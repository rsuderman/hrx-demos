// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/registry.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include <limits>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
loomc_status_t Error(const std::string &message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str()));
}
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
bool SupportsNorm(const Registration &, const InvocationRequest &r,
                  std::string *error) {
  const auto *input = r.Type(Port::Input), *gamma = r.Type(Port::Gamma);
  const auto *attrs = std::get_if<RmsNormAttrs>(&r.attributes);
  return input && gamma && attrs && input->dimension(1) == 3072 &&
                 input->dimension(0) && input->dimension(0) <= 512 &&
                 gamma->encoding() == Encoding::Dense &&
                 static_cast<float>(attrs->epsilon) == 1e-5f
             ? true
             : Fail(error, "unsupported norm shape, encoding or epsilon");
}
bool SupportsFfn(const Registration &entry, const InvocationRequest &r,
                 std::string *error) {
  const bool gate = entry.operation == OpCode::GatedMatmul;
  const auto *input = r.Type(Port::Input),
             *weight = r.Type(gate ? Port::GateWeight : Port::Weight);
  if (!input || !weight)
    return Fail(error, "missing FFN input or weight");
  const auto rows = input->dimension(0);
  const auto hidden = gate ? input->dimension(1) : r.result.type.dimension(1);
  const auto ffn = gate ? r.result.type.dimension(1) : input->dimension(1);
  if (hidden != 3072 || ffn != 8192 || !rows || rows > 512 ||
      (entry.phase == Phase::Decode && rows != 1))
    return Fail(error, "unsupported FFN geometry or phase");
  if (gate) {
    const auto *up = r.Type(Port::UpWeight);
    if (!up || up->encoding() != Encoding::Q4_K ||
        weight->encoding() != Encoding::Q4_K)
      return Fail(error, "gate/up provider requires Q4_K weights");
  } else if (weight->encoding() != Encoding::Q4_K &&
             weight->encoding() != Encoding::Q6_K)
    return Fail(error, "unsupported FFN weight encoding");
  return true;
}
loomc_status_t BuildOwned(const Registration &registration, BuildContext &ctx,
                          Value *out) {
  const auto *entry = &registration;
  auto operand = [&](Port port) { return ctx.Operand(port); };
  const auto *input = operand(Port::Input);
  const auto *type = ctx.request.graph.value(input->logical)->tensorType();
  const uint64_t rows = type->dimension(0);
  const bool norm = entry->operation == OpCode::RmsNorm;
  const bool gate = entry->operation == OpCode::GatedMatmul;
  const bool decode = entry->phase == Phase::Decode;
  const auto *weight = operand(norm   ? Port::Gamma
                               : gate ? Port::GateWeight
                                      : Port::Weight);
  const auto *weight_type =
      ctx.request.graph.value(weight->logical)->tensorType();
  const auto encoding = weight_type->encoding();
  if (norm) {
    auto checked = ctx.program.InspectNative({}, [&](const loom_module_t
                                                         *native,
                                                     const std::vector<
                                                         loom_value_id_t> &) {
      bool width = false, epsilon = false;
      for (size_t i = 0; i < native->symbols.count; ++i) {
        const auto &symbol = native->symbols.entries[i];
        if (symbol.name_id == LOOM_STRING_ID_INVALID || !symbol.defining_op ||
            !loom_config_def_isa(symbol.defining_op))
          continue;
        auto name = loom_string_table_get(&native->strings, symbol.name_id);
        auto value = loom_config_def_value(symbol.defining_op);
        if (std::string_view(name.data, name.size) == "llama.model.hidden_size")
          width = value.kind == LOOM_ATTR_I64 && value.i64 == 3072;
        if (std::string_view(name.data, name.size) == "llama.model.rms_epsilon")
          epsilon =
              value.kind == LOOM_ATTR_F64 &&
              static_cast<float>(value.f64) ==
                  static_cast<float>(
                      std::get<RmsNormAttrs>(ctx.request.attributes).epsilon);
      }
      if (!width || !epsilon)
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "native norm configuration disagrees with operation attributes");
      return iree_ok_status();
    });
    if (!loomc_status_is_ok(checked))
      return checked;
  }
  if (!norm && !decode && !ctx.has_control)
    return Error("prefill requires existing control buffer");
  if (!input->parameter.empty() ||
      (!gate && !norm && !operand(Port::Residual)->parameter.empty()))
    return Error(
        "activation and residual must be boundary or generated buffers");
  if (!norm && (weight->parameter.empty() ||
                (gate && operand(Port::UpWeight)->parameter.empty())))
    return Error("FFN weights require named parameters");
  KernelRef kernel, pack;
  CommandFamilyRef family;
  auto status = norm ? ctx.module.Kernel(entry->symbol, &kernel)
                     : ctx.module.CommandFamily(entry->symbol, &family);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index_type = ctx.module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer_type = ctx.module.Buffer();
  if (norm) {
    status = ctx.module.CheckKernelInterface(
        kernel, {index_type},
        {index_type, buffer_type, buffer_type, buffer_type});
    if (!loomc_status_is_ok(status))
      return status;
  } else {
    // The public family's native declaration remains the ABI authority. Compare
    // selected overrides against it, including dependent shape relationships,
    // while both declarations still own their argument SSA values.
    CommandFamilyRef expected;
    status = ctx.module.CommandFamily(entry->interface_symbol, &expected);
    if (!loomc_status_is_ok(status))
      return status;
    status = ctx.module.CheckCommandFamilyInterface(family, expected);
    if (!loomc_status_is_ok(status))
      return status;
  }
  if (gate && decode) {
    status = ctx.module.Kernel("ggml_quantize_q8_1_x4_qk_compat_f32", &pack);
    if (!loomc_status_is_ok(status))
      return status;
  }
  if (gate && decode) {
    status = ctx.module.CheckKernelInterface(
        pack, {index_type, index_type},
        {index_type, index_type, index_type,
         ctx.module.Scalar(LOOM_SCALAR_TYPE_I32),
         ctx.module.Scalar(LOOM_SCALAR_TYPE_OFFSET), buffer_type, buffer_type,
         buffer_type});
    if (!loomc_status_is_ok(status))
      return status;
  }
  ctx.staged = true;
  // From here all failures dispose the existing mutable owner: allocation,
  // dependent shape SSA, and calls are one synchronous native staging step.
#define STAGE(call)                                                            \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return s;                                                                \
  } while (0)
  auto index = [&](uint64_t n, Value *v) {
    return ctx.values.Constant(n, LOOM_SCALAR_TYPE_INDEX, v);
  };
  auto scratch = [&](uint64_t bytes, Value *v) {
    return ctx.Allocate(bytes, v);
  };
  Value native_output = ctx.output;
  if (!ctx.has_output)
    STAGE(scratch(rows * ctx.request.result.type.dimension(1) * 4,
                  &native_output));
  Value count;
  STAGE(index(rows, &count));
  if (norm) {
    Value gamma = weight->native;
    if (!weight->parameter.empty()) {
      Value width;
      STAGE(
          index(std::get<RmsNormAttrs>(ctx.request.attributes).width, &width));
      STAGE(ctx.values.Parameter(weight->native, weight->parameter, {},
                                 LOOM_SCALAR_TYPE_F32, {width}, &gamma));
    }
    STAGE(ctx.program.Invoke(kernel, {count},
                             {count, input->native, gamma, native_output}));
  } else {
    Value hidden, ffn, yes, no, bytes, ffn_row, hidden_row, zero, origin, four;
    STAGE(index(3072, &hidden));
    STAGE(index(8192, &ffn));
    STAGE(ctx.values.Constant(1, LOOM_SCALAR_TYPE_I1, &yes));
    STAGE(ctx.values.Constant(0, LOOM_SCALAR_TYPE_I1, &no));
    STAGE(index(8192 * 4, &ffn_row));
    STAGE(index(3072 * 4, &hidden_row));
    STAGE(index(0, &zero));
    STAGE(ctx.values.Constant(0, LOOM_SCALAR_TYPE_OFFSET, &origin));
    STAGE(index(4, &four));
    const uint64_t weight_bytes =
        3072ull * 8192 / 256 * (encoding == Encoding::Q4_K ? 144 : 210);
    STAGE(index(weight_bytes, &bytes));
    Value w, up;
    STAGE(ctx.values.Parameter(weight->native, weight->parameter, {},
                               LOOM_SCALAR_TYPE_I8, {bytes}, &w));
    if (gate) {
      const auto *up_binding = operand(Port::UpWeight);
      STAGE(ctx.values.Parameter(up_binding->native, up_binding->parameter, {},
                                 LOOM_SCALAR_TYPE_I8, {bytes}, &up));
    }
    std::vector<Value> args{hidden,
                            ffn,
                            yes,
                            yes,
                            encoding == Encoding::Q4_K ? yes : no,
                            encoding == Encoding::Q6_K ? yes : no};
    Value activation = input->native;
    if (decode) {
      if (gate) {
        Value packed, one32;
        STAGE(scratch(3072 / 128 * 144, &packed));
        STAGE(ctx.values.Constant(1, LOOM_SCALAR_TYPE_I32, &one32));
        STAGE(ctx.program.Invoke(pack, {count, hidden},
                                 {count, hidden, zero, one32, origin,
                                  activation, activation, packed}));
        activation = packed;
      }
      args.insert(args.end(), {bytes, ffn_row});
      if (!gate)
        args.push_back(hidden_row);
      args.push_back(w);
      if (gate)
        args.push_back(up);
      args.insert(args.end(), {activation, native_output});
    } else {
      Value ffn_capacity, hidden_capacity, packed_capacity, packed_depth;
      STAGE(index(rows * 8192 * 4, &ffn_capacity));
      STAGE(index(rows * 3072 * 4, &hidden_capacity));
      STAGE(index(rows * 3072 / 128 * 144, &packed_capacity));
      STAGE(index(4 * 3072 / 128 * 144, &packed_depth));
      args.insert(args.end(), {count, count, count, zero, origin, no, four,
                               ffn_capacity, bytes});
      if (gate)
        args.insert(args.end(), {packed_capacity, packed_depth, w, up,
                                 activation, ctx.control, native_output});
      else
        args.insert(args.end(), {hidden_capacity, w, ctx.control, activation,
                                 native_output});
    }
    STAGE(ctx.program.Apply(family, args));
  }
  *out = native_output;
#undef STAGE
  return loomc_ok_status();
}
} // namespace
const operations::TensorType *
InvocationRequest::Type(operations::Port port) const {
  for (const auto &input : inputs)
    if (input.port == port) {
      auto *value = graph.value(input.value);
      return value ? value->tensorType() : nullptr;
    }
  return nullptr;
}
const NativeOperand *BuildContext::Operand(operations::Port port) const {
  for (const auto &input : inputs)
    if (input.port == port)
      return &input;
  return nullptr;
}
loomc_status_t BuildContext::Allocate(uint64_t bytes,
                                      loomify_gguf::Value *out) {
  if (!bytes || bytes > uint64_t(INT64_MAX))
    return Error("native allocation size overflow");
  staged = true;
  loomify_gguf::Value size;
  auto status = values.Constant(bytes, LOOM_SCALAR_TYPE_OFFSET, &size);
  if (!loomc_status_is_ok(status))
    return status;
  return values.Allocate(size, 256, out);
}
OperationBuilder NormBuilder() {
  return {OpCode::RmsNorm,
          Kind::Kernel,
          {{Port::Input, Access::Read},
           {Port::Gamma, Access::Read},
           {Port::Output, Access::Write}},
          SupportsNorm,
          BuildOwned,
          {"llm_rmsnorm_row"}};
}
OperationBuilder FfnBuilder(operations::OpCode opcode) {
  return {
      opcode, Kind::CommandFamily,
      opcode == OpCode::GatedMatmul
          ? std::vector<OperandContract>{{Port::Input, Access::Read},
                                         {Port::GateWeight, Access::Read},
                                         {Port::UpWeight, Access::Read},
                                         {Port::Output, Access::Write}}
          : std::vector<OperandContract>{{Port::Input, Access::Read},
                                         {Port::Weight, Access::Read},
                                         {Port::Residual, Access::ReadWrite}},
      SupportsFfn, BuildOwned};
}
} // namespace hrx::dispatch
