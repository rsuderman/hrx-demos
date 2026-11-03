// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include <utility>

namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
loomc_status_t Error(const std::string &message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str()));
}
} // namespace
NativeEmissionSession::NativeEmissionSession(
    const Registry &registry, const operations::OperationGraph &graph,
    ModuleBuilder &module, CommandProgramBuilder &program)
    : registry_(registry), graph_(graph), module_(module), program_(program),
      values_(program) {}
const NativeEmissionSession::Binding *
NativeEmissionSession::Find(ValueId id) const {
  for (const auto &binding : bindings_)
    if (binding.logical == id)
      return &binding;
  return nullptr;
}
loomc_status_t NativeEmissionSession::CheckBuffer(Value native) const {
  return program_.InspectNative(
      {native},
      [](const loom_module_t *module, const std::vector<loom_value_id_t> &ids) {
        if (!loom_type_equal(loom_module_value_type(module, ids[0]),
                             loom_type_buffer()))
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "boundary value must be a buffer");
        return iree_ok_status();
      });
}
loomc_status_t NativeEmissionSession::Bind(ValueId logical, Value native) {
  if (failed_ || !graph_.value(logical) ||
      !graph_.value(logical)->tensorType() || Find(logical))
    return Error("invalid, foreign or already bound logical tensor");
  auto status = CheckBuffer(native);
  if (!loomc_status_is_ok(status))
    return status;
  if (graph_.value(logical)->tensorType()->encoding() != Encoding::Dense)
    return Error("quantized weights require named parameter binding");
  bindings_.push_back({logical, native, {}});
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::Parameter(ValueId logical,
                                                Value parameters,
                                                std::string key) {
  if (failed_ || !graph_.value(logical) ||
      !graph_.value(logical)->tensorType() || Find(logical) || key.empty() ||
      key.find('\0') != std::string::npos ||
      key.find('{') != std::string::npos || key.find('}') != std::string::npos)
    return Error("invalid, foreign or already bound named parameter");
  auto status = CheckBuffer(parameters);
  if (!loomc_status_is_ok(status))
    return status;
  bindings_.push_back({logical, parameters, std::move(key)});
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::Control(Value control) {
  if (failed_)
    return Error("failed invocation owner");
  auto status = CheckBuffer(control);
  if (!loomc_status_is_ok(status))
    return status;
  control_ = control;
  has_control_ = true;
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::Native(ValueId logical,
                                             Value *out) const {
  const auto *binding = Find(logical);
  if (!out || failed_ || !binding || !binding->parameter.empty())
    return Error("missing native logical buffer");
  auto status = CheckBuffer(binding->native);
  if (loomc_status_is_ok(status))
    *out = binding->native;
  return status;
}
loomc_status_t NativeEmissionSession::FailMutation(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return status;
  failed_ = true;
  // Discard the mutable owner after any partial staging, using its existing
  // terminal native-error path. No staged graph/native output is published.
  std::vector<Value> ignored;
  auto terminal = program_.BuildNative(
      {},
      [](loom_module_t *, loom_builder_t *,
         const std::vector<loom_value_id_t> &, std::vector<loom_value_id_t> *) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "invocation staging failed");
      },
      &ignored);
  loomc_status_free(terminal);
  return status;
}
loomc_status_t
NativeEmissionSession::Emit(Id id, const OperationAttrs &attrs,
                            const std::vector<shapes::Operand> &inputs,
                            ValueId existing_result, shapes::Result *out_result,
                            Value *out_native) {
  if (!out_result || !out_native || failed_)
    return Error("missing result or failed invocation owner");
  const auto *entry = registry_.Find(id);
  if (!entry)
    return Error("missing dispatch registration");
  std::string error;
  Registry validation;
  if (!validation.Add(*entry, &error))
    return Error(error);
  shapes::Result inferred;
  if (!shapes::InferResult(graph_, entry->operation, attrs, inputs, &inferred,
                           &error))
    return Error(error);
  if (!graph_.definition(entry->operation))
    return Error("graph lacks operation definition");
  if (existing_result.valid()) {
    const auto *description = graph_.value(existing_result);
    const auto *actual = description ? description->tensorType() : nullptr;
    if (!actual || actual->rank() != inferred.type.rank() ||
        actual->elementType() != inferred.type.elementType() ||
        actual->encoding() != inferred.type.encoding() ||
        actual->layout() != inferred.type.layout())
      return Error("existing output type disagrees with inferred result");
    for (size_t i = 0; i < actual->rank(); ++i)
      if (actual->dimension(i) != inferred.type.dimension(i))
        return Error("existing output shape disagrees with inferred result");
    if (inferred.access == Access::ReadWrite &&
        existing_result != inferred.alias)
      return Error("existing ReadWrite output must retain its explicit alias");
    if (inferred.access == Access::Write) {
      if (const auto *binding = Find(existing_result)) {
        if (!binding->parameter.empty())
          return Error(
              "existing output cannot overwrite an immutable parameter");
        auto status = CheckBuffer(binding->native);
        if (!loomc_status_is_ok(status))
          return status;
      }
    }
  }
  // Keep original graph controls for shape verification and inspection; only
  // this explicit proof specialization permits omitting their native mapping.
  std::vector<shapes::Operand> native_inputs;
  for (const auto &input : inputs) {
    if (input.port == Port::RowBegin || input.port == Port::RowCount) {
      if (!one_row_ ||
          input.value !=
              (input.port == Port::RowBegin ? row_begin_ : row_count_))
        return Error(
            "active subspan requires explicit matching one-row specialization");
    } else
      native_inputs.push_back(input);
  }
  auto operand = [&](Port port) -> const Binding * {
    for (const auto &input : inputs)
      if (input.port == port)
        return Find(input.value);
    return nullptr;
  };
  for (const auto &input : native_inputs) {
    auto *binding = Find(input.value);
    if (!binding)
      return Error("unbound logical input");
    auto status = CheckBuffer(binding->native);
    if (!loomc_status_is_ok(status))
      return status;
  }
  const auto *input = operand(Port::Input);
  const auto *type = graph_.value(input->logical)->tensorType();
  const uint64_t rows = type->dimension(0);
  if (one_row_ && rows != 1)
    return Error(
        "explicit one-row entrypoint requires the whole static row domain");
  const bool norm = id == Id::RmsNorm;
  const bool gate = entry->operation == OpCode::GatedMatmul;
  const bool decode = id == Id::GateUpDecode || id == Id::DownDecode;
  const auto *weight = operand(norm   ? Port::Gamma
                               : gate ? Port::GateWeight
                                      : Port::Weight);
  const auto *weight_type = graph_.value(weight->logical)->tensorType();
  const auto encoding = weight_type->encoding();
  if (!CheckProfile(*entry,
                    norm || gate ? type->dimension(1)
                                 : inferred.type.dimension(1),
                    norm   ? 8192
                    : gate ? inferred.type.dimension(1)
                           : type->dimension(1),
                    rows, encoding, PackingPolicy::QkCompat, &error))
    return Error(error);
  if (norm &&
      static_cast<float>(std::get<RmsNormAttrs>(attrs).epsilon) != 1e-5f)
    return Error("norm epsilon differs from selected provider configuration");
  if (norm) {
    auto checked = program_.InspectNative({}, [&](const loom_module_t *native,
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
                  static_cast<float>(std::get<RmsNormAttrs>(attrs).epsilon);
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
  if (gate && graph_.value(operand(Port::UpWeight)->logical)
                      ->tensorType()
                      ->encoding() != Encoding::Q4_K)
    return Error("gate/up provider requires Q4_K for both weights");
  if (!norm && !decode && !has_control_)
    return Error("prefill requires existing control buffer");
  if (!norm && !decode) {
    auto status = CheckBuffer(control_);
    if (!loomc_status_is_ok(status))
      return status;
  }
  if (!input->parameter.empty() ||
      (!gate && !norm && !operand(Port::Residual)->parameter.empty()))
    return Error(
        "activation and residual must be boundary or generated buffers");
  if (!norm && (weight->parameter.empty() ||
                (gate && operand(Port::UpWeight)->parameter.empty())))
    return Error("FFN weights require named parameters");
  KernelRef kernel, pack;
  CommandFamilyRef family;
  auto status = norm ? module_.Kernel(entry->symbol, &kernel)
                     : module_.CommandFamily(entry->symbol, &family);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index_type = module_.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer_type = module_.Buffer();
  if (norm) {
    status = module_.CheckKernelInterface(
        kernel, {index_type},
        {index_type, buffer_type, buffer_type, buffer_type});
    if (!loomc_status_is_ok(status))
      return status;
  } else {
    // The public family's native declaration remains the ABI authority. Compare
    // selected overrides against it, including dependent shape relationships,
    // while both declarations still own their argument SSA values.
    CommandFamilyRef expected;
    const auto known = Registry::Llama3B();
    status = module_.CommandFamily(known.Find(id)->symbol, &expected);
    if (!loomc_status_is_ok(status))
      return status;
    status = module_.CheckCommandFamilyInterface(family, expected);
    if (!loomc_status_is_ok(status))
      return status;
  }
  if (gate && decode) {
    status = module_.Kernel("ggml_quantize_q8_1_x4_qk_compat_f32", &pack);
    if (!loomc_status_is_ok(status))
      return status;
  }
  if (gate && decode) {
    status =
        module_.CheckKernelInterface(pack, {index_type, index_type},
                                     {index_type, index_type, index_type,
                                      module_.Scalar(LOOM_SCALAR_TYPE_I32),
                                      module_.Scalar(LOOM_SCALAR_TYPE_OFFSET),
                                      buffer_type, buffer_type, buffer_type});
    if (!loomc_status_is_ok(status))
      return status;
  }
  started_ = true;
  // From here all failures dispose the existing mutable owner: allocation,
  // dependent shape SSA, and calls are one synchronous native staging step.
#define STAGE(call)                                                            \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return FailMutation(s);                                                  \
  } while (0)
  auto index = [&](uint64_t n, Value *v) {
    return values_.Constant(n, LOOM_SCALAR_TYPE_INDEX, v);
  };
  auto scratch = [&](uint64_t bytes, Value *v) {
    Value size;
    auto s = values_.Constant(bytes, LOOM_SCALAR_TYPE_OFFSET, &size);
    if (!loomc_status_is_ok(s))
      return s;
    return values_.Allocate(size, 256, v);
  };
  Value native_output;
  if (inferred.access == Access::ReadWrite)
    native_output = operand(Port::Residual)->native;
  else if (existing_result.valid() && Find(existing_result))
    native_output = Find(existing_result)->native;
  else
    STAGE(scratch(rows * inferred.type.dimension(1) * 4, &native_output));
  Value count;
  STAGE(index(rows, &count));
  if (norm) {
    Value gamma = weight->native;
    if (!weight->parameter.empty()) {
      Value width;
      STAGE(index(std::get<RmsNormAttrs>(attrs).width, &width));
      STAGE(values_.Parameter(weight->native, weight->parameter, {},
                              LOOM_SCALAR_TYPE_F32, {width}, &gamma));
    }
    STAGE(program_.Invoke(kernel, {count},
                          {count, input->native, gamma, native_output}));
  } else {
    Value hidden, ffn, yes, no, bytes, ffn_row, hidden_row, zero, origin, four;
    STAGE(index(3072, &hidden));
    STAGE(index(8192, &ffn));
    STAGE(values_.Constant(1, LOOM_SCALAR_TYPE_I1, &yes));
    STAGE(values_.Constant(0, LOOM_SCALAR_TYPE_I1, &no));
    STAGE(index(8192 * 4, &ffn_row));
    STAGE(index(3072 * 4, &hidden_row));
    STAGE(index(0, &zero));
    STAGE(values_.Constant(0, LOOM_SCALAR_TYPE_OFFSET, &origin));
    STAGE(index(4, &four));
    const uint64_t weight_bytes =
        3072ull * 8192 / 256 * (encoding == Encoding::Q4_K ? 144 : 210);
    STAGE(index(weight_bytes, &bytes));
    Value w, up;
    STAGE(values_.Parameter(weight->native, weight->parameter, {},
                            LOOM_SCALAR_TYPE_I8, {bytes}, &w));
    if (gate) {
      const auto *up_binding = operand(Port::UpWeight);
      STAGE(values_.Parameter(up_binding->native, up_binding->parameter, {},
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
        STAGE(values_.Constant(1, LOOM_SCALAR_TYPE_I32, &one32));
        STAGE(program_.Invoke(pack, {count, hidden},
                              {count, hidden, zero, one32, origin, activation,
                               activation, packed}));
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
                                 activation, control_, native_output});
      else
        args.insert(args.end(),
                    {hidden_capacity, w, control_, activation, native_output});
    }
    STAGE(program_.Apply(family, args));
  }
  *out_result = inferred;
  *out_native = native_output;
#undef STAGE
  return loomc_ok_status();
}

loomc_status_t NativeEmissionSession::SpecializeOneRow(ValueId begin,
                                                       ValueId count) {
  if (failed_ || started_ || one_row_)
    return Error("one-row specialization must be set once before emission");
  const auto *begin_value = graph_.value(begin),
             *count_value = graph_.value(count);
  if (!begin_value || !count_value || begin == count ||
      !begin_value->scalarType() || !count_value->scalarType() ||
      *begin_value->scalarType() != ScalarType::I64 ||
      *count_value->scalarType() != ScalarType::I64)
    return Error("one-row specialization requires distinct owned I64 control "
                 "identities");
  row_begin_ = begin;
  row_count_ = count;
  one_row_ = true;
  return loomc_ok_status();
}

loomc_status_t
NativeEmissionSession::EmitExisting(Id id, const Invocation &invocation) {
  size_t index = 0;
  for (; index < graph_.invocations().size(); ++index)
    if (&graph_.invocations()[index] == &invocation)
      break;
  if (index == graph_.invocations().size() ||
      (has_emitted_existing_ && index <= last_existing_))
    return Error("existing invocation must belong to this graph and follow "
                 "append order");
  const auto *entry = registry_.Find(id);
  if (!entry || entry->operation != invocation.opcode())
    return Error("dispatch does not match existing invocation");
  const auto *definition = graph_.definition(invocation.opcode());
  if (!definition)
    return Error("missing existing operation definition");
  std::vector<shapes::Operand> inputs;
  ValueId output;
  for (const auto &binding : invocation.bindings()) {
    const auto *port = definition->port(binding.port());
    if (!port)
      return Error("unknown existing operation port");
    if (port->access() == Access::Write) {
      if (output.valid())
        return Error("dispatch expects a single existing output");
      output = binding.value();
    } else {
      inputs.push_back({binding.port(), binding.value()});
      if (port->access() == Access::ReadWrite)
        output = binding.value();
    }
  }
  if (!output.valid())
    return Error("missing existing output identity");
  shapes::Result inferred;
  Value native;
  auto status =
      Emit(id, invocation.attributes(), inputs, output, &inferred, &native);
  if (!loomc_status_is_ok(status))
    return status;
  if (inferred.access == Access::Write && !Find(output))
    bindings_.push_back({output, native, {}});
  last_existing_ = index;
  has_emitted_existing_ = true;
  return loomc_ok_status();
}

loomc_status_t
InvocationProgram::Invoke(Id id, const OperationAttrs &attrs,
                          const std::vector<shapes::Operand> &inputs,
                          ValueId *out) {
  if (!out)
    return Error("missing logical result");
  shapes::Result inferred;
  Value native;
  auto status = session_.Emit(id, attrs, inputs, {}, &inferred, &native);
  if (!loomc_status_is_ok(status))
    return status;
  std::string error;
  ValueId result = inferred.alias;
  if (!result.valid()) {
    result = graph_.tensor(inferred.type, &error);
    if (!result.valid())
      return session_.FailMutation(Error(error));
  }
  auto invocation = graph_.operation(session_.registry_.Find(id)->operation);
  std::visit([&](const auto &a) { invocation.attributes(a); }, attrs);
  for (const auto &input : inputs)
    invocation.bind(input.port, input.value);
  if (inferred.access == Access::Write)
    invocation.bind(inferred.port, result);
  if (!invocation.append(&error))
    return session_.FailMutation(Error(error));
  if (inferred.access == Access::Write)
    session_.bindings_.push_back({result, native, {}});
  *out = result;
  return loomc_ok_status();
}
} // namespace hrx::dispatch
