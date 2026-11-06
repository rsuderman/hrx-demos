// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/projection.h"
#include "lib/dispatch/attention_sequence.h"
#include "lib/dispatch/projection_internal.h"
#include "loomc/iree.h"
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
bool Supports(const Registration &entry, const InvocationRequest &r,
              std::string *error) {
  const auto *x = r.Type(Port::Input), *w = r.Type(Port::Weight);
  const auto *a = std::get_if<LinearAttrs>(&r.attributes);
  const bool q4 = entry.variant == "q4_k";
  if (!x || !w || !a || x->elementType() != ElementType::F32 ||
      x->encoding() != Encoding::Dense || x->layout() != Layout::RowMajor ||
      !x->dimension(0) || x->dimension(0) > 512 || a->input_width < 256 ||
      a->input_width > (q4 ? 32768u : 17408u) || a->input_width % 256 ||
      !a->output_width || a->output_width > (q4 ? 8192u : 128256u) ||
      x->dimension(0) * a->input_width / 128 > 32768 ||
      w->encoding() != (q4 ? Encoding::Q4_K : Encoding::Q6_K)) {
    if (error)
      *error = "unsupported projection shape, activation or weight encoding";
    return false;
  }
  return true;
}
loomc_status_t Build(const Registration &entry, BuildContext &ctx, Value *out) {
  const auto &attrs = std::get<LinearAttrs>(ctx.request.attributes);
  return detail::StageProjection(entry, ctx, attrs.input_width,
                                 attrs.output_width, *ctx.Operand(Port::Input),
                                 *ctx.Operand(Port::Weight), ctx.output,
                                 ctx.has_output, out);
}
bool SupportsAttentionOutput(const Registration &entry,
                             const InvocationRequest &r, std::string *error) {
  const auto *a = std::get_if<LinearResidualAttrs>(&r.attributes);
  const auto *x = r.Type(Port::Input), *w = r.Type(Port::Weight),
             *residual = r.Type(Port::Residual);
  const bool valid =
      a && a->input_width == 3072 && a->output_width == 3072 && x && w &&
      residual && x->rank() == 2 && x->dimension(0) && x->dimension(0) <= 512 &&
      x->dimension(1) == 3072 && x->elementType() == ElementType::F32 &&
      x->encoding() == Encoding::Dense && x->layout() == Layout::RowMajor &&
      residual->rank() == 2 && residual->dimension(0) == x->dimension(0) &&
      residual->dimension(1) == 3072 &&
      residual->elementType() == ElementType::F32 &&
      residual->encoding() == Encoding::Dense &&
      residual->layout() == Layout::RowMajor &&
      w->encoding() ==
          (entry.variant == "q4_k" ? Encoding::Q4_K : Encoding::Q6_K);
  if (!valid && error)
    *error = "unsupported attention output geometry or encoding";
  return valid;
}
loomc_status_t BuildAttentionOutput(const Registration &entry,
                                    BuildContext &ctx, Value *out) {
  const auto *residual = ctx.Operand(Port::Residual);
  if (!ctx.llama_rows || !residual || !residual->parameter.empty())
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "attention output requires configured "
                                      "packet rows and residual buffer"));
  KernelRef add;
  auto status = ctx.module.Kernel("llama32_attention_residual", &add);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index = ctx.module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto offset = ctx.module.Scalar(LOOM_SCALAR_TYPE_OFFSET);
  const auto buffer = ctx.module.Buffer();
  status = ctx.module.CheckKernelInterface(
      add, {index}, {index, offset, index, buffer, buffer, buffer});
  if (!loomc_status_is_ok(status))
    return status;
  Value projected;
  // The mutable residual is never a projection destination.
  status = detail::StageProjection(
      entry, ctx, 3072, 3072, *ctx.Operand(Port::Input),
      *ctx.Operand(Port::Weight), Value{}, false, &projected);
  if (!loomc_status_is_ok(status))
    return status;
#define STAGE(call)                                                            \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return s;                                                                \
  } while (0)
  Value rows, phase, origin, begin;
  STAGE(ctx.values.Constant(ctx.llama_rows->row_capacity,
                            LOOM_SCALAR_TYPE_INDEX, &rows));
  STAGE(ctx.values.Constant(ctx.llama_rows->phase == Phase::Decode,
                            LOOM_SCALAR_TYPE_INDEX, &phase));
  STAGE(ctx.values.Constant(ctx.llama_rows->control_origin,
                            LOOM_SCALAR_TYPE_OFFSET, &origin));
  STAGE(ctx.values.Constant(ctx.llama_rows->token_begin, LOOM_SCALAR_TYPE_INDEX,
                            &begin));
  STAGE(ctx.program.Invoke(add, {rows},
                           {phase, origin, begin, ctx.llama_rows->packet,
                            projected, residual->native}));
  *out = residual->native;
  return loomc_ok_status();
#undef STAGE
}
OperationBuilder AttentionOutputBuilder() {
  return {OpCode::LinearResidual,
          Kind::Kernel,
          {{Port::Input, Access::Read},
           {Port::Weight, Access::Read},
           {Port::Residual, Access::ReadWrite}},
          SupportsAttentionOutput,
          BuildAttentionOutput,
          {"ggml_quantize_q8_1_x4_qk_compat_f32", "llama_active_row",
           "llama32_head_count", "llama32_attention_residual"}};
}

} // namespace
namespace detail {
using namespace operations;
using namespace loomify_gguf;
loomc_status_t StageProjection(const Registration &entry, BuildContext &ctx,
                               uint64_t input_width, uint64_t output_width,
                               const NativeOperand &activation,
                               const NativeOperand &weight_binding,
                               Value existing_output, bool has_existing_output,
                               Value *out) {
  const auto *input = &activation;
  const auto *weight = &weight_binding;
  if (!input->parameter.empty() || weight->parameter.empty())
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "projection requires buffer activations and named weight parameters"));
  const auto rows = ctx.request.Type(Port::Input)->dimension(0);
  KernelRef pack, projection;
  auto status = ctx.module.Kernel("ggml_quantize_q8_1_x4_qk_compat_f32", &pack);
  if (!loomc_status_is_ok(status))
    return status;
  status = ctx.module.Kernel(entry.symbol, &projection);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index = ctx.module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto i32 = ctx.module.Scalar(LOOM_SCALAR_TYPE_I32);
  const auto offset = ctx.module.Scalar(LOOM_SCALAR_TYPE_OFFSET);
  const auto buffer = ctx.module.Buffer();
  status = ctx.module.CheckKernelInterface(
      pack, {index, index},
      {index, index, index, i32, offset, buffer, buffer, buffer});
  if (!loomc_status_is_ok(status))
    return status;
  status =
      ctx.module.CheckKernelInterface(projection, {index, index, index},
                                      {index, index, index, index, i32, offset,
                                       buffer, buffer, buffer, buffer});
  if (!loomc_status_is_ok(status))
    return status;
#define STAGE(call)                                                            \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return s;                                                                \
  } while (0)
  ctx.staged = true;
  Value n, in, width, zero, count, origin, bytes, w, packed;
  STAGE(ctx.values.Constant(rows, LOOM_SCALAR_TYPE_INDEX, &n));
  STAGE(ctx.values.Constant(input_width, LOOM_SCALAR_TYPE_INDEX, &in));
  STAGE(ctx.values.Constant(output_width, LOOM_SCALAR_TYPE_INDEX, &width));
  STAGE(ctx.values.Constant(ctx.llama_rows ? ctx.llama_rows->token_begin : 0,
                            LOOM_SCALAR_TYPE_INDEX, &zero));
  STAGE(ctx.values.Constant(ctx.llama_rows ? int64_t(-1) : int64_t(rows),
                            LOOM_SCALAR_TYPE_I32, &count));
  STAGE(
      ctx.values.Constant(ctx.count_origin, LOOM_SCALAR_TYPE_OFFSET, &origin));
  const auto weight_bytes =
      input_width / 256 * output_width * (entry.variant == "q4_k" ? 144 : 210);
  STAGE(ctx.values.Constant(weight_bytes, LOOM_SCALAR_TYPE_INDEX, &bytes));
  STAGE(ctx.values.Parameter(weight->native, weight->parameter, {},
                             LOOM_SCALAR_TYPE_I8, {bytes}, &w));
  STAGE(ctx.Allocate(rows * input_width / 128 * 144, &packed));
  Value output = existing_output;
  if (!has_existing_output)
    STAGE(ctx.Allocate(rows * output_width * 4, &output));
  // Capacity still sizes physical storage; actual count selects the existing
  // numerical policy and the provider-owned active-row guard.
  STAGE(ctx.program.Invoke(pack, {n, in},
                           {n, in, zero, count, origin,
                            ctx.llama_rows ? ctx.count_control : input->native,
                            input->native, packed}));
  STAGE(ctx.program.Invoke(projection, {n, in, width},
                           {n, in, width, zero, count, origin,
                            ctx.llama_rows ? ctx.count_control : packed, packed,
                            w, output}));
  *out = output;
#undef STAGE
  return loomc_ok_status();
}
} // namespace detail

OperationBuilder ProjectionBuilder() {
  return {operations::OpCode::Linear,
          Kind::Kernel,
          {{operations::Port::Input, operations::Access::Read},
           {operations::Port::Weight, operations::Access::Read},
           {operations::Port::Output, operations::Access::Write}},
          Supports,
          Build,
          {"ggml_quantize_q8_1_x4_qk_compat_f32", "llama_active_row",
           "llama32_head_count"}};
}
void RegisterProjections(Registry *registry) {
  for (const bool q4 : {true, false}) {
    const std::string root = "lib/kernels/llama/";
    Registration entry{Id::Projection,
                       operations::OpCode::Linear,
                       Kind::Kernel,
                       "llama.projection",
                       q4 ? "ggml_linear_q4k_q8_1_x4_compat"
                          : "ggml_linear_q6k_q8_1_x4_compat",
                       root + "projection/projection_interface.loom",
                       {},
                       {},
                       {"ggml.linear_q4k_q8_1_x4.token_capacity",
                        "ggml.linear_q4k_q8_1_x4.output_capacity",
                        "ggml.linear_q6k_q8_1_x4.token_capacity",
                        "ggml.linear_q6k_q8_1_x4.output_capacity",
                        "ggml.quantize_q8_1_x4.group_capacity"}};
    for (const char *file :
         {"lib/kernels/llama/config.loom", "lib/kernels/llama/active_rows.loom",
          "lib/kernels/llama/output/head_count.loom",
          "lib/kernels/projection/q4k.loom",
          "lib/kernels/projection/q4k_compat.loom",
          "lib/kernels/projection/q6k.loom",
          "lib/kernels/projection/q6k_compat.loom",
          "lib/kernels/projection/qk_common.loom",
          "lib/kernels/projection/qk_mmvq_compat.loom",
          "lib/kernels/quantization/q8_1_x4.loom",
          "lib/kernels/quantization/q8_1_x4_mmvq_compat.loom"})
      entry.provider_files.push_back(file);
    entry.variant = q4 ? "q4_k" : "q6_k";
    entry.interface_symbol = entry.symbol;
    entry.builder = ProjectionBuilder();
    entry.operands = entry.builder.operands;
    std::string error;
    Registration output = entry;
    registry->Add(std::move(entry), &error);
    output.id = Id::AttentionOutput;
    output.operation = OpCode::LinearResidual;
    output.name = "llama.attention_output";
    output.builder = AttentionOutputBuilder();
    output.operands = output.builder.operands;
    const auto native = AttentionSequenceSources();
    output.provider_files.push_back(native.front().interface_file);
    output.provider_files.insert(output.provider_files.end(),
                                 native.front().provider_files.begin(),
                                 native.front().provider_files.end());
    output.required_config.insert(output.required_config.end(),
                                  native.front().required_config.begin(),
                                  native.front().required_config.end());
    registry->Add(std::move(output), &error);
  }
}
} // namespace hrx::dispatch
