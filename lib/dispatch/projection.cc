// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/projection.h"
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
  const auto *input = ctx.Operand(Port::Input);
  const auto *weight = ctx.Operand(Port::Weight);
  if (!input->parameter.empty() || weight->parameter.empty())
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "projection requires buffer activations and named weight parameters"));
  const auto &attrs = std::get<LinearAttrs>(ctx.request.attributes);
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
  STAGE(ctx.values.Constant(attrs.input_width, LOOM_SCALAR_TYPE_INDEX, &in));
  STAGE(
      ctx.values.Constant(attrs.output_width, LOOM_SCALAR_TYPE_INDEX, &width));
  STAGE(ctx.values.Constant(0, LOOM_SCALAR_TYPE_INDEX, &zero));
  STAGE(ctx.values.Constant(rows, LOOM_SCALAR_TYPE_I32, &count));
  STAGE(ctx.values.Constant(0, LOOM_SCALAR_TYPE_OFFSET, &origin));
  const auto weight_bytes = attrs.input_width / 256 * attrs.output_width *
                            (entry.variant == "q4_k" ? 144 : 210);
  STAGE(ctx.values.Constant(weight_bytes, LOOM_SCALAR_TYPE_INDEX, &bytes));
  STAGE(ctx.values.Parameter(weight->native, weight->parameter, {},
                             LOOM_SCALAR_TYPE_I8, {bytes}, &w));
  STAGE(ctx.Allocate(rows * attrs.input_width / 128 * 144, &packed));
  Value output = ctx.output;
  if (!ctx.has_output)
    STAGE(ctx.Allocate(rows * attrs.output_width * 4, &output));
  // Fixed logical count selects existing native MMVQ/MMQ numerical policy.
  // Nonnegative counts bypass the count-view buffer entirely.
  STAGE(ctx.program.Invoke(
      pack, {n, in},
      {n, in, zero, count, origin, input->native, input->native, packed}));
  STAGE(ctx.program.Invoke(
      projection, {n, in, width},
      {n, in, width, zero, count, origin, packed, packed, w, output}));
  *out = output;
#undef STAGE
  return loomc_ok_status();
}
} // namespace
OperationBuilder ProjectionBuilder() {
  return {operations::OpCode::Linear,
          Kind::Kernel,
          {{operations::Port::Input, operations::Access::Read},
           {operations::Port::Weight, operations::Access::Read},
           {operations::Port::Output, operations::Access::Write}},
          Supports,
          Build,
          {"ggml_quantize_q8_1_x4_qk_compat_f32"}};
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
         {"config.loom", "projection/linear_q4k_q8_1_x4.loom",
          "projection/linear_q4k_q8_1_x4_compat.loom",
          "projection/linear_q6k_q8_1_x4.loom",
          "projection/linear_q6k_q8_1_x4_compat.loom",
          "projection/linear_qk_common.loom",
          "projection/linear_qk_mmvq_compat.loom",
          "projection/quantize_q8_1_x4.loom",
          "projection/quantize_q8_1_x4_mmvq_compat.loom"})
      entry.provider_files.push_back(root + file);
    entry.variant = q4 ? "q4_k" : "q6_k";
    entry.interface_symbol = entry.symbol;
    entry.builder = ProjectionBuilder();
    entry.operands = entry.builder.operands;
    std::string error;
    registry->Add(std::move(entry), &error);
  }
}
} // namespace hrx::dispatch
