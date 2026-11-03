// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/registry.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include <string_view>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
loomc_status_t Error(const char *message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
}
bool Supports(const Registration &, const InvocationRequest &r,
              std::string *error) {
  const auto *ids = r.Type(Port::TokenIds), *weight = r.Type(Port::Weight);
  const auto *a = std::get_if<EmbeddingAttrs>(&r.attributes);
  for (const auto &input : r.inputs)
    if (input.port == Port::RowBegin || input.port == Port::RowCount)
      return Fail(
          error,
          "embedding provider computes all rows; active controls unsupported");
  return ids && weight && a && ids->rank() == 1 &&
                 ids->elementType() == ElementType::I32 && ids->dimension(0) &&
                 ids->dimension(0) <= 512 && a->embedding_width == 3072 &&
                 a->vocabulary_size == 128256 &&
                 weight->encoding() == Encoding::Q6_K
             ? true
             : Fail(error, "embedding provider requires Llama 3B Q6_K weights "
                           "and 1..512 token IDs");
}
loomc_status_t Build(const Registration &entry, BuildContext &ctx, Value *out) {
  const auto *ids = ctx.Operand(Port::TokenIds);
  const auto *weight = ctx.Operand(Port::Weight);
  if (!ids || !weight || !ids->parameter.empty() || weight->parameter.empty())
    return Error(
        "embedding requires token buffer and named immutable weight parameter");
  KernelRef kernel;
  auto status = ctx.module.Kernel(entry.symbol, &kernel);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index = ctx.module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer = ctx.module.Buffer();
  status = ctx.module.CheckKernelInterface(kernel, {index},
                                           {index, buffer, buffer, buffer});
  if (!loomc_status_is_ok(status))
    return status;
  status = ctx.program.InspectNative(
      {}, [](const loom_module_t *m, const std::vector<loom_value_id_t> &) {
        bool hidden = false, vocabulary = false;
        for (size_t i = 0; i < m->symbols.count; ++i) {
          const auto &s = m->symbols.entries[i];
          if (s.name_id == LOOM_STRING_ID_INVALID || !s.defining_op ||
              !loom_config_def_isa(s.defining_op))
            continue;
          const auto n = loom_string_table_get(&m->strings, s.name_id);
          const auto v = loom_config_def_value(s.defining_op);
          if (std::string_view(n.data, n.size) == "llama.model.hidden_size")
            hidden = v.kind == LOOM_ATTR_I64 && v.i64 == 3072;
          if (std::string_view(n.data, n.size) == "llama.model.vocabulary_size")
            vocabulary = v.kind == LOOM_ATTR_I64 && v.i64 == 128256;
        }
        return hidden && vocabulary
                   ? iree_ok_status()
                   : iree_make_status(
                         IREE_STATUS_INVALID_ARGUMENT,
                         "native embedding geometry disagrees with attributes");
      });
  if (!loomc_status_is_ok(status))
    return status;
#define STAGE(call)                                                            \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return s;                                                                \
  } while (0)
  const auto rows = ctx.request.result.type.dimension(0);
  ctx.staged = true;
  Value output = ctx.output, count, bytes, weights;
  if (!ctx.has_output)
    STAGE(ctx.Allocate(rows * 3072 * 4, &output));
  STAGE(ctx.values.Constant(rows, LOOM_SCALAR_TYPE_INDEX, &count));
  STAGE(ctx.values.Constant(128256ull * (3072 / 256) * 210,
                            LOOM_SCALAR_TYPE_INDEX, &bytes));
  STAGE(ctx.values.Parameter(weight->native, weight->parameter, {},
                             LOOM_SCALAR_TYPE_I8, {bytes}, &weights));
  STAGE(ctx.program.Invoke(kernel, {count},
                           {count, ids->native, weights, output}));
  *out = output;
#undef STAGE
  return loomc_ok_status();
}
} // namespace
OperationBuilder EmbeddingBuilder() {
  return {OpCode::Embedding,
          Kind::Kernel,
          {{Port::TokenIds, Access::Read},
           {Port::Weight, Access::Read},
           {Port::Output, Access::Write}},
          Supports,
          Build,
          {"llm_token_embedding_q6k_f32_row", "ggml_q6k_f32_vector4"}};
}
Registration EmbeddingRegistration() {
  const std::string root = "lib/kernels/llama/";
  Registration r{Id::Embedding,
                 OpCode::Embedding,
                 Kind::Kernel,
                 "llama.embedding",
                 "llama32_token_embedding_q6k_f32",
                 root + "embedding/embedding_interface.loom",
                 {root + "embedding/embedding.loom",
                  root + "embedding/embedding_rows.loom",
                  root + "embedding/embedding_config.loom",
                  root + "config.loom"},
                 {},
                 {"llama.model.hidden_size", "llama.model.vocabulary_size"}};
  r.interface_symbol = r.symbol;
  r.builder = EmbeddingBuilder();
  r.operands = r.builder.operands;
  return r;
}
} // namespace hrx::dispatch
