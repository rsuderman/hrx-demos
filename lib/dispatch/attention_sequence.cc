// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/attention_sequence.h"
#include "loomc/iree.h"
namespace hrx::dispatch {
using namespace loomify_gguf;
loomc_status_t LoadAttentionSequence(loomc_context_t *context,
                                     loomc_workspace_t *workspace,
                                     const std::string &root,
                                     const std::string &capacity_config,
                                     loomc_module_t **out,
                                     loomc_result_t **result) {
  std::vector<NativeSource> entries;
  const std::string prefix = "lib/kernels/llama/";
  for (const char *symbol :
       {"llama32_rotary_query", "llama32_publish_kv", "llama32_attention_rows",
        "llama32_attention_residual"}) {
    // Load inspects only native kind/declaration, source closure and roots.
    NativeSource e{};
    e.kind = Kind::Kernel;
    e.symbol = e.interface_symbol = symbol;
    e.interface_file = prefix + "attention/attention_native_interface.loom";
    for (const char *file :
         {"attention/attention_prepare.loom", "attention/attention_native.loom",
          "attention/rotary.loom", "attention/attention_core.loom",
          "attention/attention_map.loom",
          "attention/attention_materialized.loom", "rows.loom", "config.loom",
          "attention/attention_config.loom"})
      e.provider_files.push_back(prefix + file);
    e.provider_files.push_back(capacity_config);
    entries.push_back(std::move(e));
  }
  return LoadNative(context, workspace, entries, root, out, result);
}
loomc_status_t BuildAttentionSequence(ModuleBuilder &module,
                                      CommandProgramBuilder &program,
                                      const AttentionSequenceOptions &o,
                                      const AttentionSequenceBindings &b,
                                      Value *attention) {
  if (!attention || !o.row_capacity || o.row_capacity > 512 || o.layer >= 28 ||
      o.token_begin >= o.row_capacity || o.control_origin > INT64_MAX ||
      o.map_origin > INT64_MAX || o.cache_base > INT64_MAX ||
      (o.phase != operations::Phase::Prefill &&
       o.phase != operations::Phase::Decode) ||
      (o.phase == operations::Phase::Decode && o.row_capacity != 1))
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "invalid native Llama attention sequence options"));
#define TRY(call)                                                              \
  do {                                                                         \
    auto status = (call);                                                      \
    if (!loomc_status_is_ok(status))                                           \
      return status;                                                           \
  } while (0)
  const auto index = module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto offset = module.Scalar(LOOM_SCALAR_TYPE_OFFSET);
  const auto buffer = module.Buffer();
  KernelRef rotary, publish, causal;
  TRY(module.Kernel("llama32_rotary_query", &rotary));
  TRY(module.Kernel("llama32_publish_kv", &publish));
  TRY(module.Kernel("llama32_attention_rows", &causal));
  TRY(module.CheckKernelInterface(
      rotary, {index}, {index, offset, index, buffer, buffer, buffer, buffer}));
  TRY(module.CheckKernelInterface(publish, {index},
                                  {index, offset, index, offset, index, offset,
                                   buffer, buffer, buffer, buffer, buffer,
                                   buffer}));
  TRY(module.CheckKernelInterface(causal, {index},
                                  {index, offset, index, offset, index, offset,
                                   buffer, buffer, buffer, buffer, buffer}));
  CommandValues values(program);
  Value rows, layer, kind, control_origin, begin, cache_base, map_origin;
  TRY(values.Constant(o.row_capacity, LOOM_SCALAR_TYPE_INDEX, &rows));
  TRY(values.Constant(o.layer, LOOM_SCALAR_TYPE_INDEX, &layer));
  TRY(values.Constant(o.phase == operations::Phase::Decode,
                      LOOM_SCALAR_TYPE_INDEX, &kind));
  TRY(values.Constant(o.control_origin, LOOM_SCALAR_TYPE_OFFSET,
                      &control_origin));
  TRY(values.Constant(o.token_begin, LOOM_SCALAR_TYPE_INDEX, &begin));
  TRY(values.Constant(o.cache_base, LOOM_SCALAR_TYPE_OFFSET, &cache_base));
  TRY(values.Constant(o.map_origin, LOOM_SCALAR_TYPE_OFFSET, &map_origin));
  Value size, query, output;
  TRY(values.Constant(o.row_capacity * 3072 * 4, LOOM_SCALAR_TYPE_OFFSET,
                      &size));
  TRY(values.Allocate(size, 256, &query));
  TRY(values.Allocate(size, 256, &output));
  TRY(program.Invoke(rotary, {rows},
                     {kind, control_origin, begin, b.control, b.factors,
                      b.query_projection, query}));
  TRY(program.Invoke(publish, {rows},
                     {layer, cache_base, kind, control_origin, begin,
                      map_origin, b.control, b.map, b.factors, b.key_projection,
                      b.value_projection, b.cache}));
  TRY(program.Invoke(causal, {rows},
                     {layer, cache_base, kind, control_origin, begin,
                      map_origin, b.control, b.map, query, b.cache, output}));
  *attention = output;
  return loomc_ok_status();
#undef TRY
}
} // namespace hrx::dispatch
