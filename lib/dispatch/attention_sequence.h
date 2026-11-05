// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_ATTENTION_SEQUENCE_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_ATTENTION_SEQUENCE_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch {
// Existing DefineLlamaStep packet ABI. Control/map/cache remain caller owned;
// cache_base + layer*2*physical_capacity*1024*2 selects this layer's K plane.
// Control prefill={active_count,absolute_base}; decode word0=absolute position.
// Pooled maps use unique valid 64-token block IDs; dense pool capacity is zero.
struct AttentionSequenceOptions {
  uint64_t row_capacity = 1;
  uint64_t layer = 0;
  operations::Phase phase = operations::Phase::Prefill;
  uint64_t control_origin = 0;
  uint64_t token_begin = 0;
  uint64_t cache_base = 0;
  uint64_t map_origin = 0;
};
struct AttentionSequenceBindings {
  loomify_gguf::Value control, map, cache, factors;
  loomify_gguf::Value query_projection, key_projection, value_projection;
};
struct AttentionNativeConfig {
  uint64_t layer_count = 28;
  uint64_t cache_capacity = 0;
  uint64_t pool_capacity = 0;
};
// The actual packet sequence closure, excluding geometry/capacity providers.
// A combined model loader may merge it with registered family closures once.
std::vector<NativeSource> AttentionSequenceSources();
loomc_status_t CreateAttentionConfiguration(loomc_context_t *,
                                            loomc_workspace_t *,
                                            const AttentionNativeConfig &,
                                            loomc_module_t **,
                                            loomc_result_t **);
loomc_status_t LoadAttentionSequence(loomc_context_t *, loomc_workspace_t *,
                                     const std::string &source_root,
                                     const AttentionNativeConfig &,
                                     loomc_module_t **, loomc_result_t **);
// Combines registered operation families and the packet-attention providers
// under one admitted configuration; implementation variants stay
// registry-owned.
loomc_status_t LoadLlamaNative(loomc_context_t *, loomc_workspace_t *,
                               const Registry &, const std::vector<Id> &,
                               const std::string &source_root,
                               const AttentionNativeConfig &, loomc_module_t **,
                               loomc_result_t **);
// Loads actual packet kernels and their owned numerical closure. Uses the
// native declarations for ABI/link checks; no typed graph operation is created.
loomc_status_t LoadAttentionSequence(loomc_context_t *, loomc_workspace_t *,
                                     const std::string &source_root,
                                     const std::string &capacity_config,
                                     loomc_module_t **, loomc_result_t **);
// Validates loaded configuration, offsets and kernel interfaces without
// staging.
loomc_status_t
CheckAttentionSequence(loomify_gguf::ModuleBuilder &,
                       loomify_gguf::CommandProgramBuilder &,
                       const AttentionSequenceOptions &,
                       AttentionNativeConfig *configuration = nullptr);
// Stages all active K/V publication before causal attention, retaining existing
// mixed F16/F32 policy and inactive attention-row zeroing. Owns Q/output
// scratch. Caller discards the mutable program after staging failure.
loomc_status_t BuildAttentionSequence(loomify_gguf::ModuleBuilder &,
                                      loomify_gguf::CommandProgramBuilder &,
                                      const AttentionSequenceOptions &,
                                      const AttentionSequenceBindings &,
                                      loomify_gguf::Value *attention);
} // namespace hrx::dispatch
#endif
