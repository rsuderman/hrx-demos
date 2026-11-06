// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_QWEN_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_QWEN_H_
#include "lib/dispatch/registry.h"
#include "tools/model/model_descriptor.h"
#include <utility>
namespace hrx::dispatch {
// Explicit deployment bounds, independent of checkpoint training context.
struct QwenNativeOptions {
  uint64_t token_capacity = 0;
  uint64_t span_capacity = 0;
  uint64_t selected_output_capacity = 0;
  uint64_t cache_capacity = 0;
  uint64_t pool_capacity = 0;
  uint64_t row_capacity = 0;
};
struct QwenNativeGeometry {
  uint64_t embedding_width = 0, feed_forward_width = 0, vocabulary_size = 0;
  uint64_t residual_bytes = 0;
  uint64_t gdn_history_bytes = 0, gdn_recurrent_bytes = 0, gdn_layer_bytes = 0;
  uint64_t gdn_layer_count = 0, gdn_total_bytes = 0;
  uint64_t attention_layer_count = 0, kv_plane_count = 0,
           kv_plane_token_bytes = 0;
  uint64_t kv_plane_stride = 0, kv_pool_bytes = 0;
  uint64_t gdn_slot_origin = 0, private_row_stride = 0, map_origin = 0;
};
enum class QwenStateRole { GdnHistory, GdnMatrix, Key, Value };
struct QwenStateBinding {
  operations::ValueId value;
  size_t layer_slot = 0;
  uint32_t original_layer_index = 0;
  uint64_t ordinal = 0;
  QwenStateRole role = QwenStateRole::GdnHistory;
};
// Model-owned structure supplies these graph associations. Native staging
// validates them against the session's graph without depending on lib/qwen.
struct NativeQwenEpoch {
  operations::ValueId logical_epoch, logical_rows;
  loomify_gguf::Value epoch, rows, resident_arena;
  QwenNativeOptions options;
  std::vector<QwenStateBinding> states;
};
// Pure native-profile admission; failure leaves geometry unchanged.
bool DeriveQwenNativeGeometry(const model::ModelDescriptor &,
                              const QwenNativeOptions &, QwenNativeGeometry *,
                              std::string *error);
// Loads requested real registrations with one descriptor-derived configuration.
// Descriptor/options are borrowed. On success module/result are caller-owned;
// geometry is copied only after a successful diagnosed load. No layer stubs.
loomc_status_t
LoadQwenNative(loomc_context_t *, loomc_workspace_t *, const Registry &,
               const std::vector<Id> &, const model::ModelDescriptor &,
               const QwenNativeOptions &, const std::string &source_root,
               QwenNativeGeometry *out_geometry, loomc_module_t **out_module,
               loomc_result_t **out_result);
std::vector<Registration> QwenCommonRegistrations();
// Shared native dependencies for complete layer registrations, with no
// dependency on either numerical layer helper or full-registry assembly.
std::vector<NativeSource> QwenSharedSources();
// Checks loaded per-layer encoding selectors against the graph weight types.
// Pure preflight: no allocation, configuration replacement or native staging.
loomc_status_t CheckQwenLayerConfiguration(const BuildContext &);
// The actual layer command takes one parameter root. Callers provide its
// explicit role/key contract; this preflight never stages or picks a root.
loomc_status_t CheckQwenLayerParameters(
    const BuildContext &,
    const std::vector<std::pair<operations::Port, std::string>> &,
    loomify_gguf::Value *parameter_root);
} // namespace hrx::dispatch
#endif
