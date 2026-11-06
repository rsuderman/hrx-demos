// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef LOOMIFY_QWEN_SERVING_TEST_FIXTURE_H_
#define LOOMIFY_QWEN_SERVING_TEST_FIXTURE_H_
#include "lib/dispatch/qwen.h"
namespace hrx::loomify_gguf::testing {
// Known admitted0.8B geometry for testing host publication, not numerical
// admission.
inline dispatch::QwenNativeGeometry Geometry(uint64_t pool = 512) {
  dispatch::QwenNativeGeometry g;
  g.embedding_width = 1024;
  g.feed_forward_width = 3584;
  g.vocabulary_size = 248320;
  g.residual_bytes = 2097152;
  g.gdn_history_bytes = 73728;
  g.gdn_recurrent_bytes = 1048576;
  g.gdn_layer_bytes = 1122304;
  g.gdn_layer_count = 18;
  g.gdn_total_bytes = 20201472;
  g.attention_layer_count = 6;
  g.kv_plane_count = 12;
  g.kv_plane_token_bytes = 1024;
  g.kv_plane_stride = pool * 1024;
  g.kv_pool_bytes = 12 * g.kv_plane_stride;
  g.gdn_slot_origin = 256;
  g.private_row_stride = 20204032;
  g.map_origin = 2048;
  return g;
}
} // namespace hrx::loomify_gguf::testing
#endif
