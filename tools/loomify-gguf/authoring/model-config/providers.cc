// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model-config/providers.h"
namespace hrx::loomify_gguf {
ConfiguredLinkRequest MakeLlamaLinkedProviders(const std::string &source_root, bool relocatable) {
  ConfiguredLinkRequest request;
  request.source_root = source_root;
  if (relocatable) {
    request.providers.push_back({"lib/kernels/llama/configured_providers.loombc"});
  } else {
    for (const char *path : {
        "lib/kernels/llama/norm/norm.loom",
        "lib/kernels/normalization/rmsnorm_row.loom",
        "lib/kernels/elementwise/residual_row.loom",
        "lib/kernels/elementwise/swiglu_row.loom",
        "lib/kernels/llama/active_rows.loom",
        "lib/kernels/llama/ffn/ffn.loom",
        "lib/kernels/elementwise/ffn_launches.loom",
        "lib/kernels/projection/q4k.loom",
        "lib/kernels/projection/q4k_compat.loom",
        "lib/kernels/projection/q6k.loom",
        "lib/kernels/projection/q6k_compat.loom",
        "lib/kernels/projection/qk_common.loom",
        "lib/kernels/projection/qk_mmvq_compat.loom",
        "lib/kernels/quantization/q8_1_x4.loom",
        "lib/kernels/quantization/q8_1_x4_mmvq_compat.loom",
        "lib/kernels/llama/output/head_count.loom",
        "lib/kernels/llama/output/output.loom",
        "lib/kernels/llama/output/output_interface.loom",
        "lib/kernels/llama/output/output_norm_last.loom",
        "lib/kernels/selection/greedy_argmax.loom",
        "lib/kernels/llama/publication/publication.loom",
        "lib/kernels/llama/publication/publication_interface.loom",
        "lib/kernels/llama/attention/attention_prepare.loom",
        "lib/kernels/llama/attention/attention_native.loom",
        "lib/kernels/llama/attention/rotary.loom",
        "lib/kernels/attention/core.loom",
        "lib/kernels/attention/kv_position.loom",
        "lib/kernels/llama/attention/attention_materialized.loom",
        "lib/kernels/llama/embedding/embedding.loom",
        "lib/kernels/embedding/q6k_rows.loom",
    }) request.providers.push_back({path});
  }
  return request;
}
ConfiguredLinkRequest MakeQwenFfnLinkedProviders(const std::string &source_root, bool relocatable) {
  ConfiguredLinkRequest request;
  request.source_root = source_root;
  if (relocatable) {
    request.providers.push_back({"lib/kernels/qwen/configured_providers.loombc"});
  } else {
    for (const char *path : {
        "lib/kernels/qwen/common/shared_interface.loom",
        "lib/kernels/quantization/q8_1_x4_canonical.loom",
        "lib/kernels/projection/q4k_canonical.loom",
        "lib/kernels/projection/q5k.loom",
        "lib/kernels/projection/q6k_canonical.loom",
        "lib/kernels/qwen/common/layer_decode.loom",
        "lib/kernels/qwen/ffn/sequence.loom",
        "lib/kernels/qwen/common/layer_prefill.loom",
        "lib/kernels/qwen/ffn/children.loom",
        "lib/kernels/elementwise/ffn_launches.loom",
        "lib/kernels/projection/q4k.loom",
        "lib/kernels/projection/q6k.loom",
        "lib/kernels/projection/qk_common.loom",
        "lib/kernels/quantization/q8_1_x4.loom",
        "lib/kernels/normalization/rmsnorm_row.loom",
        "lib/kernels/elementwise/residual_row.loom",
        "lib/kernels/elementwise/swiglu_row.loom",
        "lib/kernels/qwen/common/spans.loom",
        "lib/kernels/attention/kv_position.loom",
    }) request.providers.push_back({path});
  }
  return request;
}
} // namespace hrx::loomify_gguf
