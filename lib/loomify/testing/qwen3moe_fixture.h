// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_QWEN3MOE_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_QWEN3MOE_FIXTURE_H_

#include "tools/model/model_descriptor.h"
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::testing::qwen3moe {
// Header facts only. Synthetic layer parity deliberately exercises independent
// V/down tags; actual checkpoint layer assignments are checked from its header.
inline model::GgufCatalog Catalog(uint32_t layers = 2, bool q8_backbone = false,
                                  uint64_t dense_width = 5472) {
  model::GgufCatalog c;
  c.architecture = "qwen3moe";
  c.name = "Qwen3MoE header fixture";
  for (auto [key, value] : std::vector<std::pair<const char *, uint64_t>>{
           {"block_count", layers},
           {"embedding_length", 2048},
           {"context_length", 262144},
           {"feed_forward_length", dense_width},
           {"attention.head_count", 32},
           {"attention.head_count_kv", 4},
           {"attention.key_length", 128},
           {"attention.value_length", 128},
           {"expert_count", 128},
           {"expert_used_count", 8},
           {"expert_feed_forward_length", 768}})
    c.metadata.push_back({std::string("qwen3moe.") + key, 4, value});
  c.metadata.push_back({"qwen3moe.rope.freq_base", 6, 1e7});
  c.metadata.push_back({"qwen3moe.attention.layer_norm_rms_epsilon", 6, 1e-6});
  model::MetadataRecord vocabulary;
  vocabulary.key = "tokenizer.ggml.tokens";
  vocabulary.source_type = 9;
  vocabulary.element_type = 8;
  vocabulary.element_count = 151936;
  vocabulary.elements_omitted = true;
  c.metadata.push_back(vocabulary);
  uint64_t next = 4096;
  const auto tensor = [&](std::string name, uint32_t tag, uint64_t a,
                          uint64_t b = 1, uint64_t x = 1) {
    model::TensorRecord t;
    t.name = std::move(name);
    t.rank = x != 1 ? 3 : b != 1 ? 2 : 1;
    t.normalized_extents = {int64_t(a), int64_t(b), int64_t(x), 1};
    t.storage_type = tag;
    if (!model::EncodedTensorByteSize(tag, a, b * x, &t.byte_size))
      std::abort();
    t.absolute_offset = next;
    next += (t.byte_size + 31) / 32 * 32;
    c.tensors.push_back(std::move(t));
  };
  const uint32_t matrix = q8_backbone ? 8 : 12;
  tensor("token_embd.weight", matrix, 2048, 151936);
  tensor("output.weight", q8_backbone ? 8 : 14, 2048, 151936);
  tensor("output_norm.weight", 0, 2048);
  for (uint32_t i = 0; i < layers; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    const uint32_t varied = i % 2 ? 14 : 12;
    tensor(p + "attn_norm.weight", 0, 2048);
    tensor(p + "ffn_norm.weight", 0, 2048);
    tensor(p + "attn_q_norm.weight", 0, 128);
    tensor(p + "attn_k_norm.weight", 0, 128);
    tensor(p + "attn_q.weight", matrix, 2048, 4096);
    tensor(p + "attn_k.weight", matrix, 2048, 512);
    tensor(p + "attn_v.weight", q8_backbone ? 8 : varied, 2048, 512);
    tensor(p + "attn_output.weight", matrix, 4096, 2048);
    tensor(p + "ffn_gate_inp.weight", 0, 2048, 128);
    tensor(p + "ffn_gate_exps.weight", 12, 2048, 768, 128);
    tensor(p + "ffn_up_exps.weight", 12, 2048, 768, 128);
    tensor(p + "ffn_down_exps.weight", q8_backbone ? 12 : varied, 768, 2048,
           128);
  }
  return c;
}
} // namespace hrx::loomify_gguf::testing::qwen3moe
#endif
