// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_TESTING_FIXTURE_H_
#define HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_TESTING_FIXTURE_H_
#include "tools/model/model_descriptor.h"
namespace hrx::loomify_gguf::testing {
inline model::ModelDescriptor Fixture(bool llama = true, size_t layers = 3) {
  model::ModelDescriptor m;
  m.family = llama ? "llama" : "qwen35";
  m.embedding_width = llama ? 3072 : 1024;
  m.vocabulary_size = 128256;
  m.key_head_width = m.value_head_width = 128;
  m.rms_epsilon = llama ? 1e-5 : 1e-6;
  m.rope.width = 128;
  m.rope.base = 500000;
  m.output_tied_to_embedding = true;
  uint64_t offset = 4096;
  auto tensor = [&](std::string role, std::string key, uint32_t tag, uint64_t a,
                    uint64_t b = 1) {
    model::TensorRecord t;
    t.name = key;
    t.rank = b == 1 ? 1 : 2;
    t.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
    t.storage_type = tag;
    t.byte_size = tag == 0 ? a * b * 4
                           : a / 256 * b *
                                 (tag == 12   ? 144
                                  : tag == 13 ? 176
                                              : 210);
    t.absolute_offset = offset;
    offset += (t.byte_size + 255) / 256 * 256;
    return model::TensorBinding{role, t};
  };
  const uint64_t E = m.embedding_width, F = llama ? 8192 : 3584;
  if (llama) {
    m.globals = {tensor("token_embedding", "token_embd.weight", 14, E, 128256),
                 tensor("output_norm", "output_norm.weight", 0, E),
                 tensor("rope_frequencies", "rope_freqs.weight", 0, 64)};
    auto output = m.globals[0];
    output.role = "output";
    m.globals.push_back(output);
  }
  for (size_t i = 0; i < layers; ++i) {
    model::LayerDescriptor l;
    l.index = i;
    l.feed_forward_width = F;
    l.query_heads = 24;
    l.kv_heads = 8;
    auto prefix = "blk." + std::to_string(i) + ".";
    const auto gate = llama ? 12u : 12u + uint32_t(i % 3);
    l.tensors = {tensor("post_norm",
                        prefix + (llama ? "ffn_norm.weight"
                                        : "post_attention_norm.weight"),
                        0, E),
                 tensor("ffn_gate", prefix + "ffn_gate.weight", gate, E, F),
                 tensor("ffn_up", prefix + "ffn_up.weight", gate, E, F),
                 tensor("ffn_down", prefix + "ffn_down.weight",
                        llama ? 12 : 13 + i % 2, F, E)};
    if (llama) {
      l.tensors.push_back(
          tensor("input_norm", prefix + "attn_norm.weight", 0, E));
      l.tensors.push_back(
          tensor("query", prefix + "attn_q.weight", i % 2 ? 14 : 12, E, E));
      l.tensors.push_back(tensor("key", prefix + "attn_k.weight", 14, E, 1024));
      l.tensors.push_back(
          tensor("value", prefix + "attn_v.weight", 12, E, 1024));
      l.tensors.push_back(
          tensor("attention_output", prefix + "attn_output.weight", 12, E, E));
    }
    m.layers.push_back(l);
  }
  return m;
}
} // namespace hrx::loomify_gguf::testing
#endif
