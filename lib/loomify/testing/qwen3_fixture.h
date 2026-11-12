// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_QWEN3_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_QWEN3_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <cstdlib>
namespace hrx::loomify_gguf::testing {
inline model::ModelDescriptor Qwen3ModelFacts(uint64_t hidden = 1024,
                                              uint32_t layers = 2) {
  model::ModelDescriptor m;
  m.family = "qwen3";
  m.name = "Qwen3 configuration fixture";
  m.embedding_width = hidden;
  m.vocabulary_size = 151936;
  m.training_context = 40960;
  m.key_head_width = m.value_head_width = m.rope.width = 128;
  m.rms_epsilon = 1e-6;
  m.rope.base = 1000000;
  m.output_tied_to_embedding = true;
  uint64_t next = 4096;
  auto tensor = [&](std::string role, std::string name, uint32_t tag,
                    uint64_t a, uint64_t b = 1) {
    model::TensorRecord t;
    t.name = std::move(name);
    t.rank = b == 1 ? 1 : 2;
    t.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
    t.storage_type = tag;
    if (!model::EncodedTensorByteSize(tag, a, b, &t.byte_size))
      std::abort();
    t.absolute_offset = next;
    next += (t.byte_size + 255) / 256 * 256;
    return model::TensorBinding{std::move(role), std::move(t)};
  };
  m.globals = {tensor("token_embedding", "token_embd.weight", 14, hidden,
                      m.vocabulary_size),
               tensor("output_norm", "output_norm.weight", 0, hidden)};
  auto output = m.globals.front();
  output.role = "output";
  m.globals.push_back(std::move(output));
  for (uint32_t i = 0; i < layers; ++i) {
    model::LayerDescriptor l;
    l.index = i;
    l.attention_ordinal = i;
    l.feed_forward_width = 3072 + (i % 2) * 256;
    l.query_heads = 16;
    l.kv_heads = 8;
    l.rope_base = m.rope.base;
    const auto p = "blk." + std::to_string(i) + ".";
    l.tensors = {
        tensor("input_norm", p + "attn_norm.weight", 0, hidden),
        tensor("post_norm", p + "ffn_norm.weight", 0, hidden),
        tensor("query_norm", p + "attn_q_norm.weight", 0, 128),
        tensor("key_norm", p + "attn_k_norm.weight", 0, 128),
        tensor("query", p + "attn_q.weight", 12, hidden, 2048),
        tensor("key", p + "attn_k.weight", 12, hidden, 1024),
        tensor("value", p + "attn_v.weight", 14, hidden, 1024),
        tensor("attention_output", p + "attn_output.weight", 12, 2048, hidden),
        tensor("ffn_gate", p + "ffn_gate.weight", 12, hidden,
               l.feed_forward_width),
        tensor("ffn_up", p + "ffn_up.weight", 12, hidden, l.feed_forward_width),
        tensor("ffn_down", p + "ffn_down.weight", 14, l.feed_forward_width,
               hidden)};
    m.layers.push_back(std::move(l));
  }
  return m;
}
} // namespace hrx::loomify_gguf::testing
#endif
