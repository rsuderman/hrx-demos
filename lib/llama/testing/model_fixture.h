// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LLAMA_TESTING_MODEL_FIXTURE_H_
#define HRX_SOLUTIONS_LLAMA_TESTING_MODEL_FIXTURE_H_
#include "tools/model/model_descriptor.h"
namespace hrx::llama::testing {
inline model::ModelDescriptor ModelFacts(size_t layers = 2,
                                         uint64_t hidden = 3072,
                                         uint64_t ffn = 8192,
                                         uint64_t head = 128) {
  model::ModelDescriptor f;
  f.family = "llama";
  f.embedding_width = hidden;
  f.vocabulary_size = 128256;
  f.key_head_width = f.value_head_width = head;
  f.rms_epsilon = 1e-5;
  f.rope.width = head;
  f.rope.base = 500000;
  uint64_t offset = 4096;
  auto weight = [&](std::string role, std::string name, uint32_t storage,
                    uint64_t a, uint64_t b = 1) {
    model::TensorBinding p;
    p.role = role;
    p.tensor.name = name;
    p.tensor.storage_type = storage;
    p.tensor.rank = b == 1 ? 1 : 2;
    p.tensor.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
    p.tensor.byte_size =
        storage == 0 ? a * b * 4 : a / 256 * b * (storage == 12 ? 144 : 210);
    p.tensor.absolute_offset = offset;
    offset += (p.tensor.byte_size + 255) / 256 * 256;
    return p;
  };
  f.globals = {weight("rope_frequencies", "rope_freqs.weight", 0, head / 2),
               weight("token_embedding", "token_embd.weight", 14, hidden,
                      f.vocabulary_size),
               weight("output_norm", "output_norm.weight", 0, hidden)};
  auto output = f.globals[1];
  output.role = "output";
  f.globals.push_back(output);
  f.output_tied_to_embedding = true;
  for (size_t i = 0; i < layers; ++i) {
    model::LayerDescriptor l;
    l.index = 7 + 4 * i;
    l.query_heads = hidden / head;
    l.kv_heads = 8;
    l.feed_forward_width = ffn;
    auto prefix = "blk." + std::to_string(l.index);
    l.tensors = {
        weight("input_norm", prefix + ".attn_norm.weight", 0, hidden),
        weight("query", prefix + ".attn_q.weight", 12, hidden, hidden),
        weight("key", prefix + ".attn_k.weight", 14, hidden, 8 * head),
        weight("value", prefix + ".attn_v.weight", 14, hidden, 8 * head),
        weight("attention_output", prefix + ".attn_output.weight", 12, hidden,
               hidden),
        weight("post_norm", prefix + ".ffn_norm.weight", 0, hidden),
        weight("ffn_gate", prefix + ".ffn_gate.weight", 12, hidden, ffn),
        weight("ffn_up", prefix + ".ffn_up.weight", 12, hidden, ffn),
        weight("ffn_down", prefix + ".ffn_down.weight", 12, ffn, hidden)};
    f.layers.push_back(std::move(l));
  }
  return f;
}
} // namespace hrx::llama::testing
#endif
