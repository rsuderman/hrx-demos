// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_GEMMA3_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_GEMMA3_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <cstdlib>
namespace hrx::loomify_gguf::testing::gemma3 {
inline void Integer(model::GgufCatalog &c, const char *key, uint64_t value) {
  c.metadata.push_back({std::string("gemma3.") + key, 4, value});
}
inline void Real(model::GgufCatalog &c, const char *key, double value) {
  c.metadata.push_back({std::string("gemma3.") + key, 6, value});
}
inline void Tensor(model::GgufCatalog &c, const std::string &name, uint64_t a,
                   uint64_t b, uint32_t tag) {
  model::TensorRecord t;
  t.name = name;
  t.rank = b == 1 ? 1 : 2;
  t.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
  t.storage_type = tag;
  if (!model::EncodedTensorByteSize(tag, a, b, &t.byte_size))
    std::abort();
  t.absolute_offset = 64 + c.tensors.size() * uint64_t(1 << 20);
  c.tensors.push_back(t);
}
inline model::GgufCatalog Catalog() {
  model::GgufCatalog c;
  c.architecture = "gemma3";
  for (auto p : std::vector<std::pair<const char *, uint64_t>>{
           {"block_count", 6},
           {"embedding_length", 1152},
           {"context_length", 32768},
           {"feed_forward_length", 6912},
           {"attention.head_count", 4},
           {"attention.head_count_kv", 1},
           {"attention.key_length", 256},
           {"attention.value_length", 256},
           {"vocab_size", 256},
           {"attention.sliding_window", 512}})
    Integer(c, p.first, p.second);
  Real(c, "attention.layer_norm_rms_epsilon", 1e-6);
  Real(c, "rope.freq_base", 1e6);
  Tensor(c, "token_embd.weight", 1152, 256, 8);
  Tensor(c, "output_norm.weight", 1152, 1, 0);
  for (int i = 0; i < 6; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    for (auto n : {"attn_norm.weight", "post_attention_norm.weight",
                   "ffn_norm.weight", "post_ffw_norm.weight"})
      Tensor(c, p + n, 1152, 1, 0);
    Tensor(c, p + "attn_q_norm.weight", 256, 1, 0);
    Tensor(c, p + "attn_k_norm.weight", 256, 1, 0);
    Tensor(c, p + "attn_q.weight", 1152, 1024, 6);
    Tensor(c, p + "attn_k.weight", 1152, 256, 6);
    Tensor(c, p + "attn_v.weight", 1152, 256, i % 2 ? 8 : 6);
    Tensor(c, p + "attn_output.weight", 1024, 1152, 12);
    Tensor(c, p + "ffn_gate.weight", 1152, 6912, 6);
    Tensor(c, p + "ffn_up.weight", 1152, 6912, 6);
    Tensor(c, p + "ffn_down.weight", 6912, 1152, i % 2 ? 12 : 14);
  }
  return c;
}

inline model::ModelDescriptor ModelFacts() {
  model::ModelDescriptor m;
  std::string error;
  if (!model::LoadModelDescriptor(Catalog(), &m, &error))
    std::abort();
  return m;
}
} // namespace hrx::loomify_gguf::testing::gemma3
#endif
