// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_QWEN2_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_QWEN2_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <algorithm>
#include <cstdlib>
namespace hrx::loomify_gguf::testing::qwen2 {
// Header-only geometry/data fixture. No checkpoint payloads or native objects.
inline model::GgufCatalog Catalog(uint64_t hidden = 1536, uint32_t layers = 28,
                                  uint64_t ffn = 8960) {
  model::GgufCatalog c;
  c.architecture = "qwen2";
  c.name = "Qwen2 configuration fixture";
  for (auto [key, value] : std::vector<std::pair<const char *, uint64_t>>{
           {"block_count", layers},
           {"embedding_length", hidden},
           {"context_length", 32768},
           {"feed_forward_length", ffn},
           {"attention.head_count", hidden / 128},
           {"attention.head_count_kv", 2}})
    c.metadata.push_back({std::string("qwen2.") + key, 4, value});
  c.metadata.push_back({"qwen2.rope.freq_base", 6, 1e6});
  c.metadata.push_back({"qwen2.attention.layer_norm_rms_epsilon", 6, 1e-6});
  // Exercise the actual checkpoint's missing vocab_size metadata fallback.
  model::MetadataRecord vocabulary;
  vocabulary.key = "tokenizer.ggml.tokens";
  vocabulary.source_type = 9;
  vocabulary.element_type = 8;
  vocabulary.element_count = 151936;
  vocabulary.elements_omitted = true;
  c.metadata.push_back(vocabulary);
  uint64_t offset = 4096;
  const auto tensor = [&](const std::string &name, uint32_t tag, uint64_t a,
                          uint64_t b = 1) {
    model::TensorRecord t;
    t.name = name;
    t.rank = b == 1 ? 1 : 2;
    t.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
    t.storage_type = tag;
    if (!model::EncodedTensorByteSize(tag, a, b, &t.byte_size))
      std::abort();
    t.absolute_offset = offset;
    offset += (t.byte_size + 31) / 32 * 32;
    c.tensors.push_back(std::move(t));
  };
  tensor("token_embd.weight", 14, hidden, 151936);
  tensor("output_norm.weight", 0, hidden);
  const std::vector<uint32_t> q6_layers{0,  1,  5,  6,  7,  8,  9,
                                        10, 13, 16, 19, 21, 24, 27};
  for (uint32_t i = 0; i < layers; ++i) {
    const auto prefix = "blk." + std::to_string(i) + ".";
    const uint32_t varied =
        std::find(q6_layers.begin(), q6_layers.end(), i) != q6_layers.end()
            ? 14
            : 12;
    tensor(prefix + "attn_norm.weight", 0, hidden);
    tensor(prefix + "ffn_norm.weight", 0, hidden);
    tensor(prefix + "attn_q.weight", 12, hidden, hidden);
    tensor(prefix + "attn_k.weight", 12, hidden, 256);
    tensor(prefix + "attn_v.weight", varied, hidden, 256);
    tensor(prefix + "attn_output.weight", 12, hidden, hidden);
    tensor(prefix + "ffn_gate.weight", 12, hidden, ffn);
    tensor(prefix + "ffn_up.weight", 12, hidden, ffn);
    tensor(prefix + "ffn_down.weight", varied, ffn, hidden);
    tensor(prefix + "attn_q.bias", 0, hidden);
    tensor(prefix + "attn_k.bias", 0, 256);
    tensor(prefix + "attn_v.bias", 0, 256);
  }
  return c;
}
inline model::ModelDescriptor
ModelFacts(uint64_t hidden = 1536, uint32_t layers = 28, uint64_t ffn = 8960) {
  model::ModelDescriptor m;
  std::string error;
  if (!model::LoadModelDescriptor(Catalog(hidden, layers, ffn), &m, &error))
    std::abort();
  return m;
}
} // namespace hrx::loomify_gguf::testing::qwen2
#endif
