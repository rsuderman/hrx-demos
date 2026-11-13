// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_SPARK2_5_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_SPARK2_5_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <cstdlib>
namespace hrx::loomify_gguf::testing {
inline model::GgufCatalog
Spark25Catalog(uint64_t hidden = 2560, uint64_t layers = 2, uint64_t heads = 16,
               uint64_t kvheads = 4, uint64_t ffn = 10240, bool tied = true,
               uint64_t vocab = 1024) {
  model::GgufCatalog c;
  c.architecture = "spark2_5";
  c.name = "Spark constructed fixture";
  for (const auto &v : std::vector<std::pair<const char *, uint64_t>>{
           {"block_count", layers},
           {"embedding_length", hidden},
           {"context_length", 1048576},
           {"feed_forward_length", ffn},
           {"attention.head_count", heads},
           {"attention.head_count_kv", kvheads},
           {"attention.key_length", 256},
           {"attention.value_length", 256},
           {"vocab_size", vocab},
           {"rope.dimension_count", 64},
           {"rope.dimension_count_swa", 256},
           {"attention.sliding_window", 512}})
    c.metadata.push_back({std::string("spark2_5.") + v.first, 4, v.second});
  for (const auto &v : std::vector<std::pair<const char *, double>>{
           {"attention.layer_norm_rms_epsilon", 1e-6},
           {"rope.freq_base", 5e6},
           {"rope.freq_base_swa", 1e4}})
    c.metadata.push_back({std::string("spark2_5.") + v.first, 6, v.second});
  model::MetadataRecord window;
  window.key = "spark2_5.attention.sliding_window_pattern";
  window.source_type = 9;
  window.element_type = 7;
  window.element_count = layers;
  for (uint64_t i = 0; i < layers; ++i)
    window.elements.push_back(i % 2 == 0);
  c.metadata.push_back(window);
  uint64_t next = 4096;
  auto tensor = [&](const std::string &name, uint32_t tag, uint64_t a,
                    uint64_t b = 1) {
    model::TensorRecord t;
    t.name = name;
    t.rank = name.find("norm.weight") != std::string::npos ? 1 : 2;
    t.storage_type = tag;
    t.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
    if (!model::EncodedTensorByteSize(tag, a, b, &t.byte_size))
      std::abort();
    t.absolute_offset = next;
    next += (t.byte_size + 255) / 256 * 256;
    c.tensors.push_back(t);
  };
  tensor("token_embd.weight", 14, hidden, vocab);
  tensor("output_norm.weight", 0, hidden);
  if (!tied)
    tensor("output.weight", 12, hidden, vocab);
  for (uint64_t i = 0; i < layers; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    tensor(p + "attn_norm.weight", 0, hidden);
    tensor(p + "ffn_norm.weight", 0, hidden);
    tensor(p + "attn_qkv.weight", hidden <= 17408 ? 14 : 12, hidden,
           (heads + 2 * kvheads) * 256);
    tensor(p + "attn_gate.weight", 0, hidden, heads);
    tensor(p + "attn_output.weight", 12, heads * 256, hidden);
    tensor(p + "ffn_gate.weight", 12, hidden, ffn);
    tensor(p + "ffn_up.weight", i % 2 ? 13 : 12, hidden, ffn);
    tensor(p + "ffn_down.weight", 12, ffn, hidden);
  }
  return c;
}
} // namespace hrx::loomify_gguf::testing
#endif
