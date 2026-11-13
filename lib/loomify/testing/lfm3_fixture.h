// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_LFM3_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_LFM3_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <cstdlib>
namespace hrx::loomify_gguf::testing {
inline model::GgufCatalog Lfm3Catalog(uint64_t hidden = 2048,
                                      bool tied = true) {
  model::GgufCatalog c;
  c.architecture = "lfm3";
  c.name = "LFM3 constructed fixture";
  auto integer = [&](const char *key, uint64_t n) {
    c.metadata.push_back({std::string("lfm3.") + key, 4, n});
  };
  for (const auto &v : std::vector<std::pair<const char *, uint64_t>>{
           {"block_count", 3},
           {"embedding_length", hidden},
           {"context_length", 262144},
           {"feed_forward_length", 3072},
           {"attention.head_count", 16},
           {"attention.key_length", 128},
           {"attention.value_length", 128},
           {"vocab_size", 1024},
           {"rope.dimension_count", 128},
           {"attention.sliding_window", 256},
           {"shortconv.l_cache", 7}})
    integer(v.first, v.second);
  for (const auto &v : std::vector<std::pair<const char *, double>>{
           {"attention.layer_norm_rms_epsilon", 1e-5},
           {"rope.freq_base", 1e6},
           {"rope.freq_base_swa", 1e4}})
    c.metadata.push_back({std::string("lfm3.") + v.first, 6, v.second});
  model::MetadataRecord kv;
  kv.key = "lfm3.attention.head_count_kv";
  kv.source_type = 9;
  kv.element_type = 4;
  kv.element_count = 3;
  kv.elements = {uint64_t(0), uint64_t(4), uint64_t(4)};
  c.metadata.push_back(kv);
  model::MetadataRecord window;
  window.key = "lfm3.attention.sliding_window_pattern";
  window.source_type = 9;
  window.element_type = 7;
  window.element_count = 3;
  window.elements = {false, false, true};
  c.metadata.push_back(window);
  uint64_t next = 4096;
  auto tensor = [&](const std::string &name, uint32_t tag, uint64_t a,
                    uint64_t b = 1) {
    model::TensorRecord t;
    t.name = name;
    t.rank = b == 1 ? 1 : 2;
    t.storage_type = tag;
    t.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
    if (!model::EncodedTensorByteSize(tag, a, b, &t.byte_size))
      std::abort();
    t.absolute_offset = next;
    next += (t.byte_size + 255) / 256 * 256;
    c.tensors.push_back(t);
  };
  tensor("token_embd.weight", 13, hidden, 1024);
  tensor("output_norm.weight", 0, hidden);
  tensor("rope_freqs.weight", 0, 64);
  if (!tied)
    tensor("output.weight", 14, hidden, 1024);
  for (uint32_t i = 0; i < 3; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    for (const char *suffix : {"attn_norm.weight", "post_attention_norm.weight",
                               "ffn_norm.weight", "post_ffw_norm.weight"})
      tensor(p + suffix, 0, hidden);
    tensor(p + "ffn_gate.weight", 12, hidden, 3072);
    tensor(p + "ffn_up.weight", 14, hidden, 3072);
    tensor(p + "ffn_down.weight", 13, 3072, hidden);
    if (i == 0) {
      tensor(p + "shortconv.conv.weight", 0, 7, hidden);
      tensor(p + "shortconv.in_proj.weight", 12, hidden, 3 * hidden);
      tensor(p + "shortconv.out_proj.weight", 8, hidden, hidden);
    } else {
      tensor(p + "attn_q.weight", 12, hidden, 2048);
      tensor(p + "attn_k.weight", 8, hidden, 512);
      tensor(p + "attn_v.weight", 14, hidden, 512);
      tensor(p + "attn_output.weight", 8, 2048, hidden);
      tensor(p + "attn_gate.weight", 0, hidden, 16);
      tensor(p + "attn_q_norm.weight", 0, 128);
      tensor(p + "attn_k_norm.weight", 0, 128);
    }
  }
  return c;
}
} // namespace hrx::loomify_gguf::testing
#endif
