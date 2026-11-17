// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_LFM2_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_LFM2_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <cstdlib>
namespace hrx::loomify_gguf::testing {
// Exact numerical directory geometry/formats shared by the two cached inputs.
// This constructs descriptors only; it does not contain checkpoint weights.
inline model::GgufCatalog Lfm2Catalog() {
  model::GgufCatalog c;
  c.architecture = "lfm2";
  c.name = "LFM2 constructed fixture";
  for (const auto &v : std::vector<std::pair<const char *, uint64_t>>{
           {"block_count", 16},
           {"embedding_length", 2048},
           {"context_length", 128000},
           {"feed_forward_length", 8192},
           {"attention.head_count", 32},
           {"vocab_size", 65536},
           {"shortconv.l_cache", 3}})
    c.metadata.push_back({std::string("lfm2.") + v.first, 4, v.second});
  c.metadata.push_back({"lfm2.attention.layer_norm_rms_epsilon", 6,
                        double(9.999999747378752e-6)});
  c.metadata.push_back({"lfm2.rope.freq_base", 6, double(1e6)});
  model::MetadataRecord kv;
  kv.key = "lfm2.attention.head_count_kv";
  kv.source_type = 9;
  kv.element_type = 4;
  kv.element_count = 16;
  for (uint64_t n : {0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0})
    kv.elements.push_back(n);
  c.metadata.push_back(kv);
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
  tensor("token_embd.weight", 14, 2048, 65536);
  tensor("token_embd_norm.weight", 0, 2048);
  for (uint32_t i = 0; i < 16; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    tensor(p + "attn_norm.weight", 0, 2048);
    tensor(p + "ffn_norm.weight", 0, 2048);
    tensor(p + "ffn_gate.weight", 12, 2048, 8192);
    tensor(p + "ffn_up.weight", 12, 2048, 8192);
    const bool q6down = i == 0 || i == 1 || i == 4 || i == 7 || i == 10 ||
                        i == 13 || i == 14 || i == 15;
    tensor(p + "ffn_down.weight", q6down ? 14 : 12, 8192, 2048);
    if (!std::get<uint64_t>(kv.elements[i])) {
      tensor(p + "shortconv.conv.weight", 0, 3, 2048);
      tensor(p + "shortconv.in_proj.weight", 12, 2048, 6144);
      tensor(p + "shortconv.out_proj.weight", 12, 2048, 2048);
    } else {
      tensor(p + "attn_q.weight", 12, 2048, 2048);
      tensor(p + "attn_k.weight", 12, 2048, 512);
      tensor(p + "attn_v.weight", i == 8 || i == 14 ? 14 : 12, 2048, 512);
      tensor(p + "attn_output.weight", 12, 2048, 2048);
      tensor(p + "attn_q_norm.weight", 0, 64);
      tensor(p + "attn_k_norm.weight", 0, 64);
    }
  }
  return c;
}
} // namespace hrx::loomify_gguf::testing
#endif
