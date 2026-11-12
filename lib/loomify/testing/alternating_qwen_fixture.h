// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_TESTING_ALTERNATING_QWEN_FIXTURE_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_TESTING_ALTERNATING_QWEN_FIXTURE_H_

#include "tools/model/model_descriptor.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace hrx::loomify_gguf::testing {
// Independently authored wire fixture, not derived from configuration output or
// the production checkpoint presets. Only its header is stored; tensor extents
// are sparse, so no model payload is allocated or downloaded.
inline bool LoadAlternatingQwenFixture(model::ModelDescriptor *out,
                                       model::GgufCatalog *catalog,
                                       std::string *error,
                                       uint32_t ffn_width = 3584,
                                       uint32_t query_tag = 14,
                                       bool tied_output = true) {
  auto number = [](std::string &s, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
      s.push_back(char(value >> (8 * i)));
  };
  auto string = [&](std::string &s, const std::string &value) {
    number(s, value.size(), 8);
    s += value;
  };
  std::string metadata;
  uint64_t metadata_count = 0;
  auto integer = [&](const char *suffix, uint32_t value) {
    string(metadata, std::string("qwen35.") + suffix);
    number(metadata, 4, 4);
    number(metadata, value, 4);
    ++metadata_count;
  };
  auto real = [&](const char *suffix, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    string(metadata, std::string("qwen35.") + suffix);
    number(metadata, 6, 4);
    number(metadata, bits, 4);
    ++metadata_count;
  };
  string(metadata, "general.architecture");
  number(metadata, 8, 4);
  string(metadata, "qwen35");
  ++metadata_count;
  string(metadata, "general.name");
  number(metadata, 8, 4);
  string(metadata, "independent eight-layer alternating fixture");
  ++metadata_count;
  integer("block_count", 8);
  integer("embedding_length", 1024);
  integer("context_length", 262144);
  integer("feed_forward_length", ffn_width);
  integer("attention.head_count", 8);
  integer("attention.head_count_kv", 2);
  integer("attention.key_length", 256);
  integer("attention.value_length", 256);
  integer("vocab_size", 248320);
  real("attention.layer_norm_rms_epsilon", 1e-6f);
  integer("rope.dimension_count", 64);
  real("rope.freq_base", 1e7f);
  string(metadata, "qwen35.rope.dimension_sections");
  number(metadata, 9, 4);
  number(metadata, 4, 4);
  number(metadata, 4, 8);
  for (uint32_t v : {11, 11, 10, 0})
    number(metadata, v, 4);
  ++metadata_count;
  integer("ssm.conv_kernel", 4);
  integer("ssm.inner_size", 2048);
  integer("ssm.state_size", 128);
  integer("ssm.group_count", 16);
  integer("ssm.time_step_rank", 16);
  // The explicit map must override this conflicting three-to-one fallback.
  integer("full_attention_interval", 4);
  string(metadata, "qwen35.attention.recurrent_layers");
  number(metadata, 9, 4);
  number(metadata, 7, 4);
  number(metadata, 8, 8);
  for (bool recurrent : {true, false, true, false, true, false, true, false})
    number(metadata, recurrent, 1);
  ++metadata_count;

  std::string tensors;
  uint64_t tensor_count = 0, payload_bytes = 0;
  auto tensor = [&](const std::string &name, uint32_t tag, uint64_t in,
                    uint64_t columns = 1) {
    string(tensors, name);
    number(tensors, columns == 1 ? 1 : 2, 4);
    number(tensors, in, 8);
    if (columns != 1)
      number(tensors, columns, 8);
    number(tensors, tag, 4);
    number(tensors, payload_bytes, 8);
    // Independently stated GGML wire sizes; do not use the product size helper
    // to manufacture the expected source records.
    const uint64_t bytes = tag == 0   ? in * columns * 4
                           : tag == 8 ? in / 32 * columns * 34
                                      : in / 256 * columns *
                                            (tag == 12   ? 144
                                             : tag == 13 ? 176
                                                         : 210);
    payload_bytes += (bytes + 31) / 32 * 32;
    ++tensor_count;
  };
  tensor("token_embd.weight", 14, 1024, 248320);
  tensor("output_norm.weight", 0, 1024);
  if (!tied_output)
    tensor("output.weight", 14, 1024, 248320);
  for (unsigned i = 0; i < 8; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    tensor(p + "attn_norm.weight", 0, 1024);
    tensor(p + "post_attention_norm.weight", 0, 1024);
    tensor(p + "ffn_gate.weight", 12, 1024, ffn_width);
    tensor(p + "ffn_up.weight", 12, 1024, ffn_width);
    tensor(p + "ffn_down.weight", 13, ffn_width, 1024);
    if (i % 2 == 0) {
      tensor(p + "attn_qkv.weight", 14, 1024, 6144);
      tensor(p + "attn_gate.weight", 14, 1024, 2048);
      tensor(p + "ssm_conv1d.weight", 0, 4, 6144);
      tensor(p + "ssm_dt.bias", 0, 16);
      tensor(p + "ssm_a", 0, 16);
      tensor(p + "ssm_alpha.weight", 13, 1024, 16);
      tensor(p + "ssm_beta.weight", 13, 1024, 16);
      tensor(p + "ssm_norm.weight", 0, 128);
      tensor(p + "ssm_out.weight", 13, 2048, 1024);
    } else {
      tensor(p + "attn_q.weight", query_tag, 1024, 4096);
      tensor(p + "attn_k.weight", 14, 1024, 512);
      tensor(p + "attn_v.weight", 14, 1024, 512);
      tensor(p + "attn_output.weight", 13, 2048, 1024);
      tensor(p + "attn_q_norm.weight", 0, 256);
      tensor(p + "attn_k_norm.weight", 0, 256);
    }
  }
  std::string bytes = "GGUF";
  number(bytes, 3, 4);
  number(bytes, tensor_count, 8);
  number(bytes, metadata_count, 8);
  bytes += metadata;
  bytes += tensors;
  while (bytes.size() % 32)
    bytes.push_back(0);
  const auto extent = bytes.size() + payload_bytes;
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(),
                                                     std::fclose);
  if (!file || std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0 ||
      std::fwrite(bytes.data(), 1, bytes.size(), file.get()) != bytes.size() ||
      fseeko(file.get(), int64_t(extent - 1), SEEK_SET) != 0 ||
      std::fputc(0, file.get()) == EOF) {
    *error = "failed to create independent sparse GGUF fixture";
    return false;
  }
  model::GgufCatalog loaded;
  if (!model::ReadGgufCatalog(file.get(), &loaded, error) ||
      !model::LoadModelDescriptor(loaded, out, error))
    return false;
  if (catalog)
    *catalog = std::move(loaded);
  return true;
}
} // namespace hrx::loomify_gguf::testing
#endif
