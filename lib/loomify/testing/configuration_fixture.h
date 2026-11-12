// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_CONFIGURATION_TEST_FIXTURE_H_
#define HRX_SOLUTIONS_LOOMIFY_CONFIGURATION_TEST_FIXTURE_H_
#include "tools/model/model_descriptor.h"
#include <cstdlib>
namespace hrx::loomify_gguf::testing {
// Plain admitted full-profile descriptors. No native/compiler/graph objects.
inline model::ModelDescriptor LlamaModelFacts(
    uint64_t hidden = 3072, uint64_t vocabulary = 128256,
    uint64_t feed_forward = 8192, size_t layers = 28,
    uint64_t heads = 24, uint32_t embedding_tag = 14,
    bool tied = true, bool rope_factors = true) {
  constexpr bool llama = true;
  model::ModelDescriptor m;
  m.family = llama ? "llama" : "qwen35";
  m.embedding_width = hidden;
  m.vocabulary_size = vocabulary;
  m.key_head_width = m.value_head_width = 128;
  m.rms_epsilon = llama ? 1e-5 : 1e-6;
  m.rope.width = 128;
  m.rope.base = 500000;
  m.output_tied_to_embedding = tied;
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
  const uint64_t E = m.embedding_width, F = feed_forward;
  if (llama) {
    m.globals = {tensor("token_embedding", "token_embd.weight", embedding_tag,
                        E, vocabulary),
                 tensor("output_norm", "output_norm.weight", 0, E)};
    if (rope_factors)
      m.globals.push_back(tensor("rope_frequencies", "rope_freqs.weight", 0, 64));
    if (tied) {
      auto output = m.globals[0];
      output.role = "output";
      m.globals.push_back(output);
    } else {
      m.globals.push_back(tensor("output", "output.weight", 14, E, vocabulary));
    }
  }
  for (size_t i = 0; i < layers; ++i) {
    model::LayerDescriptor l;
    l.index = i;
    l.feed_forward_width = F;
    l.query_heads = heads;
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
inline model::ModelDescriptor LlamaPlainModelFacts() {
  auto m = LlamaModelFacts(2048, 110592, 7168, 24, 16, 12, false, false);
  m.rope.base = 1e8;
  m.training_context = 131072;
  // Deliberately mixed layer formats; the model must consume every row rather
  // than infer a uniform route from the first layer or filename.
  for (auto &l : m.layers) {
    const auto i = l.index;
    const bool q6 = i < 3 || i >= 20 || i == 5 || i == 8 || i == 11 ||
                    i == 14 || i == 17;
    for (auto &binding : l.tensors) {
      auto &t = binding.tensor;
      t.storage_type = binding.role == "input_norm" || binding.role == "post_norm"
                           ? 0
                           : (binding.role == "value" || binding.role == "ffn_down") && q6
                                 ? 14 : 12;
      if (!model::EncodedTensorByteSize(t.storage_type, t.normalized_extents[0],
                                      t.normalized_extents[1], &t.byte_size))
        std::abort();
    }
  }
  uint64_t offset = 4096;
  auto place = [&](auto &bindings) {
    for (auto &binding : bindings) {
      binding.tensor.absolute_offset = offset;
      offset += (binding.tensor.byte_size + 255) / 256 * 256;
    }
  };
  place(m.globals);
  for (auto &l : m.layers) place(l.tensors);
  return m;
}
inline model::ModelDescriptor QwenModelFacts(bool large = false) {
  const size_t layers = large ? 64 : 24;
  constexpr bool small = false;
  model::ModelDescriptor f;
  f.family = "qwen35";
  f.name = "Qwen structural fixture";
  f.embedding_width = large ? 5120 : 1024;
  f.vocabulary_size = small ? 256 : 248320;
  f.training_context = 262144;
  f.key_head_width = f.value_head_width = small ? 64 : 256;
  f.rms_epsilon = 1e-6;
  f.rope.width = small ? 32 : 64;
  f.rope.sections = small ? std::array<uint64_t, 4>{6, 5, 5, 0}
                          : std::array<uint64_t, 4>{11, 11, 10, 0};
  f.rope.base = 1e7;
  model::SsmDescriptor s;
  s.convolution_kernel = 4;
  s.state_width = small ? 64 : 128;
  s.key_heads = small ? 2 : 16;
  s.value_heads = large ? 48 : 16;
  s.inner_width = s.state_width * s.value_heads;
  auto channels = s.state_width * (2 * s.key_heads + s.value_heads);
  s.convolution_history_elements = 3 * channels;
  s.recurrent_state_elements = s.state_width * s.inner_width;
  f.ssm = s;
  uint64_t next = 4096;
  auto tensor = [&](std::string role, std::string name, uint32_t storage,
                    uint64_t a, uint64_t b = 1) {
    model::TensorBinding t;
    t.role = role;
    t.tensor.name = name;
    t.tensor.storage_type = storage;
    t.tensor.rank = b == 1 ? 1 : 2;
    t.tensor.normalized_extents = {int64_t(a), int64_t(b), 1, 1};
    if (!model::EncodedTensorByteSize(storage, a, b, &t.tensor.byte_size)) std::abort();
    t.tensor.absolute_offset = next;
    next += (t.tensor.byte_size + 255) / 256 * 256;
    return t;
  };
  f.globals = {
      tensor("token_embedding", "token_embd.weight", large ? 12 : 14, f.embedding_width,
             f.vocabulary_size),
      tensor("output_norm", "output_norm.weight", 0, f.embedding_width)};
  auto output = f.globals[0];
  output.role = "output";
  if (large) output = tensor("output", "output.weight", 14, f.embedding_width, f.vocabulary_size);
  f.globals.push_back(output);
  f.output_tied_to_embedding = !large;
  uint32_t attention = 0, gdn = 0;
  for (size_t i = 0; i < layers; ++i) {
    model::LayerDescriptor l;
    l.index = i;
    l.feed_forward_width = large ? 17408 : 3584;
    l.query_heads = large ? 24 : 8;
    l.kv_heads = large ? 4 : 2;
    auto p = "blk." + std::to_string(i) + ".";
    auto h = f.embedding_width, ff = l.feed_forward_width;
    l.tensors = {tensor("input_norm", p + "attn_norm.weight", 0, h),
                 tensor("post_norm", p + "post_attention_norm.weight", 0, h),
                 tensor("ffn_gate", p + "ffn_gate.weight", large ? 12 : 12 + i % 3, h, ff),
                 tensor("ffn_up", p + "ffn_up.weight", large ? 12 : 12 + i % 3, h, ff),
                 tensor("ffn_down", p + "ffn_down.weight", large ? 12 : 13 + i % 2, ff, h)};
    auto add = [&](model::TensorBinding t) {
      l.tensors.push_back(std::move(t));
    };
    if ((i + 1) % 4) {
      l.kind = model::LayerKind::kGatedDeltaNet;
      l.recurrent_ordinal = gdn++;
      add(tensor("recurrent_qkv", p + "attn_qkv.weight", large ? 8 : 14, h, channels));
      add(tensor("recurrent_gate", p + "attn_gate.weight", large ? 8 : 14, h,
                 s.inner_width));
      add(tensor("convolution", p + "ssm_conv1d.weight", 0, 4, channels));
      add(tensor("dt_bias", p + "ssm_dt.bias", 0, s.value_heads));
      add(tensor("decay", p + "ssm_a", 0, s.value_heads));
      add(tensor("alpha", p + "ssm_alpha.weight", large ? 8 : 13, h, s.value_heads));
      add(tensor("beta", p + "ssm_beta.weight", large ? 8 : 13, h, s.value_heads));
      add(tensor("recurrent_norm", p + "ssm_norm.weight", 0, s.state_width));
      add(tensor("recurrent_output", p + "ssm_out.weight", large ? 8 : 13, s.inner_width,
                 h));
    } else {
      l.attention_ordinal = attention++;
      auto q = l.query_heads * f.key_head_width,
           k = l.kv_heads * f.key_head_width;
      add(tensor("query", p + "attn_q.weight", large ? 8 : 14, h, 2 * q));
      add(tensor("key", p + "attn_k.weight", large ? 8 : 14, h, k));
      add(tensor("value", p + "attn_v.weight", large ? 8 : 14, h, k));
      add(tensor("attention_output", p + "attn_output.weight", large ? 14 : 13, q, h));
      add(tensor("query_norm", p + "attn_q_norm.weight", 0, f.key_head_width));
      add(tensor("key_norm", p + "attn_k_norm.weight", 0, f.key_head_width));
    }
    f.layers.push_back(std::move(l));
  }
  return f;
}
} // namespace hrx::loomify_gguf::testing
#endif
