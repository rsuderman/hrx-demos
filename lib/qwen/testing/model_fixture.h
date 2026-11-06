// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_QWEN_TESTING_MODEL_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_QWEN_TESTING_MODEL_FIXTURE_H_
#include "tools/model/model_descriptor.h"
namespace hrx::qwen::testing {
inline model::ModelDescriptor ModelFacts(size_t layers = 24,
                                         bool small = false) {
  model::ModelDescriptor f;
  f.family = "qwen35";
  f.name = "Qwen structural fixture";
  f.embedding_width = small ? 256 : 1024;
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
  s.value_heads = small ? 4 : 16;
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
    t.tensor.byte_size = storage == 0 ? a * b * 4
                                      : a / 256 * b *
                                            (storage == 12   ? 144
                                             : storage == 13 ? 176
                                                             : 210);
    t.tensor.absolute_offset = next;
    next += (t.tensor.byte_size + 255) / 256 * 256;
    return t;
  };
  f.globals = {
      tensor("token_embedding", "token_embd.weight", 14, f.embedding_width,
             f.vocabulary_size),
      tensor("output_norm", "output_norm.weight", 0, f.embedding_width)};
  auto output = f.globals[0];
  output.role = "output";
  f.globals.push_back(output);
  f.output_tied_to_embedding = true;
  uint32_t attention = 0, gdn = 0;
  for (size_t i = 0; i < layers; ++i) {
    model::LayerDescriptor l;
    l.index = i;
    l.feed_forward_width = small ? 512 : 3584;
    l.query_heads = 8;
    l.kv_heads = 2;
    auto p = "blk." + std::to_string(i) + ".";
    auto h = f.embedding_width, ff = l.feed_forward_width;
    l.tensors = {tensor("input_norm", p + "attn_norm.weight", 0, h),
                 tensor("post_norm", p + "post_attention_norm.weight", 0, h),
                 tensor("ffn_gate", p + "ffn_gate.weight", 12 + i % 3, h, ff),
                 tensor("ffn_up", p + "ffn_up.weight", 12 + i % 3, h, ff),
                 tensor("ffn_down", p + "ffn_down.weight", 13 + i % 2, ff, h)};
    auto add = [&](model::TensorBinding t) {
      l.tensors.push_back(std::move(t));
    };
    if ((i + 1) % 4) {
      l.kind = model::LayerKind::kGatedDeltaNet;
      l.recurrent_ordinal = gdn++;
      add(tensor("recurrent_qkv", p + "attn_qkv.weight", 14, h, channels));
      add(tensor("recurrent_gate", p + "attn_gate.weight", 14, h,
                 s.inner_width));
      add(tensor("convolution", p + "ssm_conv1d.weight", 0, 4, channels));
      add(tensor("dt_bias", p + "ssm_dt.bias", 0, s.value_heads));
      add(tensor("decay", p + "ssm_a", 0, s.value_heads));
      add(tensor("alpha", p + "ssm_alpha.weight", 13, h, s.value_heads));
      add(tensor("beta", p + "ssm_beta.weight", 13, h, s.value_heads));
      add(tensor("recurrent_norm", p + "ssm_norm.weight", 0, s.state_width));
      add(tensor("recurrent_output", p + "ssm_out.weight", 13, s.inner_width,
                 h));
    } else {
      l.attention_ordinal = attention++;
      auto q = l.query_heads * f.key_head_width,
           k = l.kv_heads * f.key_head_width;
      add(tensor("query", p + "attn_q.weight", 14, h, 2 * q));
      add(tensor("key", p + "attn_k.weight", 13, h, k));
      add(tensor("value", p + "attn_v.weight", 14, h, k));
      add(tensor("attention_output", p + "attn_output.weight", 13, q, h));
      add(tensor("query_norm", p + "attn_q_norm.weight", 0, f.key_head_width));
      add(tensor("key_norm", p + "attn_k_norm.weight", 0, f.key_head_width));
    }
    f.layers.push_back(std::move(l));
  }
  return f;
}
} // namespace hrx::qwen::testing
#endif
