// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
#include "tools/model/architectures/qwen2.h"
namespace hrx::model::internal {
namespace {
const char *Qwen2TensorSuffix(std::string_view role, bool is_layer) {
  if (!is_layer) return nullptr;

  if (role == "post_norm")
    return "ffn_norm.weight";
  if (role == "query_bias")
    return "attn_q.bias";
  if (role == "key_bias")
    return "attn_k.bias";
  if (role == "value_bias")
    return "attn_v.bias";
  return nullptr;
}
void LoadQwen2(const Loader &l, ModelDescriptor &m) {
  RejectHeadNorms(l, "Qwen2");
  l.Unsupported("attention.sliding_window");
  ReadDecoderDimensions(l, m);
  if (m.rope.width != m.key_head_width)
    l.Fail("invalid RoPE width");
  ReadRope(l, m, l.Real("rope.freq_base", 10000));
  BindGlobals(l, m);
  for (auto &d : m.layers) {
    d.attention_ordinal = d.index;
    const auto q = Product(m.key_head_width, d.query_heads);
    const auto kv = Product(m.key_head_width, d.kv_heads);
    l.Bind(m, d.tensors, "input_norm", {m.embedding_width}, d.index);
    l.Bind(m, d.tensors, "post_norm", {m.embedding_width}, d.index);
    BindFeedForward(l, m, d);
    BindAttention(l, m, d, q);
    l.Bind(m, d.tensors, "query_bias", {q}, d.index);
    l.Bind(m, d.tensors, "key_bias", {kv}, d.index);
    l.Bind(m, d.tensors, "value_bias", {kv}, d.index);
  }
}
} // namespace
const ModelArchitecture kQwen2Architecture{
    "qwen2", LoadQwen2, Qwen2TensorSuffix};
} // namespace hrx::model::internal
