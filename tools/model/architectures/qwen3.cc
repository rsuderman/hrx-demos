// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
#include "tools/model/architectures/qwen3.h"
namespace hrx::model::internal {
namespace {
const char *Qwen3TensorSuffix(std::string_view role, bool is_layer) {
  if (!is_layer) return nullptr;

  if (role == "post_norm")
    return "ffn_norm.weight";
  return HeadNormTensorSuffix(role);
}
void LoadQwen3(const Loader &l, ModelDescriptor &m) {
  l.Unsupported("attention.sliding_window");
  ReadDecoderDimensions(l, m);
  if (m.rope.width != m.key_head_width)
    l.Fail("invalid RoPE width");
  ReadRope(l, m, l.Real("rope.freq_base", 10000));
  BindGlobals(l, m);
  for (auto &d : m.layers) {
    d.attention_ordinal = d.index;
    l.Bind(m, d.tensors, "input_norm", {m.embedding_width}, d.index);
    l.Bind(m, d.tensors, "post_norm", {m.embedding_width}, d.index);
    BindFeedForward(l, m, d);
    BindAttention(l, m, d, Product(m.key_head_width, d.query_heads));
    BindHeadNorms(l, m, d);
  }
}
} // namespace
const ModelArchitecture kQwen3Architecture{
    "qwen3", LoadQwen3, Qwen3TensorSuffix};
} // namespace hrx::model::internal
