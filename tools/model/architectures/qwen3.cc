// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
namespace hrx::model::internal {
const char *Qwen3TensorSuffix(std::string_view role) {
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
} // namespace hrx::model::internal
