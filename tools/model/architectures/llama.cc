// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
namespace hrx::model::internal {
const char *LlamaTensorSuffix(std::string_view role) {
  return role == "post_norm" ? "ffn_norm.weight" : nullptr;
}
void LoadLlama(const Loader &l, ModelDescriptor &m) {
  for (const auto &tensor : l.catalog.tensors) {
    const auto &name = tensor.name;
    if (name.rfind("blk.", 0) != 0)
      continue;
    size_t end = 4;
    while (end < name.size() && name[end] >= '0' && name[end] <= '9')
      ++end;
    if (end > 4 && (name.substr(end) == ".attn_q_norm.weight" ||
                    name.substr(end) == ".attn_k_norm.weight"))
      l.Fail("unsupported Llama Q/K normalization tensor: " + name);
  }

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
  }
  l.Bind(m, m.globals, "rope_frequencies", {m.rope.width / 2}, std::nullopt,
         true);
}
} // namespace hrx::model::internal
