// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
#include "tools/model/architectures/llama.h"
namespace hrx::model::internal {
namespace {
const char *LlamaTensorSuffix(std::string_view role, bool is_layer) {
  if (!is_layer)
    return role == "rope_frequencies" ? "rope_freqs.weight" : nullptr;

  return role == "post_norm" ? "ffn_norm.weight" : nullptr;
}
void LoadLlama(const Loader &l, ModelDescriptor &m) {
  RejectHeadNorms(l, "Llama");
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
} // namespace
const ModelArchitecture kLlamaArchitecture{
    "llama", LoadLlama, LlamaTensorSuffix};
} // namespace hrx::model::internal
