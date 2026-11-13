// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/assembly_registry.h"
#include "tools/model/model_descriptor_internal.h"
namespace hrx::model::internal {
namespace {
const char *Spark25TensorSuffix(std::string_view role, bool layer) {
  if (!layer)
    return nullptr;
  if (role == "query_key_value")
    return "attn_qkv.weight";
  if (role == "attention_gate")
    return "attn_gate.weight";
  if (role == "ffn_norm")
    return "ffn_norm.weight";
  return nullptr;
}
void LoadSpark25(const Loader &l, ModelDescriptor &m) {
  ReadDecoderDimensions(l, m);
  ReadRope(l, m, l.Real("rope.freq_base"));
  const auto window = l.Int("attention.sliding_window", {}, false);
  const auto localbase = l.Real("rope.freq_base_swa");
  const auto localwidth = l.Int("rope.dimension_count_swa");
  if (localwidth > m.key_head_width || localwidth % 2)
    l.Fail("invalid Spark local RoPE width");
  const auto *pattern = l.Field("attention.sliding_window_pattern");
  if (!pattern || pattern->source_type != 9 || pattern->element_type != 7 ||
      pattern->elements_omitted || pattern->element_count != m.layers.size() ||
      pattern->elements.size() != m.layers.size())
    l.Fail("Spark requires a per-layer boolean sliding-window map");
  RejectHeadNorms(l, "Spark");
  BindGlobals(l, m);
  for (auto &d : m.layers) {
    const auto *local = std::get_if<bool>(&pattern->elements[d.index]);
    if (!local)
      l.Fail("Spark sliding-window map must be boolean");
    if (*local && !window)
      l.Fail("Spark local layer requires positive sliding window");
    d.attention_ordinal = d.index;
    d.attention_window = *local ? window : 0;
    d.rope_base = *local ? localbase : m.rope.base;
    d.rope_width = *local ? localwidth : m.rope.width;
    const auto e = m.embedding_width;
    const auto q = Product(d.query_heads, m.key_head_width);
    const auto kv = Product(d.kv_heads, m.key_head_width);
    if (q > uint64_t(INT64_MAX) - Product(2, kv))
      l.Fail("Spark fused QKV extent overflow");
    l.Bind(m, d.tensors, "input_norm", {e}, d.index);
    l.Bind(m, d.tensors, "ffn_norm", {e}, d.index);
    BindFeedForward(l, m, d);
    l.Bind(m, d.tensors, "query_key_value", {e, q + 2 * kv}, d.index);
    l.Bind(m, d.tensors, "attention_gate", {e, d.query_heads}, d.index);
    l.Bind(m, d.tensors, "attention_output", {q, e}, d.index);
    for (const auto &t : l.catalog.tensors) {
      const auto p = "blk." + std::to_string(d.index) + ".";
      if (t.name == p + "post_attention_norm.weight" ||
          t.name == p + "post_ffw_norm.weight" ||
          t.name == p + "attn_q.weight" || t.name == p + "attn_k.weight" ||
          t.name == p + "attn_v.weight")
        l.Fail("unsupported Spark attention/postnorm tensor: " + t.name);
    }
  }
}
} // namespace
const ModelArchitecture kSpark25Architecture{"spark2_5", LoadSpark25,
                                             Spark25TensorSuffix};
} // namespace hrx::model::internal
