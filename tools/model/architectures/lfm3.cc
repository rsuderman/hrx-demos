// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/architectures/lfm3.h"
#include "tools/model/model_descriptor_internal.h"
namespace hrx::model::internal {
namespace {
const char *Lfm3TensorSuffix(std::string_view role, bool is_layer) {
  if (!is_layer)
    return role == "rope_frequencies" ? "rope_freqs.weight" : nullptr;
  if (role == "attention_post_norm")
    return "post_attention_norm.weight";
  if (role == "ffn_norm")
    return "ffn_norm.weight";
  if (role == "ffn_post_norm")
    return "post_ffw_norm.weight";
  if (role == "attention_gate")
    return "attn_gate.weight";
  if (role == "convolution")
    return "shortconv.conv.weight";
  if (role == "convolution_input")
    return "shortconv.in_proj.weight";
  if (role == "convolution_output")
    return "shortconv.out_proj.weight";
  return HeadNormTensorSuffix(role);
}
void LoadLfm3(const Loader &l, ModelDescriptor &m) {
  ReadDecoderDimensions(l, m, true);
  ReadRope(l, m, l.Real("rope.freq_base"));
  if (m.rope.width != m.key_head_width)
    l.Fail("LFM3 rotary width must equal head width");
  BindGlobals(l, m);
  l.Bind(m, m.globals, "rope_frequencies", {m.key_head_width / 2});
  ShortConvolutionDescriptor conv;
  conv.width = m.embedding_width;
  conv.kernel = l.Int("shortconv.l_cache");
  if (conv.kernel < 2)
    l.Fail("LFM3 short-convolution requires at least two taps");
  conv.history_rows = conv.kernel - 1;
  m.short_convolution = conv;
  const auto *pattern = l.Field("attention.sliding_window_pattern");
  if (!pattern || pattern->source_type != 9 || pattern->elements_omitted ||
      pattern->element_count != m.layers.size() ||
      pattern->elements.size() != m.layers.size())
    l.Fail("LFM3 requires per-layer sliding-window map");
  const auto window = l.Int("attention.sliding_window");
  const auto localbase = l.Real("rope.freq_base_swa");
  uint32_t attention = 0, convolution = 0;
  for (auto &layer : m.layers) {
    const auto *local = std::get_if<bool>(&pattern->elements[layer.index]);
    if (!local)
      l.Fail("LFM3 sliding-window map must be boolean");
    const auto e = m.embedding_width;
    for (const char *role :
         {"input_norm", "attention_post_norm", "ffn_norm", "ffn_post_norm"})
      l.Bind(m, layer.tensors, role, {e}, layer.index);
    BindFeedForward(l, m, layer);
    if (!layer.kv_heads) {
      if (*local)
        l.Fail("short-convolution layer cannot have attention window");
      layer.kind = LayerKind::kShortConvolution;
      layer.convolution_ordinal = convolution++;
      l.Bind(m, layer.tensors, "convolution", {conv.kernel, e}, layer.index);
      l.Bind(m, layer.tensors, "convolution_input", {e, Product(e, 3)},
             layer.index);
      l.Bind(m, layer.tensors, "convolution_output", {e, e}, layer.index);
    } else {
      layer.attention_ordinal = attention++;
      layer.attention_window = *local ? window : 0;
      layer.rope_base = *local ? localbase : m.rope.base;
      BindAttention(l, m, layer, Product(m.key_head_width, layer.query_heads));
      BindHeadNorms(l, m, layer);
      l.Bind(m, layer.tensors, "attention_gate", {e, layer.query_heads},
             layer.index);
    }
  }
  if (!attention || !convolution)
    l.Fail("LFM3 requires attention and short-convolution layers");
}
} // namespace
const ModelArchitecture kLfm3Architecture{"lfm3", LoadLfm3, Lfm3TensorSuffix};
} // namespace hrx::model::internal
