// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/assembly_registry.h"
#include "tools/model/model_descriptor_internal.h"
namespace hrx::model::internal {
namespace {
const char *Lfm2TensorSuffix(std::string_view role, bool is_layer) {
  if (!is_layer)
    return role == "output_norm" ? "token_embd_norm.weight" : nullptr;
  if (role == "ffn_norm")
    return "ffn_norm.weight";
  if (role == "convolution")
    return "shortconv.conv.weight";
  if (role == "convolution_input")
    return "shortconv.in_proj.weight";
  if (role == "convolution_output")
    return "shortconv.out_proj.weight";
  return HeadNormTensorSuffix(role);
}
void LoadLfm2(const Loader &l, ModelDescriptor &m) {
  ReadDecoderDimensions(l, m, true);
  ReadRope(l, m, l.Real("rope.freq_base"));
  l.Unsupported("attention.sliding_window");
  l.Unsupported("attention.sliding_window_pattern");
  if (m.rope.width != m.key_head_width)
    l.Fail("LFM2 rotary width must equal head width");
  BindGlobals(l, m);
  ShortConvolutionDescriptor conv;
  conv.width = m.embedding_width;
  conv.kernel = l.Int("shortconv.l_cache");
  if (conv.kernel < 2)
    l.Fail("LFM2 short-convolution requires at least two taps");
  conv.history_rows = conv.kernel - 1;
  m.short_convolution = conv;
  uint32_t attention = 0, convolution = 0;
  for (auto &layer : m.layers) {
    const auto e = m.embedding_width;
    l.Bind(m, layer.tensors, "input_norm", {e}, layer.index);
    l.Bind(m, layer.tensors, "ffn_norm", {e}, layer.index);
    BindFeedForward(l, m, layer);
    if (!layer.kv_heads) {
      layer.kind = LayerKind::kShortConvolution;
      layer.convolution_ordinal = convolution++;
      l.Bind(m, layer.tensors, "convolution", {conv.kernel, e}, layer.index);
      l.Bind(m, layer.tensors, "convolution_input", {e, Product(e, 3)},
             layer.index);
      l.Bind(m, layer.tensors, "convolution_output", {e, e}, layer.index);
    } else {
      layer.attention_ordinal = attention++;
      BindAttention(l, m, layer, Product(m.key_head_width, layer.query_heads));
      BindHeadNorms(l, m, layer);
    }
  }
  if (!attention || !convolution)
    l.Fail("LFM2 requires attention and short-convolution layers");
  // The dense LFM2 sequence has no additional gamma, mixer gates or factors.
  // Reject unbound tensors rather than silently admitting another graph.
  for (const auto &tensor : l.catalog.tensors) {
    bool bound = false;
    for (const auto &binding : m.globals)
      bound |= binding.tensor.name == tensor.name;
    for (const auto &layer : m.layers)
      for (const auto &binding : layer.tensors)
        bound |= binding.tensor.name == tensor.name;
    if (!bound)
      l.Fail("unsupported LFM2 tensor: " + tensor.name);
  }
}
} // namespace
const ModelArchitecture kLfm2Architecture{"lfm2", LoadLfm2, Lfm2TensorSuffix};
} // namespace hrx::model::internal
