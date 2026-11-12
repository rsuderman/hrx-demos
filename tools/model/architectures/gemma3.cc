// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
#include "tools/model/architectures/gemma3.h"
namespace hrx::model::internal {
namespace {
const char *Gemma3TensorSuffix(std::string_view role, bool is_layer) {
  if (!is_layer)
    return nullptr;

  if (role == "attention_post_norm")
    return "post_attention_norm.weight";
  if (role == "ffn_norm")
    return "ffn_norm.weight";
  if (role == "ffn_post_norm")
    return "post_ffw_norm.weight";
  return HeadNormTensorSuffix(role);
}
void LoadGemma3(const Loader &l, ModelDescriptor &m) {
  const auto window = l.Int("attention.sliding_window", 0, false);
  const auto period = l.Int("attention.sliding_window_pattern", 6, false);
  const auto local_rope_base = l.Real("rope.freq_base_swa", 10000);
  if (l.Real("final_logit_softcapping", 0, false) != 0)
    l.Fail("unsupported Gemma3 final logit softcap");
  ReadDecoderDimensions(l, m);
  ReadRope(l, m, l.Real("rope.freq_base"));
  BindGlobals(l, m);
  for (auto &d : m.layers) {
    d.attention_ordinal = d.index;
    const bool local = window && (period == 0 || d.index % period < period - 1);
    d.attention_window = local ? window : 0;
    d.rope_base = local ? local_rope_base : m.rope.base;
    l.Bind(m, d.tensors, "input_norm", {m.embedding_width}, d.index);
    l.Bind(m, d.tensors, "attention_post_norm", {m.embedding_width}, d.index);
    l.Bind(m, d.tensors, "ffn_norm", {m.embedding_width}, d.index);
    l.Bind(m, d.tensors, "ffn_post_norm", {m.embedding_width}, d.index);
    BindFeedForward(l, m, d);
    BindAttention(l, m, d, Product(m.key_head_width, d.query_heads));
    BindHeadNorms(l, m, d);
  }
}
} // namespace
const ModelArchitecture kGemma3Architecture{"gemma3", LoadGemma3,
                                            Gemma3TensorSuffix};
} // namespace hrx::model::internal
