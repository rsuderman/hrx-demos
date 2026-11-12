// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
#include "tools/model/architectures/qwen35.h"
namespace hrx::model::internal {
namespace {
std::vector<bool> Recurrent(const Loader &l, size_t count) {
  const std::string key = "attention.recurrent_layers";
  const auto *f = l.Field(key);
  if (!f) {
    const auto interval = l.Int("full_attention_interval", 4);
    std::vector<bool> out;
    for (size_t i = 0; i < count; ++i)
      out.push_back((i + 1) % interval != 0);
    return out;
  }
  auto value = [&](const MetadataScalar &v) {
    if (auto b = std::get_if<bool>(&v))
      return *b;
    auto n = l.Integer(v, key, false);
    if (n > 1)
      l.Fail("expected boolean recurrent map: " + l.prefix + key);
    return n != 0;
  };
  if (f->source_type != 9)
    return std::vector<bool>(count, value(f->scalar));
  if (f->elements_omitted || f->elements.size() != count ||
      f->element_count != count)
    l.Fail("wrong recurrent array length: " + l.prefix + key);
  std::vector<bool> out;
  for (const auto &v : f->elements)
    out.push_back(value(v));
  return out;
}
const char *Qwen35TensorSuffix(std::string_view role, bool is_layer) {
  if (!is_layer) return nullptr;

  if (role == "post_norm")
    return "post_attention_norm.weight";
  if (role == "recurrent_qkv")
    return "attn_qkv.weight";
  if (role == "recurrent_gate")
    return "attn_gate.weight";
  if (role == "convolution")
    return "ssm_conv1d.weight";
  if (role == "dt_bias")
    return "ssm_dt.bias";
  if (role == "decay")
    return "ssm_a";
  if (role == "beta")
    return "ssm_beta.weight";
  if (role == "alpha")
    return "ssm_alpha.weight";
  if (role == "recurrent_norm")
    return "ssm_norm.weight";
  if (role == "recurrent_output")
    return "ssm_out.weight";
  return HeadNormTensorSuffix(role);
}
void LoadQwen35(const Loader &l, ModelDescriptor &m) {
  l.Unsupported("attention.sliding_window");
  ReadDecoderDimensions(l, m);
  ReadRope(l, m, l.Real("rope.freq_base", 10000));
  BindGlobals(l, m);
  auto sections = l.Field("rope.dimension_sections");
  if (!sections)
    l.Fail("missing required metadata: " + l.prefix +
           "rope.dimension_sections");
  std::vector<MetadataScalar> values;
  if (sections->source_type == 9) {
    if (sections->elements_omitted || sections->element_count != 4 ||
        sections->elements.size() != 4)
      l.Fail("RoPE sections must have length4");
    values = sections->elements;
  } else
    values.assign(4, sections->scalar);
  uint64_t sum = 0;
  for (size_t i = 0; i < 4; ++i) {
    m.rope.sections[i] = l.Integer(values[i], "rope.dimension_sections", false);
    sum += m.rope.sections[i];
  }
  if (!sum || Product(sum, 2) != m.rope.width)
    l.Fail("RoPE sections must sum to half RoPE width");
  SsmDescriptor s;
  s.convolution_kernel = l.Int("ssm.conv_kernel");
  s.inner_width = l.Int("ssm.inner_size");
  s.state_width = l.Int("ssm.state_size");
  s.key_heads = l.Int("ssm.group_count");
  s.value_heads = l.Int("ssm.time_step_rank");
  if (s.inner_width % s.value_heads ||
      s.inner_width / s.value_heads != s.state_width ||
      s.value_heads % s.key_heads)
    l.Fail("unsupported/inconsistent SSM head geometry");
  auto conv_width = Product(s.state_width, 2 * s.key_heads + s.value_heads);
  s.convolution_history_elements =
      Product(s.convolution_kernel - 1, conv_width);
  s.recurrent_state_elements = Product(s.state_width, s.inner_width);
  m.ssm = s;
  const auto recurrent = Recurrent(l, m.layers.size());
  uint32_t attention_ordinal = 0, recurrent_ordinal = 0;
  for (auto &d : m.layers) {
    const auto e = m.embedding_width;
    l.Bind(m, d.tensors, "input_norm", {e}, d.index);
    l.Bind(m, d.tensors, "post_norm", {e}, d.index);
    BindFeedForward(l, m, d);
    if (recurrent[d.index]) {
      d.kind = LayerKind::kGatedDeltaNet;
      d.recurrent_ordinal = recurrent_ordinal++;
      const auto v = Product(s.state_width, s.value_heads);
      const auto k = Product(s.state_width, s.key_heads);
      l.Bind(m, d.tensors, "recurrent_qkv", {e, 2 * k + v}, d.index);
      l.Bind(m, d.tensors, "recurrent_gate", {e, v}, d.index);
      l.Bind(m, d.tensors, "convolution", {s.convolution_kernel, 2 * k + v},
             d.index);
      l.Bind(m, d.tensors, "dt_bias", {s.value_heads}, d.index);
      l.Bind(m, d.tensors, "decay", {s.value_heads}, d.index);
      l.Bind(m, d.tensors, "beta", {e, s.value_heads}, d.index);
      l.Bind(m, d.tensors, "alpha", {e, s.value_heads}, d.index);
      l.Bind(m, d.tensors, "recurrent_norm", {s.state_width}, d.index);
      l.Bind(m, d.tensors, "recurrent_output", {v, e}, d.index);
    } else {
      d.attention_ordinal = attention_ordinal++;
      BindAttention(l, m, d,
                    Product(Product(m.key_head_width, d.query_heads), 2));
      BindHeadNorms(l, m, d);
    }
  }
}
} // namespace
const ModelArchitecture kQwen35Architecture{
    "qwen35", LoadQwen35, Qwen35TensorSuffix};
} // namespace hrx::model::internal
