// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace hrx::model {
namespace {
// Rules mirror llama.cpp dace2471 src/llama-model.cpp1086-1209 and
// src/models/{llama,qwen35}.cpp. These limits bound structural arithmetic;
// they do not restrict loading to an existing execution profile.
constexpr uint64_t kDimensionLimit = 1ULL << 30;
struct Loader {
  const GgufCatalog &catalog;
  std::string prefix;
  [[noreturn]] void Fail(const std::string &message) const {
    throw std::runtime_error(message);
  }
  const MetadataRecord *Field(const std::string &key) const {
    return FindMetadata(catalog, prefix + key);
  }
  uint64_t Integer(const MetadataScalar &value, const std::string &key,
                   bool positive = true) const {
    uint64_t n;
    if (auto v = std::get_if<uint64_t>(&value))
      n = *v;
    else if (auto v = std::get_if<int64_t>(&value); v && *v >= 0)
      n = static_cast<uint64_t>(*v);
    else
      Fail("expected integer metadata: " + prefix + key);
    if ((positive && n == 0) || n > kDimensionLimit)
      Fail("invalid/bounded geometry: " + prefix + key);
    return n;
  }
  uint64_t Int(const std::string &key, std::optional<uint64_t> fallback = {},
               bool positive = true) const {
    const auto *f = Field(key);
    if (!f) {
      if (fallback)
        return *fallback;
      Fail("missing required metadata: " + prefix + key);
    }
    if (f->source_type == 9)
      Fail("expected scalar integer metadata: " + prefix + key);
    return Integer(f->scalar, key, positive);
  }
  double Real(const std::string &key, std::optional<double> fallback = {},
              bool positive = true) const {
    const auto *f = Field(key);
    if (!f) {
      if (fallback)
        return *fallback;
      Fail("missing required metadata: " + prefix + key);
    }
    const auto *v = std::get_if<double>(&f->scalar);
    if (f->source_type == 9 || !v || !std::isfinite(*v) ||
        (positive && *v <= 0))
      Fail("expected finite floating metadata: " + prefix + key);
    return *v;
  }
  std::string String(const std::string &key,
                     const std::string &fallback) const {
    const auto *f = Field(key);
    if (!f)
      return fallback;
    const auto *s = std::get_if<std::string>(&f->scalar);
    if (f->source_type != 8 || !s)
      Fail("expected string metadata: " + prefix + key);
    return *s;
  }
  std::vector<uint64_t>
  Dimensions(const std::string &key, size_t count,
             const std::vector<uint64_t> &fallback = {}) const {
    const auto *f = Field(key);
    if (!f) {
      if (!fallback.empty())
        return fallback;
      Fail("missing required metadata: " + prefix + key);
    }
    if (f->source_type != 9)
      return std::vector<uint64_t>(count, Integer(f->scalar, key));
    if (f->elements_omitted || f->element_count != count ||
        f->elements.size() != count)
      Fail("wrong array length: " + prefix + key);
    std::vector<uint64_t> out;
    for (const auto &v : f->elements)
      out.push_back(Integer(v, key));
    return out;
  }
  std::vector<bool> Recurrent(size_t count) const {
    const std::string key = "attention.recurrent_layers";
    const auto *f = Field(key);
    if (!f) {
      const auto interval = Int("full_attention_interval", 4);
      std::vector<bool> out;
      for (size_t i = 0; i < count; ++i)
        out.push_back((i + 1) % interval != 0);
      return out;
    }
    auto value = [&](const MetadataScalar &v) {
      if (auto b = std::get_if<bool>(&v))
        return *b;
      auto n = Integer(v, key, false);
      if (n > 1)
        Fail("expected boolean recurrent map: " + prefix + key);
      return n != 0;
    };
    if (f->source_type != 9)
      return std::vector<bool>(count, value(f->scalar));
    if (f->elements_omitted || f->elements.size() != count ||
        f->element_count != count)
      Fail("wrong recurrent array length: " + prefix + key);
    std::vector<bool> out;
    for (const auto &v : f->elements)
      out.push_back(value(v));
    return out;
  }
  void Bind(std::vector<TensorBinding> &out, const std::string &role,
            const std::string &name, std::initializer_list<uint64_t> shape,
            bool optional = false) const {
    const auto *t = FindTensor(catalog, name);
    if (!t) {
      if (optional)
        return;
      Fail("missing required tensor: " + name);
    }
    std::array<int64_t, 4> expected{1, 1, 1, 1};
    size_t i = 0;
    for (auto extent : shape) {
      if (extent > uint64_t(INT64_MAX))
        Fail("tensor geometry overflow: " + name);
      expected[i++] = extent;
    }
    if (t->normalized_extents != expected)
      Fail("tensor logical shape mismatch: " + name);
    out.push_back({role, *t});
  }
  void Unsupported(const std::string &key) const {
    if (Field(key))
      Fail("unsupported architecture feature: " + prefix + key);
  }
};
uint64_t Product(uint64_t a, uint64_t b) {
  if (b && a > uint64_t(INT64_MAX) / b)
    throw std::runtime_error("architecture dimension product overflow");
  return a * b;
}
} // namespace
bool LoadModelDescriptor(const GgufCatalog &catalog, ModelDescriptor *output,
                         std::string *error) {
  if (!output || !error)
    return false;
  try {
    Loader l{catalog, catalog.architecture + "."};
    if (catalog.architecture == "llama") {
      for (const auto &tensor : catalog.tensors) {
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
    }

    ModelDescriptor m;
    m.family = catalog.architecture;
    m.name = catalog.name;
    if (m.family != "llama" && m.family != "qwen35")
      l.Fail("unsupported architecture: " + m.family);
    for (const char *key :
         {"expert_count", "expert_used_count", "nextn_predict_layers"})
      if (l.Int(key, 0, false) != 0)
        l.Fail(std::string("unsupported MoE/MTP feature: ") + l.prefix + key);
    for (const char *key :
         {"attention.sliding_window", "attention.logit_softcap",
          "attention.scale", "embedding_length_out", "pooling_type"})
      l.Unsupported(key);
    if (const auto *f = l.Field("attention.causal")) {
      const auto *v = std::get_if<bool>(&f->scalar);
      if (f->source_type != 7 || !v || !*v)
        l.Fail("unsupported noncausal attention");
    }
    auto layers = l.Int("block_count");
    if (layers > 512)
      l.Fail("block_count exceeds structural layer limit512");
    m.embedding_width = l.Int("embedding_length");
    m.training_context = l.Int("context_length");
    auto ff = l.Dimensions("feed_forward_length", layers);
    auto heads = l.Dimensions("attention.head_count", layers);
    auto kv = l.Dimensions("attention.head_count_kv", layers, heads);
    if ((!l.Field("attention.key_length") ||
         !l.Field("attention.value_length")) &&
        m.embedding_width % heads.front())
      l.Fail("embedding width not divisible by default head count");
    m.key_head_width =
        l.Int("attention.key_length", m.embedding_width / heads.front());
    m.value_head_width =
        l.Int("attention.value_length", m.embedding_width / heads.front());
    if (m.key_head_width != m.value_head_width)
      l.Fail("unsupported unequal key/value head widths");
    m.rms_epsilon = l.Real("attention.layer_norm_rms_epsilon");
    m.rope.width = l.Int("rope.dimension_count", m.key_head_width);
    if (m.rope.width > m.key_head_width || m.rope.width % 2 ||
        (m.family == "llama" && m.rope.width != m.key_head_width))
      l.Fail("invalid RoPE width");
    m.rope.base = l.Real("rope.freq_base", 10000);
    m.rope.scaling_kind = l.String("rope.scaling.type", "linear");
    if (m.rope.scaling_kind != "linear" && m.rope.scaling_kind != "none")
      l.Fail("unsupported RoPE scaling: " + m.rope.scaling_kind);
    double factor = l.Field("rope.scaling.factor")
                        ? l.Real("rope.scaling.factor", {}, false)
                        : l.Real("rope.scale_linear", 0, false);
    if (factor < 0)
      l.Fail("negative RoPE scaling factor");
    m.rope.scaling_factor = factor == 0 ? 1 : factor;
    m.rope.original_context =
        l.Int("rope.scaling.original_context_length", m.training_context);
    for (const char *key : {"rope.scaling.attn_factor", "rope.scaling.alpha",
                            "rope.scaling.finetuned"})
      l.Unsupported(key);
    if (l.Field("vocab_size"))
      m.vocabulary_size = l.Int("vocab_size");
    else {
      auto *tokens = FindMetadata(catalog, "tokenizer.ggml.tokens");
      if (!tokens || tokens->source_type != 9 || tokens->element_type != 8 ||
          !tokens->element_count || tokens->element_count > kDimensionLimit)
        l.Fail("missing/invalid tokenizer vocabulary array");
      m.vocabulary_size = tokens->element_count;
    }
    l.Bind(m.globals, "token_embedding", "token_embd.weight",
           {m.embedding_width, m.vocabulary_size});
    l.Bind(m.globals, "output_norm", "output_norm.weight", {m.embedding_width});
    m.output_tied_to_embedding = !FindTensor(catalog, "output.weight");
    l.Bind(m.globals, "output",
           m.output_tied_to_embedding ? "token_embd.weight" : "output.weight",
           {m.embedding_width, m.vocabulary_size});
    std::vector<bool> recurrent(layers, false);
    if (m.family == "qwen35") {
      auto sections = l.Field("rope.dimension_sections");
      if (!sections)
        l.Fail("missing required metadata: qwen35.rope.dimension_sections");
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
        m.rope.sections[i] =
            l.Integer(values[i], "rope.dimension_sections", false);
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
      recurrent = l.Recurrent(layers);
    }
    uint32_t attention_ordinal = 0, recurrent_ordinal = 0;
    for (uint32_t i = 0; i < layers; ++i) {
      LayerDescriptor d;
      d.index = i;
      d.feed_forward_width = ff[i];
      d.query_heads = heads[i];
      d.kv_heads = kv[i];
      if (heads[i] % kv[i])
        l.Fail("query heads must be divisible by KV heads at layer" +
               std::to_string(i));
      const auto p = "blk." + std::to_string(i) + ".";
      auto bind = [&](const std::string &role, const std::string &suffix,
                      std::initializer_list<uint64_t> shape) {
        l.Bind(d.tensors, role, p + suffix, shape);
      };
      auto e = m.embedding_width;
      auto f = ff[i];
      bind("input_norm", "attn_norm.weight", {e});
      bind("post_norm",
           m.family == "llama" ? "ffn_norm.weight"
                               : "post_attention_norm.weight",
           {e});
      bind("ffn_gate", "ffn_gate.weight", {e, f});
      bind("ffn_up", "ffn_up.weight", {e, f});
      bind("ffn_down", "ffn_down.weight", {f, e});
      if (recurrent[i]) {
        d.kind = LayerKind::kGatedDeltaNet;
        d.recurrent_ordinal = recurrent_ordinal++;
        const auto &s = *m.ssm;
        auto v = Product(s.state_width, s.value_heads),
             k = Product(s.state_width, s.key_heads);
        bind("recurrent_qkv", "attn_qkv.weight", {e, 2 * k + v});
        bind("recurrent_gate", "attn_gate.weight", {e, v});
        bind("convolution", "ssm_conv1d.weight",
             {s.convolution_kernel, 2 * k + v});
        bind("dt_bias", "ssm_dt.bias", {s.value_heads});
        bind("decay", "ssm_a", {s.value_heads});
        bind("beta", "ssm_beta.weight", {e, s.value_heads});
        bind("alpha", "ssm_alpha.weight", {e, s.value_heads});
        bind("recurrent_norm", "ssm_norm.weight", {s.state_width});
        bind("recurrent_output", "ssm_out.weight", {v, e});
      } else {
        d.attention_ordinal = attention_ordinal++;
        if (FindTensor(catalog, p + "attn_qkv.weight"))
          l.Fail("unsupported fused attention QKV layout: " + p);
        auto q = Product(m.key_head_width, heads[i]),
             k = Product(m.key_head_width, kv[i]),
             v = Product(m.value_head_width, kv[i]);
        bind("query", "attn_q.weight",
             {e, Product(q, m.family == "qwen35" ? 2 : 1)});
        bind("key", "attn_k.weight", {e, k});
        bind("value", "attn_v.weight", {e, v});
        bind("attention_output", "attn_output.weight", {q, e});
        if (m.family == "qwen35") {
          bind("query_norm", "attn_q_norm.weight", {m.key_head_width});
          bind("key_norm", "attn_k_norm.weight", {m.key_head_width});
        }
      }
      m.layers.push_back(std::move(d));
    }
    // Known optional weights must not silently become unbound extras whose
    // graph effects would be lost. Non-model extras remain in the catalog.
    for (const auto &t : catalog.tensors) {
      if (t.name.find(".scale") != std::string::npos ||
          t.name.find(".input_scale") != std::string::npos)
        l.Fail("unsupported optional scale tensor: " + t.name);
      if (t.name.find(".bias") != std::string::npos &&
          t.name.find(".ssm_dt.bias") == std::string::npos)
        l.Fail("unsupported optional bias tensor: " + t.name);
      if (t.name.find("rope_") != std::string::npos) {
        if (t.name == "rope_freqs.weight" && m.family == "llama")
          l.Bind(m.globals, "rope_frequencies", t.name, {m.rope.width / 2});
        else
          l.Fail("unsupported optional RoPE tensor: " + t.name);
      }
      if (t.name.find("_exps.") != std::string::npos ||
          t.name.rfind("mtp.", 0) == 0)
        l.Fail("unsupported MoE/MTP tensor: " + t.name);
    }
    *output = std::move(m);
    return true;
  } catch (const std::exception &ex) {
    *error = ex.what();
    return false;
  }
}
} // namespace hrx::model
