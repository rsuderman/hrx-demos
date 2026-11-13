// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
#include "tools/model/architectures.h"
#include <algorithm>
namespace hrx::model::internal {
uint64_t Product(uint64_t a, uint64_t b) {
  if (b && a > uint64_t(INT64_MAX) / b)
    throw std::runtime_error("architecture dimension product overflow");
  return a * b;
}
void RejectHeadNorms(const Loader &l, std::string_view architecture) {
  for (const auto &tensor : l.catalog.tensors) {
    const auto &name = tensor.name;
    if (name.rfind("blk.", 0) != 0)
      continue;
    size_t end = 4;
    while (end < name.size() && name[end] >= '0' && name[end] <= '9')
      ++end;
    if (end > 4 && (name.substr(end) == ".attn_q_norm.weight" ||
                    name.substr(end) == ".attn_k_norm.weight"))
      l.Fail("unsupported " + std::string(architecture) +
             " Q/K normalization tensor: " + name);
  }
}
void ReadDecoderDimensions(const Loader &l, ModelDescriptor &m,
                           bool allow_zero_kv) {
  for (const char *key :
       {"expert_count", "expert_used_count", "nextn_predict_layers"})
    if (l.Int(key, 0, false) != 0)
      l.Fail(std::string("unsupported MoE/MTP feature: ") + l.prefix + key);
  for (const char *key : {"attention.logit_softcap", "attention.scale",
                          "embedding_length_out", "pooling_type"})
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
  auto kv =
      l.Dimensions("attention.head_count_kv", layers, heads, allow_zero_kv);
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
  if (m.rope.width > m.key_head_width || m.rope.width % 2)
    l.Fail("invalid RoPE width");
  if (l.Field("vocab_size"))
    m.vocabulary_size = l.Int("vocab_size");
  else {
    const auto *tokens = FindMetadata(l.catalog, "tokenizer.ggml.tokens");
    if (!tokens || tokens->source_type != 9 || tokens->element_type != 8 ||
        !tokens->element_count || tokens->element_count > kDimensionLimit)
      l.Fail("missing/invalid tokenizer vocabulary array");
    m.vocabulary_size = tokens->element_count;
  }
  for (uint32_t i = 0; i < layers; ++i) {
    if (kv[i] && heads[i] % kv[i])
      l.Fail("query heads must be divisible by KV heads at layer" +
             std::to_string(i));
    LayerDescriptor d;
    d.index = i;
    d.feed_forward_width = ff[i];
    d.query_heads = heads[i];
    d.kv_heads = kv[i];
    m.layers.push_back(std::move(d));
  }
}
void ReadRope(const Loader &l, ModelDescriptor &m, double base) {
  m.rope.base = base;
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
  for (auto &d : m.layers)
    d.rope_base = base;
}
void BindGlobals(const Loader &l, ModelDescriptor &m) {
  m.output_tied_to_embedding = !FindTensor(l.catalog, "output.weight");
  l.Bind(m, m.globals, "token_embedding",
         {m.embedding_width, m.vocabulary_size});
  l.Bind(m, m.globals, "output_norm", {m.embedding_width});
  l.Bind(m, m.globals, "output", {m.embedding_width, m.vocabulary_size});
}
void BindFeedForward(const Loader &l, const ModelDescriptor &m,
                     LayerDescriptor &d) {
  const auto e = m.embedding_width, f = d.feed_forward_width;
  l.Bind(m, d.tensors, "ffn_gate", {e, f}, d.index);
  l.Bind(m, d.tensors, "ffn_up", {e, f}, d.index);
  l.Bind(m, d.tensors, "ffn_down", {f, e}, d.index);
}
void BindAttention(const Loader &l, const ModelDescriptor &m,
                   LayerDescriptor &d, uint64_t query_width) {
  const auto prefix = "blk." + std::to_string(d.index) + ".";
  if (FindTensor(l.catalog, prefix + "attn_qkv.weight"))
    l.Fail("unsupported fused attention QKV layout: " + prefix);
  const auto e = m.embedding_width;
  l.Bind(m, d.tensors, "query", {e, query_width}, d.index);
  l.Bind(m, d.tensors, "key", {e, Product(m.key_head_width, d.kv_heads)},
         d.index);
  l.Bind(m, d.tensors, "value", {e, Product(m.value_head_width, d.kv_heads)},
         d.index);
  l.Bind(m, d.tensors, "attention_output",
         {Product(m.key_head_width, d.query_heads), e}, d.index);
}
void BindHeadNorms(const Loader &l, const ModelDescriptor &m,
                   LayerDescriptor &d) {
  l.Bind(m, d.tensors, "query_norm", {m.key_head_width}, d.index);
  l.Bind(m, d.tensors, "key_norm", {m.key_head_width}, d.index);
}
const char *HeadNormTensorSuffix(std::string_view role) {
  if (role == "query_norm")
    return "attn_q_norm.weight";
  if (role == "key_norm")
    return "attn_k_norm.weight";
  return nullptr;
}
} // namespace hrx::model::internal
namespace hrx::model {
bool ExpectedTensorName(const ModelDescriptor &m, std::string_view role,
                        std::optional<uint32_t> layer, std::string *name) {
  const auto *architecture = internal::SelectModelArchitecture(m.family);
  if (!name || !architecture)
    return false;
  const char *suffix = nullptr;
  if (!layer) {
    if (role == "token_embedding")
      suffix = "token_embd.weight";
    else if (role == "output_norm")
      suffix = "output_norm.weight";
    else if (role == "output")
      suffix =
          m.output_tied_to_embedding ? "token_embd.weight" : "output.weight";
  } else {
    if (role == "input_norm")
      suffix = "attn_norm.weight";
    else if (role == "ffn_gate")
      suffix = "ffn_gate.weight";
    else if (role == "ffn_up")
      suffix = "ffn_up.weight";
    else if (role == "ffn_down")
      suffix = "ffn_down.weight";
    else if (role == "query")
      suffix = "attn_q.weight";
    else if (role == "key")
      suffix = "attn_k.weight";
    else if (role == "value")
      suffix = "attn_v.weight";
    else if (role == "attention_output")
      suffix = "attn_output.weight";
  }
  if (!suffix)
    suffix = architecture->extra_tensor_name(role, layer.has_value());
  if (!suffix)
    return false;
  *name = layer ? "blk." + std::to_string(*layer) + "." + suffix : suffix;
  return true;
}
bool LoadModelDescriptor(const GgufCatalog &catalog, ModelDescriptor *output,
                         std::string *error) {
  if (!output || !error)
    return false;
  try {
    internal::Loader l{catalog, catalog.architecture + "."};
    ModelDescriptor m;
    m.family = catalog.architecture;
    m.name = catalog.name;
    const auto *architecture = internal::SelectModelArchitecture(m.family);
    if (!architecture)
      l.Fail("unsupported architecture: " + m.family);
    architecture->load(l, m);
    // Optional tensors with graph effects must not silently become unbound
    // extras.
    for (const auto &t : catalog.tensors) {
      if (t.name.find(".scale") != std::string::npos ||
          t.name.find(".input_scale") != std::string::npos)
        l.Fail("unsupported optional scale tensor: " + t.name);
      if (t.name.find(".bias") != std::string::npos &&
          t.name.find(".ssm_dt.bias") == std::string::npos) {
        // Biases are admitted only when the selected architecture explicitly
        // binds them. Unknown extras must not silently change the model graph.
        const auto bound = [&](const auto &bindings) {
          return std::any_of(
              bindings.begin(), bindings.end(),
              [&](const auto &b) { return b.tensor.name == t.name; });
        };
        if (!bound(m.globals) && !std::any_of(m.layers.begin(), m.layers.end(),
                                              [&](const auto &layer) {
                                                return bound(layer.tensors);
                                              }))
          l.Fail("unsupported optional bias tensor: " + t.name);
      }
      if (t.name.find("rope_") != std::string::npos) {
        const auto bound = std::find_if(
            m.globals.begin(), m.globals.end(), [&](const auto &b) {
              return b.role == "rope_frequencies" && b.tensor.name == t.name;
            });
        if (bound == m.globals.end())
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
