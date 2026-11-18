// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration_internal.h"
#include "lib/loomify/assembly_registry.h"
#include <tuple>
namespace hrx::loomify_gguf::internal {
bool EmitGemma(const model::ModelDescriptor &m, const std::string &asset,
               const ConfigurationSettings &s, ConfigurationFacts *out,
               std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  if (m.family != "gemma3" || m.layers.empty() || m.layers.size() > 64 || !E ||
      E > 32768 || H != 256 || m.value_head_width != H || !V || V > 262208 ||
      m.ssm || !m.output_tied_to_embedding || m.rope.width != H ||
      !PositiveF32(m.rope.scaling_factor) ||
      (m.rope.scaling_kind == "none" && m.rope.scaling_factor != 1) ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !PositiveF32(m.rms_epsilon))
    return Fail(error, "unsupported Gemma3 geometry, rotary or output binding");
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  OrdinaryEmbeddingFacts embedding;
  CanonicalProjectionFacts output;
  if (!p.OrdinaryEmbedding(m.globals, "token_embedding", E, V, &embedding) ||
      (embedding.storage_tag != 8 && embedding.storage_tag != 14) ||
      !p.CanonicalProjection(m.globals, "output", E, V, std::nullopt,
                             &output) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  const auto QH = m.layers.front().query_heads, KH = m.layers.front().kv_heads;
  if (!QH || QH > 32 || !KH || KH > QH || QH % KH)
    return Fail(error, "unsupported Gemma3 head grouping");
  const auto Q = QH * H, KV = KH * H;
  std::vector<std::vector<uint64_t>> ffn, attention, window;
  double localbase = 0, globalbase = m.rope.base;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    const auto F = l.feed_forward_width;
    if (l.index != i || l.kind != model::LayerKind::kAttention ||
        l.query_heads != QH || l.kv_heads != KH || !F || F > 32768 || F % 256 ||
        !PositiveF32(l.rope_base) || l.rope_base < 1)
      return Fail(error, "unsupported Gemma3 layer geometry or rotary base");
    if (l.attention_window) {
      if (localbase && localbase != l.rope_base)
        return Fail(error, "inconsistent Gemma local rotary bases");
      localbase = l.rope_base;
    } else if (l.rope_base != globalbase)
      return Fail(error, "inconsistent Gemma global rotary base");
    for (const char *role :
         {"input_norm", "attention_post_norm", "ffn_norm", "ffn_post_norm"})
      if (!p.Tensor(l.tensors, role, E, 1, {0}, l.index))
        return false;
    if (!p.Tensor(l.tensors, "query_norm", H, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "key_norm", H, 1, {0}, l.index))
      return false;
    for (const auto &shape :
         std::vector<std::tuple<const char *, uint64_t, uint64_t>>{
             {"query", E, Q},
             {"key", E, KV},
             {"value", E, KV},
             {"attention_output", Q, E},
             {"ffn_gate", E, F},
             {"ffn_up", E, F},
             {"ffn_down", F, E}}) {
      CanonicalProjectionFacts projection;
      const auto *tensor = Role(l.tensors, std::get<0>(shape));
      if (!tensor || (tensor->storage_type != 6 && tensor->storage_type != 8 &&
                      tensor->storage_type != 12 && tensor->storage_type != 14))
        return Fail(error, "unsupported Gemma3 projection encoding");
      if (!p.CanonicalProjection(l.tensors, std::get<0>(shape),
                                 std::get<1>(shape), std::get<2>(shape),
                                 l.index, &projection))
        return false;
    }
    const auto &q = *Role(l.tensors, "query"), &k = *Role(l.tensors, "key"),
               &v = *Role(l.tensors, "value"),
               &o = *Role(l.tensors, "attention_output");
    const auto &g = *Role(l.tensors, "ffn_gate"),
               &u = *Role(l.tensors, "ffn_up"),
               &d = *Role(l.tensors, "ffn_down");
    attention.push_back({q.storage_type, k.storage_type, v.storage_type,
                         o.storage_type, q.byte_size, k.byte_size, v.byte_size,
                         o.byte_size});
    ffn.push_back({F, g.storage_type, u.storage_type, d.storage_type,
                   g.byte_size, u.byte_size, d.byte_size});
    window.push_back({l.attention_window, l.attention_window ? 1u : 0u});
  }
  if (!PositiveF32(globalbase) || globalbase < 1)
    return Fail(error, "invalid global rotary base");
  if (!localbase)
    localbase = globalbase;
  const auto history = std::max<uint64_t>(s.token_capacity, 129);
  p.Index("gemma3.model.hidden_size", E);
  p.Index("gemma3.model.vocabulary_size", V);
  p.Index("gemma3.model.layer_count", m.layers.size());
  p.Index("gemma3.model.history_capacity", history);
  p.Index("gemma3.model.query_heads", QH);
  p.Index("gemma3.model.kv_heads", KH);
  p.Index("gemma3.model.head_size", H);
  p.Index("gemma3.model.kv_size", KV);
  p.Index("gemma3.model.query_size", Q);
  p.Index("gemma3.model.embedding_weight_tag", embedding.storage_tag);
  p.Index("gemma3.model.output_weight_tag", output.storage_tag);
  p.Real("gemma3.model.rope_factor_global", float(m.rope.scaling_factor));
  p.Real("gemma3.model.rms_epsilon", float(m.rms_epsilon));
  p.Real("gemma3.model.embedding_scale", float(std::sqrt(double(E))));
  p.Real("gemma3.model.attention_scale", float(1 / std::sqrt(double(H))));
  p.Real("gemma3.model.rope_log_step_local",
         float(-2 * std::log(localbase) / H));
  p.Real("gemma3.model.rope_log_step_global",
         float(-2 * std::log(globalbase) / H));
  p.Index("gemma3.model.embedding_weight_bytes",
          Role(m.globals, "token_embedding")->byte_size);
  p.Index("gemma3.model.output_weight_bytes",
          Role(m.globals, "output")->byte_size);
  p.Index("runner.gemma3.prefill_token_count", s.token_capacity);
  p.Index("gemma3.attention.cache_capacity", s.context_capacity);
  p.Index("gemma3.attention.pool_capacity",
          (s.context_capacity + 63) / 64 * 64);
  p.Index("gemma3.attention.cache_base", 768 + 4 * history, "offset");
  p.Index("gemma3.attention.map_origin", 2048, "offset");
  p.Index("qwen38.greedy_argmax.output_capacity", V);
  p.OrdinaryPreparation(E, m.layers.size(), KV, history, s.token_capacity,
                        s.context_capacity, (s.context_capacity + 63) / 64 * 64,
                        768 + 4 * history, 2048);
  p.ProjectionCapacities();
  p.Lookup("loomify.config.layer_attention", attention);
  p.Lookup("loomify.config.layer_ffn", ffn);
  p.Lookup("loomify.config.layer_window", window);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
