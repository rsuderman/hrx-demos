// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration_internal.h"
#include "lib/loomify/architectures/gemma3.h"
namespace hrx::loomify_gguf::internal {
bool EmitGemma(const model::ModelDescriptor &m, const std::string &asset,
               const ConfigurationSettings &s, ConfigurationFacts *out,
               std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  if (m.family != "gemma3" || m.layers.empty() || m.layers.size() > 64 ||
      E != 1152 || H != 256 || m.value_head_width != H || !V || V > 262144 ||
      m.ssm || !m.output_tied_to_embedding || m.rope.width != H ||
      m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !PositiveF32(m.rms_epsilon))
    return Fail(error, "unsupported Gemma3 geometry, rotary or output binding");
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  if (!p.Tensor(m.globals, "token_embedding", E, V, {8}) ||
      !p.Tensor(m.globals, "output", E, V, {8}) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  p.Projection(*Role(m.globals, "output"), E, V);
  std::vector<std::vector<uint64_t>> ffn, attention, window;
  double localbase = 0, globalbase = m.rope.base;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    const auto F = l.feed_forward_width;
    if (l.index != i || l.kind != model::LayerKind::kAttention ||
        l.query_heads != 4 || l.kv_heads != 1 || !F || F > 32768 || F % 256 ||
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
        !p.Tensor(l.tensors, "key_norm", H, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "query", E, 4 * H, {6}, l.index) ||
        !p.Tensor(l.tensors, "key", E, H, {6}, l.index) ||
        !p.Tensor(l.tensors, "value", E, H, {6, 8}, l.index) ||
        !p.Tensor(l.tensors, "attention_output", 4 * H, E, {12}, l.index) ||
        !p.Tensor(l.tensors, "ffn_gate", E, F, {6}, l.index) ||
        !p.Tensor(l.tensors, "ffn_up", E, F, {6}, l.index) ||
        !p.Tensor(l.tensors, "ffn_down", F, E, {12, 14}, l.index))
      return false;
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
    p.Projection(q, E, 4 * H);
    p.Projection(k, E, H);
    p.Projection(v, E, H);
    p.Projection(o, 4 * H, E);
    p.Projection(g, E, F);
    p.Projection(u, E, F);
    p.Projection(d, F, E);
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
  p.Index("gemma3.model.query_heads", 4);
  p.Index("gemma3.model.kv_heads", 1);
  p.Index("gemma3.model.head_size", H);
  p.Index("gemma3.model.kv_size", H);
  p.Index("gemma3.model.query_size", 4 * H);
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
  p.OrdinaryPreparation(E, m.layers.size(), H, history, s.token_capacity,
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
