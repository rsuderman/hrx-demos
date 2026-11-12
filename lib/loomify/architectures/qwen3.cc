// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration_internal.h"

namespace hrx::loomify_gguf::internal {
bool EmitQwen3(const model::ModelDescriptor &m, const std::string &asset,
               const ConfigurationSettings &s, ConfigurationFacts *out,
               std::string *error) {
  if (!out || asset.empty())
    return Fail(error, "configuration requires output and source asset identity");
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  if (m.family != "qwen3" || m.layers.empty() || m.layers.size() > 64 ||
      !E || E > 32768 || E % 1024 || !V || V > 262144 || H != 128 ||
      m.value_head_width != H || m.ssm || !m.output_tied_to_embedding ||
      m.rope.width != H || m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !PositiveF32(m.rms_epsilon) || !PositiveF32(m.rope.base) || m.rope.base < 1)
    return Fail(error, "unsupported Qwen3 geometry, rotary or output binding");
  const auto QH = m.layers.front().query_heads, KH = m.layers.front().kv_heads;
  uint64_t Q, KV;
  if (!QH || QH > 128 || !KH || KH > QH || QH % KH ||
      !Product(QH, H, &Q) || Q > 32768 || !Product(KH, H, &KV))
    return Fail(error, "unsupported Qwen3 attention head geometry");
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  if (!p.Tensor(m.globals, "token_embedding", E, V, {14}) ||
      !p.Tensor(m.globals, "output", E, V, {14}) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  p.Projection(*Role(m.globals, "output"), E, V);
  std::vector<std::vector<uint64_t>> ffn, attention;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    const auto F = l.feed_forward_width;
    if (l.index != i || l.kind != model::LayerKind::kAttention ||
        l.query_heads != QH || l.kv_heads != KH || !F || F > 32768 || F % 256 ||
        l.attention_window || l.rope_base != m.rope.base)
      return Fail(error, "unsupported Qwen3 layer identity or geometry");
    if (!p.Tensor(l.tensors, "input_norm", E, 1,
                  {0}, l.index) ||
        !p.Tensor(l.tensors, "post_norm", E, 1,
                  {0}, l.index) ||
        !p.Tensor(l.tensors, "query_norm", H, 1,
                  {0}, l.index) ||
        !p.Tensor(l.tensors, "key_norm", H, 1,
                  {0}, l.index) ||
        !p.Tensor(l.tensors, "query", E, Q,
                  {12, 14}, l.index) ||
        !p.Tensor(l.tensors, "key", E, KV,
                  {12, 14}, l.index) ||
        !p.Tensor(l.tensors, "value", E, KV,
                  {12, 14}, l.index) ||
        !p.Tensor(l.tensors, "attention_output",
                  Q, E, {12, 14}, l.index) ||
        !p.Tensor(l.tensors, "ffn_gate", E, F,
                  {12}, l.index) ||
        !p.Tensor(l.tensors, "ffn_up", E, F,
                  {12}, l.index) ||
        !p.Tensor(l.tensors, "ffn_down", F, E,
                  {12, 14}, l.index))
      return false;
    const auto &q = *Role(l.tensors, "query"), &k = *Role(l.tensors, "key"),
               &v = *Role(l.tensors, "value"),
               &o = *Role(l.tensors, "attention_output"),
               &g = *Role(l.tensors, "ffn_gate"),
               &u = *Role(l.tensors, "ffn_up"),
               &d = *Role(l.tensors, "ffn_down");
    if (g.storage_type != u.storage_type || g.byte_size != u.byte_size)
      return Fail(error, "FFN gate/up must have matching encoded extents");
    attention.push_back({q.storage_type, k.storage_type, v.storage_type,
                         o.storage_type, q.byte_size, k.byte_size, v.byte_size,
                         o.byte_size});
    ffn.push_back({F, g.storage_type, u.storage_type, d.storage_type,
                   g.byte_size, u.byte_size, d.byte_size});
    p.Projection(q, E, Q);
    p.Projection(k, E, KV);
    p.Projection(v, E, KV);
    p.Projection(o, Q, E);
    p.Projection(g, E, F);
    p.Projection(u, E, F);
    p.Projection(d, F, E);
  }
  const auto history = std::max<uint64_t>(s.token_capacity, 129);
  p.Index("qwen3.model.hidden_size", E);
  p.Index("qwen3.model.query_size", Q);
  p.Index("qwen3.model.kv_size", KV);
  p.Index("qwen3.model.vocabulary_size", V);
  p.Index("qwen3.model.layer_count", m.layers.size());
  p.Index("qwen3.model.query_heads", QH);
  p.Index("qwen3.model.kv_heads", KH);
  p.Index("qwen3.model.head_size", H);
  p.Index("qwen3.model.history_capacity", history);
  p.Real("qwen3.model.rms_epsilon", float(m.rms_epsilon));
  p.Real("qwen3.model.rope_log_step", float(-2 * std::log(m.rope.base) / H));
  p.Real("qwen3.model.attention_scale", float(1 / std::sqrt(double(H))));
  p.Index("qwen3.model.embedding_weight_bytes",
          Role(m.globals, "token_embedding")->byte_size);
  p.Index("qwen3.model.output_weight_bytes", Role(m.globals, "output")->byte_size);
  p.Index("runner.qwen3.prefill_token_count", s.token_capacity);
  p.Index("qwen3.attention.cache_capacity", s.context_capacity);
  p.Index("qwen3.attention.pool_capacity", (s.context_capacity + 63) / 64 * 64);
  p.Index("qwen3.attention.cache_base", 768 + 4 * history, "offset");
  p.Index("qwen3.attention.map_origin", 2048, "offset");
  p.Index("qwen38.greedy_argmax.output_capacity", V);
  p.OrdinaryPreparation(E, m.layers.size(), KV, history, s.token_capacity,
                        s.context_capacity, (s.context_capacity + 63) / 64 * 64,
                        768 + 4 * history, 2048);
  p.ProjectionCapacities();
  p.Lookup("loomify.config.layer_attention", attention);
  p.Lookup("loomify.config.layer_ffn", ffn);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
