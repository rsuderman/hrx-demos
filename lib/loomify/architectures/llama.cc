// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration_internal.h"
namespace hrx::loomify_gguf::internal {
bool EmitLlama(const model::ModelDescriptor &m, const std::string &asset,
               const ConfigurationSettings &s, ConfigurationFacts *out,
               std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  if (!s.token_capacity || s.token_capacity > 512)
    return Fail(error, "token capacity must be between 1 and 512");
  if (m.family != "llama" || m.layers.empty() || m.layers.size() > 64)
    return Fail(error, "Llama configuration requires supported layer identities");
  const uint64_t E = m.embedding_width, V = m.vocabulary_size;
  const uint64_t QH = m.layers.front().query_heads, KH = m.layers.front().kv_heads;
  const uint64_t H = m.key_head_width;
  uint64_t Q, KV;
  if (!E || !V || !QH || !KH || QH % KH || !H || H != m.value_head_width ||
      !Product(QH, H, &Q) || Q != E || !Product(KH, H, &KV) ||
      !PositiveF32(m.rms_epsilon) || !PositiveF32(m.rope.base) || m.rope.base < 1 ||
      m.ssm || m.rope.width != H || m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !m.output_tied_to_embedding)
    return Fail(error, "inconsistent Llama geometry or unsupported rotary/output binding");
  Producer p;
  p.asset = asset;
  p.error = error;
  if (!p.Tensor(m.globals, "token_embedding", "token_embd.weight", E, V,
                {14}) ||
      !p.Tensor(m.globals, "output", "token_embd.weight", E, V, {14}) ||
      !p.Tensor(m.globals, "output_norm", "output_norm.weight", E, 1, {0}) ||
      !p.Tensor(m.globals, "rope_frequencies", "rope_freqs.weight", m.rope.width / 2, 1, {0}))
    return false;

  p.Projection(*Role(m.globals, "output"), E, V);
  std::vector<std::vector<uint64_t>> ffn, attention;
  // Order is architecture order; require that it is also the native layer key.
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    const uint64_t F = l.feed_forward_width;
    if (l.index != i || !F || l.kind != model::LayerKind::kAttention ||
        l.query_heads != QH || l.kv_heads != KH)
      return Fail(error, "unsupported layer identity or geometry at layer " +
                             std::to_string(i));
    const auto prefix = "blk." + std::to_string(i) + ".";
    if (!p.Tensor(l.tensors, "post_norm", prefix + "ffn_norm.weight", E, 1, {0},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_gate", prefix + "ffn_gate.weight", E, F, {},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_up", prefix + "ffn_up.weight", E, F, {},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_down", prefix + "ffn_down.weight", F, E,
                  {}, l.index))
      return false;
    const auto &gate = *Role(l.tensors, "ffn_gate"),
               &up = *Role(l.tensors, "ffn_up"),
               &down = *Role(l.tensors, "ffn_down");
    if (gate.storage_type != up.storage_type || gate.byte_size != up.byte_size)
      return Fail(
          error,
          "FFN gate/up must have matching encoding and encoded byte extents");
    ffn.push_back({F, gate.storage_type, up.storage_type, down.storage_type,
                   gate.byte_size, up.byte_size, down.byte_size});
    p.Projection(gate, E, F);
    p.Projection(up, E, F);
    p.Projection(down, F, E);
    if (!p.Tensor(l.tensors, "input_norm", prefix + "attn_norm.weight", E, 1,
                  {0}, l.index) ||
        !p.Tensor(l.tensors, "query", prefix + "attn_q.weight", E, E, {},
                  l.index) ||
        !p.Tensor(l.tensors, "key", prefix + "attn_k.weight", E, KV, {},
                  l.index) ||
        !p.Tensor(l.tensors, "value", prefix + "attn_v.weight", E, KV,
                  {}, l.index) ||
        !p.Tensor(l.tensors, "attention_output", prefix + "attn_output.weight",
                  E, E, {}, l.index))
      return false;
    const auto &q = *Role(l.tensors, "query"), &k = *Role(l.tensors, "key"),
               &v = *Role(l.tensors, "value"),
               &o = *Role(l.tensors, "attention_output");
    attention.push_back({q.storage_type, k.storage_type, v.storage_type,
                         o.storage_type, q.byte_size, k.byte_size, v.byte_size,
                         o.byte_size});
    p.Projection(q, E, E);
    p.Projection(k, E, KV);
    p.Projection(v, E, KV);
    p.Projection(o, E, E);
  }
  p.Index("loomify.ffn.profile", 0);
  const auto history = std::max<uint64_t>(s.token_capacity, 129);
  const auto pool = (s.context_capacity + 63) / 64 * 64;
  p.Index("llama.model.history_capacity", history);
  p.Index("llama.model.hidden_size", E);
  p.Index("llama.model.vocabulary_size", m.vocabulary_size);
  p.Index("llama.model.layer_count", m.layers.size());
  p.Index("llama.model.query_heads", QH);
  p.Index("llama.model.kv_heads", KH);
  p.Index("llama.model.head_size", H);
  p.Index("llama.model.kv_size", KV);
  p.Real("llama.model.rms_epsilon", float(m.rms_epsilon));
  p.Real("llama.model.rotary_step",
         float(std::pow(m.rope.base, -2.0 / double(m.rope.width))));
  p.Real("llama.model.attention_scale",
         float(1.0 / std::sqrt(double(m.key_head_width))));
  p.Index("llama.attention.cache_capacity", s.context_capacity);
  p.Index("llama.attention.pool_capacity", pool);
  p.Index("runner.llama32.prefill_token_count", s.token_capacity);
  p.Index("llama.attention.cache_base", 768 + 4 * history, "offset");
  p.Index("llama.attention.map_origin", 2048, "offset");
  p.Index("llama.model.embedding_weight_bytes",
          Role(m.globals, "token_embedding")->byte_size);
  p.Index("llama.model.output_weight_bytes",
          Role(m.globals, "output")->byte_size);
  p.Index("qwen38.greedy_argmax.output_capacity", m.vocabulary_size);

  p.ProjectionCapacities();
  p.Lookup("loomify.config.layer_ffn", ffn);
  p.Lookup("loomify.config.layer_attention", attention);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}

} // namespace hrx::loomify_gguf::internal
