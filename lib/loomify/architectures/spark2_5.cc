// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/assembly_registry.h"
#include "lib/loomify/configuration_internal.h"
namespace hrx::loomify_gguf::internal {
bool EmitSpark25(const model::ModelDescriptor &m, const std::string &asset,
                 const ConfigurationSettings &s, ConfigurationFacts *out,
                 std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  if (m.family != "spark2_5" || m.layers.empty() || m.layers.size() > 64 ||
      E < 1024 || E > 32768 || E % 256 || V < 256 || V > 262144 || H != 256 ||
      m.value_head_width != H || m.ssm || m.short_convolution ||
      !m.rope.width || m.rope.width > H || m.rope.width % 2 ||
      m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !PositiveF32(m.rms_epsilon) || !PositiveF32(m.rope.base) ||
      m.rope.base < 1)
    return Fail(error, "unsupported Spark geometry or rotary");
  const auto QH = m.layers.front().query_heads;
  const auto KH = m.layers.front().kv_heads;
  uint64_t Q, KV, twiceKV, fused;
  if (!QH || QH > 128 || !KH || KH > QH || QH % KH || !Product(QH, H, &Q) ||
      !Product(KH, H, &KV) || !Product(KV, 2, &twiceKV) ||
      !Sum(Q, twiceKV, &fused))
    return Fail(error, "unsupported Spark attention head geometry");
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  if (m.globals.size() != 3)
    return Fail(error, "invalid Spark global tensor bindings");
  OrdinaryEmbeddingFacts embedding;
  CanonicalProjectionFacts output;
  if (!p.OrdinaryEmbedding(m.globals, "token_embedding", E, V, &embedding) ||
      !p.CanonicalProjection(m.globals, "output", E, V, {}, &output) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  std::vector<std::vector<uint64_t>> attention, ffn;
  std::vector<LayerRotaryFacts> rotary;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    const auto F = l.feed_forward_width;
    const auto rw = l.rope_width ? l.rope_width : m.rope.width;
    if (l.index != i || l.kind != model::LayerKind::kAttention ||
        !l.attention_ordinal || *l.attention_ordinal != i ||
        l.recurrent_ordinal || l.convolution_ordinal || l.query_heads != QH ||
        l.kv_heads != KH || F < 256 || F > 32768 || F % 256 || !rw || rw > H ||
        rw % 2 || !PositiveF32(l.rope_base) || l.rope_base < 1 ||
        l.attention_window > 262144 || l.tensors.size() != 8)
      return Fail(error,
                  "unsupported Spark layer identity, geometry or rotary");
    const auto step = float(-2 * std::log(l.rope_base) / rw);
    if (!std::isfinite(step))
      return Fail(error, "nonfinite Spark rotary step");
    for (const char *role : {"input_norm", "ffn_norm"})
      if (!p.Tensor(l.tensors, role, E, 1, {0}, l.index))
        return false;
    CanonicalProjectionFacts qkv, o, g, u, d;
    if (!p.CanonicalProjection(l.tensors, "query_key_value", E, fused, l.index,
                               &qkv) ||
        !p.CanonicalProjection(l.tensors, "attention_output", Q, E, l.index,
                               &o) ||
        !p.Tensor(l.tensors, "attention_gate", E, QH, {0}, l.index, 2) ||
        !p.CanonicalProjection(l.tensors, "ffn_gate", E, F, l.index, &g) ||
        !p.CanonicalProjection(l.tensors, "ffn_up", E, F, l.index, &u) ||
        !p.CanonicalProjection(l.tensors, "ffn_down", F, E, l.index, &d))
      return false;
    attention.push_back(
        {qkv.storage_tag, o.storage_tag, qkv.weight_bytes, o.weight_bytes});
    ffn.push_back({F, g.storage_tag, u.storage_tag, d.storage_tag,
                   g.weight_bytes, u.weight_bytes, d.weight_bytes});
    rotary.push_back({l.attention_window, rw, step});
  }
  const auto history = std::max<uint64_t>(s.token_capacity, 129);
  const auto pool = (s.context_capacity + 63) / 64 * 64;
  const auto base = 768 + 4 * history;
  uint64_t kvbytes, statebytes;
  if (!Product(m.layers.size(), 2 * pool * KV * 2, &kvbytes) ||
      !Sum(base, kvbytes, &statebytes))
    return Fail(error, "Spark state extent overflow");
  for (const auto &fact : std::vector<std::pair<const char *, uint64_t>>{
           {"hidden_size", E},
           {"vocabulary_size", V},
           {"layer_count", m.layers.size()},
           {"query_size", Q},
           {"kv_size", KV},
           {"query_heads", QH},
           {"kv_heads", KH},
           {"head_size", H},
           {"history_capacity", history},
           {"embedding_weight_bytes", embedding.weight_bytes},
           {"embedding_tag", embedding.storage_tag},
           {"output_weight_bytes", output.weight_bytes},
           {"output_tag", output.storage_tag},
           {"tied_output", m.output_tied_to_embedding}})
    p.Index(std::string("spark2_5.model.") + fact.first, fact.second);
  p.Real("spark2_5.model.rms_epsilon", float(m.rms_epsilon));
  p.Real("spark2_5.model.attention_scale", float(1 / std::sqrt(double(H))));
  p.Index("runner.spark2_5.prefill_token_count", s.token_capacity);
  p.Index("spark2_5.attention.cache_capacity", s.context_capacity);
  p.Index("spark2_5.attention.pool_capacity", pool);
  p.Index("spark2_5.attention.cache_base", base, "offset");
  p.Index("spark2_5.attention.map_origin", 2048, "offset");
  p.Index("qwen38.greedy_argmax.output_capacity", V);
  p.OrdinaryPreparation(E, m.layers.size(), KV, history, s.token_capacity,
                        s.context_capacity, pool, base, 2048);
  p.ProjectionCapacities();
  p.Lookup("loomify.config.layer_attention", attention);
  p.Lookup("loomify.config.layer_ffn", ffn);
  p.LayerRotaryLookup(rotary);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
