// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures/spark2_5.h"
#include "lib/loomify/configuration_internal.h"
namespace hrx::loomify_gguf::internal {
namespace {
bool ProjectionGeometry(const model::TensorRecord &t, uint64_t in,
                        uint64_t out) {
  if (!out || out > 262144)
    return false;
  switch (t.storage_type) {
  case 12:
  case 13:
    return in >= 256 && in <= 32768 && in % 256 == 0;
  case 14:
    return in >= 256 && in <= 17408 && in % 256 == 0;
  case 8:
    return in >= 128 && in <= 32768 && in % 128 == 0;
  default:
    return false;
  }
}
} // namespace
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
  if (!p.Tensor(m.globals, "token_embedding", E, V, {14}, {}, 2) ||
      !p.Tensor(m.globals, "output", E, V, {12, 13, 14, 8}, {}, 2) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  const auto &embedding = *Role(m.globals, "token_embedding");
  const auto &output = *Role(m.globals, "output");
  if (!ProjectionGeometry(output, E, V))
    return Fail(error, "unsupported Spark output projection geometry");
  p.Projection(output, E, V);
  std::vector<std::vector<uint64_t>> attention, ffn, windows;
  double localbase = m.rope.base;
  uint64_t localwidth = m.rope.width;
  bool haveLocal = false;
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
    if (l.attention_window) {
      if (haveLocal && (localbase != l.rope_base || localwidth != rw))
        return Fail(error, "inconsistent Spark local rotary profile");
      haveLocal = true;
      localbase = l.rope_base;
      localwidth = rw;
    } else if (l.rope_base != m.rope.base || rw != m.rope.width)
      return Fail(error, "inconsistent Spark global rotary profile");
    for (const char *role : {"input_norm", "ffn_norm"})
      if (!p.Tensor(l.tensors, role, E, 1, {0}, l.index))
        return false;
    if (!p.Tensor(l.tensors, "query_key_value", E, fused, {12, 13, 14, 8},
                  l.index) ||
        !p.Tensor(l.tensors, "attention_output", Q, E, {12, 13, 14, 8},
                  l.index) ||
        !p.Tensor(l.tensors, "attention_gate", E, QH, {0}, l.index, 2) ||
        !p.Tensor(l.tensors, "ffn_gate", E, F, {12, 13, 14, 8}, l.index) ||
        !p.Tensor(l.tensors, "ffn_up", E, F, {12, 13, 14, 8}, l.index) ||
        !p.Tensor(l.tensors, "ffn_down", F, E, {12, 13, 14, 8}, l.index))
      return false;
    const auto &qkv = *Role(l.tensors, "query_key_value"),
               &o = *Role(l.tensors, "attention_output"),
               &g = *Role(l.tensors, "ffn_gate"),
               &u = *Role(l.tensors, "ffn_up"),
               &d = *Role(l.tensors, "ffn_down");
    for (const auto &projection : std::vector<
             std::tuple<const model::TensorRecord *, uint64_t, uint64_t>>{
             {&qkv, E, fused},
             {&o, Q, E},
             {&g, E, F},
             {&u, E, F},
             {&d, F, E}}) {
      const auto &[t, in, width] = projection;
      if (!ProjectionGeometry(*t, in, width))
        return Fail(error, "unsupported Spark projection geometry: " + t->name);
      p.Projection(*t, in, width);
    }
    attention.push_back(
        {qkv.storage_type, o.storage_type, qkv.byte_size, o.byte_size});
    ffn.push_back({F, g.storage_type, u.storage_type, d.storage_type,
                   g.byte_size, u.byte_size, d.byte_size});
    windows.push_back({l.attention_window, rw});
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
           {"embedding_weight_bytes", embedding.byte_size},
           {"output_weight_bytes", output.byte_size},
           {"output_tag", output.storage_type},
           {"tied_output", m.output_tied_to_embedding}})
    p.Index(std::string("spark2_5.model.") + fact.first, fact.second);
  p.Real("spark2_5.model.rms_epsilon", float(m.rms_epsilon));
  p.Real("spark2_5.model.attention_scale", float(1 / std::sqrt(double(H))));
  p.Real("spark2_5.model.rope_log_step_local",
         float(-2 * std::log(localbase) / localwidth));
  p.Real("spark2_5.model.rope_log_step_global",
         float(-2 * std::log(m.rope.base) / m.rope.width));
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
  p.Lookup("loomify.config.layer_window", windows);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
