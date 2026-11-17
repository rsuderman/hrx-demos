// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/assembly_registry.h"
#include "lib/loomify/configuration_internal.h"
namespace hrx::loomify_gguf::internal {
bool EmitLfm3(const model::ModelDescriptor &m, const std::string &asset,
              const ConfigurationSettings &s, ConfigurationFacts *out,
              std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  if (m.family != "lfm3" || m.layers.empty() || m.layers.size() > 64 ||
      E < 1024 || E > 8192 || E % 1024 || !V || V > 262144 || H != 128 ||
      m.value_head_width != H || m.rope.width != H || m.ssm ||
      !m.short_convolution || m.short_convolution->width != E ||
      m.short_convolution->kernel != 7 ||
      m.short_convolution->history_rows != 6 || m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !PositiveF32(m.rms_epsilon) || !PositiveF32(m.rope.base) ||
      m.rope.base < 1)
    return Fail(error, "unsupported LFM3 geometry, convolution or rotary");
  const auto QH = m.layers.front().query_heads;
  uint64_t KH = 0;
  for (const auto &l : m.layers)
    if (l.kind == model::LayerKind::kAttention) {
      KH = l.kv_heads;
      break;
    }
  uint64_t Q, KV;
  if (!QH || QH > 128 || !KH || KH > QH || QH % KH || !Product(QH, H, &Q) ||
      Q > 32768 || !Product(KH, H, &KV))
    return Fail(error, "unsupported LFM3 attention head geometry");
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  OrdinaryEmbeddingFacts embedding;
  CanonicalProjectionFacts output;
  if (!p.OrdinaryEmbedding(m.globals, "token_embedding", E, V, &embedding) ||
      !p.CanonicalProjection(m.globals, "output", E, V, {}, &output) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}) ||
      !p.Tensor(m.globals, "rope_frequencies", H / 2, 1, {0}))
    return false;
  std::vector<std::vector<uint64_t>> info, attention, attention_gate, ffn, conv;
  uint64_t na = 0, nc = 0;
  double localbase = 0;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    const auto F = l.feed_forward_width;
    if (l.index != i || l.query_heads != QH || !F || F > 32768 || F % 256 ||
        l.recurrent_ordinal)
      return Fail(error, "unsupported LFM3 layer identity or FFN geometry");
    for (const char *role :
         {"input_norm", "attention_post_norm", "ffn_norm", "ffn_post_norm"})
      if (!p.Tensor(l.tensors, role, E, 1, {0}, l.index))
        return false;
    CanonicalProjectionFacts g, u, d;
    if (!p.CanonicalProjection(l.tensors, "ffn_gate", E, F, l.index, &g) ||
        !p.CanonicalProjection(l.tensors, "ffn_up", E, F, l.index, &u) ||
        !p.CanonicalProjection(l.tensors, "ffn_down", F, E, l.index, &d))
      return false;
    ffn.push_back({F, g.storage_tag, u.storage_tag, d.storage_tag,
                   g.weight_bytes, u.weight_bytes, d.weight_bytes});
    if (l.kind == model::LayerKind::kShortConvolution) {
      if (l.kv_heads || l.attention_ordinal || l.attention_window ||
          !l.convolution_ordinal || *l.convolution_ordinal != nc)
        return Fail(error, "invalid LFM3 short-convolution ordinal or routing");
      CanonicalProjectionFacts in, out;
      if (!p.Tensor(l.tensors, "convolution", 7, E, {0}, l.index) ||
          !p.CanonicalProjection(l.tensors, "convolution_input", E, 3 * E,
                                 l.index, &in) ||
          !p.CanonicalProjection(l.tensors, "convolution_output", E, E, l.index,
                                 &out))
        return false;
      info.push_back({1, nc++, 0});
      attention.push_back(std::vector<uint64_t>(8, 0));
      attention_gate.push_back({0, 0});
      conv.push_back(
          {in.storage_tag, in.weight_bytes, out.storage_tag, out.weight_bytes});
    } else if (l.kind == model::LayerKind::kAttention) {
      if (l.kv_heads != KH || l.convolution_ordinal || !l.attention_ordinal ||
          *l.attention_ordinal != na || !PositiveF32(l.rope_base) ||
          l.rope_base < 1 || l.attention_window > 262144)
        return Fail(error, "invalid LFM3 attention ordinal, window or rotary");
      if (l.attention_window) {
        if (localbase && localbase != l.rope_base)
          return Fail(error, "inconsistent LFM3 local rotary bases");
        localbase = l.rope_base;
      } else if (l.rope_base != m.rope.base)
        return Fail(error, "inconsistent LFM3 global rotary base");
      CanonicalProjectionFacts q, k, v, o;
      if (!p.Tensor(l.tensors, "query_norm", H, 1, {0}, l.index) ||
          !p.Tensor(l.tensors, "key_norm", H, 1, {0}, l.index) ||
          !p.Tensor(l.tensors, "attention_gate", E, QH, {}, l.index, 2) ||
          !p.CanonicalProjection(l.tensors, "query", E, Q, l.index, &q) ||
          !p.CanonicalProjection(l.tensors, "key", E, KV, l.index, &k) ||
          !p.CanonicalProjection(l.tensors, "value", E, KV, l.index, &v) ||
          !p.CanonicalProjection(l.tensors, "attention_output", Q, E, l.index,
                                 &o))
        return false;
      const auto &gate = *Role(l.tensors, "attention_gate");
      if (gate.storage_type != 0) {
        CanonicalProjectionFacts checked;
        if (!CheckCanonicalProjection(gate, E, QH, &checked, error))
          return false;
        p.Projection(gate, E, QH);
      }
      attention_gate.push_back({gate.storage_type, gate.byte_size});
      info.push_back({0, na++, l.attention_window});
      conv.push_back(std::vector<uint64_t>(4, 0));
      attention.push_back({q.storage_tag, k.storage_tag, v.storage_tag,
                           o.storage_tag, q.weight_bytes, k.weight_bytes,
                           v.weight_bytes, o.weight_bytes});
    } else
      return Fail(error, "unsupported LFM3 layer kind");
  }
  if (!na || !nc)
    return Fail(error, "LFM3 requires attention and convolution layers");
  if (!localbase)
    localbase = m.rope.base;
  const auto history = std::max<uint64_t>(s.token_capacity, 129),
             pool = (s.context_capacity + 63) / 64 * 64;
  const auto base = 768 + 4 * history;
  uint64_t kvbytes, historybytes, historybase;
  if (!Product(na, 2 * pool * KV * 2, &kvbytes) ||
      !Sum(base, kvbytes, &historybase) ||
      !Product(nc, 6 * E * 4, &historybytes) ||
      historybytes > uint64_t(INT64_MAX) - historybase)
    return Fail(error, "LFM3 state extent overflow");
  for (const auto &fact : std::vector<std::pair<const char *, uint64_t>>{
           {"hidden_size", E},
           {"vocabulary_size", V},
           {"layer_count", m.layers.size()},
           {"attention_layer_count", na},
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
    p.Index(std::string("lfm3.model.") + fact.first, fact.second);
  p.Real("lfm3.model.rms_epsilon", float(m.rms_epsilon));
  p.Real("lfm3.model.post_rms_epsilon", float(m.rms_epsilon));
  p.Real("lfm3.model.attention_scale", float(1 / std::sqrt(double(H))));
  p.Real("lfm3.model.rope_log_step", float(-2 * std::log(m.rope.base) / H));
  p.Real("lfm3.model.rope_log_step_swa", float(-2 * std::log(localbase) / H));
  p.Index("runner.lfm3.prefill_token_count", s.token_capacity);
  p.Index("lfm3.shortconv.taps", 7);
  p.Index("lfm3.shortconv.history_base", historybase, "offset");
  p.Index("lfm3.attention.cache_capacity", s.context_capacity);
  p.Index("lfm3.attention.pool_capacity", pool);
  p.Index("lfm3.attention.cache_base", base, "offset");
  p.Index("lfm3.attention.map_origin", 2048, "offset");
  p.Index("qwen38.greedy_argmax.output_capacity", V);
  p.OrdinaryPreparation(E, na, KV, history, s.token_capacity,
                        s.context_capacity, pool, base, 2048, historybytes);
  p.ProjectionCapacities();
  p.Lookup("loomify.config.layer_info", info);
  p.Lookup("loomify.config.layer_attention", attention);
  p.Lookup("loomify.config.layer_attention_gate", attention_gate);
  p.Lookup("loomify.config.layer_ffn", ffn);
  p.Lookup("loomify.config.layer_conv", conv);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
