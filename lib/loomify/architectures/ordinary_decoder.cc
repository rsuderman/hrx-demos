// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures/ordinary_decoder.h"

namespace hrx::loomify_gguf::internal {
bool BindOrdinaryDecoderMatrices(Producer &p, const model::LayerDescriptor &l,
                                 uint64_t E, uint64_t Q, uint64_t KV,
                                 ConfigRows &attention, ConfigRows &ffn) {
  const auto F = l.feed_forward_width;
  if (!p.Tensor(l.tensors, "query", E, Q, {12, 14}, l.index) ||
      !p.Tensor(l.tensors, "key", E, KV, {12, 14}, l.index) ||
      !p.Tensor(l.tensors, "value", E, KV, {12, 14}, l.index) ||
      !p.Tensor(l.tensors, "attention_output", Q, E, {12, 14}, l.index) ||
      !p.Tensor(l.tensors, "ffn_gate", E, F, {12}, l.index) ||
      !p.Tensor(l.tensors, "ffn_up", E, F, {12}, l.index) ||
      !p.Tensor(l.tensors, "ffn_down", F, E, {12, 14}, l.index))
    return false;
  const auto &q = *Role(l.tensors, "query"), &k = *Role(l.tensors, "key"),
             &v = *Role(l.tensors, "value"),
             &o = *Role(l.tensors, "attention_output"),
             &g = *Role(l.tensors, "ffn_gate"), &u = *Role(l.tensors, "ffn_up"),
             &d = *Role(l.tensors, "ffn_down");
  if (g.storage_type != u.storage_type || g.byte_size != u.byte_size)
    return Fail(p.error, "FFN gate/up must have matching encoded extents");
  attention.push_back({q.storage_type, k.storage_type, v.storage_type,
                       o.storage_type, q.byte_size, k.byte_size, v.byte_size,
                       o.byte_size});
  ffn.push_back({F, g.storage_type, u.storage_type, d.storage_type, g.byte_size,
                 u.byte_size, d.byte_size});
  p.Projection(q, E, Q);
  p.Projection(k, E, KV);
  p.Projection(v, E, KV);
  p.Projection(o, Q, E);
  p.Projection(g, E, F);
  p.Projection(u, E, F);
  p.Projection(d, F, E);
  return true;
}
void EmitOrdinaryDecoderFacts(Producer &p, const model::ModelDescriptor &m,
                              const ConfigurationSettings &s,
                              std::string_view family_prefix, uint64_t Q,
                              uint64_t KV,
                              const OrdinaryEmbeddingFacts &embedding,
                              const ConfigRows &attention,
                              const ConfigRows &ffn) {
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  const auto QH = m.layers.front().query_heads, KH = m.layers.front().kv_heads;
  const auto prefix = std::string(family_prefix) + ".";
  const auto history = std::max<uint64_t>(s.token_capacity, 129);
  p.Index(prefix + "model.hidden_size", E);
  p.Index(prefix + "model.query_size", Q);
  p.Index(prefix + "model.kv_size", KV);
  p.Index(prefix + "model.vocabulary_size", V);
  p.Index(prefix + "model.layer_count", m.layers.size());
  p.Index(prefix + "model.query_heads", QH);
  p.Index(prefix + "model.kv_heads", KH);
  p.Index(prefix + "model.head_size", H);
  p.Index(prefix + "model.history_capacity", history);
  p.Real(prefix + "model.rms_epsilon", float(m.rms_epsilon));
  p.Real(prefix + "model.rope_log_step", float(-2 * std::log(m.rope.base) / H));
  p.Real(prefix + "model.attention_scale", float(1 / std::sqrt(double(H))));
  p.Index(prefix + "model.embedding_tag", embedding.storage_tag);
  p.Index(prefix + "model.embedding_weight_bytes", embedding.weight_bytes);
  p.Index(prefix + "model.output_weight_bytes",
          Role(m.globals, "output")->byte_size);
  p.Index("runner." + std::string(family_prefix) + ".prefill_token_count",
          s.token_capacity);
  p.Index(prefix + "attention.cache_capacity", s.context_capacity);
  p.Index(prefix + "attention.pool_capacity",
          (s.context_capacity + 63) / 64 * 64);
  p.Index(prefix + "attention.cache_base", 768 + 4 * history, "offset");
  p.Index(prefix + "attention.map_origin", 2048, "offset");
  p.Index("qwen38.greedy_argmax.output_capacity", V);
  p.OrdinaryPreparation(E, m.layers.size(), KV, history, s.token_capacity,
                        s.context_capacity, (s.context_capacity + 63) / 64 * 64,
                        768 + 4 * history, 2048);
  p.ProjectionCapacities();
  p.Lookup("loomify.config.layer_attention", attention);
  p.Lookup("loomify.config.layer_ffn", ffn);
}
} // namespace hrx::loomify_gguf::internal
