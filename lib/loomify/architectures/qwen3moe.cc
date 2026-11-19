// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures/ordinary_decoder.h"
#include "lib/loomify/assembly_registry.h"

namespace hrx::loomify_gguf::internal {
bool EmitQwen3Moe(const model::ModelDescriptor &m, const std::string &asset,
                  const ConfigurationSettings &s, ConfigurationFacts *out,
                  std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  if (m.family != "qwen3moe" || m.layers.empty() || m.layers.size() > 64 ||
      E != 2048 || !V || V > 262144 || H != 128 || m.value_head_width != H ||
      m.ssm || m.short_convolution || !m.routed_experts ||
      m.routed_experts->expert_count != 128 ||
      m.routed_experts->experts_per_token != 8 ||
      m.routed_experts->feed_forward_width != 768 || m.rope.width != H ||
      m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !PositiveF32(m.rms_epsilon) || !PositiveF32(m.rope.base) ||
      m.rope.base < 1)
    return Fail(error,
                "unsupported Qwen3MoE geometry, rotary or expert profile");
  const auto QH = m.layers.front().query_heads, KH = m.layers.front().kv_heads;
  uint64_t Q, KV;
  if (!QH || QH > 128 || !KH || KH > QH || QH % KH || !Product(QH, H, &Q) ||
      Q > 32768 || !Product(KH, H, &KV))
    return Fail(error, "unsupported Qwen3MoE attention head geometry");
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  OrdinaryEmbeddingFacts embedding;
  CanonicalProjectionFacts output;
  if (!p.OrdinaryEmbedding(m.globals, "token_embedding", E, V, &embedding) ||
      !p.CanonicalProjection(m.globals, "output", E, V, std::nullopt,
                             &output) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  p.Projection(*Role(m.globals, "output"), E, V);
  ConfigRows attention, experts;
  const auto F = m.routed_experts->feed_forward_width;
  const auto X = m.routed_experts->expert_count;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    if (l.index != i || l.kind != model::LayerKind::kAttention ||
        l.query_heads != QH || l.kv_heads != KH || l.attention_window ||
        l.rope_base != m.rope.base || (l.rope_width && l.rope_width != H))
      return Fail(error, "unsupported Qwen3MoE layer identity or geometry");
    if (!p.Tensor(l.tensors, "input_norm", E, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "post_norm", E, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "query_norm", H, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "key_norm", H, 1, {0}, l.index) ||
        !BindOrdinaryAttentionMatrices(p, l, E, Q, KV, attention) ||
        !p.Tensor(l.tensors, "expert_router", E, X, {0}, l.index, 2) ||
        !p.TensorShape(l.tensors, "expert_gate", {E, F, X}, {12, 13, 14, 8},
                       l.index) ||
        !p.TensorShape(l.tensors, "expert_up", {E, F, X}, {12, 13, 14, 8},
                       l.index) ||
        !p.TensorShape(l.tensors, "expert_down", {F, E, X}, {12, 13, 14, 8, 6},
                       l.index))
      return false;
    const auto &gate = *Role(l.tensors, "expert_gate");
    const auto &up = *Role(l.tensors, "expert_up");
    const auto &down = *Role(l.tensors, "expert_down");
    experts.push_back({gate.storage_type, up.storage_type, down.storage_type,
                       gate.byte_size, up.byte_size, down.byte_size});
  }
  // qwen3 is the reused backbone ABI; public resources/root remain qwen3moe.
  EmitOrdinaryDecoderBaseFacts(p, m, s, "qwen3", Q, KV, embedding, attention);
  p.Index("qwen3.model.output_tag", output.storage_tag);
  p.Index("qwen3.model.tie_word_embeddings", m.output_tied_to_embedding);
  p.Lookup("loomify.config.layer_experts", experts);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
