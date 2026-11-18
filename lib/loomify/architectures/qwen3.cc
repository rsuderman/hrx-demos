// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures/ordinary_decoder.h"
#include "lib/loomify/assembly_registry.h"

namespace hrx::loomify_gguf::internal {
bool EmitQwen3(const model::ModelDescriptor &m, const std::string &asset,
               const ConfigurationSettings &s, ConfigurationFacts *out,
               std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto E = m.embedding_width, V = m.vocabulary_size, H = m.key_head_width;
  if (m.family != "qwen3" || m.layers.empty() || m.layers.size() > 64 || !E ||
      E > 32768 || E % 256 || !V || V > 262144 || H != 128 ||
      m.value_head_width != H || m.ssm || !m.output_tied_to_embedding ||
      m.rope.width != H || m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !PositiveF32(m.rms_epsilon) || !PositiveF32(m.rope.base) ||
      m.rope.base < 1)
    return Fail(error, "unsupported Qwen3 geometry, rotary or output binding");
  const auto QH = m.layers.front().query_heads, KH = m.layers.front().kv_heads;
  uint64_t Q, KV;
  if (!QH || QH > 128 || !KH || KH > QH || QH % KH || !Product(QH, H, &Q) ||
      Q > 32768 || !Product(KH, H, &KV))
    return Fail(error, "unsupported Qwen3 attention head geometry");
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  OrdinaryEmbeddingFacts embedding;
  if (!p.OrdinaryEmbedding(m.globals, "token_embedding", E, V, &embedding) ||
      !p.Tensor(m.globals, "output", E, V, {14}) ||
      !p.Tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  p.Projection(*Role(m.globals, "output"), E, V);
  ConfigRows ffn, attention;
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    const auto F = l.feed_forward_width;
    if (l.index != i || l.kind != model::LayerKind::kAttention ||
        l.query_heads != QH || l.kv_heads != KH || !F || F > 32768 || F % 256 ||
        l.attention_window || l.rope_base != m.rope.base)
      return Fail(error, "unsupported Qwen3 layer identity or geometry");
    if (!p.Tensor(l.tensors, "input_norm", E, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "post_norm", E, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "query_norm", H, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "key_norm", H, 1, {0}, l.index))
      return false;
    if (!BindOrdinaryDecoderMatrices(p, l, E, Q, KV, attention, ffn))
      return false;
  }
  EmitOrdinaryDecoderFacts(p, m, s, "qwen3", Q, KV, embedding, attention, ffn);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
