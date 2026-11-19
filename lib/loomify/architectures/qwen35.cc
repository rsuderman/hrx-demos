// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration_internal.h"
#include "lib/loomify/assembly_registry.h"
#include "lib/loomify/architectures/ffn_binding.h"
namespace hrx::loomify_gguf::internal {
bool EmitQwen(const model::ModelDescriptor &m, const std::string &asset,
              const ConfigurationSettings &o, ConfigurationFacts *out,
              std::string *error) {
  Producer p;
  p.model = &m;
  p.asset = asset;
  p.error = error;
  std::optional<uint32_t> layer;
  auto tensor = [&](const auto &bindings, const std::string &role, uint64_t in,
                    uint64_t out, std::initializer_list<uint32_t> tags) {
    return p.Tensor(bindings, role, in, out, tags, layer);
  };
  if (m.family != "qwen35" || !m.ssm || m.layers.empty() ||
      m.layers.size() > 64 ||
      (m.stored_layer_count && m.stored_layer_count < m.layers.size()))
    return Fail(
        error,
        "Qwen configuration requires supported layer identities and SSM facts");
  const uint64_t E = m.embedding_width, V = m.vocabulary_size;
  const uint64_t F = m.layers.front().feed_forward_width, L = m.layers.size();
  const uint64_t QH = m.layers.front().query_heads,
                 KH = m.layers.front().kv_heads;
  const uint64_t G = m.ssm->value_heads, K = m.ssm->key_heads;
  const uint64_t D = m.ssm->state_width, H = m.key_head_width;
  const uint64_t W = m.ssm->inner_width;
  if (!E || !V || !F || !QH || !KH || QH % KH || !H ||
      H != m.value_head_width || !K || !G || G % K || !D || !W || W % D ||
      W / D != G || !m.ssm->convolution_kernel || !PositiveF32(m.rms_epsilon) ||
      !m.rope.width || m.rope.width % 2 || m.rope.width > H ||
      !PositiveF32(m.rope.base) || m.rope.base < 1 ||
      m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear"))
    return Fail(error,
                "inconsistent Qwen numerical or unsupported rotary facts");
  uint64_t sections = 0;
  for (auto section : m.rope.sections)
    if (!Sum(sections, section, &sections))
      return Fail(error, "overflowing Qwen rotary sections");
  uint64_t section_width;
  if (!Product(2, sections, &section_width) || section_width != m.rope.width)
    return Fail(error, "inconsistent Qwen rotary sections");
  uint64_t attention_width, query_gate_width, kv_width;
  if (!Product(QH, H, &attention_width) ||
      !Product(2, attention_width, &query_gate_width) ||
      !Product(KH, H, &kv_width))
    return Fail(error, "overflowing Qwen attention geometry");
  uint64_t key_channels, QKV, expected_history, expected_state;
  if (!Product(K, D, &key_channels) || !Product(2, key_channels, &QKV) ||
      !Sum(QKV, W, &QKV) ||
      !Product(m.ssm->convolution_kernel - 1, QKV, &expected_history) ||
      !Product(D, W, &expected_state) ||
      expected_history != m.ssm->convolution_history_elements ||
      expected_state != m.ssm->recurrent_state_elements)
    return Fail(error, "inconsistent or overflowing Qwen recurrent geometry");
  if (!tensor(m.globals, "token_embedding", E, V, {}) ||
      !tensor(m.globals, "output", E, V, {}) ||
      !tensor(m.globals, "output_norm", E, 1, {0}))
    return false;
  p.Projection(*Role(m.globals, "output"), E, V);
  uint64_t gdn_count = 0, attention_count = 0;
  std::vector<std::vector<uint64_t>> routes, ffn, attention_projections,
      gdn_projections;
  for (size_t slot = 0; slot < m.layers.size(); ++slot) {
    const auto &l = m.layers[slot];
    layer = l.index;
    if (l.index != slot || l.feed_forward_width != F || l.query_heads != QH ||
        l.kv_heads != KH)
      return Fail(error, "unsupported Qwen layer identity or geometry");
    const auto &t = l.tensors;
    if (!tensor(t, "input_norm", E, 1, {0}) ||
        !tensor(t, "post_norm", E, 1, {0}) ||
        !BindFfnProjections(p, l, E, {}, ffn))
      return false;
    if (l.kind == model::LayerKind::kGatedDeltaNet) {
      if (!l.recurrent_ordinal || *l.recurrent_ordinal != gdn_count ||
          l.attention_ordinal)
        return Fail(error,
                    "Qwen recurrent ordinal disagrees with descriptor order");
      routes.push_back({1, l.index, gdn_count++});
      if (!tensor(t, "recurrent_qkv", E, QKV, {}) ||
          !tensor(t, "recurrent_gate", E, W, {}) ||
          !tensor(t, "convolution", m.ssm->convolution_kernel, QKV, {0}) ||
          !tensor(t, "dt_bias", G, 1, {0}) || !tensor(t, "decay", G, 1, {0}) ||
          !tensor(t, "alpha", E, G, {}) || !tensor(t, "beta", E, G, {}) ||
          !tensor(t, "recurrent_norm", D, 1, {0}) ||
          !tensor(t, "recurrent_output", W, E, {}))
        return false;
      for (const auto &use :
           std::vector<std::tuple<const char *, uint64_t, uint64_t>>{
               {"recurrent_qkv", E, QKV},
               {"recurrent_gate", E, W},
               {"alpha", E, G},
               {"beta", E, G},
               {"recurrent_output", W, E}})
        p.Projection(*Role(t, std::get<0>(use)), std::get<1>(use),
                     std::get<2>(use));
      std::vector<uint64_t> row;
      for (const char *role : {"recurrent_qkv", "recurrent_gate", "alpha",
                               "beta", "recurrent_output"})
        row.push_back(Role(t, role)->storage_type);
      for (const char *role : {"recurrent_qkv", "recurrent_gate", "alpha",
                               "beta", "recurrent_output"})
        row.push_back(Role(t, role)->byte_size);
      gdn_projections.push_back(std::move(row));
    } else if (l.kind == model::LayerKind::kAttention) {
      if (!l.attention_ordinal || *l.attention_ordinal != attention_count ||
          l.recurrent_ordinal)
        return Fail(error,
                    "Qwen attention ordinal disagrees with descriptor order");
      routes.push_back({0, l.index, attention_count++});
      if (!tensor(t, "query", E, query_gate_width, {}) ||
          !tensor(t, "key", E, kv_width, {}) ||
          !tensor(t, "value", E, kv_width, {}) ||
          !tensor(t, "attention_output", attention_width, E, {}) ||
          !tensor(t, "query_norm", H, 1, {0}) ||
          !tensor(t, "key_norm", H, 1, {0}))
        return false;
      for (const auto &use :
           std::vector<std::tuple<const char *, uint64_t, uint64_t>>{
               {"query", E, query_gate_width},
               {"key", E, kv_width},
               {"value", E, kv_width},
               {"attention_output", attention_width, E}})
        p.Projection(*Role(t, std::get<0>(use)), std::get<1>(use),
                     std::get<2>(use));
      std::vector<uint64_t> row;
      for (const char *role : {"query", "key", "value", "attention_output"})
        row.push_back(Role(t, role)->storage_type);
      for (const char *role : {"query", "key", "value", "attention_output"})
        row.push_back(Role(t, role)->byte_size);
      attention_projections.push_back(std::move(row));
    } else
      return Fail(error, "unsupported Qwen layer kind");
  }
  // These are the complete consumers' actual ordinal bounds, not a checkpoint
  // layer ratio. Both lookup products currently require at least one row.
  if (!gdn_count || gdn_count > 48 || !attention_count || attention_count > 16)
    return Fail(error, "unsupported Qwen recurrent/attention ordinal counts");
  const auto pool = (o.context_capacity + 63) / 64 * 64;
  uint64_t history_bytes, recurrent_bytes, layer_bytes, state_bytes;
  uint64_t plane_token_bytes, all_token_bytes, capacity_bytes, private_stride,
      arena_bytes;
  const auto plane_count = 2 * attention_count;
  if (!Product(4, expected_history, &history_bytes) ||
      !Product(4, expected_state, &recurrent_bytes) ||
      !Sum(history_bytes, recurrent_bytes, &layer_bytes) ||
      !Product(gdn_count, layer_bytes, &state_bytes) ||
      !Product(KH, H, &plane_token_bytes) ||
      !Product(2, plane_token_bytes, &plane_token_bytes) ||
      !Product(plane_count, plane_token_bytes, &all_token_bytes) ||
      !Product(all_token_bytes, pool, &capacity_bytes) ||
      !Sum(state_bytes, 256 + 2048 + 256, &private_stride) ||
      !Sum(private_stride, capacity_bytes, &arena_bytes))
    return Fail(error, "overflowing Qwen state or cache allocation");
  for (auto width : {E, F, W, QKV, query_gate_width, kv_width, V})
    if (!Product(width, 512 * 4, &capacity_bytes))
      return Fail(error, "overflowing Qwen scratch allocation");
  p.Index("qwen.model.hidden_size", E);
  p.Index("qwen.model.ffn_size", F);
  p.Index("qwen.model.vocabulary_size", V);
  p.Index("qwen.model.base_layers", m.layers.size());
  p.Index("qwen.model.checkpoint_layers",
          m.stored_layer_count ? m.stored_layer_count : m.layers.size());
  p.Index("qwen.model.group_count", attention_count);
  p.Index("qwen.model.attention_heads", QH);
  p.Index("qwen.model.kv_heads", KH);
  p.Index("qwen.model.head_size", H);
  p.Index("qwen.model.attention_width", attention_width);
  p.Index("qwen.model.kv_size", kv_width);
  p.Index("qwen.model.query_gate_size", query_gate_width);
  p.Index("qwen.model.rope_size", m.rope.width);
  p.Index("qwen.model.rope_half", m.rope.width / 2);
  p.Real("qwen.model.rope_theta", m.rope.base);
  p.Real("qwen.model.rope_log_step",
         -2.0 * std::log(m.rope.base) / m.rope.width);
  p.Real("qwen.model.rms_epsilon", m.rms_epsilon);
  p.Real("qwen.model.attention_scale", 1.0 / std::sqrt(double(H)));
  p.Real("qwen.model.head_divisor", H);
  p.Real("qwen.model.hidden_divisor", E);
  p.Real("qwen.model.gdn_divisor", D);
  p.Real("qwen.model.gdn_scale", 1.0 / std::sqrt(double(D)));
  p.Index("qwen.model.gdn_head_size", D);
  p.Index("qwen.model.gdn_key_heads", K);
  p.Index("qwen.model.gdn_value_heads", G);
  p.Index("qwen.model.gdn_width", W);
  p.Index("qwen.model.gdn_q_channels", K * D);
  p.Index("qwen.model.gdn_qkv_size", QKV);
  p.Index("qwen.model.gdn_v_base", 2 * K * D);
  p.Index("qwen.model.gdn_convolution_taps", m.ssm->convolution_kernel);
  p.Index("qwen.model.gdn_history_values", m.ssm->convolution_kernel - 1);
  p.Index("qwen.model.gdn_history_elements",
          m.ssm->convolution_history_elements);
  p.Index("qwen.model.gdn_history_bytes", history_bytes);
  p.Index("qwen.model.gdn_state_elements", m.ssm->recurrent_state_elements);
  p.Index("qwen.model.gdn_recurrent_bytes", recurrent_bytes);
  p.Index("qwen.model.gdn_layer_bytes", layer_bytes);
  p.Index("qwen.model.gdn_layers", gdn_count);
  p.Index("qwen.model.gdn_state_bytes", state_bytes);
  p.Index("qwen.model.gdn_conv_elements", QKV * m.ssm->convolution_kernel);
  p.Index("qwen.model.gdn_value_channels_per_key_head", W / K);
  p.Index("qwen.model.gdn_column_tiles", D / 4);
  p.Index("qwen.model.gdn_qk_norm_elements", 2 * K);
  p.Index("qwen.model.gdn_inverse_row_bytes", 2 * K * 4);
  p.Index("qwen.model.gdn_gate_row_bytes", G * 4);
  p.Index("qwen.model.gdn_qkv_row_bytes", QKV * 4);
  // This inherited name is a byte row stride in the selected numerical body.
  p.Index("qwen.model.gdn_capture_qkv_elements", QKV * 4);
  p.Index("qwen.model.projection_row_bytes", W * 4);
  p.Index("qwen.model.hidden_row_bytes", E * 4);
  p.Index("qwen.model.ffn_row_bytes", F * 4);
  p.Index("qwen.model.query_gate_row_bytes", query_gate_width * 4);
  p.Index("qwen.model.kv_plane_token_bytes", plane_token_bytes);
  p.Index("qwen.model.kv_row_bytes", plane_token_bytes * 2);
  p.Index("qwen.model.target_kv_planes", plane_count);
  p.Index("qwen.model.target_kv_token_bytes", all_token_bytes);
  p.Index("qwen.model.head_gate_size", 2 * H);
  p.Index("qwen.model.hidden_qk_blocks", E / 256);
  p.Index("qwen.model.ffn_qk_blocks", F / 256);
  p.Index("qwen.model.hidden_q8_row_bytes", E / 128 * 144);
  p.Index("qwen.model.hidden_q8_depth4_bytes", 4 * E / 128 * 144);
  p.Index("qwen.model.projection_q8_row_bytes", W / 128 * 144);
  p.Index("qwen.model.projection_q8_depth4_bytes", 4 * W / 128 * 144);
  p.Index("qwen.model.ffn_q8_row_bytes", F / 128 * 144);
  p.Index("qwen.model.logits_row_bytes", V * 4);
  p.Index("qwen.model.argmax_partial_row_bytes", ((V + 4095) / 4096) * 4);
  p.Index("qwen.model.embedding_storage_type",
          Role(m.globals, "token_embedding")->storage_type);
  p.Index("qwen.model.output_storage_type",
          Role(m.globals, "output")->storage_type);
  p.Index("qwen.model.tie_word_embeddings", m.output_tied_to_embedding);
  p.Index("qwen.model.embedding_weight_bytes",
          Role(m.globals, "token_embedding")->byte_size);
  p.Index("qwen.model.output_weight_bytes",
          Role(m.globals, "output")->byte_size);
  p.Index("qwen.model.embedding_row_bytes",
          Role(m.globals, "token_embedding")->byte_size / V);
  p.Index("qwen.model.output_weight_row_bytes",
          Role(m.globals, "output")->byte_size / V);
  p.Index("qwen.model.projection_implementation", 1);
  p.Index("qwen.model.hidden_capacity_bytes", 512 * m.embedding_width * 4);
  p.Index("qwen.model.ffn_capacity_bytes", 512 * F * 4);
  p.Index("qwen.model.gdn_f16_capacity_bytes", 512 * W * 2);
  p.Index("qwen.model.gdn_gate_capacity_bytes", 512 * G * 4);
  p.Index("qwen.model.hidden_q8_capacity_bytes", 512 * E / 128 * 144);
  p.Index("qwen.model.kv_capacity_bytes", 512 * kv_width * 4);
  p.Index("qwen.model.projection_capacity_bytes",
          512 * std::max(W, attention_width) * 4);
  p.Index("qwen.model.qkv_capacity_bytes", 512 * QKV * 4);
  p.Index("qwen.model.query_gate_capacity_bytes", 512 * query_gate_width * 4);
  p.Index("qwen.model.embedding_chunks", E / 16);
  p.Index("qwen.model.output_projection_iterations", 1);
  p.Index("qwen.model.head_scratch_bytes", 1024);
  p.Index("qwen.model.attention_partial_split_bytes", 8 * (H + 2) * 4);
  p.Index("qwen.model.attention_partial_bytes", 10 * 8 * (H + 2) * 4);
  p.Index("qwen38.attention.decode_split_count", 10);
  p.Index("qwen38.attention.cache_capacity", o.context_capacity);
  p.Index("qwen38.attention.pool_capacity", pool);
  p.Index("runner.qwen38.prefill_token_count", o.token_capacity);
  // Experimental first qualification profile. Keep every other configuration
  // on its existing stage routing until its state and performance gates pass.
  p.facts.specialized_decode =
      o.token_capacity == 128 &&
      (o.context_capacity == 512 || o.context_capacity == 2048) && E == 1024 &&
      F == 3584 && L == 24 && QH == 8 && KH == 2 && H == 256 && D == 128 &&
      K == 16 && G == 16;
  p.Index("runner.qwen38.specialized_decode", p.facts.specialized_decode);
  p.Index("runner.qwen38.span_capacity", 1);
  p.Index("qwen38.greedy_argmax.output_capacity", V);
  p.ProjectionCapacities();
  p.Index("qwen.model.gdn_slot_origin", 256);
  p.Index("qwen.model.private_row_stride", private_stride);
  p.Index("qwen.model.map_origin", 2048);
  p.Index("loomify.ffn.profile", 1);
  p.Lookup("loomify.config.layer_route", routes);
  p.Lookup("loomify.config.attention_projections", attention_projections);
  p.Lookup("loomify.config.gdn_projections", gdn_projections);
  p.Lookup("loomify.config.layer_ffn", ffn);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
