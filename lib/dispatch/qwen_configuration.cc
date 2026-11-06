// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen.h"
#include "loomc/iree.h"
#include "loomc/source.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>

namespace hrx::dispatch {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
loomc_status_t Error(const std::string &message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str()));
}
const model::TensorBinding *
Role(const std::vector<model::TensorBinding> &bindings,
     const std::string &role) {
  const model::TensorBinding *found = nullptr;
  for (const auto &binding : bindings)
    if (binding.role == role) {
      if (found)
        return nullptr;
      found = &binding;
    }
  return found;
}
bool Tensor(const std::vector<model::TensorBinding> &bindings,
            const std::string &role, const std::string &name, uint64_t in,
            uint64_t out, std::initializer_list<uint32_t> tags,
            std::string *error) {
  const auto *binding = Role(bindings, role);
  if (!binding)
    return Fail(error, "Qwen requires one " + role + " binding");
  const auto &t = binding->tensor;
  if (t.name != name || t.rank != (out == 1 ? 1u : 2u) ||
      std::find(tags.begin(), tags.end(), t.storage_type) == tags.end() ||
      t.normalized_extents !=
          std::array<int64_t, 4>{int64_t(in), int64_t(out), 1, 1})
    return Fail(error,
                "unsupported Qwen tensor name, shape or encoding: " + role);
  uint64_t bytes = in * out * 4;
  if (t.storage_type != 0) {
    if (in % 256)
      return Fail(error, "Qwen encoded input width must be block256");
    bytes = in / 256 * out *
            (t.storage_type == 12   ? 144
             : t.storage_type == 13 ? 176
                                    : 210);
  }
  if (t.byte_size != bytes || t.absolute_offset > uint64_t(INT64_MAX) - bytes)
    return Fail(error, "invalid Qwen encoded tensor byte range: " + role);
  return true;
}
struct Selectors {
  uint32_t gate_q4 = 0, gate_q6 = 0, down_q6 = 0, attention_q6 = 0;
};
bool Admit(const model::ModelDescriptor &m, const QwenNativeOptions &o,
           QwenNativeGeometry *out, Selectors *selectors, std::string *error) {
  if (!out || !selectors || m.family != "qwen35" || m.embedding_width != 1024 ||
      m.vocabulary_size != 248320 || m.layers.size() != 24 ||
      m.key_head_width != 256 || m.value_head_width != 256 ||
      !std::isfinite(m.rms_epsilon) || float(m.rms_epsilon) != 1e-6f ||
      m.rope.width != 64 || m.rope.base != 10000000 ||
      m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      m.rope.sections != std::array<uint64_t, 4>{11, 11, 10, 0} || !m.ssm ||
      m.ssm->convolution_kernel != 4 || m.ssm->inner_width != 2048 ||
      m.ssm->state_width != 128 || m.ssm->key_heads != 16 ||
      m.ssm->value_heads != 16 ||
      m.ssm->convolution_history_elements != 18432 ||
      m.ssm->recurrent_state_elements != 262144 || !m.output_tied_to_embedding)
    return Fail(error, "unsupported Qwen 0.8B native profile");
  if (!o.token_capacity || o.token_capacity > 512 || o.span_capacity != 1 ||
      o.selected_output_capacity != 1 || o.row_capacity != 1 ||
      !o.cache_capacity || o.cache_capacity > 262144 ||
      o.pool_capacity != (o.cache_capacity + 63) / 64 * 64)
    return Fail(error, "unsupported Qwen deployment capacities");
  if (!Tensor(m.globals, "token_embedding", "token_embd.weight", 1024, 248320,
              {14}, error) ||
      !Tensor(m.globals, "output", "token_embd.weight", 1024, 248320, {14},
              error) ||
      !Tensor(m.globals, "output_norm", "output_norm.weight", 1024, 1, {0},
              error))
    return false;
  const auto &embedding = Role(m.globals, "token_embedding")->tensor;
  const auto &output = Role(m.globals, "output")->tensor;
  if (embedding.absolute_offset != output.absolute_offset ||
      embedding.byte_size != output.byte_size)
    return Fail(error, "Qwen tied output must retain the embedding byte range");
  QwenNativeGeometry g;
  g.embedding_width = m.embedding_width;
  g.feed_forward_width = 3584;
  g.vocabulary_size = m.vocabulary_size;
  Selectors selected;
  uint32_t seen = 0;
  for (const auto &l : m.layers) {
    if (l.index >= 32 || seen & (uint32_t(1) << l.index) ||
        l.feed_forward_width != g.feed_forward_width || l.query_heads != 8 ||
        l.kv_heads != 2)
      return Fail(error, "unsupported Qwen layer identity or geometry");
    const auto bit = uint32_t(1) << l.index;
    seen |= bit;
    const auto prefix = "blk." + std::to_string(l.index) + ".";
    const auto &t = l.tensors;
    if (!Tensor(t, "input_norm", prefix + "attn_norm.weight", 1024, 1, {0},
                error) ||
        !Tensor(t, "post_norm", prefix + "post_attention_norm.weight", 1024, 1,
                {0}, error) ||
        !Tensor(t, "ffn_gate", prefix + "ffn_gate.weight", 1024, 3584,
                {12, 13, 14}, error) ||
        !Tensor(t, "ffn_up", prefix + "ffn_up.weight", 1024, 3584, {12, 13, 14},
                error) ||
        !Tensor(t, "ffn_down", prefix + "ffn_down.weight", 3584, 1024, {13, 14},
                error))
      return false;
    const auto gate = Role(t, "ffn_gate")->tensor.storage_type;
    if (gate != Role(t, "ffn_up")->tensor.storage_type)
      return Fail(error, "Qwen whole FFN requires matching gate/up encodings");
    if (gate == 12)
      selected.gate_q4 |= bit;
    if (gate == 14)
      selected.gate_q6 |= bit;
    if (Role(t, "ffn_down")->tensor.storage_type == 14)
      selected.down_q6 |= bit;
    if (l.kind == model::LayerKind::kGatedDeltaNet) {
      if (!l.recurrent_ordinal || *l.recurrent_ordinal != g.gdn_layer_count ||
          l.attention_ordinal)
        return Fail(error,
                    "Qwen recurrent ordinal disagrees with descriptor order");
      ++g.gdn_layer_count;
      if (!Tensor(t, "recurrent_qkv", prefix + "attn_qkv.weight", 1024, 6144,
                  {14}, error) ||
          !Tensor(t, "recurrent_gate", prefix + "attn_gate.weight", 1024, 2048,
                  {14}, error) ||
          !Tensor(t, "convolution", prefix + "ssm_conv1d.weight", 4, 6144, {0},
                  error) ||
          !Tensor(t, "dt_bias", prefix + "ssm_dt.bias", 16, 1, {0}, error) ||
          !Tensor(t, "decay", prefix + "ssm_a", 16, 1, {0}, error) ||
          !Tensor(t, "alpha", prefix + "ssm_alpha.weight", 1024, 16, {13},
                  error) ||
          !Tensor(t, "beta", prefix + "ssm_beta.weight", 1024, 16, {13},
                  error) ||
          !Tensor(t, "recurrent_norm", prefix + "ssm_norm.weight", 128, 1, {0},
                  error) ||
          !Tensor(t, "recurrent_output", prefix + "ssm_out.weight", 2048, 1024,
                  {13}, error))
        return false;
    } else if (l.kind == model::LayerKind::kAttention) {
      if (!l.attention_ordinal ||
          *l.attention_ordinal != g.attention_layer_count ||
          l.recurrent_ordinal)
        return Fail(error,
                    "Qwen attention ordinal disagrees with descriptor order");
      ++g.attention_layer_count;
      if (!Tensor(t, "query", prefix + "attn_q.weight", 1024, 4096, {13, 14},
                  error) ||
          !Tensor(t, "key", prefix + "attn_k.weight", 1024, 512, {13, 14},
                  error) ||
          !Tensor(t, "value", prefix + "attn_v.weight", 1024, 512, {14},
                  error) ||
          !Tensor(t, "attention_output", prefix + "attn_output.weight", 2048,
                  1024, {13}, error) ||
          !Tensor(t, "query_norm", prefix + "attn_q_norm.weight", 256, 1, {0},
                  error) ||
          !Tensor(t, "key_norm", prefix + "attn_k_norm.weight", 256, 1, {0},
                  error))
        return false;
      const auto query = Role(t, "query")->tensor.storage_type;
      if (query != Role(t, "key")->tensor.storage_type)
        return Fail(error,
                    "Qwen attention requires matching query/key encodings");
      if (query == 14)
        selected.attention_q6 |= bit;
    } else
      return Fail(error, "unsupported Qwen layer kind");
  }
  if (g.gdn_layer_count != 18 || g.attention_layer_count != 6)
    return Fail(error, "unsupported Qwen recurrent/attention layer counts");
  g.residual_bytes = 512 * g.embedding_width * 4;
  g.gdn_history_bytes = m.ssm->convolution_history_elements * 4;
  g.gdn_recurrent_bytes = m.ssm->recurrent_state_elements * 4;
  g.gdn_layer_bytes = g.gdn_history_bytes + g.gdn_recurrent_bytes;
  g.gdn_total_bytes = g.gdn_layer_count * g.gdn_layer_bytes;
  g.kv_plane_count = 2 * g.attention_layer_count;
  g.kv_plane_token_bytes = 2 * m.layers.front().kv_heads * m.key_head_width;
  g.kv_plane_stride = o.pool_capacity * g.kv_plane_token_bytes;
  g.kv_pool_bytes = g.kv_plane_stride * g.kv_plane_count;
  g.gdn_slot_origin = 256;
  g.private_row_stride = g.gdn_slot_origin + g.gdn_total_bytes + 2048 + 256;
  g.map_origin = 2048;
  *out = g;
  *selectors = selected;
  return true;
}
std::string Configuration(const model::ModelDescriptor &m,
                          const QwenNativeOptions &o,
                          const QwenNativeGeometry &g, const Selectors &s) {
  std::ostringstream text;
  auto index = [&](const char *key, uint64_t value) {
    text << "config.def @" << key << " = " << value << " : index\n";
  };
  auto real = [&](const char *key, double value) {
    text << "config.def @" << key << " = " << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10)
         << float(value) << " : f32\n";
  };
  auto mask = [&](const char *key, uint32_t value) {
    text << "config.def @" << key << " = " << int64_t(int32_t(value))
         << " : i32\n";
  };
  const auto E = g.embedding_width, F = g.feed_forward_width,
             V = g.vocabulary_size;
  const auto H = m.key_head_width, D = m.ssm->state_width, K = m.ssm->key_heads,
             G = m.ssm->value_heads, W = m.ssm->inner_width,
             QKV = 2 * K * D + W;
  index("qwen.model.hidden_size", E);
  index("qwen.model.ffn_size", F);
  index("qwen.model.vocabulary_size", V);
  index("qwen.model.base_layers", m.layers.size());
  index("qwen.model.checkpoint_layers", m.layers.size());
  index("qwen.model.group_count", g.attention_layer_count);
  index("qwen.model.attention_heads", 8);
  index("qwen.model.kv_heads", 2);
  index("qwen.model.head_size", H);
  index("qwen.model.attention_width", 8 * H);
  index("qwen.model.kv_size", 2 * H);
  index("qwen.model.query_gate_size", 16 * H);
  index("qwen.model.rope_size", m.rope.width);
  index("qwen.model.rope_half", m.rope.width / 2);
  real("qwen.model.rope_theta", m.rope.base);
  real("qwen.model.rope_log_step", -2.0 * std::log(m.rope.base) / m.rope.width);
  real("qwen.model.rms_epsilon", m.rms_epsilon);
  real("qwen.model.attention_scale", 1.0 / std::sqrt(double(H)));
  real("qwen.model.head_divisor", H);
  real("qwen.model.hidden_divisor", E);
  real("qwen.model.gdn_divisor", D);
  real("qwen.model.gdn_scale", 1.0 / std::sqrt(double(D)));
  index("qwen.model.gdn_head_size", D);
  index("qwen.model.gdn_key_heads", K);
  index("qwen.model.gdn_value_heads", G);
  index("qwen.model.gdn_width", W);
  index("qwen.model.gdn_q_channels", K * D);
  index("qwen.model.gdn_qkv_size", QKV);
  index("qwen.model.gdn_v_base", 2 * K * D);
  index("qwen.model.gdn_convolution_taps", m.ssm->convolution_kernel);
  index("qwen.model.gdn_history_values", m.ssm->convolution_kernel - 1);
  index("qwen.model.gdn_history_elements", m.ssm->convolution_history_elements);
  index("qwen.model.gdn_history_bytes", g.gdn_history_bytes);
  index("qwen.model.gdn_state_elements", m.ssm->recurrent_state_elements);
  index("qwen.model.gdn_recurrent_bytes", g.gdn_recurrent_bytes);
  index("qwen.model.gdn_layer_bytes", g.gdn_layer_bytes);
  index("qwen.model.gdn_layers", g.gdn_layer_count);
  index("qwen.model.gdn_state_bytes", g.gdn_total_bytes);
  index("qwen.model.gdn_conv_elements", QKV * m.ssm->convolution_kernel);
  index("qwen.model.gdn_value_channels_per_key_head", W / K);
  index("qwen.model.gdn_column_tiles", D / 4);
  index("qwen.model.gdn_qk_norm_elements", 2 * K);
  index("qwen.model.gdn_inverse_row_bytes", 2 * K * 4);
  index("qwen.model.gdn_gate_row_bytes", G * 4);
  index("qwen.model.gdn_qkv_row_bytes", QKV * 4);
  // This inherited name is a byte row stride in the selected numerical body.
  index("qwen.model.gdn_capture_qkv_elements", QKV * 4);
  index("qwen.model.projection_row_bytes", W * 4);
  index("qwen.model.hidden_row_bytes", E * 4);
  index("qwen.model.ffn_row_bytes", F * 4);
  index("qwen.model.query_gate_row_bytes", 16 * H * 4);
  index("qwen.model.kv_plane_token_bytes", g.kv_plane_token_bytes);
  index("qwen.model.kv_row_bytes", g.kv_plane_token_bytes * 2);
  index("qwen.model.target_kv_planes", g.kv_plane_count);
  index("qwen.model.target_kv_token_bytes",
        g.kv_plane_count * g.kv_plane_token_bytes);
  index("qwen.model.head_gate_size", 2 * H);
  index("qwen.model.hidden_qk_blocks", E / 256);
  index("qwen.model.ffn_qk_blocks", F / 256);
  index("qwen.model.hidden_q8_row_bytes", E / 128 * 144);
  index("qwen.model.hidden_q8_depth4_bytes", 4 * E / 128 * 144);
  index("qwen.model.projection_q8_row_bytes", W / 128 * 144);
  index("qwen.model.projection_q8_depth4_bytes", 4 * W / 128 * 144);
  index("qwen.model.ffn_q8_row_bytes", F / 128 * 144);
  index("qwen.model.logits_row_bytes", V * 4);
  index("qwen.model.argmax_partial_row_bytes", ((V + 4095) / 4096) * 4);
  index("qwen.model.embedding_quantization", 6);
  index("qwen.model.tie_word_embeddings", 1);
  index("qwen.model.embedding_weight_bytes", E / 256 * V * 210);
  index("qwen.model.output_weight_bytes", E / 256 * V * 210);
  index("qwen.model.embedding_row_bytes", E / 256 * 210);
  index("qwen.model.output_weight_row_bytes", E / 256 * 210);
  index("qwen.model.ffn_gate_q4_weight_bytes", E / 256 * F * 144);
  index("qwen.model.ffn_gate_q5_weight_bytes", E / 256 * F * 176);
  index("qwen.model.ffn_gate_q6_weight_bytes", E / 256 * F * 210);
  index("qwen.model.ffn_down_q5_weight_bytes", F / 256 * E * 176);
  index("qwen.model.ffn_down_q6_weight_bytes", F / 256 * E * 210);
  index("qwen.model.ffn_gate_weight_bytes", E / 256 * F * 176);
  index("qwen.model.ffn_down_weight_bytes", F / 256 * E * 210);
  index("qwen.model.query_q5_weight_bytes", E / 256 * 16 * H * 176);
  index("qwen.model.query_q6_weight_bytes", E / 256 * 16 * H * 210);
  index("qwen.model.key_q5_weight_bytes", E / 256 * 2 * H * 176);
  index("qwen.model.key_q6_weight_bytes", E / 256 * 2 * H * 210);
  index("qwen.model.query_weight_bytes", E / 256 * 16 * H * 210);
  index("qwen.model.key_weight_bytes", E / 256 * 2 * H * 210);
  index("qwen.model.value_weight_bytes", E / 256 * 2 * H * 210);
  index("qwen.model.projection_weight_bytes", W / 256 * E * 176);
  index("qwen.model.gdn_qkv_weight_bytes", E / 256 * QKV * 210);
  index("qwen.model.gdn_gate_weight_bytes", E / 256 * W * 210);
  index("qwen.model.gdn_alpha_weight_bytes", E / 256 * G * 176);
  index("qwen.model.gdn_beta_weight_bytes", E / 256 * G * 176);
  mask("qwen.model.ffn_gate_q4_layers", s.gate_q4);
  mask("qwen.model.ffn_gate_q6_layers", s.gate_q6);
  mask("qwen.model.ffn_down_q6_layers", s.down_q6);
  mask("qwen.model.attention_q6_layers", s.attention_q6);
  index("qwen.model.projection_implementation", 1);
  index("qwen.model.hidden_capacity_bytes", g.residual_bytes);
  index("qwen.model.ffn_capacity_bytes", 512 * F * 4);
  index("qwen.model.gdn_f16_capacity_bytes", 512 * W * 2);
  index("qwen.model.gdn_gate_capacity_bytes", 512 * G * 4);
  index("qwen.model.hidden_q8_capacity_bytes", 512 * E / 128 * 144);
  index("qwen.model.kv_capacity_bytes", 512 * 2 * H * 4);
  index("qwen.model.projection_capacity_bytes", 512 * W * 4);
  index("qwen.model.qkv_capacity_bytes", 512 * QKV * 4);
  index("qwen.model.query_gate_capacity_bytes", 512 * 16 * H * 4);
  index("qwen.model.embedding_chunks", E / 16);
  index("qwen.model.output_projection_iterations", 1);
  index("qwen.model.head_scratch_bytes", 1024);
  index("qwen.model.attention_partial_split_bytes", 8 * (H + 2) * 4);
  index("qwen.model.attention_partial_bytes", 10 * 8 * (H + 2) * 4);
  index("qwen38.attention.decode_split_count", 10);
  index("qwen38.attention.cache_capacity", o.cache_capacity);
  index("qwen38.attention.pool_capacity", o.pool_capacity);
  index("runner.qwen38.prefill_token_count", o.token_capacity);
  index("runner.qwen38.span_capacity", o.span_capacity);
  index("qwen38.greedy_argmax.output_capacity", V);
  index("ggml.linear_q4k_q8_1_x4.output_capacity", F);
  index("ggml.linear_q5k_q8_1_x4.output_capacity", QKV);
  index("ggml.linear_q6k_q8_1_x4.output_capacity", V);
  index("ggml.linear_q4k_q8_1_x4.token_capacity", 512);
  index("ggml.linear_q5k_q8_1_x4.token_capacity", 512);
  index("ggml.linear_q6k_q8_1_x4.token_capacity", 512);
  index("ggml.quantize_q8_1_x4.group_capacity", 32768);
  return text.str();
}
} // namespace
bool DeriveQwenNativeGeometry(const model::ModelDescriptor &m,
                              const QwenNativeOptions &o,
                              QwenNativeGeometry *out, std::string *error) {
  QwenNativeGeometry geometry;
  Selectors selectors;
  if (!out || !Admit(m, o, &geometry, &selectors, error))
    return false;
  *out = geometry;
  return true;
}
loomc_status_t
LoadQwenNative(loomc_context_t *context, loomc_workspace_t *workspace,
               const Registry &registry, const std::vector<Id> &ids,
               const model::ModelDescriptor &m, const QwenNativeOptions &o,
               const std::string &root, QwenNativeGeometry *geometry,
               loomc_module_t **out, loomc_result_t **result) {
  if (out)
    *out = nullptr;
  if (result)
    *result = nullptr;
  if (!context || !workspace || !geometry || !out || !result || ids.empty())
    return Error(
        "Qwen load requires context, workspace, registrations and outputs");
  QwenNativeGeometry admitted;
  Selectors selectors;
  std::string error;
  if (!Admit(m, o, &admitted, &selectors, &error))
    return Error(error);
  std::vector<NativeSource> sources;
  std::vector<Id> seen;
  Registry validation;
  for (auto id : ids) {
    if (std::find(seen.begin(), seen.end(), id) != seen.end())
      continue;
    seen.push_back(id);
    const auto entries = registry.Implementations(id);
    if (entries.empty())
      return Error("missing Qwen dispatch registration");
    for (const auto &entry : entries) {
      if (!validation.Add(entry, &error))
        return Error(error);
      sources.push_back({entry.symbol, entry.kind, entry.interface_file,
                         entry.provider_files, entry.required_config,
                         entry.interface_symbol,
                         entry.builder.required_definitions});
    }
  }
  const auto text = Configuration(m, o, admitted, selectors);
  loomc_source_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  options.identifier = loomc_make_cstring_view("qwen-configuration.loom");
  options.contents = loomc_make_byte_span(text.data(), text.size());
  loomc_source_t *source = nullptr;
  auto status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  if (!loomc_status_is_ok(status))
    return status;
  loomc_module_t *configuration = nullptr;
  status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(),
      &configuration, result);
  loomc_source_release(source);
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> owned(
      configuration, loomc_module_release);
  if (!loomc_status_is_ok(status) || !*result ||
      !loomc_result_succeeded(*result))
    return status;
  loomc_result_release(*result);
  *result = nullptr;
  status =
      LoadNative(context, workspace, sources, root, configuration, out, result);
  if (loomc_status_is_ok(status) && *result && loomc_result_succeeded(*result))
    *geometry = admitted;
  return status;
}
} // namespace hrx::dispatch
