// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/configuration.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
const model::TensorRecord *Role(const std::vector<model::TensorBinding> &items,
                                const std::string &role) {
  const model::TensorRecord *found = nullptr;
  for (const auto &item : items)
    if (item.role == role) {
      if (found)
        return nullptr;
      found = &item.tensor;
    }
  return found;
}
bool SameTensor(const model::TensorRecord &a, const model::TensorRecord &b) {
  return a.name == b.name && a.rank == b.rank &&
         a.normalized_extents == b.normalized_extents &&
         a.storage_type == b.storage_type &&
         a.absolute_offset == b.absolute_offset && a.byte_size == b.byte_size;
}
// Keep source checkpoint ranges separate from native origins. Restrict both
// metadata and derived geometry to the native signed index/offset domain.
bool Product(uint64_t a, uint64_t b, uint64_t *out) {
  if (b && a > uint64_t(INT64_MAX) / b)
    return false;
  *out = a * b;
  return true;
}
bool Sum(uint64_t a, uint64_t b, uint64_t *out) {
  if (a > uint64_t(INT64_MAX) || b > uint64_t(INT64_MAX) - a)
    return false;
  *out = a + b;
  return true;
}
bool PositiveF32(double value) {
  return std::isfinite(value) && std::isfinite(float(value)) && float(value) > 0;
}
struct Producer {
  ConfigurationFacts facts;
  std::map<std::string, model::TensorRecord> identities;
  std::ostringstream text;
  std::string asset;
  std::string *error;
  std::map<uint32_t, uint64_t> projection_widths;
  uint64_t packed_input_width = 0;
  void Projection(const model::TensorRecord &tensor, uint64_t input,
                  uint64_t output) {
    projection_widths[tensor.storage_type] =
        std::max(projection_widths[tensor.storage_type], output);
    packed_input_width = std::max(packed_input_width, input);
  }
  void ProjectionCapacities() {
    for (const auto &entry : std::vector<std::pair<uint32_t, const char *>>{
             {12, "ggml.linear_q4k_q8_1_x4."},
             {13, "ggml.linear_q5k_q8_1_x4."},
             {14, "ggml.linear_q6k_q8_1_x4."},
             {8, "ggml.linear_q8_0_q8_1_x4."}}) {
      // The operation owns format support. This only sizes the reused product
      // from its actual tensor consumers; unused products receive a minimal bound.
      Index(std::string(entry.second) + "token_capacity", 512);
      Index(std::string(entry.second) + "output_capacity",
            std::max<uint64_t>(1, projection_widths[entry.first]));
    }
    Index("ggml.quantize_q8_1_x4.group_capacity",
          std::max<uint64_t>(1, 512 * packed_input_width / 128));
  }
  void Index(const std::string &name, uint64_t value,
             const char *type = "index") {
    text << "config.def @" << name << " = " << value << " : " << type << '\n';
  }
  void Real(const std::string &name, float value) {
    text << "config.def @" << name << " = " << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << value
         << " : f32\n";
  }
  bool Tensor(const std::vector<model::TensorBinding> &bindings,
              const std::string &role, const std::string &key, uint64_t in,
              uint64_t out, std::initializer_list<uint32_t> tags,
              std::optional<uint32_t> layer = std::nullopt) {
    const auto *t = Role(bindings, role);
    if (!t)
      return Fail(error, "requires exactly one tensor role: " + role);
    if (t->name != key || t->rank != (out == 1 ? 1u : 2u) ||
        t->normalized_extents !=
            std::array<int64_t, 4>{int64_t(in), int64_t(out), 1, 1} ||
        (tags.size() && std::find(tags.begin(), tags.end(), t->storage_type) == tags.end()))
      return Fail(error, "unsupported tensor name, shape or encoding: " + key);
    uint64_t bytes;
    if (!model::EncodedTensorByteSize(t->storage_type, in, out, &bytes) || bytes > uint64_t(INT64_MAX))
      return Fail(error, "unsupported or overflowing encoded tensor extent: " + key);
    uint64_t end;
    if (bytes != t->byte_size || !Sum(t->absolute_offset, bytes, &end))
      return Fail(error, "invalid encoded tensor byte range: " + key);
    auto [existing, inserted] = identities.emplace(key, *t);
    if (!inserted && !SameTensor(existing->second, *t))
      return Fail(error, "inconsistent aliases for tensor: " + key);
    facts.parameters.push_back({role, layer, asset, *t});
    return true;
  }
  // Scalar cases contain facts only. The layer range is a caller precondition;
  // the final arm represents the last valid layer rather than a new fallback.
  void Lookup(const std::string &name,
              const std::vector<std::vector<uint64_t>> &rows) {
    const auto width = rows.front().size();
    std::string types;
    for (size_t j = 0; j < width; ++j)
      types += (j ? ", index" : "index");
    text << "func.def public pure inline @" << name << "(%layer: index) -> ("
         << types << ") where [range(%layer, 0, " << rows.size() - 1
         << ")] {\n";
    for (size_t j = 0; j < width; ++j)
      text << "  %v0_" << j << " = index.constant " << rows.back()[j]
           << " : index\n";
    size_t version = 0;
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
      text << "  %l" << i << " = index.constant " << i << " : index\n"
           << "  %is" << i << " = index.cmp eq, %layer, %l" << i
           << " : index\n  ";
      for (size_t j = 0; j < width; ++j)
        text << (j ? ", " : "") << "%v" << version + 1 << "_" << j;
      text << " = scf.if %is" << i << " -> (" << types << ") {\n";
      for (size_t j = 0; j < width; ++j)
        text << "    %c" << i << "_" << j << " = index.constant " << rows[i][j]
             << " : index\n";
      text << "    scf.yield ";
      for (size_t j = 0; j < width; ++j)
        text << (j ? ", " : "") << "%c" << i << "_" << j;
      text << " : " << types << "\n  } else {\n    scf.yield ";
      for (size_t j = 0; j < width; ++j)
        text << (j ? ", " : "") << "%v" << version << "_" << j;
      text << " : " << types << "\n  }\n";
      ++version;
    }
    text << "  func.return ";
    for (size_t j = 0; j < width; ++j)
      text << (j ? ", " : "") << "%v" << version << "_" << j;
    text << " : " << types << "\n}\n";
  }
};
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

bool EmitQwen(const model::ModelDescriptor &m, const std::string &asset,
              const ConfigurationSettings &o, ConfigurationFacts *out,
              std::string *error) {
  Producer p;
  p.asset = asset;
  p.error = error;
  std::optional<uint32_t> layer;
  auto tensor = [&](const auto &bindings, const std::string &role,
                    const std::string &name, uint64_t in, uint64_t out,
                    std::initializer_list<uint32_t> tags) {
    return p.Tensor(bindings, role, name, in, out, tags, layer);
  };
  if (m.family != "qwen35" || !m.ssm || m.layers.empty() || m.layers.size() > 64)
    return Fail(error, "Qwen configuration requires supported layer identities and SSM facts");
  const uint64_t E = m.embedding_width, V = m.vocabulary_size;
  const uint64_t F = m.layers.front().feed_forward_width, L = m.layers.size();
  const uint64_t QH = m.layers.front().query_heads, KH = m.layers.front().kv_heads;
  const uint64_t G = m.ssm->value_heads, K = m.ssm->key_heads;
  const uint64_t D = m.ssm->state_width, H = m.key_head_width;
  const uint64_t W = m.ssm->inner_width;
  if (!E || !V || !F || !QH || !KH || QH % KH || !H || H != m.value_head_width ||
      !K || !G || G % K || !D || !W || W % D || W / D != G ||
      !m.ssm->convolution_kernel || !PositiveF32(m.rms_epsilon) ||
      !m.rope.width || m.rope.width % 2 || m.rope.width > H ||
      !PositiveF32(m.rope.base) || m.rope.base < 1 || m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear"))
    return Fail(error, "inconsistent Qwen numerical or unsupported rotary facts");
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
  const std::string output_name = m.output_tied_to_embedding ? "token_embd.weight" : "output.weight";
  if (!tensor(m.globals, "token_embedding", "token_embd.weight", E, V, {}) ||
      !tensor(m.globals, "output", output_name, E, V, {}) ||
      !tensor(m.globals, "output_norm", "output_norm.weight", E, 1, {0}))
    return false;
  p.Projection(*Role(m.globals, "output"), E, V);
  uint64_t gdn_count = 0, attention_count = 0;
  std::vector<std::vector<uint64_t>> routes, ffn, attention_projections, gdn_projections;
  for (size_t slot = 0; slot < m.layers.size(); ++slot) {
    const auto &l = m.layers[slot];
    layer = l.index;
    if (l.index != slot || l.feed_forward_width != F || l.query_heads != QH ||
        l.kv_heads != KH)
      return Fail(error, "unsupported Qwen layer identity or geometry");
    const auto prefix = "blk." + std::to_string(l.index) + ".";
    const auto &t = l.tensors;
    if (!tensor(t, "input_norm", prefix + "attn_norm.weight", E, 1, {0}) ||
        !tensor(t, "post_norm", prefix + "post_attention_norm.weight", E, 1,
                {0}) ||
        !tensor(t, "ffn_gate", prefix + "ffn_gate.weight", E, F,
                {}) ||
        !tensor(t, "ffn_up", prefix + "ffn_up.weight", E, F,
                {}) ||
        !tensor(t, "ffn_down", prefix + "ffn_down.weight", F, E,
                {}))
      return false;
    const auto gate = Role(t, "ffn_gate")->storage_type;
    if (gate != Role(t, "ffn_up")->storage_type)
      return Fail(error, "Qwen whole FFN requires matching gate/up encodings");
    const auto &gate_tensor = *Role(t, "ffn_gate");
    const auto &up_tensor = *Role(t, "ffn_up");
    const auto &down_tensor = *Role(t, "ffn_down");
    ffn.push_back({F, gate, up_tensor.storage_type, down_tensor.storage_type,
                   gate_tensor.byte_size, up_tensor.byte_size,
                   down_tensor.byte_size});
    p.Projection(gate_tensor, E, F);
    p.Projection(up_tensor, E, F);
    p.Projection(down_tensor, F, E);
    if (l.kind == model::LayerKind::kGatedDeltaNet) {
      if (!l.recurrent_ordinal || *l.recurrent_ordinal != gdn_count ||
          l.attention_ordinal)
        return Fail(error,
                    "Qwen recurrent ordinal disagrees with descriptor order");
      routes.push_back({1, l.index, gdn_count++});
      if (!tensor(t, "recurrent_qkv", prefix + "attn_qkv.weight", E, QKV, {}) ||
          !tensor(t, "recurrent_gate", prefix + "attn_gate.weight", E, W, {}) ||
          !tensor(t, "convolution", prefix + "ssm_conv1d.weight", m.ssm->convolution_kernel, QKV,
                  {0}) ||
          !tensor(t, "dt_bias", prefix + "ssm_dt.bias", G, 1, {0}) ||
          !tensor(t, "decay", prefix + "ssm_a", G, 1, {0}) ||
          !tensor(t, "alpha", prefix + "ssm_alpha.weight", E, G, {}) ||
          !tensor(t, "beta", prefix + "ssm_beta.weight", E, G, {}) ||
          !tensor(t, "recurrent_norm", prefix + "ssm_norm.weight", D, 1,
                  {0}) ||
          !tensor(t, "recurrent_output", prefix + "ssm_out.weight", W, E, {}))
        return false;
      for (const auto &use : std::vector<std::tuple<const char *, uint64_t, uint64_t>>{
               {"recurrent_qkv", E, QKV}, {"recurrent_gate", E, W},
               {"alpha", E, G}, {"beta", E, G}, {"recurrent_output", W, E}})
        p.Projection(*Role(t, std::get<0>(use)), std::get<1>(use), std::get<2>(use));
      std::vector<uint64_t> row;
      for (const char *role : {"recurrent_qkv", "recurrent_gate", "alpha", "beta", "recurrent_output"}) row.push_back(Role(t, role)->storage_type);
      for (const char *role : {"recurrent_qkv", "recurrent_gate", "alpha", "beta", "recurrent_output"}) row.push_back(Role(t, role)->byte_size);
      gdn_projections.push_back(std::move(row));
    } else if (l.kind == model::LayerKind::kAttention) {
      if (!l.attention_ordinal || *l.attention_ordinal != attention_count ||
          l.recurrent_ordinal)
        return Fail(error,
                    "Qwen attention ordinal disagrees with descriptor order");
      routes.push_back({0, l.index, attention_count++});
      if (!tensor(t, "query", prefix + "attn_q.weight", E, query_gate_width, {}) ||
          !tensor(t, "key", prefix + "attn_k.weight", E, kv_width, {}) ||
          !tensor(t, "value", prefix + "attn_v.weight", E, kv_width, {}) ||
          !tensor(t, "attention_output", prefix + "attn_output.weight", attention_width, E, {}) ||
          !tensor(t, "query_norm", prefix + "attn_q_norm.weight", H, 1,
                  {0}) ||
          !tensor(t, "key_norm", prefix + "attn_k_norm.weight", H, 1, {0}))
        return false;
      for (const auto &use : std::vector<std::tuple<const char *, uint64_t, uint64_t>>{
               {"query", E, query_gate_width}, {"key", E, kv_width},
               {"value", E, kv_width}, {"attention_output", attention_width, E}})
        p.Projection(*Role(t, std::get<0>(use)), std::get<1>(use), std::get<2>(use));
      std::vector<uint64_t> row;
      for (const char *role : {"query", "key", "value", "attention_output"}) row.push_back(Role(t, role)->storage_type);
      for (const char *role : {"query", "key", "value", "attention_output"}) row.push_back(Role(t, role)->byte_size);
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
  uint64_t plane_token_bytes, all_token_bytes, capacity_bytes, private_stride, arena_bytes;
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
  p.Index("qwen.model.checkpoint_layers", m.layers.size());
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
  p.Index("qwen.model.embedding_storage_type", Role(m.globals, "token_embedding")->storage_type);
  p.Index("qwen.model.output_storage_type", Role(m.globals, "output")->storage_type);
  p.Index("qwen.model.tie_word_embeddings", m.output_tied_to_embedding);
  p.Index("qwen.model.embedding_weight_bytes", Role(m.globals, "token_embedding")->byte_size);
  p.Index("qwen.model.output_weight_bytes", Role(m.globals, "output")->byte_size);
  p.Index("qwen.model.embedding_row_bytes", Role(m.globals, "token_embedding")->byte_size / V);
  p.Index("qwen.model.output_weight_row_bytes", Role(m.globals, "output")->byte_size / V);
  p.Index("qwen.model.projection_implementation", 1);
  p.Index("qwen.model.hidden_capacity_bytes", 512 * m.embedding_width * 4);
  p.Index("qwen.model.ffn_capacity_bytes", 512 * F * 4);
  p.Index("qwen.model.gdn_f16_capacity_bytes", 512 * W * 2);
  p.Index("qwen.model.gdn_gate_capacity_bytes", 512 * G * 4);
  p.Index("qwen.model.hidden_q8_capacity_bytes", 512 * E / 128 * 144);
  p.Index("qwen.model.kv_capacity_bytes", 512 * kv_width * 4);
  p.Index("qwen.model.projection_capacity_bytes", 512 * std::max(W, attention_width) * 4);
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
} // namespace
bool DeriveConfiguration(const model::ModelDescriptor &m,
                         const std::string &asset,
                         const ConfigurationSettings &s,
                         ConfigurationFacts *out, std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto limit = m.family == "llama" ? 131072u : 262144u;
  if (!s.token_capacity || s.token_capacity > 512 ||
      s.context_capacity < s.token_capacity || s.context_capacity > limit)
    return Fail(error, "unsupported configuration token/context capacities");
  if (m.family == "llama")
    return EmitLlama(m, asset, s, out, error);
  if (m.family == "qwen35")
    return EmitQwen(m, asset, s, out, error);
  return Fail(error, "unsupported model family");
}
} // namespace hrx::loomify_gguf
