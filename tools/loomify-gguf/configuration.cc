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
struct Producer {
  ConfigurationFacts facts;
  std::map<std::string, model::TensorRecord> identities;
  std::ostringstream text;
  std::string asset;
  std::string *error;
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
        std::find(tags.begin(), tags.end(), t->storage_type) == tags.end())
      return Fail(error, "unsupported tensor name, shape or encoding: " + key);
    uint64_t bytes;
    if (t->storage_type == 0) {
      if (!Product(in, out, &bytes) || !Product(bytes, 4, &bytes))
        return Fail(error, "tensor extent overflow: " + key);
    } else {
      const uint64_t block = t->storage_type == 12   ? 144
                             : t->storage_type == 13 ? 176
                                                     : 210;
      if (in % 256 || !Product(in / 256, out, &bytes) ||
          !Product(bytes, block, &bytes))
        return Fail(error, "encoded tensor extent overflow: " + key);
    }
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
  const uint64_t E = 3072, F = 8192;
  if (m.family != "llama" || m.embedding_width != E ||
      !std::isfinite(m.rms_epsilon) || float(m.rms_epsilon) != 1e-5f ||
      m.layers.size() != 28)
    return Fail(error, "unsupported linked configuration model profile");
  if (m.vocabulary_size != 128256 || m.key_head_width != 128 ||
      m.value_head_width != 128 || m.ssm || m.rope.width != 128 ||
      m.rope.base != 500000 || m.rope.scaling_factor != 1 ||
      (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
      !m.output_tied_to_embedding)
    return Fail(error, "unsupported Llama head, rotary or tied-output profile");
  Producer p;
  p.asset = asset;
  p.error = error;
  if (!p.Tensor(m.globals, "token_embedding", "token_embd.weight", E, 128256,
                {14}) ||
      !p.Tensor(m.globals, "output", "token_embd.weight", E, 128256, {14}) ||
      !p.Tensor(m.globals, "output_norm", "output_norm.weight", E, 1, {0}) ||
      !p.Tensor(m.globals, "rope_frequencies", "rope_freqs.weight", 64, 1, {0}))
    return false;

  std::vector<std::vector<uint64_t>> ffn, attention;
  // Order is architecture order; require that it is also the native layer key.
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    if (l.index != i || l.feed_forward_width != F ||
        (l.kind != model::LayerKind::kAttention || l.query_heads != 24 ||
         l.kv_heads != 8))
      return Fail(error, "unsupported layer identity or geometry at layer " +
                             std::to_string(i));
    const auto prefix = "blk." + std::to_string(i) + ".";
    if (!p.Tensor(l.tensors, "post_norm", prefix + "ffn_norm.weight", E, 1, {0},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_gate", prefix + "ffn_gate.weight", E, F, {12},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_up", prefix + "ffn_up.weight", E, F, {12},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_down", prefix + "ffn_down.weight", F, E,
                  {12, 14}, l.index))
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
    if (!p.Tensor(l.tensors, "input_norm", prefix + "attn_norm.weight", E, 1,
                  {0}, l.index) ||
        !p.Tensor(l.tensors, "query", prefix + "attn_q.weight", E, E, {12, 14},
                  l.index) ||
        !p.Tensor(l.tensors, "key", prefix + "attn_k.weight", E, 1024, {12, 14},
                  l.index) ||
        !p.Tensor(l.tensors, "value", prefix + "attn_v.weight", E, 1024,
                  {12, 14}, l.index) ||
        !p.Tensor(l.tensors, "attention_output", prefix + "attn_output.weight",
                  E, E, {12, 14}, l.index))
      return false;
    const auto &q = *Role(l.tensors, "query"), &k = *Role(l.tensors, "key"),
               &v = *Role(l.tensors, "value"),
               &o = *Role(l.tensors, "attention_output");
    attention.push_back({q.storage_type, k.storage_type, v.storage_type,
                         o.storage_type, q.byte_size, k.byte_size, v.byte_size,
                         o.byte_size});
  }
  p.Index("loomify.ffn.profile", 0);
  const auto history = std::max<uint64_t>(s.token_capacity, 129);
  const auto pool = (s.context_capacity + 63) / 64 * 64;
  p.Index("llama.model.history_capacity", history);
  p.Index("llama.model.hidden_size", E);
  p.Index("llama.model.vocabulary_size", m.vocabulary_size);
  p.Index("llama.model.layer_count", m.layers.size());
  p.Index("llama.model.query_heads", 24);
  p.Index("llama.model.kv_heads", 8);
  p.Index("llama.model.head_size", 128);
  p.Index("llama.model.kv_size", 1024);
  p.Real("llama.model.rms_epsilon", float(m.rms_epsilon));
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

  for (const auto tag : std::vector<int>{4, 6}) {
    const auto prefix = "ggml.linear_q" + std::to_string(tag) + "k_q8_1_x4.";
    p.Index(prefix + "token_capacity", 512);
    p.Index(prefix + "output_capacity", tag == 6 ? 128256 : F);
  }
  p.Index("ggml.quantize_q8_1_x4.group_capacity", 32768);
  p.Lookup("loomify.config.layer_ffn", ffn);
  p.Lookup("loomify.config.layer_attention", attention);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  if (error)
    error->clear();
  return true;
}

struct Selectors {
  uint32_t gate_q4 = 0, gate_q6 = 0, down_q6 = 0, attention_q6 = 0;
};
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
  if (m.family != "qwen35" || m.embedding_width != 1024 ||
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
  if (!tensor(m.globals, "token_embedding", "token_embd.weight", 1024, 248320,
              {14}) ||
      !tensor(m.globals, "output", "token_embd.weight", 1024, 248320, {14}) ||
      !tensor(m.globals, "output_norm", "output_norm.weight", 1024, 1, {0}))
    return false;
  Selectors selected;
  uint64_t gdn_count = 0, attention_count = 0;
  std::vector<std::vector<uint64_t>> routes, ffn;
  for (size_t slot = 0; slot < m.layers.size(); ++slot) {
    const auto &l = m.layers[slot];
    layer = l.index;
    if (l.index != slot || l.feed_forward_width != 3584 || l.query_heads != 8 ||
        l.kv_heads != 2)
      return Fail(error, "unsupported Qwen layer identity or geometry");
    const auto bit = uint32_t(1) << l.index;
    const auto prefix = "blk." + std::to_string(l.index) + ".";
    const auto &t = l.tensors;
    if (!tensor(t, "input_norm", prefix + "attn_norm.weight", 1024, 1, {0}) ||
        !tensor(t, "post_norm", prefix + "post_attention_norm.weight", 1024, 1,
                {0}) ||
        !tensor(t, "ffn_gate", prefix + "ffn_gate.weight", 1024, 3584,
                {12, 13, 14}) ||
        !tensor(t, "ffn_up", prefix + "ffn_up.weight", 1024, 3584,
                {12, 13, 14}) ||
        !tensor(t, "ffn_down", prefix + "ffn_down.weight", 3584, 1024,
                {13, 14}))
      return false;
    const auto gate = Role(t, "ffn_gate")->storage_type;
    if (gate != Role(t, "ffn_up")->storage_type)
      return Fail(error, "Qwen whole FFN requires matching gate/up encodings");
    const auto &gate_tensor = *Role(t, "ffn_gate");
    const auto &up_tensor = *Role(t, "ffn_up");
    const auto &down_tensor = *Role(t, "ffn_down");
    ffn.push_back({3584, gate, up_tensor.storage_type, down_tensor.storage_type,
                   gate_tensor.byte_size, up_tensor.byte_size,
                   down_tensor.byte_size});
    if (gate == 12)
      selected.gate_q4 |= bit;
    if (gate == 14)
      selected.gate_q6 |= bit;
    if (Role(t, "ffn_down")->storage_type == 14)
      selected.down_q6 |= bit;
    if (l.kind == model::LayerKind::kGatedDeltaNet) {
      if (!l.recurrent_ordinal || *l.recurrent_ordinal != gdn_count ||
          l.attention_ordinal)
        return Fail(error,
                    "Qwen recurrent ordinal disagrees with descriptor order");
      routes.push_back({1, l.index, gdn_count++});
      if (!tensor(t, "recurrent_qkv", prefix + "attn_qkv.weight", 1024, 6144,
                  {14}) ||
          !tensor(t, "recurrent_gate", prefix + "attn_gate.weight", 1024, 2048,
                  {14}) ||
          !tensor(t, "convolution", prefix + "ssm_conv1d.weight", 4, 6144,
                  {0}) ||
          !tensor(t, "dt_bias", prefix + "ssm_dt.bias", 16, 1, {0}) ||
          !tensor(t, "decay", prefix + "ssm_a", 16, 1, {0}) ||
          !tensor(t, "alpha", prefix + "ssm_alpha.weight", 1024, 16, {13}) ||
          !tensor(t, "beta", prefix + "ssm_beta.weight", 1024, 16, {13}) ||
          !tensor(t, "recurrent_norm", prefix + "ssm_norm.weight", 128, 1,
                  {0}) ||
          !tensor(t, "recurrent_output", prefix + "ssm_out.weight", 2048, 1024,
                  {13}))
        return false;
    } else if (l.kind == model::LayerKind::kAttention) {
      if (!l.attention_ordinal || *l.attention_ordinal != attention_count ||
          l.recurrent_ordinal)
        return Fail(error,
                    "Qwen attention ordinal disagrees with descriptor order");
      routes.push_back({0, l.index, attention_count++});
      if (!tensor(t, "query", prefix + "attn_q.weight", 1024, 4096, {13, 14}) ||
          !tensor(t, "key", prefix + "attn_k.weight", 1024, 512, {13, 14}) ||
          !tensor(t, "value", prefix + "attn_v.weight", 1024, 512, {14}) ||
          !tensor(t, "attention_output", prefix + "attn_output.weight", 2048,
                  1024, {13}) ||
          !tensor(t, "query_norm", prefix + "attn_q_norm.weight", 256, 1,
                  {0}) ||
          !tensor(t, "key_norm", prefix + "attn_k_norm.weight", 256, 1, {0}))
        return false;
      const auto query = Role(t, "query")->storage_type;
      if (query != Role(t, "key")->storage_type)
        return Fail(error,
                    "Qwen attention requires matching query/key encodings");
      if (query == 14)
        selected.attention_q6 |= bit;
    } else
      return Fail(error, "unsupported Qwen layer kind");
  }
  if (gdn_count != 18 || attention_count != 6)
    return Fail(error, "unsupported Qwen recurrent/attention layer counts");
  const auto pool = (o.context_capacity + 63) / 64 * 64;
  const auto history_bytes = 4 * m.ssm->convolution_history_elements;
  const auto recurrent_bytes = 4 * m.ssm->recurrent_state_elements;
  const auto layer_bytes = history_bytes + recurrent_bytes;
  const auto state_bytes = gdn_count * layer_bytes;
  const auto plane_count = 2 * attention_count;
  const auto plane_token_bytes =
      2 * m.layers.front().kv_heads * m.key_head_width;
  auto mask = [&](const char *key, uint32_t value) {
    p.text << "config.def @" << key << " = " << int64_t(int32_t(value))
           << " : i32\n";
  };
  const auto E = m.embedding_width, F = uint64_t(3584), V = m.vocabulary_size;
  const auto H = m.key_head_width, D = m.ssm->state_width, K = m.ssm->key_heads,
             G = m.ssm->value_heads, W = m.ssm->inner_width,
             QKV = 2 * K * D + W;
  p.Index("qwen.model.hidden_size", E);
  p.Index("qwen.model.ffn_size", F);
  p.Index("qwen.model.vocabulary_size", V);
  p.Index("qwen.model.base_layers", m.layers.size());
  p.Index("qwen.model.checkpoint_layers", m.layers.size());
  p.Index("qwen.model.group_count", attention_count);
  p.Index("qwen.model.attention_heads", 8);
  p.Index("qwen.model.kv_heads", 2);
  p.Index("qwen.model.head_size", H);
  p.Index("qwen.model.attention_width", 8 * H);
  p.Index("qwen.model.kv_size", 2 * H);
  p.Index("qwen.model.query_gate_size", 16 * H);
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
  p.Index("qwen.model.query_gate_row_bytes", 16 * H * 4);
  p.Index("qwen.model.kv_plane_token_bytes", plane_token_bytes);
  p.Index("qwen.model.kv_row_bytes", plane_token_bytes * 2);
  p.Index("qwen.model.target_kv_planes", plane_count);
  p.Index("qwen.model.target_kv_token_bytes", plane_count * plane_token_bytes);
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
  p.Index("qwen.model.embedding_quantization", 6);
  p.Index("qwen.model.tie_word_embeddings", 1);
  p.Index("qwen.model.embedding_weight_bytes", E / 256 * V * 210);
  p.Index("qwen.model.output_weight_bytes", E / 256 * V * 210);
  p.Index("qwen.model.embedding_row_bytes", E / 256 * 210);
  p.Index("qwen.model.output_weight_row_bytes", E / 256 * 210);
  p.Index("qwen.model.ffn_gate_q4_weight_bytes", E / 256 * F * 144);
  p.Index("qwen.model.ffn_gate_q5_weight_bytes", E / 256 * F * 176);
  p.Index("qwen.model.ffn_gate_q6_weight_bytes", E / 256 * F * 210);
  p.Index("qwen.model.ffn_down_q5_weight_bytes", F / 256 * E * 176);
  p.Index("qwen.model.ffn_down_q6_weight_bytes", F / 256 * E * 210);
  p.Index("qwen.model.ffn_gate_weight_bytes", E / 256 * F * 176);
  p.Index("qwen.model.ffn_down_weight_bytes", F / 256 * E * 210);
  p.Index("qwen.model.query_q5_weight_bytes", E / 256 * 16 * H * 176);
  p.Index("qwen.model.query_q6_weight_bytes", E / 256 * 16 * H * 210);
  p.Index("qwen.model.key_q5_weight_bytes", E / 256 * 2 * H * 176);
  p.Index("qwen.model.key_q6_weight_bytes", E / 256 * 2 * H * 210);
  p.Index("qwen.model.query_weight_bytes", E / 256 * 16 * H * 210);
  p.Index("qwen.model.key_weight_bytes", E / 256 * 2 * H * 210);
  p.Index("qwen.model.value_weight_bytes", E / 256 * 2 * H * 210);
  p.Index("qwen.model.projection_weight_bytes", W / 256 * E * 176);
  p.Index("qwen.model.gdn_qkv_weight_bytes", E / 256 * QKV * 210);
  p.Index("qwen.model.gdn_gate_weight_bytes", E / 256 * W * 210);
  p.Index("qwen.model.gdn_alpha_weight_bytes", E / 256 * G * 176);
  p.Index("qwen.model.gdn_beta_weight_bytes", E / 256 * G * 176);
  mask("qwen.model.ffn_gate_q4_layers", selected.gate_q4);
  mask("qwen.model.ffn_gate_q6_layers", selected.gate_q6);
  mask("qwen.model.ffn_down_q6_layers", selected.down_q6);
  mask("qwen.model.attention_q6_layers", selected.attention_q6);
  p.Index("qwen.model.projection_implementation", 1);
  p.Index("qwen.model.hidden_capacity_bytes", 512 * m.embedding_width * 4);
  p.Index("qwen.model.ffn_capacity_bytes", 512 * F * 4);
  p.Index("qwen.model.gdn_f16_capacity_bytes", 512 * W * 2);
  p.Index("qwen.model.gdn_gate_capacity_bytes", 512 * G * 4);
  p.Index("qwen.model.hidden_q8_capacity_bytes", 512 * E / 128 * 144);
  p.Index("qwen.model.kv_capacity_bytes", 512 * 2 * H * 4);
  p.Index("qwen.model.projection_capacity_bytes", 512 * W * 4);
  p.Index("qwen.model.qkv_capacity_bytes", 512 * QKV * 4);
  p.Index("qwen.model.query_gate_capacity_bytes", 512 * 16 * H * 4);
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
  p.Index("ggml.linear_q4k_q8_1_x4.output_capacity", F);
  p.Index("ggml.linear_q5k_q8_1_x4.output_capacity", QKV);
  p.Index("ggml.linear_q6k_q8_1_x4.output_capacity", V);
  p.Index("ggml.linear_q4k_q8_1_x4.token_capacity", 512);
  p.Index("ggml.linear_q5k_q8_1_x4.token_capacity", 512);
  p.Index("ggml.linear_q6k_q8_1_x4.token_capacity", 512);
  p.Index("ggml.quantize_q8_1_x4.group_capacity", 32768);
  p.Index("qwen.model.gdn_slot_origin", 256);
  p.Index("qwen.model.private_row_stride", 256 + state_bytes + 2048 + 256);
  p.Index("qwen.model.map_origin", 2048);
  p.Index("loomify.ffn.profile", 1);
  p.Lookup("loomify.config.layer_route", routes);
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
