// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model-config/configuration.h"

#include "loomc/iree.h"
#include "loomc/source.h"
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
loomc_status_t Error(const std::string &message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str()));
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
  LinkedConfigurationFacts facts;
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
bool Layout(const LinkedConfigurationSettings &s, size_t layers,
            LlamaLinkedLayout *out, std::string *error) {
  if (!s.context_capacity || s.context_capacity < s.token_capacity ||
      s.context_capacity > 131072)
    return Fail(error, "Llama context capacity must satisfy T <= C <= 131072");
  LlamaLinkedLayout g;
  g.token_capacity = s.token_capacity;
  g.context_capacity = s.context_capacity;
  g.pool_capacity = (s.context_capacity + 63) / 64 * 64;
  if (s.pool_capacity && *s.pool_capacity != g.pool_capacity)
    return Fail(error, "Llama physical pool must equal round_up(context,64)");
  g.layer_count = layers;
  g.history_capacity = std::max<uint64_t>(s.token_capacity, 129);
  g.cache_base = 768 + 4 * g.history_capacity;
  uint64_t residual, arena, planes, stride;
  if (!Product(4 * 3072, s.token_capacity, &residual) ||
      !Product(2, layers, &planes) ||
      !Product(2 * 1024, g.pool_capacity, &stride) ||
      !Product(planes, stride, &arena) || !Sum(arena, g.cache_base, &arena))
    return Fail(error, "Llama layout exceeds native signed geometry bounds");
  const auto table = 2048 + 4 * ((s.context_capacity + 63) / 64);
  g.prefill = {residual, 12, table, arena, 4 * g.history_capacity, 32};
  g.decode = {4 * 3072, 12, table, arena, 4 * 129, 32};
  g.epoch_minima = {0, residual, 2064, table, arena, 4 * s.token_capacity, 64};
  g.cold_allocations = {{{residual, 256, residual},
                         {arena, 256, g.cache_base},
                         {2064, 256, 0},
                         {table, 256, 0},
                         {4 * g.history_capacity, 256, 0},
                         {64, 256, 0},
                         {0, 256, 0},
                         {0, 256, 0},
                         {0, 256, 0},
                         {0, 256, 0},
                         {0, 256, 0}}};
  g.row_views = {
      0, 12, 512, 0, 512, 4 * g.history_capacity, 512 + 4 * g.history_capacity,
      32};
  g.origins = {256, 256, g.cache_base, 0};
  g.pooled_geometry = {64,           2048,   0,         64, 256,
                       g.cache_base, 4,      1,         1,  g.cache_base,
                       planes,       stride, 128 * 1024};
  *out = g;
  return true;
}
} // namespace

bool DeriveLinkedConfiguration(const model::ModelDescriptor &m,
                               const std::string &asset,
                               const LinkedConfigurationSettings &s,
                               LinkedConfigurationFacts *out,
                               std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const bool llama = s.profile == LinkedProfile::Llama3BModel;
  if (!llama && s.profile != LinkedProfile::Qwen08BFfn)
    return Fail(error, "unknown linked configuration profile");
  if (!s.token_capacity || s.token_capacity > 512)
    return Fail(error, "token capacity must be between 1 and 512");
  const uint64_t E = llama ? 3072 : 1024, F = llama ? 8192 : 3584;
  if (m.family != (llama ? "llama" : "qwen35") || m.embedding_width != E ||
      !std::isfinite(m.rms_epsilon) ||
      float(m.rms_epsilon) != (llama ? 1e-5f : 1e-6f) || m.layers.empty())
    return Fail(error, "unsupported linked configuration model profile");
  if (llama &&
      (m.vocabulary_size != 128256 || m.key_head_width != 128 ||
       m.value_head_width != 128 || m.ssm || m.rope.width != 128 ||
       m.rope.base != 500000 || m.rope.scaling_factor != 1 ||
       (m.rope.scaling_kind != "none" && m.rope.scaling_kind != "linear") ||
       !m.output_tied_to_embedding))
    return Fail(error, "unsupported Llama head, rotary or tied-output profile");
  Producer p;
  p.asset = asset;
  p.error = error;
  p.facts.profile = s.profile;
  if (llama) {
    LlamaLinkedLayout layout;
    if (!Layout(s, m.layers.size(), &layout, error))
      return false;
    p.facts.llama_layout = layout;
    if (!p.Tensor(m.globals, "token_embedding", "token_embd.weight", E, 128256,
                  {14}) ||
        !p.Tensor(m.globals, "output", "token_embd.weight", E, 128256, {14}) ||
        !p.Tensor(m.globals, "output_norm", "output_norm.weight", E, 1, {0}) ||
        !p.Tensor(m.globals, "rope_frequencies", "rope_freqs.weight", 64, 1,
                  {0}))
      return false;
  }
  std::vector<std::vector<uint64_t>> ffn, attention;
  // Order is architecture order; require that it is also the native layer key.
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &l = m.layers[i];
    if (l.index != i || l.feed_forward_width != F ||
        (llama && (l.kind != model::LayerKind::kAttention ||
                   l.query_heads != 24 || l.kv_heads != 8)))
      return Fail(error, "unsupported layer identity or geometry at layer " +
                             std::to_string(i));
    const auto prefix = "blk." + std::to_string(i) + ".";
    if (!p.Tensor(
            l.tensors, "post_norm",
            prefix + (llama ? "ffn_norm.weight" : "post_attention_norm.weight"),
            E, 1, {0}, l.index) ||
        !p.Tensor(l.tensors, "ffn_gate", prefix + "ffn_gate.weight", E, F,
                  llama ? std::initializer_list<uint32_t>{12}
                        : std::initializer_list<uint32_t>{12, 13, 14},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_up", prefix + "ffn_up.weight", E, F,
                  llama ? std::initializer_list<uint32_t>{12}
                        : std::initializer_list<uint32_t>{12, 13, 14},
                  l.index) ||
        !p.Tensor(l.tensors, "ffn_down", prefix + "ffn_down.weight", F, E,
                  llama ? std::initializer_list<uint32_t>{12, 14}
                        : std::initializer_list<uint32_t>{13, 14},
                  l.index))
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
    if (llama) {
      if (!p.Tensor(l.tensors, "input_norm", prefix + "attn_norm.weight", E, 1,
                    {0}, l.index) ||
          !p.Tensor(l.tensors, "query", prefix + "attn_q.weight", E, E,
                    {12, 14}, l.index) ||
          !p.Tensor(l.tensors, "key", prefix + "attn_k.weight", E, 1024,
                    {12, 14}, l.index) ||
          !p.Tensor(l.tensors, "value", prefix + "attn_v.weight", E, 1024,
                    {12, 14}, l.index) ||
          !p.Tensor(l.tensors, "attention_output",
                    prefix + "attn_output.weight", E, E, {12, 14}, l.index))
        return false;
      const auto &q = *Role(l.tensors, "query"), &k = *Role(l.tensors, "key"),
                 &v = *Role(l.tensors, "value"),
                 &o = *Role(l.tensors, "attention_output");
      attention.push_back({q.storage_type, k.storage_type, v.storage_type,
                           o.storage_type, q.byte_size, k.byte_size,
                           v.byte_size, o.byte_size});
    }
  }
  p.Index("loomify.ffn.profile", llama ? 0 : 1);
  if (llama) {
    const auto &g = *p.facts.llama_layout;
    p.Index("llama.model.hidden_size", E);
    p.Index("llama.model.vocabulary_size", m.vocabulary_size);
    p.Index("llama.model.layer_count", m.layers.size());
    p.Index("llama.model.query_heads", 24);
    p.Index("llama.model.kv_heads", 8);
    p.Index("llama.model.head_size", 128);
    p.Index("llama.model.kv_size", 1024);
    p.Real("llama.model.rms_epsilon", float(m.rms_epsilon));
    p.Index("llama.attention.cache_capacity", g.context_capacity);
    p.Index("llama.attention.pool_capacity", g.pool_capacity);
    p.Index("runner.llama32.prefill_token_count", s.token_capacity);
    p.Index("llama.attention.cache_base", g.cache_base, "offset");
    p.Index("llama.attention.map_origin", g.map_origin, "offset");
    p.Index("llama.model.embedding_weight_bytes",
            Role(m.globals, "token_embedding")->byte_size);
    p.Index("llama.model.output_weight_bytes",
            Role(m.globals, "output")->byte_size);
    p.Index("qwen38.greedy_argmax.output_capacity", m.vocabulary_size);
  } else {
    p.Index("qwen.model.hidden_size", E);
    p.Index("qwen.model.ffn_size", F);
    p.Real("qwen.model.hidden_divisor", float(E));
    p.Real("qwen.model.rms_epsilon", float(m.rms_epsilon));
    p.Index("qwen.model.hidden_row_bytes", 4 * E);
    p.Index("qwen.model.ffn_row_bytes", 4 * F);
    p.Index("qwen.model.hidden_q8_row_bytes", E / 128 * 144);
    p.Index("qwen.model.hidden_q8_depth4_bytes", 4 * E / 128 * 144);
    p.Index("qwen.model.ffn_q8_row_bytes", F / 128 * 144);
    p.Index("qwen.model.hidden_capacity_bytes", 512 * E * 4);
    p.Index("qwen.model.ffn_capacity_bytes", 512 * F * 4);
    p.Index("qwen.model.hidden_q8_capacity_bytes", 512 * E / 128 * 144);
  }
  for (const auto tag :
       (llama ? std::vector<int>{4, 6} : std::vector<int>{4, 5, 6})) {
    const auto prefix = "ggml.linear_q" + std::to_string(tag) + "k_q8_1_x4.";
    p.Index(prefix + "token_capacity", 512);
    p.Index(prefix + "output_capacity", llama && tag == 6 ? 128256 : F);
  }
  p.Index("ggml.quantize_q8_1_x4.group_capacity", 32768);
  p.Lookup("loomify.config.layer_ffn", ffn);
  if (llama)
    p.Lookup("loomify.config.layer_attention", attention);
  p.facts.source = p.text.str();
  *out = std::move(p.facts);
  return true;
}

loomc_status_t CreateLinkedConfiguration(
    loomc_context_t *context, loomc_workspace_t *workspace,
    const model::ModelDescriptor &descriptor, const std::string &asset,
    const LinkedConfigurationSettings &settings, LinkedConfiguration *out,
    loomc_result_t **result) {
  if (!context || !workspace || !out || !result)
    return Error("configuration requires context, workspace and outputs");
  LinkedConfiguration candidate;
  std::string error;
  if (!DeriveLinkedConfiguration(descriptor, asset, settings, &candidate.facts,
                                 &error))
    return Error(error);
  loomc_source_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  options.identifier =
      loomc_make_cstring_view("linked-model-configuration.loom");
  const auto &text = candidate.facts.source;
  options.contents = loomc_make_byte_span(text.data(), text.size());
  loomc_source_t *source = nullptr;
  auto status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  if (!loomc_status_is_ok(status))
    return status;
  loomc_module_t *module = nullptr;
  loomc_result_t *diagnostics = nullptr;
  status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &diagnostics);
  loomc_source_release(source);
  candidate.module.reset(module);
  // Diagnostics are an explicit result even when parsing fails. Product output
  // remains untouched until the complete native candidate is valid.
  *result = diagnostics;
  if (loomc_status_is_ok(status) && diagnostics &&
      loomc_result_succeeded(diagnostics))
    *out = std::move(candidate);
  return status;
}
} // namespace hrx::loomify_gguf
