// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/qwen/structure.h"
#include "lib/operations/qwen_operations.h"
#include "lib/operations/tensor_operations.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
namespace hrx::qwen {
using namespace operations;
namespace {
bool Fail(std::string *e, const std::string &message) {
  if (e)
    *e = message;
  return false;
}
bool Multiply(uint64_t a, uint64_t b, uint64_t *out) {
  if (!b || a > uint64_t(INT64_MAX) / b)
    return false;
  *out = a * b;
  return true;
}
} // namespace
QwenStructure::QwenStructure(model::ModelDescriptor facts,
                             QwenBuildOptions options,
                             const OperationRegistry &registry)
    : facts_(std::move(facts)), options_(std::move(options)), graph_(registry) {
}
const model::TensorBinding *
QwenStructure::parameter(ParameterCoordinate c) const {
  if (c.global)
    return c.layer_slot == 0 && c.tensor_slot < facts_.globals.size()
               ? &facts_.globals[c.tensor_slot]
               : nullptr;
  if (c.layer_slot >= facts_.layers.size() ||
      c.tensor_slot >= facts_.layers[c.layer_slot].tensors.size())
    return nullptr;
  return &facts_.layers[c.layer_slot].tensors[c.tensor_slot];
}
bool BuildQwenStructure(model::ModelDescriptor facts, const QwenBuildOptions &o,
                        std::shared_ptr<const QwenStructure> *out,
                        std::string *error) {
  if (!out || !error || facts.family != "qwen35" || !facts.ssm ||
      facts.layers.empty() || !facts.embedding_width ||
      !facts.vocabulary_size || !facts.key_head_width ||
      facts.key_head_width != facts.value_head_width ||
      !std::isfinite(facts.rms_epsilon) || facts.rms_epsilon <= 0 ||
      !o.token_capacity || o.token_capacity > 512 || !o.span_capacity ||
      o.span_capacity > 64 || !o.selected_output_capacity ||
      o.selected_output_capacity > o.span_capacity || !o.cache_capacity ||
      o.cache_capacity > uint64_t(INT64_MAX) || o.source_asset.empty())
    return Fail(error, "invalid complete Qwen facts/options/output");
  const auto ssm = *facts.ssm;
  if (!ssm.state_width || !ssm.key_heads || !ssm.value_heads ||
      ssm.value_heads % ssm.key_heads || ssm.convolution_kernel < 2 ||
      ssm.convolution_kernel > uint64_t(INT64_MAX) ||
      ssm.state_width > uint64_t(INT64_MAX) / ssm.value_heads ||
      ssm.inner_width != ssm.state_width * ssm.value_heads)
    return Fail(error, "inconsistent Qwen SSM geometry");
  uint64_t key_width, channels, history;
  if (!Multiply(ssm.key_heads, ssm.state_width, &key_width) ||
      key_width > (uint64_t(INT64_MAX) - ssm.inner_width) / 2 ||
      !Multiply(2 * key_width + ssm.inner_width, ssm.convolution_kernel - 1,
                &history))
    return Fail(error, "Qwen SSM geometry overflow");
  channels = 2 * key_width + ssm.inner_width;
  uint64_t recurrent;
  if (!Multiply(ssm.state_width, ssm.inner_width, &recurrent) ||
      history != ssm.convolution_history_elements ||
      recurrent != ssm.recurrent_state_elements)
    return Fail(error, "inconsistent Qwen retained state extents");
  OperationRegistry registry;
  if (!DefineEmbedding(&registry, error) ||
      !DefineQwenGdnLayer(&registry, error) ||
      !DefineQwenAttentionLayer(&registry, error) ||
      !DefineQwenEpochHead(&registry, error))
    return false;
  auto candidate = std::shared_ptr<QwenStructure>(
      new QwenStructure(std::move(facts), o, registry));
  auto &g = candidate->graph_;
  const auto &f = candidate->facts_;
  auto &in = candidate->inputs_;
  auto tensor = [&](TensorType t) {
    uint64_t bytes = t.elementType() == ElementType::F16  ? 2
                     : t.elementType() == ElementType::I8 ? 1
                                                          : 4;
    for (size_t i = 0; i < t.rank(); ++i)
      if (!Multiply(bytes, t.dimension(i), &bytes)) {
        Fail(error, "Qwen logical tensor byte extent overflow");
        return ValueId{};
      }
    return g.tensor(t, error);
  };
  in.tokens = tensor(TensorType::vector(o.token_capacity, ElementType::I32));
  in.residual = tensor(TensorType::matrix(o.token_capacity, f.embedding_width,
                                          ElementType::F32));
  in.epoch = tensor(TensorType::vector(1048, ElementType::I32));
  // Full 64-record origins prefix plus one map per bounded resident row/span.
  if (o.cache_capacity > uint64_t(INT64_MAX) - 63)
    return Fail(error, "Qwen map capacity overflow");
  uint64_t map_bytes;
  if (!Multiply((o.cache_capacity + 63) / 64, 4 * o.span_capacity,
                &map_bytes) ||
      map_bytes > uint64_t(INT64_MAX) - 2048)
    return Fail(error, "Qwen map bytes overflow");
  in.rows = tensor(TensorType::vector(2048 + map_bytes, ElementType::I8));
  in.output =
      tensor(TensorType::vector(o.selected_output_capacity, ElementType::I32));
  if (!in.tokens.valid() || !in.residual.valid() || !in.epoch.valid() ||
      !in.rows.valid() || !in.output.valid())
    return false;
  auto resolve = [&](bool global, size_t layer, const char *role,
                     const std::string &key, ValueId *value) {
    const auto &bindings = global ? f.globals : f.layers[layer].tensors;
    size_t count = 0, slot = 0;
    for (size_t i = 0; i < bindings.size(); ++i)
      if (bindings[i].role == role) {
        ++count;
        slot = i;
      }
    if (count != 1)
      return Fail(
          error, std::string("requires exactly one Qwen tensor role: ") + role);
    const auto &t = bindings[slot].tensor;
    if (t.name != key)
      return Fail(error, "Qwen tensor role has noncanonical key: " + t.name);
    const auto &e = t.normalized_extents;
    if (e[0] <= 0 || e[1] <= 0 || e[2] != 1 || e[3] != 1 ||
        t.rank != (e[1] == 1 ? 1u : 2u))
      return Fail(error,
                  "Qwen parameter requires exact vector/matrix rank/extents");
    uint64_t expected;
    TensorType type;
    if (t.storage_type == 0) {
      if (!Multiply(uint64_t(e[0]), uint64_t(e[1]), &expected) ||
          !Multiply(expected, 4, &expected))
        return Fail(error, "Qwen F32 parameter byte overflow");
      if (e[1] != 1 && std::string(role) != "convolution")
        return Fail(
            error, "only Qwen convolution admits a dense F32 parameter matrix");
      // GGML [taps,channels] bytes are channel-major contiguous taps; describe
      // them as row-major [channels,taps] without changing any payload bytes.
      type = e[1] == 1 ? TensorType::vector(e[0], ElementType::F32)
                       : TensorType::matrix(e[1], e[0], ElementType::F32);
    } else {
      Encoding encoding;
      uint64_t block_bytes;
      switch (t.storage_type) {
      case 12:
        encoding = Encoding::Q4_K;
        block_bytes = 144;
        break;
      case 13:
        encoding = Encoding::Q5_K;
        block_bytes = 176;
        break;
      case 14:
        encoding = Encoding::Q6_K;
        block_bytes = 210;
        break;
      default:
        return Fail(error, "unsupported Qwen parameter storage");
      }
      if (e[0] % 256 ||
          !Multiply(uint64_t(e[0]) / 256, uint64_t(e[1]), &expected) ||
          !Multiply(expected, block_bytes, &expected))
        return Fail(error, "invalid Qwen encoded matrix range");
      type = TensorType::matrix(e[0], e[1], ElementType::F32)
                 .setEncoding(encoding)
                 .setLayout(Layout::GgmlWeightRows);
    }
    if (t.byte_size != expected ||
        t.absolute_offset > uint64_t(INT64_MAX) - t.byte_size)
      return Fail(error, "Qwen parameter encoded byte range mismatch");
    ValueId id;
    for (const auto &a : candidate->parameters_) {
      const auto &prior = candidate->parameter(a.coordinate)->tensor;
      if (prior.name != t.name)
        continue;
      if (prior.normalized_extents != t.normalized_extents ||
          prior.rank != t.rank || prior.storage_type != t.storage_type ||
          prior.byte_size != t.byte_size ||
          prior.absolute_offset != t.absolute_offset)
        return Fail(error, "conflicting Qwen repeated parameter identity");
      id = a.value;
      break;
    }
    if (!id.valid())
      id = tensor(type);
    if (!id.valid())
      return false;
    candidate->parameters_.push_back({id, {global, layer, slot}});
    *value = id;
    return true;
  };
  ValueId embedding;
  if (!resolve(true, 0, "token_embedding", "token_embd.weight", &embedding) ||
      !g.operation(OpCode::Embedding)
           .bind(Port::TokenIds, in.tokens)
           .bind(Port::Weight, embedding)
           .bind(Port::Output, in.residual)
           .attributes(EmbeddingAttrs{f.vocabulary_size, f.embedding_width})
           .append(error))
    return false;
  uint64_t next_gdn = 0, next_attention = 0;
  std::set<uint32_t> indices;
  // The only architecture/routing loop: native emission traverses these
  // records.
  for (size_t slot = 0; slot < f.layers.size(); ++slot) {
    const auto &layer = f.layers[slot];
    if (!indices.insert(layer.index).second)
      return Fail(error, "duplicate Qwen layer index");
    bool gdn = layer.kind == model::LayerKind::kGatedDeltaNet;
    if (!gdn && layer.kind != model::LayerKind::kAttention)
      return Fail(error, "unknown Qwen layer kind");
    uint64_t ordinal = gdn ? next_gdn++ : next_attention++;
    if ((gdn &&
         (layer.recurrent_ordinal != ordinal || layer.attention_ordinal)) ||
        (!gdn &&
         (layer.attention_ordinal != ordinal || layer.recurrent_ordinal)))
      return Fail(error, "inconsistent Qwen local layer ordinal");
    QwenInputs::LayerState state;
    state.kind = layer.kind;
    state.layer_slot = slot;
    state.original_layer_index = layer.index;
    state.ordinal = ordinal;
    auto op =
        g.operation(gdn ? OpCode::QwenGdnLayer : OpCode::QwenAttentionLayer);
    op.bind(Port::Residual, in.residual)
        .bind(Port::Epoch, in.epoch)
        .bind(Port::Rows, in.rows);
    for (const auto &binding : QwenLayerWeightRoles(
             gdn ? OpCode::QwenGdnLayer : OpCode::QwenAttentionLayer)) {
      ValueId value;
      if (!resolve(false, slot, binding.role,
                   "blk." + std::to_string(layer.index) + "." + binding.suffix,
                   &value))
        return false;
      op.bind(binding.port, value);
    }
    if (gdn) {
      state.history = tensor(TensorType::matrix(
          channels, ssm.convolution_kernel - 1, ElementType::F32));
      auto matrix = TensorType::matrix(ssm.value_heads, ssm.state_width,
                                       ElementType::F32);
      matrix.addDimension(ssm.state_width);
      state.recurrent = tensor(matrix);
      if (!state.history.valid() || !state.recurrent.valid())
        return false;
      op.bind(Port::ConvolutionHistory, state.history)
          .bind(Port::RecurrentState, state.recurrent);
      QwenGdnLayerAttrs attrs;
      attrs.layer_index = layer.index;
      attrs.recurrent_ordinal = ordinal;
      attrs.token_capacity = o.token_capacity;
      attrs.span_capacity = o.span_capacity;
      attrs.hidden_width = f.embedding_width;
      attrs.ffn_width = layer.feed_forward_width;
      attrs.convolution_taps = ssm.convolution_kernel;
      attrs.state_width = ssm.state_width;
      attrs.key_heads = ssm.key_heads;
      attrs.value_heads = ssm.value_heads;
      attrs.epsilon = f.rms_epsilon;
      op.attributes(attrs);
    } else {
      auto cache = TensorType::matrix(o.cache_capacity, layer.kv_heads,
                                      ElementType::F16);
      cache.addDimension(f.key_head_width);
      state.key_cache = tensor(cache);
      state.value_cache = tensor(cache);
      if (!state.key_cache.valid() || !state.value_cache.valid())
        return false;
      op.bind(Port::KeyCache, state.key_cache)
          .bind(Port::ValueCache, state.value_cache);
      QwenAttentionLayerAttrs attrs;
      attrs.layer_index = layer.index;
      attrs.attention_ordinal = ordinal;
      attrs.token_capacity = o.token_capacity;
      attrs.span_capacity = o.span_capacity;
      attrs.hidden_width = f.embedding_width;
      attrs.ffn_width = layer.feed_forward_width;
      attrs.query_heads = layer.query_heads;
      attrs.kv_heads = layer.kv_heads;
      attrs.head_width = f.key_head_width;
      attrs.rotary_width = f.rope.width;
      attrs.cache_capacity = o.cache_capacity;
      attrs.epsilon = f.rms_epsilon;
      attrs.rope_base = f.rope.base;
      attrs.rope_scaling_factor = f.rope.scaling_factor;
      if (f.rope.scaling_kind != "none" && f.rope.scaling_kind != "linear")
        return Fail(error, "unsupported Qwen rotary scaling");
      attrs.rope_scaling = f.rope.scaling_kind == "none" ? RopeScaling::None
                                                         : RopeScaling::Linear;
      attrs.rope_sections = f.rope.sections;
      attrs.attention_scale = 1 / std::sqrt(double(f.key_head_width));
      op.attributes(attrs);
    }
    if (!op.append(error))
      return false;
    in.layers.push_back(state);
  }
  ValueId norm, output;
  if (!resolve(true, 0, "output_norm", "output_norm.weight", &norm) ||
      !resolve(true, 0, "output",
               f.output_tied_to_embedding ? "token_embd.weight"
                                          : "output.weight",
               &output) ||
      !g.operation(OpCode::QwenEpochHead)
           .bind(Port::Input, in.residual)
           .bind(Port::Gamma, norm)
           .bind(Port::Weight, output)
           .bind(Port::Epoch, in.epoch)
           .bind(Port::Output, in.output)
           .attributes(QwenEpochHeadAttrs{
               f.embedding_width, f.vocabulary_size, f.rms_epsilon,
               o.token_capacity, o.span_capacity, o.selected_output_capacity})
           .append(error))
    return false;
  *out = std::move(candidate);
  error->clear();
  return true;
}
} // namespace hrx::qwen
