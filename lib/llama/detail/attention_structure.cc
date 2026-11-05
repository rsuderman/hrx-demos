// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/detail/attention_structure.h"
#include <array>
#include <cmath>
#include <limits>
namespace hrx::llama::detail {
namespace {
using namespace operations;
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
} // namespace
bool AppendNormalizedAttention(StructureConstruction *owner, size_t layer_slot,
                               NormalizedAttentionRegion *out,
                               std::string *error) {
  if (!owner || !out || !error)
    return Fail(error, "attention requires owner, region and error outputs");
  const auto &s = owner->structure();
  const auto &f = s.facts();
  const auto &inputs = s.inputs();
  auto &g = owner->graph();
  if (f.family != "llama" || layer_slot >= f.layers.size() ||
      layer_slot >= inputs.layers.size() || !g.value(inputs.residual))
    return Fail(error, "attention requires Llama layer and created inputs");
  const auto &l = f.layers[layer_slot];
  const auto max = uint64_t(std::numeric_limits<int64_t>::max());
  const auto d = f.key_head_width;
  if (l.kind != model::LayerKind::kAttention || !d || d != f.value_head_width ||
      !l.query_heads || !l.kv_heads || l.query_heads % l.kv_heads ||
      l.query_heads > max / d || l.kv_heads > max / d || !f.embedding_width ||
      f.embedding_width > max || f.rope.width != d || d % 2 ||
      !std::isfinite(f.rms_epsilon) || f.rms_epsilon <= 0 ||
      !std::isfinite(f.rope.base) || f.rope.base <= 0 ||
      !std::isfinite(f.rope.scaling_factor) || f.rope.scaling_factor <= 0 ||
      (f.rope.scaling_kind != "none" && f.rope.scaling_kind != "linear") ||
      (f.rope.scaling_kind == "none" && f.rope.scaling_factor != 1))
    return Fail(error, "unsupported Llama attention geometry or rotary facts");
  for (auto section : f.rope.sections)
    if (section)
      return Fail(error, "sectioned rotary is not Llama attention");
  for (auto code :
       {OpCode::RmsNorm, OpCode::Linear, OpCode::Rope, OpCode::RotatePublishKV,
        OpCode::CausalAttention, OpCode::LinearResidual})
    if (!g.definition(code))
      return Fail(error, std::string("missing attention definition ") +
                             OpCodeName(code));
  const uint64_t qwidth = l.query_heads * d, kvwidth = l.kv_heads * d;
  constexpr std::array<const char *, 5> roles = {"input_norm", "query", "key",
                                                 "value", "attention_output"};
  std::array<size_t, 5> slots{};
  size_t factor_slot = 0, factor_count = 0;
  for (size_t role = 0; role < roles.size(); ++role) {
    size_t count = 0;
    for (size_t i = 0; i < l.tensors.size(); ++i)
      if (l.tensors[i].role == roles[role]) {
        slots[role] = i;
        ++count;
      }
    if (count != 1)
      return Fail(error, std::string(roles[role]) + " requires one binding");
    const auto &t = l.tensors[slots[role]].tensor;
    const uint64_t in = role == 4 ? qwidth : f.embedding_width;
    const uint64_t cols = role == 0   ? 1
                          : role == 1 ? qwidth
                          : role == 4 ? f.embedding_width
                                      : kvwidth;
    Encoding encoding;
    if (t.normalized_extents !=
            std::array<int64_t, 4>{int64_t(in), int64_t(cols), 1, 1} ||
        (role == 0 ? t.storage_type != 0
                   : !MatrixEncoding(t.storage_type, &encoding)))
      return Fail(error, std::string(roles[role]) +
                             " shape or numeric storage mismatch");
  }
  // RoPE factors are a global semantic binding, matching model descriptor
  // facts.
  for (size_t i = 0; i < f.globals.size(); ++i)
    if (f.globals[i].role == "rope_frequencies") {
      factor_slot = i;
      ++factor_count;
    }
  if (factor_count > 1)
    return Fail(error, "duplicate rope_frequencies binding");
  if (factor_count) {
    const auto &t = f.globals[factor_slot].tensor;
    if (t.storage_type != 0 ||
        t.normalized_extents != std::array<int64_t, 4>{int64_t(d / 2), 1, 1, 1})
      return Fail(error,
                  "rope_frequencies requires dense F32 per-pair factors");
  }
  std::array<ValueId, 5> weights;
  for (size_t i = 0; i < weights.size(); ++i)
    if (!owner->resolve({ParameterScope::Layer, layer_slot, slots[i]},
                        &weights[i], error))
      return false;
  ValueId factors;
  if (factor_count && !owner->resolve({ParameterScope::Global, 0, factor_slot},
                                      &factors, error))
    return false;
  const uint64_t rows = s.options().row_capacity;
  auto temporary = [&](uint64_t width) {
    return g.tensor(TensorType::matrix(rows, width, ElementType::F32), error);
  };
  const auto norm = temporary(f.embedding_width), q = temporary(qwidth),
             k = temporary(kvwidth), v = temporary(kvwidth),
             rotated = temporary(qwidth), attended = temporary(qwidth);
  if (!norm.valid() || !q.valid() || !k.valid() || !v.valid() ||
      !rotated.valid() || !attended.valid())
    return false;
  NormalizedAttentionRegion staged{layer_slot,
                                   g.invocations().size(),
                                   0,
                                   inputs.residual,
                                   inputs.layers[layer_slot].key_cache,
                                   inputs.layers[layer_slot].value_cache};
  auto op = [&](OpCode code) {
    auto b = g.operation(code);
    b.bind(Port::RowBegin, inputs.row_begin)
        .bind(Port::RowCount, inputs.row_count);
    return b;
  };
  if (!op(OpCode::RmsNorm)
           .bind(Port::Input, inputs.residual)
           .bind(Port::Gamma, weights[0])
           .bind(Port::Output, norm)
           .attributes(RmsNormAttrs{f.embedding_width, f.rms_epsilon})
           .append(error))
    return false;
  for (size_t i = 0; i < 3; ++i)
    if (!op(OpCode::Linear)
             .bind(Port::Input, norm)
             .bind(Port::Weight, weights[i + 1])
             .bind(Port::Output, std::array<ValueId, 3>{q, k, v}[i])
             .attributes(
                 LinearAttrs{f.embedding_width, i == 0 ? qwidth : kvwidth})
             .append(error))
      return false;
  const auto scaling =
      f.rope.scaling_kind == "none" ? RopeScaling::None : RopeScaling::Linear;
  const auto mode = factor_count ? FactorMode::PerPair : FactorMode::None;
  auto rope = op(OpCode::Rope);
  if (factor_count)
    rope.bind(Port::Factors, factors);
  if (!rope.bind(Port::Input, q)
           .bind(Port::Output, rotated)
           .bind(Port::Positions, inputs.positions)
           .attributes(RopeAttrs{l.query_heads, d, d, f.rope.base, scaling,
                                 f.rope.scaling_factor, mode})
           .append(error))
    return false;
  auto publish = op(OpCode::RotatePublishKV);
  if (factor_count)
    publish.bind(Port::Factors, factors);
  if (!publish.bind(Port::Key, k)
           .bind(Port::Value, v)
           .bind(Port::Positions, inputs.positions)
           .bind(Port::KeyCache, staged.key_cache)
           .bind(Port::ValueCache, staged.value_cache)
           .attributes(RotatePublishKVAttrs{
               l.kv_heads, d, d, s.options().cache_capacity, d, f.rope.base,
               scaling, f.rope.scaling_factor, mode})
           .append(error) ||
      !op(OpCode::CausalAttention)
           .bind(Port::Query, rotated)
           .bind(Port::Positions, inputs.positions)
           .bind(Port::KeyCache, staged.key_cache)
           .bind(Port::ValueCache, staged.value_cache)
           .bind(Port::Output, attended)
           .attributes(CausalAttentionAttrs{
               l.query_heads, l.kv_heads, d, d, s.options().cache_capacity,
               1.0 / std::sqrt(double(d)), s.options().attention_policy})
           .append(error) ||
      !op(OpCode::LinearResidual)
           .bind(Port::Input, attended)
           .bind(Port::Weight, weights[4])
           .bind(Port::Residual, inputs.residual)
           .attributes(LinearResidualAttrs{qwidth, f.embedding_width})
           .append(error))
    return false;
  staged.operation_end = g.invocations().size();
  if (!owner->record({LlamaRegionKind::Attention, layer_slot, 1},
                     staged.operation_begin, staged.operation_end, error))
    return false;
  *out = staged;
  return true;
}
} // namespace hrx::llama::detail
