// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "loomc/iree.h"
#include <cmath>

namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
loomc_status_t Error(const char *message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
}
ValueId Operand(const Invocation &op, Port port) {
  const auto *binding = op.binding(port);
  return binding ? binding->value() : ValueId{};
}
bool Rotary(uint64_t width, double base, RopeScaling scaling, double factor,
            FactorMode mode) {
  return width == 128 && base == 500000 && factor == 1 &&
         mode == FactorMode::PerPair &&
         (scaling == RopeScaling::None || scaling == RopeScaling::Linear);
}
} // namespace
loomc_status_t NativeEmissionSession::EmitAttentionRegion(
    const llama::LlamaStructure &owner, llama::LlamaRegion selection,
    const AttentionSequenceOptions &o, Value map, Value cache) {
  if (failed_ || &owner.graph() != &graph_ || !has_llama_rows_ ||
      selection.kind != llama::LlamaRegionKind::Attention ||
      selection.layer_count != 1 ||
      selection.layer_begin >= owner.inputs().layers.size())
    return Error("attention region requires its configured structure owner");
  llama::LlamaRegionView view;
  std::string error;
  if (!owner.select(selection, &view, &error) ||
      view.operation_end - view.operation_begin != 8)
    return Error("attention region requires a recorded normalized layer");
  const auto start = view.operation_begin + 4;
  if (has_emitted_existing_ && start <= last_existing_)
    return Error("attention region must follow append order");
  const auto &ops = graph_.invocations();
  constexpr OpCode order[] = {
      OpCode::RmsNorm,         OpCode::Linear,        OpCode::Linear,
      OpCode::Linear,          OpCode::Rope,          OpCode::RotatePublishKV,
      OpCode::CausalAttention, OpCode::LinearResidual};
  for (size_t i = 0; i < 8; ++i)
    if (ops[view.operation_begin + i].opcode() != order[i])
      return Error("attention region has incorrect operation order");
  const auto &rope = ops[start], &publish = ops[start + 1],
             &causal = ops[start + 2];
  if (rope.opcode() != OpCode::Rope ||
      publish.opcode() != OpCode::RotatePublishKV ||
      causal.opcode() != OpCode::CausalAttention ||
      ops[start + 3].opcode() != OpCode::LinearResidual)
    return Error("attention region has incorrect operation order");
  const auto &in = owner.inputs();
  const auto &rows = llama_rows_;
  if (rows.row_begin != in.row_begin || rows.row_count != in.row_count ||
      rows.packet_positions != in.positions || !in.positions.valid() ||
      Find(in.positions) || owner.options().row_capacity != rows.row_capacity ||
      owner.options().phase != rows.phase ||
      o.row_capacity != rows.row_capacity || o.phase != rows.phase ||
      o.token_begin != rows.token_begin ||
      o.control_origin != rows.control_origin ||
      o.layer != selection.layer_begin)
    return Error("attention region disagrees with configured packet rows");
  for (size_t i = view.operation_begin; i < view.operation_end; ++i)
    if (Operand(ops[i], Port::RowBegin) != in.row_begin ||
        Operand(ops[i], Port::RowCount) != in.row_count)
      return Error("attention region has inconsistent row identities");
  const auto rotated = Operand(rope, Port::Output);
  const auto attended = Operand(causal, Port::Output);
  const auto factors = Operand(rope, Port::Factors);
  const auto &planes = in.layers[selection.layer_begin];
  if (!rotated.valid() || !attended.valid() || !factors.valid() ||
      Find(rotated) || Find(attended) || Reserved(attended) ||
      Operand(causal, Port::Query) != rotated ||
      Operand(ops[start + 3], Port::Input) != attended ||
      Operand(publish, Port::Factors) != factors ||
      planes.key_cache == planes.value_cache || Find(planes.key_cache) ||
      Find(planes.value_cache))
    return Error(
        "attention region output, factors or arena-plane binding conflict");
  for (const auto *op : {&rope, &publish, &causal})
    if (Operand(*op, Port::Positions) != in.positions)
      return Error("attention positions must be the packet-derived identity");
  for (const auto *op : {&publish, &causal})
    if (Operand(*op, Port::KeyCache) != planes.key_cache ||
        Operand(*op, Port::ValueCache) != planes.value_cache)
      return Error("attention region must use its layer-local cache planes");
  if (Operand(rope, Port::Input) != Operand(ops[start - 3], Port::Output) ||
      Operand(publish, Port::Key) != Operand(ops[start - 2], Port::Output) ||
      Operand(publish, Port::Value) != Operand(ops[start - 1], Port::Output))
    return Error("attention region has inconsistent projection dataflow");
  for (const auto &port : view.outside)
    if (port.value == rotated)
      return Error("rotated query cannot escape the packet region");
  for (size_t i = 0; i < ops.size(); ++i)
    if (i < start || i > start + 2)
      for (const auto &binding : ops[i].bindings())
        if (binding.value() == rotated)
          return Error("rotated query has an outside consumer");
  const auto *r = std::get_if<RopeAttrs>(&rope.attributes());
  const auto *p = std::get_if<RotatePublishKVAttrs>(&publish.attributes());
  const auto *a = std::get_if<CausalAttentionAttrs>(&causal.attributes());
  if (!r || !p || !a || r->head_count != 24 || r->head_width != 128 ||
      !Rotary(r->rotary_width, r->base, r->scaling, r->scaling_factor,
              r->factor_mode) ||
      p->kv_heads != 8 || p->key_width != 128 || p->value_width != 128 ||
      !Rotary(p->rotary_width, p->base, p->scaling, p->scaling_factor,
              p->factor_mode) ||
      r->scaling != p->scaling || a->query_heads != 24 || a->kv_heads != 8 ||
      a->key_width != 128 || a->value_width != 128 ||
      a->numerical_policy != NumericalPolicy::LlamaPacketMixed ||
      static_cast<float>(a->scale) !=
          static_cast<float>(1.0 / std::sqrt(128.0)) ||
      p->cache_capacity != owner.options().cache_capacity ||
      a->cache_capacity != owner.options().cache_capacity)
    return Error("attention region differs from the packet numerical profile");
  const Binding *factor_binding = Find(factors);
  bool owned_factors = false;
  for (const auto &association : owner.parameters())
    if (association.value == factors &&
        association.coordinate.scope == llama::ParameterScope::Global) {
      const auto *parameter = owner.parameter(association.coordinate);
      owned_factors = parameter && parameter->role == "rope_frequencies" &&
                      parameter->tensor.storage_type == 0 &&
                      parameter->tensor.normalized_extents ==
                          std::array<int64_t, 4>{64, 1, 1, 1} &&
                      factor_binding &&
                      (factor_binding->parameter.empty() ||
                       factor_binding->parameter == parameter->tensor.name);
    }
  const Binding *q = Find(Operand(rope, Port::Input));
  const Binding *k = Find(Operand(publish, Port::Key));
  const Binding *v = Find(Operand(publish, Port::Value));
  if (!owned_factors || !q || !k || !v || !q->parameter.empty() ||
      !k->parameter.empty() || !v->parameter.empty())
    return Error("attention requires resolved factors and projection buffers");
#define TRY(call)                                                              \
  do {                                                                         \
    auto status = (call);                                                      \
    if (!loomc_status_is_ok(status))                                           \
      return status;                                                           \
  } while (0)
  TRY(CheckBuffer(map));
  TRY(CheckBuffer(cache));
  AttentionNativeConfig config;
  TRY(CheckAttentionSequence(module_, program_, o, &config));
  if (config.layer_count != owner.facts().layers.size() ||
      config.cache_capacity != owner.options().cache_capacity)
    return Error("attention owner disagrees with loaded native configuration");
  // All admission checks precede the first mutation. Native failures thereafter
  // use the session's existing discard-owner protocol.
  Value factor = factor_binding->native;
  const auto factor_key = factor_binding->parameter;
  const Value query = q->native, key = k->native, value = v->native;
  auto status = StageCount();
  if (!loomc_status_is_ok(status))
    return FailMutation(status);
  if (!factor_key.empty()) {
    Value count;
    status = values_.Constant(64, LOOM_SCALAR_TYPE_INDEX, &count);
    if (loomc_status_is_ok(status))
      status = values_.Parameter(factor, factor_key, {}, LOOM_SCALAR_TYPE_F32,
                                 {count}, &factor);
    if (!loomc_status_is_ok(status))
      return FailMutation(status);
  }
  Value output;
  status = BuildAttentionSequence(
      module_, program_, o,
      {rows.packet, map, cache, factor, query, key, value}, &output);
  if (!loomc_status_is_ok(status))
    return FailMutation(status);
  status = CheckBuffer(output);
  if (!loomc_status_is_ok(status))
    return FailMutation(status);
  bindings_.push_back({attended, output, {}});
  started_ = true;
  last_existing_ = start + 2;
  has_emitted_existing_ = true;
  return loomc_ok_status();
#undef TRY
}
} // namespace hrx::dispatch
