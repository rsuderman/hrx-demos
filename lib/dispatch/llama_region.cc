// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/llama_region.h"
#include "lib/dispatch/parameters.h"
namespace hrx::dispatch {
namespace {
using namespace operations;
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
const TensorType *Type(const OperationGraph &graph, const Invocation &op,
                       Port port) {
  auto *binding = op.binding(port);
  auto *value = binding ? graph.value(binding->value()) : nullptr;
  return value ? value->tensorType() : nullptr;
}
bool HasRole(const llama::LlamaStructure &owner, ValueId value, size_t layer,
             const char *role) {
  for (const auto &association : owner.parameters()) {
    if (association.value != value ||
        association.coordinate.scope != llama::ParameterScope::Layer ||
        association.coordinate.layer_slot != layer)
      continue;
    auto *binding = owner.parameter(association.coordinate);
    if (binding && binding->role == role)
      return true;
  }
  return false;
}
} // namespace
bool FfnRegion::Select(std::shared_ptr<const llama::LlamaStructure> owner,
                       llama::LlamaRegion selection, const Registry &registry,
                       std::unique_ptr<FfnRegion> *out, std::string *error) {
  if (!owner || !out)
    return Fail(error, "FFN selection requires an owner and output");
  if (selection.kind != llama::LlamaRegionKind::Ffn ||
      selection.layer_count != 1)
    return Fail(error, "expected one recorded FFN region");
  if (!owner->options().row_capacity || owner->options().row_capacity > 512 ||
      (owner->options().phase == Phase::Decode &&
       owner->options().row_capacity != 1))
    return Fail(error, "unsupported FFN row capacity for phase");
  llama::LlamaRegionView view;
  if (!owner->select(selection, &view, error))
    return false;
  const auto &graph = owner->graph();
  if (view.operation_end - view.operation_begin != 3)
    return Fail(error, "normalized FFN requires exactly three operations");
  const bool decode = owner->options().phase == Phase::Decode;
  std::vector<Id> ids;
  for (size_t i = view.operation_begin; i < view.operation_end; ++i) {
    const auto &op = graph.invocations()[i];
    Id id;
    Port parameter;
    switch (op.opcode()) {
    case OpCode::RmsNorm:
      id = Id::RmsNorm;
      parameter = Port::Gamma;
      break;
    case OpCode::GatedMatmul:
      id = decode ? Id::GateUpDecode : Id::GateUpPrefill;
      parameter = Port::GateWeight;
      break;
    case OpCode::LinearResidual:
      id = decode ? Id::DownDecode : Id::DownPrefill;
      parameter = Port::Weight;
      break;
    default:
      return Fail(error, "unsupported operation in FFN region");
    }
    const auto *entry = registry.Find(id);
    const auto *weight = Type(graph, op, parameter);
    if (!entry || entry->operation != op.opcode() || !weight)
      return Fail(error, "missing or incompatible FFN registration");
    Registry validation;
    if (!validation.Add(*entry, error))
      return false;
    std::vector<shapes::Operand> inputs;
    const auto *definition = graph.definition(op.opcode());
    for (const auto &binding : op.bindings())
      if (definition->port(binding.port())->access() != Access::Write)
        inputs.push_back({binding.port(), binding.value()});
    shapes::Result result;
    if (!shapes::InferResult(graph, op.opcode(), op.attributes(), inputs,
                             &result, error) ||
        !registry.Resolve(
            id, InvocationRequest{graph, op.attributes(), inputs, result}, {},
            error))
      return false;
    ids.push_back(id);
  }
  const std::vector<Id> expected{Id::RmsNorm,
                                 decode ? Id::GateUpDecode : Id::GateUpPrefill,
                                 decode ? Id::DownDecode : Id::DownPrefill};
  if (ids != expected)
    return Fail(error, "normalized FFN operation order differs");
  const auto &ops = graph.invocations();
  const auto &norm = ops[view.operation_begin];
  const auto &gate = ops[view.operation_begin + 1];
  const auto &down = ops[view.operation_begin + 2];
  auto value = [](const Invocation &op, Port port) {
    auto *binding = op.binding(port);
    return binding ? binding->value() : ValueId{};
  };
  if (value(norm, Port::Input) != owner->inputs().residual ||
      value(gate, Port::Input) != value(norm, Port::Output) ||
      value(down, Port::Input) != value(gate, Port::Output) ||
      value(down, Port::Residual) != owner->inputs().residual)
    return Fail(error, "FFN dataflow must preserve owner residual and outputs");
  for (size_t i = view.operation_begin; i < view.operation_end; ++i)
    if (value(ops[i], Port::RowBegin) != owner->inputs().row_begin ||
        value(ops[i], Port::RowCount) != owner->inputs().row_count)
      return Fail(error, "FFN row controls must match the retained owner");
  if (!HasRole(*owner, value(norm, Port::Gamma), selection.layer_begin,
               "post_norm") ||
      !HasRole(*owner, value(gate, Port::GateWeight), selection.layer_begin,
               "ffn_gate") ||
      !HasRole(*owner, value(gate, Port::UpWeight), selection.layer_begin,
               "ffn_up") ||
      !HasRole(*owner, value(down, Port::Weight), selection.layer_begin,
               "ffn_down"))
    return Fail(error, "FFN weights must retain selected layer role identity");
  // Existing operations remain the source of operands, output identities,
  // access domains, attributes and execution order.
  *out = std::unique_ptr<FfnRegion>(
      new FfnRegion(std::move(owner), std::move(view), std::move(ids)));
  return true;
}
bool FfnRegion::Emit(NativeEmissionSession &session,
                     loomify_gguf::Value parameters, std::string *error) const {
  if (&session.graph() != &owner_->graph())
    return Fail(error, "FFN emission session belongs to another owner");
  auto check = [&](loomc_status_t status) {
    const bool ok = loomc_status_is_ok(status);
    if (!ok && error) {
      auto message = loomc_status_message(status);
      error->assign(message.data ? message.data : "", message.size);
    }
    loomc_status_free(status);
    return ok;
  };
  if (!BindRegionParameters(*owner_, view_, parameters, session, error))
    return false;
  for (size_t i = view_.operation_begin; i < view_.operation_end; ++i)
    if (!check(session.EmitExisting(dispatches_[i - view_.operation_begin],
                                    owner_->graph().invocations()[i])))
      return false;
  return true;
}
} // namespace hrx::dispatch
