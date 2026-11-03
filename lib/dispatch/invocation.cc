// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "loomc/iree.h"
#include <utility>

namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
loomc_status_t Error(const std::string &message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message.c_str()));
}
} // namespace
NativeEmissionSession::NativeEmissionSession(
    const Registry &registry, const operations::OperationGraph &graph,
    ModuleBuilder &module, CommandProgramBuilder &program)
    : registry_(registry), graph_(graph), module_(module), program_(program),
      values_(program) {}
const NativeEmissionSession::Binding *
NativeEmissionSession::Find(ValueId id) const {
  for (const auto &binding : bindings_)
    if (binding.logical == id)
      return &binding;
  return nullptr;
}
loomc_status_t NativeEmissionSession::CheckBuffer(Value native) const {
  return program_.InspectNative(
      {native},
      [](const loom_module_t *module, const std::vector<loom_value_id_t> &ids) {
        if (!loom_type_equal(loom_module_value_type(module, ids[0]),
                             loom_type_buffer()))
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "boundary value must be a buffer");
        return iree_ok_status();
      });
}
loomc_status_t NativeEmissionSession::Bind(ValueId logical, Value native) {
  if (failed_ || !graph_.value(logical) ||
      !graph_.value(logical)->tensorType() || Find(logical))
    return Error("invalid, foreign or already bound logical tensor");
  auto status = CheckBuffer(native);
  if (!loomc_status_is_ok(status))
    return status;
  if (graph_.value(logical)->tensorType()->encoding() != Encoding::Dense)
    return Error("quantized weights require named parameter binding");
  bindings_.push_back({logical, native, {}});
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::Parameter(ValueId logical,
                                                Value parameters,
                                                std::string key) {
  if (failed_ || !graph_.value(logical) ||
      !graph_.value(logical)->tensorType() || Find(logical) || key.empty() ||
      key.find('\0') != std::string::npos ||
      key.find('{') != std::string::npos || key.find('}') != std::string::npos)
    return Error("invalid, foreign or already bound named parameter");
  auto status = CheckBuffer(parameters);
  if (!loomc_status_is_ok(status))
    return status;
  bindings_.push_back({logical, parameters, std::move(key)});
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::Control(Value control) {
  if (failed_)
    return Error("failed invocation owner");
  auto status = CheckBuffer(control);
  if (!loomc_status_is_ok(status))
    return status;
  control_ = control;
  has_control_ = true;
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::Native(ValueId logical,
                                             Value *out) const {
  const auto *binding = Find(logical);
  if (!out || failed_ || !binding || !binding->parameter.empty())
    return Error("missing native logical buffer");
  auto status = CheckBuffer(binding->native);
  if (loomc_status_is_ok(status))
    *out = binding->native;
  return status;
}
loomc_status_t NativeEmissionSession::FailMutation(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return status;
  failed_ = true;
  // Discard the mutable owner after any partial staging, using its existing
  // terminal native-error path. No staged graph/native output is published.
  std::vector<Value> ignored;
  auto terminal = program_.BuildNative(
      {},
      [](loom_module_t *, loom_builder_t *,
         const std::vector<loom_value_id_t> &, std::vector<loom_value_id_t> *) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "invocation staging failed");
      },
      &ignored);
  loomc_status_free(terminal);
  return status;
}
loomc_status_t
NativeEmissionSession::Emit(Id id, const OperationAttrs &attrs,
                            const std::vector<shapes::Operand> &inputs,
                            ValueId existing_result, shapes::Result *out_result,
                            Value *out_native, const std::string &variant) {
  if (!out_result || !out_native || failed_)
    return Error("missing result or failed invocation owner");
  const auto *entry = registry_.Find(id);
  if (!entry)
    return Error("missing dispatch registration");
  std::string error;
  Registry validation;
  if (!validation.Add(*entry, &error))
    return Error(error);
  shapes::Result inferred;
  if (!shapes::InferResult(graph_, entry->operation, attrs, inputs, &inferred,
                           &error))
    return Error(error);
  if (!graph_.definition(entry->operation))
    return Error("graph lacks operation definition");
  if (existing_result.valid()) {
    const auto *description = graph_.value(existing_result);
    const auto *actual = description ? description->tensorType() : nullptr;
    if (!actual || actual->rank() != inferred.type.rank() ||
        actual->elementType() != inferred.type.elementType() ||
        actual->encoding() != inferred.type.encoding() ||
        actual->layout() != inferred.type.layout())
      return Error("existing output type disagrees with inferred result");
    for (size_t i = 0; i < actual->rank(); ++i)
      if (actual->dimension(i) != inferred.type.dimension(i))
        return Error("existing output shape disagrees with inferred result");
    if (inferred.access == Access::ReadWrite &&
        existing_result != inferred.alias)
      return Error("existing ReadWrite output must retain its explicit alias");
    if (inferred.access == Access::Write) {
      if (const auto *binding = Find(existing_result)) {
        if (!binding->parameter.empty())
          return Error(
              "existing output cannot overwrite an immutable parameter");
        auto status = CheckBuffer(binding->native);
        if (!loomc_status_is_ok(status))
          return status;
      }
    }
  }
  // Keep original graph controls for shape verification and inspection; only
  // this explicit proof specialization permits omitting their native mapping.
  std::vector<shapes::Operand> native_inputs;
  for (const auto &input : inputs) {
    if (input.port == Port::RowBegin || input.port == Port::RowCount) {
      if (!one_row_ ||
          input.value !=
              (input.port == Port::RowBegin ? row_begin_ : row_count_))
        return Error(
            "active subspan requires explicit matching one-row specialization");
    } else
      native_inputs.push_back(input);
  }
  for (const auto &input : native_inputs) {
    auto *binding = Find(input.value);
    if (!binding)
      return Error("unbound logical input");
    auto status = CheckBuffer(binding->native);
    if (!loomc_status_is_ok(status))
      return status;
  }
  if (one_row_) {
    for (const auto &input : inputs)
      if (input.port == Port::Input &&
          graph_.value(input.value)->tensorType()->dimension(0) != 1)
        return Error(
            "explicit one-row entrypoint requires the whole static row domain");
  }
  InvocationRequest request{graph_, attrs, inputs, inferred};
  const auto *selected = registry_.Resolve(id, request, variant, &error);
  if (!selected)
    return Error(error);
  std::vector<NativeOperand> native_operands;
  for (const auto &input : native_inputs) {
    const auto *binding = Find(input.value);
    native_operands.push_back(
        {input.port, input.value, binding->native, binding->parameter});
  }
  Value existing;
  bool has_output = false;
  if (inferred.access == Access::ReadWrite) {
    existing = Find(inferred.alias)->native;
    has_output = true;
  } else if (existing_result.valid() && Find(existing_result)) {
    existing = Find(existing_result)->native;
    has_output = true;
  }
  BuildContext context{request,      module_,         program_,
                       values_,      native_operands, control_,
                       has_control_, existing,        has_output};
  Value native_output;
  auto status = selected->builder.build(*selected, context, &native_output);
  if (context.staged)
    started_ = true;
  if (!loomc_status_is_ok(status))
    return context.staged ? FailMutation(status) : status;

  status = CheckBuffer(native_output);
  if (!loomc_status_is_ok(status))
    return context.staged ? FailMutation(status) : status;
  *out_result = inferred;
  *out_native = native_output;
  return loomc_ok_status();
}

loomc_status_t NativeEmissionSession::SpecializeOneRow(ValueId begin,
                                                       ValueId count) {
  if (failed_ || started_ || one_row_)
    return Error("one-row specialization must be set once before emission");
  const auto *begin_value = graph_.value(begin),
             *count_value = graph_.value(count);
  if (!begin_value || !count_value || begin == count ||
      !begin_value->scalarType() || !count_value->scalarType() ||
      *begin_value->scalarType() != ScalarType::I64 ||
      *count_value->scalarType() != ScalarType::I64)
    return Error("one-row specialization requires distinct owned I64 control "
                 "identities");
  row_begin_ = begin;
  row_count_ = count;
  one_row_ = true;
  return loomc_ok_status();
}

loomc_status_t NativeEmissionSession::EmitExisting(Id id,
                                                   const Invocation &invocation,
                                                   const std::string &variant) {
  size_t index = 0;
  for (; index < graph_.invocations().size(); ++index)
    if (&graph_.invocations()[index] == &invocation)
      break;
  if (index == graph_.invocations().size() ||
      (has_emitted_existing_ && index <= last_existing_))
    return Error("existing invocation must belong to this graph and follow "
                 "append order");
  const auto *entry = registry_.Find(id);
  if (!entry || entry->operation != invocation.opcode())
    return Error("dispatch does not match existing invocation");
  const auto *definition = graph_.definition(invocation.opcode());
  if (!definition)
    return Error("missing existing operation definition");
  std::vector<shapes::Operand> inputs;
  ValueId output;
  for (const auto &binding : invocation.bindings()) {
    const auto *port = definition->port(binding.port());
    if (!port)
      return Error("unknown existing operation port");
    if (port->access() == Access::Write) {
      if (output.valid())
        return Error("dispatch expects a single existing output");
      output = binding.value();
    } else {
      inputs.push_back({binding.port(), binding.value()});
      if (port->access() == Access::ReadWrite)
        output = binding.value();
    }
  }
  if (!output.valid())
    return Error("missing existing output identity");
  shapes::Result inferred;
  Value native;
  auto status = Emit(id, invocation.attributes(), inputs, output, &inferred,
                     &native, variant);
  if (!loomc_status_is_ok(status))
    return status;
  if (inferred.access == Access::Write && !Find(output))
    bindings_.push_back({output, native, {}});
  last_existing_ = index;
  has_emitted_existing_ = true;
  return loomc_ok_status();
}

loomc_status_t
InvocationProgram::Invoke(Id id, const OperationAttrs &attrs,
                          const std::vector<shapes::Operand> &inputs,
                          ValueId *out, const std::string &variant) {
  if (!out)
    return Error("missing logical result");
  shapes::Result inferred;
  Value native;
  auto status =
      session_.Emit(id, attrs, inputs, {}, &inferred, &native, variant);
  if (!loomc_status_is_ok(status))
    return status;
  std::string error;
  ValueId result = inferred.alias;
  if (!result.valid()) {
    result = graph_.tensor(inferred.type, &error);
    if (!result.valid())
      return session_.FailMutation(Error(error));
  }
  auto invocation = graph_.operation(session_.registry_.Find(id)->operation);
  std::visit([&](const auto &a) { invocation.attributes(a); }, attrs);
  for (const auto &input : inputs)
    invocation.bind(input.port, input.value);
  if (inferred.access == Access::Write)
    invocation.bind(inferred.port, result);
  if (!invocation.append(&error))
    return session_.FailMutation(Error(error));
  if (inferred.access == Access::Write)
    session_.bindings_.push_back({result, native, {}});
  *out = result;
  return loomc_ok_status();
}
} // namespace hrx::dispatch
