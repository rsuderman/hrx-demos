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
bool NativeEmissionSession::Reserved(ValueId id) const {
  if (has_qwen_epoch_ && id.valid()) {
    if (id == qwen_epoch_.logical_epoch || id == qwen_epoch_.logical_rows)
      return true;
    for (const auto &state : qwen_epoch_.states)
      if (id == state.value) return true;
  }
  return has_llama_rows_ && id.valid() &&
         (id == llama_rows_.count_control ||
          id == llama_rows_.packet_positions);
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
      !graph_.value(logical)->tensorType() || Find(logical) ||
      Reserved(logical))
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
      !graph_.value(logical)->tensorType() || Find(logical) ||
      Reserved(logical) || key.empty() || key.find('\0') != std::string::npos ||
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
  if (has_llama_rows_)
    return CheckPacket(control);
  control_ = control;
  has_control_ = true;
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::CheckPacket(Value packet) const {
  return program_.InspectNative(
      {packet, llama_rows_.packet},
      [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
        return ids[0] == ids[1]
                   ? iree_ok_status()
                   : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                      "configured Llama packet is immutable");
      });
}
loomc_status_t NativeEmissionSession::LlamaRows(const NativeLlamaRows &rows) {
  if (failed_ || started_ || has_llama_rows_ || has_qwen_epoch_ || one_row_)
    return Error("Llama row binding must be configured once before emission");
  const auto *begin = graph_.value(rows.row_begin);
  const auto *count = graph_.value(rows.row_count);
  if (!begin || !count || rows.row_begin == rows.row_count ||
      !begin->scalarType() || !count->scalarType() ||
      *begin->scalarType() != ScalarType::I64 ||
      *count->scalarType() != ScalarType::I64 || !rows.row_capacity ||
      rows.row_capacity > 512 || rows.token_begin >= rows.row_capacity ||
      rows.control_origin > uint64_t(INT64_MAX) - 12 ||
      (rows.phase != Phase::Prefill && rows.phase != Phase::Decode) ||
      (rows.phase == Phase::Decode &&
       (rows.row_capacity != 1 || rows.token_begin != 0)))
    return Error(
        "invalid Llama row identities, extent, phase or packet origin");
  auto derived = [&](ValueId id, uint64_t size) {
    if (!id.valid())
      return true;
    const auto *value = graph_.value(id);
    const auto *type = value ? value->tensorType() : nullptr;
    return type && type->rank() == 1 && type->dimension(0) == size &&
           type->elementType() == ElementType::I32 &&
           type->encoding() == Encoding::Dense &&
           type->layout() == Layout::RowMajor && !Find(id);
  };
  if (!derived(rows.count_control, 3) ||
      !derived(rows.packet_positions, rows.row_capacity) ||
      (rows.count_control.valid() &&
       rows.count_control == rows.packet_positions))
    return Error("invalid or conflicting reserved Llama derivation identity");
  auto status = CheckBuffer(rows.packet);
  if (!loomc_status_is_ok(status))
    return status;
  if (has_control_) {
    status = program_.InspectNative(
        {control_, rows.packet},
        [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
          return ids[0] == ids[1]
                     ? iree_ok_status()
                     : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                        "conflicting existing packet control");
        });
    if (!loomc_status_is_ok(status))
      return status;
  }
  llama_rows_ = rows;
  has_llama_rows_ = has_control_ = true;
  control_ = rows.packet;
  return loomc_ok_status();
}
loomc_status_t NativeEmissionSession::StageCount() {
  if (!has_llama_rows_ || has_count_)
    return loomc_ok_status();
  KernelRef helper;
  auto status = module_.Kernel("llama32_head_count", &helper);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index = module_.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto offset = module_.Scalar(LOOM_SCALAR_TYPE_OFFSET);
  const auto buffer = module_.Buffer();
  status =
      module_.CheckKernelInterface(helper, {}, {index, offset, buffer, buffer});
  if (!loomc_status_is_ok(status))
    return status;
  started_ = true;
#define COUNT(call)                                                            \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return FailMutation(s);                                                  \
  } while (0)
  Value bytes, kind, origin, count;
  COUNT(values_.Constant(12, LOOM_SCALAR_TYPE_OFFSET, &bytes));
  COUNT(values_.Allocate(bytes, 4, &count));
  COUNT(values_.Constant(llama_rows_.phase == Phase::Decode,
                         LOOM_SCALAR_TYPE_INDEX, &kind));
  COUNT(values_.Constant(llama_rows_.control_origin, LOOM_SCALAR_TYPE_OFFSET,
                         &origin));
  COUNT(program_.Invoke(helper, {}, {kind, origin, llama_rows_.packet, count}));
#undef COUNT
  count_ = count;
  if (llama_rows_.count_control.valid())
    bindings_.push_back({llama_rows_.count_control, count_, {}});
  has_count_ = true;
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
                            Value *out_native, const std::string &variant,
                            bool expect_result) {
  if (!out_result || !out_native || failed_)
    return Error("missing result or failed invocation owner");
  if (has_llama_rows_ && (id == Id::Rope || id == Id::RotatePublishKV ||
                          id == Id::CausalAttention))
    return Error("packet attention requires its checked region adapter");
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
  if (inferred.has_result != expect_result)
    return Error(expect_result ? "operation has no producer result; use InvokeEffect"
                              : "producer operation requires Invoke");
  if (!graph_.definition(entry->operation))
    return Error("graph lacks operation definition");
  if (has_llama_rows_ && id != Id::OutputHead && id != Id::PublishPrefill &&
      id != Id::PublishDecode) {
    bool begin = false, count = false;
    for (const auto &input : inputs) {
      begin |=
          input.port == Port::RowBegin && input.value == llama_rows_.row_begin;
      count |=
          input.port == Port::RowCount && input.value == llama_rows_.row_count;
    }
    if (!begin || !count)
      return Error(
          "configured producer requires its exact logical row controls");
    if ((id == Id::GateUpPrefill || id == Id::GateUpDecode ||
         id == Id::DownPrefill || id == Id::DownDecode) &&
        entry->phase != llama_rows_.phase)
      return Error("FFN dispatch phase differs from configured row phase");
  }
  if (Reserved(existing_result) ||
      (inferred.has_result && Reserved(inferred.alias)))
    return Error(
        "operation output cannot overwrite a reserved Llama derivation");
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
      if (has_llama_rows_) {
        if (input.value != (input.port == Port::RowBegin
                                ? llama_rows_.row_begin
                                : llama_rows_.row_count))
          return Error("active-row controls differ from configured identities");
      } else if (!one_row_ ||
                 input.value !=
                     (input.port == Port::RowBegin ? row_begin_ : row_count_))
        return Error(
            "active subspan requires explicit matching one-row specialization");
    } else
      native_inputs.push_back(input);
  }
  for (const auto &input : native_inputs) {
    if (has_llama_rows_ && id == Id::OutputHead &&
        input.port == Port::Control) {
      if (!llama_rows_.count_control.valid() ||
          input.value != llama_rows_.count_control)
        return Error("configured head requires its reserved count control");
      continue;
    }
    auto *binding = Find(input.value);
    if (!binding)
      return Error("unbound logical input");
    auto status = CheckBuffer(binding->native);
    if (!loomc_status_is_ok(status))
      return status;
    if (has_llama_rows_ && entry->operation == OpCode::PublishToken &&
        input.port == Port::Control) {
      if (llama_rows_.control_origin != 0 ||
          std::get<PublishTokenAttrs>(attrs).phase != llama_rows_.phase)
        return Error(
            "publication requires configured phase and origin-zero packet");
      status = CheckPacket(binding->native);
      if (!loomc_status_is_ok(status))
        return status;
    }
  }
  if (one_row_) {
    for (const auto &input : inputs)
      if (input.port == Port::Input &&
          graph_.value(input.value)->tensorType()->dimension(0) != 1)
        return Error(
            "explicit one-row entrypoint requires the whole static row domain");
  }
  InvocationRequest request{graph_, attrs, inputs, inferred};
  if (has_llama_rows_) {
    const auto *input_type = request.Type(Port::Input);
    if (!input_type)
      input_type = request.Type(Port::TokenIds);
    if (input_type && input_type->dimension(0) != llama_rows_.row_capacity)
      return Error("input capacity differs from configured Llama rows");
  }
  const auto *selected = registry_.Resolve(id, request, variant, &error);
  if (!selected)
    return Error(error);
  auto count_status = StageCount();
  if (!loomc_status_is_ok(count_status))
    return count_status;
  std::vector<NativeOperand> native_operands;
  for (const auto &input : native_inputs) {
    const auto *binding = Find(input.value);
    native_operands.push_back(
        {input.port, input.value, binding->native, binding->parameter});
  }
  Value existing;
  bool has_output = false;
  if (inferred.has_result && inferred.access == Access::ReadWrite) {
    existing = Find(inferred.alias)->native;
    has_output = true;
  } else if (existing_result.valid() && Find(existing_result)) {
    existing = Find(existing_result)->native;
    has_output = true;
  }
  BuildContext context{request,      module_,         program_,
                       values_,      native_operands, control_,
                       has_control_, existing,        has_output};
  context.llama_rows = has_llama_rows_ ? &llama_rows_ : nullptr;
  context.qwen_epoch = has_qwen_epoch_ ? &qwen_epoch_ : nullptr;
  context.count_control = count_;
  context.staged = has_llama_rows_;
  Value native_output;
  auto status = selected->builder.build(*selected, context, &native_output);
  if (context.staged)
    started_ = true;
  if (!loomc_status_is_ok(status))
    return context.staged ? FailMutation(status) : status;

  status = inferred.has_result ? CheckBuffer(native_output) : loomc_ok_status();
  if (!loomc_status_is_ok(status))
    return context.staged ? FailMutation(status) : status;
  *out_result = inferred;
  *out_native = native_output;
  return loomc_ok_status();
}

loomc_status_t NativeEmissionSession::SpecializeOneRow(ValueId begin,
                                                       ValueId count) {
  if (failed_ || started_ || one_row_ || has_llama_rows_ || has_qwen_epoch_)
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
  const bool has_result = invocation.opcode() != OpCode::RotatePublishKV &&
                          invocation.opcode() != OpCode::PublishToken;
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

    }
  }
  if (has_result && !output.valid()) {
    shapes::Result result;
    std::string error;
    if (!shapes::InferResult(graph_, invocation.opcode(), invocation.attributes(),
                             inputs, &result, &error))
      return Error(error);
    if (!result.has_result || result.access != Access::ReadWrite || !result.alias.valid())
      return Error("missing existing output identity");
    output = result.alias;
  }
  shapes::Result inferred;
  Value native;
  auto status = Emit(id, invocation.attributes(), inputs, output, &inferred,
                     &native, variant, has_result);
  if (!loomc_status_is_ok(status))
    return status;
  if (inferred.has_result && inferred.access == Access::Write && !Find(output))
    bindings_.push_back({output, native, {}});
  last_existing_ = index;
  has_emitted_existing_ = true;
  return loomc_ok_status();
}

loomc_status_t InvocationProgram::InvokeEffect(
    Id id, const OperationAttrs &attrs,
    const std::vector<shapes::Operand> &inputs, const std::string &variant) {
  shapes::Result inferred;
  Value native;
  auto status = session_.Emit(id, attrs, inputs, {}, &inferred, &native, variant,
                              false);
  if (!loomc_status_is_ok(status))
    return status;
  auto invocation = graph_.operation(session_.registry_.Find(id)->operation);
  std::visit([&](const auto &a) { invocation.attributes(a); }, attrs);
  for (const auto &input : inputs)
    invocation.bind(input.port, input.value);
  std::string error;
  if (!invocation.append(&error))
    return session_.FailMutation(Error(error));
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
