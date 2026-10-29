// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/command/command_program_builder.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "loom/ops/command/ops.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/scf/ops.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/link.h"

namespace hrx::loomify_gguf {
namespace {
std::atomic<uint64_t> next_identity{1};
loomc_status_t Error(loomc_status_code_t code, const std::string &message) {
  return loomc_make_status(code, message.c_str());
}
std::vector<loom_value_id_t> BlockArguments(const loom_region_t *region) {
  const auto *block = loom_region_const_entry_block(region);
  if (!block->arg_count)
    return {};
  return {block->arg_ids, block->arg_ids + block->arg_count};
}
std::vector<loom_value_id_t> Copy(loom_value_slice_t values) {
  if (!values.count)
    return {};
  return {values.values, values.values + values.count};
}
bool ValidName(std::string_view name) {
  return !name.empty() && name.find('\0') == std::string_view::npos;
}
} // namespace

namespace detail {
struct AuthoringState {
  ~AuthoringState() {
    for (auto *provider : providers)
      loomc_module_release(provider);
    loomc_module_release(module);
    loomc_context_release(context);
    loomc_workspace_release(workspace);
  }
  loomc_status_t Mutable() const {
    if (!alive || terminal)
      return Error(LOOMC_STATUS_FAILED_PRECONDITION,
                   "module builder is finalized, destroyed or failed");
    return loomc_ok_status();
  }
  loomc_status_t Mutation(iree_status_t status) {
    if (!iree_status_is_ok(status))
      terminal = true;
    return loomc_status_from_iree(status);
  }
  loomc_status_t Find(std::string_view name, loom_symbol_id_t *out) const {
    auto status = Mutable();
    if (!loomc_status_is_ok(status))
      return status;
    if (!ValidName(name))
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "nonempty symbol name required");
    for (size_t i = 0; i < native->symbols.count; ++i) {
      auto id = native->symbols.entries[i].name_id;
      if (id == LOOM_STRING_ID_INVALID || id >= native->strings.count)
        continue;
      auto actual = loom_string_table_get(&native->strings, id);
      if (name.size() == actual.size &&
          std::memcmp(name.data(), actual.data, name.size()) == 0) {
        *out = static_cast<loom_symbol_id_t>(i);
        return loomc_ok_status();
      }
    }
    return Error(LOOMC_STATUS_NOT_FOUND,
                 "symbol not found: " + std::string(name));
  }
  loomc_status_t NewSymbol(std::string_view name, loom_symbol_id_t *out) {
    auto status = Mutable();
    if (!loomc_status_is_ok(status))
      return status;
    if (!ValidName(name))
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "nonempty symbol name required");
    loom_symbol_id_t existing = LOOM_SYMBOL_ID_INVALID;
    auto found = Find(name, &existing);
    if (loomc_status_is_ok(found))
      return Error(LOOMC_STATUS_ALREADY_EXISTS,
                   "symbol already exists: " + std::string(name));
    loomc_status_free(found);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    status = Mutation(loom_module_intern_string(
        native, iree_make_string_view(name.data(), name.size()), &name_id));
    if (!loomc_status_is_ok(status))
      return status;
    return Mutation(loom_module_add_symbol(native, name_id, out));
  }
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  loomc_module_t *module = nullptr;
  loom_module_t *native = nullptr;
  std::vector<loomc_module_t *> providers;
  uint64_t identity = next_identity.fetch_add(1);
  size_t unfinished = 0;
  bool alive = true;
  bool terminal = false;
};
} // namespace detail

struct CommandProgramBuilder::Impl {
  loomc_status_t Mutable() const {
    auto status = state->Mutable();
    if (!loomc_status_is_ok(status))
      return status;
    if (finished)
      return Error(LOOMC_STATUS_FAILED_PRECONDITION,
                   "command body is finished");
    return loomc_ok_status();
  }
  bool Valid(Value value) const {
    return value.owner_ == state->identity &&
           std::find(scopes.begin(), scopes.end(), value.scope_) !=
               scopes.end();
  }
  Value Wrap(loom_value_id_t id) const {
    Value value;
    value.owner_ = state->identity;
    value.scope_ = scopes.back();
    value.id_ = id;
    return value;
  }
  std::shared_ptr<detail::AuthoringState> state;
  loom_builder_t body = {};
  loom_builder_t schedule = {};
  std::vector<loom_value_id_t> inputs;
  size_t specialization_count = 0;
  loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
  std::vector<uint64_t> scopes{next_identity.fetch_add(1)};
  bool finished = false;
};

ModuleBuilder::ModuleBuilder(std::shared_ptr<detail::AuthoringState> state)
    : state_(std::move(state)) {}
ModuleBuilder::~ModuleBuilder() { state_->alive = false; }
CommandProgramBuilder::CommandProgramBuilder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CommandProgramBuilder::~CommandProgramBuilder() = default;

loomc_status_t ModuleBuilder::Create(loomc_context_t *context,
                                     loomc_workspace_t *workspace,
                                     const loomc_module_t *definitions,
                                     std::unique_ptr<ModuleBuilder> *out) {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing owner output");
  out->reset();
  if (!context || !workspace || !definitions)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "context, workspace and definitions required");
  auto state = std::make_shared<detail::AuthoringState>();
  auto status = loomc_module_clone(definitions, workspace,
                                   loomc_allocator_system(), &state->module);
  if (!loomc_status_is_ok(status))
    return status;
  loomc_result_t *result = nullptr;
  status = loomc_module_verify(state->module, nullptr, loomc_allocator_system(),
                               &result);
  if (!loomc_status_is_ok(status))
    return status;
  const bool valid = result && loomc_result_succeeded(result);
  loomc_result_release(result);
  if (!valid)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "input definitions failed Loom verification");
  state->context = context;
  loomc_context_retain(context);
  state->workspace = workspace;
  loomc_workspace_retain(workspace);
  state->native = loomc_module_get_mutable_interop_view(state->module).module;
  if (!state->native)
    return Error(LOOMC_STATUS_FAILED_PRECONDITION, "no mutable native module");
  out->reset(new ModuleBuilder(std::move(state)));
  return loomc_ok_status();
}
Type ModuleBuilder::Scalar(loom_scalar_type_t type) const {
  Type out;
  if (state_->alive && !state_->terminal && loom_scalar_type_is_valid(type)) {
    out.owner_ = state_->identity;
    out.type_ = loom_type_scalar(type);
  }
  return out;
}
Type ModuleBuilder::Buffer() const {
  Type out;
  if (state_->alive && !state_->terminal) {
    out.owner_ = state_->identity;
    out.type_ = loom_type_buffer();
  }
  return out;
}
loomc_status_t ModuleBuilder::Kernel(std::string_view name,
                                     KernelRef *out) const {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing kernel output");
  *out = {};
  loom_symbol_id_t id;
  auto status = state_->Find(name, &id);
  if (!loomc_status_is_ok(status))
    return status;
  const auto *op = state_->native->symbols.entries[id].defining_op;
  if (!op || (!loom_kernel_def_isa(op) && !loom_kernel_decl_isa(op)))
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "symbol is not a configured kernel: " + std::string(name));
  out->owner_ = state_->identity;
  out->symbol_ = id;
  return loomc_ok_status();
}
loomc_status_t ModuleBuilder::KernelPipeline(std::string_view name,
                                             KernelPipelineRef *out) const {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing pipeline output");
  *out = {};
  loom_symbol_id_t id;
  auto status = state_->Find(name, &id);
  if (!loomc_status_is_ok(status))
    return status;
  const auto *op = state_->native->symbols.entries[id].defining_op;
  if (!op || (!loom_pipeline_def_isa(op) && !loom_pipeline_decl_isa(op)))
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "symbol is not a pipeline: " + std::string(name));
  auto scope = loom_pipeline_def_isa(op) ? loom_pipeline_def_scope(op)
                                         : loom_pipeline_decl_scope(op);
  if (scope != LOOM_PIPELINE_SCOPE_KERNEL)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "only kernel-scoped pipelines can be issued by commands");
  out->owner_ = state_->identity;
  out->symbol_ = id;
  return loomc_ok_status();
}
loomc_status_t ModuleBuilder::Program(std::string_view name,
                                      ProgramRef *out) const {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing program output");
  *out = {};
  loom_symbol_id_t id;
  auto status = state_->Find(name, &id);
  if (!loomc_status_is_ok(status))
    return status;
  const auto *op = state_->native->symbols.entries[id].defining_op;
  if (!op ||
      (!loom_command_program_def_isa(op) && !loom_command_program_decl_isa(op)))
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "symbol is not a command program: " + std::string(name));
  out->owner_ = state_->identity;
  out->symbol_ = id;
  return loomc_ok_status();
}
loomc_status_t ModuleBuilder::Root(std::string_view name, RootRef *out) const {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing root output");
  *out = {};
  loom_symbol_id_t id;
  auto status = state_->Find(name, &id);
  if (!loomc_status_is_ok(status))
    return status;
  if (!state_->native->symbols.entries[id].defining_op)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "root has no definition or declaration");
  out->owner_ = state_->identity;
  out->symbol_ = id;
  return loomc_ok_status();
}
loomc_status_t ModuleBuilder::DeclareKernel(std::string_view name,
                                            const std::vector<Type> &workloads,
                                            const std::vector<Type> &arguments,
                                            KernelRef *out) {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing kernel output");
  *out = {};
  auto status = state_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (workloads.size() > UINT16_MAX ||
      arguments.size() > UINT16_MAX - workloads.size())
    return Error(LOOMC_STATUS_RESOURCE_EXHAUSTED, "too many kernel arguments");
  std::vector<loom_type_t> w, a;
  for (const auto &type : workloads) {
    if (type.owner_ != state_->identity)
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "foreign or invalid workload type");
    w.push_back(type.type_);
  }
  for (const auto &type : arguments) {
    if (type.owner_ != state_->identity)
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "foreign or invalid argument type");
    a.push_back(type.type_);
  }
  loom_symbol_id_t id;
  status = state_->NewSymbol(name, &id);
  if (!loomc_status_is_ok(status))
    return status;
  loom_builder_t top;
  loom_builder_initialize(state_->native, &state_->native->arena,
                          loom_module_block(state_->native), &top);
  loom_op_t *op = nullptr;
  status = state_->Mutation(loom_kernel_decl_build(
      &top, 0, 0, loom_symbol_ref_null(), LOOM_STRING_ID_INVALID, 0,
      loom_symbol_ref_t{0, id}, w.data(), w.size(), a.data(), a.size(), nullptr,
      0, LOOM_LOCATION_UNKNOWN, &op));
  if (!loomc_status_is_ok(status))
    return status;
  out->owner_ = state_->identity;
  out->symbol_ = id;
  return loomc_ok_status();
}
loomc_status_t ModuleBuilder::DefineProgram(
    std::string_view name,
    const std::vector<loom_scalar_type_t> &specialization_types,
    size_t binding_count, std::unique_ptr<CommandProgramBuilder> *out) {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing program output");
  out->reset();
  auto status = state_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (binding_count > UINT16_MAX ||
      specialization_types.size() > UINT16_MAX - binding_count)
    return Error(LOOMC_STATUS_RESOURCE_EXHAUSTED, "too many program arguments");
  std::vector<loom_type_t> types;
  for (auto type : specialization_types) {
    if (!loom_scalar_type_is_valid(type))
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "invalid specialization type");
    types.push_back(loom_type_scalar(type));
  }
  auto impl = std::make_unique<CommandProgramBuilder::Impl>();
  impl->state = state_;
  status = state_->NewSymbol(name, &impl->symbol);
  if (!loomc_status_is_ok(status))
    return status;
  loom_builder_t top;
  loom_builder_initialize(state_->native, &state_->native->arena,
                          loom_module_block(state_->native), &top);
  std::vector<loom_type_t> bindings(binding_count, loom_type_buffer());
  loom_op_t *op = nullptr;
  status = state_->Mutation(loom_command_program_def_build(
      &top, LOOM_COMMAND_PROGRAM_DEF_BUILD_FLAG_HAS_VISIBILITY,
      LOOM_COMMAND_VISIBILITY_PUBLIC, 0, loom_symbol_ref_null(),
      {0, impl->symbol}, types.data(), types.size(), bindings.data(),
      bindings.size(), nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
  if (!loomc_status_is_ok(status))
    return status;
  ++state_->unfinished;
  impl->body = top;
  loom_builder_enter_region(&impl->body, op, loom_command_program_def_body(op));
  impl->inputs = BlockArguments(loom_command_program_def_body(op));
  impl->specialization_count = types.size();
  loom_op_t *serial = nullptr;
  status = state_->Mutation(
      loom_command_serial_build(&impl->body, LOOM_LOCATION_UNKNOWN, &serial));
  if (!loomc_status_is_ok(status))
    return status;
  impl->schedule = impl->body;
  loom_builder_enter_region(&impl->schedule, serial,
                            loom_command_serial_body(serial));
  out->reset(new CommandProgramBuilder(std::move(impl)));
  return loomc_ok_status();
}
loomc_status_t ModuleBuilder::AddProvider(const loomc_module_t *provider) {
  auto status = state_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (!provider)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing provider");
  loomc_module_t *copy = nullptr;
  status = loomc_module_clone(provider, state_->workspace,
                              loomc_allocator_system(), &copy);
  if (!loomc_status_is_ok(status))
    return status;
  state_->providers.push_back(copy);
  return loomc_ok_status();
}
loomc_status_t ModuleBuilder::BuildNative(
    const std::function<iree_status_t(loom_module_t *, loom_builder_t *)>
        &build) {
  auto status = state_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (!build)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "missing native definition callback");
  loom_builder_t top;
  loom_builder_initialize(state_->native, &state_->native->arena,
                          loom_module_block(state_->native), &top);
  return state_->Mutation(build(state_->native, &top));
}
loomc_status_t ModuleBuilder::LinkAndVerify(const std::vector<RootRef> &roots,
                                            loomc_module_t **out_module,
                                            loomc_result_t **out_result) {
  if (out_module)
    *out_module = nullptr;
  if (out_result)
    *out_result = nullptr;
  if (!out_module || !out_result)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "module and result outputs required");
  auto status = state_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (state_->unfinished)
    return Error(LOOMC_STATUS_FAILED_PRECONDITION,
                 "unfinished command definitions block module finalization");
  if (roots.empty())
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "at least one root required");
  std::vector<loomc_string_view_t> names;
  for (auto root : roots) {
    if (root.owner_ != state_->identity)
      return Error(LOOMC_STATUS_INVALID_ARGUMENT, "foreign or invalid root");
    auto name = loom_string_table_get(
        &state_->native->strings,
        state_->native->symbols.entries[root.symbol_].name_id);
    names.push_back(loomc_make_string_view(name.data, name.size));
  }
  loomc_result_t *verified = nullptr;
  status = loomc_module_verify(state_->module, nullptr,
                               loomc_allocator_system(), &verified);
  if (!loomc_status_is_ok(status)) {
    state_->terminal = true;
    return status;
  }
  if (!verified || !loomc_result_succeeded(verified)) {
    state_->terminal = true;
    *out_result = verified;
    return loomc_ok_status();
  }
  loomc_result_release(verified);
  loomc_linker_t *linker = nullptr;
  status = loomc_linker_create(state_->context, nullptr,
                               loomc_allocator_system(), &linker);
  if (!loomc_status_is_ok(status)) {
    state_->terminal = true;
    return status;
  }
  std::vector<loomc_link_module_provider_t> providers;
  loomc_link_module_provider_t primary = {};
  primary.module = state_->module;
  primary.role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
  providers.push_back(primary);
  for (auto *provider : state_->providers) {
    loomc_link_module_provider_t entry = {};
    entry.module = provider;
    entry.role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
    providers.push_back(entry);
  }
  loomc_link_options_t options = {};
  options.structure_size = sizeof(options);
  options.mode = LOOMC_LINK_MODE_LINK;
  options.module_providers = providers.data();
  options.module_provider_count = providers.size();
  options.root_symbols = names.data();
  options.root_symbol_count = names.size();
  state_->terminal = true;
  status = loomc_link_module(linker, state_->workspace, &options, out_module,
                             out_result);
  loomc_linker_release(linker);
  return status;
}
ProgramRef CommandProgramBuilder::Ref() const {
  ProgramRef out;
  if (impl_->state->alive && !impl_->state->terminal) {
    out.owner_ = impl_->state->identity;
    out.symbol_ = impl_->symbol;
  }
  return out;
}
Value CommandProgramBuilder::Specialization(size_t index) const {
  if (!impl_->state->alive || impl_->state->terminal || impl_->finished ||
      index >= impl_->specialization_count)
    return {};
  auto out = impl_->Wrap(impl_->inputs[index]);
  out.scope_ = impl_->scopes.front();
  return out;
}
Value CommandProgramBuilder::Binding(size_t index) const {
  if (!impl_->state->alive || impl_->state->terminal || impl_->finished ||
      index >= impl_->inputs.size() - impl_->specialization_count)
    return {};
  auto out = impl_->Wrap(impl_->inputs[impl_->specialization_count + index]);
  out.scope_ = impl_->scopes.front();
  return out;
}
loomc_status_t CommandProgramBuilder::IndexConstant(int64_t value, Value *out) {
  if (!out)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing value output");
  *out = {};
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  loom_op_t *op = nullptr;
  status = impl_->state->Mutation(loom_index_constant_build(
      &impl_->schedule, loom_attr_i64(value),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
  if (loomc_status_is_ok(status))
    *out = impl_->Wrap(loom_index_constant_result(op));
  return status;
}
loomc_status_t CommandProgramBuilder::InvokeSymbol(
    loom_symbol_id_t symbol, const std::vector<loom_value_id_t> &expected_first,
    const std::vector<loom_value_id_t> &expected_second,
    const std::vector<Value> &first, const std::vector<Value> &second,
    bool remap_first, bool program_call) {
  if (first.size() != expected_first.size() ||
      second.size() != expected_second.size())
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "callee operand group count mismatch");
  std::vector<loom_value_id_t> a, b;
  for (auto value : first) {
    if (!impl_->Valid(value))
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "foreign, invalid or out-of-scope first operand");
    a.push_back(value.id_);
  }
  for (auto value : second) {
    if (!impl_->Valid(value))
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "foreign, invalid or out-of-scope second operand");
    b.push_back(value.id_);
  }
  loom_type_value_remap_t first_remap = {};
  first_remap.source_values = expected_first.data();
  first_remap.target_values = a.data();
  first_remap.count = a.size();
  loom_type_value_remap_t second_remap = {};
  second_remap.source_values = expected_second.data();
  second_remap.target_values = b.data();
  second_remap.count = b.size();
  second_remap.next = remap_first ? &first_remap : nullptr;
  for (int group = 0; group < 2; ++group) {
    const auto &actual = group ? b : a;
    const auto &expected = group ? expected_second : expected_first;
    const auto *remap = group ? &second_remap : &first_remap;
    for (size_t i = 0; i < actual.size(); ++i) {
      bool matches = false;
      auto native_status = loom_function_call_argument_type_matches(
          impl_->state->native,
          loom_module_value_type(impl_->state->native, actual[i]),
          loom_module_value_type(impl_->state->native, expected[i]), remap,
          LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_ALLOW_BUFFER_MATERIALIZATION,
          &matches);
      if (!iree_status_is_ok(native_status))
        return loomc_status_from_iree(native_status);
      if (!matches)
        return Error(
            LOOMC_STATUS_INVALID_ARGUMENT,
            std::string(group ? "argument" : "specialization/workload") +
                " type mismatch at " + std::to_string(i));
    }
  }
  loom_op_t *op = nullptr;
  if (program_call)
    return impl_->state->Mutation(loom_command_program_launch_build(
        &impl_->schedule, {0, symbol}, a.data(), a.size(), b.data(), b.size(),
        LOOM_LOCATION_UNKNOWN, &op));
  return impl_->state->Mutation(loom_kernel_launch_build(
      &impl_->schedule, {0, symbol}, a.data(), a.size(), b.data(), b.size(),
      LOOM_LOCATION_UNKNOWN, &op));
}
loomc_status_t
CommandProgramBuilder::Invoke(KernelRef kernel,
                              const std::vector<Value> &workloads,
                              const std::vector<Value> &arguments) {
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (kernel.owner_ != impl_->state->identity)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "foreign or invalid kernel reference");
  const auto *op =
      impl_->state->native->symbols.entries[kernel.symbol_].defining_op;
  std::vector<loom_value_id_t> w, a;
  if (loom_kernel_def_isa(op)) {
    w = BlockArguments(loom_kernel_def_config(op));
    a = BlockArguments(loom_kernel_def_body(op));
  } else {
    w = Copy(loom_kernel_decl_workloads(op));
    a = Copy(loom_kernel_decl_args(op));
  }
  return InvokeSymbol(kernel.symbol_, w, a, workloads, arguments, false, false);
}
loomc_status_t
CommandProgramBuilder::Invoke(KernelPipelineRef pipeline,
                              const std::vector<Value> &specializations,
                              const std::vector<Value> &arguments) {
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (pipeline.owner_ != impl_->state->identity)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "foreign or invalid kernel-pipeline reference");
  const auto *op =
      impl_->state->native->symbols.entries[pipeline.symbol_].defining_op;
  std::vector<loom_value_id_t> args;
  size_t count;
  if (loom_pipeline_def_isa(op)) {
    args = BlockArguments(loom_pipeline_def_body(op));
    count = loom_pipeline_def_specialization_count(op);
  } else {
    args = Copy(loom_pipeline_decl_args(op));
    count = loom_pipeline_decl_specialization_count(op);
  }
  return InvokeSymbol(pipeline.symbol_, {args.begin(), args.begin() + count},
                      {args.begin() + count, args.end()}, specializations,
                      arguments, true, false);
}
loomc_status_t
CommandProgramBuilder::Call(ProgramRef program,
                            const std::vector<Value> &specializations,
                            const std::vector<Value> &bindings) {
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (program.owner_ != impl_->state->identity)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "foreign or invalid program reference");
  if (program.symbol_ == impl_->symbol)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "direct recursive command call rejected");
  const auto *op =
      impl_->state->native->symbols.entries[program.symbol_].defining_op;
  std::vector<loom_value_id_t> args;
  size_t count;
  if (loom_command_program_def_isa(op)) {
    args = BlockArguments(loom_command_program_def_body(op));
    count = loom_command_program_def_specialization_count(op);
  } else {
    args = Copy(loom_command_program_decl_args(op));
    count = loom_command_program_decl_specialization_count(op);
  }
  return InvokeSymbol(program.symbol_, {args.begin(), args.begin() + count},
                      {args.begin() + count, args.end()}, specializations,
                      bindings, true, true);
}
loomc_status_t CommandProgramBuilder::BuildNative(
    const std::vector<Value> &operands,
    const std::function<iree_status_t(loom_module_t *, loom_builder_t *,
                                      const std::vector<loom_value_id_t> &,
                                      std::vector<loom_value_id_t> *)> &build,
    std::vector<Value> *results) {
  if (!results || !build)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "callback and results required");
  results->clear();
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  std::vector<loom_value_id_t> ids, outputs;
  for (auto value : operands) {
    if (!impl_->Valid(value))
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "foreign or out-of-scope native operand");
    ids.push_back(value.id_);
  }
  // A callback may enter nested regions using its local insertion point; that
  // must not redirect subsequent ordinary command authoring or scope checks.
  auto local_builder = impl_->schedule;
  status = impl_->state->Mutation(
      build(impl_->state->native, &local_builder, ids, &outputs));
  if (!loomc_status_is_ok(status))
    return status;
  for (auto id : outputs) {
    bool found = false;
    const loom_op_t *op;
    loom_block_for_each_op(impl_->schedule.ip.block, op) {
      for (size_t i = 0; i < op->result_count; ++i)
        if (loom_op_const_results(op)[i] == id)
          found = true;
    }
    if (!found) {
      impl_->state->terminal = true;
      results->clear();
      return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                   "native result is not defined in current scope");
    }
    results->push_back(impl_->Wrap(id));
  }
  return loomc_ok_status();
}
loomc_status_t CommandProgramBuilder::EnterRegion(loom_op_t *op,
                                                  loom_region_t *region,
                                                  const RegionBody &body,
                                                  bool scf_yield) {
  auto saved = impl_->schedule;
  loom_builder_enter_region(&impl_->schedule, op, region);
  impl_->scopes.push_back(next_identity.fetch_add(1));
  auto status = body(*this);
  if (loomc_status_is_ok(status)) {
    loom_op_t *end = nullptr;
    status = impl_->state->Mutation(
        scf_yield ? loom_scf_yield_build(&impl_->schedule, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &end)
                  : loom_command_yield_build(&impl_->schedule,
                                             LOOM_LOCATION_UNKNOWN, &end));
  }
  if (!loomc_status_is_ok(status))
    impl_->state->terminal = true;
  impl_->scopes.pop_back();
  impl_->schedule = saved;
  return status;
}
loomc_status_t CommandProgramBuilder::Serial(const RegionBody &body) {
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (!body)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing serial body");
  loom_op_t *op = nullptr;
  status = impl_->state->Mutation(
      loom_command_serial_build(&impl_->schedule, LOOM_LOCATION_UNKNOWN, &op));
  if (!loomc_status_is_ok(status))
    return status;
  return EnterRegion(op, loom_command_serial_body(op), body, false);
}
loomc_status_t CommandProgramBuilder::Concurrent(const RegionBody &body) {
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (!body)
    return Error(LOOMC_STATUS_INVALID_ARGUMENT, "missing concurrent body");
  loom_op_t *op = nullptr;
  status = impl_->state->Mutation(loom_command_concurrent_build(
      &impl_->schedule, LOOM_LOCATION_UNKNOWN, &op));
  if (!loomc_status_is_ok(status))
    return status;
  return EnterRegion(op, loom_command_concurrent_body(op), body, false);
}
loomc_status_t CommandProgramBuilder::If(Value condition,
                                         const RegionBody &then_body,
                                         const RegionBody &else_body) {
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (!then_body || !impl_->Valid(condition) ||
      !loom_type_equal(
          loom_module_value_type(impl_->state->native, condition.id_),
          loom_type_scalar(LOOM_SCALAR_TYPE_I1)))
    return Error(LOOMC_STATUS_INVALID_ARGUMENT,
                 "if requires scoped i1 condition and then body");
  loom_op_t *op = nullptr;
  status = impl_->state->Mutation(loom_scf_if_build(
      &impl_->schedule, else_body ? LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION : 0,
      condition.id_, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));
  if (!loomc_status_is_ok(status))
    return status;
  status = EnterRegion(op, loom_scf_if_then_region(op), then_body, true);
  if (!loomc_status_is_ok(status) || !else_body)
    return status;
  return EnterRegion(op, loom_scf_if_else_region(op), else_body, true);
}
loomc_status_t CommandProgramBuilder::Finish() {
  auto status = impl_->Mutable();
  if (!loomc_status_is_ok(status))
    return status;
  if (impl_->scopes.size() != 1)
    return Error(LOOMC_STATUS_FAILED_PRECONDITION,
                 "cannot finish inside a nested region");
  loom_op_t *end = nullptr;
  status = impl_->state->Mutation(
      loom_command_yield_build(&impl_->schedule, LOOM_LOCATION_UNKNOWN, &end));
  if (!loomc_status_is_ok(status))
    return status;
  status = impl_->state->Mutation(
      loom_command_return_build(&impl_->body, LOOM_LOCATION_UNKNOWN, &end));
  if (!loomc_status_is_ok(status))
    return status;
  impl_->finished = true;
  --impl_->state->unfinished;
  return loomc_ok_status();
}
} // namespace hrx::loomify_gguf
