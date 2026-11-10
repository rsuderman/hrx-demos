// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/testing/prepare_vm.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loomc/target/vm.h"
#include <cstring>
#include <memory>

namespace hrx::loomify_gguf::testing {
namespace {
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
using Signature = iree_vm_module_signature_type_t;
constexpr Signature kRequire[] = {{IREE_VM_SCALAR_TYPE_I32, 0},
                                  {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
constexpr Signature kMembersIn[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
constexpr Signature kMembersOut[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
constexpr Signature kParameter[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
constexpr Signature kStageIn[] = {{IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
                                  {IREE_VM_SCALAR_TYPE_I64, 0}};
constexpr Signature kStageOut[] = {{IREE_VM_SCALAR_TYPE_I32, 0}};
const iree_vm_module_callable_type_declaration_t kInputTypes[] = {
    {{{kRequire, 2, 1, 1, 0}, {nullptr, 0, 0, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0}};
const iree_vm_module_callable_type_declaration_t kJsonTypes[] = {
    {{{kMembersIn, 1, 0, 1, 0}, {kMembersOut, 3, 2, 1, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0}};
const iree_vm_module_callable_type_declaration_t kPrepareTypes[] = {
    // Native callable types are canonicalized by signature arity, independently
    // of the alphabetically ordered export names and function ordinals.
    {{{kStageIn, 2, 1, 1, 0}, {kStageOut, 1, 1, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kParameter, 6, 3, 3, 0}, {nullptr, 0, 0, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0}};
const iree_vm_module_export_declaration_t kInputExports[] = {
    {IREE_SVL("require"), 0, 0, 0}};
const iree_vm_module_export_declaration_t kJsonExports[] = {
    {IREE_SVL("members"), 0, 0, 0}};
const iree_vm_module_export_declaration_t kPrepareExports[] = {
    {IREE_SVL("parameter"), 1, 0, 0}, {IREE_SVL("stage"), 0, 1, 0}};
struct Native {
  iree_vm_module_t module{};
  iree_vm_module_descriptor_t descriptor{};
  const iree_vm_module_callable_type_declaration_t *types = nullptr;
  const iree_vm_module_export_declaration_t *exports = nullptr;
  const PreparationRequest *request = nullptr;
  PreparationObservation *observations = nullptr;
  int kind = 0;
};
std::string StringArg(const iree_vm_call_packet_t &call, unsigned ordinal) {
  iree_vm_ref_t ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(&call, ordinal, &ref);
  auto *buffer = static_cast<iree_vm_buffer_t *>(ref.object);
  const auto length = iree_vm_buffer_length(buffer);
  return length ? std::string(static_cast<const char *>(
                                  iree_vm_buffer_const_data(buffer)),
                              length)
                : std::string{};
}
iree_status_t Start(iree_vm_module_t *module,
                    const iree_vm_module_function_start_params_t *params,
                    iree_vm_execution_outcome_t *outcome) {
  auto &self = *reinterpret_cast<Native *>(module);
  const auto &call = params->call;
  auto value = [&](unsigned i) {
    return iree_vm_call_value_argument_load(&call, i);
  };
  if (self.kind == 0) {
    const int32_t condition = static_cast<int32_t>(value(0));
    self.observations->requirements.push_back(condition);
    if (!condition)
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "prepare requirement rejected: %s",
                              StringArg(call, 0).c_str());
  } else if (self.kind == 1) {
    iree_vm_ref_t fields = iree_vm_ref_null();
    iree_vm_call_ref_argument_load_move(&call, 0, &fields);
    iree_vm_call_ref_result_store_move(&call, 0, &fields);
    iree_vm_call_value_result_store(&call, 0, self.request->options_type);
    iree_vm_call_value_result_store(&call, 1, self.request->options_fields);
  } else if (params->function_ordinal == 0) {
    self.observations->parameters.push_back(
        {static_cast<int32_t>(value(0)), static_cast<int32_t>(value(1)),
         static_cast<int32_t>(value(2)), StringArg(call, 0), StringArg(call, 1),
         StringArg(call, 2)});
  } else {
    const int32_t id = 101 + self.observations->stages.size();
    self.observations->stages.push_back(
        {StringArg(call, 0), static_cast<int64_t>(value(0)), id});
    iree_vm_call_value_result_store(&call, 0, id);
  }
  *outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}
void QueryExport(const iree_vm_module_t *module, iree_host_size_t i,
                 iree_vm_module_export_declaration_t *out) {
  *out = reinterpret_cast<const Native *>(module)->exports[i];
}
void QueryType(const iree_vm_module_t *module, iree_host_size_t i,
               iree_vm_module_callable_type_declaration_t *out) {
  *out = reinterpret_cast<const Native *>(module)->types[i];
}
void QueryGroup(const iree_vm_module_t *, iree_host_size_t,
                iree_vm_module_import_group_t *) {
  IREE_CHECK_UNREACHABLE("no imports");
}
void QueryImport(const iree_vm_module_t *, iree_host_size_t,
                 iree_vm_module_import_declaration_t *) {
  IREE_CHECK_UNREACHABLE("no imports");
}
void Destroy(iree_vm_module_t *) {}
const iree_vm_module_vtable_t kVtable = {
    sizeof(kVtable),
    IREE_VM_MODULE_ABI_VERSION_0,
    Destroy,
    Start,
    iree_vm_module_function_resume_unreachable,
    nullptr,
    nullptr,
    nullptr,
    QueryGroup,
    QueryImport,
    QueryExport,
    QueryType,
    iree_vm_module_query_presentation_none,
    iree_vm_module_metadata_by_ordinal_none};
bool NativeStatus(iree_status_t status, std::string *error) {
  if (iree_status_is_ok(status))
    return true;
  char *text = nullptr;
  iree_host_size_t length = 0;
  const auto allocator = iree_allocator_system();
  iree_status_to_string(status, &allocator, &text, &length);
  if (error)
    error->assign(text, length);
  iree_allocator_free(iree_allocator_system(), text);
  iree_status_ignore(status);
  return false;
}
bool LoomStatus(loomc_status_t status, std::string *error) {
  if (loomc_status_is_ok(status)) {
    loomc_status_free(status);
    return true;
  }
  auto text = loomc_status_message(status);
  if (error)
    error->assign(text.data, text.size);
  loomc_status_free(status);
  return false;
}
bool Result(loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (error) {
    *error = "preparation compilation failed";
    if (result)
      for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
        const auto *d = loomc_result_diagnostic_at(result, i);
        error->append("\n").append(d->message.data, d->message.size);
      }
  }
  return false;
}
struct Runtime {
  iree_vm_environment_t *environment = nullptr;
  iree_vm_module_t *bytecode = nullptr;
  iree_vm_program_t *program = nullptr;
  iree_vm_process_t *process = nullptr;
  iree_vm_invocation_t *invocation = nullptr;
  Native providers[3];
  int initialized = 0;
  std::array<iree_vm_variant_t, 10> arguments{};
  std::array<iree_vm_variant_t, 9> results{};
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage{};
  ~Runtime() {
    iree_vm_variant_span_reset({results.data(), results.size()});
    iree_vm_variant_span_reset({arguments.data(), arguments.size()});
    iree_vm_process_release(process);
    if (invocation)
      iree_vm_invocation_deinitialize(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(bytecode);
    for (int i = 0; i < initialized; ++i)
      iree_vm_module_release(&providers[i].module);
    iree_vm_environment_free(environment);
  }
};
} // namespace
bool RunPreparation(loomc_target_environment_t *environment,
                    loomc_context_t *context, loomc_workspace_t *workspace,
                    loomc_module_t *module, const PreparationRequest &request,
                    PreparationObservation *out, std::string *error) {
  if (!environment || !context || !workspace || !module || !out) {
    if (error)
      *error = "preparation VM needs context/workspace/module/output";
    return false;
  }
  *out = {};
  loomc_compiler_t *raw_compiler = nullptr;
  if (!LoomStatus(loomc_compiler_create(context, nullptr,
                                        loomc_allocator_system(),
                                        &raw_compiler),
                  error))
    return false;
  Owner<loomc_compiler_t, loomc_compiler_release> compiler(
      raw_compiler, loomc_compiler_release);
  // As in Serve's loom_serve_jit_compile_vm, select a concrete VM profile.
  // An emitter format alone does not target otherwise targetless host
  // functions.
  loomc_target_profile_t *raw_profile = nullptr;
  if (!LoomStatus(loomc_target_profile_select(
                      environment, loomc_make_cstring_view("vm:core"),
                      loomc_allocator_system(), &raw_profile),
                  error))
    return false;
  Owner<loomc_target_profile_t, loomc_target_profile_release> profile(
      raw_profile, loomc_target_profile_release);
  const auto root = loomc_make_cstring_view("prepare");
  loomc_emit_options_t emit{};
  emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
  emit.structure_size = sizeof(emit);
  emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
  loomc_compile_artifact_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
  options.structure_size = sizeof(options);
  options.roots = &root;
  options.root_count = 1;
  options.target_profile = profile.get();
  options.emit_options = &emit;
  loomc_result_t *raw_result = nullptr;
  auto status =
      loomc_compile_artifact(compiler.get(), workspace, nullptr, module,
                             &options, loomc_allocator_system(), &raw_result);
  Owner<loomc_result_t, loomc_result_release> compiled(raw_result,
                                                       loomc_result_release);
  if (!LoomStatus(status, error) || !Result(compiled.get(), error))
    return false;
  const loomc_artifact_t *artifact = nullptr;
  for (size_t i = 0; i < loomc_result_artifact_count(compiled.get()); ++i) {
    auto *candidate = loomc_result_artifact_at(compiled.get(), i);
    if (candidate->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      artifact = candidate;
      break;
    }
  }
  if (!artifact) {
    if (error)
      *error = "no VM executable";
    return false;
  }
  // VM artifacts may be segmented; Serve likewise clones their byte sequence.
  // Keep this image alive until the borrowing bytecode module is destroyed.
  struct Image {
    loomc_byte_span_t bytes{};
    ~Image() {
      loomc_allocator_free(loomc_allocator_system(),
                           const_cast<uint8_t *>(bytes.data));
    }
  } image;
  if (!LoomStatus(loomc_byte_sequence_clone(artifact->contents,
                                            loomc_allocator_system(),
                                            &image.bytes),
                  error))
    return false;
  const auto artifact_bytes = image.bytes;
  Runtime runtime;
  auto ok = [&](iree_status_t s) { return NativeStatus(s, error); };
  if (!ok(iree_vm_environment_allocate(iree_allocator_system(),
                                       &runtime.environment)))
    return false;
  iree_vm_ref_types_t types{};
  if (!ok(iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                        runtime.environment, IREE_SV("vm")),
                                    &types)))
    return false;
  if (!ok(iree_vm_bytecode_module_create(
          runtime.environment, IREE_SV("compiled"),
          {iree_make_const_byte_span(artifact_bytes.data,
                                     artifact_bytes.data_length),
           iree_allocator_null()},
          iree_allocator_system(), &runtime.bytecode)))
    return false;
  const char *names[] = {"input", "json", "prepare"};
  const iree_vm_module_callable_type_declaration_t *signatures[] = {
      kInputTypes, kJsonTypes, kPrepareTypes};
  const iree_vm_module_export_declaration_t *exports[] = {
      kInputExports, kJsonExports, kPrepareExports};
  const unsigned counts[] = {1, 1, 2};
  const unsigned values[] = {1, 2, 5}, refs[] = {1, 2, 4};
  for (int i = 0; i < 3; ++i) {
    auto &p = runtime.providers[i];
    p.kind = i;
    p.request = &request;
    p.observations = out;
    p.types = signatures[i];
    p.exports = exports[i];
    p.descriptor = {
        iree_make_cstring_view(names[i]),
        IREE_VM_MODULE_FLAG_LINKABLE,
        {&types.buffer, 1},
        {counts[i], counts[i], 0, 0, counts[i], 0, {values[i], refs[i], 0}},
        0};
    if (!ok(iree_vm_module_initialize(&kVtable, &p.descriptor, &p.module)))
      return false;
    ++runtime.initialized;
  }
  iree_vm_module_t *libraries[] = {&runtime.providers[0].module,
                                   &runtime.providers[1].module,
                                   &runtime.providers[2].module};
  if (!ok(iree_vm_program_create(
          {runtime.bytecode, iree_vm_module_span_from_array(libraries)},
          iree_allocator_system(), &runtime.program)) ||
      !ok(iree_vm_invocation_initialize(
          iree_make_byte_span(runtime.storage.data(), runtime.storage.size()),
          &runtime.invocation)) ||
      !ok(iree_vm_process_create(runtime.program, runtime.invocation,
                                 iree_vm_variant_span_empty(),
                                 iree_allocator_system(), &runtime.process)))
    return false;
  auto &args = runtime.arguments;
  args[0] = iree_vm_variant_from_i64(request.prefill);
  args[1] = iree_vm_variant_from_i64(request.context);
  args[2] = iree_vm_variant_from_i64(request.pool);
  args[3] = iree_vm_variant_from_i64(request.rows);
  args[4] = iree_vm_variant_from_i64(request.pins);
  args[5] = iree_vm_variant_from_i32(request.draft_depth);
  auto buffer = [&](const void *data, size_t size, iree_vm_variant_t *variant) {
    iree_vm_buffer_t *b = nullptr;
    if (!ok(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                 iree_make_const_byte_span(data, size), 8,
                                 iree_allocator_system(), &b)))
      return false;
    *variant = iree_vm_buffer_variant_from_ptr_move(&types, &b);
    return true;
  };
  if (!buffer(request.shapes.data(),
              request.shapes.size() * sizeof(request.shapes[0]), &args[6]) ||
      !buffer(request.weights.data(), request.weights.size(), &args[8]) ||
      !buffer(nullptr, 0, &args[9]))
    return false;
  args[7] = iree_vm_variant_from_i64(request.shapes.size());
  // Artifact compilation materializes the explicitly selected Loom root; the
  // emitted VM export may carry its native function-version suffix. Inspect
  // the actual single-root artifact instead of inventing that export spelling.
  if (runtime.bytecode->descriptor->counts.export_count != 1) {
    if (error)
      *error =
          "expected one VM export for explicit @prepare root; got " +
          std::to_string(runtime.bytecode->descriptor->counts.export_count);
    return false;
  }
  iree_vm_module_export_declaration_t entry{};
  runtime.bytecode->vtable->query_export(runtime.bytecode, 0, &entry);
  iree_vm_function_t function = iree_vm_function_null();
  if (!ok(iree_vm_process_lookup_function(runtime.process, IREE_SV("compiled"),
                                          entry.export_name, &function)) ||
      !ok(iree_vm_invoke(runtime.invocation, function,
                         {args.data(), args.size()},
                         {runtime.results.data(), runtime.results.size()})))
    return false;
  if (!ok(iree_vm_i64_from_variant(runtime.results[0],
                                   &out->effective_prefill)) ||
      !ok(iree_vm_i32_from_variant(runtime.results[1], &out->shared)))
    return false;
  for (size_t i = 0; i < 7; ++i) {
    void *ptr = nullptr;
    if (!ok(iree_vm_ptr_from_variant_borrowed(runtime.results[i + 2],
                                              types.buffer, &ptr)))
      return false;
    auto *b = static_cast<iree_vm_buffer_t *>(ptr);
    const auto *bytes =
        static_cast<const uint8_t *>(iree_vm_buffer_const_data(b));
    const auto length = iree_vm_buffer_length(b);
    if (length)
      out->buffers[i].assign(bytes, bytes + length);
  }
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::testing
