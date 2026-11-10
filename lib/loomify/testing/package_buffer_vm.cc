// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/testing/package_buffer_vm.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loomc/loomc.h"
#include "loomc/target/vm.h"
#include <array>
#include <memory>
namespace hrx::loomify_gguf::testing {
namespace {
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
bool Ok(loomc_status_t status, std::string *error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto m = loomc_status_message(status);
    error->assign(m.data, m.size);
  }
  loomc_status_free(status);
  return ok;
}
bool NativeOk(iree_status_t status, std::string *error) {
  if (iree_status_is_ok(status))
    return true;
  char message[4096];
  iree_host_size_t size = 0;
  iree_status_format(status, sizeof(message), message, &size);
  *error = message;
  iree_status_ignore(status);
  return false;
}
bool Result(loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result))
    return true;
  *error = "package host compilation failed";
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto *d = loomc_result_diagnostic_at(result, i);
      error->append("\n").append(d->message.data, d->message.size);
    }
  return false;
}
struct Runtime {
  ~Runtime() {
    iree_vm_process_release(process);
    if (invocation)
      iree_vm_invocation_deinitialize(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(module);
    iree_vm_environment_free(environment);
    loomc_allocator_free(loomc_allocator_system(),
                         const_cast<uint8_t *>(image.data));
  }
  loomc_byte_span_t image{};
  iree_vm_environment_t *environment = nullptr;
  iree_vm_module_t *module = nullptr;
  iree_vm_program_t *program = nullptr;
  iree_vm_invocation_t *invocation = nullptr;
  iree_vm_process_t *process = nullptr;
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage{};
};
} // namespace
bool InvokePackageBuffer(const std::string &path, const char *root,
                         bool bytecode, std::string *value,
                         std::string *error) {
  loomc_target_environment_t *e = nullptr;
  if (!Ok(loomc_target_environment_create_vm(loomc_allocator_system(), &e),
          error))
    return false;
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment(e, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = e;
  loomc_context_options_t co{};
  co.next = &target;
  loomc_context_t *c = nullptr;
  if (!Ok(loomc_context_create(&co, loomc_allocator_system(), &c), error))
    return false;
  Owner<loomc_context_t, loomc_context_release> context(c,
                                                        loomc_context_release);
  loomc_workspace_t *w = nullptr;
  if (!Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w), error))
    return false;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace(
      w, loomc_workspace_release);
  loomc_module_t *m = nullptr;
  loomc_result_t *r = nullptr;
  auto status = loomc_module_deserialize_from_path(
      c, w, loomc_make_cstring_view(path.c_str()), nullptr,
      loomc_allocator_system(), &m, &r);
  Owner<loomc_module_t, loomc_module_release> module(m, loomc_module_release);
  Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
  if (!Ok(status, error) || !Result(r, error))
    return false;
  if (bytecode) {
    loomc_module_serialize_options_t serialization{};
    serialization.format = LOOMC_SOURCE_FORMAT_BYTECODE;
    loomc_source_t *s = nullptr;
    if (!Ok(loomc_module_serialize_to_source(m, &serialization,
                                             loomc_allocator_system(), &s),
            error))
      return false;
    Owner<loomc_source_t, loomc_source_release> source(s, loomc_source_release);
    loomc_module_t *restored = nullptr;
    r = nullptr;
    status = loomc_module_deserialize_from_source(
        c, w, s, nullptr, loomc_allocator_system(), &restored, &r);
    module.reset(restored);
    result.reset(r);
    if (!Ok(status, error) || !Result(r, error))
      return false;
  }
  loomc_compiler_t *compiler_raw = nullptr;
  if (!Ok(loomc_compiler_create(c, nullptr, loomc_allocator_system(),
                                &compiler_raw),
          error))
    return false;
  Owner<loomc_compiler_t, loomc_compiler_release> compiler(
      compiler_raw, loomc_compiler_release);
  loomc_target_profile_t *p = nullptr;
  if (!Ok(loomc_target_profile_select(e, loomc_make_cstring_view("vm:core"),
                                      loomc_allocator_system(), &p),
          error))
    return false;
  Owner<loomc_target_profile_t, loomc_target_profile_release> profile(
      p, loomc_target_profile_release);
  const auto symbol = loomc_make_cstring_view(root);
  loomc_emit_options_t emit{};
  emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
  emit.structure_size = sizeof(emit);
  emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
  loomc_compile_artifact_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
  options.structure_size = sizeof(options);
  options.roots = &symbol;
  options.root_count = 1;
  options.target_profile = p;
  options.emit_options = &emit;
  r = nullptr;
  status = loomc_compile_artifact(compiler_raw, w, nullptr, module.get(),
                                  &options, loomc_allocator_system(), &r);
  result.reset(r);
  if (!Ok(status, error) || !Result(r, error))
    return false;
  Runtime runtime;
  const loomc_artifact_t *artifact = nullptr;
  for (size_t i = 0; i < loomc_result_artifact_count(r); ++i)
    if (loomc_result_artifact_at(r, i)->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE)
      artifact = loomc_result_artifact_at(r, i);
  if (!artifact) {
    *error = "package host root produced no executable";
    return false;
  }
  if (!Ok(loomc_byte_sequence_clone(artifact->contents,
                                    loomc_allocator_system(), &runtime.image),
          error))
    return false;
  if (!NativeOk(iree_vm_environment_allocate(iree_allocator_system(),
                                             &runtime.environment),
                error))
    return false;
  iree_vm_ref_types_t types{};
  if (!NativeOk(
          iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                        runtime.environment, IREE_SV("vm")),
                                    &types),
          error))
    return false;
  if (!NativeOk(iree_vm_bytecode_module_create(
                    runtime.environment, IREE_SV("package"),
                    {iree_make_const_byte_span(runtime.image.data,
                                               runtime.image.data_length),
                     iree_allocator_null()},
                    iree_allocator_system(), &runtime.module),
                error))
    return false;
  if (!NativeOk(
          iree_vm_program_create({runtime.module, iree_vm_module_span_empty()},
                                 iree_allocator_system(), &runtime.program),
          error))
    return false;
  if (!NativeOk(iree_vm_invocation_initialize(
                    iree_make_byte_span(runtime.storage.data(),
                                        runtime.storage.size()),
                    &runtime.invocation),
                error))
    return false;
  if (!NativeOk(iree_vm_process_create(runtime.program, runtime.invocation,
                                       iree_vm_variant_span_empty(),
                                       iree_allocator_system(),
                                       &runtime.process),
                error))
    return false;
  iree_vm_function_t fn = iree_vm_function_null();
  bool found = false;
  for (uint32_t i = 0; i < runtime.module->descriptor->counts.export_count;
       ++i) {
    iree_vm_module_export_declaration_t entry{};
    runtime.module->vtable->query_export(runtime.module, i, &entry);
    const std::string name(entry.export_name.data, entry.export_name.size);
    if (name == root || name.rfind(std::string(root) + "$", 0) == 0 ||
        name.rfind(std::string(root) + ".", 0) == 0) {
      if (!NativeOk(iree_vm_process_lookup_function(runtime.process,
                                                    IREE_SV("package"),
                                                    entry.export_name, &fn),
                    error))
        return false;
      found = true;
      break;
    }
  }
  if (!found) {
    *error = "missing package host export";
    return false;
  }
  iree_vm_variant_t out{};
  const bool invoked =
      NativeOk(iree_vm_invoke(runtime.invocation, fn,
                              iree_vm_variant_span_empty(), {&out, 1}),
               error);
  void *ptr = nullptr;
  const bool typed =
      invoked &&
      NativeOk(iree_vm_ptr_from_variant_borrowed(out, types.buffer, &ptr),
               error);
  if (typed) {
    auto *buffer = static_cast<iree_vm_buffer_t *>(ptr);
    value->assign(static_cast<const char *>(iree_vm_buffer_const_data(buffer)),
                  iree_vm_buffer_length(buffer));
  }
  iree_vm_variant_span_reset({&out, 1});
  if (!typed)
    return false;
  error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::testing
