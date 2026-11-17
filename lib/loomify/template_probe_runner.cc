// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/template_probe_runner.h"

#include "lib/loomify/linking.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/json.h"
#include "loom/runtime/program/program.h"
#include "loomc/target/vm.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::jinja {
namespace {
iree_string_view_t View(std::string_view value) {
  return iree_make_string_view(value.data(), value.size());
}

bool Fail(std::string message, std::string *error) {
  if (error)
    *error = std::move(message);
  return false;
}

bool RuntimeOk(iree_status_t status, std::string *error) {
  if (iree_status_is_ok(status))
    return true;
  char text[4096] = {};
  iree_host_size_t size = 0;
  iree_status_format(status, sizeof(text), text, &size);
  iree_status_ignore(status);
  return Fail(std::string(text, std::min<size_t>(size, sizeof(text) - 1)),
              error);
}

bool CompilerOk(loomc_status_t status, std::string *error) {
  if (loomc_status_is_ok(status)) {
    loomc_status_free(status);
    return true;
  }
  const auto message = loomc_status_message(status);
  const std::string copy(message.data, message.size);
  loomc_status_free(status);
  return Fail(copy, error);
}

bool ResolveHelpers(const GeneratedInputRenderer &generated,
                    const std::string &root, std::vector<std::string> *out,
                    std::string *error) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (root.empty() || !fs::is_directory(root, ec) || ec)
    return Fail("template helper root must be an existing directory", error);
  for (const auto &helper : generated.helper_requirements) {
    const fs::path relative(helper);
    if (relative.empty() || relative.is_absolute() || relative.has_root_path())
      return Fail("invalid template helper resource: " + helper, error);
    for (const auto &part : relative)
      if (part.empty() || part == "." || part == "..")
        return Fail("invalid template helper resource: " + helper, error);
    // BUILD-owned runfiles may themselves be symlinks. The explicit resource
    // root authorizes those resources; template source never supplies paths.
    const fs::path path = fs::path(root) / relative;
    ec.clear();
    if (!fs::is_regular_file(path, ec) || ec)
      return Fail("missing template helper resource: " + path.string(), error);
    out->push_back(path.string());
  }
  return true;
}
} // namespace

struct PreparedTemplate::Impl {
  ~Impl() {
    loom_runtime_program_free(program);
    iree_vm_module_release(input);
    iree_vm_module_release(json);
    iree_vm_environment_free(environment);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(target);
    if (!scratch.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(scratch, ignored);
    }
  }

  bool Initialize(const GeneratedInputRenderer &generated,
                  const std::string &helper_root,
                  const std::string &scratch_parent, bool bytecode,
                  std::string *error, const iree_tokenizer_t *tokenizer) {
    if (generated.source.empty() || generated.entrypoint.empty())
      return Fail("template source and entrypoint must be nonempty", error);
    std::vector<std::string> helpers;
    if (!ResolveHelpers(generated, helper_root, &helpers, error))
      return false;
    std::error_code ec;
    if (scratch_parent.empty() ||
        !std::filesystem::is_directory(scratch_parent, ec) || ec)
      return Fail("template scratch parent must be an existing directory",
                  error);
    std::string pattern =
        (std::filesystem::path(scratch_parent) / "template-probe-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char *created = ::mkdtemp(writable.data());
    if (!created)
      return Fail("cannot create isolated template scratch directory", error);
    scratch = created;
    if (!CompilerOk(loomc_target_environment_create_vm(loomc_allocator_system(),
                                                       &target),
                    error))
      return false;
    loomc_context_target_options_t target_options{};
    target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.target_environment = target;
    loomc_context_options_t context_options{};
    context_options.next = &target_options;
    if (!CompilerOk(loomc_context_create(&context_options,
                                         loomc_allocator_system(), &context),
                    error) ||
        !CompilerOk(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                           &workspace),
                    error))
      return false;
    const std::vector<const char *> roots{generated.entrypoint.c_str()};
    loomc_module_t *linked = nullptr;
    if (!LinkConfiguredProducts(context, workspace, generated.source, helpers,
                                roots, false, &linked, error))
      return false;
    loomc_module_serialize_options_t options{};
    options.format =
        bytecode ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
    loomc_source_t *serialized = nullptr;
    const bool serialized_ok =
        CompilerOk(loomc_module_serialize_to_source(
                       linked, &options, loomc_allocator_system(), &serialized),
                   error);
    loomc_module_release(linked);
    if (!serialized_ok)
      return false;
    const auto path = scratch / "template.loom";
    const auto bytes = loomc_source_contents(serialized);
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
    output.close();
    loomc_source_release(serialized);
    if (!output)
      return Fail("cannot write isolated template product", error);
    if (!RuntimeOk(
            iree_vm_environment_allocate(iree_allocator_system(), &environment),
            error) ||
        !RuntimeOk(
            iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                          environment, IREE_SV("vm")),
                                      &types),
            error) ||
        !RuntimeOk(loom_runtime_json_module_create(
                       environment, iree_allocator_system(), &json),
                   error) ||
        !RuntimeOk(loom_runtime_input_module_create(
                       environment, tokenizer, iree_allocator_system(), &input),
                   error))
      return false;
    iree_vm_module_t *libraries[] = {json, input};
    const auto runtime_root = View(generated.entrypoint);
    loom_runtime_sources_t sources{};
    loom_runtime_compilation_report_t report{};
    const auto status = loom_runtime_program_allocate(
        target, environment, &sources, View(path.string()), 1, &runtime_root,
        iree_vm_module_span_from_array(libraries), &report,
        iree_allocator_system(), &program);
    if (!RuntimeOk(status, error)) {
      if (error)
        for (iree_host_size_t i = 0; i < report.diagnostic_count; ++i) {
          const auto &entry = report.diagnostics[i];
          error->append("\n").append(
              entry.message,
              std::min<uint64_t>(entry.message_length, sizeof(entry.message)));
        }
      return false;
    }
    return RuntimeOk(
        iree_vm_process_lookup_function(loom_runtime_program_process(program),
                                        IREE_SV("model"), runtime_root, &entry),
        error);
  }

  loomc_target_environment_t *target = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  iree_vm_environment_t *environment = nullptr;
  iree_vm_module_t *json = nullptr;
  iree_vm_module_t *input = nullptr;
  loom_runtime_program_t *program = nullptr;
  iree_vm_ref_types_t types{};
  iree_vm_function_t entry = iree_vm_function_null();
  std::filesystem::path scratch;
};

PreparedTemplate::PreparedTemplate() : impl_(std::make_unique<Impl>()) {}
PreparedTemplate::~PreparedTemplate() = default;

bool PreparedTemplate::Prepare(const GeneratedInputRenderer &generated,
                               const std::string &helper_root,
                               const std::string &scratch_parent, bool bytecode,
                               std::unique_ptr<PreparedTemplate> *out,
                               std::string *error,
                               const iree_tokenizer_t *tokenizer) {
  if (error)
    error->clear();
  if (!out)
    return Fail("template output must not be null", error);
  std::unique_ptr<PreparedTemplate> prepared(new PreparedTemplate());
  if (!prepared->impl_->Initialize(generated, helper_root, scratch_parent,
                                   bytecode, error, tokenizer))
    return false;
  *out = std::move(prepared);
  return true;
}

bool PreparedTemplate::Render(std::string_view environment_json,
                              std::string *out, std::string *error,
                              int32_t *native_status_code) {
  if (error)
    error->clear();
  if (native_status_code)
    *native_status_code = IREE_STATUS_OK;
  if (!out) {
    if (native_status_code)
      *native_status_code = IREE_STATUS_INVALID_ARGUMENT;
    return Fail("rendered output must not be null", error);
  }
  iree_vm_buffer_t *input = nullptr;
  auto status =
      iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                           iree_make_const_byte_span(environment_json.data(),
                                                     environment_json.size()),
                           1, iree_allocator_system(), &input);
  if (native_status_code)
    *native_status_code = iree_status_code(status);
  if (!RuntimeOk(status, error))
    return false;
  iree_vm_variant_t argument =
      iree_vm_buffer_variant_from_ptr_move(&impl_->types, &input);
  iree_vm_variant_t result = {0};
  status =
      iree_vm_invoke(loom_runtime_program_invocation(impl_->program),
                     impl_->entry, iree_vm_variant_span_from_ptr(&argument, 1),
                     iree_vm_variant_span_from_ptr(&result, 1));
  std::string rendered;
  if (iree_status_is_ok(status)) {
    iree_vm_buffer_t *buffer = nullptr;
    status = iree_vm_buffer_ptr_from_variant_borrowed(&impl_->types, result,
                                                      &buffer);
    if (iree_status_is_ok(status)) {
      if (!buffer)
        status = iree_make_status(IREE_STATUS_INTERNAL, "null rendered buffer");
      else if (iree_vm_buffer_length(buffer))
        rendered.assign(
            static_cast<const char *>(iree_vm_buffer_const_data(buffer)),
            iree_vm_buffer_length(buffer));
    }
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_ptr(&result, 1));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_ptr(&argument, 1));
  if (native_status_code)
    *native_status_code = iree_status_code(status);
  if (!RuntimeOk(status, error))
    return false;
  *out = std::move(rendered);
  return true;
}

bool PreparedTemplate::SelectFormat(std::string_view environment_json,
                                    int64_t *format_id, int64_t *startup_mode,
                                    std::string *error) {
  if (error)
    error->clear();
  if (!format_id || !startup_mode)
    return Fail("selector outputs must not be null", error);
  iree_vm_buffer_t *input = nullptr;
  if (!RuntimeOk(iree_vm_buffer_clone(
                     IREE_VM_BUFFER_ACCESS_FLAG_READ,
                     iree_make_const_byte_span(environment_json.data(),
                                               environment_json.size()),
                     1, iree_allocator_system(), &input),
                 error))
    return false;
  iree_vm_variant_t argument =
      iree_vm_buffer_variant_from_ptr_move(&impl_->types, &input);
  iree_vm_variant_t results[2] = {};
  auto status =
      iree_vm_invoke(loom_runtime_program_invocation(impl_->program),
                     impl_->entry, iree_vm_variant_span_from_ptr(&argument, 1),
                     iree_vm_variant_span_from_array(results));
  int64_t selected_format = 0, selected_startup = 0;
  if (iree_status_is_ok(status))
    status = iree_vm_i64_from_variant(results[0], &selected_format);
  if (iree_status_is_ok(status))
    status = iree_vm_i64_from_variant(results[1], &selected_startup);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_ptr(&argument, 1));
  if (!RuntimeOk(status, error))
    return false;
  *format_id = selected_format;
  *startup_mode = selected_startup;
  return true;
}

} // namespace hrx::loomify_gguf::jinja
