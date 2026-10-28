// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/registry/loading/module_loader.h"

#include <memory>

#include "loomc/source.h"

namespace hrx::loomify_gguf {
namespace {
using SourceOwner =
    std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)>;
} // namespace

loomc_status_t LoadBytecode(loomc_context_t *context,
                            loomc_workspace_t *workspace,
                            loomc_string_view_t path,
                            loomc_module_t **out_module,
                            loomc_result_t **out_result) {
  if (out_module)
    *out_module = nullptr;
  if (out_result)
    *out_result = nullptr;
  if (!context || !workspace || !out_module || !out_result) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace and output pointers are required");
  }
  loomc_source_load_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS;
  options.structure_size = sizeof(options);
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  loomc_source_t *source = nullptr;
  loomc_status_t status = loomc_source_create_from_path(
      path, &options, loomc_allocator_system(), &source);
  if (!loomc_status_is_ok(status))
    return status;
  SourceOwner source_owner(source, loomc_source_release);
  return loomc_module_deserialize_bytecode_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), out_module,
      out_result);
}

loomc_status_t DumpText(const loomc_module_t *module, std::string *out_text) {
  if (!out_text) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "output text pointer is required");
  }
  out_text->clear();
  if (!module) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module is required");
  }
  loomc_module_serialize_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
  options.structure_size = sizeof(options);
  options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
  options.print_locations = false;
  loomc_source_t *source = nullptr;
  loomc_status_t status = loomc_module_serialize_text_to_source(
      module, &options, loomc_allocator_system(), &source);
  if (!loomc_status_is_ok(status))
    return status;
  SourceOwner source_owner(source, loomc_source_release);
  const auto contents = loomc_source_contents(source);
  if (contents.data_length) {
    out_text->assign(reinterpret_cast<const char *>(contents.data),
                     contents.data_length);
  }
  return loomc_ok_status();
}

} // namespace hrx::loomify_gguf
