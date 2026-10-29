// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include "loomc/target/configured_ir.h"
#include "tools/loomify-gguf/authoring/model-full/llama_step.h"
#include "tools/model/gguf_catalog.h"
#include "tools/model/model_descriptor.h"

namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(message.size), message.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  return false;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: emit_llama_step MODEL.gguf "
                         "NUMERICAL_DEFINITIONS.loombc OUTPUT.loombc\n");
    return 1;
  }
  for (int input : {1, 2}) {
    std::error_code error;
    if (std::filesystem::equivalent(argv[input], argv[3], error)) {
      std::fprintf(stderr,
                   "output aliases an input; choose a distinct output path\n");
      return 1;
    }
  }
  FILE *file = std::fopen(argv[1], "rb");
  if (!file) {
    std::perror(argv[1]);
    return 1;
  }
  std::setvbuf(file, nullptr, _IONBF, 0);
  hrx::model::GgufCatalog catalog;
  hrx::model::ModelDescriptor descriptor;
  std::string error;
  bool loaded = hrx::model::ReadGgufCatalog(file, &catalog, &error);
  std::fclose(file);
  if (!loaded ||
      !hrx::model::LoadModelDescriptor(catalog, &descriptor, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  loomc_target_environment_t *environment = nullptr;
  if (!Ok(loomc_target_environment_create_configured_ir(
          loomc_allocator_system(), &environment)))
    return 1;
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment_owner(environment);
  loomc_context_target_options_t target_options = {};
  target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target_options.structure_size = sizeof(target_options);
  target_options.target_environment = environment;
  loomc_context_options_t context_options = {};
  context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
  context_options.structure_size = sizeof(context_options);
  context_options.next = &target_options;
  loomc_context_t *context = nullptr;
  if (!Ok(loomc_context_create(&context_options, loomc_allocator_system(),
                               &context)))
    return 1;
  Owner<loomc_context_t, loomc_context_release> context_owner(context);
  loomc_workspace_t *workspace = nullptr;
  if (!Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_owner(workspace);
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(loomc_module_deserialize_from_path(
          context, workspace, loomc_make_cstring_view(argv[2]), nullptr,
          loomc_allocator_system(), &definitions, &result)))
    return 1;
  Owner<loomc_result_t, loomc_result_release> parse_result(result);
  Owner<loomc_module_t, loomc_module_release> definitions_owner(definitions);
  if (!Succeeded(result))
    return 1;
  std::unique_ptr<hrx::loomify_gguf::ModuleBuilder> module;
  if (!Ok(hrx::loomify_gguf::ModuleBuilder::Create(context, workspace,
                                                   definitions, &module)))
    return 1;
  hrx::loomify_gguf::ProgramRef prefill, decode;
  if (!Ok(hrx::loomify_gguf::DefineLlamaStep(
          descriptor, *module, hrx::loomify_gguf::LlamaStepKind::kPrefill,
          "generated.llama_prefill", &prefill)))
    return 1;
  if (!Ok(hrx::loomify_gguf::DefineLlamaStep(
          descriptor, *module, hrx::loomify_gguf::LlamaStepKind::kDecode,
          "generated.llama_decode", &decode)))
    return 1;
  loomc_module_t *output = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify({prefill, decode}, &output, &result)))
    return 1;
  Owner<loomc_result_t, loomc_result_release> link_result(result);
  Owner<loomc_module_t, loomc_module_release> output_owner(output);
  if (!Succeeded(result))
    return 1;
  loomc_module_serialize_options_t options = {};
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  return Ok(loomc_module_serialize_to_path(output, &options,
                                           loomc_make_cstring_view(argv[3]),
                                           loomc_allocator_system()))
             ? 0
             : 1;
}
