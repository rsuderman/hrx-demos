// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "iree/base/tooling/flags.h"
#include "tools/loomify-gguf/source_builder.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
IREE_FLAG(string, gguf, nullptr,
          "Input GGUF checkpoint (weights remain external).");
IREE_FLAG(string, output, nullptr, "Output .loombc or .loom file path.");
IREE_FLAG(string, format, "bc", "Output serialization: bc or loom.");
IREE_FLAG(string, stage, "embedding",
          "Bounded generation stage: embedding (F32 Llama).");
namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
bool Check(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    std::fprintf(stderr, "loomify-gguf: %.*s\n", int(message.size),
                 message.data);
  }
  loomc_status_free(status);
  return ok;
}
} // namespace
int main(int argc, char **argv) {
  iree_flags_set_usage(
      "loomify-gguf",
      "Construct a complete Loom module for a bounded GGUF model stage.\n"
      "\nUsage: loomify-gguf --gguf=MODEL.gguf --output=embedding.loombc\n"
      "       [--format=bc|loom] [--stage=embedding]\n"
      "\nEmbedding currently supports F32 Llama token_embd.weight only.\n"
      "Output is the embedding command, not a complete model. Weights stay "
      "external.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1 || !FLAG_gguf || !*FLAG_gguf || !FLAG_output || !*FLAG_output) {
    std::fprintf(stderr, "loomify-gguf: --gguf=PATH and --output=PATH are "
                         "required; see --help\n");
    return 1;
  }
  if (std::strcmp(FLAG_stage, "embedding") != 0 ||
      (std::strcmp(FLAG_format, "bc") != 0 &&
       std::strcmp(FLAG_format, "loom") != 0)) {
    std::fprintf(stderr, "loomify-gguf: supported stage is embedding; format "
                         "must be bc or loom\n");
    return 1;
  }
  std::error_code path_error;
  if (std::filesystem::equivalent(FLAG_gguf, FLAG_output, path_error)) {
    std::fprintf(stderr,
                 "loomify-gguf: output must not alias the GGUF input\n");
    return 1;
  }
  FILE *input = std::fopen(FLAG_gguf, "rb");
  if (!input) {
    std::perror("loomify-gguf: opening GGUF");
    return 1;
  }
  std::unique_ptr<FILE, decltype(&std::fclose)> file(input, &std::fclose);
  if (std::setvbuf(input, nullptr, _IONBF, 0) != 0) {
    std::fprintf(stderr, "loomify-gguf: unable to disable input buffering\n");
    return 1;
  }
  hrx::model::GgufCatalog catalog;
  hrx::model::ModelDescriptor model;
  std::string error;
  if (!hrx::model::ReadGgufCatalog(input, &catalog, &error) ||
      !hrx::model::LoadModelDescriptor(catalog, &model, &error)) {
    std::fprintf(stderr, "loomify-gguf: %s\n", error.c_str());
    return 1;
  }
  loomc_context_t *context = nullptr;
  if (!Check(loomc_context_create(nullptr, loomc_allocator_system(), &context)))
    return 1;
  Owner<loomc_context_t, loomc_context_release> context_owner(context);
  loomc_workspace_t *workspace = nullptr;
  if (!Check(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                    &workspace)))
    return 1;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_owner(workspace);
  loomc_module_t *module = nullptr;
  if (!hrx::loomify_gguf::BuildEmbeddingProgram(model, context, workspace,
                                                &module, &error)) {
    std::fprintf(stderr, "loomify-gguf: %s\n", error.c_str());
    return 1;
  }
  Owner<loomc_module_t, loomc_module_release> module_owner(module);
  loomc_module_serialize_options_t options = {};
  options.format = std::strcmp(FLAG_format, "bc") == 0
                       ? LOOMC_SOURCE_FORMAT_BYTECODE
                       : LOOMC_SOURCE_FORMAT_TEXT;
  if (!Check(loomc_module_serialize_to_path(
          module, &options, loomc_make_cstring_view(FLAG_output),
          loomc_allocator_system())))
    return 1;
  std::fprintf(stderr,
               "loomify-gguf: wrote model.embedding (%llu x %llu F32 table); "
               "full model not generated\n",
               static_cast<unsigned long long>(model.vocabulary_size),
               static_cast<unsigned long long>(model.embedding_width));
  return 0;
}
