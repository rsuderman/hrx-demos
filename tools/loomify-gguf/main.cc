// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "iree/base/tooling/flags.h"
#include "loomc/target/amdgpu.h"
#include "tools/loomify-gguf/gguf_model_builder.h"
#include "tools/loomify-gguf/embedding.h"
#include "tools/model/model_descriptor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
IREE_FLAG(string, gguf, nullptr,
          "Input GGUF checkpoint (weights remain external).");
IREE_FLAG(string, output, nullptr, "Output .loombc or .loom file path.");
IREE_FLAG(string, format, "bc", "Output serialization: bc or loom.");
IREE_FLAG_NAMED(string, output_dir, "output-dir", nullptr,
                "New directory for a complete serving package.");
IREE_FLAG_NAMED(string, chat_template, "chat-template", nullptr,
                "Explicit named chat template from the input GGUF.");
IREE_FLAG(string, tokenizer, nullptr,
          "External matching tokenizer.json for full-model export.");
IREE_FLAG_NAMED(int64_t, context_capacity, "context-capacity", 512,
                "Baked model context capacity.");
IREE_FLAG_NAMED(int64_t, prefill_capacity, "prefill-capacity", 128,
                "Baked maximum prefill chunk size (1..512).");
IREE_FLAG(string, stage, "embedding",
          "Generation stage: embedding (F32) or model (supported Llama/Qwen).");
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
std::filesystem::path LibraryRoot(const char *executable) {
  std::vector<std::filesystem::path> candidates;
  if (const char *runfiles = std::getenv("RUNFILES_DIR"))
    candidates.emplace_back(std::filesystem::path(runfiles) / "_main");
  candidates.emplace_back(
      std::filesystem::path(std::string(executable) + ".runfiles") / "_main");
  for (const auto &path : candidates)
    if (std::filesystem::is_directory(path / "lib/architectures"))
      return path;
  return {};
}
int ExportModel(const char *executable) {
  if (!FLAG_output_dir || !*FLAG_output_dir || !FLAG_tokenizer ||
      !*FLAG_tokenizer || FLAG_output || FLAG_prefill_capacity < 1 ||
      FLAG_prefill_capacity > 512 ||
      FLAG_context_capacity < FLAG_prefill_capacity ||
      FLAG_context_capacity > 262144) {
    std::fprintf(stderr, "loomify-gguf: model requires "
                         "--output-dir=NEW_DIRECTORY, --tokenizer=PATH and "
                         "valid capacities; use --output only for embedding\n");
    return 1;
  }
  std::unique_ptr<hrx::loomify_gguf::GgufModelBuilder> builder;
  std::string error;
  if (!hrx::loomify_gguf::GgufModelBuilder::Open(FLAG_gguf, &builder, &error)) {
    std::fprintf(stderr, "loomify-gguf: %s\n", error.c_str());
    return 1;
  }
  hrx::loomify_gguf::GgufBuildOptions build;
  build.library_root = LibraryRoot(executable).string();
  build.prefill_capacity = FLAG_prefill_capacity;
  build.context_capacity = FLAG_context_capacity;
  hrx::loomify_gguf::GgufPackageOptions package;
  package.output_directory = FLAG_output_dir;
  package.tokenizer = FLAG_tokenizer;
  if (FLAG_chat_template) package.template_name = FLAG_chat_template;
  package.format = std::strcmp(FLAG_format, "bc") == 0
      ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
  if (build.library_root.empty()) {
    std::fprintf(stderr, "loomify-gguf: declared model products are missing from runfiles\n");
    return 1;
  }
  loomc_target_environment_t *environment = nullptr;
  if (!Check(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                    &environment)))
    return 1;
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment_owner(environment);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t context_options{};
  context_options.next = &target;
  loomc_context_t *context = nullptr;
  if (!Check(loomc_context_create(&context_options, loomc_allocator_system(),
                                  &context)))
    return 1;
  Owner<loomc_context_t, loomc_context_release> context_owner(context);
  loomc_workspace_t *workspace = nullptr;
  if (!Check(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                    &workspace)))
    return 1;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_owner(workspace);
  if (!builder->WritePackage(context, workspace, build, package, &error)) {
    std::fprintf(stderr, "loomify-gguf: %s\n", error.c_str());
    return 1;
  }
  std::fprintf(stderr, "loomify-gguf: wrote Serve model package to %s\n",
               FLAG_output_dir);
  return 0;
}
} // namespace
int main(int argc, char **argv) {
  iree_flags_set_usage(
      "loomify-gguf",
      "Construct a complete Loom module for a bounded GGUF model stage.\n"
      "\nUsage: loomify-gguf --gguf=MODEL.gguf --output=embedding.loombc\n"
      "       [--format=bc|loom] [--stage=embedding]\n"
      "       loomify-gguf --stage=model --gguf=MODEL.gguf "
      "--tokenizer=tokenizer.json\n"
      "       --output-dir=NEW_DIRECTORY [--context-capacity=512] "
      "[--prefill-capacity=128]\n"
      "\nEmbedding currently supports F32 Llama token_embd.weight only.\n"
      "--stage=embedding outputs only the embedding command. --stage=model "
      "outputs a complete serving package. Weights stay external.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1 || !FLAG_gguf || !*FLAG_gguf) {
    std::fprintf(stderr, "loomify-gguf: --gguf=PATH is "
                         "required; see --help\n");
    return 1;
  }
  if ((std::strcmp(FLAG_stage, "embedding") != 0 &&
       std::strcmp(FLAG_stage, "model") != 0) ||
      (std::strcmp(FLAG_format, "bc") != 0 &&
       std::strcmp(FLAG_format, "loom") != 0)) {
    std::fprintf(
        stderr,
        "loomify-gguf: supported stages are embedding and model; format "
        "must be bc or loom\n");
    return 1;
  }
  if (std::strcmp(FLAG_stage, "model") == 0)
    return ExportModel(argv[0]);
  if (!FLAG_output || !*FLAG_output || FLAG_output_dir || FLAG_tokenizer || FLAG_chat_template) {
    std::fprintf(stderr, "loomify-gguf: embedding requires --output=PATH and "
                         "no model package flags\n");
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
