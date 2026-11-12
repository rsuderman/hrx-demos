// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "iree/base/tooling/flags.h"
#include "lib/loomify/gguf_model_builder.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

IREE_FLAG(
    string, platform, "amdgpu:gfx11-generic",
    "Maintained generation platform recipe (not a live device selector).");
IREE_FLAG(string, gguf, nullptr,
          "Input GGUF checkpoint (weights remain external).");
IREE_FLAG(string, output, nullptr,
          "New numerical .loombc or .loom output file.");
IREE_FLAG(string, format, "bc",
          "Module serialization: bc or loom; inspection is always text.");
IREE_FLAG_NAMED(
    string, output_dir, "output-dir", nullptr,
    "New directory for a complete Serve package (requires tokenizer).");
IREE_FLAG_NAMED(
    string, configuration_output, "configuration-output", nullptr,
    "New inspection directory: numeric.loom and optional serve-data.loom. "
    "Written before native export and retained if later native export fails.");
IREE_FLAG_NAMED(string, chat_template, "chat-template", nullptr,
                "Named GGUF chat template; requires tokenizer.");
IREE_FLAG(string, tokenizer, nullptr,
          "Matching external tokenizer.json for package data.");
IREE_FLAG_NAMED(int64_t, context_capacity, "context-capacity", 512,
                "Baked context capacity.");
IREE_FLAG_NAMED(int64_t, prefill_capacity, "prefill-capacity", 128,
                "Baked prefill chunk size (1..512).");

namespace {
namespace fs = std::filesystem;
using namespace hrx::loomify_gguf;
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
int Error(const std::string &message) {
  std::fprintf(stderr, "loomify-gguf: %s\n", message.c_str());
  return 1;
}
bool Check(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    std::fprintf(stderr, "loomify-gguf: %.*s\n", int(message.size),
                 message.data);
  }
  loomc_status_free(status);
  return ok;
}
fs::path LibraryRoot(const char *executable) {
  // Choose the conventional caller resource root without reading products.
  // Availability is checked only after optional inspection publication.
  if (const auto *root = std::getenv("RUNFILES_DIR"))
    if (*root)
      return fs::path(root) / "_main";
  return fs::path(std::string(executable) + ".runfiles") / "_main";
}
bool WriteText(const fs::path &path, const std::string &source) {
  std::ofstream file(path, std::ios::binary);
  file.write(source.data(), source.size());
  file.close();
  return bool(file);
}
bool Inspect(const fs::path &directory, const GgufConfiguration &configuration,
             std::string *error) {
  // create_directory acquires exclusive ownership; existing files/symlinks and
  // missing parents fail without taking ownership or deleting anything.
  std::error_code ec;
  if (!fs::create_directory(directory, ec) || ec) {
    *error =
        "cannot create new configuration directory: " + directory.string() +
        ": " + ec.message();
    return false;
  }
  if (WriteText(directory / "numeric.loom",
                configuration.configuration_source()) &&
      (!configuration.has_package_configuration() ||
       WriteText(directory / "serve-data.loom",
                 configuration.package_data_source())))
    return true;
  *error = "cannot write or close configuration output";
  fs::remove_all(directory, ec);
  if (ec)
    *error += "; cannot remove owned partial configuration: " + ec.message();
  return false;
}
bool SameDestination(const char *a, const char *b, std::string *error) {
  std::error_code ec;
  const auto first = fs::weakly_canonical(fs::absolute(a, ec), ec);
  if (ec) {
    *error = "cannot resolve output destination: " + ec.message();
    return true;
  }
  const auto second = fs::weakly_canonical(fs::absolute(b, ec), ec);
  if (ec) {
    *error = "cannot resolve output destination: " + ec.message();
    return true;
  }
  if (first == second) {
    *error = "inspection and native output destinations must be distinct";
    return true;
  }
  return false;
}
} // namespace

int main(int argc, char **argv) {
  iree_flags_set_usage(
      "loomify-gguf",
      "Generate a supported complete model from GGUF using maintained Loom products.\n      Platform recipe: --platform=amdgpu:gfx11-generic (default); Serve chooses the live device.\n"
      "  --gguf=MODEL --output=NEW_FILE [--format=bc|loom]\n"
      "  --gguf=MODEL --tokenizer=JSON --output-dir=NEW_DIRECTORY\n"
      "  --gguf=MODEL --configuration-output=NEW_DIRECTORY [--tokenizer=JSON]\n"
      "Configuration inspection may accompany either export; numerical export rejects tokenizer/template flags.\n"
      "Inspection alone performs no Loom compilation or product loading. Successful inspection is retained\n"
      "if a later native export fails. Existing output paths are never replaced.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const bool module = FLAG_output != nullptr,
             package = FLAG_output_dir != nullptr;
  const bool inspect = FLAG_configuration_output != nullptr,
             tokenizer = FLAG_tokenizer != nullptr;
  if (argc != 1 || !FLAG_gguf || !*FLAG_gguf)
    return Error("--gguf=PATH is required");
  for (const auto *path :
       {FLAG_output, FLAG_output_dir, FLAG_configuration_output, FLAG_tokenizer,
        FLAG_chat_template})
    if (path && !*path)
      return Error("explicit paths/template names must not be empty");
  if ((!module && !package && !inspect) || (module && package))
    return Error(
        "select --output, --output-dir, or --configuration-output; module and package outputs conflict");
  if ((package && !tokenizer) ||
      (module && (tokenizer || FLAG_chat_template)) ||
      (FLAG_chat_template && !tokenizer) || (tokenizer && !package && !inspect))
    return Error(
        "tokenizer is required for packages and accepted for inspection-only package preparation; chat-template requires tokenizer");
  if ((std::strcmp(FLAG_format, "bc") && std::strcmp(FLAG_format, "loom")) ||
      FLAG_prefill_capacity < 1 || FLAG_prefill_capacity > 512 ||
      FLAG_context_capacity < FLAG_prefill_capacity ||
      FLAG_context_capacity > 262144)
    return Error("unsupported format or context/prefill capacities");
  std::string error;
  if (inspect && (module || package) &&
      SameDestination(FLAG_configuration_output,
                      module ? FLAG_output : FLAG_output_dir, &error))
    return Error(error);

  std::unique_ptr<GgufModelBuilder> builder;
  if (!GgufModelBuilder::Open(FLAG_gguf, &builder, &error))
    return Error(error);
  GgufBuildOptions build;
  build.platform = FLAG_platform;
  build.prefill_capacity = FLAG_prefill_capacity;
  build.context_capacity = FLAG_context_capacity;
  // Copy the conventional resource root without reading any model products.
  // Actual availability checks follow successful optional inspection.
  if (module || package)
    build.library_root = LibraryRoot(argv[0]).string();
  GgufPackageOptions options;
  if (package)
    options.output_directory = FLAG_output_dir;
  if (tokenizer)
    options.tokenizer = FLAG_tokenizer;
  if (FLAG_chat_template)
    options.template_name = FLAG_chat_template;
  options.format = std::strcmp(FLAG_format, "bc") == 0
                       ? LOOMC_SOURCE_FORMAT_BYTECODE
                       : LOOMC_SOURCE_FORMAT_TEXT;
  std::unique_ptr<GgufConfiguration> configuration;
  if (!builder->Configure(build, tokenizer ? &options : nullptr, &configuration,
                          &error))
    return Error(error);
  builder.reset();
  if (inspect && !Inspect(FLAG_configuration_output, *configuration, &error))
    return Error(error);
  if (!module && !package)
    return 0;
  // Native loaders validate only the selected products. Numerical export must
  // not require unrelated text or other-architecture products from the
  // inventory.
  if (build.library_root.empty())
    return Error("declared Loom resource root is missing");

  loomc_target_environment_t *environment = nullptr;
  if (!Check(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                    &environment)))
    return 1;
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment_owner(environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t context_options{};
  context_options.next = &target;
  loomc_context_t *context = nullptr;
  if (!Check(loomc_context_create(&context_options, loomc_allocator_system(),
                                  &context)))
    return 1;
  Owner<loomc_context_t, loomc_context_release> context_owner(
      context, loomc_context_release);
  loomc_workspace_t *workspace = nullptr;
  if (!Check(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                    &workspace)))
    return 1;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_owner(
      workspace, loomc_workspace_release);
  if (package) {
    if (!configuration->WritePackage(context, workspace, &error))
      return Error(error);
  } else {
    std::unique_ptr<BuiltGgufModule> numerical;
    if (!configuration->BuildModule(context, workspace, &numerical, &error) ||
        !numerical->WriteModule(FLAG_output, options.format, &error))
      return Error(error);
  }
  return 0;
}
