// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual GGUF facts plus maintained wrappers; arithmetic/state oracles live in
// the existing kernel tests. Old-versus-new parity is external migration
// evidence.
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "tools/model/model_descriptor.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <filesystem>
#include <memory>
using namespace hrx;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "usage: emitter LIBRARY_ROOT GGUF llama|qwen OUTPUT_STEM\n");
    return 1;
  }
  const std::string family = argv[3];
  if (family != "llama" && family != "qwen")
    return 1;
  std::unique_ptr<FILE, decltype(&std::fclose)> input(std::fopen(argv[2], "rb"),
                                                      std::fclose);
  if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0))
    return 1;
  model::ModelDescriptor descriptor;
  std::string error;
  model::GgufCatalog catalog;
  if (!model::ReadGgufCatalog(input.get(), &catalog, &error) ||
      !model::LoadModelDescriptor(catalog, &descriptor, &error) ||
      descriptor.family != (family == "llama" ? "llama" : "qwen35")) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  ConfigurationFacts config;
  if (!DeriveConfiguration(descriptor, argv[2], {4, 512}, &config, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  loomc_target_environment_t *env = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &env)))
    return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = env;
  loomc_context_options_t options{};
  options.next = &target;
  if (!Ok(loomc_context_create(&options, loomc_allocator_system(), &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  std::vector<std::string> names;
  for (int layer :
       family == "llama" ? std::vector<int>{0, 1, 3} : std::vector<int>{0, 1})
    for (const char *phase : {"prefill", "decode"})
      names.push_back("real." + family + ".ffn.layer" + std::to_string(layer) +
                      "." + phase);
  std::vector<const char *> roots;
  for (const auto &name : names)
    roots.push_back(name.c_str());
  auto product = (std::filesystem::path(argv[1]) / "lib/programs/ffn/testing" /
                  (family + "_fixture.loombc"))
                     .string();
  loomc_module_t *module = nullptr;
  if (!LinkConfiguredProducts(context, workspace, config.source, {product},
                              roots, true, &module, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  bool ok = true;
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t serialize{};
    serialize.format = format;
    auto path = std::string(argv[4]) +
                (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc");
    ok = Ok(loomc_module_serialize_to_path(
             module, &serialize, loomc_make_cstring_view(path.c_str()),
             loomc_allocator_system())) &&
         ok;
  }
  loomc_module_release(module);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(env);
  return ok ? 0 : 1;
}
