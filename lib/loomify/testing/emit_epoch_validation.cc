// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
// Finite actual-GGUF epoch contract gate through the public numerical builder.
#include "lib/loomify/gguf_model_builder.h"
#include "loomc/target/amdgpu.h"
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(message.size), message.data);
  }
  loomc_status_free(status);
  return ok;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 4 && argc != 6) {
    std::fprintf(
        stderr,
        "usage: emitter LIBRARY_ROOT GGUF OUTPUT_DIRECTORY [PREFILL CONTEXT]\n");
    return 1;
  }
  std::string error;
  std::unique_ptr<GgufModelBuilder> builder;
  if (!GgufModelBuilder::Open(argv[2], &builder, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const auto &descriptor = builder->descriptor();
  if (descriptor.family != "llama" && descriptor.family != "qwen35" &&
      descriptor.family != "gemma3")
    return 1;
  GgufBuildOptions settings;
  settings.library_root = argv[1];
  settings.prefill_capacity = 8;
  settings.context_capacity = descriptor.family == "gemma3" ? 576 : 64;
  if (argc == 6) {
    auto read = [](const char *text, uint64_t *value) {
      const std::string word(text);
      const auto parsed =
          std::from_chars(word.data(), word.data() + word.size(), *value);
      return parsed.ec == std::errc() &&
             parsed.ptr == word.data() + word.size();
    };
    if (!read(argv[4], &settings.prefill_capacity) ||
        !read(argv[5], &settings.context_capacity))
      return 1;
  }
  std::unique_ptr<GgufConfiguration> configuration;
  if (!builder->Configure(settings, nullptr, &configuration, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  loomc_target_environment_t *environment = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &environment)))
    return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *context = nullptr;
  const bool context_ok =
      Ok(loomc_context_create(&options, loomc_allocator_system(), &context));
  loomc_target_environment_release(environment);
  if (!context_ok)
    return 1;
  loomc_workspace_t *workspace = nullptr;
  if (!Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  std::unique_ptr<BuiltGgufModule> module;
  if (!configuration->BuildModule(context, workspace, &module, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::filesystem::create_directories(argv[3]);
  bool ok = true;
  for (const auto format :
       {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t serialization{};
    serialization.format = format;
    const auto path =
        (std::filesystem::path(argv[3]) /
         (format == LOOMC_SOURCE_FORMAT_TEXT ? "epoch.loom" : "epoch.loombc"))
            .string();
    ok =
        Ok(loomc_module_serialize_to_path(module->module(), &serialization,
                                          loomc_make_cstring_view(path.c_str()),
                                          loomc_allocator_system())) &&
        ok;
  }
  std::ofstream facts(std::filesystem::path(argv[3]) / "configuration.loom");
  facts << configuration->configuration_source();
  std::ofstream layout(std::filesystem::path(argv[3]) / "layout.json");
  layout << "{\"family\":\"" << descriptor.family
         << "\",\"tokens\":" << settings.prefill_capacity
         << ",\"context\":" << settings.context_capacity
         << ",\"hidden\":" << descriptor.embedding_width;
  if (const auto *decode = configuration->specialized_decode_root())
    layout << ",\"decode_root\":\"" << decode << "\"";
  if (descriptor.family == "gemma3") {
    layout << ",\"layer_windows\":[";
    for (size_t i = 0; i < descriptor.layers.size(); ++i)
      layout << (i ? "," : "") << descriptor.layers[i].attention_window;
    layout << "],\"layer_rope_bases\":[";
    for (size_t i = 0; i < descriptor.layers.size(); ++i)
      layout << (i ? "," : "") << descriptor.layers[i].rope_base;
    layout << "]";
  }
  layout << "}\n";
  ok = bool(facts) && bool(layout) && ok;
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  return ok ? 0 : 1;
}
