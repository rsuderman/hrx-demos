// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual GGUF facts plus maintained wrappers; arithmetic/state oracles live in
// the existing kernel tests. Old-versus-new parity is external migration
// evidence.
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "tools/model/model_descriptor.h"
#include <fstream>
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
  if (argc != 4) {
    std::fprintf(stderr, "usage: emitter LIBRARY_ROOT GGUF OUTPUT_DIRECTORY\n");
    return 1;
  }
  std::filesystem::create_directories(argv[3]);
  std::unique_ptr<FILE, decltype(&std::fclose)> input(std::fopen(argv[2], "rb"),
                                                      std::fclose);
  if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0))
    return 1;
  model::ModelDescriptor descriptor;
  std::string error;
  model::GgufCatalog catalog;
  if (!model::ReadGgufCatalog(input.get(), &catalog, &error) ||
      !model::LoadModelDescriptor(catalog, &descriptor, &error) ||
      (descriptor.family != "lfm3" && descriptor.family != "spark2_5")) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  ConfigurationFacts config;
  if (!DeriveConfiguration(descriptor, argv[2], {3, 64}, &config, &error)) {
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
  const bool lfm = descriptor.family == "lfm3";
  std::vector<const char *> roots =
      lfm ? std::vector<const char *>{"validated.lfm3.layer0",
                                      "validated.lfm3.layer2",
                                      "validated.lfm3.layer4"}
          : std::vector<const char *>{"validated.spark2_5.layer0",
                                      "validated.spark2_5.layer3"};
  auto product =
      (std::filesystem::path(argv[1]) /
       ("lib/architectures/" + descriptor.family + "/layer_fixture.loombc"))
          .string();
  loomc_module_t *module = nullptr;
  if (!LinkConfiguredProducts(
          context, workspace, config.source,
          {product, (std::filesystem::path(argv[1]) /
                     "lib/platforms/amdgpu/gfx11/targets.loombc")
                        .string()},
          roots, true, &module, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  bool ok = true;
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t serialize{};
    serialize.format = format;
    auto path = std::string(argv[3]) + "/configured_layers" +
                (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc");
    ok = Ok(loomc_module_serialize_to_path(
             module, &serialize, loomc_make_cstring_view(path.c_str()),
             loomc_allocator_system())) &&
         ok;
  }
  const uint64_t hidden = descriptor.embedding_width, pool = 64;
  const uint64_t base = 768 + 4 * 129;
  uint64_t attention = 0, convolution = 0, kv = 0;
  for (const auto &layer : descriptor.layers) {
    if (layer.kind == model::LayerKind::kAttention) {
      ++attention;
      kv = layer.kv_heads * descriptor.key_head_width;
    } else if (layer.kind == model::LayerKind::kShortConvolution) {
      ++convolution;
    }
  }
  const uint64_t kvbytes = attention * 2 * pool * kv * 2;
  const uint64_t historybase = base + kvbytes;
  const uint64_t statebytes = historybase + convolution * 6 * hidden * 4;
  std::ofstream layout(std::string(argv[3]) + "/layout.json");
  layout << "{\"family\":\"" << descriptor.family << "\",\"hidden\":" << hidden
         << ",\"rows\":3,\"pool\":" << pool << ",\"kv\":" << kv
         << ",\"attention_layers\":" << attention << ",\"cache_base\":" << base
         << ",\"history_base\":" << historybase
         << ",\"state_bytes\":" << statebytes << ",\"blocks\":[";
  const auto selected =
      lfm ? std::vector<uint32_t>{0, 2, 4} : std::vector<uint32_t>{0, 3};
  bool first = true;
  for (auto layer : selected) {
    if (!first)
      layout << ",";
    first = false;
    const auto &l = descriptor.layers.at(layer);
    const bool conv = l.kind == model::LayerKind::kShortConvolution;
    layout << "{\"layer\":" << layer
           << ",\"convolution\":" << (conv ? "true" : "false")
           << ",\"ordinal\":"
           << (conv ? *l.convolution_ordinal : *l.attention_ordinal)
           << ",\"window\":" << l.attention_window << ",\"bindings\":[";
    bool first_binding = true;
    for (const auto &binding : config.parameters) {
      if (binding.layer != layer &&
          (conv || binding.role != "rope_frequencies"))
        continue;
      if (!first_binding)
        layout << ",";
      first_binding = false;
      layout << "\"" << binding.tensor.name << "\"";
    }
    layout << "]}";
  }
  layout << "]}\n";
  ok = bool(layout) && ok;
  loomc_module_release(module);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(env);
  return ok ? 0 : 1;
}
