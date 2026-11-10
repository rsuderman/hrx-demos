// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual GGUF facts plus maintained wrappers; arithmetic/state oracles live in
// the existing kernel tests. Old-versus-new parity is external migration evidence.
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
  bool ok=loomc_status_is_ok(status);
  if (!ok) { auto m=loomc_status_message(status); std::fprintf(stderr,"%.*s\n",int(m.size),m.data); }
  loomc_status_free(status);return ok;
}
}
int main(int argc,char **argv) {
  const bool attention_partial = argc == 5 && std::string(argv[4]) == "--attention-partial";
  const bool partial = attention_partial || (argc == 5 && std::string(argv[4]) == "--partial-active");
  if (argc != 4 && !partial) {
    std::fprintf(stderr,"usage: emitter LIBRARY_ROOT GGUF OUTPUT_DIRECTORY [--partial-active | --attention-partial]\n");
    return 1;
  }
  std::filesystem::create_directories(argv[3]);
  std::unique_ptr<FILE,decltype(&std::fclose)> input(std::fopen(argv[2],"rb"),std::fclose);
  if (!input || std::setvbuf(input.get(),nullptr,_IONBF,0)) return 1;
  model::ModelDescriptor descriptor;std::string error;
  model::GgufCatalog catalog;
  if (!model::ReadGgufCatalog(input.get(), &catalog, &error) ||
      !model::LoadModelDescriptor(catalog, &descriptor, &error) ||
      descriptor.family != "qwen35") {
    std::fprintf(stderr,"%s\n",error.c_str());return 1;
  }
  ConfigurationFacts config;
  if (!DeriveConfiguration(descriptor,argv[2],{partial ? 3u : 1u,64},&config,&error)) {
    std::fprintf(stderr,"%s\n",error.c_str());return 1;
  }
  loomc_target_environment_t *env=nullptr;loomc_context_t *context=nullptr;loomc_workspace_t *workspace=nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),&env))) return 1;
  loomc_context_target_options_t target{};target.type=LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;target.target_environment=env;
  loomc_context_options_t options{};options.next=&target;
  if (!Ok(loomc_context_create(&options,loomc_allocator_system(),&context)) ||
      !Ok(loomc_workspace_create(nullptr,loomc_allocator_system(),&workspace))) return 1;
  std::vector<const char *> roots={"validated.qwen.gdn0", "validated.qwen.attention3"};
  if (partial) roots={"validated.qwen.gdn0", "validated.qwen.gdn0_t3"};
  if (attention_partial) roots={"validated.qwen.attention3", "validated.qwen.attention3_t3"};
  auto product=(std::filesystem::path(argv[1])/"lib/architectures/qwen35/block_fixture.loombc").string();
  loomc_module_t *module=nullptr;
  if (!LinkConfiguredProducts(context,workspace,config.source, {product},roots,true,&module,&error)) {
    std::fprintf(stderr,"%s\n",error.c_str());return 1;
  }
  bool ok=true;
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT,LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t serialize{};serialize.format=format;
    auto path=std::string(argv[3])+"/configured_blocks"+(format==LOOMC_SOURCE_FORMAT_TEXT?".loom":".loombc");
    ok=Ok(loomc_module_serialize_to_path(module,&serialize,loomc_make_cstring_view(path.c_str()),loomc_allocator_system()))&&ok;
  }
  // Test allocation geometry from the admitted descriptor. No alternate
  // production configuration or model sequence is emitted by this fixture.
  if (!descriptor.ssm || (descriptor.embedding_width != 1024 &&
                          descriptor.embedding_width != 5120)) return 1;
  const uint64_t history=4*descriptor.ssm->convolution_history_elements;
  const uint64_t recurrent=4*descriptor.ssm->recurrent_state_elements;
  const uint64_t plane=2*descriptor.layers.front().kv_heads*descriptor.key_head_width;
  const uint64_t attention_width=descriptor.layers.at(3).query_heads*descriptor.key_head_width;
  std::ofstream layout(std::string(argv[3])+"/layout.json");
  layout << "{\"hidden\":" << descriptor.embedding_width
         << ",\"residual_bytes\":" << 512*descriptor.embedding_width*4
         << ",\"gdn_history_bytes\":" << history
         << ",\"gdn_recurrent_bytes\":" << recurrent
         << ",\"gdn_state_bytes\":" << history+recurrent
         << ",\"plane_token_bytes\":" << plane
         << ",\"pool\":64,\"map_origin\":2048,\"state_origin\":256,\"capture_bytes\":"
         << 512*(attention_partial ? attention_width : descriptor.ssm->inner_width)*4;
  if (attention_partial) layout << ",\"attention_width\":" << attention_width;
  layout << ",\"bindings\":{";
  for (uint32_t layer : {0u,3u}) {
    if (layer) layout << ",";
    layout << "\"" << layer << "\":[";
    bool first=true;
    for (const auto &binding : config.parameters) if (binding.layer == layer) {
      if (!first) layout << ",";
      first=false;
      layout << "\"" << binding.tensor.name << "\"";
    }
    layout << "]";
  }
  layout << "}}\n";
  ok=bool(layout)&&ok;
  loomc_module_release(module);loomc_workspace_release(workspace);loomc_context_release(context);loomc_target_environment_release(env);
  return ok?0:1;
}
