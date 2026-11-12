// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual GGUF facts plus maintained wrappers; arithmetic/state oracles live in
// the existing kernel tests. Old-versus-new parity is external migration evidence.
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "tools/model/model_descriptor.h"
#include <fstream>
#include <cmath>
#include <iomanip>
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <cstdlib>
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
  if (argc!=4 && argc!=5) { std::fprintf(stderr,"usage: emitter LIBRARY_ROOT GGUF OUTPUT_DIRECTORY [TOKENS:8|17|64]\n");return 1; }
  std::filesystem::create_directories(argv[3]);
  std::unique_ptr<FILE,decltype(&std::fclose)> input(std::fopen(argv[2],"rb"),std::fclose);
  if (!input || std::setvbuf(input.get(),nullptr,_IONBF,0)) return 1;
  model::ModelDescriptor descriptor;std::string error;
  model::GgufCatalog catalog;
  if (!model::ReadGgufCatalog(input.get(), &catalog, &error) ||
      !model::LoadModelDescriptor(catalog, &descriptor, &error) ||
      descriptor.family != "qwen2") {
    std::fprintf(stderr,"%s\n",error.c_str());return 1;
  }
  unsigned long tokens = 17;
  if (argc == 5) {
    char *end = nullptr;
    tokens = std::strtoul(argv[4], &end, 10);
    if (!end || *end || (tokens != 8 && tokens != 17 && tokens != 64)) return 2;
  }
  ConfigurationFacts config;
  if (!DeriveConfiguration(descriptor,argv[2],{tokens,128},&config,&error)) {
    std::fprintf(stderr,"%s\n",error.c_str());return 1;
  }
  loomc_target_environment_t *env=nullptr;loomc_context_t *context=nullptr;loomc_workspace_t *workspace=nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),&env))) return 1;
  loomc_context_target_options_t target{};target.type=LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;target.target_environment=env;
  loomc_context_options_t options{};options.next=&target;
  if (!Ok(loomc_context_create(&options,loomc_allocator_system(),&context)) ||
      !Ok(loomc_workspace_create(nullptr,loomc_allocator_system(),&workspace))) return 1;
  std::vector<const char *> roots={"validated.qwen2.layer0", "validated.qwen2.ffn0", "validated.qwen2.attention0", "validated.qwen2.layer1", "validated.qwen2.layer2", "validated.qwen2.layer4", "validated.qwen2.layer5"};
  const auto library = std::filesystem::path(argv[1]);
  auto product=(library/"lib/architectures/qwen2/layer_fixture.loombc").string();
  loomc_module_t *module=nullptr;
  if (!LinkConfiguredProducts(context,workspace,config.source, {product, (library/"lib/platforms/amdgpu/gfx11/targets.loombc").string(), (library/"lib/platforms/amdgpu/gfx11/projection/implementation.loombc").string()},roots,true,&module,&error)) {
    std::fprintf(stderr,"%s\n",error.c_str());return 1;
  }
  bool ok=true;
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT,LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t serialize{};serialize.format=format;
    auto path=std::string(argv[3])+"/configured_layer"+(format==LOOMC_SOURCE_FORMAT_TEXT?".loom":".loombc");
    ok=Ok(loomc_module_serialize_to_path(module,&serialize,loomc_make_cstring_view(path.c_str()),loomc_allocator_system()))&&ok;
  }
  const uint64_t layers=descriptor.layers.size(),pool=128,base=768+4*129;
  const auto hidden=descriptor.embedding_width,head=descriptor.key_head_width;
  const auto qheads=descriptor.layers.front().query_heads,kvheads=descriptor.layers.front().kv_heads;
  const auto kv=kvheads*head;
  std::ofstream layout(std::string(argv[3])+"/layout.json");
  layout << "{\"rows\":" << tokens << ",\"layers\":" << layers << ",\"pool\":" << pool
         << ",\"cache_base\":" << base << ",\"map_origin\":2048,\"cache_bytes\":"
         << base+4*layers*pool*kv << ",\"map_bytes\":2056,\"hidden\":" << hidden
         << ",\"head\":" << head << ",\"query_heads\":" << qheads << ",\"kv_heads\":" << kvheads
         << ",\"ffn\":" << descriptor.layers.front().feed_forward_width
         << ",\"epsilon\":" << std::setprecision(17) << descriptor.rms_epsilon
         << ",\"rope_log_step\":" << float(-2*std::log(descriptor.rope.base)/head)
         << ",\"attention_scale\":" << float(1/std::sqrt(double(head))) << ",\"tensors\":[";
  bool first = true;
  for (const auto &tensor : catalog.tensors) {
    if (tensor.name.rfind("blk.", 0) != 0) continue;
    if (!first) layout << ",";
    first = false;
    layout << "{\"name\":\"" << tensor.name << "\",\"tag\":" << tensor.storage_type
           << ",\"offset\":" << tensor.absolute_offset << ",\"bytes\":" << tensor.byte_size
           << ",\"input\":" << tensor.normalized_extents[0]
           << ",\"output\":" << tensor.normalized_extents[1] << "}";
  }
  layout << "]}\n";
  ok=bool(layout)&&ok;
  loomc_module_release(module);loomc_workspace_release(workspace);loomc_context_release(context);loomc_target_environment_release(env);
  return ok?0:1;
}
