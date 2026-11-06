// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
// Emits actual maintained and historical registered layers for an external
// real-weight numerical/state gate. The historical C++ body is a test oracle;
// it is never linked into the maintained export route.
#include "lib/dispatch/llama_region.h"
#include "lib/dispatch/parameters.h"
#include "lib/llama/program.h"
#include "tools/loomify-gguf/authoring/model-config/configuration.h"
#include "tools/loomify-gguf/authoring/model-config/providers.h"
#include "tools/model/llama_configuration.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
using namespace hrx;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) { const auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data); }
  loomc_status_free(status); return ok;
}
bool Result(loomc_result_t *r) {
  const bool ok = r && loomc_result_succeeded(r);
  if (!ok && r) for (size_t i=0;i<loomc_result_diagnostic_count(r);++i) {
    const auto *d=loomc_result_diagnostic_at(r,i);
    std::fprintf(stderr,"%.*s\n",int(d->message.size),d->message.data);
  }
  if (r) loomc_result_release(r); return ok;
}
bool Save(loomc_module_t *module, const std::string &stem) {
  for (const auto format : {LOOMC_SOURCE_FORMAT_TEXT,LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t options{}; options.format=format;
    const auto path=stem+(format==LOOMC_SOURCE_FORMAT_TEXT?".loom":".loombc");
    if (!Ok(loomc_module_serialize_to_path(module,&options,
          loomc_make_cstring_view(path.c_str()),loomc_allocator_system()))) return false;
  }
  return true;
}
std::string Name(unsigned layer,bool decode) {
  return "validated.layer"+std::to_string(layer)+(decode?".decode":".prefill");
}
bool Baseline(loomc_context_t *context,loomc_workspace_t *workspace,
              const model::ModelDescriptor &facts,const std::string &source,
              const LlamaLinkedLayout &layout,const std::string &output) {
  auto registry=dispatch::Registry::Llama3B();
  loomc_module_t *raw=nullptr; loomc_result_t *result=nullptr;
  if (!Ok(dispatch::LoadLlamaNative(context,workspace,registry,
      llama::LlamaProgramDispatches(),source,
      dispatch::AttentionNativeConfig{facts.layers.size(),layout.context_capacity,
                                     layout.pool_capacity},&raw,&result)) || !Result(result)) return false;
  std::unique_ptr<loomc_module_t,decltype(&loomc_module_release)> definitions(raw,loomc_module_release);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context,workspace,raw,&module))) return false;
  std::vector<RootRef> roots; std::string error;
  for (unsigned layer : {0u,1u}) for (bool decode : {false,true}) {
    const auto phase=decode?operations::Phase::Decode:operations::Phase::Prefill;
    llama::LlamaBuildOptions options{phase,decode?1:layout.token_capacity,
                                    layout.context_capacity,128,"real-layer-gate"};
    options.attention_policy=operations::NumericalPolicy::LlamaPacketMixed;
    std::shared_ptr<const llama::LlamaStructure> owner;
    if (!llama::BuildLlamaStructure(facts,options,&owner,&error)) {
      std::fprintf(stderr,"%s\n",error.c_str()); return false; }
    std::unique_ptr<CommandProgramBuilder> program;
    if (!Ok(module->DefineProgram(Name(layer,decode),{},5,&program))) return false;
    const auto &in=owner->inputs();
    dispatch::NativeEmissionSession session(registry,owner->graph(),*module,*program);
    dispatch::NativeLlamaRows rows;
    rows.row_begin=in.row_begin; rows.row_count=in.row_count;
    rows.count_control=in.count_control; rows.packet_positions=in.positions;
    rows.row_capacity=options.row_capacity; rows.phase=phase; rows.packet=program->Binding(2);
    if (!Ok(session.Bind(in.residual,program->Binding(1))) ||
        !Ok(session.LlamaRows(rows))) return false;
    llama::LlamaRegionView attention;
    if (!owner->select({llama::LlamaRegionKind::Attention,layer,1},&attention,&error) ||
        !dispatch::BindRegionParameters(*owner,attention,program->Binding(0),session,&error)) {
      std::fprintf(stderr,"%s\n",error.c_str());return false; }
    const auto &ops=owner->graph().invocations();
    for (size_t i=attention.operation_begin;i<attention.operation_begin+4;++i)
      if (!Ok(session.EmitExisting(i==attention.operation_begin?dispatch::Id::RmsNorm:
                                  dispatch::Id::Projection,ops[i]))) return false;
    dispatch::AttentionSequenceOptions sequence;
    sequence.row_capacity=options.row_capacity;sequence.phase=phase;sequence.layer=layer;
    sequence.cache_base=layout.cache_base;sequence.map_origin=layout.map_origin;
    if (!Ok(session.EmitAttentionRegion(*owner,{llama::LlamaRegionKind::Attention,layer,1},
                                       sequence,program->Binding(3),program->Binding(4))) ||
        !Ok(session.EmitExisting(dispatch::Id::AttentionOutput,ops[attention.operation_end-1]))) return false;
    std::unique_ptr<dispatch::FfnRegion> ffn;
    if (!dispatch::FfnRegion::Select(owner,{llama::LlamaRegionKind::Ffn,layer,1},registry,&ffn,&error) ||
        !ffn->Emit(session,program->Binding(0),&error)) {
      std::fprintf(stderr,"%s\n",error.c_str());return false; }
    if (!Ok(program->Finish())) return false;
    roots.push_back(program->Ref());
  }
  raw=nullptr;result=nullptr;
  if (!Ok(module->LinkAndVerify(roots,&raw,&result)) || !Result(result)) return false;
  std::unique_ptr<loomc_module_t,decltype(&loomc_module_release)> linked(raw,loomc_module_release);
  return Save(raw,output+"/baseline_layer");
}
}
int main(int argc,char **argv) {
  if (argc!=4) { std::fprintf(stderr,"usage: emitter SOURCE_ROOT GGUF OUTPUT_DIRECTORY\n");return 1; }
  std::filesystem::create_directories(argv[3]);
  std::unique_ptr<FILE,decltype(&std::fclose)> input(std::fopen(argv[2],"rb"),std::fclose);
  model::ModelDescriptor facts;std::string error;
  if (!input || !model::ExtractLlamaConfiguration(input.get(),&facts,&error)) {
    std::fprintf(stderr,"%s\n",error.c_str());return 1; }
  loomc_target_environment_t *environment=nullptr;
  loomc_context_t *context=nullptr;loomc_workspace_t *workspace=nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),&environment))) return 1;
  loomc_context_target_options_t target{};target.type=LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment=environment;loomc_context_options_t context_options{};context_options.next=&target;
  if (!Ok(loomc_context_create(&context_options,loomc_allocator_system(),&context)) ||
      !Ok(loomc_workspace_create(nullptr,loomc_allocator_system(),&workspace))) return 1;
  LinkedConfigurationSettings settings;settings.token_capacity=17;settings.context_capacity=512;
  LinkedConfiguration configuration;loomc_result_t *result=nullptr;
  if (!Ok(CreateLinkedConfiguration(context,workspace,facts,argv[2],settings,&configuration,&result)) || !Result(result)) return 1;
  auto request=MakeLlamaLinkedProviders(argv[1]);
  for (const char *path : {"lib/programs/ffn/interface.loom","lib/programs/ffn/ffn.loom",
                          "lib/llama/programs/ffn_adapter.loom","lib/llama/programs/attention.loom",
                          "lib/llama/programs/layer.loom","lib/llama/testing/testdata/configured_layer_roots.loom"})
    request.providers.push_back({path,LOOMC_LINK_PROVIDER_ROLE_INPUT});
  for (unsigned layer : {0u,1u}) for (bool decode : {false,true})
    request.roots.push_back({Name(layer,decode),ConfiguredCallableKind::CommandProgram,{},
                             std::vector<loom_type_t>(5,loom_type_buffer())});
  loomc_module_t *linked=nullptr;result=nullptr;
  if (!Ok(LinkConfiguredModules(context,workspace,configuration.module.get(),request,&linked,&result)) || !Result(result)) return 1;
  const bool ok=Save(linked,std::string(argv[3])+"/configured_layer") &&
    Baseline(context,workspace,facts,argv[1],*configuration.facts.llama_layout,argv[3]);
  std::ofstream layout(std::string(argv[3])+"/layout.json");
  const auto &l=*configuration.facts.llama_layout;
  layout<<"{\"rows\":17,\"layers\":"<<l.layer_count<<",\"pool\":"<<l.pool_capacity
        <<",\"cache_base\":"<<l.cache_base<<",\"map_origin\":"<<l.map_origin
        <<",\"cache_bytes\":"<<l.prefill.cache<<",\"map_bytes\":"<<l.prefill.map<<"}\n";
  loomc_module_release(linked);configuration.module.reset();
  loomc_workspace_release(workspace);loomc_context_release(context);loomc_target_environment_release(environment);
  return ok?0:1;
}
