// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Bounded qualification emitter. No model weights or serving outputs are owned
// here; the caller supplies an immutable GGUF and an external output stem.
#include "tools/loomify-gguf/authoring/model-config/configuration.h"
#include "tools/loomify-gguf/authoring/model-config/providers.h"
#include "tools/model/llama_configuration.h"
#include "tools/model/qwen_configuration.h"
#include "lib/dispatch/llama_region.h"
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/testing/ffn_fixture.h"
#include "loomc/target/amdgpu.h"
#include "loomc/source.h"
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
using namespace hrx;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(message.size), message.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  const bool ok = result && loomc_result_succeeded(result);
  if (!ok && result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto *d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  if (result) loomc_result_release(result);
  return ok;
}
bool Configured(loomc_context_t *context, loomc_workspace_t *workspace,
                const model::ModelDescriptor &facts, const std::string &source,
                const std::string &asset, bool llama, loomc_module_t **out) {
  LinkedConfigurationSettings settings;
  settings.profile = llama ? LinkedProfile::Llama3BModel : LinkedProfile::Qwen08BFfn;
  settings.token_capacity = 4;
  LinkedConfiguration configuration;
  loomc_result_t *result = nullptr;
  if (!Ok(CreateLinkedConfiguration(context, workspace, facts, asset, settings,
                                     &configuration, &result)) || !Succeeded(result)) return false;
  auto request = llama ? MakeLlamaLinkedProviders(source)
                       : MakeQwenFfnLinkedProviders(source);
  const std::string family = llama ? "llama" : "qwen";
  request.providers.push_back({"lib/programs/ffn/ffn.loom"});
  request.providers.push_back({"lib/" + family + "/programs/ffn_adapter.loom"});
  const auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer = loom_type_buffer();
  request.roots.push_back({"loomify." + family + ".ffn",
      ConfiguredCallableKind::CommandProgram, {index,index,index}, {buffer,buffer,buffer}});
  result = nullptr;
  return Ok(LinkConfiguredModules(context, workspace, configuration.module.get(),
                                    request, out, &result)) && Succeeded(result);
}
bool Baseline(loomc_context_t *context, loomc_workspace_t *workspace,
              const model::ModelDescriptor &facts, const std::string &source,
              bool llama, loomc_module_t **out) {
  loomc_result_t *result = nullptr;
  if (llama) {
    const auto registry = dispatch::Registry::Llama3B();
    return Ok(dispatch::Load(context, workspace, registry,
        {dispatch::Id::RmsNorm, dispatch::Id::GateUpDecode,
         dispatch::Id::GateUpPrefill, dispatch::Id::DownDecode,
         dispatch::Id::DownPrefill}, source, out, &result)) && Succeeded(result);
  }
  const auto registry = dispatch::Registry::Qwen08B();
  dispatch::QwenNativeGeometry geometry;
  return Ok(dispatch::LoadQwenNative(context, workspace, registry,
      {dispatch::Id::QwenGdnLayer}, facts, {4,1,1,65,128,1}, source,
      &geometry, out, &result)) && Succeeded(result);
}
bool QwenBaseline(ModuleBuilder &module, CommandProgramBuilder &program,
                  size_t layer, bool decode) {
  CommandValues values(program);
  auto index = [&](uint64_t n, Value *out) {
    return Ok(values.Constant(n, LOOM_SCALAR_TYPE_INDEX, out));
  };
  Value n,e,f,l,zero,span,gamma,bytes,normalized,swiglu,epsilon;
  if (!index(decode ? 1 : 4,&n) || !index(1024,&e) || !index(3584,&f) ||
      !index(layer,&l) || !index(0,&zero) || !index(1,&span) ||
      !Ok(values.Parameter(program.Binding(0),"blk.{}.post_attention_norm.weight",
          {l},LOOM_SCALAR_TYPE_F32,{e},&gamma)) ||
      !Ok(values.Config("qwen.model.rms_epsilon",&epsilon)) ||
      !Ok(values.Constant((decode ? 1 : 4)*1024*4,LOOM_SCALAR_TYPE_OFFSET,&bytes)) ||
      !Ok(values.Allocate(bytes,256,&normalized)) ||
      !Ok(values.Constant((decode ? 1 : 4)*3584*4,LOOM_SCALAR_TYPE_OFFSET,&bytes)) ||
      !Ok(values.Allocate(bytes,256,&swiglu))) return false;
  KernelRef norm;
  if (!Ok(module.Kernel(decode ? "qwen38_rmsnorm_f32_decode" :
                                   "qwen38_rmsnorm_f32_prefill",&norm))) return false;
  if (decode) {
    Value packed;
    KernelRef pack;
    ProgramRef gate,down;
    if (!Ok(program.Invoke(norm,{}, {epsilon,program.Binding(1),gamma,normalized})) ||
        !Ok(values.Constant(1024*4,LOOM_SCALAR_TYPE_OFFSET,&bytes)) ||
        !Ok(values.Allocate(bytes,256,&packed)) ||
        !Ok(module.Kernel("ggml_quantize_q8_1_x4_f32",&pack)) ||
        !Ok(program.Invoke(pack,{n,e},{n,e,normalized,packed})) ||
        !Ok(module.Program("qwen38_ffn_gate_up_decode",&gate)) ||
        !Ok(module.Program("qwen38_ffn_down_decode",&down)) ||
        !Ok(program.Call(gate,{l},{program.Binding(0),packed,swiglu})) ||
        !Ok(program.Call(down,{l},{program.Binding(0),swiglu,program.Binding(1)}))) return false;
  } else {
    ProgramRef gate,down;
    if (!Ok(program.Invoke(norm,{n},{n,epsilon,program.Binding(1),gamma,normalized})) ||
        !Ok(module.Program("qwen38_ffn_gate_up_prefill",&gate)) ||
        !Ok(module.Program("qwen38_ffn_down_prefill",&down)) ||
        !Ok(program.Call(gate,{n,n,l,zero,span},
            {program.Binding(0),normalized,program.Binding(2),swiglu})) ||
        !Ok(program.Call(down,{n,n,l,zero,span},
            {program.Binding(0),program.Binding(2),swiglu,program.Binding(1)}))) return false;
  }
  return true;
}
} // namespace
int main(int argc,char **argv) {
  if (argc != 6) {
    std::fprintf(stderr,"usage: emit_real_ffn SOURCE_ROOT GGUF llama|qwen configured|baseline OUT_STEM\n");
    return 1;
  }
  const std::string family=argv[3],mode=argv[4],stem=argv[5];
  if ((family!="llama" && family!="qwen") || (mode!="configured" && mode!="baseline")) return 1;
  const bool llama=family=="llama", configured=mode=="configured";
  std::unique_ptr<FILE,decltype(&std::fclose)> file(std::fopen(argv[2],"rb"),std::fclose);
  if (!file || std::setvbuf(file.get(),nullptr,_IONBF,0)) return 1;
  model::ModelDescriptor facts; std::string error;
  if (!(llama ? model::ExtractLlamaConfiguration(file.get(),&facts,&error)
              : model::ExtractQwenConfiguration(file.get(),&facts,&error))) {
    std::fprintf(stderr,"%s\n",error.c_str()); return 1;
  }
  if (facts.layers.size()<2) return 1;
  loomc_target_environment_t *environment=nullptr;
  loomc_context_t *context=nullptr;
  loomc_workspace_t *workspace=nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),&environment))) return 1;
  loomc_context_target_options_t target{};
  target.type=LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment=environment;
  loomc_context_options_t options{}; options.next=&target;
  if (!Ok(loomc_context_create(&options,loomc_allocator_system(),&context)) ||
      !Ok(loomc_workspace_create(nullptr,loomc_allocator_system(),&workspace))) return 1;
  loomc_module_t *definitions=nullptr;
  if (!(configured ? Configured(context,workspace,facts,argv[1],argv[2],llama,&definitions)
                   : Baseline(context,workspace,facts,argv[1],llama,&definitions))) return 1;
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context,workspace,definitions,&module))) return 1;
  loomc_module_release(definitions);
  std::vector<RootRef> roots;
  // Llama layers0/1 use Q6 down in the local checkpoint; layer3 supplies its
  // actual Q4 down variant. Qwen's first two layers cover its selected profile.
  const std::vector<size_t> selected_layers = llama
      ? std::vector<size_t>{0,1,3} : std::vector<size_t>{0,1};
  if (facts.layers.size() <= selected_layers.back()) return 1;
  for (size_t layer : selected_layers) for (bool decode : {false,true}) {
    std::unique_ptr<CommandProgramBuilder> program;
    const auto name="real."+family+".ffn.layer"+std::to_string(layer)+(decode ? ".decode" : ".prefill");
    if (!Ok(module->DefineProgram(name,{},3,&program))) return 1;
    if (configured) {
      CommandValues values(*program); Value l,p,n; ProgramRef ffn;
      if (!Ok(values.Constant(layer,LOOM_SCALAR_TYPE_INDEX,&l)) ||
          !Ok(values.Constant(decode?1:0,LOOM_SCALAR_TYPE_INDEX,&p)) ||
          !Ok(values.Constant(decode?1:4,LOOM_SCALAR_TYPE_INDEX,&n)) ||
          !Ok(module->Program("loomify."+family+".ffn",&ffn)) ||
          !Ok(program->Call(ffn,{l,p,n},{program->Binding(0),program->Binding(1),program->Binding(2)}))) return 1;
    } else if (llama) {
      const auto phase=decode ? operations::Phase::Decode : operations::Phase::Prefill;
      auto owner=dispatch::testing::FfnFixture(facts,phase,decode?1:4,&error,layer);
      const auto registry=dispatch::Registry::Llama3B();
      std::unique_ptr<dispatch::FfnRegion> region;
      if (!owner || !dispatch::FfnRegion::Select(owner,{llama::LlamaRegionKind::Ffn,layer,1},registry,&region,&error)) {
        std::fprintf(stderr,"%s\n",error.c_str()); return 1;
      }
      dispatch::NativeEmissionSession session(registry,owner->graph(),*module,*program);
      dispatch::NativeLlamaRows rows;
      rows.row_begin=owner->inputs().row_begin; rows.row_count=owner->inputs().row_count;
      rows.row_capacity=decode?1:4; rows.phase=phase; rows.packet=program->Binding(2);
      if (!Ok(session.Bind(owner->inputs().residual,program->Binding(1))) ||
          !Ok(session.LlamaRows(rows)) || !region->Emit(session,program->Binding(0),&error)) {
        std::fprintf(stderr,"%s\n",error.c_str()); return 1;
      }
    } else if (!QwenBaseline(*module,*program,layer,decode)) return 1;
    if (!Ok(program->Finish())) return 1;
    roots.push_back(program->Ref());
  }
  loomc_module_t *linked=nullptr; loomc_result_t *result=nullptr;
  if (!Ok(module->LinkAndVerify(roots,&linked,&result)) || !Succeeded(result)) return 1;
  for (bool bytecode : {false,true}) {
    const auto path=stem+(bytecode?".loombc":".loom");
    loomc_module_serialize_options_t serialize{};
    serialize.format=bytecode?LOOMC_SOURCE_FORMAT_BYTECODE:LOOMC_SOURCE_FORMAT_TEXT;
    if (!Ok(loomc_module_serialize_to_path(linked,&serialize,loomc_make_cstring_view(path.c_str()),loomc_allocator_system()))) return 1;
  }
  loomc_module_release(linked); module.reset();
  loomc_workspace_release(workspace); loomc_context_release(context);
  loomc_target_environment_release(environment);
  return 0;
}
