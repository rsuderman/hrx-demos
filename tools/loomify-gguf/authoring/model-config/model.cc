// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model-config/model.h"
#include "loomc/pass.h"
#include <algorithm>
#include <utility>
namespace hrx::loomify_gguf {
loomc_status_t BuildConfiguredLlamaModel(
    loomc_context_t *context, loomc_workspace_t *workspace,
    const LinkedConfiguration &configuration,
    const ConfiguredLinkRequest &providers, ConfiguredLlamaModel *out,
    loomc_result_t **result) {
  if (result)
    *result = nullptr;
  if (!out || !result || !configuration.module ||
      configuration.facts.profile != LinkedProfile::Llama3BModel ||
      !configuration.facts.llama_layout || configuration.facts.parameters.empty())
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                            "Llama model requires admitted Llama configuration, layout and outputs");
  const auto &asset = configuration.facts.parameters.front().source_asset;
  if (asset.empty() || std::any_of(configuration.facts.parameters.begin(),
      configuration.facts.parameters.end(), [&](const ParameterRequirement &p) {
        return p.source_asset != asset;
      }))
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                            "Llama model requires one admitted source asset");
  auto request = providers;
  // Maintained architecture units jointly own private helper definitions.
  // Numerical providers remain libraries with their exported entry points.
  for (const char *path : {"lib/programs/ffn/interface.loom",
                           "lib/programs/ffn/ffn.loom",
                           "lib/llama/programs/ffn_adapter.loom",
                           "lib/llama/programs/attention.loom",
                           "lib/llama/programs/layer.loom",
                           "lib/llama/programs/model.loom",
                           "lib/llama/programs/epoch.loom",
                           "tools/loomify-gguf/serving/epoch.loom"}) {
    if (std::none_of(request.providers.begin(), request.providers.end(),
                     [&](const ConfiguredProvider &p) { return p.path == path; }))
      request.providers.push_back({path, LOOMC_LINK_PROVIDER_ROLE_INPUT});
  }
  const auto buffer = loom_type_buffer();
  for (const char *name : {"generated.llama_prefill", "generated.llama_decode",
                           "generated.llama_epoch"})
    request.roots.push_back({name, ConfiguredCallableKind::CommandProgram, {},
                             std::vector<loom_type_t>(7, buffer)});
  for (const char *name : {"runner.llama32.prefill_token_count",
                           "llama.model.hidden_size", "llama.model.layer_count",
                           "llama.model.vocabulary_size",
                           "llama.model.embedding_weight_bytes",
                           "llama.model.output_weight_bytes"})
    request.required_configuration.push_back(name);
  for (const char *name : {"loomify.llama.layer", "llama32_head_count",
                           "llama32_token_embedding_active_q6k_f32",
                           "llm_output_last_greedy", "llama32_finish_isolated"})
    request.required_definitions.push_back(name);
  loomc_module_t *module = nullptr;
  auto status = LinkConfiguredModules(context, workspace, configuration.module.get(),
                                      request, &module, result);
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> owned(
      module, loomc_module_release);
  if (!loomc_status_is_ok(status) || !*result ||
      !loomc_result_succeeded(*result) || !owned)
    return status;
  // The pinned command materializer selects templates before unrolling loops.
  // Normalize the authored model loop first so mixed per-layer encodings are
  // exact when its calls reach the shared FFN's numerical family selection.
  // These native passes transform the maintained source; C++ adds no sequence.
  loomc_result_release(*result);
  *result = nullptr;
  loomc_pass_program_t *raw_passes = nullptr;
  status = loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view("canonicalize,unroll-scf-for,canonicalize"),
      nullptr, loomc_allocator_system(), &raw_passes, result);
  std::unique_ptr<loomc_pass_program_t, decltype(&loomc_pass_program_release)> passes(
      raw_passes, loomc_pass_program_release);
  if (!loomc_status_is_ok(status) || !*result ||
      !loomc_result_succeeded(*result) || !passes)
    return status;
  loomc_result_release(*result);
  *result = nullptr;
  status = loomc_pass_program_run(passes.get(), workspace, owned.get(), nullptr,
                                  loomc_allocator_system(), result);
  if (!loomc_status_is_ok(status) || !*result || !loomc_result_succeeded(*result))
    return status;
  ConfiguredLlamaModel built;
  built.module = std::move(owned);
  built.layout = *configuration.facts.llama_layout;
  built.source_asset = asset;
  *out = std::move(built);
  return status;
}
} // namespace hrx::loomify_gguf
