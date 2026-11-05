// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/model_export.h"
#include "lib/llama/program.h"
#include "loomc/link.h"
#include "tools/loomify-gguf/serving/epoch.h"
#include <algorithm>
#include <filesystem>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
template <class T, void (*Release)(T *)> struct Deleter {
  void operator()(T *p) const { Release(p); }
};
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
bool Check(loomc_status_t s, std::string *error) {
  bool ok = loomc_status_is_ok(s);
  if (!ok && error) {
    auto m = loomc_status_message(s);
    error->assign(m.data, m.size);
  }
  loomc_status_free(s);
  return ok;
}
bool Result(loomc_result_t *r, std::string *error) {
  if (r && loomc_result_succeeded(r))
    return true;
  if (error) {
    error->clear();
    if (r)
      for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
        auto d = loomc_result_diagnostic_at(r, i);
        error->append(d->message.data, d->message.size);
        error->push_back('\n');
      }
  }
  return false;
}
} // namespace
bool BuildLlamaServeModule(const model::ModelDescriptor &model,
                           const LlamaPackageOptions &o,
                           loomc_context_t *context,
                           loomc_workspace_t *workspace, loomc_module_t **out,
                           std::string *error) {
  if (!out || !context || !workspace || o.prefill_capacity < 1 ||
      o.prefill_capacity > 512 || o.context_capacity < o.prefill_capacity ||
      o.context_capacity > 131072 || model.layers.empty()) {
    if (error)
      *error = "invalid bounded serving build configuration";
    return false;
  }
  *out = nullptr;
  const uint64_t pool = ((o.context_capacity + 63) / 64) * 64;
  const uint64_t cache_base =
      768 + 4 * std::max<uint64_t>(o.prefill_capacity, 129);
  auto registry = dispatch::Registry::Llama3B();
  dispatch::AttentionNativeConfig config{model.layers.size(),
                                         o.context_capacity, pool};
  loomc_module_t *native = nullptr;
  loomc_result_t *r = nullptr;
  auto s = dispatch::LoadLlamaNative(context, workspace, registry,
                                     llama::LlamaProgramDispatches(),
                                     o.source_root, config, &native, &r);
  Owner<loomc_module_t, loomc_module_release> native_owner(native);
  Owner<loomc_result_t, loomc_result_release> native_result(r);
  if (!Check(s, error) || !Result(r, error))
    return false;
  loomc_module_t *adapter = nullptr;
  r = nullptr;
  const auto path = std::filesystem::path(o.source_root) /
                    "tools/loomify-gguf/serving/epoch.loom";
  s = loomc_module_deserialize_from_path(
      context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
      loomc_allocator_system(), &adapter, &r);
  Owner<loomc_module_t, loomc_module_release> adapter_owner(adapter);
  Owner<loomc_result_t, loomc_result_release> adapter_result(r);
  if (!Check(s, error) || !Result(r, error))
    return false;
  loomc_link_module_provider_t providers[2]{};
  providers[0].module = native;
  providers[0].role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
  providers[1].module = adapter;
  providers[1].role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
  loomc_link_options_t options{};
  options.mode = LOOMC_LINK_MODE_MERGE;
  options.module_providers = providers;
  options.module_provider_count = 2;
  loomc_linker_t *linker = nullptr;
  if (!Check(loomc_linker_create(context, nullptr, loomc_allocator_system(),
                                 &linker),
             error))
    return false;
  Owner<loomc_linker_t, loomc_linker_release> linker_owner(linker);
  loomc_module_t *definitions = nullptr;
  r = nullptr;
  s = loomc_link_module(linker, workspace, &options, &definitions, &r);
  Owner<loomc_module_t, loomc_module_release> definitions_owner(definitions);
  Owner<loomc_result_t, loomc_result_release> merge_result(r);
  if (!Check(s, error) || !Result(r, error))
    return false;
  std::unique_ptr<ModuleBuilder> module;
  if (!Check(ModuleBuilder::Create(context, workspace, definitions, &module),
             error))
    return false;
  llama::LlamaProgramSettings settings;
  settings.row_capacity = o.prefill_capacity;
  settings.cache_capacity = o.context_capacity;
  settings.pool_capacity = pool;
  settings.cache_base = cache_base;
  settings.map_origin = 2048;
  settings.source_asset = o.gguf;
  llama::LlamaProgram prefill, decode;
  if (!llama::DefineLlamaProgram(model, settings, registry, *module,
                                 o.prefill_root, &prefill, error))
    return false;
  settings.phase = operations::Phase::Decode;
  settings.row_capacity = 1;
  if (!llama::DefineLlamaProgram(model, settings, registry, *module,
                                 o.decode_root, &decode, error))
    return false;
  ProgramRef epoch;
  if (!Check(DefineLlamaEpoch(*module, prefill.program, o.prefill_capacity,
                              o.epoch_root, &epoch),
             error))
    return false;
  r = nullptr;
  s = module->LinkAndVerify({prefill.program, decode.program, epoch}, out, &r);
  Owner<loomc_result_t, loomc_result_release> link_result(r);
  if (!Check(s, error) || !Result(r, error)) {
    if (*out) {
      loomc_module_release(*out);
      *out = nullptr;
    }
    return false;
  }
  return true;
}
} // namespace hrx::loomify_gguf
