// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/model_export.h"
#include "tools/loomify-gguf/authoring/model-config/providers.h"
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
                           const ModelPackageOptions &o,
                           loomc_context_t *context,
                           loomc_workspace_t *workspace,
                           ConfiguredLlamaModel *out, std::string *error) {
  if (!out || !context || !workspace) {
    if (error)
      *error = "serving build requires context, workspace and output";
    return false;
  }
  LinkedConfigurationSettings settings;
  settings.profile = LinkedProfile::Llama3BModel;
  settings.token_capacity = o.prefill_capacity;
  settings.context_capacity = o.context_capacity;
  LinkedConfiguration configuration;
  loomc_result_t *raw = nullptr;
  auto status = CreateLinkedConfiguration(context, workspace, model, o.gguf,
                                          settings, &configuration, &raw);
  Owner<loomc_result_t, loomc_result_release> config_result(raw);
  if (!Check(status, error) || !Result(raw, error))
    return false;
  raw = nullptr;
  status = BuildConfiguredLlamaModel(context, workspace, configuration,
                                     MakeLlamaLinkedProviders(o.source_root),
                                     out, &raw);
  Owner<loomc_result_t, loomc_result_release> build_result(raw);
  return Check(status, error) && Result(raw, error);
}
} // namespace hrx::loomify_gguf
