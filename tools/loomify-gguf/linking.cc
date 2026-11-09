// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/linking.h"
#include "loomc/link.h"
#include "loomc/pass.h"
#include <memory>
namespace hrx::loomify_gguf {
namespace {
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
bool Status(loomc_status_t status, std::string *error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok && error) {
    auto message = loomc_status_message(status);
    error->assign(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Result(loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result)) return true;
  if (error) {
    *error = "native Loom operation failed";
    if (result)
      for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
        auto *d = loomc_result_diagnostic_at(result, i);
        error->append("\n").append(d->message.data, d->message.size);
      }
  }
  return false;
}
} // namespace
bool LinkConfiguredProduct(loomc_context_t *context, loomc_workspace_t *workspace,
                           const std::string &configuration,
                           const std::string &product_path,
                           const std::vector<const char *> &roots,
                           bool normalize_numerical, loomc_module_t **out,
                           std::string *error) {
  if (!context || !workspace || !out || roots.empty()) {
    if (error) *error = "native link requires context, workspace, roots and output";
    return false;
  }
  loomc_source_options_t source_options{};
  source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
  source_options.identifier = loomc_make_cstring_view("gguf-configuration.loom");
  source_options.contents = loomc_make_byte_span(configuration.data(), configuration.size());
  source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
  loomc_source_t *raw_source = nullptr;
  if (!Status(loomc_source_create(&source_options, loomc_allocator_system(), &raw_source), error)) return false;
  Owner<loomc_source_t, loomc_source_release> source(raw_source, loomc_source_release);
  loomc_module_t *raw_config = nullptr;
  loomc_result_t *raw_result = nullptr;
  auto status = loomc_module_deserialize_from_source(context, workspace, source.get(),
      nullptr, loomc_allocator_system(), &raw_config, &raw_result);
  Owner<loomc_module_t, loomc_module_release> config(raw_config, loomc_module_release);
  Owner<loomc_result_t, loomc_result_release> parsed(raw_result, loomc_result_release);
  if (!Status(status, error) || !Result(parsed.get(), error)) return false;
  loomc_module_t *raw_product = nullptr;
  raw_result = nullptr;
  status = loomc_module_deserialize_from_path(context, workspace,
      loomc_make_cstring_view(product_path.c_str()), nullptr, loomc_allocator_system(),
      &raw_product, &raw_result);
  Owner<loomc_module_t, loomc_module_release> product(raw_product, loomc_module_release);
  Owner<loomc_result_t, loomc_result_release> loaded(raw_result, loomc_result_release);
  if (!Status(status, error) || !Result(loaded.get(), error)) return false;
  loomc_linker_t *raw_linker = nullptr;
  if (!Status(loomc_linker_create(context, nullptr, loomc_allocator_system(), &raw_linker), error)) return false;
  Owner<loomc_linker_t, loomc_linker_release> linker(raw_linker, loomc_linker_release);
  const loomc_link_module_provider_t providers[] = {
      {config.get(), loomc_make_cstring_view("configuration"), LOOMC_LINK_PROVIDER_ROLE_INPUT},
      {product.get(), loomc_make_cstring_view(product_path.c_str()), LOOMC_LINK_PROVIDER_ROLE_INPUT}};
  std::vector<loomc_string_view_t> root_names;
  for (const char *root : roots) root_names.push_back(loomc_make_cstring_view(root));
  loomc_link_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  options.structure_size = sizeof(options);
  options.mode = LOOMC_LINK_MODE_LINK;
  options.module_providers = providers;
  options.module_provider_count = 2;
  options.root_symbols = root_names.data();
  options.root_symbol_count = root_names.size();
  options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  loomc_module_t *raw_linked = nullptr;
  raw_result = nullptr;
  status = loomc_link_module(linker.get(), workspace, &options, &raw_linked, &raw_result);
  Owner<loomc_module_t, loomc_module_release> linked(raw_linked, loomc_module_release);
  Owner<loomc_result_t, loomc_result_release> result(raw_result, loomc_result_release);
  if (!Status(status, error) || !Result(result.get(), error)) return false;
  if (normalize_numerical) {
    loomc_pass_program_t *raw_passes = nullptr;
    raw_result = nullptr;
    status = loomc_pass_program_create_from_pipeline_text(context,
        loomc_make_cstring_view("canonicalize,unroll-scf-for,canonicalize"), nullptr,
        loomc_allocator_system(), &raw_passes, &raw_result);
    Owner<loomc_pass_program_t, loomc_pass_program_release> passes(raw_passes, loomc_pass_program_release);
    result.reset(raw_result);
    if (!Status(status, error) || !Result(result.get(), error)) return false;
    raw_result = nullptr;
    status = loomc_pass_program_run(passes.get(), workspace, linked.get(), nullptr,
        loomc_allocator_system(), &raw_result);
    result.reset(raw_result);
    if (!Status(status, error) || !Result(result.get(), error)) return false;
  }
  *out = linked.release();
  if (error) error->clear();
  return true;
}
} // namespace hrx::loomify_gguf
