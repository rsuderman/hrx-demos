// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/linking.h"
#include "lib/loomify/architectures.h"
#include "lib/loomify/platforms.h"
#include <algorithm>
#include <filesystem>
#include "lib/loomify/materialize_source.h"
#include "loomc/link.h"
#include "loomc/link_index.h"
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
bool RequireRootDefinitions(loomc_context_t *context, const loomc_module_t *module,
                            const std::vector<const char *> &roots,
                            std::string *error) {
  // Native LINK permits declaration-only global roots in relocatable modules.
  // This constructor promises concrete requested outputs, including data roots.
  loomc_link_index_builder_t *raw_builder = nullptr;
  if (!Status(loomc_link_index_builder_allocate(context, nullptr,
      loomc_allocator_system(), &raw_builder), error)) return false;
  Owner<loomc_link_index_builder_t, loomc_link_index_builder_free> builder(raw_builder, loomc_link_index_builder_free);
  if (!Status(loomc_link_index_builder_add_module(builder.get(), module, nullptr,
      nullptr), error)) return false;
  loomc_link_index_t *raw_index = nullptr;
  loomc_result_t *raw_result = nullptr;
  auto status = loomc_link_index_builder_finish(builder.get(), &raw_index, &raw_result);
  Owner<loomc_link_index_t, loomc_link_index_release> index(raw_index, loomc_link_index_release);
  Owner<loomc_result_t, loomc_result_release> result(raw_result, loomc_result_release);
  if (!Status(status, error) || !Result(result.get(), error)) return false;
  for (const char *root : roots) {
    loomc_link_index_symbol_t symbol{};
    if (!loomc_link_index_lookup_global(index.get(), loomc_make_cstring_view(root), &symbol) ||
        !(symbol.flags & LOOMC_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION) ||
        (symbol.flags & LOOMC_LINK_SYMBOL_FLAG_DECLARATION)) {
      if (error) *error = std::string("native LINK root has no concrete definition: ") + root;
      return false;
    }
  }
  return true;
}
} // namespace
bool LinkNumericalProducts(loomc_context_t *context, loomc_workspace_t *workspace,
                           const std::string &library_root,
                           const std::string &configuration,
                           const ArchitectureFiles &architecture,
                           const PlatformFiles &platform, loomc_module_t **out,
                           std::string *error) {
  const auto fail = [&](const char *reason) {
    if (error) *error = std::string("numerical composition for ") +
        architecture.family + " on " + platform.selector + ": " + reason;
    return false;
  };
  if (library_root.empty() || !architecture.implementation || !platform.targets)
    return fail("missing numerical product or library root");
  std::vector<std::string> paths;
  const auto append = [&](const char *product) {
    const auto path = (std::filesystem::path(library_root) / product).string();
    if (std::find(paths.begin(), paths.end(), path) == paths.end())
      paths.push_back(path);
  };
  append(architecture.implementation);
  append(platform.targets);
  std::vector<RequiredOperation> visited;
  for (const auto required : architecture.required_operations) {
    if (std::find(visited.begin(), visited.end(), required) != visited.end())
      return fail("duplicate required operation");
    visited.push_back(required);
    const char *product = nullptr;
    for (const auto &implementation : platform.implementations) {
      if (implementation.operation != required) continue;
      if (product) return fail("duplicate platform operation implementation");
      if (!implementation.product || !*implementation.product)
        return fail("empty platform operation product");
      product = implementation.product;
    }
    if (!product) return fail("missing platform operation implementation");
    append(product);
  }
  // Native LINK owns actual symbols/types/predicates. Selection metadata is not
  // a verifier for arbitrary provider IR; loading every selected input is required.
  return LinkConfiguredProducts(context, workspace, configuration, paths,
                                architecture.roots, true, out, error);
}
bool LinkConfiguredProducts(loomc_context_t *context, loomc_workspace_t *workspace,
                           const std::string &configuration,
                           const std::vector<std::string> &product_paths,
                           const std::vector<const char *> &roots,
                           bool normalize_numerical, loomc_module_t **out,
                           std::string *error) {
  if (!context || !workspace || !out || roots.empty() || product_paths.empty()) {
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
  std::vector<Owner<loomc_module_t, loomc_module_release>> products;
  std::vector<loomc_link_module_provider_t> providers;
  products.reserve(product_paths.size());
  providers.reserve(product_paths.size() + 1);
  providers.push_back({config.get(), loomc_make_cstring_view("configuration"),
                       LOOMC_LINK_PROVIDER_ROLE_INPUT});
  for (const auto &path : product_paths) {
    loomc_module_t *raw_product = nullptr;
    raw_result = nullptr;
    status = loomc_module_deserialize_from_path(context, workspace,
        loomc_make_cstring_view(path.c_str()), nullptr, loomc_allocator_system(),
        &raw_product, &raw_result);
    products.emplace_back(raw_product, loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> loaded(raw_result, loomc_result_release);
    if (!Status(status, error) || !Result(loaded.get(), error)) return false;
    providers.push_back({products.back().get(), loomc_make_cstring_view(path.c_str()),
                         LOOMC_LINK_PROVIDER_ROLE_INPUT});
  }
  loomc_linker_t *raw_linker = nullptr;
  if (!Status(loomc_linker_create(context, nullptr, loomc_allocator_system(), &raw_linker), error)) return false;
  Owner<loomc_linker_t, loomc_linker_release> linker(raw_linker, loomc_linker_release);
  std::vector<loomc_string_view_t> root_names;
  for (const char *root : roots) root_names.push_back(loomc_make_cstring_view(root));
  loomc_link_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  options.structure_size = sizeof(options);
  options.mode = LOOMC_LINK_MODE_LINK;
  options.module_providers = providers.data();
  options.module_provider_count = providers.size();
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
    // Instantiate nested authoring templates to a native fixed point before
    // enforcing final operation predicates. Ordinary DCE must follow selection:
    // template providers remain needed while family applications are pending.
    const auto* pass_files = materialize_source_create();
    if (materialize_source_size() != 1) {
      if (error) *error = "native materialization asset is missing";
      return false;
    }

    loomc_source_options_t pass_source_options{};
    pass_source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
    pass_source_options.identifier = loomc_make_cstring_view("loomify-numerical-passes.loom");
    pass_source_options.contents = loomc_make_byte_span(pass_files[0].data, pass_files[0].size);
    pass_source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t *raw_pass_source = nullptr;
    if (!Status(loomc_source_create(&pass_source_options, loomc_allocator_system(),
        &raw_pass_source), error)) return false;
    Owner<loomc_source_t, loomc_source_release> pass_source(raw_pass_source, loomc_source_release);
    loomc_module_t *raw_pass_module = nullptr;
    raw_result = nullptr;
    status = loomc_module_deserialize_from_source(context, workspace, pass_source.get(),
        nullptr, loomc_allocator_system(), &raw_pass_module, &raw_result);
    Owner<loomc_module_t, loomc_module_release> pass_module(raw_pass_module, loomc_module_release);
    result.reset(raw_result);
    if (!Status(status, error) || !Result(result.get(), error)) return false;
    loomc_pass_program_t *raw_passes = nullptr;
    raw_result = nullptr;
    status = loomc_pass_program_create_from_module_symbol(pass_module.get(),
        loomc_make_cstring_view("materialize"), nullptr, loomc_allocator_system(),
        &raw_passes, &raw_result);
    Owner<loomc_pass_program_t, loomc_pass_program_release> passes(raw_passes, loomc_pass_program_release);
    result.reset(raw_result);
    if (!Status(status, error) || !Result(result.get(), error)) return false;
    raw_result = nullptr;
    status = loomc_pass_program_run(passes.get(), workspace, linked.get(), nullptr,
        loomc_allocator_system(), &raw_result);
    result.reset(raw_result);
    if (!Status(status, error) || !Result(result.get(), error)) return false;
  }
  if (!RequireRootDefinitions(context, linked.get(), roots, error)) return false;
  *out = linked.release();
  if (error) error->clear();
  return true;
}
} // namespace hrx::loomify_gguf
