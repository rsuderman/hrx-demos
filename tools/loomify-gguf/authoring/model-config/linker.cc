// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model-config/linker.h"

#include "loom/ops/command/ops.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <set>

namespace hrx::loomify_gguf {
namespace {
using Module = std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)>;
loomc_status_t Error(const std::string &message) {
  return loomc_status_from_iree(iree_status_allocate_f(
      IREE_STATUS_INVALID_ARGUMENT, __FILE__, __LINE__, "%s", message.c_str()));
}
bool Name(const std::string &name) {
  return !name.empty() && name.find('\0') == std::string::npos;
}
const loom_op_t *Symbol(const loom_module_t *module, const std::string &name) {
  for (size_t i = 0; i < module->symbols.count; ++i) {
    auto id = module->symbols.entries[i].name_id;
    if (id == LOOM_STRING_ID_INVALID || id >= module->strings.count)
      continue;
    auto text = loom_string_table_get(&module->strings, id);
    if (text.size == name.size() &&
        !std::memcmp(text.data, name.data(), text.size))
      return module->symbols.entries[i].defining_op;
  }
  return nullptr;
}
bool Definition(const loom_module_t *module, const std::string &name) {
  auto *op = Symbol(module, name);
  if (!op)
    return false;
  for (size_t i = 0; i < module->symbols.count; ++i) {
    const auto &entry = module->symbols.entries[i];
    if (entry.defining_op == op && entry.definition)
      return !loom_symbol_definition_is_declaration(entry.definition);
  }
  return false;
}
loomc_status_t View(const loomc_module_t *module, const loom_module_t **out,
                    loomc_result_t **result) {
  loomc_module_interop_view_t view{};
  // The pinned SDK's read-only projection populates verification caches and
  // therefore takes a mutable handle. The borrowed IR remains unchanged.
  auto status =
      loomc_module_get_interop_view(const_cast<loomc_module_t *>(module),
                                    loomc_allocator_system(), &view, result);
  if (loomc_status_is_ok(status) && *result &&
      loomc_result_succeeded(*result)) {
    *out = view.module;
    loomc_result_release(*result);
    *result = nullptr;
  }
  return status;
}
loomc_status_t CheckRoot(const loom_module_t *module,
                         const ConfiguredRoot &root) {
  auto *op = Symbol(module, root.symbol);
  const loom_region_t *first = nullptr;
  const loom_region_t *second = nullptr;
  size_t split = 0;
  if (op && root.kind == ConfiguredCallableKind::Kernel &&
      loom_kernel_def_isa(op)) {
    first = loom_kernel_def_config(op);
    second = loom_kernel_def_body(op);
    split = loom_region_const_entry_block(first)->arg_count;
  } else if (op && root.kind == ConfiguredCallableKind::CommandProgram &&
             loom_command_program_def_isa(op)) {
    first = loom_command_program_def_body(op);
    split = loom_command_program_def_specialization_count(op);
  } else {
    return Error("root has no definition of the required callable kind: " +
                 root.symbol);
  }
  const auto *a = loom_region_const_entry_block(first);
  const auto *b = second ? loom_region_const_entry_block(second) : a;
  size_t argument_count =
      second ? b->arg_count
             : a->arg_count - std::min(split, size_t(a->arg_count));
  if (split > a->arg_count || split != root.specializations.size() ||
      argument_count != root.arguments.size())
    return Error("root operand group count mismatch: " + root.symbol);
  for (size_t i = 0; i < split + argument_count; ++i) {
    auto expected =
        i < split ? root.specializations[i] : root.arguments[i - split];
    auto value = i < split ? a->arg_ids[i] : b->arg_ids[second ? i - split : i];
    if (!loom_type_equal(expected, loom_module_value_type(module, value)))
      return Error("root operand type mismatch: " + root.symbol + " at " +
                   std::to_string(i));
  }
  return loomc_ok_status();
}
} // namespace

loomc_status_t LinkConfiguredModules(loomc_context_t *context,
                                     loomc_workspace_t *workspace,
                                     const loomc_module_t *configuration,
                                     const ConfiguredLinkRequest &request,
                                     loomc_module_t **out_module,
                                     loomc_result_t **out_result) {
  if (out_module)
    *out_module = nullptr;
  if (out_result)
    *out_result = nullptr;
  if (!context || !workspace || !configuration || !out_module || !out_result ||
      !Name(request.source_root) || request.providers.empty() ||
      request.roots.empty())
    return Error("configured link requires context, workspace, configuration, "
                 "providers, roots and outputs");
  std::set<std::string> root_names;
  for (const auto &root : request.roots) {
    if (!Name(root.symbol) || !root_names.insert(root.symbol).second ||
        (root.kind != ConfiguredCallableKind::Kernel &&
         root.kind != ConfiguredCallableKind::CommandProgram))
      return Error("invalid or duplicate configured root");
    for (const auto *group : {&root.specializations, &root.arguments})
      for (auto type : *group)
        if (!loom_type_is_scalar(type) && !loom_type_is_buffer(type))
          return Error(
              "root contract requires native scalar or buffer types: " +
              root.symbol);
  }
  const loom_module_t *config = nullptr;
  auto status = View(configuration, &config, out_result);
  if (!loomc_status_is_ok(status) || !config)
    return status;
  for (const auto &name : request.required_configuration)
    if (!Name(name) || !Symbol(config, name) ||
        !loom_config_def_isa(Symbol(config, name)))
      return Error("missing generated configuration definition: " + name);
  std::vector<Module> modules;
  std::vector<const loom_module_t *> views{config};
  std::vector<std::string> paths;
  std::vector<loomc_link_provider_role_t> roles;
  std::map<std::string, loomc_link_provider_role_t> seen;
  for (const auto &unit : request.providers) {
    std::filesystem::path path(unit.path);
    if (!Name(unit.path) || path.is_absolute() ||
        (unit.role != LOOMC_LINK_PROVIDER_ROLE_INPUT &&
         unit.role != LOOMC_LINK_PROVIDER_ROLE_LIBRARY))
      return Error("invalid provider path or role: " + unit.path);
    for (const auto &part : path)
      if (part == "..")
        return Error("provider escapes source root: " + unit.path);
    auto key = path.lexically_normal().generic_string();
    auto prior = seen.emplace(key, unit.role);
    if (!prior.second) {
      if (prior.first->second != unit.role)
        return Error("contradictory provider roles: " + key);
      continue;
    }
    loomc_module_t *loaded = nullptr;
    auto absolute =
        (std::filesystem::path(request.source_root) / path).string();
    loomc_module_deserialize_options_t deserialize{};
    deserialize.type = LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS;
    deserialize.structure_size = sizeof(deserialize);
    deserialize.identifier = loomc_make_cstring_view(key.c_str());
    status = loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(absolute.c_str()),
        &deserialize, loomc_allocator_system(), &loaded, out_result);
    Module owned(loaded, loomc_module_release);
    if (!loomc_status_is_ok(status) || !*out_result ||
        !loomc_result_succeeded(*out_result))
      return status;
    loomc_result_release(*out_result);
    *out_result = nullptr;
    const loom_module_t *view = nullptr;
    status = View(loaded, &view, out_result);
    if (!loomc_status_is_ok(status) || !view)
      return status;
    for (size_t i = 0; i < config->symbols.count; ++i) {
      const auto &entry = config->symbols.entries[i];
      if (!entry.defining_op || !loom_config_def_isa(entry.defining_op))
        continue;
      auto text = loom_string_table_get(&config->strings, entry.name_id);
      std::string name(text.data, text.size);
      auto *other = Symbol(view, name);
      if (other && loom_config_def_isa(other))
        return Error("provider competes with generated configuration: " + key +
                     ": " + name);
    }
    modules.push_back(std::move(owned));
    views.push_back(view);
    paths.push_back(std::move(key));
    roles.push_back(unit.role);
  }
  for (const auto &name : request.required_definitions) {
    bool found = false;
    for (auto *view : views)
      found |= Name(name) && Definition(view, name);
    if (!found)
      return Error("missing required implementation/helper definition: " +
                   name);
  }
  std::vector<loomc_link_module_provider_t> providers;
  providers.push_back({configuration,
                       loomc_make_cstring_view("generated-configuration"),
                       LOOMC_LINK_PROVIDER_ROLE_INPUT});
  for (size_t i = 0; i < modules.size(); ++i)
    providers.push_back({modules[i].get(),
                         loomc_make_cstring_view(paths[i].c_str()), roles[i]});
  std::vector<loomc_string_view_t> roots;
  for (const auto &root : request.roots)
    roots.push_back(loomc_make_cstring_view(root.symbol.c_str()));
  loomc_linker_t *raw = nullptr;
  status =
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &raw);
  if (!loomc_status_is_ok(status))
    return status;
  std::unique_ptr<loomc_linker_t, decltype(&loomc_linker_release)> linker(
      raw, loomc_linker_release);
  loomc_link_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  options.structure_size = sizeof(options);
  options.mode = LOOMC_LINK_MODE_LINK;
  options.module_providers = providers.data();
  options.module_provider_count = providers.size();
  options.root_symbols = roots.data();
  options.root_symbol_count = roots.size();
  loomc_module_t *linked = nullptr;
  status = loomc_link_module(raw, workspace, &options, &linked, out_result);
  Module output(linked, loomc_module_release);
  if (!loomc_status_is_ok(status) || !*out_result ||
      !loomc_result_succeeded(*out_result))
    return status;
  // Keep the successful LINK result while obtaining a temporary interop view.
  loomc_result_t *inspection = nullptr;
  const loom_module_t *view = nullptr;
  status = View(linked, &view, &inspection);
  if (!loomc_status_is_ok(status) || !view) {
    loomc_result_release(*out_result);
    *out_result = inspection;
    return status;
  }
  for (const auto &root : request.roots) {
    status = CheckRoot(view, root);
    if (!loomc_status_is_ok(status)) {
      loomc_result_release(*out_result);
      *out_result = nullptr;
      return status;
    }
  }
  *out_module = output.release();
  return status;
}
} // namespace hrx::loomify_gguf
