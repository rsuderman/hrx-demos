// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/registry.h"
#include "lib/dispatch/projection.h"
#include "lib/dispatch/output.h"
#include "lib/dispatch/attention_builders.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/template/ops.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/link.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
namespace hrx::dispatch {
Registration EmbeddingRegistration();
namespace {
using namespace operations;
constexpr const char *kRoot = "lib/kernels/llama/";
std::string File(const char *name) { return std::string(kRoot) + name; }
bool Fail(std::string *error, const char *text) {
  if (error)
    *error = text;
  return false;
}
loomc_status_t Error(const char *text) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", text));
}
const loom_op_t *Symbol(loomc_module_t *module, const std::string &name) {
  loomc_module_interop_view_t view{};
  loomc_result_t *result = nullptr;
  auto status = loomc_module_get_interop_view(module, loomc_allocator_system(),
                                              &view, &result);
  bool ok =
      loomc_status_is_ok(status) && result && loomc_result_succeeded(result);
  loomc_status_free(status);
  if (result)
    loomc_result_release(result);
  if (!ok)
    return nullptr;
  const auto *native = view.module;
  for (size_t i = 0; i < native->symbols.count; ++i) {
    auto id = native->symbols.entries[i].name_id;
    if (id == LOOM_STRING_ID_INVALID || id >= native->strings.count)
      continue;
    auto actual = loom_string_table_get(&native->strings, id);
    if (name.size() == actual.size &&
        !std::memcmp(name.data(), actual.data, actual.size))
      return native->symbols.entries[i].defining_op;
  }
  return nullptr;
}
} // namespace
Registry Registry::Llama3B() {
  Registry registry;
  registry.entries_.push_back(EmbeddingRegistration());
  Registration norm{
      Id::RmsNorm,
      OpCode::RmsNorm,
      Kind::Kernel,
      "llama.rmsnorm",
      "llama32_rmsnorm_f32",
      File("norm/norm_interface.loom"),
      {File("norm/norm.loom"), "lib/kernels/normalization/rmsnorm_row.loom",
       "lib/kernels/elementwise/residual_row.loom",
       "lib/kernels/elementwise/swiglu_row.loom", File("config.loom"),
       File("active_rows.loom"), File("output/head_count.loom")},
      {{Port::Input, Access::Read},
       {Port::Gamma, Access::Read},
       {Port::Output, Access::Write}},
      {"llama.model.hidden_size", "llama.model.rms_epsilon"}};
  norm.interface_symbol = norm.symbol;
  norm.builder = NormBuilder();
  registry.entries_.push_back(std::move(norm));
  std::vector<std::string> files;
  for (const char *file :
       {"lib/kernels/llama/ffn/ffn.loom",
        "lib/kernels/elementwise/ffn_launches.loom",
        "lib/kernels/normalization/rmsnorm_row.loom",
        "lib/kernels/elementwise/residual_row.loom",
        "lib/kernels/elementwise/swiglu_row.loom",
        "lib/kernels/llama/config.loom", "lib/kernels/llama/active_rows.loom",
        "lib/kernels/llama/output/head_count.loom",
        "lib/kernels/projection/q4k.loom",
        "lib/kernels/projection/q4k_compat.loom",
        "lib/kernels/projection/q6k.loom",
        "lib/kernels/projection/q6k_compat.loom",
        "lib/kernels/projection/qk_common.loom",
        "lib/kernels/projection/qk_mmvq_compat.loom",
        "lib/kernels/quantization/q8_1_x4.loom",
        "lib/kernels/quantization/q8_1_x4_mmvq_compat.loom"})
    files.push_back(file);
  for (Id id :
       {Id::GateUpPrefill, Id::GateUpDecode, Id::DownPrefill, Id::DownDecode}) {
    bool gate = id == Id::GateUpPrefill || id == Id::GateUpDecode;
    const char *suffix = id == Id::GateUpPrefill  ? "gate_up_prefill"
                         : id == Id::GateUpDecode ? "gate_up_decode"
                         : id == Id::DownPrefill  ? "down_prefill"
                                                  : "down_decode";
    Registration entry{id,
                       gate ? OpCode::GatedMatmul : OpCode::LinearResidual,
                       Kind::CommandFamily,
                       std::string("llama.ffn.") + suffix,
                       std::string("llm.ffn.") + suffix,
                       File("ffn/ffn_interface.loom"),
                       files,
                       {},
                       {"ggml.linear_q4k_q8_1_x4.token_capacity",
                        "ggml.linear_q4k_q8_1_x4.output_capacity",
                        "ggml.linear_q6k_q8_1_x4.token_capacity",
                        "ggml.linear_q6k_q8_1_x4.output_capacity",
                        "ggml.quantize_q8_1_x4.group_capacity"}};
    entry.operands =
        gate
            ? std::vector<OperandContract>{{Port::Input, Access::Read},
                                           {Port::GateWeight, Access::Read},
                                           {Port::UpWeight, Access::Read},
                                           {Port::Output, Access::Write}}
            : std::vector<OperandContract>{{Port::Input, Access::Read},
                                           {Port::Weight, Access::Read},
                                           {Port::Residual, Access::ReadWrite}};
    entry.interface_symbol = entry.symbol;
    entry.phase = (id == Id::GateUpDecode || id == Id::DownDecode)
                      ? operations::Phase::Decode
                      : operations::Phase::Prefill;
    entry.builder = FfnBuilder(entry.operation);
    registry.entries_.push_back(std::move(entry));
  }
  RegisterProjections(&registry);
  for (auto entry : OutputRegistrations())
    registry.entries_.push_back(std::move(entry));
  for (auto entry : AttentionRegistrations())
    registry.entries_.push_back(std::move(entry));
  return registry;
}
const Registration *Registry::Find(Id id) const {
  for (const auto &entry : entries_)
    if (entry.id == id)
      return &entry;
  return nullptr;
}
const Registration *Registry::Find(const std::string &name) const {
  for (const auto &entry : entries_)
    if (entry.name == name)
      return &entry;
  return nullptr;
}
std::vector<Registration> Registry::Implementations(Id id) const {
  std::vector<Registration> result;
  for (const auto &entry : entries_)
    if (entry.id == id)
      result.push_back(entry);
  return result;
}
bool Registry::Add(Registration entry, std::string *error) {
  if (error)
    error->clear();
  if ((entry.kind != Kind::Kernel && entry.kind != Kind::CommandFamily &&
       entry.kind != Kind::CommandProgram) ||
      entry.name.empty() || entry.symbol.empty() ||
      entry.interface_file.empty() || entry.provider_files.empty() ||
      entry.variant.empty() || entry.interface_symbol.empty() ||
      !entry.builder.supports || !entry.builder.build ||
      entry.builder.operation != entry.operation ||
      entry.builder.kind != entry.kind ||
      entry.operands.size() != entry.builder.operands.size())
    return Fail(error, "incompatible dispatch registration");
  for (size_t i = 0; i < entry.operands.size(); ++i)
    if (entry.operands[i].port != entry.builder.operands[i].port ||
        entry.operands[i].access != entry.builder.operands[i].access)
      return Fail(error, "incompatible operand contract");
  for (const auto &old : entries_) {
    if (old.id != entry.id && old.name != entry.name)
      continue;
    if (old.id != entry.id || old.name != entry.name ||
        old.operation != entry.operation || old.phase != entry.phase ||
        old.operands.size() != entry.operands.size())
      return Fail(error, "incompatible operation variants");
    if (old.variant == entry.variant)
      return Fail(error, "duplicate dispatch variant");
    for (size_t i = 0; i < old.operands.size(); ++i)
      if (old.operands[i].port != entry.operands[i].port ||
          old.operands[i].access != entry.operands[i].access)
        return Fail(error, "incompatible operation variants");
  }
  entries_.push_back(std::move(entry));
  return true;
}
const Registration *Registry::Resolve(Id id, const InvocationRequest &request,
                                      const std::string &variant,
                                      std::string *error) const {
  if (error)
    error->clear();
  std::string reason = "missing dispatch or variant";
  for (const auto &entry : entries_) {
    if (entry.id != id || (!variant.empty() && entry.variant != variant))
      continue;
    std::string rejected;
    if (entry.builder.supports(entry, request, &rejected))
      return &entry;
    reason = entry.name + "/" + entry.variant + ": " + rejected;
    if (!variant.empty())
      break;
  }
  if (error)
    *error = reason;
  return nullptr;
}
loomc_status_t Load(loomc_context_t *context, loomc_workspace_t *workspace,
                    const Registry &registry, const std::vector<Id> &operations,
                    const std::string &source_root, loomc_module_t **out,
                    loomc_result_t **result) {
  if (out)
    *out = nullptr;
  if (result)
    *result = nullptr;
  std::vector<Registration> entries;
  std::vector<Id> seen;
  for (auto id : operations) {
    if (std::find(seen.begin(), seen.end(), id) != seen.end())
      continue;
    seen.push_back(id);
    auto implementations = registry.Implementations(id);
    if (implementations.empty())
      return Error("missing dispatch registration");
    entries.insert(entries.end(), implementations.begin(),
                   implementations.end());
  }
  return Load(context, workspace, entries, source_root, out, result);
}
loomc_status_t Load(loomc_context_t *context, loomc_workspace_t *workspace,
                    const Registration &entry, const std::string &source_root,
                    loomc_module_t **out, loomc_result_t **result) {
  if (out)
    *out = nullptr;
  if (result)
    *result = nullptr;
  if (!context || !workspace || !out || !result)
    return Error("load requires context, workspace and outputs");
  return Load(context, workspace, std::vector<Registration>{entry}, source_root,
              out, result);
}
loomc_status_t Load(loomc_context_t *context, loomc_workspace_t *workspace,
                    const std::vector<Registration> &entries,
                    const std::string &source_root, loomc_module_t **out,
                    loomc_result_t **result) {
  if (out)
    *out = nullptr;
  if (result)
    *result = nullptr;
  if (!context || !workspace || !out || !result || entries.empty())
    return Error("load requires context, workspace, registrations and outputs");
  Registry validation;
  std::string error;
  std::vector<NativeSource> sources;
  for (const auto &entry : entries) {
    if (!validation.Add(entry, &error))
      return Error(error.c_str());
    sources.push_back({entry.symbol, entry.kind, entry.interface_file,
                       entry.provider_files, entry.required_config,
                       entry.interface_symbol, entry.builder.required_definitions});
  }
  return LoadNative(context, workspace, sources, source_root, out, result);
}
loomc_status_t LoadNative(loomc_context_t *context, loomc_workspace_t *workspace,
                         const std::vector<NativeSource> &entries,
                         const std::string &source_root, loomc_module_t **out,
                         loomc_result_t **result) {
  return LoadNative(context, workspace, entries, source_root, nullptr, out,
                    result);
}
loomc_status_t LoadNative(loomc_context_t *context,
                          loomc_workspace_t *workspace,
                          const std::vector<NativeSource> &entries,
                          const std::string &source_root,
                          const loomc_module_t *configuration,
                          loomc_module_t **out, loomc_result_t **result) {
  if (out) *out = nullptr;
  if (result) *result = nullptr;
  if (!context || !workspace || !out || !result || entries.empty())
    return Error("native load requires context, workspace, sources and outputs");
  std::vector<std::string> files;
  for (const auto &entry : entries) {
    if ((entry.kind != Kind::Kernel && entry.kind != Kind::CommandFamily &&
         entry.kind != Kind::CommandProgram) ||
        entry.symbol.empty() || entry.interface_symbol.empty() ||
        entry.interface_file.empty() || entry.provider_files.empty())
      return Error("native source requires callable and interface/provider closure");
    auto add = [&](const std::string &file) {
      if (std::find(files.begin(), files.end(), file) == files.end())
        files.push_back(file);
    };
    add(entry.interface_file);
    for (const auto &file : entry.provider_files)
      add(file);
  }
  using Module =
      std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)>;
  std::vector<Module> modules;
  for (const auto &file : files) {
    loomc_module_t *module = nullptr;
    std::string path = source_root + "/" + file;
    auto status = loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
        loomc_allocator_system(), &module, result);
    Module owned(module, loomc_module_release);
    if (!loomc_status_is_ok(status) || !*result ||
        !loomc_result_succeeded(*result))
      return status;
    loomc_result_release(*result);
    *result = nullptr;
    modules.push_back(std::move(owned));
  }
  for (const auto &entry : entries) {
    auto i = std::find(files.begin(), files.end(), entry.interface_file) -
             files.begin();
    const auto *declaration = Symbol(modules[i].get(), entry.interface_symbol);
    const bool kind_matches = declaration &&
        (entry.kind == Kind::Kernel ? loom_kernel_decl_isa(declaration)
         : entry.kind == Kind::CommandFamily ? loom_template_decl_isa(declaration)
         : loom_command_program_decl_isa(declaration));
    if (!kind_matches)
      return Error("missing symbol or wrong native callable kind");
  }
  std::vector<loomc_link_module_provider_t> providers;
  if (configuration) {
    loomc_link_module_provider_t provider{};
    provider.module = configuration;
    provider.provider_name = loomc_make_cstring_view("configuration");
    provider.role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
    providers.push_back(provider);
  }
  for (size_t i = 0; i < modules.size(); ++i) {
    loomc_link_module_provider_t provider{};
    provider.module = modules[i].get();
    provider.provider_name = loomc_make_cstring_view(files[i].c_str());
    provider.role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
    providers.push_back(provider);
  }
  loomc_linker_t *linker = nullptr;
  auto status =
      loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker);
  if (!loomc_status_is_ok(status))
    return status;
  loomc_link_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  options.structure_size = sizeof(options);
  options.mode = LOOMC_LINK_MODE_MERGE;
  options.module_providers = providers.data();
  options.module_provider_count = providers.size();
  status = loomc_link_module(linker, workspace, &options, out, result);
  if (loomc_status_is_ok(status) && *result &&
      loomc_result_succeeded(*result)) {
    loomc_result_release(*result);
    *result = nullptr;
    for (const auto &entry : entries)
      for (const auto &name : entry.required_config) {
        const auto *config = Symbol(*out, name);
        if (!config || !loom_config_def_isa(config)) {
          loomc_module_release(*out);
          *out = nullptr;
          loomc_linker_release(linker);
          return Error("missing required configuration definition");
        }
      }
    for (const auto &entry : entries)
      for (const auto &name : entry.required_definitions) {
        const auto *definition = Symbol(*out, name);
        if (!definition || loom_kernel_decl_isa(definition) ||
            loom_template_decl_isa(definition) ||
            loom_command_program_decl_isa(definition)) {
          loomc_module_release(*out);
          *out = nullptr;
          loomc_linker_release(linker);
          return Error("missing dispatch implementation dependency");
        }
      }
    std::vector<loomc_string_view_t> roots;
    for (const auto &entry : entries)
      roots.push_back(loomc_make_cstring_view(entry.symbol.c_str()));
    loomc_link_module_provider_t merged{};
    merged.module = *out;
    merged.role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
    options.mode = LOOMC_LINK_MODE_LINK;
    options.module_providers = &merged;
    options.module_provider_count = 1;
    options.root_symbols = roots.data();
    options.root_symbol_count = roots.size();
    loomc_module_t *checked = nullptr;
    status = loomc_link_module(linker, workspace, &options, &checked, result);
    if (checked)
      loomc_module_release(checked);
    if (!loomc_status_is_ok(status) || !*result ||
        !loomc_result_succeeded(*result)) {
      loomc_module_release(*out);
      *out = nullptr;
    }
  }
  loomc_linker_release(linker);
  return status;
}
} // namespace hrx::dispatch
