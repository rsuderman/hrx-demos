// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/registry/providers/provider_registry.h"

#include <utility>

namespace hrx::loomify_gguf {
namespace {

std::string Copy(loomc_string_view_t value) {
  return value.size ? std::string(value.data, value.size) : std::string();
}

loomc_status_t QueryStatus(loomc_result_t* result) {
  if (loomc_result_succeeded(result)) return loomc_ok_status();
  const auto* diagnostic = loomc_result_diagnostic_at(result, 0);
  return loomc_status_allocate(
      loomc_result_status_code(result), __FILE__, __LINE__,
      diagnostic ? diagnostic->message
                 : loomc_make_cstring_view("provider metadata query failed"));
}

}  // namespace

FrozenProviders::FrozenProviders(loomc_context_t* context,
                                 loomc_link_index_t* index,
                                 std::vector<ProviderSymbol> symbols)
    : context_(context), index_(index), symbols_(std::move(symbols)) {
  loomc_context_retain(context_);
}

FrozenProviders::~FrozenProviders() {
  loomc_link_index_release(index_);
  loomc_context_release(context_);
}

FrozenProviders::FrozenProviders(FrozenProviders&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      index_(std::exchange(other.index_, nullptr)),
      symbols_(std::move(other.symbols_)) {}

FrozenProviders& FrozenProviders::operator=(FrozenProviders&& other) noexcept {
  if (this != &other) {
    loomc_link_index_release(index_);
    loomc_context_release(context_);
    context_ = std::exchange(other.context_, nullptr);
    index_ = std::exchange(other.index_, nullptr);
    symbols_ = std::move(other.symbols_);
  }
  return *this;
}

std::vector<ProviderSymbol> FrozenProviders::FindCandidates(
    loomc_string_view_t name) const {
  std::vector<ProviderSymbol> candidates;
  if (!index_) return candidates;
  loomc_link_index_symbol_t symbol = {};
  if (loomc_link_index_lookup_global(index_, name, &symbol)) {
    do {
      candidates.push_back(symbols_[symbol.ordinal]);
    } while (loomc_link_index_next_global_duplicate(index_, &symbol, &symbol));
  }
  return candidates;
}

ProviderRegistryBuilder::ProviderRegistryBuilder(
    loomc_context_t* context, loomc_link_index_builder_t* builder)
    : context_(context), builder_(builder) {
  loomc_context_retain(context_);
}

ProviderRegistryBuilder::~ProviderRegistryBuilder() {
  loomc_link_index_builder_free(builder_);
  loomc_context_release(context_);
}

loomc_status_t ProviderRegistryBuilder::Create(
    loomc_context_t* context,
    std::unique_ptr<ProviderRegistryBuilder>* out_builder) {
  if (!out_builder) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_builder is required");
  }
  out_builder->reset();
  if (!context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "registry context is required");
  }
  loomc_link_index_builder_t* builder = nullptr;
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_allocate(
      context, nullptr, loomc_allocator_system(), &builder));
  out_builder->reset(new ProviderRegistryBuilder(context, builder));
  return loomc_ok_status();
}

loomc_status_t ProviderRegistryBuilder::AddModule(loomc_string_view_t label,
                                                const loomc_module_t* module) {
  if (finished_) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "registry builder is finished");
  }
  if (!label.data || !label.size || !module) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "nonempty label and module are required");
  }
  ProviderMetadata metadata;
  metadata.label = Copy(label);
  for (const auto& provider : providers_) {
    if (provider.label == metadata.label) {
      return loomc_make_status(LOOMC_STATUS_ALREADY_EXISTS,
                               "provider label is already registered");
    }
  }

  loomc_host_size_t count = 0;
  loomc_result_t* result = nullptr;
  LOOMC_RETURN_IF_ERROR(loomc_module_query_functions(
      module, nullptr, loomc_allocator_system(), 0, nullptr, &count, &result));
  auto query_status = QueryStatus(result);
  loomc_result_release(result);
  LOOMC_RETURN_IF_ERROR(query_status);
  std::vector<loomc_module_function_t> functions(count);
  if (count) {
    result = nullptr;
    LOOMC_RETURN_IF_ERROR(loomc_module_query_functions(
        module, nullptr, loomc_allocator_system(), count, functions.data(),
        &count, &result));
    query_status = QueryStatus(result);
    loomc_result_release(result);
    LOOMC_RETURN_IF_ERROR(query_status);
  }
  for (const auto& function : functions) {
    FunctionMetadata copied{Copy(function.symbol_name), function.kind,
                            std::nullopt};
    loomc_module_function_export_info_t export_info = {};
    if (loomc_module_function_try_get_export_info_at(
            module, function.function_ordinal, &export_info) &&
        (export_info.flags & LOOMC_MODULE_FUNCTION_EXPORT_FLAG_HAS_SYMBOL)) {
      copied.export_symbol = Copy(export_info.export_symbol);
    }
    metadata.functions.push_back(std::move(copied));
  }

  loomc_link_index_source_options_t options = {};
  options.provider_name = label;
  options.role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_add_module(
      builder_, module, &options, nullptr));
  providers_.push_back(std::move(metadata));
  return loomc_ok_status();
}

loomc_status_t ProviderRegistryBuilder::Freeze(
    std::unique_ptr<FrozenProviders>* out_frozen, loomc_result_t** out_result) {
  if (out_frozen) out_frozen->reset();
  if (out_result) *out_result = nullptr;
  if (!out_frozen || !out_result) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "frozen output and result output are required");
  }
  if (finished_) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "registry builder is finished");
  }
  loomc_link_index_t* index = nullptr;
  auto status = loomc_link_index_builder_finish(builder_, &index, out_result);
  if (!loomc_status_is_ok(status)) return status;
  finished_ = true;
  if (!loomc_result_succeeded(*out_result)) return status;

  std::vector<ProviderSymbol> symbols;
  const auto count = loomc_link_index_symbol_count(index);
  symbols.reserve(count);
  for (loomc_host_size_t i = 0; i < count; ++i) {
    loomc_link_index_symbol_t symbol = {};
    loomc_link_index_module_t module = {};
    loomc_link_index_provider_t provider = {};
    loomc_link_index_symbol_at(index, i, &symbol);
    loomc_link_index_module_at(index, symbol.module_ordinal, &module);
    loomc_link_index_provider_at(index, symbol.provider_ordinal, &provider);
    ProviderSymbol copied;
    copied.ordinal = symbol.ordinal;
    copied.provider_ordinal = symbol.provider_ordinal;
    copied.module_ordinal = symbol.module_ordinal;
    copied.provider_name = Copy(provider.name);
    copied.module_name = Copy(module.name);
    copied.name = Copy(symbol.name);
    copied.kind = symbol.kind;
    copied.identity = symbol.identity;
    copied.flags = symbol.flags;
    for (const auto& function : providers_[symbol.provider_ordinal].functions) {
      if (function.name == copied.name) {
        copied.function_kind = function.kind;
        copied.export_symbol = function.export_symbol;
        break;
      }
    }
    symbols.push_back(std::move(copied));
  }
  out_frozen->reset(new FrozenProviders(context_, index, std::move(symbols)));
  return status;
}

}  // namespace hrx::loomify_gguf
