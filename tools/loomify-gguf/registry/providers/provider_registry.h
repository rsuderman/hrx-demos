// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_PROVIDERS_PROVIDER_REGISTRY_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_PROVIDERS_PROVIDER_REGISTRY_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "loomc/link_index.h"
#include "loomc/module.h"

namespace hrx::loomify_gguf {

// Copied inspection metadata. Ordinals identify symbols only within their owning
// frozen index; they are not cross-module IR handles or callable ABI contracts.
struct ProviderSymbol {
  loomc_host_size_t ordinal = 0;
  loomc_host_size_t provider_ordinal = 0;
  loomc_host_size_t module_ordinal = 0;
  std::string provider_name;
  std::string module_name;
  std::string name;
  loomc_link_symbol_kind_t kind = LOOMC_LINK_SYMBOL_KIND_UNKNOWN;
  loomc_link_symbol_identity_t identity = LOOMC_LINK_SYMBOL_IDENTITY_PRIVATE;
  loomc_link_symbol_flags_t flags = 0;
  loomc_module_function_kind_t function_kind = LOOMC_MODULE_FUNCTION_KIND_UNKNOWN;
  std::optional<std::string> export_symbol;
};

class ProviderRegistryBuilder;

// Immutable inspection and native-linking view. Accessors borrow while this owner
// is alive. Copies of symbol descriptions remain valid after it is destroyed.
class FrozenProviders {
 public:
  ~FrozenProviders();
  FrozenProviders(FrozenProviders&& other) noexcept;
  FrozenProviders& operator=(FrozenProviders&& other) noexcept;
  FrozenProviders(const FrozenProviders&) = delete;
  FrozenProviders& operator=(const FrozenProviders&) = delete;

  loomc_context_t* context() const { return context_; }
  loomc_link_index_t* native_index() const { return index_; }
  std::vector<ProviderSymbol> EnumerateSymbols() const { return symbols_; }
  // Returns all globally visible candidates in native order, never a winner.
  // Private symbols are available only through EnumerateSymbols().
  std::vector<ProviderSymbol> FindCandidates(loomc_string_view_t name) const;

 private:
  friend class ProviderRegistryBuilder;
  FrozenProviders(loomc_context_t* context, loomc_link_index_t* index,
                  std::vector<ProviderSymbol> symbols);
  loomc_context_t* context_ = nullptr;
  loomc_link_index_t* index_ = nullptr;
  std::vector<ProviderSymbol> symbols_;
};

// Exclusively mutated builder. AddModule takes a synchronous independent snapshot;
// the original module may be mutated or released immediately on success.
class ProviderRegistryBuilder {
 public:
  static loomc_status_t Create(
      loomc_context_t* context,
      std::unique_ptr<ProviderRegistryBuilder>* out_builder);
  ~ProviderRegistryBuilder();
  ProviderRegistryBuilder(const ProviderRegistryBuilder&) = delete;
  ProviderRegistryBuilder& operator=(const ProviderRegistryBuilder&) = delete;

  loomc_status_t AddModule(loomc_string_view_t label,
                          const loomc_module_t* module);
  // Outputs owned references; caller releases result even for semantic failure.
  // Native OK finish consumes the builder's mutation state, not its allocation.
  loomc_status_t Freeze(std::unique_ptr<FrozenProviders>* out_frozen,
                       loomc_result_t** out_result);

 private:
  struct FunctionMetadata {
    std::string name;
    loomc_module_function_kind_t kind;
    std::optional<std::string> export_symbol;
  };
  struct ProviderMetadata {
    std::string label;
    std::vector<FunctionMetadata> functions;
  };
  ProviderRegistryBuilder(loomc_context_t* context,
                          loomc_link_index_builder_t* builder);
  loomc_context_t* context_;
  loomc_link_index_builder_t* builder_;
  bool finished_ = false;
  std::vector<ProviderMetadata> providers_;
};

}  // namespace hrx::loomify_gguf

#endif  // HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_PROVIDERS_PROVIDER_REGISTRY_H_
