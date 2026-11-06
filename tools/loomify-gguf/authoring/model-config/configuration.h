// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_CONFIGURATION_H_
#define HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_CONFIGURATION_H_

#include "loomc/module.h"
#include "tools/model/model_descriptor.h"
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hrx::loomify_gguf {
enum class LinkedProfile { Llama3BModel, Qwen08BFfn };
struct LinkedConfigurationSettings {
  LinkedProfile profile = LinkedProfile::Llama3BModel;
  uint64_t token_capacity = 128;
  uint64_t context_capacity = 4096;
  // Omit to derive the rounded pool; when supplied it must agree exactly.
  std::optional<uint64_t> pool_capacity;
};
struct ParameterRequirement {
  std::string role;
  std::optional<uint32_t> layer;
  std::string source_asset;
  model::TensorRecord tensor;
};
struct ColdAllocation {
  uint64_t bytes = 0, alignment = 256, clear_bytes = 0;
};
struct DirectBindingMinima {
  uint64_t residual = 0, control = 12, map = 0, cache = 0;
  uint64_t tokens_history = 0, progress = 32;
  // Parameters are described by ParameterRequirement, not a flat file bound.
};
struct LlamaLinkedLayout {
  uint64_t token_capacity = 0, context_capacity = 0, pool_capacity = 0;
  uint64_t layer_count = 0, hidden_size = 3072, kv_size = 1024;
  uint64_t history_capacity = 0, cache_base = 0, map_origin = 2048;
  DirectBindingMinima prefill, decode;
  // Epoch binding order is parameters,residual,metadata,map,cache,input,output.
  std::array<uint64_t, 7> epoch_minima{};
  std::array<ColdAllocation, 11> cold_allocations{};
  std::array<uint64_t, 8> row_views{};
  std::array<uint64_t, 64> origins{};
  // Eight header words, followed by the five-word region record.
  std::array<uint64_t, 13> pooled_geometry{};
};
struct LinkedConfigurationFacts {
  LinkedProfile profile = LinkedProfile::Llama3BModel;
  std::vector<ParameterRequirement> parameters;
  std::optional<LlamaLinkedLayout> llama_layout;
  // Data-only Loom source. Kept unsealed until linked with maintained roots.
  std::string source;
};
// Both functions borrow input and leave output unchanged on failure. Facts own
// every name/range and may outlive the descriptor and its source file.
bool DeriveLinkedConfiguration(const model::ModelDescriptor &descriptor,
                               const std::string &source_asset,
                               const LinkedConfigurationSettings &settings,
                               LinkedConfigurationFacts *out,
                               std::string *error);
struct LinkedConfiguration {
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module{
      nullptr, loomc_module_release};
  LinkedConfigurationFacts facts;
};
loomc_status_t CreateLinkedConfiguration(
    loomc_context_t *context, loomc_workspace_t *workspace,
    const model::ModelDescriptor &descriptor, const std::string &source_asset,
    const LinkedConfigurationSettings &settings, LinkedConfiguration *out,
    loomc_result_t **result);
} // namespace hrx::loomify_gguf
#endif
