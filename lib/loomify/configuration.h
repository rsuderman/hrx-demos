// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_CONFIGURATION_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_CONFIGURATION_H_
#include "tools/model/model_descriptor.h"
#include <optional>
#include <string>
#include <vector>
namespace hrx::loomify_gguf {
struct ConfigurationSettings {
  uint64_t token_capacity = 128;
  uint64_t context_capacity = 512;
};
struct TensorRequirement {
  std::string role;
  std::optional<uint32_t> layer;
  std::string source_asset;
  model::TensorRecord tensor;
};
struct ConfigurationFacts {
  std::vector<TensorRequirement> parameters;
  std::string source;
};
// Data only: no file I/O, native module or numerical invocation construction.
// Input facts are borrowed; output owns names/ranges/source and may outlive
// them. Family selects the maintained architecture fact mapping. Deployment
// bounds are explicit, with physical pool rounded to 64. Failure preserves
// *out; success clears error. Native parsing/linking belongs to the caller.
bool DeriveConfiguration(const model::ModelDescriptor &,
                         const std::string &source_asset,
                         const ConfigurationSettings &, ConfigurationFacts *out,
                         std::string *error);
} // namespace hrx::loomify_gguf
#endif
