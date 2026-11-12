// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_H_
#include "lib/loomify/platforms.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/publication.h"
#include <string_view>
#include <vector>
namespace hrx::loomify_gguf {
using ConfigurationEmitter = bool (*)(const model::ModelDescriptor &,
    const std::string &, const ConfigurationSettings &, ConfigurationFacts *, std::string *);
// Private, finite BUILD-owned products and data-only architecture policy.
struct ArchitectureFiles {
  const char *family;
  const char *implementation;
  const char *preparation;
  const char *execution;
  const char *weights;
  const char *package_name;
  std::vector<const char *> roots;
  std::vector<RequiredOperation> required_operations;
  ConfigurationEmitter emit_configuration = nullptr;
  PublicationLayout publication_layout = PublicationLayout::kUnspecified;
  uint64_t context_capacity_limit = 0;
};
const ArchitectureFiles *SelectArchitecture(std::string_view family);
} // namespace hrx::loomify_gguf
#endif
