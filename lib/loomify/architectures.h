// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_H_
#include "lib/loomify/platforms.h"
#include <string_view>
#include <vector>
namespace hrx::loomify_gguf {
// Private, finite BUILD-owned product selection. No numerical routing callbacks.
struct ArchitectureFiles {
  const char *family;
  const char *implementation;
  const char *preparation;
  const char *execution;
  const char *weights;
  const char *package_name;
  std::vector<const char *> roots;
  std::vector<RequiredOperation> required_operations;
};
const ArchitectureFiles *SelectArchitecture(std::string_view family);
} // namespace hrx::loomify_gguf
#endif
