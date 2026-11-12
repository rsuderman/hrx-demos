// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_PLATFORMS_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_PLATFORMS_H_
#include <string_view>
#include <vector>
namespace hrx::loomify_gguf {
enum class RequiredOperation { kScaledF16ProjectionV1, kAttentionPrefillH256V1 };
struct OperationImplementation {
  RequiredOperation operation;
  const char* product;
};
// A maintained generation recipe, not a physical device or a numerical policy.
struct PlatformFiles {
  const char* selector;
  const char* backend;
  const char* targets;
  std::vector<OperationImplementation> implementations;
};
const PlatformFiles* SelectPlatform(std::string_view selector);
}  // namespace hrx::loomify_gguf
#endif
