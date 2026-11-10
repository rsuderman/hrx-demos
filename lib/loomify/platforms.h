// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_PLATFORMS_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_PLATFORMS_H_
#include <string_view>
namespace hrx::loomify_gguf {
// A maintained generation recipe, not a physical device or a numerical policy.
struct PlatformFiles {
  const char* selector;
  const char* backend;
  const char* targets;
};
const PlatformFiles* SelectPlatform(std::string_view selector);
}  // namespace hrx::loomify_gguf
#endif
