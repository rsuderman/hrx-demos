// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/platforms.h"
#include "lib/serving/library_resources.h"
namespace hrx::loomify_gguf {
const PlatformFiles* SelectPlatform(std::string_view selector) {
  static const PlatformFiles gfx11{
      "amdgpu:gfx11-generic", "amdgpu", resources::kGfx11Targets,
      {{RequiredOperation::kScaledF16ProjectionV1, resources::kGfx11ScaledF16Projection}}};
  return selector == gfx11.selector ? &gfx11 : nullptr;
}
}  // namespace hrx::loomify_gguf
