// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/qwen_attention/builder.h"
#include "lib/dispatch/qwen_gdn.h"
namespace hrx::dispatch {
// This complete-model assembly links real independently qualified families.
// Common provider libraries do not depend on the two layer implementations.
Registry Registry::Qwen08B() {
  Registry registry;
  for (auto family : {QwenCommonRegistrations(), QwenGdnRegistrations(),
                      QwenAttentionRegistrations()})
    for (auto &registration : family)
      registry.entries_.push_back(std::move(registration));
  return registry;
}
} // namespace hrx::dispatch
