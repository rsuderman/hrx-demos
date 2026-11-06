// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_QWEN_ATTENTION_BUILDER_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_QWEN_ATTENTION_BUILDER_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch {
// Complete ordinary canonical Qwen0.8B layer, with both norms/residuals and
// shared FFN. Registry assembly/configuration remain common-owned.
OperationBuilder QwenAttentionLayerBuilder();
std::vector<Registration> QwenAttentionRegistrations();
} // namespace hrx::dispatch
#endif
