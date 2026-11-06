// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_QWEN_OUTPUT_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_QWEN_OUTPUT_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch {
// Ordinary Qwen0.8B Q6/F16 embedding and selected-epoch head registrations.
std::vector<Registration> QwenOutputRegistrations();
} // namespace hrx::dispatch
#endif
