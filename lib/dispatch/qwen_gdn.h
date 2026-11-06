// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_QWEN_GDN_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_QWEN_GDN_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch {
// Complete normalized ordinary0.8B GDN layer, including shared whole FFN.
std::vector<Registration> QwenGdnRegistrations();
} // namespace hrx::dispatch
#endif
