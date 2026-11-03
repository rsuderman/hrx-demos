// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_ATTENTION_BUILDERS_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_ATTENTION_BUILDERS_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch {
OperationBuilder AttentionBuilder(operations::OpCode operation);
std::vector<Registration> AttentionRegistrations();
} // namespace hrx::dispatch
#endif
