// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_PROJECTION_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_PROJECTION_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch {
OperationBuilder ProjectionBuilder();
void RegisterProjections(Registry *registry);
} // namespace hrx::dispatch
#endif
