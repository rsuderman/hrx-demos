// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_OUTPUT_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_OUTPUT_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch {
// Fused head reads active_count from explicit Control[0], with native
// precondition1<=active_count<=input capacity and static begin0. Gamma/input,
// tied weights and fresh token/scratch are disjoint. Last-active normalization
// rebases the selected row to row zero before projection and greedy selection.
OperationBuilder OutputBuilder();
OperationBuilder PublicationBuilder();
std::vector<Registration> OutputRegistrations();
} // namespace hrx::dispatch
#endif
