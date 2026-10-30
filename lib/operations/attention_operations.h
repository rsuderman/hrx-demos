// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_ATTENTION_OPERATIONS_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_ATTENTION_OPERATIONS_H_
#include "lib/operations/operation_graph.h"
namespace hrx::operations {
bool DefineRope(OperationRegistry *registry, std::string *error);
bool DefineRotatePublishKV(OperationRegistry *registry, std::string *error);
bool DefineCausalAttention(OperationRegistry *registry, std::string *error);
} // namespace hrx::operations
#endif
