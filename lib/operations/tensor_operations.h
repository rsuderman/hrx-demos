// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_TENSOR_OPERATIONS_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_TENSOR_OPERATIONS_H_
#include "lib/operations/operation_graph.h"
namespace hrx::operations {
bool DefineEmbedding(OperationRegistry *registry, std::string *error);
bool DefineRmsNorm(OperationRegistry *registry, std::string *error);
bool DefineQuantize(OperationRegistry *registry, std::string *error);
bool DefineLinear(OperationRegistry *registry, std::string *error);
} // namespace hrx::operations
#endif
