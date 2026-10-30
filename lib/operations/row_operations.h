// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_ROW_OPERATIONS_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_ROW_OPERATIONS_H_
#include "lib/operations/operation_graph.h"
namespace hrx::operations {
bool DefineCopyRows(OperationRegistry *registry, std::string *error);
bool DefineArgMax(OperationRegistry *registry, std::string *error);
} // namespace hrx::operations
#endif
