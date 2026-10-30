// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_CONTROL_OPERATIONS_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_CONTROL_OPERATIONS_H_
#include "lib/operations/operation_graph.h"
namespace hrx::operations {
bool DefinePublishToken(OperationRegistry *registry, std::string *error);
}
#endif
