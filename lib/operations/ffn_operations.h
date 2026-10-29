// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_FFN_OPERATIONS_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_FFN_OPERATIONS_H_

#include <string>

#include "lib/operations/operation_graph.h"

namespace hrx::operations {

// Illustrative authoring contracts, not native kernel registrations. Each call
// atomically registers one definition; previously registered operations remain
// unchanged on failure. Both pointers are required.
bool DefineGatedMatmul(OperationRegistry *registry, std::string *error);
bool DefineLinearResidual(OperationRegistry *registry, std::string *error);

} // namespace hrx::operations

#endif // HRX_SOLUTIONS_LIB_OPERATIONS_FFN_OPERATIONS_H_
