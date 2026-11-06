// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_QWEN_OPERATIONS_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_QWEN_OPERATIONS_H_
#include "lib/operations/operation_graph.h"
namespace hrx::operations {
struct QwenWeightRole {
  Port port;
  const char *role;
  const char *suffix;
};
// Canonical descriptor role/key mapping for complete layer parameter
// validation.
const std::vector<QwenWeightRole> &QwenLayerWeightRoles(OpCode);
bool DefineQwenGdnLayer(OperationRegistry *, std::string *error);
bool DefineQwenAttentionLayer(OperationRegistry *, std::string *error);
bool DefineQwenEpochHead(OperationRegistry *, std::string *error);
} // namespace hrx::operations
#endif
