// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_SHAPES_RESULT_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_SHAPES_RESULT_H_

#include "lib/operations/operation_graph.h"

namespace hrx::dispatch::shapes {
struct Operand {
  operations::Port port;
  operations::ValueId value;
};
struct Result {
  operations::TensorType type;
  // State-effect operations have no producer result; their explicit operands
  // retain all logical identities and native buffers.
  bool has_result = true;
  operations::Port port = operations::Port::Output;
  operations::Access access = operations::Access::Write;
  // Invalid for a fresh Write output. ReadWrite returns exactly the source
  // graph's operand identity, not a new value or an inferred physical alias.
  operations::ValueId alias;
};

// Supports registered producer families and the explicit RotatePublishKV/
// PublishToken effects. A state effect returns has_result=false. Inputs bind
// existing graph values and omit fresh outputs. Attributes and types retain
// their existing operation meanings. Checks use the built-in operation
// definitions, including operand identity and optional active-row controls. No
// source graph mutation, output allocation, runtime validation or native
// implementation selection. Failure leaves *out unchanged; graph must outlive
// any returned alias.
bool InferResult(const operations::OperationGraph &graph,
                 operations::OpCode opcode,
                 const operations::OperationAttrs &attributes,
                 const std::vector<Operand> &inputs, Result *out,
                 std::string *error);
} // namespace hrx::dispatch::shapes
#endif
