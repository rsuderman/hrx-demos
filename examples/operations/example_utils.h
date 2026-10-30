// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_EXAMPLES_OPERATIONS_EXAMPLE_UTILS_H_
#define HRX_SOLUTIONS_EXAMPLES_OPERATIONS_EXAMPLE_UTILS_H_
#include "lib/operations/operation_graph.h"
#include <iostream>
namespace hrx::operations::example {
inline ValueId M(OperationGraph &g, uint64_t rows, uint64_t width,
                 std::string *error, ElementType e = ElementType::F32,
                 Encoding encoding = Encoding::Dense,
                 Layout layout = Layout::RowMajor) {
  auto t = TensorType::matrix(rows, width, e);
  t.setEncoding(encoding).setLayout(layout);
  return g.tensor(t, error);
}
inline ValueId V(OperationGraph &g, uint64_t n, ElementType e,
                 std::string *error) {
  return g.tensor(TensorType::vector(n, e), error);
}
inline ValueId H(OperationGraph &g, uint64_t rows, uint64_t heads,
                 uint64_t width, ElementType e, std::string *error) {
  auto t = TensorType::vector(rows, e);
  t.addDimension(heads).addDimension(width);
  return g.tensor(t, error);
}
inline bool Print(const OperationGraph &g, size_t expected,
                  std::string *error) {
  if (g.invocations().size() != expected) {
    *error = "unexpected invocation count";
    return false;
  }
  for (const auto &call : g.invocations())
    std::cout << OpCodeName(call.opcode()) << '\n';
  std::cout.flush();
  if (!std::cout) {
    *error = "writing example failed";
    return false;
  }
  return true;
}
} // namespace hrx::operations::example
#endif
