// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_TESTING_OPERATION_TEST_UTIL_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_TESTING_OPERATION_TEST_UTIL_H_
#include "lib/operations/operation_graph.h"
#include "gtest/gtest.h"
namespace hrx::operations::test {
inline ValueId M(OperationGraph &g, uint64_t rows, uint64_t width,
                 ElementType e = ElementType::F32,
                 Encoding encoding = Encoding::Dense,
                 Layout layout = Layout::RowMajor) {
  auto t = TensorType::matrix(rows, width, e);
  t.setEncoding(encoding).setLayout(layout);
  std::string error;
  auto id = g.tensor(t, &error);
  EXPECT_TRUE(id.valid()) << error;
  return id;
}
inline ValueId V(OperationGraph &g, uint64_t width,
                 ElementType e = ElementType::F32) {
  std::string error;
  auto id = g.tensor(TensorType::vector(width, e), &error);
  EXPECT_TRUE(id.valid()) << error;
  return id;
}
inline ValueId H(OperationGraph &g, uint64_t rows, uint64_t heads,
                 uint64_t width, ElementType e = ElementType::F32) {
  auto t = TensorType::vector(rows, e);
  t.addDimension(heads).addDimension(width);
  std::string error;
  auto id = g.tensor(t, &error);
  EXPECT_TRUE(id.valid()) << error;
  return id;
}
inline ValueId S(OperationGraph &g, ScalarType type = ScalarType::I64) {
  std::string error;
  return g.scalar(type, &error);
}
} // namespace hrx::operations::test
#endif
