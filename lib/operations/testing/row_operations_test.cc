// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/row_operations.h"
#include "lib/operations/testing/operation_test_util.h"
#include <limits>
namespace hrx::operations {
namespace {
using namespace test;
TEST(RowOperations, RangeAndLastActivePreserveExplicitReadWrite) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineCopyRows(&r, &e));
  OperationGraph g(r);
  auto x = M(g, 7, 5), y = M(g, 3, 5), one = M(g, 1, 5), begin = S(g),
       count = S(g);
  CopyRowsAttrs a;
  a.width = 5;
  auto append = [&](ValueId out, ValueId control) {
    return g.operation(OpCode::CopyRows)
        .bind(Port::Input, x)
        .bind(Port::Output, out)
        .bind(Port::RowBegin, begin)
        .bind(Port::RowCount, control)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(y, count)) << e;
  a.selection = RowSelection::LastActive;
  ASSERT_TRUE(append(one, count));
  EXPECT_FALSE(append(y, count));
  EXPECT_FALSE(append(M(g, 1, 6), count));
  EXPECT_FALSE(append(M(g, 1, 5, ElementType::F16), count));
  EXPECT_FALSE(append(one, S(g, ScalarType::I32)));
  a.selection = RowSelection::Range;
  EXPECT_FALSE(append(x, count));
  a.selection = static_cast<RowSelection>(99);
  EXPECT_FALSE(append(one, count));
  EXPECT_EQ(g.invocations().size(), 2);
  EXPECT_EQ(r.find(OpCode::CopyRows)->port(Port::Output)->access(),
            Access::ReadWrite);
}
TEST(RowOperations, ArgMaxWidthAxisTieAndVectorOutput) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineArgMax(&r, &e));
  OperationGraph g(r);
  auto x = M(g, 3, 7), y = V(g, 3, ElementType::I32);
  ArgMaxAttrs a;
  a.width = 7;
  auto append = [&](ValueId out) {
    return g.operation(OpCode::ArgMax)
        .bind(Port::Input, x)
        .bind(Port::Output, out)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(y)) << e;
  EXPECT_FALSE(append(V(g, 7, ElementType::I32)));
  EXPECT_FALSE(append(V(g, 3, ElementType::I64)));
  EXPECT_FALSE(append(M(g, 3, 1, ElementType::I32)));
  a.axis = 0;
  EXPECT_FALSE(append(y));
  a.axis = 1;
  a.width = 3;
  EXPECT_FALSE(append(y));
  a.width = 7;
  a.tie_break = static_cast<TieBreak>(99);
  EXPECT_FALSE(append(y));
  EXPECT_EQ(g.invocations().size(), 1);
  EXPECT_EQ(std::get<ArgMaxAttrs>(g.invocations()[0].attributes()).tie_break,
            TieBreak::LowestIndex);
}
TEST(RowOperations, ArgMaxWidthsKeepEveryPossibleIndexRepresentableAsI32) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(DefineArgMax(&registry, &error));
  OperationGraph graph(registry);
  auto output = V(graph, 1, ElementType::I32);
  ArgMaxAttrs attrs;
  attrs.width = uint64_t(std::numeric_limits<int32_t>::max()) + 1;
  auto append = [&]() {
    return graph.operation(OpCode::ArgMax)
        .bind(Port::Input, M(graph, 1, attrs.width))
        .bind(Port::Output, output)
        .attributes(attrs)
        .append(&error);
  };
  ASSERT_TRUE(append()) << error;
  ++attrs.width;
  EXPECT_FALSE(append());
  EXPECT_NE(error.find("fit I32"), std::string::npos);
  attrs.width = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(append());
  EXPECT_NE(error.find("fit I32"), std::string::npos);
  EXPECT_EQ(graph.invocations().size(), 1);
}
} // namespace
} // namespace hrx::operations
