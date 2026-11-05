// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/attention_builders.h"
#include "lib/operations/attention_operations.h"
#include "gtest/gtest.h"
#include <cmath>
namespace hrx::dispatch {
namespace {
using namespace operations;
class AttentionDispatchTest : public ::testing::Test {
protected:
  OperationRegistry ops;
  OperationGraph graph{ops};
  Registry registry;
  std::string error;
  void SetUp() override {
    ASSERT_TRUE(DefineRope(&ops, &error));
    ASSERT_TRUE(DefineRotatePublishKV(&ops, &error));
    ASSERT_TRUE(DefineCausalAttention(&ops, &error));
    for (auto entry : AttentionRegistrations())
      ASSERT_TRUE(registry.Add(entry, &error)) << error;
  }
  ValueId Matrix(uint64_t width) {
    return graph.tensor(TensorType::matrix(1, width, ElementType::F32), &error);
  }
  ValueId Vector(uint64_t width, ElementType t) {
    return graph.tensor(TensorType::vector(width, t), &error);
  }
  ValueId Cache(uint64_t width = 128) {
    auto t = TensorType::vector(4, ElementType::F16);
    t.addDimension(8).addDimension(width);
    return graph.tensor(t, &error);
  }
};
TEST_F(AttentionDispatchTest, RopeRequiresActualFactorsAndOwnedFreshResult) {
  auto x = Matrix(3072), p = Vector(1, ElementType::I32),
       f = Vector(64, ElementType::F32);
  RopeAttrs a{24, 128, 128, 500000, RopeScaling::None, 1, FactorMode::PerPair};
  std::vector<shapes::Operand> operands{
      {Port::Input, x}, {Port::Positions, p}, {Port::Factors, f}};
  shapes::Result result;
  ASSERT_TRUE(
      shapes::InferResult(graph, OpCode::Rope, a, operands, &result, &error))
      << error;
  EXPECT_TRUE(result.has_result);
  EXPECT_FALSE(result.alias.valid());
  EXPECT_EQ(result.type.dimension(1), 3072);
  OperationAttrs attrs = a;
  EXPECT_TRUE(
      registry.Resolve(Id::Rope, {graph, attrs, operands, result}, {}, &error));
  a.base = 10000;
  attrs = a;
  EXPECT_FALSE(
      registry.Resolve(Id::Rope, {graph, attrs, operands, result}, {}, &error));
  operands.pop_back();
  EXPECT_FALSE(
      shapes::InferResult(graph, OpCode::Rope, a, operands, &result, &error));
}
TEST_F(AttentionDispatchTest,
       StateEffectRetainsTwoDistinctCachesAndRejectsUnequalWidths) {
  auto k = Matrix(1024), v = Matrix(1024), kc = Cache(), vc = Cache(),
       p = Vector(1, ElementType::I32), f = Vector(64, ElementType::F32);
  RotatePublishKVAttrs a{
      8, 128, 128, 4, 128, 500000, RopeScaling::None, 1, FactorMode::PerPair};
  std::vector<shapes::Operand> operands{
      {Port::Key, k},         {Port::Value, v},     {Port::KeyCache, kc},
      {Port::ValueCache, vc}, {Port::Positions, p}, {Port::Factors, f}};
  shapes::Result result;
  ASSERT_TRUE(shapes::InferResult(graph, OpCode::RotatePublishKV, a, operands,
                                  &result, &error))
      << error;
  EXPECT_FALSE(result.has_result);
  EXPECT_FALSE(result.alias.valid());
  OperationAttrs attrs = a;
  EXPECT_TRUE(registry.Resolve(Id::RotatePublishKV,
                               {graph, attrs, operands, result}, {}, &error));
  operands[3].value = kc;
  EXPECT_FALSE(shapes::InferResult(graph, OpCode::RotatePublishKV, a, operands,
                                   &result, &error));
  operands[3].value = vc;
  a.value_width = 64;
  attrs = a;
  EXPECT_FALSE(registry.Resolve(Id::RotatePublishKV,
                                {graph, attrs, operands, result}, {}, &error));
  operands.push_back(operands[0]);
  EXPECT_FALSE(shapes::InferResult(graph, OpCode::RotatePublishKV, a, operands,
                                   &result, &error));
}
TEST_F(AttentionDispatchTest,
       AttentionUsesQueryShapeAndChecksActualNumericalPolicy) {
  auto q = Matrix(3072), kc = Cache(), vc = Cache(),
       p = Vector(1, ElementType::I32);
  auto begin = graph.scalar(ScalarType::I64, &error),
       count = graph.scalar(ScalarType::I64, &error);
  CausalAttentionAttrs a{24, 8, 128, 128, 4, 1.0 / std::sqrt(128.0)};
  std::vector<shapes::Operand> operands{
      {Port::Query, q},     {Port::KeyCache, kc},    {Port::ValueCache, vc},
      {Port::Positions, p}, {Port::RowBegin, begin}, {Port::RowCount, count}};
  shapes::Result result;
  ASSERT_TRUE(shapes::InferResult(graph, OpCode::CausalAttention, a, operands,
                                  &result, &error))
      << error;
  EXPECT_EQ(result.type.dimension(1), 3072);
  OperationAttrs attrs = a;
  EXPECT_TRUE(registry.Resolve(Id::CausalAttention,
                               {graph, attrs, operands, result}, {}, &error));
  a.numerical_policy = NumericalPolicy::LlamaPacketMixed;
  ASSERT_TRUE(shapes::InferResult(graph, OpCode::CausalAttention, a, operands,
                                  &result, &error))
      << error;
  attrs = a;
  EXPECT_FALSE(registry.Resolve(Id::CausalAttention,
                                {graph, attrs, operands, result}, {}, &error));
  a.numerical_policy = NumericalPolicy::F32Accumulate;
  a.scale = 1;
  attrs = a;
  EXPECT_FALSE(registry.Resolve(Id::CausalAttention,
                                {graph, attrs, operands, result}, {}, &error));
}
} // namespace
} // namespace hrx::dispatch
