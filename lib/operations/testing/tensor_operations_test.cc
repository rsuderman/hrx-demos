// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/tensor_operations.h"
#include "lib/operations/testing/operation_test_util.h"
#include <limits>
namespace hrx::operations {
namespace {
using namespace test;
TEST(TensorOperations, EmbeddingAxesVocabularyAndTokenTypes) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineEmbedding(&r, &e));
  OperationGraph g(r);
  auto ids = V(g, 3, ElementType::I32),
       weight =
           M(g, 5, 7, ElementType::F32, Encoding::Q8_0, Layout::GgmlWeightRows),
       out = M(g, 3, 5);
  EmbeddingAttrs a;
  a.vocabulary_size = 7;
  a.embedding_width = 5;
  auto append = [&](ValueId tokens, ValueId w, ValueId y) {
    return g.operation(OpCode::Embedding)
        .bind(Port::TokenIds, tokens)
        .bind(Port::Weight, w)
        .bind(Port::Output, y)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(ids, weight, out)) << e;
  EXPECT_FALSE(append(V(g, 3, ElementType::F32), weight, out));
  EXPECT_FALSE(append(
      ids, M(g, 7, 5, ElementType::F32, Encoding::Q8_0, Layout::GgmlWeightRows),
      out));
  EXPECT_FALSE(append(ids, weight, M(g, 3, 7)));
  a.vocabulary_size = 0;
  EXPECT_FALSE(append(ids, weight, out));
  EXPECT_EQ(g.invocations().size(), 1);
  EXPECT_TRUE(r.find(OpCode::Embedding)->expectsAttributes<EmbeddingAttrs>());
  EXPECT_FALSE(DefineEmbedding(&r, &e));
  EXPECT_FALSE(DefineEmbedding(nullptr, &e));
}
TEST(TensorOperations, NormGammaFiniteEpsilonAndNoInPlaceOutput) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineRmsNorm(&r, &e));
  OperationGraph g(r);
  auto x = M(g, 3, 5), gamma = V(g, 5), y = M(g, 3, 5);
  RmsNormAttrs a;
  a.width = 5;
  auto append = [&](ValueId scale, ValueId out) {
    return g.operation(OpCode::RmsNorm)
        .bind(Port::Input, x)
        .bind(Port::Gamma, scale)
        .bind(Port::Output, out)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(gamma, y)) << e;
  EXPECT_FALSE(append(V(g, 6), y));
  EXPECT_FALSE(append(gamma, x));
  a.epsilon = 0;
  EXPECT_FALSE(append(gamma, y));
  a.epsilon = -1;
  EXPECT_FALSE(append(gamma, y));
  a.epsilon = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(append(gamma, y));
  a.epsilon = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(append(gamma, y));
  EXPECT_EQ(g.invocations().size(), 1);
}
TEST(TensorOperations, QuantizeUsesExactPackedShapePolicyAndScalarControls) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineQuantize(&r, &e));
  OperationGraph g(r);
  auto x = M(g, 3, 128),
       y = M(g, 3, 128, ElementType::F32, Encoding::Q8_1X4,
             Layout::PackedQ8_1X4Rows),
       begin = S(g), count = S(g);
  QuantizeAttrs a;
  a.width = 128;
  auto append = [&](ValueId out, ValueId row_count) {
    return g.operation(OpCode::Quantize)
        .bind(Port::Input, x)
        .bind(Port::Output, out)
        .bind(Port::RowBegin, begin)
        .bind(Port::RowCount, row_count)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(y, count)) << e;
  a.policy = PackingPolicy::QkCompat;
  ASSERT_TRUE(append(y, count));
  EXPECT_FALSE(append(M(g, 3, 256, ElementType::F32, Encoding::Q8_1X4,
                        Layout::PackedQ8_1X4Rows),
                      count));
  EXPECT_FALSE(append(y, S(g, ScalarType::I32)));
  EXPECT_FALSE(append(x, count));
  a.width = 127;
  EXPECT_FALSE(append(y, count));
  a.width = 128;
  a.policy = static_cast<PackingPolicy>(99);
  EXPECT_FALSE(append(y, count));
  EXPECT_EQ(g.invocations().size(), 2);
}
TEST(TensorOperations, LinearDensePackedAndAsymmetricWeightOrientation) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineLinear(&r, &e));
  OperationGraph g(r);
  auto x = M(g, 3, 128), packed = M(g, 3, 128, ElementType::F32,
                                    Encoding::Q8_1X4, Layout::PackedQ8_1X4Rows);
  auto w = M(g, 128, 7, ElementType::F32, Encoding::Q5_K,
             Layout::GgmlWeightRows),
       y = M(g, 3, 7);
  LinearAttrs a;
  a.input_width = 128;
  a.output_width = 7;
  auto append = [&](ValueId input, ValueId weight, ValueId out) {
    return g.operation(OpCode::Linear)
        .bind(Port::Input, input)
        .bind(Port::Weight, weight)
        .bind(Port::Output, out)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(x, w, y)) << e;
  ASSERT_TRUE(append(packed, w, y));
  ASSERT_TRUE(append(
      M(g, 3, 128, ElementType::F16),
      M(g, 128, 7, ElementType::F16, Encoding::Dense, Layout::GgmlWeightRows),
      y));
  EXPECT_FALSE(append(
      x, M(g, 7, 128, ElementType::F32, Encoding::Q5_K, Layout::GgmlWeightRows),
      y));
  EXPECT_FALSE(append(x, w, M(g, 3, 7, ElementType::F16)));
  a.numerical_policy = static_cast<NumericalPolicy>(99);
  EXPECT_FALSE(append(x, w, y));
  a.numerical_policy = NumericalPolicy::F32Accumulate;
  a.output_width = 128;
  EXPECT_FALSE(append(
      x,
      M(g, 128, 128, ElementType::F32, Encoding::Dense, Layout::GgmlWeightRows),
      x));
  EXPECT_EQ(g.invocations().size(), 3);
}
} // namespace
} // namespace hrx::operations
