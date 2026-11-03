// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/shapes/result.h"
#include "lib/llama/structure.h"
#include "gtest/gtest.h"
#include <limits>

namespace hrx::dispatch::shapes {
namespace {
using namespace operations;
class ResultTest : public ::testing::Test {
protected:
  OperationRegistry registry;
  OperationGraph graph{registry};
  std::string error;
  Result result;
  model::ModelDescriptor model;
  void SetUp() override {
    model.family = "llama";
    model.embedding_width = 3072;
    model.key_head_width = 128;
    model.layers.resize(1);
    model.layers[0].feed_forward_width = 8192;
    model.layers[0].query_heads = 24;
    model.layers[0].kv_heads = 8;
  }
  ValueId Matrix(uint64_t rows, uint64_t width) {
    return graph.tensor(TensorType::matrix(rows, width, ElementType::F32),
                        &error);
  }
  ValueId Weight(uint64_t input, uint64_t output,
                 Encoding encoding = Encoding::Q4_K) {
    auto type = TensorType::matrix(input, output, ElementType::F32);
    type.setEncoding(encoding).setLayout(Layout::GgmlWeightRows);
    return graph.tensor(type, &error);
  }
  void Shape(uint64_t rows, uint64_t width) {
    EXPECT_EQ(result.type.rank(), 2u);
    EXPECT_EQ(result.type.dimension(0), rows);
    EXPECT_EQ(result.type.dimension(1), width);
    EXPECT_EQ(result.type.elementType(), ElementType::F32);
    EXPECT_EQ(result.type.encoding(), Encoding::Dense);
    EXPECT_EQ(result.type.layout(), Layout::RowMajor);
  }
};
TEST_F(ResultTest, NormPreservesCapacityAndFreshResultForBothPhases) {
  for (auto phase : {Phase::Prefill, Phase::Decode}) {
    for (uint64_t rows : {1, 7}) {
      llama::LlamaBuildOptions options;
      options.phase = phase;
      options.row_capacity = rows;
      options.cache_capacity = 4096;
      options.history_capacity = 4096;
      options.source_asset = "fixture.gguf";
      auto input = Matrix(options.row_capacity, model.embedding_width);
      auto gamma = graph.tensor(
          TensorType::vector(model.embedding_width, ElementType::F32), &error);
      auto begin = graph.scalar(ScalarType::I64, &error);
      auto count = graph.scalar(ScalarType::I64, &error);
      const auto values = graph.values().size();
      ASSERT_TRUE(InferResult(graph, OpCode::RmsNorm,
                              RmsNormAttrs{model.embedding_width},
                              {{Port::Input, input},
                               {Port::Gamma, gamma},
                               {Port::RowBegin, begin},
                               {Port::RowCount, count}},
                              &result, &error))
          << error;
      Shape(rows, model.embedding_width);
      EXPECT_FALSE(result.alias.valid());
      EXPECT_EQ(result.access, Access::Write);
      EXPECT_EQ(result.port, Port::Output);
      EXPECT_EQ(graph.values().size(), values);
      EXPECT_TRUE(graph.invocations().empty());
    }
  }
}
TEST_F(ResultTest, LinearDerivesQueryAndKeyWidthsFromModelFacts) {
  for (uint64_t rows : {1, 7}) {
    for (auto heads : {model.layers[0].query_heads, model.layers[0].kv_heads}) {
      auto width = heads * model.key_head_width;
      auto input = Matrix(rows, model.embedding_width);
      auto weight = Weight(model.embedding_width, width);
      ASSERT_TRUE(InferResult(
          graph, OpCode::Linear, LinearAttrs{model.embedding_width, width},
          {{Port::Input, input}, {Port::Weight, weight}}, &result, &error))
          << error;
      Shape(rows, width);
      EXPECT_FALSE(result.alias.valid());
    }
  }
}
TEST_F(ResultTest, GateUpAndResidualDownIncludeKnownEncodingsAndExactAlias) {
  for (auto encoding : {Encoding::Q4_K, Encoding::Q6_K, Encoding::Q8_0}) {
    for (uint64_t rows : {1, 7}) {
      const auto hidden = model.embedding_width;
      const auto ffn = model.layers[0].feed_forward_width;
      auto input = Matrix(rows, hidden);
      auto gate = Weight(hidden, ffn, encoding);
      auto up = Weight(hidden, ffn, encoding);
      ASSERT_TRUE(InferResult(graph, OpCode::GatedMatmul,
                              GatedMatmulAttrs{hidden, ffn},
                              {{Port::Input, input},
                               {Port::GateWeight, gate},
                               {Port::UpWeight, up}},
                              &result, &error))
          << error;
      Shape(rows, ffn);
      EXPECT_FALSE(result.alias.valid());
      auto activation = graph.tensor(result.type, &error);
      auto down = Weight(ffn, hidden, encoding);
      auto residual = Matrix(rows, hidden);
      ASSERT_TRUE(InferResult(graph, OpCode::LinearResidual,
                              LinearResidualAttrs{ffn, hidden},
                              {{Port::Input, activation},
                               {Port::Weight, down},
                               {Port::Residual, residual}},
                              &result, &error))
          << error;
      Shape(rows, hidden);
      EXPECT_EQ(result.alias, residual);
      EXPECT_NE(result.alias, input);
      EXPECT_EQ(result.port, Port::Residual);
      EXPECT_EQ(result.access, Access::ReadWrite);
    }
  }
}
TEST_F(ResultTest, FailurePreservesInputsAndPreviousResult) {
  auto input = Matrix(3, 3072);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  ASSERT_TRUE(InferResult(graph, OpCode::RmsNorm, RmsNormAttrs{3072},
                          {{Port::Input, input}, {Port::Gamma, gamma}}, &result,
                          &error));
  const auto values = graph.values().size();
  for (auto width : {0, 3071}) {
    EXPECT_FALSE(InferResult(
        graph, OpCode::RmsNorm, RmsNormAttrs{uint64_t(width)},
        {{Port::Input, input}, {Port::Gamma, gamma}}, &result, &error));
    EXPECT_FALSE(error.empty());
    Shape(3, 3072);
  }
  EXPECT_EQ(graph.values().size(), values);
  EXPECT_EQ(graph.value(input)->tensorType()->dimension(1), 3072u);
  EXPECT_TRUE(graph.invocations().empty());
}
TEST_F(ResultTest, RejectsIncompatibleAxesAndEncodingsForEveryRule) {
  auto input = Matrix(3, 3072);
  auto bad_gamma =
      graph.tensor(TensorType::vector(3071, ElementType::F32), &error);
  EXPECT_FALSE(InferResult(graph, OpCode::RmsNorm, RmsNormAttrs{3072},
                           {{Port::Input, input}, {Port::Gamma, bad_gamma}},
                           &result, &error));
  auto good = Weight(3072, 8192);
  for (auto bad : {Weight(3071, 8192), Weight(3072, 8191),
                   Weight(3072, 8192, Encoding::Q5_K),
                   Weight(3072, 8192, Encoding::Dense)}) {
    EXPECT_FALSE(InferResult(
        graph, OpCode::GatedMatmul, GatedMatmulAttrs{3072, 8192},
        {{Port::Input, input}, {Port::GateWeight, good}, {Port::UpWeight, bad}},
        &result, &error));
  }
  for (auto bad : {Weight(3071, 1024), Weight(3072, 1023),
                   Weight(3072, 1024, Encoding::Q4_0)}) {
    EXPECT_FALSE(InferResult(graph, OpCode::Linear, LinearAttrs{3072, 1024},
                             {{Port::Input, input}, {Port::Weight, bad}},
                             &result, &error));
  }
  auto activation = Matrix(3, 8192);
  auto down = Weight(8192, 3072);
  for (auto residual : {Matrix(2, 3072), Matrix(3, 3071)}) {
    EXPECT_FALSE(InferResult(graph, OpCode::LinearResidual,
                             LinearResidualAttrs{8192, 3072},
                             {{Port::Input, activation},
                              {Port::Weight, down},
                              {Port::Residual, residual}},
                             &result, &error));
  }
}
TEST_F(ResultTest, ReusesBuiltinIdentityAndActiveControlChecks) {
  auto input = Matrix(3, 3072);
  auto weight = Weight(3072, 3072);
  EXPECT_FALSE(InferResult(
      graph, OpCode::LinearResidual, LinearResidualAttrs{3072, 3072},
      {{Port::Input, input}, {Port::Weight, weight}, {Port::Residual, input}},
      &result, &error));
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  for (auto control : {graph.scalar(ScalarType::I32, &error),
                       graph.scalar(ScalarType::I64, &error)}) {
    EXPECT_FALSE(InferResult(
        graph, OpCode::RmsNorm, RmsNormAttrs{3072},
        {{Port::Input, input}, {Port::Gamma, gamma}, {Port::RowBegin, control}},
        &result, &error));
  }
  EXPECT_FALSE(InferResult(
      graph, OpCode::RmsNorm, RmsNormAttrs{3072},
      {{Port::Input, input}, {Port::Input, input}, {Port::Gamma, gamma}},
      &result, &error));
  EXPECT_FALSE(InferResult(
      graph, OpCode::RmsNorm, RmsNormAttrs{3072},
      {{Port::Input, input}, {Port::Gamma, gamma}, {Port::Output, input}},
      &result, &error));
  OperationGraph foreign(registry);
  auto foreign_input =
      foreign.tensor(TensorType::matrix(3, 3072, ElementType::F32), &error);
  EXPECT_FALSE(InferResult(graph, OpCode::RmsNorm, RmsNormAttrs{3072},
                           {{Port::Input, foreign_input}, {Port::Gamma, gamma}},
                           &result, &error));
}
TEST_F(ResultTest, RejectsOverflowInInputsWeightsAndInferredOutput) {
  const auto max = uint64_t(std::numeric_limits<int64_t>::max());
  auto gamma = graph.tensor(TensorType::vector(4, ElementType::F32), &error);
  auto input = Matrix(max / 4, 4);
  EXPECT_FALSE(InferResult(graph, OpCode::RmsNorm, RmsNormAttrs{4},
                           {{Port::Input, input}, {Port::Gamma, gamma}},
                           &result, &error));
  auto small = Matrix(1, 1);
  auto huge_weight = Weight(max, 2);
  EXPECT_FALSE(InferResult(graph, OpCode::Linear, LinearAttrs{max, 2},
                           {{Port::Input, small}, {Port::Weight, huge_weight}},
                           &result, &error));
  // Input and weight each fit, but their projected result does not.
  input = Matrix(max / 8, 1);
  auto weight = Weight(1, 3);
  EXPECT_FALSE(InferResult(graph, OpCode::Linear, LinearAttrs{1, 3},
                           {{Port::Input, input}, {Port::Weight, weight}},
                           &result, &error));
  auto huge = Matrix(std::numeric_limits<uint64_t>::max(), 2);
  EXPECT_FALSE(InferResult(graph, OpCode::Linear, LinearAttrs{2, 3},
                           {{Port::Input, huge}, {Port::Weight, weight}},
                           &result, &error));
}
TEST_F(ResultTest, RejectsUnsupportedRuleAttributesAndMissingArguments) {
  auto input = Matrix(1, 3);
  EXPECT_FALSE(InferResult(graph, OpCode::Embedding, EmbeddingAttrs{4, 3},
                           {{Port::Input, input}}, &result, &error));
  EXPECT_FALSE(InferResult(graph, OpCode::Linear, RmsNormAttrs{3},
                           {{Port::Input, input}}, &result, &error));
  EXPECT_FALSE(InferResult(graph, OpCode::RmsNorm, RmsNormAttrs{3}, {}, &result,
                           &error));
  EXPECT_FALSE(InferResult(graph, OpCode::RmsNorm, RmsNormAttrs{3}, {}, nullptr,
                           &error));
  EXPECT_FALSE(InferResult(graph, OpCode::RmsNorm, RmsNormAttrs{3}, {}, &result,
                           nullptr));
}
} // namespace
} // namespace hrx::dispatch::shapes
