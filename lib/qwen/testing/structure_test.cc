// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/shapes/result.h"
#include "lib/operations/qwen_operations.h"
#include "lib/qwen/structure.h"
#include "lib/qwen/testing/model_fixture.h"
#include "gtest/gtest.h"
namespace hrx::qwen {
namespace {
using namespace operations;
TEST(QwenStructure, CompleteOrderRolesStateAndDeclaredResidualAlias) {
  for (bool small : {false, true}) {
    auto facts = testing::ModelFacts(24, small);
    std::string error;
    std::shared_ptr<const QwenStructure> owner;
    ASSERT_TRUE(
        BuildQwenStructure(facts, {128, 1, 1, 96, "fixture"}, &owner, &error))
        << error;
    ASSERT_EQ(owner->graph().invocations().size(), 26u);
    ASSERT_EQ(owner->inputs().layers.size(), 24u);
    size_t gdn = 0, attention = 0;
    const auto &ops = owner->graph().invocations();
    EXPECT_EQ(ops.front().opcode(), OpCode::Embedding);
    EXPECT_EQ(ops.back().opcode(), OpCode::QwenEpochHead);
    EXPECT_EQ(ops.front().binding(Port::Weight)->value(),
              ops.back().binding(Port::Weight)->value());
    for (size_t slot = 0; slot < 24; ++slot) {
      const auto &op = ops[slot + 1];
      const auto &state = owner->inputs().layers[slot];
      EXPECT_EQ(op.binding(Port::Residual)->value(), owner->inputs().residual);
      if ((slot + 1) % 4) {
        EXPECT_EQ(op.opcode(), OpCode::QwenGdnLayer);
        EXPECT_EQ(
            std::get<QwenGdnLayerAttrs>(op.attributes()).recurrent_ordinal,
            gdn++);
        EXPECT_EQ(op.binding(Port::ConvolutionHistory)->value(), state.history);
        EXPECT_NE(state.history, state.recurrent);
      } else {
        EXPECT_EQ(op.opcode(), OpCode::QwenAttentionLayer);
        EXPECT_EQ(std::get<QwenAttentionLayerAttrs>(op.attributes())
                      .attention_ordinal,
                  attention++);
        EXPECT_EQ(
            owner->graph().value(state.key_cache)->tensorType()->dimension(0),
            96u);
        EXPECT_NE(state.key_cache, state.value_cache);
      }
      std::vector<dispatch::shapes::Operand> inputs;
      for (const auto &b : op.bindings())
        inputs.push_back({b.port(), b.value()});
      dispatch::shapes::Result result;
      ASSERT_TRUE(dispatch::shapes::InferResult(owner->graph(), op.opcode(),
                                                op.attributes(), inputs,
                                                &result, &error))
          << error;
      EXPECT_EQ(result.alias, owner->inputs().residual);
      EXPECT_EQ(result.access, Access::ReadWrite);
      EXPECT_EQ(result.port, Port::Residual);
      for (const auto &role : QwenLayerWeightRoles(op.opcode())) {
        auto binding = op.binding(role.port);
        ASSERT_NE(binding, nullptr);
        bool found = false;
        for (const auto &a : owner->parameters())
          if (a.value == binding->value() && !a.coordinate.global &&
              a.coordinate.layer_slot == slot) {
            auto t = owner->parameter(a.coordinate);
            found |= t->role == role.role &&
                     t->tensor.name ==
                         "blk." + std::to_string(slot) + "." + role.suffix;
          }
        EXPECT_TRUE(found) << role.role;
      }
    }
    EXPECT_EQ(gdn, 18u);
    EXPECT_EQ(attention, 6u);
    std::vector<dispatch::shapes::Operand> inputs;
    for (const auto &b : ops.back().bindings())
      if (b.port() != Port::Output)
        inputs.push_back({b.port(), b.value()});
    dispatch::shapes::Result head;
    ASSERT_TRUE(dispatch::shapes::InferResult(
        owner->graph(), OpCode::QwenEpochHead, ops.back().attributes(), inputs,
        &head, &error))
        << error;
    EXPECT_EQ(head.type.dimension(0), 1u);
    EXPECT_EQ(head.type.elementType(), ElementType::I32);
  }
}
TEST(QwenStructure, OriginalIndicesAreIndependentOfLocalOrdinals) {
  auto facts = testing::ModelFacts(4);
  std::swap(facts.layers[0], facts.layers[3]);
  uint32_t gdn = 0, attention = 0;
  for (auto &l : facts.layers) {
    if (l.kind == model::LayerKind::kAttention)
      l.attention_ordinal = attention++;
    else
      l.recurrent_ordinal = gdn++;
  }
  std::shared_ptr<const QwenStructure> owner;
  std::string error;
  ASSERT_TRUE(
      BuildQwenStructure(facts, {8, 1, 1, 96, "fixture"}, &owner, &error))
      << error;
  EXPECT_EQ(owner->graph().invocations()[1].opcode(),
            OpCode::QwenAttentionLayer);
  EXPECT_EQ(owner->inputs().layers[0].original_layer_index, 3u);
  EXPECT_EQ(owner->inputs().layers[0].ordinal, 0u);
}
TEST(QwenStructure, HeadReadablePrefixAndSelectedCapacity) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(DefineQwenEpochHead(&registry, &error));
  for (uint64_t words : {386, 387, 1048}) {
    OperationGraph graph(registry);
    auto input =
        graph.tensor(TensorType::matrix(8, 1024, ElementType::F32), &error);
    auto gamma =
        graph.tensor(TensorType::vector(1024, ElementType::F32), &error);
    auto weight =
        graph.tensor(TensorType::matrix(1024, 248320, ElementType::F32)
                         .setEncoding(Encoding::Q6_K)
                         .setLayout(Layout::GgmlWeightRows),
                     &error);
    auto epoch =
        graph.tensor(TensorType::vector(words, ElementType::I32), &error);
    dispatch::shapes::Result result;
    const bool ok = dispatch::shapes::InferResult(
        graph, OpCode::QwenEpochHead,
        QwenEpochHeadAttrs{1024, 248320, 1e-6, 8, 1, 1},
        {{Port::Input, input},
         {Port::Gamma, gamma},
         {Port::Weight, weight},
         {Port::Epoch, epoch}},
        &result, &error);
    EXPECT_EQ(ok, words >= 387) << error;
    if (ok) {
      EXPECT_EQ(result.type.dimension(0), 1u);
      EXPECT_FALSE(result.alias.valid());
    }
  }
}
TEST(QwenStructure, ConvolutionLogicalAxesDescribeUnchangedRawChannels) {
  auto facts = testing::ModelFacts(4);
  std::string error;
  std::shared_ptr<const QwenStructure> owner;
  ASSERT_TRUE(
      BuildQwenStructure(facts, {8, 1, 1, 96, "fixture"}, &owner, &error))
      << error;
  auto value =
      owner->graph().invocations()[1].binding(Port::ConvolutionWeight)->value();
  auto type = owner->graph().value(value)->tensorType();
  ASSERT_NE(type, nullptr);
  EXPECT_EQ(type->dimension(0), 6144u);
  EXPECT_EQ(type->dimension(1), 4u);
  for (const auto &association : owner->parameters())
    if (association.value == value) {
      const auto &tensor = owner->parameter(association.coordinate)->tensor;
      EXPECT_EQ(tensor.normalized_extents,
                (std::array<int64_t, 4>{4, 6144, 1, 1}));
      EXPECT_EQ(tensor.byte_size, 4 * 6144 * 4u);
    }
}
TEST(QwenStructure,
     DirectContractsRejectWrongStateAndHeadShapesTransactionally) {
  auto facts = testing::ModelFacts(4);
  std::string error;
  std::shared_ptr<const QwenStructure> owner;
  ASSERT_TRUE(
      BuildQwenStructure(facts, {8, 1, 1, 96, "fixture"}, &owner, &error))
      << error;
  auto heads = [](uint64_t a, uint64_t b, uint64_t c, ElementType type) {
    auto t = TensorType::matrix(a, b, type);
    t.addDimension(c);
    return t;
  };
  struct Case {
    const char *name;
    size_t index;
    Port port;
    TensorType wrong;
  };
  std::vector<Case> cases = {
      {"swapped convolution axes", 1, Port::ConvolutionWeight,
       TensorType::matrix(4, 6144, ElementType::F32)},
      {"history extent", 1, Port::ConvolutionHistory,
       TensorType::matrix(6144, 4, ElementType::F32)},
      {"recurrent element", 1, Port::RecurrentState,
       heads(16, 128, 128, ElementType::F16)},
      {"recurrent extent", 1, Port::RecurrentState,
       heads(16, 128, 127, ElementType::F32)},
      {"KV element", 4, Port::KeyCache, heads(96, 2, 256, ElementType::F32)},
      {"KV extent", 4, Port::ValueCache, heads(97, 2, 256, ElementType::F16)},
      {"head output element", 5, Port::Output,
       TensorType::vector(1, ElementType::F32)},
      {"head output extent", 5, Port::Output,
       TensorType::vector(4, ElementType::I32)},
      {"head input element", 5, Port::Input,
       TensorType::matrix(8, 1024, ElementType::F16)}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    OperationRegistry registry;
    ASSERT_TRUE(DefineQwenGdnLayer(&registry, &error));
    ASSERT_TRUE(DefineQwenAttentionLayer(&registry, &error));
    ASSERT_TRUE(DefineQwenEpochHead(&registry, &error));
    OperationGraph graph(registry);
    const auto &source = owner->graph().invocations()[test.index];
    auto invocation = graph.operation(source.opcode());
    std::vector<dispatch::shapes::Operand> inputs;
    for (const auto &binding : source.bindings()) {
      const auto *description = owner->graph().value(binding.value());
      ASSERT_NE(description, nullptr);
      auto type =
          binding.port() == test.port ? test.wrong : *description->tensorType();
      auto value = graph.tensor(type, &error);
      ASSERT_TRUE(value.valid()) << error;
      invocation.bind(binding.port(), value);
      if (binding.port() != Port::Output)
        inputs.push_back({binding.port(), value});
    }
    std::visit([&](const auto &a) { invocation.attributes(a); },
               source.attributes());
    EXPECT_FALSE(invocation.append(&error));
    EXPECT_EQ(graph.invocations().size(), 0u);
    if (test.port != Port::Output) {
      dispatch::shapes::Result result;
      result.type = TensorType::vector(7, ElementType::I32);
      result.alias = owner->inputs().residual;
      result.access = Access::ReadWrite;
      EXPECT_FALSE(dispatch::shapes::InferResult(graph, source.opcode(),
                                                 source.attributes(), inputs,
                                                 &result, &error));
      EXPECT_EQ(result.type.dimension(0), 7u);
      EXPECT_EQ(result.alias, owner->inputs().residual);
      EXPECT_EQ(result.access, Access::ReadWrite);
      EXPECT_TRUE(graph.invocations().empty());
    }
  }
}
TEST(QwenStructure, FailedConstructionDoesNotPublish) {
  auto facts = testing::ModelFacts(4);
  std::shared_ptr<const QwenStructure> owner;
  std::string error;
  ASSERT_TRUE(
      BuildQwenStructure(facts, {8, 1, 1, 96, "fixture"}, &owner, &error))
      << error;
  auto saved = owner;
  facts.layers[0].tensors.back().tensor.byte_size++;
  EXPECT_FALSE(
      BuildQwenStructure(facts, {8, 1, 1, 96, "fixture"}, &owner, &error));
  EXPECT_EQ(owner, saved);
}
} // namespace
} // namespace hrx::qwen
