// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/structure.h"
#include "lib/llama/testing/model_fixture.h"
#include "gtest/gtest.h"
using namespace hrx::llama;
using namespace hrx::operations;
TEST(CompleteModel, RepeatedLayersRegionsAndTiedIdentity) {
  for (size_t count : {1, 2, 28})
    for (bool smaller : {false, true}) {
      auto facts = hrx::llama::testing::ModelFacts(count, smaller ? 1024 : 3072,
                                                   smaller ? 2048 : 8192,
                                                   smaller ? 64 : 128);
      std::shared_ptr<const LlamaStructure> owner;
      std::string error;
      ASSERT_TRUE(BuildLlamaStructure(
          facts, {Phase::Prefill, 8, 512, 128, "fixture"}, &owner, &error))
          << error;
      ASSERT_EQ(owner->graph().invocations().size(), 1 + 11 * count + 2);
      EXPECT_EQ(owner->inputs().layers.size(), count);
      EXPECT_NE(owner->inputs().tokens, owner->inputs().history);
      size_t heads = 0, norms = 0;
      for (const auto &op : owner->graph().invocations()) {
        heads += op.opcode() == OpCode::OutputHead;
        norms += op.opcode() == OpCode::RmsNorm;
      }
      EXPECT_EQ(heads, 1);
      EXPECT_EQ(norms, 2 * count);
      auto embedding =
          owner->graph().invocations().front().binding(Port::Weight)->value();
      auto head = owner->graph()
                      .invocations()[1 + 11 * count]
                      .binding(Port::Weight)
                      ->value();
      EXPECT_EQ(embedding, head);
      LlamaRegionView view;
      ASSERT_TRUE(
          owner->select({LlamaRegionKind::Layers, 0, count}, &view, &error))
          << error;
      EXPECT_EQ(view.operation_begin, 1);
      EXPECT_EQ(view.operation_end, 1 + 11 * count);
      for (size_t i = 0; i < count; ++i) {
        ASSERT_TRUE(
            owner->select({LlamaRegionKind::Attention, i, 1}, &view, &error))
            << error;
        EXPECT_EQ(view.operation_end - view.operation_begin, 8);
        ASSERT_TRUE(owner->select({LlamaRegionKind::Ffn, i, 1}, &view, &error))
            << error;
        EXPECT_EQ(view.operation_end - view.operation_begin, 3);
      }
      ASSERT_TRUE(owner->select({LlamaRegionKind::Output, 0, 0}, &view, &error))
          << error;
      EXPECT_EQ(view.operation_end - view.operation_begin, 2);
    }
}
TEST(CompleteModel, FailureLeavesOwnerUnchanged) {
  auto facts = hrx::llama::testing::ModelFacts();
  std::shared_ptr<const LlamaStructure> owner;
  std::string error;
  ASSERT_TRUE(BuildLlamaStructure(
      facts, {Phase::Decode, 1, 512, 128, "fixture"}, &owner, &error))
      << error;
  auto saved = owner;
  facts.layers[1].tensors.pop_back();
  EXPECT_FALSE(BuildLlamaStructure(
      facts, {Phase::Decode, 1, 512, 128, "fixture"}, &owner, &error));
  EXPECT_EQ(owner, saved);
}
