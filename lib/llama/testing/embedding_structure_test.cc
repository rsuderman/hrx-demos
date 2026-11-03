// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/detail/embedding_structure.h"
#include "lib/operations/tensor_operations.h"
#include "gtest/gtest.h"

namespace hrx::llama::detail {
namespace {
using namespace operations;
model::ModelDescriptor Facts(int64_t hidden = 256, uint32_t storage = 14,
                             bool tied = true) {
  model::ModelDescriptor f;
  f.family = "llama";
  f.embedding_width = hidden;
  f.vocabulary_size = 1024;
  f.key_head_width = f.value_head_width = 64;
  f.rms_epsilon = 1e-5;
  model::TensorBinding weight;
  weight.role = "token_embedding";
  weight.tensor.name = "embedding.weight";
  weight.tensor.storage_type = storage;
  weight.tensor.storage_name = "ignored";
  weight.tensor.normalized_extents = {hidden, 1024, 1, 1};
  weight.tensor.absolute_offset = 4096;
  weight.tensor.byte_size = 256;
  auto output = weight;
  output.role = "output";
  if (!tied) {
    output.tensor.name = "output.weight";
    output.tensor.absolute_offset += 256;
  }
  // Embedding deliberately is not global slot zero.
  f.globals = {output, weight};
  model::LayerDescriptor layer;
  layer.query_heads = hidden / 64;
  layer.kv_heads = 2;
  layer.feed_forward_width = hidden * 2;
  f.layers.push_back(layer);
  return f;
}
std::unique_ptr<StructureConstruction> Make(model::ModelDescriptor facts,
                                            bool define = true) {
  OperationRegistry registry;
  std::string error;
  if (define) {
    EXPECT_TRUE(DefineEmbedding(&registry, &error)) << error;
  }
  EXPECT_TRUE(DefineRmsNorm(&registry, &error)) << error;
  auto c = StructureConstruction::Create(
      std::move(facts), {Phase::Prefill, 4, 16, 32, "checkpoint"}, registry,
      &error);
  EXPECT_TRUE(c) << error;
  return c;
}
bool Outside(const LlamaRegionView &view, ValueId value) {
  for (auto p : view.outside)
    if (p.value == value)
      return true;
  return false;
}
TEST(EmbeddingStructure, PreservesOwnerIdentityRolesSpansAndTiedWeights) {
  for (int64_t hidden : {256, 512})
    for (uint32_t storage : {8, 12, 14})
      for (bool tied : {false, true}) {
        auto c = Make(Facts(hidden, storage, tied));
        ASSERT_TRUE(c);
        std::string error;
        ASSERT_TRUE(c->createInputs(&error)) << error;
        const auto inputs = c->structure().inputs();
        ASSERT_TRUE(AppendEmbedding(c.get(), &error)) << error;
        ASSERT_EQ(c->graph().invocations().size(), 1u);
        const auto &op = c->graph().invocations()[0];
        EXPECT_EQ(op.opcode(), OpCode::Embedding);
        const auto &attrs = std::get<EmbeddingAttrs>(op.attributes());
        EXPECT_EQ(attrs.embedding_width, uint64_t(hidden));
        EXPECT_EQ(attrs.vocabulary_size, 1024u);
        EXPECT_EQ(op.binding(Port::Output)->value(), inputs.residual);
        EXPECT_EQ(op.binding(Port::TokenIds)->value(), inputs.tokens);
        EXPECT_EQ(op.binding(Port::RowBegin)->value(), inputs.row_begin);
        EXPECT_EQ(op.binding(Port::RowCount)->value(), inputs.row_count);
        const auto *definition = c->graph().definition(OpCode::Embedding);
        EXPECT_EQ(definition->port(Port::TokenIds)->access(), Access::Read);
        EXPECT_EQ(definition->port(Port::Output)->access(), Access::Write);
        EXPECT_EQ(definition->port(Port::Output)->domain(),
                  AccessDomain::ActiveRows);
        EXPECT_EQ(definition->port(Port::TokenIds)->domain(),
                  AccessDomain::ActiveRows);
        EXPECT_EQ(definition->port(Port::Weight)->domain(),
                  AccessDomain::Whole);
        const auto &associations = c->structure().parameters();
        ASSERT_EQ(associations.size(), 1u);
        EXPECT_EQ(associations[0].coordinate.tensor_slot, 1u);
        EXPECT_EQ(associations[0].value, op.binding(Port::Weight)->value());
        ValueId output;
        ASSERT_TRUE(
            c->resolve({ParameterScope::Global, 0, 0}, &output, &error));
        EXPECT_EQ(output == op.binding(Port::Weight)->value(), tied);
        LlamaRegionView view;
        ASSERT_TRUE(c->structure().select({LlamaRegionKind::Embedding, 0, 0},
                                          &view, &error))
            << error;
        EXPECT_EQ(view.operation_begin, 0u);
        EXPECT_EQ(view.operation_end, 1u);
        EXPECT_TRUE(Outside(view, inputs.tokens));
        EXPECT_TRUE(Outside(view, op.binding(Port::Weight)->value()));
        EXPECT_FALSE(
            Outside(view, inputs.residual)); // No implicit global export.
        ASSERT_TRUE(c->exportValue(inputs.residual, &error));
        ASSERT_TRUE(c->structure().select({LlamaRegionKind::Embedding, 0, 0},
                                          &view, &error));
        EXPECT_TRUE(Outside(view, inputs.residual));
        EXPECT_FALSE(AppendEmbedding(c.get(), &error));
        EXPECT_EQ(c->graph().invocations().size(), 1u);
      }
}
TEST(EmbeddingStructure, RejectsInvalidRoleAxesStorageAndSourceIdentity) {
  for (int invalid = 0; invalid < 10; ++invalid) {
    auto facts = Facts();
    auto &weight = facts.globals[1];
    if (invalid == 0)
      weight.role = "unrelated";
    if (invalid == 1)
      facts.globals.push_back(weight);
    if (invalid == 2)
      ++weight.tensor.normalized_extents[0];
    if (invalid == 3)
      ++weight.tensor.normalized_extents[1];
    if (invalid == 4)
      ++weight.tensor.normalized_extents[2];
    if (invalid == 5)
      ++weight.tensor.normalized_extents[3];
    if (invalid == 6)
      weight.tensor.storage_type = 0;
    if (invalid == 7)
      weight.tensor.storage_type = 13;
    if (invalid == 8)
      weight.tensor.name.clear();
    if (invalid == 9)
      weight.tensor.storage_type = 2;
    auto c = Make(std::move(facts));
    ASSERT_TRUE(c);
    std::string error;
    ASSERT_TRUE(c->createInputs(&error));
    EXPECT_FALSE(AppendEmbedding(c.get(), &error)) << invalid;
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(c->graph().invocations().empty());
    EXPECT_TRUE(c->structure().parameters().empty());
    LlamaRegionView view;
    view.operation_end = 99;
    EXPECT_FALSE(c->structure().select({LlamaRegionKind::Embedding, 0, 0},
                                       &view, &error));
    EXPECT_EQ(view.operation_end, 99u);
  }
}
TEST(EmbeddingStructure, RequiresOwnerInputsAndDefinition) {
  std::string error;
  EXPECT_FALSE(AppendEmbedding(nullptr, &error));
  auto c = Make(Facts());
  ASSERT_TRUE(c);
  EXPECT_FALSE(AppendEmbedding(c.get(), &error));
  EXPECT_TRUE(c->graph().invocations().empty());
  c = Make(Facts(), false);
  ASSERT_TRUE(c);
  ASSERT_TRUE(c->createInputs(&error));
  EXPECT_FALSE(AppendEmbedding(c.get(), &error));
  EXPECT_TRUE(c->structure().parameters().empty());
}
TEST(EmbeddingStructure,
     MatchingSpanConsumerKeepsResidualInternalInCombinedCut) {
  auto c = Make(Facts());
  ASSERT_TRUE(c);
  std::string error;
  ASSERT_TRUE(c->createInputs(&error));
  ASSERT_TRUE(AppendEmbedding(c.get(), &error));
  auto &g = c->graph();
  const auto inputs = c->structure().inputs();
  auto gamma = g.tensor(TensorType::vector(256, ElementType::F32), &error);
  auto output = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
  ASSERT_TRUE(g.operation(OpCode::RmsNorm)
                  .bind(Port::Input, inputs.residual)
                  .bind(Port::Gamma, gamma)
                  .bind(Port::Output, output)
                  .bind(Port::RowBegin, inputs.row_begin)
                  .bind(Port::RowCount, inputs.row_count)
                  .attributes(RmsNormAttrs{256, 1e-5})
                  .append(&error));
  ASSERT_TRUE(c->record({LlamaRegionKind::Output, 0, 0}, 0, 2, &error));
  ASSERT_TRUE(c->exportValue(output, &error));
  LlamaRegionView view;
  ASSERT_TRUE(
      c->structure().select({LlamaRegionKind::Output, 0, 0}, &view, &error));
  EXPECT_FALSE(Outside(view, inputs.residual));
  EXPECT_TRUE(Outside(view, output));
  ASSERT_TRUE(
      c->structure().select({LlamaRegionKind::Embedding, 0, 0}, &view, &error));
  EXPECT_TRUE(
      Outside(view, inputs.residual)); // Exported through downstream use.
}
// The owner-taking API cannot receive foreign or caller-shaped token values.
// Verify the reused operation boundary, without adding a malformed-input seam.
TEST(EmbeddingStructure, ExistingVerifierRejectsMalformedAndForeignTokens) {
  for (int invalid = 0; invalid < 5; ++invalid) {
    auto c = Make(Facts());
    auto foreign = Make(Facts());
    ASSERT_TRUE(c);
    ASSERT_TRUE(foreign);
    std::string error;
    ASSERT_TRUE(c->createInputs(&error));
    ASSERT_TRUE(foreign->createInputs(&error));
    auto &g = c->graph();
    auto type = TensorType::vector(4, ElementType::I32);
    if (invalid == 0)
      type = TensorType::vector(4, ElementType::F32);
    if (invalid == 1)
      type = TensorType::matrix(4, 1, ElementType::I32);
    if (invalid == 2)
      type.setLayout(Layout::GgmlWeightRows);
    auto tokens = invalid == 3 ? foreign->structure().inputs().tokens
                               : g.tensor(type, &error);
    ValueId weight;
    ASSERT_TRUE(c->resolve({ParameterScope::Global, 0, 1}, &weight, &error));
    const auto &input = c->structure().inputs();
    auto op = g.operation(OpCode::Embedding);
    op.bind(Port::TokenIds, tokens)
        .bind(Port::Weight, weight)
        .bind(Port::Output, input.residual)
        .bind(Port::RowBegin, input.row_begin)
        .attributes(EmbeddingAttrs{1024, 256});
    if (invalid != 4)
      op.bind(Port::RowCount, input.row_count);
    EXPECT_FALSE(op.append(&error)) << invalid;
    EXPECT_TRUE(g.invocations().empty());
  }
}
TEST(EmbeddingStructure,
     ConflictingPreviouslyResolvedIdentityDoesNotPublishRegion) {
  auto facts = Facts();
  ++facts.globals[0].tensor.absolute_offset;
  auto c = Make(std::move(facts));
  ASSERT_TRUE(c);
  std::string error;
  ASSERT_TRUE(c->createInputs(&error));
  ValueId output;
  ASSERT_TRUE(c->resolve({ParameterScope::Global, 0, 0}, &output, &error));
  EXPECT_FALSE(AppendEmbedding(c.get(), &error));
  EXPECT_TRUE(c->graph().invocations().empty());
  EXPECT_EQ(c->structure().parameters().size(), 1u);
}
} // namespace
} // namespace hrx::llama::detail
