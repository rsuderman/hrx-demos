// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "lib/llama/detail/ffn_structure.h"

#include <limits>
#include <memory>
#include <utility>

#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"
#include "gtest/gtest.h"

namespace hrx::llama::detail {
namespace {
using namespace operations;

model::TensorBinding Tensor(const char *role, const char *name,
                            uint32_t storage, int64_t input,
                            int64_t output = 1) {
  model::TensorBinding binding;
  binding.role = role;
  binding.tensor.name = name;
  binding.tensor.normalized_extents = {input, output, 1, 1};
  binding.tensor.storage_type = storage;
  binding.tensor.storage_name = "diagnostic only";
  binding.tensor.absolute_offset = 4096;
  binding.tensor.byte_size = storage == 0
                                 ? input * output * 4
                                 : input * output / (storage == 8 ? 32 : 256) *
                                       (storage == 8    ? 34
                                        : storage == 12 ? 144
                                                        : 210);
  return binding;
}

model::ModelDescriptor Facts(uint64_t width = 3072, uint64_t ffn_width = 8192) {
  model::ModelDescriptor facts;
  facts.family = "llama";
  facts.embedding_width = width;
  facts.rms_epsilon = 1e-5;
  model::LayerDescriptor layer;
  layer.index =
      7; // Vector slot selection must not confuse this with slot zero.
  layer.feed_forward_width = ffn_width;
  layer.tensors = {
      Tensor("post_norm", "blk.7.ffn_norm.weight", 0, width),
      Tensor("ffn_gate", "blk.7.ffn_gate.weight", 12, width, ffn_width),
      Tensor("ffn_up", "blk.7.ffn_up.weight", 12, width, ffn_width),
      Tensor("ffn_down", "blk.7.ffn_down.weight", 14, ffn_width, width)};
  facts.layers.push_back(std::move(layer));
  return facts;
}

bool Definitions(OperationRegistry *registry, std::string *error) {
  return DefineRmsNorm(registry, error) && DefineGatedMatmul(registry, error) &&
         DefineLinearResidual(registry, error);
}

// Test-only scoped fixture, not a partial model factory or another graph IR.
// The eventual full model owner retains facts/graph and calls the same helper.
class FfnFixture {
public:
  static bool Build(model::ModelDescriptor facts, uint64_t rows,
                    std::shared_ptr<const FfnFixture> *out,
                    std::string *error) {
    OperationRegistry registry;
    if (!Definitions(&registry, error))
      return false;
    auto staged =
        std::shared_ptr<FfnFixture>(new FfnFixture(std::move(facts), registry));
    auto residual = staged->graph_.tensor(
        TensorType::matrix(rows, staged->facts_.embedding_width,
                           ElementType::F32),
        error);
    if (!residual.valid() ||
        !AppendNormalizedFfn(&staged->graph_, staged->facts_, 0, residual,
                             &staged->region_, error))
      return false;
    *out = std::move(staged);
    return true;
  }
  const model::ModelDescriptor &facts() const { return facts_; }
  const OperationGraph &graph() const { return graph_; }
  const NormalizedFfnRegion &region() const { return region_; }

private:
  FfnFixture(model::ModelDescriptor facts, const OperationRegistry &registry)
      : facts_(std::move(facts)), graph_(registry) {}
  model::ModelDescriptor facts_;
  OperationGraph graph_;
  NormalizedFfnRegion region_;
};

TEST(FfnStructure, RealProfileTopologyRolesAndEncodedStorageRemainDistinct) {
  std::shared_ptr<const FfnFixture> fixture;
  std::string error;
  ASSERT_TRUE(FfnFixture::Build(Facts(), 1, &fixture, &error)) << error;
  const auto &graph = fixture->graph();
  const auto &region = fixture->region();
  ASSERT_EQ(graph.invocations().size(), 3);
  EXPECT_EQ(graph.values().size(), 7);
  EXPECT_EQ(region.operation_begin, 0);
  EXPECT_EQ(region.operation_end, 3);
  const auto &norm = graph.invocations()[0];
  const auto &gate = graph.invocations()[1];
  const auto &down = graph.invocations()[2];
  EXPECT_EQ(norm.opcode(), OpCode::RmsNorm);
  EXPECT_EQ(gate.opcode(), OpCode::GatedMatmul);
  EXPECT_EQ(down.opcode(), OpCode::LinearResidual);
  EXPECT_EQ(norm.binding(Port::Input)->value(), region.residual);
  EXPECT_EQ(norm.binding(Port::Output)->value(), region.normalized);
  EXPECT_EQ(gate.binding(Port::Input)->value(), region.normalized);
  EXPECT_EQ(gate.binding(Port::Output)->value(), region.activation);
  EXPECT_EQ(down.binding(Port::Input)->value(), region.activation);
  EXPECT_EQ(down.binding(Port::Residual)->value(), region.residual);
  EXPECT_EQ(graph.definition(down.opcode())->port(Port::Residual)->access(),
            Access::ReadWrite);
  EXPECT_EQ(std::get<GatedMatmulAttrs>(gate.attributes()).activation,
            Activation::SiLU);
  EXPECT_DOUBLE_EQ(std::get<RmsNormAttrs>(norm.attributes()).epsilon, 1e-5);
  ASSERT_EQ(region.parameters.size(), 4);
  for (size_t i = 0; i < region.parameters.size(); ++i) {
    const auto &association = region.parameters[i];
    EXPECT_EQ(association.layer_slot, 0);
    EXPECT_EQ(association.tensor_slot, i);
    const auto &tensor = fixture->facts()
                             .layers[association.layer_slot]
                             .tensors[association.tensor_slot]
                             .tensor;
    const auto *type = graph.value(association.value)->tensorType();
    ASSERT_NE(type, nullptr);
    EXPECT_EQ(type->elementType(), ElementType::F32);
    EXPECT_EQ(tensor.absolute_offset, 4096);
    if (i == 0) {
      EXPECT_EQ(type->rank(), 1);
      EXPECT_EQ(tensor.name, "blk.7.ffn_norm.weight");
      EXPECT_EQ(fixture->facts().layers[0].tensors[i].role, "post_norm");
    } else {
      EXPECT_EQ(type->rank(), 2);
      EXPECT_EQ(type->layout(), Layout::GgmlWeightRows);
      EXPECT_EQ(type->encoding(), i == 3 ? Encoding::Q6_K : Encoding::Q4_K);
      EXPECT_EQ(type->dimension(0), i == 3 ? 8192 : 3072);
      EXPECT_EQ(type->dimension(1), i == 3 ? 3072 : 8192);
      EXPECT_NE(tensor.byte_size, type->dimension(0) * type->dimension(1) * 4);
    }
  }
}

TEST(FfnStructure,
     DifferentGeometryAndMultipleRowsNeedNoNativeProfileAdmission) {
  for (uint64_t rows : {1, 8}) {
    auto facts = Facts(4096, 14336);
    for (size_t i = 1; i < 4; ++i) {
      facts.layers[0].tensors[i].tensor.storage_type = 8;
      facts.layers[0].tensors[i].tensor.storage_name =
          "Q4_K is a misleading label";
    }
    std::shared_ptr<const FfnFixture> fixture;
    std::string error;
    ASSERT_TRUE(FfnFixture::Build(facts, rows, &fixture, &error)) << error;
    facts.layers
        .clear(); // Fixture retains owned facts, not pointers into caller.
    const auto &region = fixture->region();
    const auto &graph = fixture->graph();
    EXPECT_EQ(graph.value(region.activation)->tensorType()->dimension(0), rows);
    EXPECT_EQ(graph.value(region.activation)->tensorType()->dimension(1),
              14336);
    EXPECT_EQ(graph.value(region.normalized)->tensorType()->dimension(1), 4096);
    EXPECT_EQ(graph.value(region.parameters[1].value)->tensorType()->encoding(),
              Encoding::Q8_0);
    EXPECT_EQ(fixture->facts().layers[0].tensors[1].tensor.storage_type, 8);
  }
}

TEST(FfnStructure, HelperComposesRegionsWithStableRangesAndOneMutableResidual) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(Definitions(&registry, &error));
  OperationGraph graph(registry);
  auto facts = Facts();
  facts.layers.push_back(facts.layers[0]);
  facts.layers[1].index = 8;
  for (auto &binding : facts.layers[1].tensors) {
    binding.tensor.name = "layer8/" + binding.role;
    binding.tensor.absolute_offset += 100000000;
  }
  auto residual =
      graph.tensor(TensorType::matrix(4, 3072, ElementType::F32), &error);
  NormalizedFfnRegion first, second;
  ASSERT_TRUE(AppendNormalizedFfn(&graph, facts, 0, residual, &first, &error))
      << error;
  ASSERT_TRUE(AppendNormalizedFfn(&graph, facts, 1, residual, &second, &error))
      << error;
  EXPECT_EQ(first.operation_begin, 0);
  EXPECT_EQ(first.operation_end, 3);
  EXPECT_EQ(second.operation_begin, 3);
  EXPECT_EQ(second.operation_end, 6);
  EXPECT_EQ(first.residual, second.residual);
  EXPECT_NE(first.normalized, second.normalized);
  EXPECT_NE(first.parameters[1].value, second.parameters[1].value);
  EXPECT_EQ(second.parameters[1].layer_slot, 1);
  EXPECT_EQ(graph.invocations()[3].binding(Port::Input)->value(), residual);
}

TEST(FfnStructure,
     MalformedDescriptorFailsBeforeGraphMutationOrOutputPublication) {
  for (int variant = 0; variant < 17; ++variant) {
    SCOPED_TRACE(variant);
    OperationRegistry registry;
    std::string error;
    ASSERT_TRUE(Definitions(&registry, &error));
    OperationGraph graph(registry);
    auto facts = Facts();
    auto residual =
        graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
    auto &bindings = facts.layers[0].tensors;
    if (variant == 0)
      facts.family = "qwen";
    if (variant == 1)
      facts.layers.clear();
    if (variant == 2)
      facts.embedding_width = 0;
    if (variant == 3)
      facts.layers[0].feed_forward_width = 0;
    if (variant == 4)
      facts.rms_epsilon = std::numeric_limits<double>::infinity();
    if (variant == 5)
      facts.rms_epsilon = -1;
    if (variant == 6)
      bindings.erase(bindings.begin());
    if (variant == 7)
      bindings.push_back(bindings[1]);
    if (variant == 8)
      bindings[0].tensor.storage_type = 1;
    if (variant == 9)
      bindings[1].tensor.storage_type = 13;
    if (variant == 10)
      bindings[2].tensor.storage_type = 0;
    if (variant == 11)
      bindings[3].tensor.storage_type = 2;
    if (variant == 12)
      bindings[1].tensor.normalized_extents = {8192, 3072, 1, 1};
    if (variant == 13)
      bindings[3].tensor.normalized_extents[2] = 2;
    if (variant == 14)
      bindings[0].tensor.name.clear();
    if (variant == 15)
      facts.embedding_width = std::numeric_limits<uint64_t>::max();
    if (variant == 16)
      facts.layers[0].kind = model::LayerKind::kGatedDeltaNet;
    NormalizedFfnRegion result;
    result.operation_begin = 123;
    EXPECT_FALSE(
        AppendNormalizedFfn(&graph, facts, 0, residual, &result, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(result.operation_begin, 123);
    EXPECT_TRUE(graph.invocations().empty());
    EXPECT_EQ(graph.values().size(), 1);
  }
}

TEST(FfnStructure, ForeignResidualMissingDefinitionsAndNullOutputsFailClearly) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(Definitions(&registry, &error));
  OperationGraph graph(registry), foreign(registry);
  const auto facts = Facts();
  auto residual =
      graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  auto other =
      foreign.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  NormalizedFfnRegion region;
  EXPECT_FALSE(AppendNormalizedFfn(&graph, facts, 0, other, &region, &error));
  EXPECT_FALSE(AppendNormalizedFfn(&graph, facts, 0, {}, &region, &error));
  EXPECT_FALSE(
      AppendNormalizedFfn(&graph, facts, 7, residual, &region, &error));
  EXPECT_FALSE(
      AppendNormalizedFfn(nullptr, facts, 0, residual, &region, &error));
  EXPECT_FALSE(
      AppendNormalizedFfn(&graph, facts, 0, residual, nullptr, &error));
  EXPECT_FALSE(
      AppendNormalizedFfn(&graph, facts, 0, residual, &region, nullptr));
  auto wrong =
      graph.tensor(TensorType::matrix(1, 3072, ElementType::F16), &error);
  EXPECT_FALSE(AppendNormalizedFfn(&graph, facts, 0, wrong, &region, &error));
  OperationRegistry empty;
  OperationGraph missing(empty);
  auto input =
      missing.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  EXPECT_FALSE(AppendNormalizedFfn(&missing, facts, 0, input, &region, &error));
  EXPECT_NE(error.find("missing definition"), std::string::npos);
  EXPECT_TRUE(graph.invocations().empty());
  EXPECT_TRUE(missing.invocations().empty());
}

TEST(FfnStructure, ScopedFixtureNeverPublishesFailedConstruction) {
  std::shared_ptr<const FfnFixture> fixture;
  std::string error;
  ASSERT_TRUE(FfnFixture::Build(Facts(), 1, &fixture, &error));
  const auto original = fixture;
  auto bad = Facts();
  bad.layers[0].tensors[2].role = "missing_up";
  EXPECT_FALSE(FfnFixture::Build(bad, 1, &fixture, &error));
  EXPECT_EQ(fixture, original);
  EXPECT_FALSE(FfnFixture::Build(Facts(), 0, &fixture, &error));
  EXPECT_EQ(fixture, original);
}

} // namespace
} // namespace hrx::llama::detail
