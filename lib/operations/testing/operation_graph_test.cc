// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "lib/operations/operation_graph.h"

#include <optional>
#include <type_traits>
#include <utility>

#include "gtest/gtest.h"

namespace hrx::operations {
namespace {
// Definition mechanics do not need a bound invocation in milestone A.
bool VerifyGate(const BoundOperands &, const GatedMatmulAttrs &,
                std::string *) {
  return true;
}
bool VerifyDown(const BoundOperands &, const LinearResidualAttrs &,
                std::string *) {
  return true;
}
bool DefineGate(OperationRegistry &registry, std::string *error) {
  return registry.define(OpCode::GatedMatmul)
      .port(Port::Input, ValueKind::Tensor, Access::Read)
      .port(Port::GateWeight, ValueKind::Tensor, Access::Read)
      .port(Port::UpWeight, ValueKind::Tensor, Access::Read)
      .port(Port::Output, ValueKind::Tensor, Access::Write)
      .attributes<GatedMatmulAttrs>()
      .verify(VerifyGate)
      .finish(error);
}

TEST(OperationRegistry, FluentDefinitionOwnsOrderedSchemaAndAttributeType) {
  OperationRegistry registry;
  std::string error = "stale";
  ASSERT_TRUE(DefineGate(registry, &error)) << error;
  EXPECT_TRUE(error.empty());
  ASSERT_EQ(registry.definitions().size(), 1);
  const auto *definition = registry.find(OpCode::GatedMatmul);
  ASSERT_NE(definition, nullptr);
  EXPECT_EQ(definition->opcode(), OpCode::GatedMatmul);
  ASSERT_EQ(definition->ports().size(), 4);
  EXPECT_EQ(definition->ports()[0].port(), Port::Input);
  EXPECT_EQ(definition->ports()[1].port(), Port::GateWeight);
  EXPECT_EQ(definition->ports()[2].port(), Port::UpWeight);
  EXPECT_EQ(definition->ports()[3].port(), Port::Output);
  EXPECT_EQ(definition->port(Port::Input)->kind(), ValueKind::Tensor);
  EXPECT_EQ(definition->port(Port::Input)->access(), Access::Read);
  EXPECT_EQ(definition->port(Port::Output)->access(), Access::Write);
  EXPECT_TRUE(definition->expectsAttributes<GatedMatmulAttrs>());
  EXPECT_FALSE(definition->expectsAttributes<LinearResidualAttrs>());
  EXPECT_EQ(definition->port(Port::Residual), nullptr);
  EXPECT_EQ(definition->port(static_cast<Port>(99)), nullptr);
  EXPECT_EQ(registry.find(OpCode::RmsNorm), nullptr);
  EXPECT_EQ(registry.find(static_cast<OpCode>(99)), nullptr);
  ASSERT_TRUE(registry.define(OpCode::LinearResidual)
                  .verify(VerifyDown)
                  .attributes<LinearResidualAttrs>()
                  .port(Port::Residual, ValueKind::Tensor, Access::ReadWrite)
                  .finish(&error))
      << error;
  EXPECT_EQ(registry.definitions()[1].opcode(), OpCode::LinearResidual);
  EXPECT_EQ(
      registry.find(OpCode::LinearResidual)->port(Port::Residual)->access(),
      Access::ReadWrite);
}

TEST(OperationRegistry, DuplicateAndInvalidDefinitionsPreserveExistingRecords) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(DefineGate(registry, &error));
  EXPECT_FALSE(DefineGate(registry, &error));
  EXPECT_NE(error.find("already registered"), std::string::npos);
  EXPECT_FALSE(registry.define(static_cast<OpCode>(-1))
                   .attributes<GatedMatmulAttrs>()
                   .verify(VerifyGate)
                   .finish(&error));
  EXPECT_EQ(error, "invalid operation code");
  EXPECT_FALSE(registry.define(OpCode::RmsNorm)
                   .port(Port::Input, ValueKind::Tensor, Access::Read)
                   .port(Port::Input, ValueKind::Tensor, Access::Write)
                   .attributes<GatedMatmulAttrs>()
                   .verify(VerifyGate)
                   .finish(&error));
  EXPECT_EQ(error, "port is already defined");
  EXPECT_FALSE(registry.define(OpCode::RmsNorm)
                   .port(static_cast<Port>(99), ValueKind::Tensor, Access::Read)
                   .attributes<GatedMatmulAttrs>()
                   .verify(VerifyGate)
                   .finish(&error));
  EXPECT_EQ(error, "invalid port");
  EXPECT_FALSE(registry.define(OpCode::RmsNorm)
                   .port(Port::Input, static_cast<ValueKind>(99), Access::Read)
                   .attributes<GatedMatmulAttrs>()
                   .verify(VerifyGate)
                   .finish(&error));
  EXPECT_EQ(error, "invalid value kind");
  EXPECT_FALSE(
      registry.define(OpCode::RmsNorm)
          .port(Port::Input, ValueKind::Tensor, static_cast<Access>(99))
          .attributes<GatedMatmulAttrs>()
          .verify(VerifyGate)
          .finish(&error));
  EXPECT_EQ(error, "invalid port access");
  ASSERT_EQ(registry.definitions().size(), 1);
  EXPECT_EQ(registry.definitions()[0].ports().size(), 4);
  EXPECT_TRUE(registry.definitions()[0].expectsAttributes<GatedMatmulAttrs>());
}

TEST(OperationRegistry,
     MissingRepeatedAndMismatchedTypedAttributesOrVerifiersFail) {
  OperationRegistry registry;
  std::string error;
  EXPECT_FALSE(
      registry.define(OpCode::GatedMatmul).verify(VerifyGate).finish(&error));
  EXPECT_EQ(error, "attributes must be declared");
  EXPECT_FALSE(registry.define(OpCode::GatedMatmul)
                   .attributes<GatedMatmulAttrs>()
                   .finish(&error));
  EXPECT_EQ(error, "verifier must be declared");
  EXPECT_FALSE(registry.define(OpCode::GatedMatmul)
                   .attributes<GatedMatmulAttrs>()
                   .attributes<GatedMatmulAttrs>()
                   .verify(VerifyGate)
                   .finish(&error));
  EXPECT_EQ(error, "attributes are already declared");
  EXPECT_FALSE(registry.define(OpCode::GatedMatmul)
                   .attributes<GatedMatmulAttrs>()
                   .verify(VerifyGate)
                   .verify(VerifyGate)
                   .finish(&error));
  EXPECT_EQ(error, "verifier is already declared");
  EXPECT_FALSE(registry.define(OpCode::GatedMatmul)
                   .attributes<GatedMatmulAttrs>()
                   .verify(VerifyDown)
                   .finish(&error));
  EXPECT_NE(error.find("attribute type does not match"), std::string::npos);
  detail::Verifier<GatedMatmulAttrs> null_verifier = nullptr;
  EXPECT_FALSE(registry.define(OpCode::GatedMatmul)
                   .attributes<GatedMatmulAttrs>()
                   .verify(null_verifier)
                   .finish(&error));
  EXPECT_EQ(error, "verifier must not be null");
  EXPECT_TRUE(registry.definitions().empty());
}

TEST(OperationRegistry, DiscardedMovedConsumedAndFirstErrorBuildersAreAtomic) {
  static_assert(!std::is_copy_constructible_v<DefinitionBuilder>);
  static_assert(std::is_move_constructible_v<DefinitionBuilder>);
  OperationRegistry registry;
  std::string error;
  {
    auto discarded = registry.define(OpCode::GatedMatmul);
    discarded.attributes<GatedMatmulAttrs>().verify(VerifyGate);
  }
  EXPECT_TRUE(registry.definitions().empty());
  auto original = registry.define(OpCode::GatedMatmul);
  original.attributes<GatedMatmulAttrs>().verify(VerifyGate);
  auto moved = std::move(original);
  EXPECT_FALSE(original.finish(&error));
  EXPECT_EQ(error, "definition builder is consumed");
  ASSERT_TRUE(moved.finish(&error));
  EXPECT_FALSE(moved.finish(&error));
  EXPECT_EQ(error, "definition builder is consumed");
  auto bad = registry.define(OpCode::RmsNorm);
  bad.port(static_cast<Port>(-1), ValueKind::Tensor, Access::Read)
      .attributes<GatedMatmulAttrs>()
      .attributes<LinearResidualAttrs>()
      .verify(VerifyDown);
  EXPECT_FALSE(bad.finish(&error));
  EXPECT_EQ(error, "invalid port");
  EXPECT_FALSE(bad.finish(&error));
  EXPECT_EQ(error, "definition builder is consumed");
  auto null_error = registry.define(OpCode::LinearResidual);
  null_error.attributes<LinearResidualAttrs>().verify(VerifyDown);
  EXPECT_FALSE(null_error.finish(nullptr));
  EXPECT_FALSE(null_error.finish(&error));
  EXPECT_EQ(registry.definitions().size(), 1);
}

TEST(OperationRegistry, InterleavedBuildersCheckDuplicatesAgainAtCommit) {
  OperationRegistry registry;
  auto first = registry.define(OpCode::GatedMatmul);
  auto second = registry.define(OpCode::GatedMatmul);
  first.attributes<GatedMatmulAttrs>().verify(VerifyGate);
  second.attributes<GatedMatmulAttrs>().verify(VerifyGate);
  std::string error;
  ASSERT_TRUE(first.finish(&error));
  EXPECT_FALSE(second.finish(&error));
  EXPECT_EQ(error, "operation code is already registered");
  auto abandoned = registry.define(OpCode::RmsNorm);
  auto replacement = registry.define(OpCode::LinearResidual);
  replacement.attributes<LinearResidualAttrs>().verify(VerifyDown);
  abandoned = std::move(replacement);
  ASSERT_TRUE(abandoned.finish(&error));
  EXPECT_EQ(registry.find(OpCode::RmsNorm), nullptr);
  EXPECT_NE(registry.find(OpCode::LinearResidual), nullptr);
  EXPECT_FALSE(replacement.finish(&error));
  EXPECT_EQ(registry.definitions().size(), 2);
}

TEST(OperationGraph,
     SnapshotSurvivesRegistryLifetimeAndIgnoresLaterRegistration) {
  static_assert(!std::is_copy_constructible_v<OperationRegistry>);
  static_assert(!std::is_move_constructible_v<OperationRegistry>);
  static_assert(!std::is_copy_constructible_v<OperationGraph>);
  static_assert(!std::is_move_constructible_v<OperationGraph>);
  std::unique_ptr<OperationGraph> graph;
  std::string error;
  {
    OperationRegistry registry;
    ASSERT_TRUE(DefineGate(registry, &error));
    graph = std::make_unique<OperationGraph>(registry);
    ASSERT_TRUE(registry.define(OpCode::LinearResidual)
                    .port(Port::Residual, ValueKind::Tensor, Access::ReadWrite)
                    .attributes<LinearResidualAttrs>()
                    .verify(VerifyDown)
                    .finish(&error));
    EXPECT_EQ(graph->definition(OpCode::LinearResidual), nullptr);
  }
  const auto *definition = graph->definition(OpCode::GatedMatmul);
  ASSERT_NE(definition, nullptr);
  EXPECT_EQ(definition->ports().size(), 4);
  EXPECT_EQ(definition->port(Port::Output)->access(), Access::Write);
  EXPECT_TRUE(definition->expectsAttributes<GatedMatmulAttrs>());
  EXPECT_EQ(graph->definition(static_cast<OpCode>(99)), nullptr);
}

TEST(OperationGraph, CopiesTypesAndExposesAllThreeValueKinds) {
  OperationRegistry registry;
  OperationGraph graph(registry);
  std::string error = "stale";
  auto type = TensorType::matrix(8, 16, ElementType::F32);
  auto tensor = graph.tensor(type, &error);
  ASSERT_TRUE(tensor.valid());
  EXPECT_TRUE(error.empty());
  type.setEncoding(Encoding::Q8_0).setLayout(Layout::GgmlWeightRows);
  auto scalar = graph.scalar(ScalarType::I64, &error);
  auto state = graph.state(StateKind::KVCache, &error);
  ASSERT_TRUE(scalar.valid());
  ASSERT_TRUE(state.valid());
  ASSERT_EQ(graph.values().size(), 3);
  EXPECT_EQ(graph.values()[0].id(), tensor);
  const auto *value = graph.value(tensor);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->kind(), ValueKind::Tensor);
  ASSERT_NE(value->tensorType(), nullptr);
  EXPECT_EQ(value->tensorType()->rank(), 2);
  EXPECT_EQ(value->tensorType()->dimension(0), 8);
  EXPECT_EQ(value->tensorType()->dimension(1), 16);
  EXPECT_EQ(value->tensorType()->dimension(2), 0);
  EXPECT_EQ(value->tensorType()->encoding(), Encoding::Dense);
  EXPECT_EQ(value->tensorType()->layout(), Layout::RowMajor);
  EXPECT_EQ(value->scalarType(), nullptr);
  EXPECT_EQ(value->stateKind(), nullptr);
  EXPECT_EQ(graph.value(scalar)->kind(), ValueKind::Scalar);
  EXPECT_EQ(*graph.value(scalar)->scalarType(), ScalarType::I64);
  EXPECT_EQ(graph.value(scalar)->tensorType(), nullptr);
  EXPECT_EQ(graph.value(scalar)->stateKind(), nullptr);
  EXPECT_EQ(graph.value(state)->kind(), ValueKind::State);
  EXPECT_EQ(*graph.value(state)->stateKind(), StateKind::KVCache);
  EXPECT_EQ(graph.value(state)->tensorType(), nullptr);
  EXPECT_EQ(graph.value(state)->scalarType(), nullptr);
}

TEST(OperationGraph,
     HandlesSurviveGrowthAndRejectForeignAndReusedOwnerStorage) {
  OperationRegistry registry;
  OperationGraph graph(registry);
  OperationGraph other(registry);
  const auto type = TensorType::matrix(4, 32, ElementType::F32);
  std::string error;
  auto first = graph.tensor(type, &error);
  auto copied = first;
  auto foreign = other.tensor(type, &error);
  ValueId invalid;
  EXPECT_FALSE(invalid.valid());
  EXPECT_EQ(graph.value(invalid), nullptr);
  EXPECT_EQ(graph.value(foreign), nullptr);
  EXPECT_NE(first, foreign);
  for (int i = 0; i < 2048; ++i)
    ASSERT_TRUE(graph.scalar(ScalarType::I32, &error).valid());
  EXPECT_EQ(first, copied);
  ASSERT_NE(graph.value(copied), nullptr);
  EXPECT_EQ(graph.value(copied)->tensorType()->dimension(1), 32);
  std::optional<OperationGraph> reused;
  reused.emplace(registry);
  auto old = reused->tensor(type, &error);
  const auto *same_address = &*reused;
  reused.reset();
  EXPECT_TRUE(old.valid()); // Identity survives; graph data does not.
  reused.emplace(registry);
  EXPECT_EQ(&*reused, same_address);
  auto fresh = reused->tensor(type, &error);
  EXPECT_NE(old, fresh);
  EXPECT_EQ(reused->value(old), nullptr);
  EXPECT_NE(reused->value(fresh), nullptr);
}

TEST(OperationGraph, InvalidDeclarationsLeaveValuesUnchanged) {
  OperationRegistry registry;
  OperationGraph graph(registry);
  std::string error;
  TensorType absent;
  EXPECT_EQ(absent.rank(), 0);
  EXPECT_EQ(absent.dimension(0), 0);
  EXPECT_FALSE(graph.tensor(absent, &error).valid());
  EXPECT_NE(error.find("positive dimensions"), std::string::npos);
  EXPECT_FALSE(graph.tensor(TensorType::matrix(0, 16, ElementType::F32), &error)
                   .valid());
  EXPECT_FALSE(
      graph.tensor(TensorType::matrix(4, 0, ElementType::F32), &error).valid());
  auto type = TensorType::matrix(4, 16, ElementType::F32);
  EXPECT_FALSE(graph.tensor(type, nullptr).valid());
  EXPECT_FALSE(graph.scalar(ScalarType::F32, nullptr).valid());
  EXPECT_FALSE(graph.state(StateKind::KVCache, nullptr).valid());
  EXPECT_FALSE(
      graph
          .tensor(TensorType::matrix(4, 16, static_cast<ElementType>(99)),
                  &error)
          .valid());
  type.setEncoding(static_cast<Encoding>(99));
  EXPECT_FALSE(graph.tensor(type, &error).valid());
  type.setEncoding(Encoding::Dense).setLayout(static_cast<Layout>(99));
  EXPECT_FALSE(graph.tensor(type, &error).valid());
  EXPECT_FALSE(graph.scalar(static_cast<ScalarType>(99), &error).valid());
  EXPECT_FALSE(graph.state(static_cast<StateKind>(99), &error).valid());
  EXPECT_TRUE(graph.values().empty());
  auto valid = graph.scalar(ScalarType::F32, &error);
  ASSERT_TRUE(valid.valid());
  EXPECT_TRUE(error.empty());
  ASSERT_EQ(graph.values().size(), 1);
  EXPECT_EQ(graph.values()[0].id(), valid);
}

TEST(OperationGraph, RepresentationCompatibilityIsExplicitWithoutConversions) {
  OperationRegistry registry;
  OperationGraph graph(registry);
  std::string error;
  for (int tag = int(Encoding::Q4_0); tag <= int(Encoding::Q8_0); ++tag) {
    auto type = TensorType::matrix(128, 64, ElementType::F32);
    type.setEncoding(static_cast<Encoding>(tag));
    const auto before = graph.values().size();
    EXPECT_FALSE(graph.tensor(type, &error).valid());
    EXPECT_EQ(graph.values().size(), before);
    type.setLayout(Layout::GgmlWeightRows);
    ASSERT_TRUE(graph.tensor(type, &error).valid()) << error;
    auto wrong_element = TensorType::matrix(128, 64, ElementType::F16);
    wrong_element.setEncoding(static_cast<Encoding>(tag))
        .setLayout(Layout::GgmlWeightRows);
    EXPECT_FALSE(graph.tensor(wrong_element, &error).valid());
  }
  auto packed = TensorType::matrix(4, 128, ElementType::F32);
  packed.setEncoding(Encoding::Q8_1X4);
  EXPECT_FALSE(graph.tensor(packed, &error).valid());
  packed.setLayout(Layout::PackedQ8_1X4Rows);
  ASSERT_TRUE(graph.tensor(packed, &error).valid()) << error;
  packed.setEncoding(Encoding::Dense);
  EXPECT_FALSE(graph.tensor(packed, &error).valid());
  for (int tag = int(ElementType::F32); tag <= int(ElementType::I64); ++tag) {
    auto dense = TensorType::matrix(2, 3, static_cast<ElementType>(tag));
    ASSERT_TRUE(graph.tensor(dense, &error).valid()) << error;
    dense.setLayout(Layout::GgmlWeightRows);
    ASSERT_TRUE(graph.tensor(dense, &error).valid()) << error;
  }
}

TEST(OperationEnums, DisplayCoversEveryDeclaredIdentityAndInvalidValues) {
  for (int i = int(OpCode::Embedding); i <= int(OpCode::PublishToken); ++i)
    EXPECT_STRNE(OpCodeName(static_cast<OpCode>(i)), "invalid_opcode");
  for (int i = int(Port::Input); i <= int(Port::Output); ++i)
    EXPECT_STRNE(PortName(static_cast<Port>(i)), "invalid_port");
  for (int i = int(Access::Read); i <= int(Access::ReadWrite); ++i)
    EXPECT_STRNE(AccessName(static_cast<Access>(i)), "invalid_access");
  EXPECT_STREQ(OpCodeName(OpCode::GatedMatmul), "gated_matmul");
  EXPECT_STREQ(PortName(Port::GateWeight), "gate_weight");
  EXPECT_STREQ(AccessName(Access::ReadWrite), "read_write");
  EXPECT_STREQ(OpCodeName(static_cast<OpCode>(-1)), "invalid_opcode");
  EXPECT_STREQ(PortName(static_cast<Port>(99)), "invalid_port");
  EXPECT_STREQ(AccessName(static_cast<Access>(99)), "invalid_access");
}

int verifier_calls = 0;
bool VerifyRows(const BoundOperands &operands, const GatedMatmulAttrs &attrs,
                std::string *error) {
  ++verifier_calls;
  const auto *input = operands.tensor(Port::Input);
  const auto *output = operands.tensor(Port::Output);
  if (!input || !output || input->dimension(0) != output->dimension(0) ||
      input->dimension(1) != attrs.input_width ||
      output->dimension(1) != attrs.ffn_width) {
    *error = "test verifier: rows or widths disagree";
    return false;
  }
  return true;
}
bool VerifyUpdate(const BoundOperands &operands,
                  const LinearResidualAttrs &attrs, std::string *error) {
  const auto *input = operands.tensor(Port::Input);
  const auto *residual = operands.tensor(Port::Residual);
  if (!input || !residual || input->dimension(0) != residual->dimension(0) ||
      input->dimension(1) != attrs.input_width ||
      residual->dimension(1) != attrs.output_width) {
    *error = "test update verifier: widths disagree";
    return false;
  }
  return true;
}
bool DefineRows(OperationRegistry &registry, std::string *error) {
  return registry.define(OpCode::GatedMatmul)
      .port(Port::Input, ValueKind::Tensor, Access::Read)
      .port(Port::Output, ValueKind::Tensor, Access::Write)
      .attributes<GatedMatmulAttrs>()
      .verify(VerifyRows)
      .finish(error);
}

TEST(OperationInvocations, ConnectedCallsCopyAttrsAndNormalizeNamedBindings) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(DefineRows(registry, &error));
  ASSERT_TRUE(registry.define(OpCode::LinearResidual)
                  .port(Port::Input, ValueKind::Tensor, Access::Read)
                  .port(Port::Residual, ValueKind::Tensor, Access::ReadWrite)
                  .attributes<LinearResidualAttrs>()
                  .verify(VerifyUpdate)
                  .finish(&error));
  OperationGraph graph(registry);
  auto input = graph.tensor(TensorType::matrix(2, 8, ElementType::F32), &error);
  auto hidden =
      graph.tensor(TensorType::matrix(2, 16, ElementType::F32), &error);
  auto residual =
      graph.tensor(TensorType::matrix(2, 8, ElementType::F32), &error);
  GatedMatmulAttrs gate_attrs;
  gate_attrs.input_width = 8;
  gate_attrs.ffn_width = 16;
  auto gate = graph.operation(OpCode::GatedMatmul);
  gate.bind(Port::Output, hidden)
      .attributes(gate_attrs)
      .bind(Port::Input, input);
  gate_attrs.ffn_width = 999;
  ASSERT_TRUE(gate.append(&error)) << error;
  LinearResidualAttrs down_attrs;
  down_attrs.input_width = 16;
  down_attrs.output_width = 8;
  error = "stale";
  ASSERT_TRUE(graph.operation(OpCode::LinearResidual)
                  .bind(Port::Residual, residual)
                  .bind(Port::Input, hidden)
                  .attributes(down_attrs)
                  .append(&error))
      << error;
  EXPECT_TRUE(error.empty());
  down_attrs.input_width = 999;
  ASSERT_EQ(graph.invocations().size(), 2);
  EXPECT_EQ(graph.values().size(), 3);
  const auto &first = graph.invocations()[0];
  const auto &second = graph.invocations()[1];
  EXPECT_EQ(first.opcode(), OpCode::GatedMatmul);
  EXPECT_EQ(second.opcode(), OpCode::LinearResidual);
  ASSERT_EQ(first.bindings().size(), 2);
  EXPECT_EQ(first.bindings()[0].port(), Port::Input);
  EXPECT_EQ(first.bindings()[1].port(), Port::Output);
  EXPECT_EQ(first.binding(Port::Output)->value(),
            second.binding(Port::Input)->value());
  EXPECT_EQ(second.bindings()[0].port(), Port::Input);
  EXPECT_EQ(second.bindings()[1].port(), Port::Residual);
  EXPECT_EQ(second.binding(Port::Residual)->value(), residual);
  EXPECT_EQ(first.binding(Port::Weight), nullptr);
  EXPECT_EQ(first.binding(static_cast<Port>(99)), nullptr);
  EXPECT_EQ(std::get<GatedMatmulAttrs>(first.attributes()).ffn_width, 16);
  EXPECT_EQ(std::get<LinearResidualAttrs>(second.attributes()).input_width, 16);
  EXPECT_EQ(graph.definition(second.opcode())->port(Port::Residual)->access(),
            Access::ReadWrite);
}

TEST(OperationInvocations,
     CommonFailuresPrecedeVerifierAndPreserveEarlierRecords) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(DefineRows(registry, &error));
  OperationGraph graph(registry);
  OperationGraph other(registry);
  auto type = TensorType::matrix(2, 8, ElementType::F32);
  auto input = graph.tensor(type, &error);
  auto output = graph.tensor(type, &error);
  auto scalar = graph.scalar(ScalarType::I32, &error);
  auto foreign = other.tensor(type, &error);
  ValueId invalid;
  GatedMatmulAttrs attrs;
  attrs.input_width = 8;
  attrs.ffn_width = 8;
  LinearResidualAttrs wrong;
  ASSERT_TRUE(graph.operation(OpCode::GatedMatmul)
                  .bind(Port::Input, input)
                  .bind(Port::Output, output)
                  .attributes(attrs)
                  .append(&error));
  const auto calls_before = verifier_calls;
  EXPECT_FALSE(
      graph.operation(OpCode::RmsNorm).attributes(attrs).append(&error));
  EXPECT_NE(error.find("not registered"), std::string::npos);
  EXPECT_FALSE(graph.operation(static_cast<OpCode>(99))
                   .attributes(attrs)
                   .append(&error));
  EXPECT_EQ(error, "invalid operation code");
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, input)
                   .bind(Port::Output, output)
                   .append(&error));
  EXPECT_EQ(error, "attributes must be bound");
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, input)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_NE(error.find("missing operation port"), std::string::npos);
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, input)
                   .bind(Port::Input, input)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_EQ(error, "port is already bound");
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Weight, input)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_EQ(error, "port is not declared by operation");
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(static_cast<Port>(99), input)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_EQ(error, "invalid port");
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, scalar)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_NE(error.find("value kind"), std::string::npos);
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, invalid)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_NE(error.find("invalid or foreign"), std::string::npos);
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, foreign)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_NE(error.find("invalid or foreign"), std::string::npos);
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, input)
                   .bind(Port::Output, output)
                   .attributes(wrong)
                   .append(&error));
  EXPECT_EQ(error, "attributes do not match operation definition");
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, input)
                   .bind(Port::Output, output)
                   .attributes(attrs)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_EQ(error, "attributes are already bound");
  EXPECT_EQ(verifier_calls, calls_before);
  attrs.ffn_width = 99;
  EXPECT_FALSE(graph.operation(OpCode::GatedMatmul)
                   .bind(Port::Input, input)
                   .bind(Port::Output, output)
                   .attributes(attrs)
                   .append(&error));
  EXPECT_EQ(error, "test verifier: rows or widths disagree");
  EXPECT_EQ(verifier_calls, calls_before + 1);
  ASSERT_EQ(graph.invocations().size(), 1);
  EXPECT_EQ(graph.values().size(), 3);
  EXPECT_EQ(graph.invocations()[0].binding(Port::Output)->value(), output);
  EXPECT_EQ(
      std::get<GatedMatmulAttrs>(graph.invocations()[0].attributes()).ffn_width,
      8);
}

TEST(OperationInvocations, StagedMovedConsumedAndDiscardedBuildersAreAtomic) {
  static_assert(!std::is_copy_constructible_v<InvocationBuilder>);
  static_assert(std::is_move_constructible_v<InvocationBuilder>);
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(DefineRows(registry, &error));
  OperationGraph graph(registry);
  auto input = graph.tensor(TensorType::matrix(2, 8, ElementType::F32), &error);
  auto output =
      graph.tensor(TensorType::matrix(2, 8, ElementType::F32), &error);
  GatedMatmulAttrs attrs;
  attrs.input_width = 8;
  attrs.ffn_width = 8;
  {
    auto discarded = graph.operation(OpCode::GatedMatmul);
    discarded.bind(Port::Input, input)
        .bind(Port::Output, output)
        .attributes(attrs);
  }
  EXPECT_TRUE(graph.invocations().empty());
  auto original = graph.operation(OpCode::GatedMatmul);
  original.bind(Port::Input, input)
      .bind(Port::Output, output)
      .attributes(attrs);
  auto moved = std::move(original);
  EXPECT_FALSE(original.append(&error));
  EXPECT_EQ(error, "invocation builder is consumed");
  ASSERT_TRUE(moved.append(&error));
  EXPECT_FALSE(moved.append(&error));
  EXPECT_EQ(error, "invocation builder is consumed");
  auto incomplete = graph.operation(OpCode::GatedMatmul);
  auto complete = graph.operation(OpCode::GatedMatmul);
  complete.bind(Port::Input, input)
      .bind(Port::Output, output)
      .attributes(attrs);
  incomplete = std::move(complete);
  ASSERT_TRUE(incomplete.append(&error));
  EXPECT_FALSE(complete.append(&error));
  auto bad = graph.operation(OpCode::GatedMatmul);
  bad.bind(Port::Weight, input).attributes(attrs).attributes(attrs);
  EXPECT_FALSE(bad.append(&error));
  EXPECT_EQ(error, "port is not declared by operation");
  EXPECT_FALSE(bad.append(&error));
  EXPECT_EQ(error, "invocation builder is consumed");
  auto null_error = graph.operation(OpCode::GatedMatmul);
  null_error.bind(Port::Input, input)
      .bind(Port::Output, output)
      .attributes(attrs);
  const auto calls_before = verifier_calls;
  EXPECT_FALSE(null_error.append(nullptr));
  EXPECT_FALSE(null_error.append(&error));
  EXPECT_EQ(verifier_calls, calls_before);
  EXPECT_EQ(graph.invocations().size(), 2);
  EXPECT_EQ(graph.values().size(), 2);
}

bool VerifyKindsAndAliases(const BoundOperands &operands,
                           const GatedMatmulAttrs &, std::string *error) {
  const auto *scalar = operands.value(Port::Weight);
  const auto *state = operands.value(Port::Residual);
  if (!operands.tensor(Port::Input) || !scalar || !scalar->scalarType() ||
      *scalar->scalarType() != ScalarType::I64 || !state ||
      !state->stateKind() || *state->stateKind() != StateKind::KVCache ||
      operands.tensor(Port::Weight) || operands.tensor(Port::Residual) ||
      operands.value(Port::GateWeight) || operands.tensor(Port::GateWeight) ||
      !operands.sameValue(Port::Input, Port::Output) ||
      operands.sameValue(Port::GateWeight, Port::UpWeight) ||
      operands.sameValue(Port::Input, Port::GateWeight)) {
    *error = "test bound operand views disagree";
    return false;
  }
  return true;
}
TEST(OperationInvocations, BoundOperandsExposeKindsAndLogicalAliasIdentity) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(registry.define(OpCode::CopyRows)
                  .port(Port::Input, ValueKind::Tensor, Access::Read)
                  .port(Port::Weight, ValueKind::Scalar, Access::Read)
                  .port(Port::Residual, ValueKind::State, Access::ReadWrite)
                  .port(Port::Output, ValueKind::Tensor, Access::Write)
                  .attributes<GatedMatmulAttrs>()
                  .verify(VerifyKindsAndAliases)
                  .finish(&error));
  OperationGraph graph(registry);
  auto tensor =
      graph.tensor(TensorType::matrix(2, 8, ElementType::F32), &error);
  auto scalar = graph.scalar(ScalarType::I64, &error);
  auto state = graph.state(StateKind::KVCache, &error);
  GatedMatmulAttrs attrs;
  ASSERT_TRUE(graph.operation(OpCode::CopyRows)
                  .bind(Port::Output, tensor)
                  .bind(Port::Residual, state)
                  .bind(Port::Weight, scalar)
                  .bind(Port::Input, tensor)
                  .attributes(attrs)
                  .append(&error))
      << error;
  EXPECT_EQ(graph.invocations().size(), 1);
  EXPECT_EQ(graph.values().size(), 3);
}

bool RejectWithoutDiagnostic(const BoundOperands &, const GatedMatmulAttrs &,
                             std::string *) {
  return false;
}
TEST(OperationInvocations, EmptyVerifierDiagnosticStillExplainsFailure) {
  OperationRegistry registry;
  std::string error;
  ASSERT_TRUE(registry.define(OpCode::PublishToken)
                  .attributes<GatedMatmulAttrs>()
                  .verify(RejectWithoutDiagnostic)
                  .finish(&error));
  OperationGraph graph(registry);
  GatedMatmulAttrs attrs;
  EXPECT_FALSE(
      graph.operation(OpCode::PublishToken).attributes(attrs).append(&error));
  EXPECT_EQ(error, "operation verifier rejected invocation");
  EXPECT_TRUE(graph.invocations().empty());
}

TEST(OperationInvocations, SnapshotVerifierAndHandlesRemainUsableAfterGrowth) {
  std::optional<OperationGraph> graph;
  std::string error;
  {
    OperationRegistry registry;
    ASSERT_TRUE(DefineRows(registry, &error));
    graph.emplace(registry);
  }
  auto input =
      graph->tensor(TensorType::matrix(2, 8, ElementType::F32), &error);
  auto output =
      graph->tensor(TensorType::matrix(2, 8, ElementType::F32), &error);
  auto pending = graph->operation(OpCode::GatedMatmul);
  GatedMatmulAttrs attrs;
  attrs.input_width = 8;
  attrs.ffn_width = 8;
  pending.bind(Port::Input, input).bind(Port::Output, output).attributes(attrs);
  for (int i = 0; i < 1024; ++i)
    ASSERT_TRUE(graph->scalar(ScalarType::I32, &error).valid());
  ASSERT_TRUE(pending.append(&error)) << error;
  std::optional<Invocation> copy;
  copy.emplace(graph->invocations()[0]);
  graph.reset();
  EXPECT_EQ(copy->binding(Port::Input)->value(), input);
  EXPECT_EQ(std::get<GatedMatmulAttrs>(copy->attributes()).ffn_width, 8);
}
} // namespace
} // namespace hrx::operations
