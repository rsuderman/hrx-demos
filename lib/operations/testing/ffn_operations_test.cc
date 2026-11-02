// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "lib/operations/ffn_operations.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"

namespace hrx::operations {
namespace {

TensorType Weight(uint64_t input, uint64_t output,
                  Encoding encoding = Encoding::Q8_0) {
  auto type = TensorType::matrix(input, output, ElementType::F32);
  type.setEncoding(encoding);
  type.setLayout(Layout::GgmlWeightRows);
  return type;
}

class FfnOperationsTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(DefineGatedMatmul(&registry, &error)) << error;
    ASSERT_TRUE(DefineLinearResidual(&registry, &error)) << error;
    Reset(3, 5, 7);
  }

  void Reset(uint64_t rows, uint64_t hidden_width, uint64_t ffn_width) {
    graph = std::make_unique<OperationGraph>(registry);
    input = Declare(TensorType::matrix(rows, hidden_width, ElementType::F32));
    hidden = Declare(TensorType::matrix(rows, ffn_width, ElementType::F32));
    gate = Declare(Weight(hidden_width, ffn_width));
    up = Declare(Weight(hidden_width, ffn_width));
    down = Declare(Weight(ffn_width, hidden_width));
    residual =
        Declare(TensorType::matrix(rows, hidden_width, ElementType::F32));
    gate_attrs.input_width = hidden_width;
    gate_attrs.ffn_width = ffn_width;
    gate_attrs.activation = Activation::SiLU;
    down_attrs.input_width = ffn_width;
    down_attrs.output_width = hidden_width;
  }

  ValueId Declare(const TensorType &type) {
    auto id = graph->tensor(type, &error);
    EXPECT_TRUE(id.valid()) << error;
    return id;
  }

  bool AppendGate() {
    return graph->operation(OpCode::GatedMatmul)
        .bind(Port::Input, input)
        .bind(Port::GateWeight, gate)
        .bind(Port::UpWeight, up)
        .bind(Port::Output, hidden)
        .attributes(gate_attrs)
        .append(&error);
  }

  bool AppendDown() {
    return graph->operation(OpCode::LinearResidual)
        .bind(Port::Input, hidden)
        .bind(Port::Weight, down)
        .bind(Port::Residual, residual)
        .attributes(down_attrs)
        .append(&error);
  }

  OperationRegistry registry;
  std::unique_ptr<OperationGraph> graph;
  std::string error;
  ValueId input;
  ValueId hidden;
  ValueId gate;
  ValueId up;
  ValueId down;
  ValueId residual;
  GatedMatmulAttrs gate_attrs;
  LinearResidualAttrs down_attrs;
};

TEST(FfnRegistration, PublicHelpersRegisterExactContractsAndRejectNullOwners) {
  OperationRegistry registry;
  std::string error;
  EXPECT_FALSE(DefineGatedMatmul(nullptr, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(DefineLinearResidual(nullptr, &error));
  EXPECT_FALSE(DefineGatedMatmul(&registry, nullptr));
  EXPECT_FALSE(DefineLinearResidual(&registry, nullptr));
  EXPECT_TRUE(registry.definitions().empty());
  ASSERT_TRUE(DefineGatedMatmul(&registry, &error)) << error;
  ASSERT_TRUE(DefineLinearResidual(&registry, &error)) << error;
  ASSERT_EQ(registry.definitions().size(), 2);
  const auto *gate = registry.find(OpCode::GatedMatmul);
  const auto *down = registry.find(OpCode::LinearResidual);
  ASSERT_NE(gate, nullptr);
  ASSERT_NE(down, nullptr);
  ASSERT_EQ(gate->ports().size(), 4);
  EXPECT_EQ(gate->ports()[0].port(), Port::Input);
  EXPECT_EQ(gate->ports()[1].port(), Port::GateWeight);
  EXPECT_EQ(gate->ports()[2].port(), Port::UpWeight);
  EXPECT_EQ(gate->ports()[3].port(), Port::Output);
  EXPECT_EQ(gate->port(Port::Input)->access(), Access::Read);
  EXPECT_EQ(gate->port(Port::GateWeight)->access(), Access::Read);
  EXPECT_EQ(gate->port(Port::UpWeight)->access(), Access::Read);
  EXPECT_EQ(gate->port(Port::Output)->access(), Access::Write);
  EXPECT_TRUE(gate->expectsAttributes<GatedMatmulAttrs>());
  ASSERT_EQ(down->ports().size(), 3);
  EXPECT_EQ(down->ports()[0].port(), Port::Input);
  EXPECT_EQ(down->ports()[1].port(), Port::Weight);
  EXPECT_EQ(down->ports()[2].port(), Port::Residual);
  EXPECT_EQ(down->port(Port::Residual)->kind(), ValueKind::Tensor);
  EXPECT_EQ(down->port(Port::Residual)->access(), Access::ReadWrite);
  EXPECT_EQ(down->port(Port::Output), nullptr);
  EXPECT_TRUE(down->expectsAttributes<LinearResidualAttrs>());
  EXPECT_FALSE(DefineGatedMatmul(&registry, &error));
  EXPECT_FALSE(DefineLinearResidual(&registry, &error));
  EXPECT_EQ(registry.definitions().size(), 2);
}

TEST_F(FfnOperationsTest, ConnectedCallsShareHiddenAndDeclareExactlySixValues) {
  for (uint64_t variant = 0; variant < 3; ++variant) {
    if (variant == 0)
      Reset(1, 2, 3);
    else if (variant == 1)
      Reset(8, 4096, 14336);
    else
      Reset(11, 17, 13);
    ASSERT_TRUE(AppendGate()) << error;
    ASSERT_TRUE(AppendDown()) << error;
    ASSERT_EQ(graph->values().size(), 6);
    const auto &calls = graph->invocations();
    ASSERT_EQ(calls.size(), 2);
    EXPECT_EQ(calls[0].opcode(), OpCode::GatedMatmul);
    EXPECT_EQ(calls[1].opcode(), OpCode::LinearResidual);
    EXPECT_EQ(calls[0].binding(Port::Output)->value(), hidden);
    EXPECT_EQ(calls[1].binding(Port::Input)->value(), hidden);
    EXPECT_EQ(calls[1].binding(Port::Residual)->value(), residual);
    EXPECT_EQ(
        graph->definition(calls[1].opcode())->port(Port::Residual)->access(),
        Access::ReadWrite);
    const auto &attrs = std::get<GatedMatmulAttrs>(calls[0].attributes());
    EXPECT_EQ(attrs.input_width, gate_attrs.input_width);
    EXPECT_EQ(attrs.ffn_width, gate_attrs.ffn_width);
    EXPECT_EQ(attrs.activation, Activation::SiLU);
  }
}

TEST_F(FfnOperationsTest, QuantizedWeightFormatsComposeForEveryMixture) {
  for (const auto gate_encoding :
       {Encoding::Q4_K, Encoding::Q6_K, Encoding::Q8_0}) {
    for (const auto up_encoding :
         {Encoding::Q4_K, Encoding::Q6_K, Encoding::Q8_0}) {
      for (const auto down_encoding :
           {Encoding::Q4_K, Encoding::Q6_K, Encoding::Q8_0}) {
        Reset(8, 3072, 8192);
        gate = Declare(Weight(3072, 8192, gate_encoding));
        up = Declare(Weight(3072, 8192, up_encoding));
        down = Declare(Weight(8192, 3072, down_encoding));
        ASSERT_TRUE(AppendGate()) << error;
        ASSERT_TRUE(AppendDown()) << error;
        const auto &calls = graph->invocations();
        ASSERT_EQ(calls.size(), 2);
        EXPECT_EQ(calls[0].binding(Port::GateWeight)->value(), gate);
        EXPECT_EQ(calls[0].binding(Port::UpWeight)->value(), up);
        EXPECT_EQ(calls[1].binding(Port::Weight)->value(), down);
        EXPECT_EQ(calls[0].binding(Port::Output)->value(),
                  calls[1].binding(Port::Input)->value());
        EXPECT_EQ(calls[1].binding(Port::Residual)->value(), residual);
      }
    }
  }
}

TEST_F(FfnOperationsTest, RejectsOtherWeightEncodingsAtEveryWeightPort) {
  for (const auto encoding :
       {Encoding::Dense, Encoding::Q4_0, Encoding::Q5_K, Encoding::Q8_1X4}) {
    for (int port = 0; port < 3; ++port) {
      Reset(3, 5, 7);
      auto type = Weight(port == 2 ? 7 : 5, port == 2 ? 5 : 7, encoding);
      if (encoding == Encoding::Q8_1X4)
        type.setLayout(Layout::PackedQ8_1X4Rows);
      auto rejected = Declare(type);
      if (port == 0)
        gate = rejected;
      else if (port == 1)
        up = rejected;
      else
        down = rejected;
      EXPECT_FALSE(port == 2 ? AppendDown() : AppendGate());
      EXPECT_NE(error.find("requires Q4_K, Q6_K or Q8_0"), std::string::npos);
      EXPECT_TRUE(graph->invocations().empty());
    }
  }
}

TEST_F(FfnOperationsTest, NewWeightEncodingsRetainBothAxisChecksAtEveryPort) {
  for (const auto encoding : {Encoding::Q4_K, Encoding::Q6_K}) {
    for (int port = 0; port < 3; ++port) {
      for (int axis = 0; axis < 2; ++axis) {
        Reset(3, 5, 7);
        const uint64_t input_width = port == 2 ? 7 : 5;
        const uint64_t output_width = port == 2 ? 5 : 7;
        auto wrong = Declare(Weight(input_width + (axis == 0),
                                    output_width + (axis == 1), encoding));
        if (port == 0)
          gate = wrong;
        else if (port == 1)
          up = wrong;
        else
          down = wrong;
        EXPECT_FALSE(port == 2 ? AppendDown() : AppendGate());
        EXPECT_NE(error.find("matrix shape"), std::string::npos);
        EXPECT_TRUE(graph->invocations().empty());
      }
    }
  }
}

TEST_F(FfnOperationsTest, GatedMatmulRejectsEveryWrongMatrixAxis) {
  for (int axis = 0; axis < 8; ++axis) {
    Reset(3, 5, 7);
    if (axis == 0)
      input = Declare(TensorType::matrix(3, 6, ElementType::F32));
    else if (axis == 1)
      gate = Declare(Weight(7, 5));
    else if (axis == 2)
      gate = Declare(Weight(5, 8));
    else if (axis == 3)
      up = Declare(Weight(7, 5));
    else if (axis == 4)
      up = Declare(Weight(6, 7));
    else if (axis == 5)
      hidden = Declare(TensorType::matrix(4, 7, ElementType::F32));
    else if (axis == 6)
      hidden = Declare(TensorType::matrix(3, 8, ElementType::F32));
    else
      gate_attrs.input_width = 6;
    EXPECT_FALSE(AppendGate());
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(graph->invocations().empty());
  }
}

TEST_F(FfnOperationsTest, LinearResidualRejectsWrongAxesAndPreservesFirstCall) {
  for (int axis = 0; axis < 6; ++axis) {
    Reset(3, 5, 7);
    ASSERT_TRUE(AppendGate()) << error;
    if (axis == 0)
      hidden = Declare(TensorType::matrix(3, 8, ElementType::F32));
    else if (axis == 1)
      down = Declare(Weight(5, 7));
    else if (axis == 2)
      down = Declare(Weight(7, 6));
    else if (axis == 3)
      residual = Declare(TensorType::matrix(4, 5, ElementType::F32));
    else if (axis == 4)
      residual = Declare(TensorType::matrix(3, 6, ElementType::F32));
    else
      down_attrs.input_width = 8;
    const auto values_before = graph->values().size();
    EXPECT_FALSE(AppendDown());
    EXPECT_FALSE(error.empty());
    ASSERT_EQ(graph->invocations().size(), 1);
    EXPECT_EQ(graph->invocations()[0].opcode(), OpCode::GatedMatmul);
    EXPECT_EQ(graph->values().size(), values_before);
  }
}

TEST_F(FfnOperationsTest, RejectsIncorrectDecodedTypesEncodingsAndLayouts) {
  for (int variant = 0; variant < 10; ++variant) {
    Reset(3, 5, 7);
    if (variant == 0)
      input = Declare(TensorType::matrix(3, 5, ElementType::F16));
    else if (variant == 1) {
      auto type = TensorType::matrix(3, 5, ElementType::F32);
      type.setLayout(Layout::GgmlWeightRows);
      input = Declare(type);
    } else if (variant == 2)
      gate = Declare(TensorType::matrix(5, 7, ElementType::F32));
    else if (variant == 3) {
      auto type = Weight(5, 7);
      type.setEncoding(Encoding::Q5_K);
      up = Declare(type);
    } else if (variant == 4)
      hidden = Declare(TensorType::matrix(3, 7, ElementType::I32));
    else if (variant == 5) {
      auto type = Weight(3, 7);
      hidden = Declare(type);
    } else if (variant == 6)
      down = Declare(TensorType::matrix(7, 5, ElementType::F32));
    else if (variant == 7) {
      auto type = Weight(7, 5);
      type.setEncoding(Encoding::Q4_0);
      down = Declare(type);
    } else if (variant == 8)
      residual = Declare(TensorType::matrix(3, 5, ElementType::F16));
    else {
      auto type = TensorType::matrix(3, 5, ElementType::F32);
      type.setLayout(Layout::GgmlWeightRows);
      residual = Declare(type);
    }
    if (variant < 6)
      EXPECT_FALSE(AppendGate());
    else
      EXPECT_FALSE(AppendDown());
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(graph->invocations().empty());
  }
}

TEST_F(FfnOperationsTest, RejectsZeroAttributesAndUnsupportedActivation) {
  gate_attrs.input_width = 0;
  EXPECT_FALSE(AppendGate());
  Reset(3, 5, 7);
  gate_attrs.ffn_width = 0;
  EXPECT_FALSE(AppendGate());
  Reset(3, 5, 7);
  gate_attrs.activation = static_cast<Activation>(99);
  EXPECT_FALSE(AppendGate());
  Reset(3, 5, 7);
  down_attrs.input_width = 0;
  EXPECT_FALSE(AppendDown());
  Reset(3, 5, 7);
  down_attrs.output_width = 0;
  EXPECT_FALSE(AppendDown());
  EXPECT_TRUE(graph->invocations().empty());
}

TEST_F(FfnOperationsTest, RejectsPackedActivationsAndDenseHalfWeights) {
  auto packed = TensorType::matrix(3, 5, ElementType::F32);
  packed.setEncoding(Encoding::Q8_1X4);
  packed.setLayout(Layout::PackedQ8_1X4Rows);
  input = Declare(packed);
  EXPECT_FALSE(AppendGate());
  Reset(3, 5, 7);
  gate = Declare(TensorType::matrix(5, 7, ElementType::F16));
  EXPECT_FALSE(AppendGate());
  Reset(3, 5, 7);
  hidden = Declare(TensorType::matrix(3, 7, ElementType::F16));
  EXPECT_FALSE(AppendDown());
  Reset(3, 5, 7);
  down = Declare(TensorType::matrix(7, 5, ElementType::F16));
  EXPECT_FALSE(AppendDown());
  EXPECT_TRUE(graph->invocations().empty());
}

TEST_F(FfnOperationsTest,
       ReadSharingAllowedButOutputAndResidualAliasesRejected) {
  up = gate;
  ASSERT_TRUE(AppendGate()) << error;
  EXPECT_EQ(graph->invocations()[0].binding(Port::GateWeight)->value(),
            graph->invocations()[0].binding(Port::UpWeight)->value());
  Reset(3, 5, 5);
  hidden = input; // Same shape/format, so alias checking is the rejection.
  EXPECT_FALSE(AppendGate());
  EXPECT_NE(error.find("differ"), std::string::npos);
  EXPECT_TRUE(graph->invocations().empty());
  Reset(3, 5, 5);
  residual = hidden;
  EXPECT_FALSE(AppendDown());
  EXPECT_NE(error.find("distinct"), std::string::npos);
  EXPECT_TRUE(graph->invocations().empty());
}

} // namespace
} // namespace hrx::operations
