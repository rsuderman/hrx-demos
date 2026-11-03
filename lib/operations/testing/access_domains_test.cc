// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/ffn_operations.h"
#include "lib/operations/row_operations.h"
#include "lib/operations/tensor_operations.h"
#include "lib/operations/testing/operation_test_util.h"

namespace hrx::operations {
namespace {
using namespace test;

class AccessDomainsTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(DefineEmbedding(&registry, &error));
    ASSERT_TRUE(DefineRmsNorm(&registry, &error));
    ASSERT_TRUE(DefineLinear(&registry, &error));
    ASSERT_TRUE(DefineGatedMatmul(&registry, &error));
    ASSERT_TRUE(DefineLinearResidual(&registry, &error));
    ASSERT_TRUE(DefineQuantize(&registry, &error));
    ASSERT_TRUE(DefineCopyRows(&registry, &error));
  }
  InvocationBuilder Call(OperationGraph &g, OpCode code) {
    auto call = g.operation(code);
    if (code == OpCode::Embedding) {
      call.bind(Port::TokenIds, V(g, 8, ElementType::I32))
          .bind(Port::Weight, M(g, 128, 256, ElementType::F32, Encoding::Q8_0,
                                Layout::GgmlWeightRows))
          .bind(Port::Output, M(g, 8, 128))
          .attributes(EmbeddingAttrs{256, 128});
    } else if (code == OpCode::RmsNorm) {
      call.bind(Port::Input, M(g, 8, 128))
          .bind(Port::Gamma, V(g, 128))
          .bind(Port::Output, M(g, 8, 128))
          .attributes(RmsNormAttrs{128});
    } else if (code == OpCode::GatedMatmul) {
      call.bind(Port::Input, M(g, 8, 128))
          .bind(Port::GateWeight, M(g, 128, 256, ElementType::F32,
                                    Encoding::Q4_K, Layout::GgmlWeightRows))
          .bind(Port::UpWeight, M(g, 128, 256, ElementType::F32, Encoding::Q6_K,
                                  Layout::GgmlWeightRows))
          .bind(Port::Output, M(g, 8, 256))
          .attributes(GatedMatmulAttrs{128, 256});
    } else {
      call.bind(Port::Input, M(g, 8, 128))
          .bind(Port::Weight, M(g, 128, 256, ElementType::F32, Encoding::Q8_0,
                                Layout::GgmlWeightRows));
      if (code == OpCode::Linear)
        call.bind(Port::Output, M(g, 8, 256)).attributes(LinearAttrs{128, 256});
      else
        call.bind(Port::Residual, M(g, 8, 256))
            .attributes(LinearResidualAttrs{128, 256});
    }
    return call;
  }
  OperationRegistry registry;
  std::string error;
};

TEST_F(AccessDomainsTest, OptionalSpanAndLegacyCallsRetainOperandDomains) {
  OperationGraph g(registry);
  for (auto code : {OpCode::Embedding, OpCode::RmsNorm, OpCode::Linear,
                    OpCode::GatedMatmul, OpCode::LinearResidual}) {
    SCOPED_TRACE(OpCodeName(code));
    ASSERT_TRUE(Call(g, code).append(&error)) << error;
    EXPECT_EQ(g.invocations().back().binding(Port::RowBegin), nullptr);
    EXPECT_EQ(g.invocations().back().binding(Port::RowCount), nullptr);
    auto begin = S(g), count = S(g);
    ASSERT_TRUE(Call(g, code)
                    .bind(Port::RowBegin, begin)
                    .bind(Port::RowCount, count)
                    .append(&error))
        << error;
    EXPECT_EQ(g.invocations().back().binding(Port::RowBegin)->value(), begin);
    EXPECT_EQ(g.invocations().back().binding(Port::RowCount)->value(), count);
    const auto *d = g.definition(code);
    EXPECT_FALSE(d->port(Port::RowBegin)->required());
    EXPECT_FALSE(d->port(Port::RowCount)->required());
    for (const auto &p : d->ports()) {
      auto expected = p.port() == Port::Input || p.port() == Port::Output ||
                              p.port() == Port::Residual ||
                              p.port() == Port::TokenIds
                          ? AccessDomain::ActiveRows
                          : AccessDomain::Whole;
      EXPECT_EQ(p.domain(), expected) << PortName(p.port());
    }
  }
  auto residual = g.definition(OpCode::LinearResidual)->port(Port::Residual);
  EXPECT_EQ(residual->access(), Access::ReadWrite);
  EXPECT_EQ(residual->domain(), AccessDomain::ActiveRows);
  EXPECT_EQ(g.definition(OpCode::RmsNorm)->port(Port::Output)->access(),
            Access::Write);
}

TEST_F(AccessDomainsTest,
       RejectsUnpairedMistypedForeignAndWrongKindControlsAtomically) {
  OperationGraph g(registry), other(registry);
  for (auto code : {OpCode::Embedding, OpCode::RmsNorm, OpCode::Linear,
                    OpCode::GatedMatmul, OpCode::LinearResidual}) {
    SCOPED_TRACE(OpCodeName(code));
    for (auto port : {Port::RowBegin, Port::RowCount}) {
      EXPECT_FALSE(Call(g, code).bind(port, S(g)).append(&error));
      EXPECT_EQ(error, "row controls must be both present or absent");
      for (auto type : {ScalarType::I32, ScalarType::F32}) {
        auto call = Call(g, code);
        call.bind(Port::RowBegin,
                  S(g, port == Port::RowBegin ? type : ScalarType::I64))
            .bind(Port::RowCount,
                  S(g, port == Port::RowCount ? type : ScalarType::I64));
        EXPECT_FALSE(call.append(&error));
        EXPECT_EQ(error, "row controls must be I64 scalars");
      }
      EXPECT_FALSE(Call(g, code).bind(port, S(other)).append(&error));
      EXPECT_NE(error.find("invalid or foreign"), std::string::npos);
      EXPECT_FALSE(Call(g, code).bind(port, V(g, 1)).append(&error));
      EXPECT_NE(error.find("value kind"), std::string::npos);
    }
  }
  EXPECT_TRUE(g.invocations().empty());
}

TEST_F(AccessDomainsTest,
       RequiredQuantizeAndCopyRowsExposeExistingSelectedDomains) {
  OperationGraph g(registry);
  const auto *quantize = g.definition(OpCode::Quantize);
  EXPECT_EQ(quantize->port(Port::Input)->domain(), AccessDomain::ActiveRows);
  EXPECT_EQ(quantize->port(Port::Output)->domain(), AccessDomain::ActiveRows);
  EXPECT_TRUE(quantize->port(Port::RowBegin)->required());
  const auto *copy = g.definition(OpCode::CopyRows);
  EXPECT_EQ(copy->port(Port::Input)->domain(), AccessDomain::CopyRowsSource);
  EXPECT_EQ(copy->port(Port::Output)->domain(),
            AccessDomain::CopyRowsDestination);
  EXPECT_EQ(copy->port(Port::Output)->access(), Access::Write);
  for (auto selection : {RowSelection::Range, RowSelection::LastActive}) {
    ASSERT_TRUE(g.operation(OpCode::CopyRows)
                    .bind(Port::Input, M(g, 8, 128))
                    .bind(Port::Output,
                          M(g, selection == RowSelection::Range ? 8 : 1, 128))
                    .bind(Port::RowBegin, S(g))
                    .bind(Port::RowCount, S(g))
                    .attributes(CopyRowsAttrs{128, selection})
                    .append(&error))
        << error;
    EXPECT_EQ(
        std::get<CopyRowsAttrs>(g.invocations().back().attributes()).selection,
        selection);
  }
}

bool Accept(const BoundOperands &, const RmsNormAttrs &, std::string *) {
  return true;
}
TEST(AccessDomainDefinitions, RejectInvalidDomainAndNonTensorPort) {
  OperationRegistry r;
  std::string error;
  for (auto kind : {ValueKind::Tensor, ValueKind::Scalar, ValueKind::State}) {
    EXPECT_FALSE(r.define(OpCode::RmsNorm)
                     .port(Port::Input, kind, Access::Read,
                           kind == ValueKind::Tensor
                               ? static_cast<AccessDomain>(99)
                               : AccessDomain::ActiveRows)
                     .attributes<RmsNormAttrs>()
                     .verify(Accept)
                     .finish(&error));
  }
  EXPECT_TRUE(r.definitions().empty());
}
TEST(AccessDomainDefinitions, RejectMissingMismatchedAndWritableControls) {
  OperationRegistry r;
  std::string error;
  for (int variant = 0; variant < 4; ++variant) {
    auto d = r.define(OpCode::RmsNorm);
    d.port(Port::Input, ValueKind::Tensor, Access::Read,
           AccessDomain::ActiveRows)
        .optionalPort(Port::RowBegin, ValueKind::Scalar, Access::Read);
    if (variant == 1)
      d.port(Port::RowCount, ValueKind::Scalar, Access::Read);
    if (variant == 2)
      d.optionalPort(Port::RowCount, ValueKind::Tensor, Access::Read);
    if (variant == 3)
      d.optionalPort(Port::RowCount, ValueKind::Scalar, Access::Write);
    EXPECT_FALSE(d.attributes<RmsNormAttrs>().verify(Accept).finish(&error));
    EXPECT_EQ(error, "row access domain requires paired scalar read controls");
  }
  EXPECT_TRUE(r.definitions().empty());
}
TEST(AccessDomainDefinitions,
     CopyDomainRequiresCopyOperationAttributesAndMatchingPort) {
  OperationRegistry r;
  std::string error;
  for (auto code : {OpCode::RmsNorm, OpCode::CopyRows}) {
    EXPECT_FALSE(r.define(code)
                     .port(Port::Weight, ValueKind::Tensor, Access::Read,
                           AccessDomain::CopyRowsSource)
                     .port(Port::RowBegin, ValueKind::Scalar, Access::Read)
                     .port(Port::RowCount, ValueKind::Scalar, Access::Read)
                     .attributes<RmsNormAttrs>()
                     .verify(Accept)
                     .finish(&error));
    EXPECT_EQ(
        error,
        "copy_rows domain requires matching operation, attributes and port");
  }
  EXPECT_TRUE(r.definitions().empty());
}
bool AcceptCopy(const BoundOperands &, const CopyRowsAttrs &, std::string *) {
  return true;
}
TEST(AccessDomainDefinitions,
     CopyDestinationRejectsInventedReadsOfOverwrittenRows) {
  OperationRegistry r;
  std::string error;
  for (auto access : {Access::Read, Access::ReadWrite}) {
    EXPECT_FALSE(r.define(OpCode::CopyRows)
                     .port(Port::Input, ValueKind::Tensor, Access::Read,
                           AccessDomain::CopyRowsSource)
                     .port(Port::Output, ValueKind::Tensor, access,
                           AccessDomain::CopyRowsDestination)
                     .port(Port::RowBegin, ValueKind::Scalar, Access::Read)
                     .port(Port::RowCount, ValueKind::Scalar, Access::Read)
                     .attributes<CopyRowsAttrs>()
                     .verify(AcceptCopy)
                     .finish(&error));
    EXPECT_EQ(
        error,
        "copy_rows domain requires matching operation, attributes and port");
  }
  EXPECT_TRUE(r.definitions().empty());
  ASSERT_TRUE(DefineCopyRows(&r, &error)) << error;
  EXPECT_EQ(r.find(OpCode::CopyRows)->port(Port::Output)->access(),
            Access::Write);
}
bool AcceptPublish(const BoundOperands &, const RotatePublishKVAttrs &,
                   std::string *) {
  return true;
}
TEST(AccessDomainDefinitions,
     CacheDomainsRejectWrongOpcodePortAccessAndPositions) {
  for (int variant = 0; variant < 6; ++variant) {
    OperationRegistry r;
    std::string error;
    auto d = r.define(variant == 0 ? OpCode::Rope : OpCode::RotatePublishKV);
    d.port(variant == 1 ? Port::Input : Port::KeyCache, ValueKind::Tensor,
           variant == 2 ? Access::Write : Access::ReadWrite,
           variant == 3 ? AccessDomain::CausalPrefix
                        : AccessDomain::ActiveTokenPositions)
        .optionalPort(Port::RowBegin, ValueKind::Scalar, Access::Read)
        .optionalPort(Port::RowCount, ValueKind::Scalar, Access::Read);
    if (variant != 4)
      d.port(Port::Positions, ValueKind::Tensor, Access::Read,
             variant == 5 ? AccessDomain::Whole : AccessDomain::ActiveRows);
    EXPECT_FALSE(d.attributes<RotatePublishKVAttrs>()
                     .verify(AcceptPublish)
                     .finish(&error));
    EXPECT_TRUE(r.definitions().empty());
  }
}
} // namespace
} // namespace hrx::operations
