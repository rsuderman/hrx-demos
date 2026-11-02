// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/detail/ffn_structure.h"
#include "lib/llama/detail/structure_construction.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/row_operations.h"
#include "lib/operations/tensor_operations.h"
#include "gtest/gtest.h"
#include <limits>
namespace hrx::llama::detail {
// Explicit incomplete test fixture publication; not the complete-model factory.
class StructureTestPeer {
public:
  static std::shared_ptr<const LlamaStructure>
  Retain(StructureConstruction &construction) {
    return construction.owner_;
  }
};
namespace {
using namespace operations;
model::TensorBinding Weight(std::string role, std::string name,
                            uint32_t storage, int64_t in, int64_t out = 1) {
  model::TensorBinding binding;
  binding.role = std::move(role);
  binding.tensor.name = std::move(name);
  binding.tensor.storage_type = storage;
  binding.tensor.normalized_extents = {in, out, 1, 1};
  binding.tensor.absolute_offset = 4096;
  binding.tensor.byte_size = 256;
  return binding;
}
model::ModelDescriptor Facts(uint64_t width = 256, uint64_t ffn = 512) {
  model::ModelDescriptor f;
  f.family = "llama";
  f.embedding_width = width;
  f.vocabulary_size = 1024;
  f.key_head_width = f.value_head_width = 64;
  f.rms_epsilon = 1e-5;
  f.globals = {Weight("token_embedding", "tied", 12, width, 1024),
               Weight("output", "tied", 12, width, 1024)};
  for (size_t i = 0; i < 2; ++i) {
    model::LayerDescriptor l;
    l.index = 7 + i;
    l.query_heads = width / 64;
    l.kv_heads = 2;
    l.feed_forward_width = ffn;
    // Valid repeated records deliberately alias gate/up and both layers.
    l.tensors = {Weight("post_norm", "norm", 0, width),
                 Weight("ffn_gate", "gate-up", 12, width, ffn),
                 Weight("ffn_up", "gate-up", 12, width, ffn),
                 Weight("ffn_down", "down", 14, ffn, width)};
    f.layers.push_back(l);
  }
  return f;
}
LlamaBuildOptions Options() {
  return {Phase::Prefill, 4, 16, 32, "checkpoint"};
}
std::unique_ptr<StructureConstruction>
Construction(model::ModelDescriptor f = Facts()) {
  OperationRegistry registry;
  std::string error;
  EXPECT_TRUE(DefineRmsNorm(&registry, &error));
  EXPECT_TRUE(DefineGatedMatmul(&registry, &error));
  EXPECT_TRUE(DefineLinearResidual(&registry, &error));
  EXPECT_TRUE(DefineCopyRows(&registry, &error));
  auto c =
      StructureConstruction::Create(std::move(f), Options(), registry, &error);
  EXPECT_TRUE(c) << error;
  return c;
}
bool Append(StructureConstruction &c, size_t layer, NormalizedFfnRegion *region,
            std::string *error) {
  std::array<ValueId, 4> weights;
  for (size_t i = 0; i < 4; ++i)
    if (!c.resolve({ParameterScope::Layer, layer, i}, &weights[i], error))
      return false;
  const auto &input = c.structure().inputs();
  return AppendNormalizedFfnResolved(&c.graph(), c.structure().facts(), layer,
                                     input.residual, weights, input.row_begin,
                                     input.row_count, region, error);
}
TEST(LlamaStructure, OwnsFactsGraphAliasesAndIndependentLifetimes) {
  std::shared_ptr<const LlamaStructure> retained;
  ValueId tied, output;
  {
    auto c = Construction();
    ASSERT_TRUE(c);
    std::string error;
    ASSERT_TRUE(c->createInputs(&error)) << error;
    ASSERT_TRUE(c->resolve({ParameterScope::Global, 0, 0}, &tied, &error));
    ASSERT_TRUE(c->resolve({ParameterScope::Global, 0, 1}, &output, &error));
    EXPECT_EQ(tied, output);
    EXPECT_EQ(c->structure().parameters().size(), 2);
    ASSERT_TRUE(c->resolve({ParameterScope::Global, 0, 1}, &output, &error));
    EXPECT_EQ(c->structure().parameters().size(), 2);
    retained = StructureTestPeer::Retain(*c);
  }
  ASSERT_TRUE(retained->graph().value(tied));
  EXPECT_EQ(retained->facts().layers[0].index, 7);
  EXPECT_EQ(retained->parameter({ParameterScope::Global, 0, 1})->role,
            "output");
  auto other = Construction();
  ASSERT_TRUE(other);
  EXPECT_FALSE(other->graph().value(tied));
  std::string error;
  EXPECT_FALSE(other->exportValue(tied, &error));
  EXPECT_EQ(retained->inputs().layers.size(), 2);
  EXPECT_NE(retained->inputs().layers[0].key_cache,
            retained->inputs().layers[1].key_cache);
  EXPECT_NE(retained->inputs().layers[0].key_cache,
            retained->inputs().layers[0].value_cache);
  EXPECT_EQ(retained->graph().value(retained->inputs().row_begin)->kind(),
            ValueKind::Scalar);
}
TEST(LlamaStructure, RejectsConflictingIdentityAndCoordinatesTransactionally) {
  for (int field = 0; field < 4; ++field) {
    auto facts = Facts();
    auto &t = facts.globals[1].tensor;
    if (field == 0)
      ++t.normalized_extents[0];
    if (field == 1)
      t.storage_type = 14;
    if (field == 2)
      ++t.absolute_offset;
    if (field == 3)
      ++t.byte_size;
    auto c = Construction(std::move(facts));
    ASSERT_TRUE(c);
    std::string error;
    ValueId first, second;
    ASSERT_TRUE(c->resolve({ParameterScope::Global, 0, 0}, &first, &error));
    second = first;
    EXPECT_FALSE(c->resolve({ParameterScope::Global, 0, 1}, &second, &error));
    EXPECT_EQ(second, first);
    EXPECT_EQ(c->structure().parameters().size(), 1);
    EXPECT_FALSE(c->resolve({ParameterScope::Global, 1, 0}, &second, &error));
    EXPECT_FALSE(c->resolve({ParameterScope::Layer, 99, 0}, &second, &error));
  }
  auto facts = Facts();
  facts.globals[1].tensor.storage_name = "misleading";
  auto c = Construction(std::move(facts));
  ASSERT_TRUE(c);
  std::string error;
  ValueId a, b;
  EXPECT_TRUE(c->resolve({ParameterScope::Global, 0, 0}, &a, &error));
  EXPECT_TRUE(c->resolve({ParameterScope::Global, 0, 1}, &b, &error));
  EXPECT_EQ(a, b);
}
TEST(LlamaStructure, ValidatesOptionsAndGeometryWithoutReadingRuntimeContents) {
  OperationRegistry registry;
  std::string error;
  for (int field = 0; field < 10; ++field) {
    auto f = Facts();
    auto o = Options();
    if (field == 0)
      o.phase = static_cast<Phase>(99);
    if (field == 1)
      o.row_capacity = 0;
    if (field == 2)
      o.row_capacity = 17;
    if (field == 3)
      o.cache_capacity = 0;
    if (field == 4)
      o.history_capacity = 0;
    if (field == 5)
      o.source_asset.clear();
    if (field == 6)
      f.key_head_width = 0;
    if (field == 7)
      f.value_head_width = 32;
    if (field == 8)
      f.layers[1].index = f.layers[0].index;
    if (field == 9)
      o.row_capacity = std::numeric_limits<uint64_t>::max();
    EXPECT_FALSE(
        StructureConstruction::Create(std::move(f), o, registry, &error));
  }
  auto o = Options();
  o.phase = Phase::Decode;
  EXPECT_TRUE(
      StructureConstruction::Create(Facts(512, 1024), o, registry, &error));
}
TEST(LlamaStructure, ResolvedFfnAliasesSpansRegionsAndExports) {
  for (auto width : {256, 512}) {
    auto c = Construction(Facts(width, width * 2));
    ASSERT_TRUE(c);
    std::string error;
    ASSERT_TRUE(c->createInputs(&error));
    NormalizedFfnRegion first, second;
    ASSERT_TRUE(Append(*c, 0, &first, &error)) << error;
    ASSERT_TRUE(c->record({LlamaRegionKind::Ffn, 0, 1}, 0, 3, &error));
    ASSERT_TRUE(c->record({LlamaRegionKind::Layer, 0, 1}, 0, 3, &error));
    ASSERT_TRUE(Append(*c, 1, &second, &error)) << error;
    ASSERT_TRUE(c->record({LlamaRegionKind::Layer, 1, 1}, 3, 6, &error));
    ASSERT_TRUE(c->exportValue(c->structure().inputs().residual, &error));
    EXPECT_EQ(first.parameters[1].value, first.parameters[2].value);
    EXPECT_EQ(first.parameters[0].value, second.parameters[0].value);
    for (const auto &op : c->graph().invocations()) {
      EXPECT_EQ(op.binding(Port::RowBegin)->value(),
                c->structure().inputs().row_begin);
      EXPECT_EQ(op.binding(Port::RowCount)->value(),
                c->structure().inputs().row_count);
      EXPECT_EQ(c->graph().definition(op.opcode())->port(Port::Input)->domain(),
                AccessDomain::ActiveRows);
    }
    auto fixture = StructureTestPeer::Retain(*c);
    LlamaRegionView view;
    ASSERT_TRUE(fixture->select({LlamaRegionKind::Ffn, 0, 1}, &view, &error));
    EXPECT_EQ(view.operation_begin, 0);
    EXPECT_EQ(view.operation_end, 3);
    bool residual = false;
    for (const auto &port : view.outside) {
      EXPECT_NE(port.value, first.normalized);
      EXPECT_NE(port.value, first.activation);
      if (port.value == first.residual) {
        residual = true;
        EXPECT_EQ(port.access, Access::ReadWrite);
      }
    }
    EXPECT_TRUE(residual);
    ASSERT_TRUE(
        fixture->select({LlamaRegionKind::Layers, 0, 2}, &view, &error));
    EXPECT_EQ(view.operation_end, 6);
    for (const auto &port : view.outside)
      EXPECT_NE(port.value, second.normalized);
    auto sentinel = view.operation_end;
    EXPECT_FALSE(fixture->select(
        {LlamaRegionKind::Layers, 1, std::numeric_limits<size_t>::max()}, &view,
        &error));
    EXPECT_EQ(view.operation_end, sentinel);
    EXPECT_FALSE(
        fixture->select({LlamaRegionKind::Output, 0, 0}, &view, &error));
    EXPECT_FALSE(c->record({LlamaRegionKind::Layer, 0, 1}, 0, 3, &error));
    EXPECT_FALSE(c->record({LlamaRegionKind::Ffn, 1, 1}, 3, 7, &error));
  }
}
TEST(LlamaStructure, RejectsBadResolvedWeightsAndControlPairs) {
  auto c = Construction();
  ASSERT_TRUE(c);
  std::string error;
  ASSERT_TRUE(c->createInputs(&error));
  std::array<ValueId, 4> weights;
  for (size_t i = 0; i < 4; ++i)
    ASSERT_TRUE(c->resolve({ParameterScope::Layer, 0, i}, &weights[i], &error));
  auto input = c->structure().inputs();
  NormalizedFfnRegion region;
  region.operation_end = 99;
  EXPECT_FALSE(AppendNormalizedFfnResolved(
      &c->graph(), c->structure().facts(), 0, input.residual, weights,
      input.row_begin, {}, &region, &error));
  EXPECT_EQ(region.operation_end, 99);
  EXPECT_TRUE(c->graph().invocations().empty());
  std::swap(weights[0], weights[1]);
  EXPECT_FALSE(AppendNormalizedFfnResolved(
      &c->graph(), c->structure().facts(), 0, input.residual, weights,
      input.row_begin, input.row_count, &region, &error));
  EXPECT_EQ(region.operation_end, 99);
  EXPECT_TRUE(c->graph().invocations().empty());
}
TEST(LlamaStructure,
     OrderedLivenessKillsOverwritesAndKeepsExplicitFinalExports) {
  auto c = Construction();
  ASSERT_TRUE(c);
  std::string error;
  auto &g = c->graph();
  auto input = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
  auto temp = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
  auto result = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
  auto gamma = g.tensor(TensorType::vector(256, ElementType::F32), &error);
  auto norm = [&](ValueId in, ValueId out) {
    return g.operation(OpCode::RmsNorm)
        .bind(Port::Input, in)
        .bind(Port::Gamma, gamma)
        .bind(Port::Output, out)
        .attributes(RmsNormAttrs{256, 1e-5})
        .append(&error);
  };
  ASSERT_TRUE(norm(input, temp));
  ASSERT_TRUE(
      norm(input, temp)); // This full overwrite kills the first version.
  ASSERT_TRUE(norm(temp, result));
  ASSERT_TRUE(c->exportValue(temp, &error));
  ASSERT_TRUE(c->exportValue(result, &error));
  ASSERT_TRUE(c->record({LlamaRegionKind::Embedding, 0, 0}, 0, 1, &error));
  ASSERT_TRUE(c->record({LlamaRegionKind::Output, 0, 0}, 1, 3, &error));
  auto f = StructureTestPeer::Retain(*c);
  LlamaRegionView view;
  ASSERT_TRUE(f->select({LlamaRegionKind::Embedding, 0, 0}, &view, &error));
  for (const auto &p : view.outside)
    EXPECT_NE(p.value, temp);
  ASSERT_TRUE(f->select({LlamaRegionKind::Output, 0, 0}, &view, &error));
  bool final = false;
  for (const auto &p : view.outside)
    if (p.value == result) {
      final = true;
      EXPECT_EQ(p.access, Access::Write);
    }
  EXPECT_TRUE(final);
}
TEST(LlamaStructure, DifferentActiveDomainsDoNotHideRequiredOutsideRows) {
  auto c = Construction();
  ASSERT_TRUE(c);
  std::string error;
  ASSERT_TRUE(c->createInputs(&error));
  auto &g = c->graph();
  auto in = c->structure().inputs();
  auto temp = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
  auto output = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
  auto gamma = g.tensor(TensorType::vector(256, ElementType::F32), &error);
  auto other_begin = g.scalar(ScalarType::I64, &error);
  for (int i = 0; i < 2; ++i)
    ASSERT_TRUE(g.operation(OpCode::RmsNorm)
                    .bind(Port::Input, i ? temp : in.residual)
                    .bind(Port::Gamma, gamma)
                    .bind(Port::Output, i ? output : temp)
                    .bind(Port::RowBegin, i ? other_begin : in.row_begin)
                    .bind(Port::RowCount, in.row_count)
                    .attributes(RmsNormAttrs{256, 1e-5})
                    .append(&error));
  ASSERT_TRUE(c->record({LlamaRegionKind::Embedding, 0, 0}, 0, 2, &error));
  LlamaRegionView view;
  ASSERT_TRUE(
      c->structure().select({LlamaRegionKind::Embedding, 0, 0}, &view, &error));
  bool required = false;
  for (const auto &p : view.outside)
    if (p.value == temp) {
      required = true;
      EXPECT_EQ(p.access, Access::Read);
    }
  EXPECT_TRUE(required); // Inspection reports the gap; it does not prove
                         // runtime initialization.
}

bool HasOutside(const LlamaRegionView &view, ValueId value) {
  for (const auto &port : view.outside)
    if (port.value == value)
      return true;
  return false;
}
TEST(LlamaStructure, ActiveProducerCoversCopyRangeAndLastActiveHead) {
  for (auto selection : {RowSelection::Range, RowSelection::LastActive}) {
    auto c = Construction();
    ASSERT_TRUE(c);
    std::string error;
    ASSERT_TRUE(c->createInputs(&error));
    auto &g = c->graph();
    auto in = c->structure().inputs();
    const auto rows = selection == RowSelection::LastActive ? 1 : 4;
    auto temp = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
    auto selected =
        g.tensor(TensorType::matrix(rows, 256, ElementType::F32), &error);
    auto result =
        g.tensor(TensorType::matrix(rows, 256, ElementType::F32), &error);
    auto gamma = g.tensor(TensorType::vector(256, ElementType::F32), &error);
    ASSERT_TRUE(g.operation(OpCode::RmsNorm)
                    .bind(Port::Input, in.residual)
                    .bind(Port::Gamma, gamma)
                    .bind(Port::Output, temp)
                    .bind(Port::RowBegin, in.row_begin)
                    .bind(Port::RowCount, in.row_count)
                    .attributes(RmsNormAttrs{256, 1e-5})
                    .append(&error));
    ASSERT_TRUE(g.operation(OpCode::CopyRows)
                    .bind(Port::Input, temp)
                    .bind(Port::Output, selected)
                    .bind(Port::RowBegin, in.row_begin)
                    .bind(Port::RowCount, in.row_count)
                    .attributes(CopyRowsAttrs{256, selection})
                    .append(&error));
    if (selection == RowSelection::LastActive) {
      ASSERT_TRUE(g.operation(OpCode::RmsNorm)
                      .bind(Port::Input, selected)
                      .bind(Port::Gamma, gamma)
                      .bind(Port::Output, result)
                      .attributes(RmsNormAttrs{256, 1e-5})
                      .append(&error));
      ASSERT_TRUE(c->exportValue(result, &error));
    } else
      ASSERT_TRUE(c->exportValue(selected, &error));
    ASSERT_TRUE(c->record({LlamaRegionKind::Output, 0, 0}, 0,
                          g.invocations().size(), &error));
    LlamaRegionView view;
    ASSERT_TRUE(
        c->structure().select({LlamaRegionKind::Output, 0, 0}, &view, &error));
    EXPECT_FALSE(HasOutside(view, temp));
    if (selection == RowSelection::LastActive) {
      EXPECT_FALSE(HasOutside(view, selected));
      EXPECT_TRUE(HasOutside(view, result));
    } else {
      ASSERT_TRUE(HasOutside(view, selected));
      for (const auto &port : view.outside)
        if (port.value == selected)
          EXPECT_EQ(port.access, Access::Write);
    }
  }
}
TEST(LlamaStructure, WriteInvalidatesPriorCoverageWhileReadWritePreservesIt) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    auto c = Construction();
    ASSERT_TRUE(c);
    std::string error;
    ASSERT_TRUE(c->createInputs(&error));
    auto &g = c->graph();
    auto in = c->structure().inputs();
    auto other_begin = g.scalar(ScalarType::I64, &error);
    auto temp = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
    auto result =
        g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
    auto gamma = g.tensor(TensorType::vector(256, ElementType::F32), &error);
    auto norm = [&](ValueId input, ValueId output, ValueId begin) {
      auto op = g.operation(OpCode::RmsNorm);
      if (begin.valid())
        op.bind(Port::RowBegin, begin).bind(Port::RowCount, in.row_count);
      return op.bind(Port::Input, input)
          .bind(Port::Gamma, gamma)
          .bind(Port::Output, output)
          .attributes(RmsNormAttrs{256, 1e-5})
          .append(&error);
    };
    // Scenario 0: A -> write B -> read A, exposing stale A coverage.
    // Scenario 1/2: whole -> write/readwrite B -> read whole.
    ASSERT_TRUE(
        norm(in.residual, temp, scenario == 0 ? in.row_begin : ValueId{}));
    if (scenario == 2) {
      auto weight = g.tensor(TensorType::matrix(256, 256, ElementType::F32)
                                 .setEncoding(Encoding::Q4_K)
                                 .setLayout(Layout::GgmlWeightRows),
                             &error);
      ASSERT_TRUE(g.operation(OpCode::LinearResidual)
                      .bind(Port::Input, in.residual)
                      .bind(Port::Weight, weight)
                      .bind(Port::Residual, temp)
                      .bind(Port::RowBegin, other_begin)
                      .bind(Port::RowCount, in.row_count)
                      .attributes(LinearResidualAttrs{256, 256})
                      .append(&error));
    } else
      ASSERT_TRUE(norm(in.residual, temp, other_begin));
    ASSERT_TRUE(norm(temp, result, scenario == 0 ? in.row_begin : ValueId{}));
    ASSERT_TRUE(c->record({LlamaRegionKind::Embedding, 0, 0}, 0, 3, &error));
    LlamaRegionView view;
    ASSERT_TRUE(c->structure().select({LlamaRegionKind::Embedding, 0, 0}, &view,
                                      &error));
    EXPECT_EQ(HasOutside(view, temp), scenario != 2);
  }
}
TEST(LlamaStructure,
     ActiveWriteKillsSelectedVersionsBeforeLaterReadsAndFinalExports) {
  for (bool same_span : {false, true}) {
    auto c = Construction();
    ASSERT_TRUE(c);
    std::string error;
    ASSERT_TRUE(c->createInputs(&error));
    auto &g = c->graph();
    auto in = c->structure().inputs();
    auto other_begin = g.scalar(ScalarType::I64, &error);
    auto temp = g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
    auto result =
        g.tensor(TensorType::matrix(4, 256, ElementType::F32), &error);
    auto gamma = g.tensor(TensorType::vector(256, ElementType::F32), &error);
    for (int i = 0; i < 3; ++i)
      ASSERT_TRUE(g.operation(OpCode::RmsNorm)
                      .bind(Port::Input, i == 2 ? temp : in.residual)
                      .bind(Port::Gamma, gamma)
                      .bind(Port::Output, i == 2 ? result : temp)
                      .bind(Port::RowBegin,
                            i == 0 || same_span ? in.row_begin : other_begin)
                      .bind(Port::RowCount, in.row_count)
                      .attributes(RmsNormAttrs{256, 1e-5})
                      .append(&error));
    ASSERT_TRUE(c->record({LlamaRegionKind::Embedding, 0, 0}, 0, 1, &error));
    ASSERT_TRUE(c->exportValue(temp, &error));
    LlamaRegionView view;
    ASSERT_TRUE(c->structure().select({LlamaRegionKind::Embedding, 0, 0}, &view,
                                      &error));
    EXPECT_FALSE(HasOutside(view, temp));
  }
}
TEST(LlamaStructure, SharedNumericMatrixAdmissionIsExplicit) {
  Encoding e;
  EXPECT_TRUE(MatrixEncoding(8, &e));
  EXPECT_EQ(e, Encoding::Q8_0);
  EXPECT_TRUE(MatrixEncoding(12, &e));
  EXPECT_EQ(e, Encoding::Q4_K);
  EXPECT_TRUE(MatrixEncoding(14, &e));
  EXPECT_EQ(e, Encoding::Q6_K);
  for (auto tag : {0, 1, 2, 13, 99})
    EXPECT_FALSE(MatrixEncoding(tag, &e));
}
} // namespace
} // namespace hrx::llama::detail
