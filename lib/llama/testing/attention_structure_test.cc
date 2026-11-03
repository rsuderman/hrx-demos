// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/detail/attention_structure.h"
#include "lib/operations/attention_operations.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <limits>
namespace hrx::llama::detail {
namespace {
using namespace operations;
model::TensorBinding Weight(const std::string &role, const std::string &name,
                            uint32_t storage, uint64_t in, uint64_t out = 1) {
  model::TensorBinding b;
  b.role = role;
  b.tensor.name = name;
  b.tensor.storage_type = storage;
  b.tensor.normalized_extents = {int64_t(in), int64_t(out), 1, 1};
  b.tensor.absolute_offset = 4096;
  b.tensor.byte_size = 256;
  return b;
}
model::ModelDescriptor Facts(uint64_t d = 64, bool factors = true) {
  model::ModelDescriptor f;
  f.family = "llama";
  f.embedding_width = 4 * d;
  f.vocabulary_size = 1024;
  f.key_head_width = f.value_head_width = d;
  f.rms_epsilon = 1e-5;
  f.rope.width = d;
  f.rope.base = 500000;
  if (factors)
    f.globals.push_back(Weight("rope_frequencies", "rope", 0, d / 2));
  for (size_t i = 0; i < 2; ++i) {
    model::LayerDescriptor l;
    l.index = 8 + i;
    l.query_heads = 4;
    l.kv_heads = 2;
    l.feed_forward_width = 8 * d;
    const auto prefix = std::to_string(i);
    l.tensors = {Weight("input_norm", prefix + "norm", 0, 4 * d),
                 Weight("query", prefix + "q", 12, 4 * d, 4 * d),
                 Weight("key", prefix + "k", 14, 4 * d, 2 * d),
                 Weight("value", prefix + "v", 8, 4 * d, 2 * d),
                 Weight("attention_output", prefix + "out", 12, 4 * d, 4 * d)};
    f.layers.push_back(l);
  }
  return f;
}
std::unique_ptr<StructureConstruction> Make(model::ModelDescriptor facts,
                                            Phase phase = Phase::Prefill) {
  OperationRegistry r;
  std::string e;
  EXPECT_TRUE(DefineRmsNorm(&r, &e));
  EXPECT_TRUE(DefineLinear(&r, &e));
  EXPECT_TRUE(DefineLinearResidual(&r, &e));
  EXPECT_TRUE(DefineRope(&r, &e));
  EXPECT_TRUE(DefineRotatePublishKV(&r, &e));
  EXPECT_TRUE(DefineCausalAttention(&r, &e));
  auto c = StructureConstruction::Create(std::move(facts),
                                         {phase, 4, 32, 32, "asset"}, r, &e);
  EXPECT_TRUE(c) << e;
  if (c) {
    EXPECT_TRUE(c->createInputs(&e)) << e;
  }
  return c;
}
TEST(AttentionStructure, TwoGeometriesPhasesAndSemanticStateCuts) {
  for (const auto d : {64u, 128u})
    for (const auto phase : {Phase::Prefill, Phase::Decode}) {
      auto c = Make(Facts(d), phase);
      ASSERT_TRUE(c);
      std::string e;
      NormalizedAttentionRegion a, b;
      ASSERT_TRUE(AppendNormalizedAttention(c.get(), 0, &a, &e)) << e;
      ASSERT_TRUE(AppendNormalizedAttention(c.get(), 1, &b, &e)) << e;
      EXPECT_EQ(a.operation_begin, 0u);
      EXPECT_EQ(a.operation_end, 8u);
      EXPECT_EQ(b.operation_begin, 8u);
      EXPECT_EQ(b.operation_end, 16u);
      EXPECT_EQ(a.residual, b.residual);
      EXPECT_NE(a.key_cache, b.key_cache);
      EXPECT_NE(a.value_cache, b.value_cache);
      EXPECT_NE(a.key_cache, a.value_cache);
      const auto &s = c->structure();
      const auto &g = s.graph();
      const std::array<OpCode, 8> order = {OpCode::RmsNorm,
                                           OpCode::Linear,
                                           OpCode::Linear,
                                           OpCode::Linear,
                                           OpCode::Rope,
                                           OpCode::RotatePublishKV,
                                           OpCode::CausalAttention,
                                           OpCode::LinearResidual};
      for (size_t i = 0; i < 8; ++i) {
        const auto &op = g.invocations()[i];
        EXPECT_EQ(op.opcode(), order[i]);
        EXPECT_EQ(op.binding(Port::RowBegin)->value(), s.inputs().row_begin);
        EXPECT_EQ(op.binding(Port::RowCount)->value(), s.inputs().row_count);
      }
      const auto &rope = g.invocations()[4];
      EXPECT_EQ(std::get<RopeAttrs>(rope.attributes()).base, 500000);
      EXPECT_EQ(std::get<RopeAttrs>(rope.attributes()).factor_mode,
                FactorMode::PerPair);
      EXPECT_EQ(rope.binding(Port::Factors)->value(),
                g.invocations()[5].binding(Port::Factors)->value());
      EXPECT_EQ(
          g.value(rope.binding(Port::Input)->value())->tensorType()->rank(),
          2u);
      EXPECT_EQ(g.value(a.key_cache)->tensorType()->rank(), 3u);
      EXPECT_EQ(s.parameters().size(),
                11u); // One shared global + five per layer.
      ASSERT_TRUE(c->exportValue(a.key_cache, &e));
      ASSERT_TRUE(c->exportValue(a.value_cache, &e));
      LlamaRegionView view;
      ASSERT_TRUE(s.select({LlamaRegionKind::Attention, 0, 1}, &view, &e)) << e;
      auto port = [&](ValueId v) {
        return std::find_if(view.outside.begin(), view.outside.end(),
                            [&](const auto &p) { return p.value == v; });
      };
      ASSERT_NE(port(a.key_cache), view.outside.end());
      EXPECT_EQ(port(a.key_cache)->access, Access::ReadWrite);
      ASSERT_NE(port(a.value_cache), view.outside.end());
      EXPECT_EQ(port(a.value_cache)->access, Access::ReadWrite);
      ASSERT_NE(port(a.residual), view.outside.end());
      EXPECT_EQ(port(a.residual)->access, Access::ReadWrite);
      for (const auto &op : g.invocations()) {
        if (&op == &g.invocations()[8])
          break;
        if (const auto *output = op.binding(Port::Output)) {
          EXPECT_EQ(port(output->value()), view.outside.end());
        }
      }
    }
}
TEST(AttentionStructure, OptionalFactorsAndFailurePublication) {
  auto c = Make(Facts(64, false));
  ASSERT_TRUE(c);
  std::string e;
  NormalizedAttentionRegion out;
  ASSERT_TRUE(AppendNormalizedAttention(c.get(), 0, &out, &e)) << e;
  EXPECT_FALSE(c->graph().invocations()[4].binding(Port::Factors));
  EXPECT_EQ(
      std::get<RopeAttrs>(c->graph().invocations()[4].attributes()).factor_mode,
      FactorMode::None);
  for (int variant = 0; variant < 8; ++variant) {
    auto f = Facts();
    switch (variant) {
    case 0:
      f.rope.width = 32;
      break;
    case 1:
      f.layers[0].tensors[1].tensor.normalized_extents[1] = 128;
      break;
    case 2:
      f.layers[0].tensors[0].tensor.storage_type = 12;
      break;
    case 3:
      f.globals[0].tensor.normalized_extents[0] = 31;
      break;
    case 4:
      f.rope.scaling_kind = "unknown";
      break;
    case 5:
      f.layers[0].tensors[1].role = "missing";
      break;
    case 6:
      f.layers[0].tensors[2].tensor.storage_type = 2;
      break;
    case 7:
      f.rope.sections[0] = 1;
      break;
    }
    auto bad = Make(f);
    ASSERT_TRUE(bad);
    NormalizedAttentionRegion sentinel;
    sentinel.operation_end = 999;
    EXPECT_FALSE(AppendNormalizedAttention(bad.get(), 0, &sentinel, &e))
        << variant;
    EXPECT_EQ(sentinel.operation_end, 999u);
    EXPECT_TRUE(bad->graph().invocations().empty());
  }
}
} // namespace
} // namespace hrx::llama::detail
