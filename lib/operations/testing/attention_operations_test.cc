// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/attention_operations.h"
#include "lib/operations/testing/operation_test_util.h"
#include <limits>
namespace hrx::operations {
namespace {
using namespace test;
TEST(AttentionOperations, RotaryGeometryFactorsPositionsAndAliases) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineRope(&r, &e));
  OperationGraph g(r);
  auto x = H(g, 3, 4, 8), y = H(g, 3, 4, 8), p = V(g, 3, ElementType::I32),
       factors = V(g, 2);
  RopeAttrs a;
  a.head_count = 4;
  a.head_width = 8;
  a.rotary_width = 4;
  auto append = [&](ValueId out, ValueId f) {
    auto b = g.operation(OpCode::Rope);
    b.bind(Port::Input, x)
        .bind(Port::Output, out)
        .bind(Port::Positions, p)
        .attributes(a);
    if (f.valid())
      b.bind(Port::Factors, f);
    return b.append(&e);
  };
  ValueId absent;
  ASSERT_TRUE(append(y, absent)) << e;
  EXPECT_FALSE(append(y, factors));
  a.factor_mode = FactorMode::PerPair;
  EXPECT_FALSE(append(y, absent));
  ASSERT_TRUE(append(y, factors));
  EXPECT_FALSE(append(y, V(g, 3)));
  EXPECT_FALSE(append(x, factors));
  a.rotary_width = 3;
  EXPECT_FALSE(append(y, factors));
  a.rotary_width = 10;
  EXPECT_FALSE(append(y, factors));
  a.rotary_width = 4;
  a.base = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(append(y, factors));
  a.base = 10000;
  a.scaling = RopeScaling::None;
  a.scaling_factor = 2;
  EXPECT_FALSE(append(y, factors));
  a.scaling_factor = 1;
  a.factor_mode = static_cast<FactorMode>(99);
  EXPECT_FALSE(append(y, factors));
  EXPECT_EQ(g.invocations().size(), 2);
  EXPECT_FALSE(r.find(OpCode::Rope)->port(Port::Factors)->required());
}
TEST(AttentionOperations, PublicationVerifiesDistinctTypedCachesAndKVWidths) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineRotatePublishKV(&r, &e));
  OperationGraph g(r);
  auto k = H(g, 3, 2, 8), v = H(g, 3, 2, 6),
       kc = H(g, 11, 2, 8, ElementType::F16),
       vc = H(g, 11, 2, 6, ElementType::F16), p = V(g, 3, ElementType::I32);
  RotatePublishKVAttrs a;
  a.kv_heads = 2;
  a.key_width = 8;
  a.value_width = 6;
  a.cache_capacity = 11;
  a.rotary_width = 4;
  auto append = [&](ValueId values, ValueId keys_cache, ValueId values_cache) {
    return g.operation(OpCode::RotatePublishKV)
        .bind(Port::Key, k)
        .bind(Port::Value, values)
        .bind(Port::Positions, p)
        .bind(Port::KeyCache, keys_cache)
        .bind(Port::ValueCache, values_cache)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(v, kc, vc)) << e;
  EXPECT_FALSE(append(H(g, 3, 4, 6), kc, vc));
  EXPECT_FALSE(append(v, H(g, 11, 2, 8), vc));
  EXPECT_FALSE(append(v, kc, H(g, 11, 2, 8, ElementType::F16)));
  a.value_width = 8;
  EXPECT_FALSE(append(H(g, 3, 2, 8), kc, kc));
  EXPECT_NE(e.find("distinct"), std::string::npos);
  a.kv_heads = 0;
  EXPECT_FALSE(append(v, kc, vc));
  EXPECT_EQ(g.invocations().size(), 1);
  EXPECT_EQ(r.find(OpCode::RotatePublishKV)->port(Port::KeyCache)->access(),
            Access::ReadWrite);
  EXPECT_EQ(r.find(OpCode::RotatePublishKV)->port(Port::ValueCache)->access(),
            Access::ReadWrite);
}
TEST(AttentionOperations, SharedCachesFlowFromPublicationToAttention) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineRotatePublishKV(&r, &e));
  ASSERT_TRUE(DefineCausalAttention(&r, &e));
  OperationGraph g(r);
  auto q = H(g, 3, 4, 8), k = H(g, 3, 2, 8), v = H(g, 3, 2, 6),
       kc = H(g, 11, 2, 8, ElementType::F16),
       vc = H(g, 11, 2, 6, ElementType::F16), p = V(g, 3, ElementType::I32),
       out = M(g, 3, 24), begin = S(g), count = S(g);
  RotatePublishKVAttrs publish;
  publish.kv_heads = 2;
  publish.key_width = 8;
  publish.value_width = 6;
  publish.cache_capacity = 11;
  publish.rotary_width = 4;
  ASSERT_TRUE(g.operation(OpCode::RotatePublishKV)
                  .bind(Port::Key, k)
                  .bind(Port::Value, v)
                  .bind(Port::Positions, p)
                  .bind(Port::KeyCache, kc)
                  .bind(Port::ValueCache, vc)
                  .attributes(publish)
                  .append(&e))
      << e;
  CausalAttentionAttrs a;
  a.query_heads = 4;
  a.kv_heads = 2;
  a.key_width = 8;
  a.value_width = 6;
  a.cache_capacity = 11;
  a.scale = 0.353553;
  auto append = [&](ValueId query, ValueId keys_cache, ValueId values_cache,
                    ValueId y, ValueId rows) {
    return g.operation(OpCode::CausalAttention)
        .bind(Port::Query, query)
        .bind(Port::KeyCache, keys_cache)
        .bind(Port::ValueCache, values_cache)
        .bind(Port::Positions, p)
        .bind(Port::RowBegin, begin)
        .bind(Port::RowCount, rows)
        .bind(Port::Output, y)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(q, kc, vc, out, count)) << e;
  EXPECT_EQ(g.invocations()[0].binding(Port::KeyCache)->value(),
            g.invocations()[1].binding(Port::KeyCache)->value());
  EXPECT_EQ(
      g.definition(OpCode::CausalAttention)->port(Port::KeyCache)->access(),
      Access::Read);
  EXPECT_FALSE(append(H(g, 3, 2, 8), kc, vc, out, count));
  EXPECT_FALSE(append(q, kc, vc, M(g, 3, 8), count));
  EXPECT_FALSE(append(q, kc, vc, out, S(g, ScalarType::I32)));
  EXPECT_FALSE(append(S(g), kc, vc, out, count));
  a.query_heads = 5;
  EXPECT_FALSE(append(q, kc, vc, out, count));
  a.query_heads = 4;
  a.value_width = 8;
  EXPECT_FALSE(append(q, kc, kc, M(g, 3, 32), count));
  a.value_width = 6;
  a.scale = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(append(q, kc, vc, out, count));
  a.scale = 1;
  a.query_heads = std::numeric_limits<uint64_t>::max();
  a.kv_heads = 1;
  a.value_width = 2;
  EXPECT_FALSE(append(q, kc, vc, out, count));
  EXPECT_NE(e.find("overflows"), std::string::npos);
  EXPECT_EQ(g.invocations().size(), 2);
  EXPECT_EQ(std::get<CausalAttentionAttrs>(g.invocations()[1].attributes())
                .value_width,
            6);
}
TEST(AttentionOperations, FlattenedProjectionsAndPairedSpanValidation) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineRope(&r, &e));
  ASSERT_TRUE(DefineRotatePublishKV(&r, &e));
  ASSERT_TRUE(DefineCausalAttention(&r, &e));
  OperationGraph g(r), foreign(r);
  const auto x = M(g, 4, 32), y = M(g, 4, 32), p = V(g, 4, ElementType::I32);
  const auto begin = S(g), count = S(g);
  RopeAttrs a{4, 8, 8};
  auto append = [&](ValueId output, ValueId b, ValueId n) {
    auto op = g.operation(OpCode::Rope);
    op.bind(Port::Input, x)
        .bind(Port::Output, output)
        .bind(Port::Positions, p)
        .attributes(a);
    if (b.valid())
      op.bind(Port::RowBegin, b);
    if (n.valid())
      op.bind(Port::RowCount, n);
    return op.append(&e);
  };
  ASSERT_TRUE(append(y, begin, count)) << e;
  ASSERT_TRUE(append(y, {}, {})) << e;
  EXPECT_FALSE(append(y, begin, {}));
  EXPECT_FALSE(append(y, {}, count));
  EXPECT_FALSE(append(y, begin, S(g, ScalarType::I32)));
  EXPECT_FALSE(append(y, S(foreign), count));
  EXPECT_FALSE(append(H(g, 4, 4, 8), begin, count));
  EXPECT_FALSE(append(M(g, 4, 31), begin, count));
  a.head_count = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(append(y, begin, count));
  a.head_count = 4;
  const auto k = M(g, 4, 16), v = H(g, 4, 2, 8),
             kc = H(g, 32, 2, 8, ElementType::F16),
             vc = H(g, 32, 2, 8, ElementType::F16);
  auto publish = [&](ValueId cache, ValueId b, ValueId n) {
    auto op = g.operation(OpCode::RotatePublishKV);
    op.bind(Port::Key, k)
        .bind(Port::Value, v)
        .bind(Port::KeyCache, cache)
        .bind(Port::ValueCache, vc)
        .bind(Port::Positions, p)
        .attributes(RotatePublishKVAttrs{2, 8, 8, 32, 8});
    if (b.valid())
      op.bind(Port::RowBegin, b);
    if (n.valid())
      op.bind(Port::RowCount, n);
    return op.append(&e);
  };
  ASSERT_TRUE(publish(kc, begin, count)) << e;
  EXPECT_FALSE(publish(M(g, 32, 16, ElementType::F16), begin, count));
  EXPECT_FALSE(publish(kc, begin, {}));
  ASSERT_TRUE(g.operation(OpCode::CausalAttention)
                  .bind(Port::Query, y)
                  .bind(Port::KeyCache, kc)
                  .bind(Port::ValueCache, vc)
                  .bind(Port::Positions, p)
                  .bind(Port::RowBegin, begin)
                  .bind(Port::RowCount, count)
                  .bind(Port::Output, M(g, 4, 32))
                  .attributes(CausalAttentionAttrs{4, 2, 8, 8, 32, 0.353553})
                  .append(&e))
      << e;
  EXPECT_EQ(r.find(OpCode::Rope)->port(Port::Input)->domain(),
            AccessDomain::ActiveRows);
  EXPECT_EQ(r.find(OpCode::Rope)->port(Port::Positions)->domain(),
            AccessDomain::ActiveRows);
  EXPECT_EQ(r.find(OpCode::RotatePublishKV)->port(Port::KeyCache)->domain(),
            AccessDomain::ActiveTokenPositions);
  EXPECT_EQ(r.find(OpCode::CausalAttention)->port(Port::ValueCache)->domain(),
            AccessDomain::CausalPrefix);
}
} // namespace
} // namespace hrx::operations
