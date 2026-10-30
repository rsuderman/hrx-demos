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
} // namespace
} // namespace hrx::operations
