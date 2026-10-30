// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "examples/operations/example_utils.h"
#include "lib/operations/attention_operations.h"
using namespace hrx::operations;
using namespace hrx::operations::example;
bool Build(std::string *e) {
  OperationRegistry r;
  if (!DefineRope(&r, e) || !DefineRotatePublishKV(&r, e) ||
      !DefineCausalAttention(&r, e))
    return false;
  OperationGraph g(r);
  auto query = H(g, 4, 4, 32, ElementType::F32, e),
       rotated = H(g, 4, 4, 32, ElementType::F32, e);
  auto key = H(g, 4, 2, 32, ElementType::F32, e),
       value = H(g, 4, 2, 48, ElementType::F32, e);
  auto kc = H(g, 64, 2, 32, ElementType::F16, e),
       vc = H(g, 64, 2, 48, ElementType::F16, e);
  auto positions = V(g, 4, ElementType::I32, e), out = M(g, 4, 192, e);
  auto begin = g.scalar(ScalarType::I64, e),
       count = g.scalar(ScalarType::I64, e);
  RopeAttrs rope;
  rope.head_count = 4;
  rope.head_width = 32;
  rope.rotary_width = 16;
  RotatePublishKVAttrs publish;
  publish.kv_heads = 2;
  publish.key_width = 32;
  publish.value_width = 48;
  publish.cache_capacity = 64;
  publish.rotary_width = 16;
  CausalAttentionAttrs attention;
  attention.query_heads = 4;
  attention.kv_heads = 2;
  attention.key_width = 32;
  attention.value_width = 48;
  attention.cache_capacity = 64;
  attention.scale = 0.176777;
  if (!g.operation(OpCode::Rope)
           .bind(Port::Input, query)
           .bind(Port::Positions, positions)
           .bind(Port::Output, rotated)
           .attributes(rope)
           .append(e))
    return false;
  if (!g.operation(OpCode::RotatePublishKV)
           .bind(Port::Key, key)
           .bind(Port::Value, value)
           .bind(Port::Positions, positions)
           .bind(Port::KeyCache, kc)
           .bind(Port::ValueCache, vc)
           .attributes(publish)
           .append(e))
    return false;
  if (!g.operation(OpCode::CausalAttention)
           .bind(Port::Query, rotated)
           .bind(Port::KeyCache, kc)
           .bind(Port::ValueCache, vc)
           .bind(Port::RowBegin, begin)
           .bind(Port::RowCount, count)
           .bind(Port::Positions, positions)
           .bind(Port::Output, out)
           .attributes(attention)
           .append(e))
    return false;
  if (g.invocations()[1].binding(Port::KeyCache)->value() !=
      g.invocations()[2].binding(Port::KeyCache)->value()) {
    *e = "cache identity lost";
    return false;
  }
  return Print(g, 3, e);
}
int main() {
  std::string error;
  if (!Build(&error)) {
    std::cerr << error << '\n';
    return 1;
  }
  return 0;
}
