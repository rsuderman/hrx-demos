// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/attention_operations.h"
#include "lib/operations/detail/checks.h"
#include <cmath>
#include <limits>
namespace hrx::operations {
namespace {
using namespace detail;
bool Rotary(const BoundOperands &o, uint64_t width, uint64_t head_width,
            double base, RopeScaling scaling, double factor, FactorMode mode,
            std::string *e) {
  if (!width || width % 2 || width > head_width || !std::isfinite(base) ||
      base <= 0 || !std::isfinite(factor) || factor <= 0 ||
      (scaling != RopeScaling::None && scaling != RopeScaling::Linear) ||
      (scaling == RopeScaling::None && factor != 1))
    return Fail(e, "invalid rotary width/base/scaling");
  if (mode == FactorMode::PerPair)
    return Vector(o, Port::Factors, width / 2, ElementType::F32, e);
  if (mode != FactorMode::None || o.value(Port::Factors))
    return Fail(e, "rotary factor mode/operand mismatch");
  return true;
}
bool Rope(const BoundOperands &o, const RopeAttrs &a, std::string *e) {
  if (!a.head_count || !a.head_width)
    return Fail(e, "rope head geometry must be positive");
  auto n = o.tensor(Port::Input)->dimension(0);
  return Rotary(o, a.rotary_width, a.head_width, a.base, a.scaling,
                a.scaling_factor, a.factor_mode, e) &&
         Heads(o, Port::Input, n, a.head_count, a.head_width, ElementType::F32,
               e) &&
         Heads(o, Port::Output, n, a.head_count, a.head_width, ElementType::F32,
               e) &&
         Vector(o, Port::Positions, n, ElementType::I32, e) &&
         Distinct(o, e, Port::Input, Port::Output);
}
bool Publish(const BoundOperands &o, const RotatePublishKVAttrs &a,
             std::string *e) {
  if (!a.kv_heads || !a.key_width || !a.value_width || !a.cache_capacity)
    return Fail(e, "KV geometry must be positive");
  auto n = o.tensor(Port::Key)->dimension(0);
  return Rotary(o, a.rotary_width, a.key_width, a.base, a.scaling,
                a.scaling_factor, a.factor_mode, e) &&
         Heads(o, Port::Key, n, a.kv_heads, a.key_width, ElementType::F32, e) &&
         Heads(o, Port::Value, n, a.kv_heads, a.value_width, ElementType::F32,
               e) &&
         Heads(o, Port::KeyCache, a.cache_capacity, a.kv_heads, a.key_width,
               ElementType::F16, e) &&
         Heads(o, Port::ValueCache, a.cache_capacity, a.kv_heads, a.value_width,
               ElementType::F16, e) &&
         Vector(o, Port::Positions, n, ElementType::I32, e) &&
         Distinct(o, e, Port::KeyCache, Port::ValueCache);
}
bool Attention(const BoundOperands &o, const CausalAttentionAttrs &a,
               std::string *e) {
  if (!a.query_heads || !a.kv_heads || a.query_heads % a.kv_heads ||
      !a.key_width || !a.value_width || !a.cache_capacity ||
      !std::isfinite(a.scale) || a.scale <= 0 ||
      a.numerical_policy != NumericalPolicy::F32Accumulate)
    return Fail(e, "invalid attention heads/widths/scale/policy");
  if (a.query_heads > std::numeric_limits<uint64_t>::max() / a.value_width)
    return Fail(e, "attention output width overflows uint64");
  auto n = o.tensor(Port::Query)->dimension(0);
  return Heads(o, Port::Query, n, a.query_heads, a.key_width, ElementType::F32,
               e) &&
         Heads(o, Port::KeyCache, a.cache_capacity, a.kv_heads, a.key_width,
               ElementType::F16, e) &&
         Heads(o, Port::ValueCache, a.cache_capacity, a.kv_heads, a.value_width,
               ElementType::F16, e) &&
         Matrix(o, Port::Output, n, a.query_heads * a.value_width,
                ElementType::F32, Encoding::Dense, Layout::RowMajor, e) &&
         Vector(o, Port::Positions, n, ElementType::I32, e) &&
         RowControls(o, e) && Distinct(o, e, Port::KeyCache, Port::ValueCache);
}
} // namespace
bool DefineRope(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::Rope)
             .port(Port::Input, ValueKind::Tensor, Access::Read)
             .port(Port::Positions, ValueKind::Tensor, Access::Read)
             .optionalPort(Port::Factors, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write)
             .attributes<RopeAttrs>()
             .verify(Rope)
             .finish(e);
}
bool DefineRotatePublishKV(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::RotatePublishKV)
             .port(Port::Key, ValueKind::Tensor, Access::Read)
             .port(Port::Value, ValueKind::Tensor, Access::Read)
             .port(Port::Positions, ValueKind::Tensor, Access::Read)
             .optionalPort(Port::Factors, ValueKind::Tensor, Access::Read)
             .port(Port::KeyCache, ValueKind::Tensor, Access::ReadWrite)
             .port(Port::ValueCache, ValueKind::Tensor, Access::ReadWrite)
             .attributes<RotatePublishKVAttrs>()
             .verify(Publish)
             .finish(e);
}
bool DefineCausalAttention(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::CausalAttention)
             .port(Port::Query, ValueKind::Tensor, Access::Read)
             .port(Port::KeyCache, ValueKind::Tensor, Access::Read)
             .port(Port::ValueCache, ValueKind::Tensor, Access::Read)
             .port(Port::RowBegin, ValueKind::Scalar, Access::Read)
             .port(Port::RowCount, ValueKind::Scalar, Access::Read)
             .port(Port::Positions, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write)
             .attributes<CausalAttentionAttrs>()
             .verify(Attention)
             .finish(e);
}
} // namespace hrx::operations
