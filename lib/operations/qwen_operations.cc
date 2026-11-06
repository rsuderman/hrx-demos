// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/qwen_operations.h"
#include "lib/operations/detail/checks.h"
#include <cmath>
#include <limits>
namespace hrx::operations {
namespace {
using namespace detail;
bool Positive(uint64_t n) { return n && n <= uint64_t(INT64_MAX); }
bool Product(uint64_t a, uint64_t b, uint64_t *out) {
  if (!Positive(a) || !Positive(b) || a > uint64_t(INT64_MAX) / b)
    return false;
  *out = a * b;
  return true;
}
bool Epoch(const BoundOperands &o, size_t words, std::string *e) {
  return Tensor(o, Port::Epoch, ElementType::I32, Encoding::Dense,
                Layout::RowMajor, 1, e) &&
         (o.tensor(Port::Epoch)->dimension(0) >= words ||
          Fail(e, "epoch readable prefix is too short"));
}
bool Quantized(const BoundOperands &o, Port p, uint64_t in, uint64_t out,
               std::string *e) {
  if (!Weight(o, p, in, out, e))
    return false;
  auto encoding = o.tensor(p)->encoding();
  return encoding == Encoding::Q4_K || encoding == Encoding::Q5_K ||
         encoding == Encoding::Q6_K ||
         Fail(e, "Qwen weight requires Q4_K/Q5_K/Q6_K");
}
bool Layer(const BoundOperands &o, uint64_t t, uint64_t s, uint64_t h,
           uint64_t f, double eps, std::string *e) {
  if (!t || t > 512 || !s || s > 64 || !Positive(h) || !Positive(f) ||
      !std::isfinite(eps) || eps <= 0)
    return Fail(e, "invalid Qwen layer dimensions, capacities or epsilon");
  return Matrix(o, Port::Residual, t, h, ElementType::F32, Encoding::Dense,
                Layout::RowMajor, e) &&
         Epoch(o, 1048, e) &&
         Tensor(o, Port::Rows, ElementType::I8, Encoding::Dense,
                Layout::RowMajor, 1, e) &&
         (o.tensor(Port::Rows)->dimension(0) >= 2048 ||
          Fail(e, "Qwen row table is too short")) &&
         Vector(o, Port::InputNormWeight, h, ElementType::F32, e) &&
         Vector(o, Port::PostNormWeight, h, ElementType::F32, e) &&
         Quantized(o, Port::GateWeight, h, f, e) &&
         Quantized(o, Port::UpWeight, h, f, e) &&
         Quantized(o, Port::DownWeight, f, h, e);
}
bool Gdn(const BoundOperands &o, const QwenGdnLayerAttrs &a, std::string *e) {
  uint64_t k, v, qkv;
  if (!Product(a.key_heads, a.state_width, &k) ||
      !Product(a.value_heads, a.state_width, &v) ||
      k > (uint64_t(INT64_MAX) - v) / 2 || a.value_heads % a.key_heads ||
      !Positive(a.convolution_taps) || a.convolution_taps < 2)
    return Fail(e, "invalid Qwen GDN geometry");
  qkv = 2 * k + v;
  return Layer(o, a.token_capacity, a.span_capacity, a.hidden_width,
               a.ffn_width, a.epsilon, e) &&
         Quantized(o, Port::QkvWeight, a.hidden_width, qkv, e) &&
         Quantized(o, Port::RecurrentGateWeight, a.hidden_width, v, e) &&
         Quantized(o, Port::AlphaWeight, a.hidden_width, a.value_heads, e) &&
         Quantized(o, Port::BetaWeight, a.hidden_width, a.value_heads, e) &&
         Quantized(o, Port::RecurrentOutputWeight, v, a.hidden_width, e) &&
         Matrix(o, Port::ConvolutionWeight, qkv, a.convolution_taps,
                ElementType::F32, Encoding::Dense, Layout::RowMajor, e) &&
         Vector(o, Port::DtBias, a.value_heads, ElementType::F32, e) &&
         Vector(o, Port::DecayWeight, a.value_heads, ElementType::F32, e) &&
         Vector(o, Port::RecurrentNormWeight, a.state_width, ElementType::F32,
                e) &&
         Matrix(o, Port::ConvolutionHistory, qkv, a.convolution_taps - 1,
                ElementType::F32, Encoding::Dense, Layout::RowMajor, e) &&
         Heads(o, Port::RecurrentState, a.value_heads, a.state_width,
               a.state_width, ElementType::F32, e) &&
         Distinct(o, e, Port::Residual, Port::ConvolutionHistory,
                  Port::RecurrentState, Port::Epoch, Port::Rows);
}
bool Attention(const BoundOperands &o, const QwenAttentionLayerAttrs &a,
               std::string *e) {
  uint64_t q, k;
  if (!Product(a.query_heads, a.head_width, &q) ||
      !Product(a.kv_heads, a.head_width, &k) || q > uint64_t(INT64_MAX) / 2 ||
      a.query_heads % a.kv_heads || !Positive(a.cache_capacity) ||
      !a.rotary_width || a.rotary_width > a.head_width || a.rotary_width % 2 ||
      !std::isfinite(a.rope_base) || a.rope_base <= 0 ||
      !std::isfinite(a.rope_scaling_factor) || a.rope_scaling_factor <= 0 ||
      !std::isfinite(a.attention_scale) || a.attention_scale <= 0 ||
      (a.rope_scaling != RopeScaling::None &&
       a.rope_scaling != RopeScaling::Linear))
    return Fail(e, "invalid Qwen attention geometry or rotary configuration");
  uint64_t sum = 0;
  for (auto section : a.rope_sections) {
    if (section > a.rotary_width / 2 - sum)
      return Fail(e, "invalid rotary sections");
    sum += section;
  }
  if (sum != a.rotary_width / 2)
    return Fail(e, "rotary sections must cover rotary width");
  return Layer(o, a.token_capacity, a.span_capacity, a.hidden_width,
               a.ffn_width, a.epsilon, e) &&
         Quantized(o, Port::QueryWeight, a.hidden_width, 2 * q, e) &&
         Quantized(o, Port::KeyWeight, a.hidden_width, k, e) &&
         Quantized(o, Port::ValueWeight, a.hidden_width, k, e) &&
         Quantized(o, Port::AttentionOutputWeight, q, a.hidden_width, e) &&
         Vector(o, Port::QueryNormWeight, a.head_width, ElementType::F32, e) &&
         Vector(o, Port::KeyNormWeight, a.head_width, ElementType::F32, e) &&
         Heads(o, Port::KeyCache, a.cache_capacity, a.kv_heads, a.head_width,
               ElementType::F16, e) &&
         Heads(o, Port::ValueCache, a.cache_capacity, a.kv_heads, a.head_width,
               ElementType::F16, e) &&
         Distinct(o, e, Port::Residual, Port::KeyCache, Port::ValueCache,
                  Port::Epoch, Port::Rows);
}
bool Head(const BoundOperands &o, const QwenEpochHeadAttrs &a, std::string *e) {
  if (!Positive(a.width) || !a.vocabulary_size ||
      a.vocabulary_size > uint64_t(INT32_MAX) + 1 || !a.token_capacity ||
      a.token_capacity > 512 || !a.span_capacity || a.span_capacity > 64 ||
      !a.selected_output_capacity || a.selected_output_capacity > 64 ||
      a.selected_output_capacity > a.span_capacity ||
      !std::isfinite(a.epsilon) || a.epsilon <= 0)
    return Fail(e, "invalid selected Qwen head configuration");
  return Matrix(o, Port::Input, a.token_capacity, a.width, ElementType::F32,
                Encoding::Dense, Layout::RowMajor, e) &&
         Vector(o, Port::Gamma, a.width, ElementType::F32, e) &&
         Quantized(o, Port::Weight, a.width, a.vocabulary_size, e) &&
         Epoch(o, 387, e) &&
         Vector(o, Port::Output, a.selected_output_capacity, ElementType::I32,
                e) &&
         Distinct(o, e, Port::Input, Port::Output, Port::Epoch, Port::Gamma,
                  Port::Weight);
}
DefinitionBuilder Common(OperationRegistry *r, OpCode op) {
  auto b = r->define(op);
  b.port(Port::Residual, ValueKind::Tensor, Access::ReadWrite)
      .port(Port::Epoch, ValueKind::Tensor, Access::Read)
      .port(Port::Rows, ValueKind::Tensor, Access::Read);
  for (auto p : {Port::InputNormWeight, Port::PostNormWeight, Port::GateWeight,
                 Port::UpWeight, Port::DownWeight})
    b.port(p, ValueKind::Tensor, Access::Read);
  return b;
}
} // namespace
const std::vector<QwenWeightRole> &QwenLayerWeightRoles(OpCode opcode) {
  static const std::vector<QwenWeightRole> common = {
      {Port::InputNormWeight, "input_norm", "attn_norm.weight"},
      {Port::PostNormWeight, "post_norm", "post_attention_norm.weight"},
      {Port::GateWeight, "ffn_gate", "ffn_gate.weight"},
      {Port::UpWeight, "ffn_up", "ffn_up.weight"},
      {Port::DownWeight, "ffn_down", "ffn_down.weight"}};
  static const auto gdn = [&] {
    auto roles = common;
    for (auto role : std::vector<QwenWeightRole>{
             {Port::QkvWeight, "recurrent_qkv", "attn_qkv.weight"},
             {Port::RecurrentGateWeight, "recurrent_gate", "attn_gate.weight"},
             {Port::ConvolutionWeight, "convolution", "ssm_conv1d.weight"},
             {Port::DtBias, "dt_bias", "ssm_dt.bias"},
             {Port::DecayWeight, "decay", "ssm_a"},
             {Port::AlphaWeight, "alpha", "ssm_alpha.weight"},
             {Port::BetaWeight, "beta", "ssm_beta.weight"},
             {Port::RecurrentNormWeight, "recurrent_norm", "ssm_norm.weight"},
             {Port::RecurrentOutputWeight, "recurrent_output",
              "ssm_out.weight"}})
      roles.push_back(role);
    return roles;
  }();
  static const auto attention = [&] {
    auto roles = common;
    for (auto role : std::vector<QwenWeightRole>{
             {Port::QueryWeight, "query", "attn_q.weight"},
             {Port::KeyWeight, "key", "attn_k.weight"},
             {Port::ValueWeight, "value", "attn_v.weight"},
             {Port::AttentionOutputWeight, "attention_output",
              "attn_output.weight"},
             {Port::QueryNormWeight, "query_norm", "attn_q_norm.weight"},
             {Port::KeyNormWeight, "key_norm", "attn_k_norm.weight"}})
      roles.push_back(role);
    return roles;
  }();
  static const std::vector<QwenWeightRole> empty;
  return opcode == OpCode::QwenGdnLayer         ? gdn
         : opcode == OpCode::QwenAttentionLayer ? attention
                                                : empty;
}
bool DefineQwenGdnLayer(OperationRegistry *r, std::string *e) {
  if (!Ready(r, e))
    return false;
  auto b = Common(r, OpCode::QwenGdnLayer);
  for (auto p :
       {Port::QkvWeight, Port::RecurrentGateWeight, Port::ConvolutionWeight,
        Port::DtBias, Port::DecayWeight, Port::AlphaWeight, Port::BetaWeight,
        Port::RecurrentNormWeight, Port::RecurrentOutputWeight})
    b.port(p, ValueKind::Tensor, Access::Read);
  return b.port(Port::ConvolutionHistory, ValueKind::Tensor, Access::ReadWrite)
      .port(Port::RecurrentState, ValueKind::Tensor, Access::ReadWrite)
      .attributes<QwenGdnLayerAttrs>()
      .verify(Gdn)
      .finish(e);
}
bool DefineQwenAttentionLayer(OperationRegistry *r, std::string *e) {
  if (!Ready(r, e))
    return false;
  auto b = Common(r, OpCode::QwenAttentionLayer);
  for (auto p : {Port::QueryWeight, Port::KeyWeight, Port::ValueWeight,
                 Port::AttentionOutputWeight, Port::QueryNormWeight,
                 Port::KeyNormWeight})
    b.port(p, ValueKind::Tensor, Access::Read);
  return b.port(Port::KeyCache, ValueKind::Tensor, Access::ReadWrite)
      .port(Port::ValueCache, ValueKind::Tensor, Access::ReadWrite)
      .attributes<QwenAttentionLayerAttrs>()
      .verify(Attention)
      .finish(e);
}
bool DefineQwenEpochHead(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::QwenEpochHead)
             .port(Port::Input, ValueKind::Tensor, Access::Read)
             .port(Port::Gamma, ValueKind::Tensor, Access::Read)
             .port(Port::Weight, ValueKind::Tensor, Access::Read)
             .port(Port::Epoch, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write)
             .attributes<QwenEpochHeadAttrs>()
             .verify(Head)
             .finish(e);
}
} // namespace hrx::operations
