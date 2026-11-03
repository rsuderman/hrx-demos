// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/attention_builders.h"
#include "loomc/iree.h"
#include <cmath>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
bool Supports(const Registration &entry, const InvocationRequest &r,
              std::string *error) {
  const bool rope = entry.operation == OpCode::Rope;
  const bool publish = entry.operation == OpCode::RotatePublishKV;
  const auto *input = r.Type(rope      ? Port::Input
                             : publish ? Port::Key
                                       : Port::Query);
  if (!input || input->rank() != 2 || !input->dimension(0) ||
      input->dimension(0) > 512)
    return Fail(error, "attention requires flattened full active rows");
  bool supported = false;
  if (rope) {
    const auto *a = std::get_if<RopeAttrs>(&r.attributes);
    supported =
        a && a->head_count == 24 && a->head_width == 128 &&
        a->rotary_width == 128 && a->base == 500000 && a->scaling_factor == 1 &&
        a->factor_mode == FactorMode::PerPair &&
        (a->scaling == RopeScaling::None || a->scaling == RopeScaling::Linear);
  } else if (publish) {
    const auto *a = std::get_if<RotatePublishKVAttrs>(&r.attributes);
    supported =
        a && a->kv_heads == 8 && a->key_width == 128 && a->value_width == 128 &&
        a->rotary_width == 128 && a->base == 500000 && a->scaling_factor == 1 &&
        a->factor_mode == FactorMode::PerPair && a->cache_capacity &&
        a->cache_capacity <= 131072 &&
        (a->scaling == RopeScaling::None || a->scaling == RopeScaling::Linear);
  } else {
    const auto *a = std::get_if<CausalAttentionAttrs>(&r.attributes);
    supported = a && a->query_heads == 24 && a->kv_heads == 8 &&
                a->key_width == 128 && a->value_width == 128 &&
                a->cache_capacity && a->cache_capacity <= 131072 &&
                static_cast<float>(a->scale) ==
                    static_cast<float>(1.0 / std::sqrt(128.0)) &&
                a->numerical_policy == NumericalPolicy::F32Accumulate;
  }
  return supported || Fail(error, "unsupported Llama attention geometry, "
                                  "rotary factors or numerical policy");
}
loomc_status_t Error(const char *s) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", s));
}
loomc_status_t Build(const Registration &entry, BuildContext &ctx, Value *out) {
  const bool rope = entry.operation == OpCode::Rope;
  const bool publish = entry.operation == OpCode::RotatePublishKV;
  const auto *input = ctx.Operand(rope      ? Port::Input
                                  : publish ? Port::Key
                                            : Port::Query);
  const auto rows = ctx.request.Type(input->port)->dimension(0);
  uint64_t capacity = 131072;
  if (publish)
    capacity =
        std::get<RotatePublishKVAttrs>(ctx.request.attributes).cache_capacity;
  else if (!rope)
    capacity =
        std::get<CausalAttentionAttrs>(ctx.request.attributes).cache_capacity;
  for (const auto &operand : ctx.inputs)
    if (!operand.parameter.empty() && operand.port != Port::Factors)
      return Error("attention projections, positions and cache state require "
                   "buffer bindings");
  KernelRef kernel;
  auto status = ctx.module.Kernel(entry.symbol, &kernel);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index_type = ctx.module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer = ctx.module.Buffer();
  std::vector<Type> specs(rope || publish ? 3 : 4, index_type);
  std::vector<Type> args{index_type};
  if (!rope && !publish)
    args.push_back(index_type);
  args.insert(args.end(), rope ? 4 : publish ? 6 : 5, buffer);
  status = ctx.module.CheckKernelInterface(kernel, specs, args);
  if (!loomc_status_is_ok(status))
    return status;
#define TRY(call)                                                              \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return s;                                                                \
  } while (0)
  ctx.staged = true;
  Value row_count, heads, cap, kvheads;
  TRY(ctx.values.Constant(rows, LOOM_SCALAR_TYPE_INDEX, &row_count));
  TRY(ctx.values.Constant(publish ? 8 : 24, LOOM_SCALAR_TYPE_INDEX, &heads));
  TRY(ctx.values.Constant(capacity, LOOM_SCALAR_TYPE_INDEX, &cap));
  std::vector<Value> constants{row_count, heads};
  if (!rope && !publish) {
    TRY(ctx.values.Constant(8, LOOM_SCALAR_TYPE_INDEX, &kvheads));
    constants.push_back(kvheads);
  }
  constants.push_back(cap);
  std::vector<Value> operands{cap};
  if (!rope && !publish)
    operands.push_back(kvheads);
  operands.push_back(ctx.Operand(Port::Positions)->native);
  if (rope || publish) {
    const auto *binding = ctx.Operand(Port::Factors);
    Value factors = binding->native;
    if (!binding->parameter.empty()) {
      Value count;
      TRY(ctx.values.Constant(64, LOOM_SCALAR_TYPE_INDEX, &count));
      TRY(ctx.values.Parameter(factors, binding->parameter, {},
                               LOOM_SCALAR_TYPE_F32, {count}, &factors));
    }
    operands.push_back(factors);
  }
  operands.push_back(input->native);
  if (publish)
    operands.push_back(ctx.Operand(Port::Value)->native);
  if (!rope) {
    operands.push_back(ctx.Operand(Port::KeyCache)->native);
    operands.push_back(ctx.Operand(Port::ValueCache)->native);
  }
  Value output = ctx.output;
  if (!publish) {
    if (!ctx.has_output)
      TRY(ctx.Allocate(rows * 3072 * 4, &output));
    operands.push_back(output);
  }
  TRY(ctx.program.Invoke(kernel, constants, operands));
  if (!publish)
    *out = output;
  return loomc_ok_status();
#undef TRY
}
} // namespace
OperationBuilder AttentionBuilder(OpCode operation) {
  std::vector<OperandContract> operands{{Port::Positions, Access::Read}};
  if (operation == OpCode::Rope) {
    operands.insert(operands.end(), {{Port::Factors, Access::Read},
                                     {Port::Input, Access::Read},
                                     {Port::Output, Access::Write}});
  } else if (operation == OpCode::RotatePublishKV) {
    operands.insert(operands.end(), {{Port::Factors, Access::Read},
                                     {Port::Key, Access::Read},
                                     {Port::Value, Access::Read},
                                     {Port::KeyCache, Access::ReadWrite},
                                     {Port::ValueCache, Access::ReadWrite}});
  } else {
    operands.insert(operands.end(), {{Port::Query, Access::Read},
                                     {Port::KeyCache, Access::Read},
                                     {Port::ValueCache, Access::Read},
                                     {Port::Output, Access::Write}});
  }
  return {operation, Kind::Kernel, std::move(operands), Supports, Build};
}
std::vector<Registration> AttentionRegistrations() {
  std::vector<Registration> result;
  for (auto id : {Id::Rope, Id::RotatePublishKV, Id::CausalAttention}) {
    const bool rope = id == Id::Rope, publish = id == Id::RotatePublishKV;
    const auto opcode = rope      ? OpCode::Rope
                        : publish ? OpCode::RotatePublishKV
                                  : OpCode::CausalAttention;
    const char *symbol = rope      ? "llama_dispatch_rope"
                         : publish ? "llama_dispatch_publish_kv"
                                   : "llama_dispatch_causal_attention";
    const std::string root = "lib/kernels/llama/";
    Registration entry{id,
                       opcode,
                       Kind::Kernel,
                       rope      ? "llama.rope"
                       : publish ? "llama.publish_kv"
                                 : "llama.causal_attention",
                       symbol,
                       root + "attention/attention_interface.loom",
                       {root + "attention/attention_dispatch.loom",
                        root + "attention/rotary.loom",
                        root + "attention/attention_core.loom",
                        root + "attention/attention_map.loom"},
                       {},
                       {}};
    entry.interface_symbol = symbol;
    entry.builder = AttentionBuilder(opcode);
    entry.operands = entry.builder.operands;
    result.push_back(std::move(entry));
  }
  return result;
}
} // namespace hrx::dispatch
