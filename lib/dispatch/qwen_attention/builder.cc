// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen_attention/builder.h"
#include "lib/dispatch/qwen.h"
#include "lib/operations/qwen_operations.h"
#include "loomc/iree.h"
#include <algorithm>
#include <array>

namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
loomc_status_t Error(const char *message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
}
bool Supports(const Registration &, const InvocationRequest &request,
              std::string *error) {
  const auto *a = std::get_if<QwenAttentionLayerAttrs>(&request.attributes);
  if (!a || a->layer_index >= 24 || a->attention_ordinal >= 6 ||
      !a->token_capacity || a->token_capacity > 512 || a->span_capacity != 1 ||
      a->hidden_width != 1024 || a->ffn_width != 3584 || a->query_heads != 8 ||
      a->kv_heads != 2 || a->head_width != 256 || a->rotary_width != 64 ||
      !a->cache_capacity || a->cache_capacity > 262144 ||
      static_cast<float>(a->epsilon) != 1e-6f || a->rope_base != 10000000 ||
      a->rope_scaling_factor != 1 ||
      (a->rope_scaling != RopeScaling::None &&
       a->rope_scaling != RopeScaling::Linear) ||
      a->rope_sections != std::array<uint64_t, 4>{11, 11, 10, 0} ||
      static_cast<float>(a->attention_scale) != 0.0625f)
    return Fail(
        error,
        "unsupported Qwen0.8B attention geometry/epsilon/rotary profile");
  const auto *q = request.Type(Port::QueryWeight);
  const auto *k = request.Type(Port::KeyWeight);
  const auto *v = request.Type(Port::ValueWeight);
  const auto *o = request.Type(Port::AttentionOutputWeight);
  const auto *gate = request.Type(Port::GateWeight);
  const auto *up = request.Type(Port::UpWeight);
  const auto *down = request.Type(Port::DownWeight);
  if (!q || !k || !v || !o || !gate || !up || !down ||
      (q->encoding() != Encoding::Q5_K && q->encoding() != Encoding::Q6_K) ||
      k->encoding() != q->encoding() || v->encoding() != Encoding::Q6_K ||
      o->encoding() != Encoding::Q5_K || gate->encoding() != up->encoding() ||
      (gate->encoding() != Encoding::Q4_K &&
       gate->encoding() != Encoding::Q5_K &&
       gate->encoding() != Encoding::Q6_K) ||
      (down->encoding() != Encoding::Q5_K &&
       down->encoding() != Encoding::Q6_K))
    return Fail(error,
                "unsupported Qwen attention projection/FFN weight encodings");
  ValueId residual;
  for (const auto &input : request.inputs)
    if (input.port == Port::Residual)
      residual = input.value;
  if (!request.result.has_result || request.result.port != Port::Residual ||
      request.result.access != Access::ReadWrite ||
      request.result.alias != residual || !residual.valid())
    return Fail(error,
                "Qwen attention must return its explicit residual alias");
  return true;
}
loomc_status_t Build(const Registration &entry, BuildContext &ctx, Value *out) {
  const auto &a = std::get<QwenAttentionLayerAttrs>(ctx.request.attributes);
  if (!out || !ctx.qwen_epoch)
    return Error("Qwen attention requires a configured epoch");
  const auto &epoch = *ctx.qwen_epoch;
  if (epoch.options.token_capacity != a.token_capacity ||
      epoch.options.span_capacity != a.span_capacity ||
      epoch.options.cache_capacity != a.cache_capacity ||
      !epoch.options.pool_capacity)
    return Error("Qwen attention attributes disagree with configured epoch");
  const auto *residual = ctx.Operand(Port::Residual);
  const auto *metadata = ctx.Operand(Port::Epoch);
  const auto *rows = ctx.Operand(Port::Rows);
  if (!residual || !metadata || !rows ||
      metadata->logical != epoch.logical_epoch ||
      rows->logical != epoch.logical_rows)
    return Error("Qwen attention requires its declared epoch/row identities");
  auto native_status = ctx.program.InspectNative(
      {metadata->native, epoch.epoch, rows->native, epoch.rows,
       residual->native, epoch.resident_arena},
      [](const loom_module_t *module, const std::vector<loom_value_id_t> &ids) {
        for (auto id : ids)
          if (!loom_type_equal(loom_module_value_type(module, id),
                               loom_type_buffer()))
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "Qwen attention native boundary must be buffers");
        if (ids[0] != ids[1] || ids[2] != ids[3])
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Qwen attention epoch/rows native mapping disagrees");
        return iree_ok_status();
      });
  if (!loomc_status_is_ok(native_status))
    return native_status;
  auto configuration_status = CheckQwenLayerConfiguration(ctx);
  if (!loomc_status_is_ok(configuration_status))
    return configuration_status;
  std::vector<std::pair<Port, std::string>> roles;
  const std::string prefix = "blk." + std::to_string(a.layer_index) + ".";
  for (const auto &role : QwenLayerWeightRoles(OpCode::QwenAttentionLayer))
    roles.emplace_back(role.port, prefix + role.suffix);
  Value parameters;
  auto status = CheckQwenLayerParameters(ctx, roles, &parameters);
  if (!loomc_status_is_ok(status))
    return status;
  const auto *keys = ctx.Operand(Port::KeyCache);
  const auto *values = ctx.Operand(Port::ValueCache);
  if (!keys || !values)
    return Error("Qwen attention requires both declared cache states");
  const auto key_state = std::find_if(
      epoch.states.begin(), epoch.states.end(),
      [&](const auto &state) { return state.value == keys->logical; });
  const auto value_state = std::find_if(
      epoch.states.begin(), epoch.states.end(),
      [&](const auto &state) { return state.value == values->logical; });
  if (key_state == epoch.states.end() || value_state == epoch.states.end() ||
      key_state->role != QwenStateRole::Key ||
      value_state->role != QwenStateRole::Value ||
      key_state->original_layer_index != a.layer_index ||
      value_state->original_layer_index != a.layer_index ||
      key_state->ordinal != a.attention_ordinal ||
      value_state->ordinal != a.attention_ordinal ||
      key_state->layer_slot != value_state->layer_slot)
    return Error(
        "Qwen attention cache identities disagree with epoch associations");
  status = ctx.program.InspectNative(
      {parameters, residual->native, epoch.epoch, epoch.rows,
       epoch.resident_arena, keys->native, values->native},
      [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
        if (ids[4] != ids[5] || ids[4] != ids[6])
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "Qwen cache states must use the epoch arena");
        for (size_t i = 0; i < 5; ++i)
          for (size_t j = i + 1; j < 5; ++j)
            if (ids[i] == ids[j])
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "Qwen layer boundary buffers must be distinct");
        return iree_ok_status();
      });
  if (!loomc_status_is_ok(status))
    return status;
  ProgramRef program;
  status = ctx.module.Program(entry.symbol, &program);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index = ctx.module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  status = ctx.module.CheckProgramInterface(
      program, std::vector<Type>(6, index),
      std::vector<Type>(6, ctx.module.Buffer()));
  if (!loomc_status_is_ok(status))
    return status;
  // All knowable profile, identity, key/root and ABI facts were checked before
  // constants/storage staging. The native command owns every numerical child
  // and the R1 full-T attended-output initialization.
#define STAGE(call)                                                            \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return s;                                                                \
  } while (0)
  Value tokens, layer, ordinal, zero, spans, attended;
  STAGE(ctx.Allocate(512ull * 2048 * sizeof(float), &attended));
  STAGE(ctx.values.Constant(a.token_capacity, LOOM_SCALAR_TYPE_INDEX, &tokens));
  STAGE(ctx.values.Constant(a.layer_index, LOOM_SCALAR_TYPE_INDEX, &layer));
  STAGE(ctx.values.Constant(a.attention_ordinal, LOOM_SCALAR_TYPE_INDEX,
                            &ordinal));
  STAGE(ctx.values.Constant(0, LOOM_SCALAR_TYPE_INDEX, &zero));
  STAGE(ctx.values.Constant(a.span_capacity, LOOM_SCALAR_TYPE_INDEX, &spans));
  STAGE(ctx.program.Call(program, {tokens, tokens, layer, ordinal, zero, spans},
                         {parameters, residual->native, epoch.resident_arena,
                          epoch.epoch, attended, epoch.rows}));
  *out = residual->native;
#undef STAGE
  return loomc_ok_status();
}
std::vector<std::string> RequiredConfig() {
  return {"qwen.model.hidden_size",
          "qwen.model.attention_width",
          "qwen.model.kv_size",
          "qwen.model.query_gate_size",
          "qwen.model.head_size",
          "qwen.model.rms_epsilon",
          "qwen.model.attention_q6_layers",
          "qwen.model.query_q5_weight_bytes",
          "qwen.model.query_q6_weight_bytes",
          "qwen.model.key_q5_weight_bytes",
          "qwen.model.key_q6_weight_bytes",
          "qwen.model.value_weight_bytes",
          "qwen.model.projection_weight_bytes",
          "qwen.model.hidden_capacity_bytes",
          "qwen.model.hidden_q8_capacity_bytes",
          "qwen.model.query_gate_capacity_bytes",
          "qwen.model.kv_capacity_bytes",
          "qwen.model.projection_capacity_bytes",
          "qwen.model.ffn_capacity_bytes",
          "qwen.model.kv_plane_token_bytes",
          "qwen.model.attention_heads",
          "qwen.model.kv_heads",
          "qwen.model.head_gate_size",
          "qwen.model.head_scratch_bytes",
          "qwen.model.head_divisor",
          "qwen.model.kv_row_bytes",
          "qwen.model.query_gate_row_bytes",
          "qwen.model.rope_half",
          "qwen.model.rope_size",
          "qwen.model.rope_log_step",
          "qwen.model.attention_scale",
          "qwen38.attention.cache_capacity",
          "qwen38.attention.pool_capacity"};
}
} // namespace
OperationBuilder QwenAttentionLayerBuilder() {
  std::vector<OperandContract> operands{{Port::Residual, Access::ReadWrite}};
  for (const auto &role : QwenLayerWeightRoles(OpCode::QwenAttentionLayer))
    operands.push_back({role.port, Access::Read});
  operands.insert(operands.end(), {{Port::Epoch, Access::Read},
                                   {Port::Rows, Access::Read},
                                   {Port::KeyCache, Access::ReadWrite},
                                   {Port::ValueCache, Access::ReadWrite}});
  return {OpCode::QwenAttentionLayer,
          Kind::CommandProgram,
          std::move(operands),
          Supports,
          Build,
          {"qwen38_attention_clear_attended",
           "qwen38_attention_query_prepare_spans",
           "qwen38_attention_kv_prepare_spans",
           "qwen38_attention_spans_f16_wmma"}};
}
std::vector<Registration> QwenAttentionRegistrations() {
  const std::string root = "lib/kernels/qwen/attention/";
  auto builder = QwenAttentionLayerBuilder();
  Registration entry{
      Id::QwenAttentionLayer,
      OpCode::QwenAttentionLayer,
      Kind::CommandProgram,
      "qwen.08b.attention.layer",
      "qwen38_attention_layer_prefill",
      root + "layer_interface.loom",
      {root + "layer.loom", root + "clear.loom", root + "attention_common.loom",
       root + "attention_prefill.loom", root + "attention_prefill_wmma.loom",
       root + "attention_spans.loom"},
      builder.operands,
      RequiredConfig()};
  entry.interface_symbol = entry.symbol;
  entry.builder = std::move(builder);
  for (const auto &source : QwenSharedSources()) {
    for (const auto &file : source.provider_files)
      if (std::find(entry.provider_files.begin(), entry.provider_files.end(),
                    file) == entry.provider_files.end())
        entry.provider_files.push_back(file);
    if (!source.interface_file.empty() &&
        std::find(entry.provider_files.begin(), entry.provider_files.end(),
                  source.interface_file) == entry.provider_files.end())
      entry.provider_files.push_back(source.interface_file);
    for (const auto &definition : source.required_definitions)
      if (std::find(entry.builder.required_definitions.begin(),
                    entry.builder.required_definitions.end(),
                    definition) == entry.builder.required_definitions.end())
        entry.builder.required_definitions.push_back(definition);
    for (const auto &key : source.required_config)
      if (std::find(entry.required_config.begin(), entry.required_config.end(),
                    key) == entry.required_config.end())
        entry.required_config.push_back(key);
  }
  return {std::move(entry)};
}
} // namespace hrx::dispatch
