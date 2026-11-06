// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen_gdn.h"
#include "lib/dispatch/qwen.h"
#include "lib/operations/qwen_operations.h"
#include "loomc/iree.h"
#include <algorithm>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
bool Fail(std::string *e, const char *s) {
  if (e)
    *e = s;
  return false;
}
loomc_status_t Error(const char *s) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", s));
}
bool Shape(const InvocationRequest &r, Port p,
           std::initializer_list<uint64_t> dimensions, ElementType element,
           Encoding encoding = Encoding::Dense,
           Layout layout = Layout::RowMajor) {
  const auto *t = r.Type(p);
  if (!t || t->rank() != dimensions.size() || t->elementType() != element ||
      t->encoding() != encoding || t->layout() != layout)
    return false;
  size_t i = 0;
  for (auto d : dimensions)
    if (t->dimension(i++) != d)
      return false;
  return true;
}
bool Weight(const InvocationRequest &r, Port p, uint64_t in, uint64_t out,
            Encoding encoding) {
  return Shape(r, p, {in, out}, ElementType::F32, encoding,
               Layout::GgmlWeightRows);
}
bool Supports(const Registration &, const InvocationRequest &r,
              std::string *e) {
  const auto *a = std::get_if<QwenGdnLayerAttrs>(&r.attributes);
  if (!a || a->hidden_width != 1024 || a->ffn_width != 3584 ||
      a->state_width != 128 || a->key_heads != 16 || a->value_heads != 16 ||
      a->convolution_taps != 4 || a->span_capacity != 1 || !a->token_capacity ||
      a->token_capacity > 512 || a->layer_index >= 24 ||
      a->recurrent_ordinal >= 18 || float(a->epsilon) != 1e-6f)
    return Fail(e,
                "unsupported ordinary Qwen0.8B GDN geometry/epsilon/capacity");
  if (!Shape(r, Port::Residual, {a->token_capacity, 1024}, ElementType::F32) ||
      !Shape(r, Port::InputNormWeight, {1024}, ElementType::F32) ||
      !Shape(r, Port::PostNormWeight, {1024}, ElementType::F32) ||
      !Weight(r, Port::QkvWeight, 1024, 6144, Encoding::Q6_K) ||
      !Weight(r, Port::RecurrentGateWeight, 1024, 2048, Encoding::Q6_K) ||
      !Weight(r, Port::AlphaWeight, 1024, 16, Encoding::Q5_K) ||
      !Weight(r, Port::BetaWeight, 1024, 16, Encoding::Q5_K) ||
      !Weight(r, Port::RecurrentOutputWeight, 2048, 1024, Encoding::Q5_K) ||
      !Shape(r, Port::ConvolutionWeight, {6144, 4}, ElementType::F32) ||
      !Shape(r, Port::DtBias, {16}, ElementType::F32) ||
      !Shape(r, Port::DecayWeight, {16}, ElementType::F32) ||
      !Shape(r, Port::RecurrentNormWeight, {128}, ElementType::F32) ||
      !Shape(r, Port::ConvolutionHistory, {6144, 3}, ElementType::F32) ||
      !Shape(r, Port::RecurrentState, {16, 128, 128}, ElementType::F32))
    return Fail(e, "unsupported Qwen GDN role shape, encoding or state layout");
  const auto *gate = r.Type(Port::GateWeight), *up = r.Type(Port::UpWeight),
             *down = r.Type(Port::DownWeight);
  if (!gate || !up || !down || gate->encoding() != up->encoding())
    return Fail(e, "Qwen FFN gate/up must use matching admitted encodings");
  auto quantized = [](Encoding x) {
    return x == Encoding::Q4_K || x == Encoding::Q5_K || x == Encoding::Q6_K;
  };
  if (!quantized(gate->encoding()) ||
      (down->encoding() != Encoding::Q5_K &&
       down->encoding() != Encoding::Q6_K) ||
      !Weight(r, Port::GateWeight, 1024, 3584, gate->encoding()) ||
      !Weight(r, Port::UpWeight, 1024, 3584, up->encoding()) ||
      !Weight(r, Port::DownWeight, 3584, 1024, down->encoding()))
    return Fail(e, "unsupported canonical mixed Qwen FFN encoding/shape");
  const auto *epoch = r.Type(Port::Epoch), *rows = r.Type(Port::Rows);
  if (!epoch || epoch->rank() != 1 ||
      epoch->elementType() != ElementType::I32 ||
      epoch->encoding() != Encoding::Dense ||
      epoch->layout() != Layout::RowMajor || epoch->dimension(0) < 1048 ||
      !rows || rows->rank() != 1 || rows->elementType() != ElementType::I8 ||
      rows->encoding() != Encoding::Dense ||
      rows->layout() != Layout::RowMajor || rows->dimension(0) < 2048)
    return Fail(e,
                "Qwen GDN requires complete epoch and resident-row metadata");
  return true;
}
loomc_status_t Build(const Registration &entry, BuildContext &ctx, Value *out) {
  const auto *a = std::get_if<QwenGdnLayerAttrs>(&ctx.request.attributes);
  const auto *residual = ctx.Operand(Port::Residual);
  const auto *history = ctx.Operand(Port::ConvolutionHistory);
  const auto *matrix = ctx.Operand(Port::RecurrentState);
  const auto *packet = ctx.Operand(Port::Epoch);
  const auto *rows = ctx.Operand(Port::Rows);
  if (!a || !ctx.qwen_epoch || !residual || !history || !matrix || !packet ||
      !rows || !residual->parameter.empty() || !ctx.request.result.has_result ||
      ctx.request.result.alias != residual->logical)
    return Error(
        "Qwen GDN requires configured epoch and explicit residual alias");
  const auto &epoch = *ctx.qwen_epoch;
  if (packet->logical != epoch.logical_epoch ||
      rows->logical != epoch.logical_rows)
    return Error(
        "Qwen GDN epoch/rows identities disagree with configured frame");
  if (epoch.options.token_capacity != a->token_capacity ||
      epoch.options.span_capacity != a->span_capacity)
    return Error("Qwen GDN invocation and epoch capacities differ");
  size_t layer_slot = 0;
  bool found_layer = false;
  for (const auto &invocation : ctx.request.graph.invocations()) {
    const auto *gdn = std::get_if<QwenGdnLayerAttrs>(&invocation.attributes());
    const auto *attention =
        std::get_if<QwenAttentionLayerAttrs>(&invocation.attributes());
    if (!gdn && !attention)
      continue;
    if (gdn && gdn->layer_index == a->layer_index &&
        gdn->recurrent_ordinal == a->recurrent_ordinal) {
      found_layer = true;
      break;
    }
    ++layer_slot;
  }
  if (!found_layer)
    return Error("Qwen GDN layer does not belong to the configured graph");
  bool history_bound = false, matrix_bound = false;
  for (const auto &state : epoch.states) {
    if (state.original_layer_index != a->layer_index ||
        state.ordinal != a->recurrent_ordinal || state.layer_slot != layer_slot)
      continue;
    history_bound |= state.role == QwenStateRole::GdnHistory &&
                     state.value == history->logical;
    matrix_bound |= state.role == QwenStateRole::GdnMatrix &&
                    state.value == matrix->logical;
  }
  if (!history_bound || !matrix_bound)
    return Error(
        "Qwen GDN history/matrix identities do not match layer ordinal");
  std::vector<std::pair<Port, std::string>> roles;
  for (const auto &role : QwenLayerWeightRoles(OpCode::QwenGdnLayer))
    roles.push_back({role.port, "blk." + std::to_string(a->layer_index) + "." +
                                    role.suffix});
  Value parameters;
  auto status = CheckQwenLayerParameters(ctx, roles, &parameters);
  if (!loomc_status_is_ok(status))
    return status;
  status = ctx.program.InspectNative(
      {parameters, residual->native, epoch.resident_arena, epoch.epoch,
       epoch.rows, packet->native, rows->native, history->native,
       matrix->native},
      [](const loom_module_t *module, const std::vector<loom_value_id_t> &ids) {
        for (auto id : ids)
          if (!loom_type_equal(loom_module_value_type(module, id),
                               loom_type_buffer()))
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "Qwen GDN configured frame requires native buffers");
        if (ids[3] != ids[5] || ids[4] != ids[6] || ids[2] != ids[7] ||
            ids[2] != ids[8])
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "Qwen GDN native operands disagree with configured frame");
        for (size_t i = 0; i < 5; ++i)
          for (size_t j = i + 1; j < 5; ++j)
            if (ids[i] == ids[j])
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "Qwen GDN boundary buffers must be distinct");
        return iree_ok_status();
      });
  if (!loomc_status_is_ok(status))
    return status;
  status = CheckQwenLayerConfiguration(ctx);
  if (!loomc_status_is_ok(status))
    return status;
  ProgramRef layer;
  status = ctx.module.Program(entry.symbol, &layer);
  if (!loomc_status_is_ok(status))
    return status;
  const auto index = ctx.module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer = ctx.module.Buffer();
  status = ctx.module.CheckProgramInterface(
      layer, {index, index, index, index, index, index},
      {buffer, buffer, buffer, buffer, buffer, buffer});
  if (!loomc_status_is_ok(status))
    return status;
  ctx.staged = true;
  std::vector<Value> settings;
  for (uint64_t n :
       {a->token_capacity, a->token_capacity, uint64_t(a->layer_index),
        a->recurrent_ordinal, uint64_t(0), a->span_capacity}) {
    Value value;
    status = ctx.values.Constant(n, LOOM_SCALAR_TYPE_INDEX, &value);
    if (!loomc_status_is_ok(status))
      return status;
    settings.push_back(value);
  }
  Value capture;
  // Verification depth0 never consumes capture; retain the actual callable ABI.
  status = ctx.Allocate(256, &capture);
  if (!loomc_status_is_ok(status))
    return status;
  status = ctx.program.Call(layer, settings,
                            {parameters, residual->native, epoch.resident_arena,
                             epoch.epoch, capture, epoch.rows});
  if (!loomc_status_is_ok(status))
    return status;
  *out = residual->native;
  return loomc_ok_status();
}
template <class T> void AppendUnique(std::vector<T> *v, const T &x) {
  if (std::find(v->begin(), v->end(), x) == v->end())
    v->push_back(x);
}
} // namespace
std::vector<Registration> QwenGdnRegistrations() {
  const std::string root = "lib/kernels/qwen/gdn/";
  OperationBuilder builder{OpCode::QwenGdnLayer,
                           Kind::CommandProgram,
                           {{Port::Residual, Access::ReadWrite},
                            {Port::ConvolutionHistory, Access::ReadWrite},
                            {Port::RecurrentState, Access::ReadWrite},
                            {Port::Epoch, Access::Read},
                            {Port::Rows, Access::Read}},
                           Supports,
                           Build,
                           {"qwen38_gdn_layer_prefill", "qwen38_gdn_conv_spans",
                            "qwen38_gdn_recurrent_spans",
                            "qwen08_gdn_clear_raw"}};
  for (const auto &role : QwenLayerWeightRoles(OpCode::QwenGdnLayer))
    builder.operands.push_back({role.port, Access::Read});
  Registration entry{Id::QwenGdnLayer,
                     OpCode::QwenGdnLayer,
                     Kind::CommandProgram,
                     "qwen08.gdn_layer",
                     "qwen38_gdn_layer_prefill",
                     root + "interface.loom",
                     {root + "layer.loom", root + "common.loom",
                      root + "prefill.loom", root + "spans.loom",
                      root + "clear.loom"},
                     builder.operands,
                     {},
                     "ordinary",
                     Phase::Prefill,
                     "qwen38_gdn_layer_prefill",
                     builder};
  for (const auto &source : QwenSharedSources()) {
    AppendUnique(&entry.provider_files, source.interface_file);
    for (const auto &p : source.provider_files)
      AppendUnique(&entry.provider_files, p);
    for (const auto &c : source.required_config)
      AppendUnique(&entry.required_config, c);
  }
  // Exact native requirements of the curated stateful core and layer.
  for (const char *name : {"gdn_capture_qkv_elements",
                           "gdn_column_tiles",
                           "gdn_conv_elements",
                           "gdn_convolution_taps",
                           "gdn_divisor",
                           "gdn_gate_row_bytes",
                           "gdn_head_size",
                           "gdn_history_elements",
                           "gdn_history_values",
                           "gdn_inverse_row_bytes",
                           "gdn_key_heads",
                           "gdn_q_channels",
                           "gdn_qk_norm_elements",
                           "gdn_qkv_size",
                           "gdn_state_elements",
                           "gdn_v_base",
                           "gdn_value_channels_per_key_head",
                           "gdn_value_heads",
                           "gdn_width",
                           "projection_row_bytes",
                           "hidden_size",
                           "gdn_qkv_weight_bytes",
                           "gdn_gate_weight_bytes",
                           "gdn_alpha_weight_bytes",
                           "gdn_beta_weight_bytes",
                           "projection_weight_bytes",
                           "gdn_recurrent_bytes",
                           "gdn_history_bytes",
                           "hidden_capacity_bytes",
                           "hidden_q8_capacity_bytes",
                           "qkv_capacity_bytes",
                           "projection_capacity_bytes",
                           "gdn_gate_capacity_bytes",
                           "ffn_capacity_bytes"})
    AppendUnique(&entry.required_config, std::string("qwen.model.") + name);
  return {std::move(entry)};
}
} // namespace hrx::dispatch
