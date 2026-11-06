// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include <map>
#include <set>
namespace hrx::dispatch {
using namespace operations;
loomc_status_t NativeEmissionSession::QwenEpoch(const NativeQwenEpoch &epoch) {
  auto error = [](const char *message) {
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
  };
  if (failed_ || started_ || has_llama_rows_ || has_qwen_epoch_ || one_row_ ||
      has_control_)
    return error("Qwen epoch must be configured once before emission");
  const auto &o = epoch.options;
  if (!o.token_capacity || o.token_capacity > 512 || o.span_capacity != 1 ||
      o.selected_output_capacity != 1 || o.row_capacity != 1 ||
      !o.cache_capacity || o.cache_capacity > 262144 ||
      o.pool_capacity != (o.cache_capacity + 63) / 64 * 64)
    return error("invalid Qwen native epoch bounds");
  auto tensor = [&](ValueId id, ElementType element,
                    std::initializer_list<uint64_t> dimensions) {
    const auto *value = graph_.value(id);
    const auto *type = value ? value->tensorType() : nullptr;
    if (!type || type->elementType() != element ||
        type->encoding() != Encoding::Dense ||
        type->layout() != Layout::RowMajor || type->rank() != dimensions.size())
      return false;
    size_t axis = 0;
    for (auto size : dimensions)
      if (type->dimension(axis++) != size)
        return false;
    return true;
  };
  if (epoch.logical_epoch == epoch.logical_rows ||
      !tensor(epoch.logical_epoch, ElementType::I32, {1048}) ||
      !tensor(epoch.logical_rows, ElementType::I8,
              {2048 + 4 * ((o.cache_capacity + 63) / 64)}) ||
      Find(epoch.logical_epoch) || Find(epoch.logical_rows))
    return error("invalid, foreign or prebound Qwen epoch/map identities");
  auto status = program_.InspectNative(
      {epoch.epoch, epoch.rows, epoch.resident_arena},
      [](const loom_module_t *module, const std::vector<loom_value_id_t> &ids) {
        for (auto id : ids)
          if (!loom_type_equal(loom_module_value_type(module, id),
                               loom_type_buffer()))
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "Qwen epoch boundary must be buffers");
        if (ids[0] == ids[1] || ids[0] == ids[2] || ids[1] == ids[2])
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "Qwen packet/map/arena must be distinct");
        return iree_ok_status();
      });
  if (!loomc_status_is_ok(status))
    return status;
  std::map<std::string, uint64_t> loaded;
  status =
      program_.InspectNative({}, [&](const loom_module_t *module,
                                     const std::vector<loom_value_id_t> &) {
        for (size_t i = 0; i < module->symbols.count; ++i) {
          const auto &symbol = module->symbols.entries[i];
          if (!symbol.defining_op || !loom_config_def_isa(symbol.defining_op))
            continue;
          const auto name =
              loom_string_table_get(&module->strings, symbol.name_id);
          const auto attr = loom_config_def_value(symbol.defining_op);
          if (attr.kind == LOOM_ATTR_I64 && attr.i64 >= 0)
            loaded.emplace(std::string(name.data, name.size),
                           uint64_t(attr.i64));
        }
        return iree_ok_status();
      });
  if (!loomc_status_is_ok(status))
    return status;
  for (auto [key, expected] : std::vector<std::pair<const char *, uint64_t>>{
           {"runner.qwen38.prefill_token_count", o.token_capacity},
           {"runner.qwen38.span_capacity", o.span_capacity},
           {"qwen38.attention.cache_capacity", o.cache_capacity},
           {"qwen38.attention.pool_capacity", o.pool_capacity},
           {"qwen.model.hidden_size", 1024},
           {"qwen.model.ffn_size", 3584},
           {"qwen.model.hidden_capacity_bytes", 2097152},
           {"qwen.model.gdn_history_bytes", 73728},
           {"qwen.model.gdn_recurrent_bytes", 1048576},
           {"qwen.model.gdn_layer_bytes", 1122304},
           {"qwen.model.gdn_layers", 18},
           {"qwen.model.group_count", 6},
           {"qwen.model.kv_plane_token_bytes", 1024},
           {"qwen.model.target_kv_planes", 12},
           {"qwen.model.projection_implementation", 1}}) {
    auto found = loaded.find(key);
    if (found == loaded.end() || found->second != expected)
      return error(
          "Qwen epoch bounds/geometry disagree with loaded configuration");
  }
  std::vector<QwenStateBinding> expected;
  size_t layer_slot = 0;
  for (const auto &invocation : graph_.invocations()) {
    const auto *gdn = std::get_if<QwenGdnLayerAttrs>(&invocation.attributes());
    const auto *attention =
        std::get_if<QwenAttentionLayerAttrs>(&invocation.attributes());
    const auto *head =
        std::get_if<QwenEpochHeadAttrs>(&invocation.attributes());
    if (!gdn && !attention && !head)
      continue;
    const auto *packet = invocation.binding(Port::Epoch);
    if (!packet || packet->value() != epoch.logical_epoch)
      return error("Qwen invocation packet differs from configured identity");
    if (head) {
      if (head->token_capacity != o.token_capacity ||
          head->span_capacity != o.span_capacity ||
          head->selected_output_capacity != o.selected_output_capacity)
        return error("Qwen head bounds disagree with configured epoch");
      continue;
    }
    const auto *rows = invocation.binding(Port::Rows);
    const uint32_t original = gdn ? gdn->layer_index : attention->layer_index;
    const uint64_t ordinal =
        gdn ? gdn->recurrent_ordinal : attention->attention_ordinal;
    if (!rows || rows->value() != epoch.logical_rows || original >= 32 ||
        ordinal >= (gdn ? 18u : 6u) ||
        (gdn ? gdn->token_capacity : attention->token_capacity) !=
            o.token_capacity ||
        (gdn ? gdn->span_capacity : attention->span_capacity) !=
            o.span_capacity ||
        (attention && attention->cache_capacity != o.cache_capacity))
      return error("Qwen layer identity/bounds disagree with configured epoch");
    auto add = [&](Port port, QwenStateRole role) {
      const auto *binding = invocation.binding(port);
      if (!binding)
        return false;
      const auto value = binding->value();
      if (Find(value) || value == epoch.logical_epoch ||
          value == epoch.logical_rows)
        return false;
      bool shape =
          role == QwenStateRole::GdnHistory
              ? tensor(value, ElementType::F32, {6144, 3})
          : role == QwenStateRole::GdnMatrix
              ? tensor(value, ElementType::F32, {16, 128, 128})
              : tensor(value, ElementType::F16, {o.cache_capacity, 2, 256});
      if (!shape)
        return false;
      for (const auto &state : expected)
        if (state.value == value ||
            (state.ordinal == ordinal && state.role == role))
          return false;
      expected.push_back({value, layer_slot, original, ordinal, role});
      return true;
    };
    if (!(gdn ? add(Port::ConvolutionHistory, QwenStateRole::GdnHistory) &&
                    add(Port::RecurrentState, QwenStateRole::GdnMatrix)
              : add(Port::KeyCache, QwenStateRole::Key) &&
                    add(Port::ValueCache, QwenStateRole::Value)))
      return error("Qwen layer state shape/identity is invalid or prebound");
    ++layer_slot;
  }
  if (expected.size() != epoch.states.size())
    return error("Qwen epoch must associate every graph state exactly once");
  for (const auto &state : expected) {
    size_t matches = 0;
    for (const auto &actual : epoch.states)
      matches += actual.value == state.value &&
                 actual.layer_slot == state.layer_slot &&
                 actual.original_layer_index == state.original_layer_index &&
                 actual.ordinal == state.ordinal && actual.role == state.role;
    if (matches != 1)
      return error("Qwen epoch state association disagrees with graph");
  }
  // All admission above is nonmutating. Only now install owner-derived
  // mappings.
  qwen_epoch_ = epoch;
  has_qwen_epoch_ = true;
  bindings_.push_back({epoch.logical_epoch, epoch.epoch, {}});
  bindings_.push_back({epoch.logical_rows, epoch.rows, {}});
  for (const auto &state : expected)
    bindings_.push_back({state.value, epoch.resident_arena, {}});
  return loomc_ok_status();
}
} // namespace hrx::dispatch
