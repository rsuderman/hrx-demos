// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/qwen/program.h"
#include "lib/dispatch/invocation.h"
#include <algorithm>
#include <limits>
namespace hrx::qwen {
using namespace operations;
using namespace loomify_gguf;
std::vector<dispatch::Id> QwenProgramDispatches() {
  return {dispatch::Id::Embedding, dispatch::Id::QwenGdnLayer,
          dispatch::Id::QwenAttentionLayer, dispatch::Id::QwenEpochHead};
}
bool DefineQwenProgram(const model::ModelDescriptor &facts,
                       const QwenProgramSettings &settings,
                       const dispatch::Registry &registry,
                       ModuleBuilder &module, const std::string &name,
                       QwenProgram *out, std::string *error) {
  auto fail = [&](const std::string &message) {
    if (error)
      *error = message;
    return false;
  };
  if (!out || !error || name.empty())
    return fail("Qwen program requires output, name and error");
  QwenProgram candidate;
  if (!dispatch::DeriveQwenNativeGeometry(facts, settings.native,
                                          &candidate.geometry, error))
    return false;
  const auto &options = settings.native;
  if (!BuildQwenStructure(facts,
                          {options.token_capacity, options.span_capacity,
                           options.selected_output_capacity,
                           options.cache_capacity, settings.source_asset},
                          &candidate.structure, error))
    return false;
  std::unique_ptr<CommandProgramBuilder> program;
  auto ok = [&](loomc_status_t status) {
    bool good = loomc_status_is_ok(status);
    if (!good) {
      auto message = loomc_status_message(status);
      error->assign(message.data, message.size);
    }
    loomc_status_free(status);
    return good;
  };
  if (!ok(module.DefineProgram(name, {}, 7, &program)))
    return false;
  const auto &owner = *candidate.structure;
  const auto &inputs = owner.inputs();
  dispatch::NativeEmissionSession session(registry, owner.graph(), module,
                                          *program);
  dispatch::NativeQwenEpoch epoch;
  epoch.logical_epoch = inputs.epoch;
  epoch.logical_rows = inputs.rows;
  epoch.epoch = program->Binding(2);
  epoch.rows = program->Binding(3);
  epoch.resident_arena = program->Binding(4);
  epoch.options = options;
  for (const auto &layer : inputs.layers) {
    auto add = [&](ValueId value, dispatch::QwenStateRole role) {
      epoch.states.push_back({value, layer.layer_slot,
                              layer.original_layer_index, layer.ordinal, role});
    };
    if (layer.kind == model::LayerKind::kGatedDeltaNet) {
      add(layer.history, dispatch::QwenStateRole::GdnHistory);
      add(layer.recurrent, dispatch::QwenStateRole::GdnMatrix);
    } else {
      add(layer.key_cache, dispatch::QwenStateRole::Key);
      add(layer.value_cache, dispatch::QwenStateRole::Value);
    }
  }
  if (!ok(session.QwenEpoch(epoch)) ||
      !ok(session.Bind(inputs.tokens, program->Binding(5))) ||
      !ok(session.Bind(inputs.residual, program->Binding(1))) ||
      !ok(session.Bind(inputs.output, program->Binding(6))))
    return false;
  std::vector<ValueId> bound;
  for (const auto &association : owner.parameters()) {
    if (std::find(bound.begin(), bound.end(), association.value) != bound.end())
      continue;
    const auto *parameter = owner.parameter(association.coordinate);
    if (!parameter)
      return fail("Qwen owner lost parameter association");
    if (!ok(session.Parameter(association.value, program->Binding(0),
                              parameter->tensor.name)))
      return false;
    bound.push_back(association.value);
  }
  // Consume the recorded architecture, never reconstruct a second model loop.
  for (const auto &invocation : owner.graph().invocations()) {
    dispatch::Id id;
    switch (invocation.opcode()) {
    case OpCode::Embedding:
      id = dispatch::Id::Embedding;
      break;
    case OpCode::QwenGdnLayer:
      id = dispatch::Id::QwenGdnLayer;
      break;
    case OpCode::QwenAttentionLayer:
      id = dispatch::Id::QwenAttentionLayer;
      break;
    case OpCode::QwenEpochHead:
      id = dispatch::Id::QwenEpochHead;
      break;
    default:
      return fail("unexpected operation in complete Qwen architecture");
    }
    if (!ok(session.EmitExisting(id, invocation)))
      return false;
  }
  if (!ok(program->Finish()))
    return false;
  const auto &geometry = candidate.geometry;
  // Geometry was admitted by the common native loader/session, including
  // physical pooled strides and native residual footprint independently of T.
  const auto max = uint64_t(INT64_MAX);
  if (geometry.private_row_stride > max / options.row_capacity ||
      geometry.kv_pool_bytes >
          max - geometry.private_row_stride * options.row_capacity ||
      options.cache_capacity > max - 63)
    return fail("Qwen external buffer requirement overflow");
  const auto blocks = (options.cache_capacity + 63) / 64;
  if (blocks > max / 4 / options.row_capacity ||
      geometry.map_origin > max - 4 * blocks * options.row_capacity)
    return fail("Qwen map buffer requirement overflow");
  candidate.program = program->Ref();
  candidate.buffers = {
      {"parameters", 0},
      {"residual", geometry.residual_bytes},
      {"epoch", 4192},
      {"rows", geometry.map_origin + 4 * blocks * options.row_capacity},
      {"resident_arena", geometry.private_row_stride * options.row_capacity +
                             geometry.kv_pool_bytes},
      {"input_ids", 4 * std::max<uint64_t>(512, options.token_capacity)},
      {"output_ids", 4 * options.selected_output_capacity}};
  *out = std::move(candidate);
  error->clear();
  return true;
}
} // namespace hrx::qwen
