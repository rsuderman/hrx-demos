// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/program.h"
#include "lib/dispatch/parameters.h"
#include <algorithm>
#include <limits>
namespace hrx::llama {
using namespace operations;
using namespace loomify_gguf;
std::vector<dispatch::Id> LlamaProgramDispatches() {
  using dispatch::Id;
  return {Id::Embedding,       Id::RmsNorm,       Id::Projection,
          Id::AttentionOutput, Id::GateUpPrefill, Id::GateUpDecode,
          Id::DownPrefill,     Id::DownDecode,    Id::OutputHead,
          Id::PublishPrefill,  Id::PublishDecode};
}
bool DefineLlamaProgram(const model::ModelDescriptor &facts,
                        const LlamaProgramSettings &settings,
                        const dispatch::Registry &registry,
                        ModuleBuilder &module, const std::string &name,
                        LlamaProgram *out, std::string *error) {
  auto fail = [&](const std::string &message) {
    if (error)
      *error = message;
    return false;
  };
  if (!out || name.empty() || !settings.row_capacity ||
      settings.row_capacity > 512 || !settings.cache_capacity ||
      (settings.phase == Phase::Decode && settings.row_capacity != 1))
    return fail("invalid Llama program output, name or capacities");
  // Checked byte arithmetic is confined to the public buffer requirements.
  auto bytes = [&](std::initializer_list<uint64_t> factors, uint64_t *result) {
    uint64_t n = 1;
    for (auto factor : factors) {
      if (factor && n > uint64_t(INT64_MAX) / factor)
        return false;
      n *= factor;
    }
    *result = n;
    return true;
  };
  uint64_t residual, cache, map;
  const auto physical =
      settings.pool_capacity ? settings.pool_capacity : settings.cache_capacity;
  if (settings.cache_base > uint64_t(INT64_MAX) ||
      settings.map_origin > uint64_t(INT64_MAX) || facts.layers.empty() ||
      !bytes({4, settings.row_capacity, facts.embedding_width}, &residual) ||
      !bytes({4, uint64_t(facts.layers.size()), physical,
              facts.layers.front().kv_heads, facts.key_head_width},
             &cache) ||
      cache > uint64_t(INT64_MAX) - settings.cache_base ||
      settings.cache_capacity > uint64_t(INT64_MAX) - 63 ||
      !bytes({4, (settings.cache_capacity + 63) / 64}, &map) ||
      map > uint64_t(INT64_MAX) - settings.map_origin)
    return fail("Llama buffer requirement overflow");
  LlamaBuildOptions options;
  options.phase = settings.phase;
  options.row_capacity = settings.row_capacity;
  options.cache_capacity = settings.cache_capacity;
  options.history_capacity = 128;
  options.source_asset = settings.source_asset;
  options.attention_policy = NumericalPolicy::LlamaPacketMixed;
  LlamaProgram candidate;
  if (!BuildLlamaStructure(facts, options, &candidate.structure, error))
    return false;
  std::unique_ptr<CommandProgramBuilder> program;
  auto ok = [&](loomc_status_t status) {
    bool success = loomc_status_is_ok(status);
    if (!success) {
      auto message = loomc_status_message(status);
      if (error)
        error->assign(message.data, message.size);
    }
    loomc_status_free(status);
    return success;
  };
  if (!ok(module.DefineProgram(name, {}, 7, &program)))
    return false;
  dispatch::AttentionSequenceOptions attention_options;
  attention_options.row_capacity = settings.row_capacity;
  attention_options.phase = settings.phase;
  attention_options.cache_base = settings.cache_base;
  attention_options.map_origin = settings.map_origin;
  dispatch::AttentionNativeConfig loaded;
  if (!ok(dispatch::CheckAttentionSequence(module, *program, attention_options,
                                           &loaded)))
    return false;
  if (loaded.layer_count != facts.layers.size() ||
      loaded.cache_capacity != settings.cache_capacity ||
      loaded.pool_capacity != settings.pool_capacity)
    return fail(
        "Llama program settings disagree with loaded attention configuration");
  const auto &owner = *candidate.structure;
  const auto &in = owner.inputs();
  dispatch::NativeEmissionSession session(registry, owner.graph(), module,
                                          *program);
  if (!ok(session.Bind(in.residual, program->Binding(1))) ||
      !ok(session.Bind(in.control, program->Binding(2))) ||
      !ok(session.Bind(in.tokens, program->Binding(5))) ||
      !ok(session.Bind(in.history, program->Binding(5))) ||
      !ok(session.Bind(in.progress, program->Binding(6))))
    return false;
  dispatch::NativeLlamaRows rows;
  rows.row_begin = in.row_begin;
  rows.row_count = in.row_count;
  rows.count_control = in.count_control;
  rows.packet_positions = in.positions;
  rows.row_capacity = settings.row_capacity;
  rows.phase = settings.phase;
  rows.packet = program->Binding(2);
  if (!ok(session.LlamaRows(rows)) ||
      !dispatch::BindRegionParameters(
          owner, {0, owner.graph().invocations().size(), {}},
          program->Binding(0), session, error))
    return false;
  struct Attention {
    size_t slot;
    LlamaRegionView view;
  };
  std::vector<Attention> regions;
  for (size_t layer = 0; layer < facts.layers.size(); ++layer) {
    LlamaRegionView view;
    if (!owner.select({LlamaRegionKind::Attention, layer, 1}, &view, error))
      return false;
    regions.push_back({layer, std::move(view)});
  }
  const auto &ops = owner.graph().invocations();
  for (size_t i = 0; i < ops.size(); ++i) {
    auto region =
        std::find_if(regions.begin(), regions.end(), [&](const Attention &a) {
          return i == a.view.operation_begin + 4;
        });
    if (region != regions.end()) {
      dispatch::AttentionSequenceOptions attention;
      attention.row_capacity = settings.row_capacity;
      attention.layer = region->slot;
      attention.phase = settings.phase;
      attention.cache_base = settings.cache_base;
      attention.map_origin = settings.map_origin;
      if (!ok(session.EmitAttentionRegion(
              owner, {LlamaRegionKind::Attention, region->slot, 1}, attention,
              program->Binding(3), program->Binding(4))))
        return false;
      i = region->view.operation_end - 2;
      continue;
    }
    using dispatch::Id;
    Id id;
    switch (ops[i].opcode()) {
    case OpCode::Embedding:
      id = Id::Embedding;
      break;
    case OpCode::RmsNorm:
      id = Id::RmsNorm;
      break;
    case OpCode::Linear:
      id = Id::Projection;
      break;
    case OpCode::LinearResidual: {
      bool attention =
          std::any_of(regions.begin(), regions.end(), [&](const Attention &a) {
            return i + 1 == a.view.operation_end;
          });
      id = attention                          ? Id::AttentionOutput
           : settings.phase == Phase::Prefill ? Id::DownPrefill
                                              : Id::DownDecode;
      break;
    }
    case OpCode::GatedMatmul:
      id = settings.phase == Phase::Prefill ? Id::GateUpPrefill
                                            : Id::GateUpDecode;
      break;
    case OpCode::OutputHead:
      id = Id::OutputHead;
      break;
    case OpCode::PublishToken:
      id = settings.phase == Phase::Prefill ? Id::PublishPrefill
                                            : Id::PublishDecode;
      break;
    default:
      return fail("unhandled operation in complete Llama structure");
    }
    if (!ok(session.EmitExisting(id, ops[i])))
      return false;
  }
  if (!ok(program->Finish()))
    return false;
  candidate.program = program->Ref();
  candidate.buffers = {
      {"parameters", 0},
      {"residual", residual},
      {"control", 12},
      {"map", settings.map_origin + std::max<uint64_t>(4, map)},
      {"cache", cache + settings.cache_base},
      {"tokens_history", 4 * std::max<uint64_t>(settings.row_capacity, 129)},
      {"progress", 32}};
  *out = std::move(candidate);
  return true;
}
} // namespace hrx::llama
