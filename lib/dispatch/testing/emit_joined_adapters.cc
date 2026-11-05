// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/dispatch/testing/attention_fixture.h"
#include "lib/operations/tensor_operations.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <memory>
using namespace hrx::dispatch;
using namespace hrx::operations;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto *d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  return false;
}

bool LayerRoot(ModuleBuilder &module, const Registry &registry,
               const std::string &name, uint64_t layers, bool decode,
               bool compact, std::vector<RootRef> *roots) {
  std::string error;
  const uint64_t capacity = decode ? 1 : compact ? 3 : 8;
  const uint64_t begin = decode || compact ? 0 : 2;
  auto construction = hrx::dispatch::testing::AttentionFixture(
      capacity, 512, decode ? Phase::Decode : Phase::Prefill, &error, 2, {},
      true);
  if (!construction) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return false;
  }
  const auto &owner = construction->structure();
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module.DefineProgram(name, {}, 24, &program)))
    return false;
  NativeEmissionSession session(registry, owner.graph(), module, *program);
  NativeLlamaRows rows;
  rows.row_begin = owner.inputs().row_begin;
  rows.row_count = owner.inputs().row_count;
  rows.packet_positions = owner.inputs().positions;
  rows.row_capacity = capacity;
  rows.token_begin = begin;
  rows.phase = decode ? Phase::Decode : Phase::Prefill;
  rows.packet = program->Binding(20);
  if (!Ok(session.LlamaRows(rows)) ||
      !Ok(session.Bind(owner.inputs().residual, program->Binding(19))))
    return false;
  for (const auto &association : owner.parameters()) {
    const auto &coordinate = association.coordinate;
    const auto *parameter = owner.parameter(coordinate);
    const auto index =
        coordinate.scope == hrx::llama::ParameterScope::Global
            ? 0
            : 1 + coordinate.layer_slot * 9 + coordinate.tensor_slot;
    if (!parameter ||
        !Ok(session.Parameter(association.value, program->Binding(index),
                              parameter->tensor.name)))
      return false;
  }
  const auto &operations = owner.graph().invocations();
  for (uint64_t layer = 0; layer < layers; ++layer) {
    const size_t base = layer * 11;
    for (size_t i = 0; i < 4; ++i)
      if (!Ok(session.EmitExisting(i ? Id::Projection : Id::RmsNorm,
                                   operations[base + i])))
        return false;
    AttentionSequenceOptions options;
    options.row_capacity = capacity;
    options.token_begin = begin;
    options.phase = rows.phase;
    options.layer = layer;
    options.cache_base = 16;
    if (!Ok(session.EmitAttentionRegion(
            owner, {hrx::llama::LlamaRegionKind::Attention, layer, 1}, options,
            program->Binding(21), program->Binding(22))) ||
        !Ok(session.EmitExisting(Id::AttentionOutput, operations[base + 7])) ||
        !Ok(session.EmitExisting(Id::RmsNorm, operations[base + 8])) ||
        !Ok(session.EmitExisting(decode ? Id::GateUpDecode : Id::GateUpPrefill,
                                 operations[base + 9])) ||
        !Ok(session.EmitExisting(decode ? Id::DownDecode : Id::DownPrefill,
                                 operations[base + 10])))
      return false;
  }
  Value residual;
  if (!Ok(session.Native(owner.inputs().residual, &residual)) ||
      !Ok(program->InspectNative(
          {residual, program->Binding(19)},
          [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
            return ids[0] == ids[1]
                       ? iree_ok_status()
                       : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                          "lost joined residual alias");
          })) ||
      !Ok(program->Finish()))
    return false;
  roots->push_back(program->Ref());
  return true;
}
bool Replay(ModuleBuilder &module, const std::string &name, uint64_t layers,
            bool decode, std::vector<RootRef> *roots) {
  ProgramRef active, compact;
  if (!Ok(module.Program(name + ".active", &active)) ||
      !Ok(module.Program(name + ".compact", &compact)))
    return false;
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module.DefineProgram(name, {}, 24, &program)))
    return false;
  CommandValues values(*program);
  auto constant = [&](uint64_t n, loom_scalar_type_t t, Value *v) {
    return Ok(values.Constant(n, t, v));
  };
  Value capacity, count, begin, phase, layer_count, residual_size, compact_size,
      cache_size, before, small, compact_cache, before_cache;
  if (!constant(decode ? 1 : 8, LOOM_SCALAR_TYPE_INDEX, &capacity) ||
      !constant(decode ? 1 : 3, LOOM_SCALAR_TYPE_INDEX, &count) ||
      !constant(decode ? 0 : 2, LOOM_SCALAR_TYPE_INDEX, &begin) ||
      !constant(decode, LOOM_SCALAR_TYPE_INDEX, &phase) ||
      !constant(layers, LOOM_SCALAR_TYPE_INDEX, &layer_count) ||
      !constant((decode ? 1 : 8) * 3072 * 4, LOOM_SCALAR_TYPE_OFFSET,
                &residual_size) ||
      !constant((decode ? 1 : 3) * 3072 * 4, LOOM_SCALAR_TYPE_OFFSET,
                &compact_size) ||
      !constant(4194368, LOOM_SCALAR_TYPE_OFFSET, &cache_size) ||
      !Ok(values.Allocate(residual_size, 256, &before)) ||
      !Ok(values.Allocate(compact_size, 256, &small)) ||
      !Ok(values.Allocate(cache_size, 256, &compact_cache)) ||
      !Ok(values.Allocate(cache_size, 256, &before_cache)))
    return false;
  KernelRef seed, compare, initialize;
  if (!Ok(module.Kernel("test.join_seed", &seed)) ||
      !Ok(module.Kernel("test.join_compare", &compare)) ||
      !Ok(module.Kernel("test.join_packet", &initialize)))
    return false;
  for (uint64_t iteration = 0; iteration < 2; ++iteration) {
    Value ordinal;
    if (!constant(iteration, LOOM_SCALAR_TYPE_INDEX, &ordinal) ||
        !Ok(program->Invoke(initialize, {},
                            {phase, ordinal, program->Binding(20)})) ||
        !Ok(program->Invoke(seed, {capacity, count},
                            {capacity, count, begin, program->Binding(19),
                             before, small, program->Binding(22), before_cache,
                             compact_cache})))
      return false;
    std::vector<Value> bindings;
    for (unsigned i = 0; i < 24; ++i)
      bindings.push_back(program->Binding(i));
    if (!Ok(program->Call(active, {}, bindings)))
      return false;
    bindings[19] = small;
    bindings[22] = compact_cache;
    if (!Ok(program->Call(compact, {}, bindings)) ||
        !Ok(program->Invoke(compare, {capacity, count},
                            {capacity, count, begin, phase, layer_count,
                             ordinal, program->Binding(20),
                             program->Binding(19), before, small,
                             program->Binding(22), before_cache, compact_cache,
                             program->Binding(23)})))
      return false;
  }
  if (!Ok(program->Finish()))
    return false;
  roots->push_back(program->Ref());
  return true;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 4)
    return 1;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  loomc_target_environment_t *environment = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &environment)))
    return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t context_options{};
  context_options.next = &target;
  if (!Ok(loomc_context_create(&context_options, loomc_allocator_system(),
                               &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  auto defaults = Registry::Llama3B();
  Registry registry;
  const std::vector<Id> ids{
      Id::RmsNorm,      Id::Projection,  Id::AttentionOutput, Id::GateUpPrefill,
      Id::GateUpDecode, Id::DownPrefill, Id::DownDecode};
  for (auto id : ids)
    for (auto entry : defaults.Implementations(id)) {
      if (id == Id::RmsNorm)
        entry.provider_files.push_back(
            "lib/dispatch/testing/testdata/joined_adapters.loom");
      std::string error;
      if (!registry.Add(std::move(entry), &error))
        return 1;
    }
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(LoadLlamaNative(context, workspace, registry, ids, argv[1],
                          AttentionNativeConfig{2, 512, 0}, &definitions,
                          &result)) ||
      !Succeeded(result))
    return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  std::vector<RootRef> roots;
  for (uint64_t layers : {1u, 2u})
    for (bool decode : {false, true}) {
      const auto name = std::string("joined.") + std::to_string(layers) +
                        (decode ? ".decode" : ".prefill");
      if (!LayerRoot(*module, registry, name + ".active", layers, decode, false,
                     &roots) ||
          !LayerRoot(*module, registry, name + ".compact", layers, decode, true,
                     &roots) ||
          !Replay(*module, name, layers, decode, &roots))
        return 1;
    }
  loomc_module_t *linked = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify(roots, &linked, &result)) || !Succeeded(result))
    return 1;
  loomc_result_release(result);
  bool ok = true;
  for (int format = 0; format < 2; ++format) {
    loomc_module_serialize_options_t options{};
    options.format =
        format ? LOOMC_SOURCE_FORMAT_TEXT : LOOMC_SOURCE_FORMAT_BYTECODE;
    ok &= Ok(loomc_module_serialize_to_path(
        linked, &options, loomc_make_cstring_view(argv[2 + format]),
        loomc_allocator_system()));
  }
  loomc_module_release(linked);
  program.reset();
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
  return ok ? 0 : 1;
}
