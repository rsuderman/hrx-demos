// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/dispatch/testing/attention_fixture.h"
#include "loomc/target/amdgpu.h"
#include "gtest/gtest.h"
#include <cstdlib>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Reject(loomc_status_t status) {
  const bool rejected = !loomc_status_is_ok(status);
  loomc_status_free(status);
  return rejected;
}
class AttentionRegionTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(
        Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    const auto root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
                      std::getenv("TEST_WORKSPACE");
    loomc_module_t *definitions = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(LoadAttentionSequence(context, workspace, root,
                                         AttentionNativeConfig{2, 128, 128},
                                         &definitions, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result));
    loomc_result_release(result);
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    loomc_module_release(definitions);
    ASSERT_TRUE(Ok(module->DefineProgram("region", {}, 8, &program)));
    construction = testing::AttentionFixture(8, 128, Phase::Prefill, &error);
    ASSERT_TRUE(construction) << error;
  }
  void TearDown() override {
    construction.reset();
    program.reset();
    module.reset();
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
  }
  bool Configure(NativeEmissionSession &session, std::string factor_key = {}) {
    const auto &owner = construction->structure();
    const auto &inputs = owner.inputs();
    NativeLlamaRows rows;
    rows.row_begin = inputs.row_begin;
    rows.row_count = inputs.row_count;
    rows.packet_positions = inputs.positions;
    rows.row_capacity = 8;
    rows.token_begin = 2;
    rows.control_origin = 8;
    rows.packet = program->Binding(0);
    if (!Ok(session.LlamaRows(rows)))
      return false;
    const auto &ops = owner.graph().invocations();
    return Ok(factor_key.empty()
                  ? session.Bind(ops[4].binding(Port::Factors)->value(),
                                 program->Binding(3))
                  : session.Parameter(ops[4].binding(Port::Factors)->value(),
                                      program->Binding(3), factor_key)) &&
           Ok(session.Bind(ops[4].binding(Port::Input)->value(),
                           program->Binding(4))) &&
           Ok(session.Bind(ops[5].binding(Port::Key)->value(),
                           program->Binding(5))) &&
           Ok(session.Bind(ops[5].binding(Port::Value)->value(),
                           program->Binding(6)));
  }
  AttentionSequenceOptions Options() {
    AttentionSequenceOptions o;
    o.row_capacity = 8;
    o.token_begin = 2;
    o.control_origin = 8;
    o.cache_base = 16;
    o.map_origin = 4;
    return o;
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::unique_ptr<ModuleBuilder> module;
  std::unique_ptr<CommandProgramBuilder> program;
  std::unique_ptr<llama::detail::StructureConstruction> construction;
  Registry registry = Registry::Llama3B();
  std::string error;
};
TEST_F(AttentionRegionTest,
       AdmissionFailuresRecoverThenEmissionConsumesRegion) {
  const auto &owner = construction->structure();
  const auto values = owner.graph().values().size();
  const auto operations = owner.graph().invocations().size();
  NativeEmissionSession session(registry, owner.graph(), *module, *program);
  ASSERT_TRUE(Configure(session));
  const llama::LlamaRegion selected{llama::LlamaRegionKind::Attention, 0, 1};
  auto invoke = [&](const llama::LlamaStructure &s, llama::LlamaRegion r,
                    const AttentionSequenceOptions &o) {
    return session.EmitAttentionRegion(s, r, o, program->Binding(1),
                                       program->Binding(2));
  };
  auto foreign = testing::AttentionFixture(8, 128, Phase::Prefill, &error);
  ASSERT_TRUE(foreign);
  EXPECT_TRUE(Reject(invoke(foreign->structure(), selected, Options())));
  auto bad = selected;
  bad.kind = llama::LlamaRegionKind::Ffn;
  EXPECT_TRUE(Reject(invoke(owner, bad, Options())));
  bad = selected;
  bad.layer_count = 2;
  EXPECT_TRUE(Reject(invoke(owner, bad, Options())));
  for (int field = 0; field < 5; ++field) {
    auto o = Options();
    if (field == 0)
      o.row_capacity = 7;
    if (field == 1)
      o.token_begin = 1;
    if (field == 2)
      o.control_origin = 0;
    if (field == 3)
      o.layer = 1;
    if (field == 4)
      o.phase = Phase::Decode;
    EXPECT_TRUE(Reject(invoke(owner, selected, o)));
  }
  ASSERT_TRUE(Ok(invoke(owner, selected, Options())));
  Value output;
  ASSERT_TRUE(Ok(session.Native(
      owner.graph().invocations()[6].binding(Port::Output)->value(), &output)));
  EXPECT_TRUE(Reject(invoke(owner, selected, Options())));
  EXPECT_EQ(owner.graph().values().size(), values);
  EXPECT_EQ(owner.graph().invocations().size(), operations);
  ASSERT_TRUE(Ok(program->Finish()));
}
TEST_F(AttentionRegionTest, PreboundOutputAndPrivateQueryRejectBeforeStaging) {
  const auto &owner = construction->structure();
  for (const auto port_index : {4u, 6u}) {
    NativeEmissionSession session(registry, owner.graph(), *module, *program);
    ASSERT_TRUE(Configure(session));
    const auto id =
        owner.graph().invocations()[port_index].binding(Port::Output)->value();
    ASSERT_TRUE(Ok(session.Bind(id, program->Binding(7))));
    EXPECT_TRUE(Reject(session.EmitAttentionRegion(
        owner, {llama::LlamaRegionKind::Attention, 0, 1}, Options(),
        program->Binding(1), program->Binding(2))));
    Value retained;
    EXPECT_TRUE(Ok(session.Native(id, &retained)));
  }
}
TEST_F(AttentionRegionTest, ExportedPrivateQueryRejects) {
  const auto &owner = construction->structure();
  ASSERT_TRUE(construction->exportValue(
      owner.graph().invocations()[4].binding(Port::Output)->value(), &error));
  NativeEmissionSession session(registry, owner.graph(), *module, *program);
  ASSERT_TRUE(Configure(session));
  EXPECT_TRUE(Reject(session.EmitAttentionRegion(
      owner, {llama::LlamaRegionKind::Attention, 0, 1}, Options(),
      program->Binding(1), program->Binding(2))));
}
TEST_F(AttentionRegionTest,
       ChangedProfileAndLoadedCapacityRejectBeforeStaging) {
  for (int field = 0; field < 5; ++field) {
    construction = testing::AttentionFixture(
        8, field == 3 ? 64 : 128, Phase::Prefill, &error, field == 4 ? 1 : 2,
        [field](model::ModelDescriptor &facts,
                llama::LlamaBuildOptions &options) {
          if (field == 0)
            facts.rope.base = 10000;
          if (field == 1) {
            facts.rope.scaling_kind = "linear";
            facts.rope.scaling_factor = 2;
          }
          if (field == 2)
            options.attention_policy = NumericalPolicy::F32Accumulate;
        });
    ASSERT_TRUE(construction) << error;
    const auto &owner = construction->structure();
    NativeEmissionSession session(registry, owner.graph(), *module, *program);
    ASSERT_TRUE(Configure(session));
    EXPECT_TRUE(
        Reject(session.Bind(owner.inputs().positions, program->Binding(7))));
    EXPECT_TRUE(Reject(session.EmitAttentionRegion(
        owner, {llama::LlamaRegionKind::Attention, 0, 1}, Options(),
        program->Binding(1), program->Binding(2))));
    Value retained;
    EXPECT_TRUE(Ok(session.Native(
        owner.graph().invocations()[4].binding(Port::Input)->value(),
        &retained)));
  }
}
TEST_F(AttentionRegionTest, TypedPlaneBindingCannotFakeArenaViews) {
  const auto &owner = construction->structure();
  NativeEmissionSession session(registry, owner.graph(), *module, *program);
  ASSERT_TRUE(Configure(session));
  ASSERT_TRUE(Ok(
      session.Bind(owner.inputs().layers[0].key_cache, program->Binding(2))));
  ASSERT_TRUE(Ok(
      session.Bind(owner.inputs().layers[0].value_cache, program->Binding(2))));
  EXPECT_TRUE(Reject(session.EmitAttentionRegion(
      owner, {llama::LlamaRegionKind::Attention, 0, 1}, Options(),
      program->Binding(1), program->Binding(2))));
}

TEST_F(AttentionRegionTest,
       WrongFactorRoleKeyRejectsAndCorrectNamedFactorsStage) {
  const auto &owner = construction->structure();
  const llama::LlamaRegion selection{llama::LlamaRegionKind::Attention, 0, 1};
  NativeEmissionSession rejected(registry, owner.graph(), *module, *program);
  ASSERT_TRUE(Configure(rejected, "unrelated.weight"));
  EXPECT_TRUE(Reject(rejected.EmitAttentionRegion(
      owner, selection, Options(), program->Binding(1), program->Binding(2))));
  NativeEmissionSession accepted(registry, owner.graph(), *module, *program);
  ASSERT_TRUE(Configure(accepted, "rope_freqs.weight"));
  ASSERT_TRUE(Ok(accepted.EmitAttentionRegion(
      owner, selection, Options(), program->Binding(1), program->Binding(2))));
}
TEST_F(AttentionRegionTest, SameShapePositionsAreNotPacketDerivation) {
  const auto &owner = construction->structure();
  auto &graph = construction->graph();
  NativeLlamaRows rows;
  rows.row_begin = owner.inputs().row_begin;
  rows.row_count = owner.inputs().row_count;
  rows.packet_positions =
      graph.tensor(TensorType::vector(8, ElementType::I32), &error);
  rows.row_capacity = 8;
  rows.token_begin = 2;
  rows.control_origin = 8;
  rows.packet = program->Binding(0);
  NativeEmissionSession session(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(session.LlamaRows(rows)));
  EXPECT_TRUE(Reject(session.EmitAttentionRegion(
      owner, {llama::LlamaRegionKind::Attention, 0, 1}, Options(),
      program->Binding(1), program->Binding(2))));
}

} // namespace
} // namespace hrx::dispatch
