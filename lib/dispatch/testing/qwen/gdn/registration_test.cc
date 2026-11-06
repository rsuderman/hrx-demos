// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/qwen_gdn.h"
#include "lib/operations/qwen_operations.h"
#include "lib/qwen/structure.h"
#include "lib/qwen/testing/model_fixture.h"
#include "loomc/iree.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "gtest/gtest.h"
#include <cstdlib>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return ::testing::AssertionSuccess();
  const auto view = loomc_status_message(status);
  const std::string message(view.data ? view.data : "", view.size);
  loomc_status_free(status);
  return ::testing::AssertionFailure() << message;
}
std::shared_ptr<const qwen::QwenStructure> Owner() {
  std::shared_ptr<const qwen::QwenStructure> owner;
  std::string error;
  EXPECT_TRUE(qwen::BuildQwenStructure(
      qwen::testing::ModelFacts(), {8, 1, 1, 96, "fixture"}, &owner, &error))
      << error;
  return owner;
}
std::vector<shapes::Operand> Inputs(const Invocation &op) {
  std::vector<shapes::Operand> result;
  for (const auto &b : op.bindings())
    result.push_back({b.port(), b.value()});
  return result;
}
TEST(QwenGdnRegistration, AdmitsAllKnownGdnLayersWithoutIndexMaskRouting) {
  const auto owner = Owner();
  ASSERT_TRUE(owner);
  const auto entries = QwenGdnRegistrations();
  ASSERT_EQ(entries.size(), 1u);
  const auto &entry = entries.front();
  EXPECT_EQ(entry.kind, Kind::CommandProgram);
  size_t count = 0;
  for (const auto &op : owner->graph().invocations()) {
    if (op.opcode() != OpCode::QwenGdnLayer)
      continue;
    auto inputs = Inputs(op);
    shapes::Result result;
    std::string error;
    ASSERT_TRUE(shapes::InferResult(owner->graph(), op.opcode(),
                                    op.attributes(), inputs, &result, &error))
        << error;
    InvocationRequest request{owner->graph(), op.attributes(), inputs, result};
    EXPECT_TRUE(entry.builder.supports(entry, request, &error)) << error;
    EXPECT_EQ(result.alias, owner->inputs().residual);
    ++count;
  }
  EXPECT_EQ(count, 18u);
}
TEST(QwenGdnRegistration, RejectsUnsupportedNativeProfileBeforeBuilding) {
  const auto owner = Owner();
  ASSERT_TRUE(owner);
  const auto entry = QwenGdnRegistrations().front();
  const auto &op = owner->graph().invocations()[1];
  auto inputs = Inputs(op);
  shapes::Result result;
  std::string error;
  ASSERT_TRUE(shapes::InferResult(owner->graph(), op.opcode(), op.attributes(),
                                  inputs, &result, &error));
  for (unsigned change = 0; change < 5; ++change) {
    auto a = std::get<QwenGdnLayerAttrs>(op.attributes());
    if (change == 0)
      a.hidden_width = 2048;
    if (change == 1)
      a.state_width = 64;
    if (change == 2)
      a.span_capacity = 2;
    if (change == 3)
      a.epsilon = 1e-5;
    if (change == 4)
      a.token_capacity = 513;
    OperationAttrs attrs = a;
    InvocationRequest request{owner->graph(), attrs, inputs, result};
    error.clear();
    EXPECT_FALSE(entry.builder.supports(entry, request, &error));
    EXPECT_FALSE(error.empty());
  }
}
TEST(QwenGdnRegistration,
     EveryWeightRequiresExactKeyAndSameScopedBankBeforeCall) {
  const auto owner = Owner();
  ASSERT_TRUE(owner);
  const auto &op = owner->graph().invocations()[5]; // original layer4, ordinal3
  ASSERT_EQ(op.opcode(), OpCode::QwenGdnLayer);
  const auto a = std::get<QwenGdnLayerAttrs>(op.attributes());
  ASSERT_NE(a.layer_index, a.recurrent_ordinal);
  auto inputs = Inputs(op);
  shapes::Result result;
  std::string error;
  ASSERT_TRUE(shapes::InferResult(owner->graph(), op.opcode(), op.attributes(),
                                  inputs, &result, &error));
  InvocationRequest request{owner->graph(), op.attributes(), inputs, result};
  const auto entry = QwenGdnRegistrations().front();
  loomc_target_environment_t *environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &environment)));
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  ASSERT_TRUE(
      Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
  ASSERT_TRUE(Ok(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  const char *srcdir = std::getenv("TEST_SRCDIR");
  const char *testworkspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(srcdir, nullptr);
  ASSERT_NE(testworkspace, nullptr);
  const auto path = std::string(srcdir) + "/" + testworkspace +
                    "/lib/kernels/qwen/gdn/interface.loom";
  loomc_module_t *definitions = nullptr;
  loomc_result_t *diagnostics = nullptr;
  ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
      context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
      loomc_allocator_system(), &definitions, &diagnostics)));
  ASSERT_TRUE(diagnostics && loomc_result_succeeded(diagnostics));
  loomc_result_release(diagnostics);
  {
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    const auto &roles = QwenLayerWeightRoles(OpCode::QwenGdnLayer);
    for (size_t role = 0; role < roles.size(); ++role)
      for (bool wrong_bank : {false, true}) {
        std::unique_ptr<CommandProgramBuilder> program;
        ASSERT_TRUE(Ok(module->DefineProgram(
            "reject_" + std::to_string(role) + (wrong_bank ? "_bank" : "_key"),
            {}, 6, &program)));
        CommandValues values(*program);
        std::vector<NativeOperand> native;
        for (const auto &input : inputs) {
          Value value;
          std::string parameter;
          if (input.port == Port::Residual)
            value = program->Binding(1);
          else if (input.port == Port::Epoch)
            value = program->Binding(3);
          else if (input.port == Port::Rows)
            value = program->Binding(4);
          else if (input.port == Port::ConvolutionHistory ||
                   input.port == Port::RecurrentState)
            value = program->Binding(2);
          else {
            value = program->Binding(0);
            for (const auto &r : roles)
              if (r.port == input.port)
                parameter = "blk.4." + std::string(r.suffix);
          }
          if (input.port == roles[role].port) {
            if (wrong_bank)
              value = program->Binding(5);
            else
              parameter += ".wrong";
          }
          native.push_back({input.port, input.value, value, parameter});
        }
        NativeQwenEpoch epoch;
        epoch.logical_epoch = owner->inputs().epoch;
        epoch.logical_rows = owner->inputs().rows;
        epoch.options.token_capacity = 8;
        epoch.options.span_capacity = 1;
        epoch.epoch = program->Binding(3);
        epoch.rows = program->Binding(4);
        epoch.resident_arena = program->Binding(2);
        epoch.states = {{op.binding(Port::ConvolutionHistory)->value(), 4, 4, 3,
                         QwenStateRole::GdnHistory},
                        {op.binding(Port::RecurrentState)->value(), 4, 4, 3,
                         QwenStateRole::GdnMatrix}};
        BuildContext ctx{request, *module, *program, values, native, {}, false};
        ctx.output = program->Binding(1);
        ctx.has_output = true;
        ctx.qwen_epoch = &epoch;
        size_t before = 0;
        ASSERT_TRUE(Ok(program->InspectNative(
            {},
            [&](const loom_module_t *m, const std::vector<loom_value_id_t> &) {
              before = m->values.count;
              return iree_ok_status();
            })));
        Value out;
        const auto status = entry.builder.build(entry, ctx, &out);
        EXPECT_FALSE(loomc_status_is_ok(status));
        const auto message_view = loomc_status_message(status);
        const std::string message(message_view.data ? message_view.data : "",
                                  message_view.size);
        // A missing native callable is not evidence that parameter preflight
        // ran.
        EXPECT_NE(message.find("parameter"), std::string::npos) << message;
        loomc_status_free(status);
        EXPECT_FALSE(ctx.staged);
        ASSERT_TRUE(Ok(program->InspectNative(
            {},
            [&](const loom_module_t *m, const std::vector<loom_value_id_t> &) {
              EXPECT_EQ(m->values.count, before);
              return iree_ok_status();
            })));
        ASSERT_TRUE(Ok(program->Finish()));
      }
  }
  loomc_module_release(definitions);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
}
} // namespace
} // namespace hrx::dispatch
