// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/qwen_gdn.h"
#include "lib/operations/qwen_operations.h"
#include "lib/qwen/structure.h"
#include "lib/qwen/testing/model_fixture.h"
#include "loomc/iree.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "tools/model/qwen_configuration.h"
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
std::string Diagnostics(loomc_result_t *r) {
  std::string text;
  if (r)
    for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
      const auto *d = loomc_result_diagnostic_at(r, i);
      text.append(d->message.data, d->message.size);
      text += '\n';
    }
  return text;
}
size_t OperationCount(const loom_region_t *region) {
  size_t count = 0;
  for (size_t b = 0; b < region->block_count; ++b)
    for (const auto *op = region->blocks[b]->first_op; op; op = op->next_op) {
      ++count;
      for (size_t r = 0; r < op->region_count; ++r)
        count += OperationCount(loom_op_regions(op)[r]);
    }
  return count;
}
model::ModelDescriptor Facts() {
  auto facts = qwen::testing::ModelFacts();
  // Structural fixture intentionally varies Q/K tags; this native fixture
  // chooses one actually supported matched pair without changing shared facts.
  for (auto &l : facts.layers)
    if (l.kind == model::LayerKind::kAttention)
      for (auto &t : l.tensors)
        if (t.role == "key") {
          t.tensor.storage_type = 14;
          t.tensor.byte_size = 1024 / 256 * 512 * 210;
        }
  return facts;
}
class QwenGdnEmission : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &env)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = env;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(
        Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
           std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(env);
  }
  loomc_target_environment_t *env = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_F(QwenGdnEmission,
       CompleteOwnedClosureRegisteredCallAliasAndSourceBytecodeReload) {
  auto facts = Facts();
  std::string error;
  if (const char *path = std::getenv("LOOMIFY_QWEN_GGUF")) {
    FILE *file = std::fopen(path, "rb");
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(std::setvbuf(file, nullptr, _IONBF, 0), 0);
    const bool extracted =
        model::ExtractQwenConfiguration(file, &facts, &error);
    std::fclose(file);
    ASSERT_TRUE(extracted) << error;
  }
  const auto entry = QwenGdnRegistrations().front();
  Registry registry;
  ASSERT_TRUE(registry.Add(entry, &error)) << error;
  uint64_t tokens = 8;
  if (const char *value = std::getenv("LOOMIFY_QWEN_TEST_T"))
    tokens = std::strtoull(value, nullptr, 10);
  ASSERT_GE(tokens, 1u);
  ASSERT_LE(tokens, 512u);
  const QwenNativeOptions options{tokens, 1, 1, 96, 128, 1};
  QwenNativeGeometry geometry;
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(
      Ok(LoadQwenNative(context, workspace, registry, {Id::QwenGdnLayer}, facts,
                        options, root, &geometry, &definitions, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::shared_ptr<const qwen::QwenStructure> owner;
  ASSERT_TRUE(qwen::BuildQwenStructure(facts, {tokens, 1, 1, 96, "fixture"},
                                       &owner, &error))
      << error;
  const auto &op = owner->graph().invocations()[5]; // original4 != recurrent3
  std::vector<shapes::Operand> inputs;
  for (const auto &b : op.bindings())
    inputs.push_back({b.port(), b.value()});
  shapes::Result logical_result;
  ASSERT_TRUE(shapes::InferResult(owner->graph(), op.opcode(), op.attributes(),
                                  inputs, &logical_result, &error))
      << error;
  {
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    std::unique_ptr<CommandProgramBuilder> program;
    ASSERT_TRUE(
        Ok(module->DefineProgram("generated_gdn_layer", {}, 5, &program)));
    NativeEmissionSession session(registry, owner->graph(), *module, *program);
    NativeQwenEpoch epoch;
    epoch.logical_epoch = owner->inputs().epoch;
    epoch.logical_rows = owner->inputs().rows;
    epoch.epoch = program->Binding(3);
    epoch.rows = program->Binding(4);
    epoch.resident_arena = program->Binding(2);
    epoch.options = options;
    for (const auto &state : owner->inputs().layers) {
      if (state.kind == model::LayerKind::kGatedDeltaNet) {
        epoch.states.push_back({state.history, state.layer_slot,
                                state.original_layer_index, state.ordinal,
                                QwenStateRole::GdnHistory});
        epoch.states.push_back({state.recurrent, state.layer_slot,
                                state.original_layer_index, state.ordinal,
                                QwenStateRole::GdnMatrix});
      } else {
        epoch.states.push_back({state.key_cache, state.layer_slot,
                                state.original_layer_index, state.ordinal,
                                QwenStateRole::Key});
        epoch.states.push_back({state.value_cache, state.layer_slot,
                                state.original_layer_index, state.ordinal,
                                QwenStateRole::Value});
      }
    }
    // Every native state shares the arena; that equality alone must never
    // admit swapped roles, another layer's state, or a changed ordinal/slot.
    for (unsigned change = 0; change < 4; ++change) {
      auto invalid = epoch;
      auto &history = invalid.states[8]; // layer4, recurrent ordinal3
      auto &matrix = invalid.states[9];
      ASSERT_EQ(history.original_layer_index, 4u);
      if (change == 0)
        ++history.ordinal;
      if (change == 1)
        ++history.layer_slot;
      if (change == 2)
        std::swap(history.value, matrix.value);
      if (change == 3)
        history.value = invalid.states[0].value;
      auto status = session.QwenEpoch(invalid);
      EXPECT_FALSE(loomc_status_is_ok(status));
      loomc_status_free(status);
    }
    // Failed pure admission leaves the same session available for the exact
    // retained owner's bindings and subsequent registered emission.
    ASSERT_TRUE(Ok(session.QwenEpoch(epoch)));
    ASSERT_TRUE(
        Ok(session.Bind(owner->inputs().residual, program->Binding(1))));
    for (const auto &role : QwenLayerWeightRoles(OpCode::QwenGdnLayer))
      ASSERT_TRUE(Ok(session.Parameter(op.binding(role.port)->value(),
                                       program->Binding(0),
                                       "blk.4." + std::string(role.suffix))));
    // Exercise the builder's frame preflight on the same native owner that
    // subsequently emits the canonical registered call successfully.
    const InvocationRequest request{owner->graph(), op.attributes(), inputs,
                                    logical_result};
    for (unsigned change = 0; change < 16; ++change) {
      auto frame = epoch;
      std::vector<Value> boundary;
      for (size_t i = 0; i < 5; ++i)
        boundary.push_back(program->Binding(i));
      // Ten pairwise boundary aliases, preserving the packet's mapping so
      // these cases specifically reach the known-alias rejection.
      if (change >= 6) {
        unsigned pair = 6;
        for (size_t i = 0; i < 5; ++i)
          for (size_t j = i + 1; j < 5; ++j, ++pair)
            if (pair == change)
              boundary[j] = boundary[i];
        frame.resident_arena = boundary[2];
        frame.epoch = boundary[3];
        frame.rows = boundary[4];
      }
      std::vector<NativeOperand> native;
      for (const auto &input : inputs) {
        NativeOperand operand{input.port, input.value, boundary[0], {}};
        if (input.port == Port::Residual)
          operand.native = boundary[1];
        else if (input.port == Port::ConvolutionHistory ||
                 input.port == Port::RecurrentState)
          operand.native = boundary[2];
        else if (input.port == Port::Epoch)
          operand.native = boundary[3];
        else if (input.port == Port::Rows)
          operand.native = boundary[4];
        else
          for (const auto &role : QwenLayerWeightRoles(OpCode::QwenGdnLayer))
            if (role.port == input.port)
              operand.parameter = "blk.4." + std::string(role.suffix);
        if (change == 0 && input.port == Port::Epoch)
          operand.logical = owner->inputs().rows;
        if (change == 1 && input.port == Port::Rows)
          operand.logical = owner->inputs().epoch;
        if ((change == 2 && input.port == Port::Epoch) ||
            (change == 3 && input.port == Port::Rows) ||
            (change == 4 && input.port == Port::ConvolutionHistory) ||
            (change == 5 && input.port == Port::RecurrentState))
          operand.native = boundary[1];
        native.push_back(operand);
      }
      CommandValues values(*program);
      BuildContext ctx{request, *module, *program, values, native, {}, false};
      ctx.qwen_epoch = &frame;
      size_t before_values = 0, before_operations = 0;
      ASSERT_TRUE(Ok(
          program->InspectNative({}, [&](const loom_module_t *m,
                                         const std::vector<loom_value_id_t> &) {
            before_values = m->values.count;
            before_operations = OperationCount(m->body);
            return iree_ok_status();
          })));
      Value rejected;
      auto status = entry.builder.build(entry, ctx, &rejected);
      ASSERT_FALSE(loomc_status_is_ok(status)) << change;
      const auto view = loomc_status_message(status);
      const std::string message(view.data ? view.data : "", view.size);
      EXPECT_NE(message.find(change < 6 ? "configured frame" : "distinct"),
                std::string::npos)
          << change << ": " << message;
      loomc_status_free(status);
      EXPECT_FALSE(ctx.staged);
      ASSERT_TRUE(Ok(
          program->InspectNative({}, [&](const loom_module_t *m,
                                         const std::vector<loom_value_id_t> &) {
            EXPECT_EQ(m->values.count, before_values);
            EXPECT_EQ(OperationCount(m->body), before_operations);
            return iree_ok_status();
          })));
    }
    const auto logical_values = owner->graph().values().size();
    const auto logical_operations = owner->graph().invocations().size();
    ASSERT_TRUE(Ok(session.EmitExisting(Id::QwenGdnLayer, op)));
    EXPECT_EQ(owner->graph().values().size(), logical_values);
    EXPECT_EQ(owner->graph().invocations().size(), logical_operations);
    Value output;
    ASSERT_TRUE(Ok(session.Native(owner->inputs().residual, &output)));
    ASSERT_TRUE(Ok(program->InspectNative(
        {output, program->Binding(1)},
        [&](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
          EXPECT_EQ(ids[0], ids[1]);
          return iree_ok_status();
        })));
    ASSERT_TRUE(Ok(program->Finish()));
    loomc_module_t *linked = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    for (auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t serialization{};
      serialization.format = format;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          linked, &serialization, loomc_allocator_system(), &source)));
      if (const char *directory = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
        const auto path =
            std::string(directory) + "/generated_gdn_layer" +
            (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc");
        ASSERT_TRUE(Ok(loomc_module_serialize_to_path(
            linked, &serialization, loomc_make_cstring_view(path.c_str()),
            loomc_allocator_system())));
      }
      loomc_module_t *reloaded = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context, workspace, source, nullptr, loomc_allocator_system(),
          &reloaded, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      loomc_module_release(reloaded);
      loomc_source_release(source);
    }
    loomc_module_release(linked);
  }
  loomc_module_release(definitions);
}
} // namespace
} // namespace hrx::dispatch
