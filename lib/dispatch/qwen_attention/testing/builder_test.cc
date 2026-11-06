// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/qwen_attention/builder.h"
#include "lib/operations/qwen_operations.h"
#include "lib/qwen/structure.h"
#include "lib/qwen/testing/model_fixture.h"
#ifdef QWEN_ATTENTION_REAL_GGUF_TEST
#include "tools/model/qwen_configuration.h"
#include <cstdio>
#endif
#include "loomc/compile.h"
#include "loomc/interop.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
namespace hrx::dispatch {
namespace {
std::string Message(loomc_status_t status) {
  auto view = loomc_status_message(status);
  std::string text(view.data ? view.data : "", view.size);
  loomc_status_free(status);
  return text;
}
::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << Message(status);
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  if (!result)
    return "no result";
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->message.data, d->message.size);
    text += '\n';
  }
  return text;
}
loomc_status_t Dump(const loomc_module_t *module, std::string *text) {
  loomc_module_serialize_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
  options.structure_size = sizeof(options);
  options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
  loomc_source_t *source = nullptr;
  auto status = loomc_module_serialize_text_to_source(
      module, &options, loomc_allocator_system(), &source);
  if (loomc_status_is_ok(status)) {
    auto bytes = loomc_source_contents(source);
    text->assign(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
    loomc_source_release(source);
  }
  return status;
}
using namespace operations;
using namespace loomify_gguf;
model::ModelDescriptor Facts(Encoding query = Encoding::Q6_K) {
  auto facts = qwen::testing::ModelFacts();
  for (auto &layer : facts.layers) {
    if (layer.kind != model::LayerKind::kAttention)
      continue;
    for (auto &binding : layer.tensors) {
      if (binding.role != "query" && binding.role != "key")
        continue;
      binding.tensor.storage_type = query == Encoding::Q5_K ? 13 : 14;
      binding.tensor.byte_size = binding.tensor.normalized_extents[0] / 256 *
                                 binding.tensor.normalized_extents[1] *
                                 (query == Encoding::Q5_K ? 176 : 210);
    }
  }
  return facts;
}
class BuilderTest : public ::testing::Test {
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
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
           std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
  }
  loomc_module_t *LoadLayer(const model::ModelDescriptor &facts,
                            Registry *registry) {
    std::string error;
    for (auto entry : QwenAttentionRegistrations())
      EXPECT_TRUE(registry->Add(std::move(entry), &error)) << error;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    QwenNativeGeometry geometry;
    auto status =
        LoadQwenNative(context, workspace, *registry, {Id::QwenAttentionLayer},
                       facts, options, root, &geometry, &module, &result);
    EXPECT_TRUE(Ok(status));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    return module;
  }
  void Compile(loomc_module_t *module, const char *name) {
    loomc_target_profile_t *profile = nullptr;
    ASSERT_TRUE(Ok(loomc_target_profile_select(
        environment, loomc_make_cstring_view("amdgpu:gfx1100"),
        loomc_allocator_system(), &profile)));
    auto symbol = loomc_make_cstring_view(name);
    std::vector<loomc_request_t *> requests;
    loomc_cmd_program_module_options_t build{};
    build.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
    build.structure_size = sizeof(build);
    build.root_symbols = &symbol;
    build.root_symbol_count = 1;
    build.kernel_target_profile = profile;
    build.request_sink.publish = [](void *user, loomc_request_t *request) {
      static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
      return loomc_ok_status();
    };
    build.request_sink.user_data = &requests;
    loomc_product_t *product = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_cmd_program_product_build_module(
        workspace, module, &build, loomc_allocator_system(), &product,
        &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    if (product)
      loomc_product_release(product);
    EXPECT_GE(requests.size(), 10u);
    loomc_compiler_t *compiler = nullptr;
    ASSERT_TRUE(Ok(loomc_compiler_create(context, nullptr,
                                         loomc_allocator_system(), &compiler)));
    for (auto *request : requests) {
      loomc_module_t *leaf = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context, workspace, loomc_request_source(request), nullptr,
          loomc_allocator_system(), &leaf, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      result = nullptr;
      loomc_compile_artifact_options_t compile{};
      compile.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
      compile.structure_size = sizeof(compile);
      compile.target_profile = profile;
      EXPECT_TRUE(Ok(loomc_compile_artifact(compiler, workspace, nullptr, leaf,
                                            &compile, loomc_allocator_system(),
                                            &result)));
      EXPECT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      if (result && loomc_result_succeeded(result)) {
        EXPECT_GT(loomc_result_artifact_count(result), 0u);
      }
      if (result)
        loomc_result_release(result);
      loomc_module_release(leaf);
      loomc_request_release(request);
    }
    loomc_compiler_release(compiler);
    loomc_target_profile_release(profile);
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  QwenNativeOptions options{4, 1, 1, 96, 128, 1};
  std::string root;
};
std::vector<shapes::Operand> Inputs(const Invocation &invocation) {
  std::vector<shapes::Operand> result;
  for (const auto &binding : invocation.bindings())
    result.push_back({binding.port(), binding.value()});
  return result;
}
TEST_F(BuilderTest, ProfileAdmissionUsesActualRolesEncodingsAndResidualResult) {
  for (auto encoding : {Encoding::Q5_K, Encoding::Q6_K}) {
    auto facts = Facts(encoding);
    std::shared_ptr<const qwen::QwenStructure> owner;
    std::string error;
    ASSERT_TRUE(qwen::BuildQwenStructure(facts, {4, 1, 1, 96, "fixture"},
                                         &owner, &error))
        << error;
    const auto &invocation = owner->graph().invocations()[4];
    auto inputs = Inputs(invocation);
    shapes::Result result;
    ASSERT_TRUE(shapes::InferResult(owner->graph(), invocation.opcode(),
                                    invocation.attributes(), inputs, &result,
                                    &error));
    auto entry = QwenAttentionRegistrations()[0];
    InvocationRequest request{owner->graph(), invocation.attributes(), inputs,
                              result};
    EXPECT_TRUE(entry.builder.supports(entry, request, &error)) << error;
    auto changed = std::get<QwenAttentionLayerAttrs>(invocation.attributes());
    changed.epsilon = 1e-5;
    OperationAttrs attrs = changed;
    InvocationRequest wrong{owner->graph(), attrs, inputs, result};
    EXPECT_FALSE(entry.builder.supports(entry, wrong, &error));
    result.alias = owner->inputs().layers[3].key_cache;
    EXPECT_FALSE(entry.builder.supports(entry, request, &error));
  }
}
TEST_F(BuilderTest,
       WholeLayerChecksBankKeysBeforeStagingAndCompilesActualCalls) {
  for (auto encoding : {Encoding::Q5_K, Encoding::Q6_K}) {
    auto facts = Facts(encoding);
    std::shared_ptr<const qwen::QwenStructure> owner;
    std::string error;
    ASSERT_TRUE(qwen::BuildQwenStructure(facts, {4, 1, 1, 96, "fixture"},
                                         &owner, &error))
        << error;
    Registry registry;
    auto *definitions = LoadLayer(facts, &registry);
    ASSERT_NE(definitions, nullptr);
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    loomc_module_release(definitions);
    std::unique_ptr<CommandProgramBuilder> program;
    ASSERT_TRUE(Ok(module->DefineProgram("attention_layer", {}, 6, &program)));
    const auto &invocation = owner->graph().invocations()[4];
    auto logical_inputs = Inputs(invocation);
    shapes::Result result;
    ASSERT_TRUE(shapes::InferResult(owner->graph(), invocation.opcode(),
                                    invocation.attributes(), logical_inputs,
                                    &result, &error));
    const auto *entry = registry.Find(Id::QwenAttentionLayer);
    ASSERT_NE(entry, nullptr);
    InvocationRequest request{owner->graph(), invocation.attributes(),
                              logical_inputs, result};
    ASSERT_TRUE(entry->builder.supports(*entry, request, &error)) << error;
    NativeQwenEpoch epoch;
    epoch.logical_epoch = owner->inputs().epoch;
    epoch.logical_rows = owner->inputs().rows;
    epoch.epoch = program->Binding(3);
    epoch.rows = program->Binding(4);
    epoch.resident_arena = program->Binding(5);
    epoch.options = options;
    for (const auto &layer : owner->inputs().layers) {
      auto add = [&](ValueId value, QwenStateRole role) {
        epoch.states.push_back({value, layer.layer_slot,
                                layer.original_layer_index, layer.ordinal,
                                role});
      };
      if (layer.kind == model::LayerKind::kGatedDeltaNet) {
        add(layer.history, QwenStateRole::GdnHistory);
        add(layer.recurrent, QwenStateRole::GdnMatrix);
      } else {
        add(layer.key_cache, QwenStateRole::Key);
        add(layer.value_cache, QwenStateRole::Value);
      }
    }
    std::vector<NativeOperand> native_inputs;
    for (const auto &binding : invocation.bindings()) {
      NativeOperand operand{binding.port(), binding.value(), {}, {}};
      if (binding.port() == Port::Residual)
        operand.native = program->Binding(2);
      else if (binding.port() == Port::Epoch)
        operand.native = epoch.epoch;
      else if (binding.port() == Port::Rows)
        operand.native = epoch.rows;
      else if (binding.port() == Port::KeyCache ||
               binding.port() == Port::ValueCache)
        operand.native = epoch.resident_arena;
      else {
        operand.native = program->Binding(0);
        const auto &roles = QwenLayerWeightRoles(OpCode::QwenAttentionLayer);
        for (const auto &role : roles)
          if (role.port == binding.port())
            operand.parameter = std::string("blk.3.") + role.suffix;
      }
      native_inputs.push_back(operand);
    }
    CommandValues values(*program);
    BuildContext ctx{request,       *module, *program, values,
                     native_inputs, {},      false};
    ctx.output = program->Binding(2);
    ctx.has_output = true;
    ctx.qwen_epoch = &epoch;
    size_t before = 0;
    ASSERT_TRUE(Ok(program->InspectNative(
        {}, [&](const loom_module_t *m, const std::vector<loom_value_id_t> &) {
          before = m->values.count;
          return iree_ok_status();
        })));
    Value output;
    // Every canonical role must reject an incorrect key or a different bank
    // before allocation. A successful same-owner emission follows below.
    for (auto &weight : native_inputs) {
      if (weight.parameter.empty())
        continue;
      const auto key = weight.parameter;
      weight.parameter += ".wrong";
      EXPECT_FALSE(Ok(entry->builder.build(*entry, ctx, &output)));
      EXPECT_FALSE(ctx.staged);
      weight.parameter = key;
      weight.native = program->Binding(1);
      EXPECT_FALSE(Ok(entry->builder.build(*entry, ctx, &output)));
      EXPECT_FALSE(ctx.staged);
      weight.native = program->Binding(0);
    }
    for (auto &operand : native_inputs) {
      if (operand.port != Port::Residual && operand.port != Port::KeyCache &&
          operand.port != Port::ValueCache)
        continue;
      const auto original = operand.native;
      operand.native = program->Binding(0);
      EXPECT_FALSE(Ok(entry->builder.build(*entry, ctx, &output)));
      EXPECT_FALSE(ctx.staged);
      operand.native = original;
    }
    // Shape-compatible caches all map to one native arena. Their logical
    // associations must still select exactly the requested layer and plane.
    auto changed = std::get<QwenAttentionLayerAttrs>(invocation.attributes());
    ++changed.attention_ordinal;
    OperationAttrs changed_attrs = changed;
    InvocationRequest wrong_ordinal{owner->graph(), changed_attrs,
                                    logical_inputs, result};
    BuildContext ordinal_ctx{wrong_ordinal, *module, *program, values,
                             native_inputs, {},      false};
    ordinal_ctx.output = program->Binding(2);
    ordinal_ctx.has_output = true;
    ordinal_ctx.qwen_epoch = &epoch;
    ASSERT_TRUE(entry->builder.supports(*entry, wrong_ordinal, &error));
    EXPECT_FALSE(Ok(entry->builder.build(*entry, ordinal_ctx, &output)));
    EXPECT_FALSE(ordinal_ctx.staged);
    auto key =
        std::find_if(native_inputs.begin(), native_inputs.end(),
                     [](const auto &o) { return o.port == Port::KeyCache; });
    auto value =
        std::find_if(native_inputs.begin(), native_inputs.end(),
                     [](const auto &o) { return o.port == Port::ValueCache; });
    ASSERT_NE(key, native_inputs.end());
    ASSERT_NE(value, native_inputs.end());
    std::swap(key->logical, value->logical);
    EXPECT_FALSE(Ok(entry->builder.build(*entry, ctx, &output)));
    EXPECT_FALSE(ctx.staged);
    std::swap(key->logical, value->logical);
    const auto original_key = key->logical;
    key->logical = owner->inputs().layers[7].key_cache;
    EXPECT_FALSE(Ok(entry->builder.build(*entry, ctx, &output)));
    EXPECT_FALSE(ctx.staged);
    key->logical = original_key;
    const auto original_value = value->logical;
    value->logical = owner->inputs().layers[7].value_cache;
    EXPECT_FALSE(Ok(entry->builder.build(*entry, ctx, &output)));
    EXPECT_FALSE(ctx.staged);
    value->logical = original_value;
    ASSERT_TRUE(Ok(program->InspectNative(
        {}, [&](const loom_module_t *m, const std::vector<loom_value_id_t> &) {
          EXPECT_EQ(m->values.count, before);
          return iree_ok_status();
        })));
    NativeEmissionSession session(registry, owner->graph(), *module, *program);
    ASSERT_TRUE(
        Ok(session.Bind(owner->inputs().residual, program->Binding(2))));
    ASSERT_TRUE(Ok(session.QwenEpoch(epoch)));
    for (const auto &input : native_inputs)
      if (!input.parameter.empty())
        ASSERT_TRUE(Ok(
            session.Parameter(input.logical, input.native, input.parameter)));
    const auto graph_size = owner->graph().invocations().size();
    ASSERT_TRUE(Ok(session.EmitExisting(Id::QwenAttentionLayer, invocation)));
    ASSERT_TRUE(Ok(session.Native(owner->inputs().residual, &output)));
    EXPECT_EQ(owner->graph().invocations().size(), graph_size);
    for (const auto logical : {owner->inputs().layers[3].key_cache,
                               owner->inputs().layers[3].value_cache}) {
      Value state;
      ASSERT_TRUE(Ok(session.Native(logical, &state)));
      ASSERT_TRUE(Ok(program->InspectNative(
          {state, epoch.resident_arena},
          [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
            EXPECT_EQ(ids[0], ids[1]);
            return iree_ok_status();
          })));
    }
    ASSERT_TRUE(Ok(program->InspectNative(
        {output, program->Binding(2)},
        [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
          EXPECT_EQ(ids[0], ids[1]);
          return iree_ok_status();
        })));
    ASSERT_TRUE(Ok(program->Finish()));
    loomc_module_t *linked = nullptr;
    loomc_result_t *checked = nullptr;
    ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &checked)));
    ASSERT_TRUE(checked && loomc_result_succeeded(checked))
        << Diagnostics(checked);
    loomc_result_release(checked);
    std::string text;
    ASSERT_TRUE(Ok(Dump(linked, &text)));
    for (const auto &role : QwenLayerWeightRoles(OpCode::QwenAttentionLayer))
      EXPECT_NE(text.find(std::string("blk.{}.") + role.suffix),
                std::string::npos);
    Compile(linked, "attention_layer");
    // Both public serialization formats must preserve the complete native
    // program and dependent specialization body, including named weights.
    for (auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      const auto path =
          std::filesystem::path(std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) /
          (std::string(encoding == Encoding::Q5_K ? "layer-q5" : "layer-q6") +
           (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc"));
      loomc_module_serialize_options_t serialization{};
      serialization.format = format;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_path(
          linked, &serialization, loomc_make_cstring_view(path.c_str()),
          loomc_allocator_system())));
      loomc_module_t *reloaded = nullptr;
      checked = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
          context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
          loomc_allocator_system(), &reloaded, &checked)));
      ASSERT_TRUE(checked && loomc_result_succeeded(checked))
          << Diagnostics(checked);
      loomc_result_release(checked);
      ASSERT_NE(reloaded, nullptr);
      Compile(reloaded, "attention_layer");
      loomc_module_release(reloaded);
    }
    loomc_module_release(linked);
  }
}
#ifdef QWEN_ATTENTION_REAL_GGUF_TEST
TEST_F(BuilderTest, ActualQ5HeaderRetainsAllSixDenseLayerRolesAndCompiles) {
  const auto *path = std::getenv("LOOMIFY_QWEN_TEST_GGUF");
  ASSERT_NE(path, nullptr)
      << "explicit immutable default Qwen Q5_K_XL required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     &std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::ModelDescriptor facts;
  std::string error;
  ASSERT_TRUE(model::ExtractQwenConfiguration(file.get(), &facts, &error))
      << error;
  file.reset(); // No checkpoint payload survives into native authoring.
  std::shared_ptr<const qwen::QwenStructure> owner;
  ASSERT_TRUE(qwen::BuildQwenStructure(facts, {4, 1, 1, 96, "actual-header"},
                                       &owner, &error))
      << error;
  Registry registry;
  auto *definitions = LoadLayer(facts, &registry);
  ASSERT_NE(definitions, nullptr);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(
      Ok(module->DefineProgram("actual_header_layers", {}, 5, &program)));
  NativeEmissionSession session(registry, owner->graph(), *module, *program);
  ASSERT_TRUE(Ok(session.Bind(owner->inputs().residual, program->Binding(1))));
  NativeQwenEpoch epoch;
  epoch.logical_epoch = owner->inputs().epoch;
  epoch.logical_rows = owner->inputs().rows;
  epoch.epoch = program->Binding(2);
  epoch.rows = program->Binding(3);
  epoch.resident_arena = program->Binding(4);
  epoch.options = options;
  for (const auto &layer : owner->inputs().layers) {
    auto add = [&](ValueId value, QwenStateRole role) {
      epoch.states.push_back({value, layer.layer_slot,
                              layer.original_layer_index, layer.ordinal, role});
    };
    if (layer.kind == model::LayerKind::kGatedDeltaNet) {
      add(layer.history, QwenStateRole::GdnHistory);
      add(layer.recurrent, QwenStateRole::GdnMatrix);
    } else {
      add(layer.key_cache, QwenStateRole::Key);
      add(layer.value_cache, QwenStateRole::Value);
    }
  }
  ASSERT_TRUE(Ok(session.QwenEpoch(epoch)));
  size_t layer_count = 0, role_count = 0;
  for (const auto &invocation : owner->graph().invocations()) {
    if (invocation.opcode() != OpCode::QwenAttentionLayer)
      continue;
    const auto &attrs =
        std::get<QwenAttentionLayerAttrs>(invocation.attributes());
    const auto prefix = "blk." + std::to_string(attrs.layer_index) + ".";
    for (const auto &role : QwenLayerWeightRoles(invocation.opcode())) {
      const auto *binding = invocation.binding(role.port);
      ASSERT_NE(binding, nullptr);
      const auto key = prefix + role.suffix;
      const auto parameter = std::find_if(
          owner->parameters().begin(), owner->parameters().end(),
          [&](const auto &p) { return p.value == binding->value(); });
      ASSERT_NE(parameter, owner->parameters().end());
      const auto *tensor = owner->parameter(parameter->coordinate);
      ASSERT_NE(tensor, nullptr);
      EXPECT_EQ(tensor->tensor.name, key);
      const auto *type = owner->graph().value(binding->value())->tensorType();
      ASSERT_NE(type, nullptr);
      const auto tag = tensor->tensor.storage_type;
      EXPECT_EQ(type->encoding(), tag == 0    ? Encoding::Dense
                                  : tag == 12 ? Encoding::Q4_K
                                  : tag == 13 ? Encoding::Q5_K
                                              : Encoding::Q6_K);
      EXPECT_TRUE(tag == 0 || tag == 12 || tag == 13 || tag == 14);
      std::printf("ACTUAL_HEADER %s tag=%u bytes=%llu\n", key.c_str(), tag,
                  static_cast<unsigned long long>(tensor->tensor.byte_size));
      ASSERT_TRUE(
          Ok(session.Parameter(binding->value(), program->Binding(0), key)));
      ++role_count;
    }
    ASSERT_TRUE(Ok(session.EmitExisting(Id::QwenAttentionLayer, invocation)));
    ++layer_count;
  }
  EXPECT_EQ(layer_count, 6u);
  EXPECT_EQ(role_count, 66u);
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  loomc_result_t *checked = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &checked)));
  ASSERT_TRUE(checked && loomc_result_succeeded(checked))
      << Diagnostics(checked);
  loomc_result_release(checked);
  Compile(linked, "actual_header_layers");
  loomc_module_release(linked);
}
#endif
} // namespace
} // namespace hrx::dispatch
