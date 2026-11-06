// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/dispatch/qwen.h"
#include "lib/qwen/structure.h"
#include "lib/qwen/testing/model_fixture.h"
#include "loom/ops/config/ops.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
namespace hrx::dispatch {
namespace {
using namespace loomify_gguf;
model::ModelDescriptor Facts() {
  auto m = qwen::testing::ModelFacts();
  for (auto &l : m.layers)
    if (l.kind == model::LayerKind::kAttention)
      for (auto &t : l.tensors)
        if (t.role == "key") {
          t.tensor.storage_type = 14;
          t.tensor.byte_size = 1024 / 256 * 512 * 210;
        }
  return m;
}
QwenNativeOptions Options() { return {128, 1, 1, 65, 128, 1}; }
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
class QwenConfigurationTest : public ::testing::Test {
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
  Registry Witness() {
    Registration r;
    r.id = Id::RmsNorm;
    r.operation = operations::OpCode::RmsNorm;
    r.kind = Kind::CommandProgram;
    r.name = "test.qwen.config";
    r.symbol = r.interface_symbol = "test.registered_program";
    r.interface_file = "lib/dispatch/testing/testdata/program_interface.loom";
    r.provider_files = {"lib/dispatch/testing/testdata/program_provider.loom"};
    r.required_config = {
        "qwen.model.hidden_size",         "qwen.model.ffn_gate_q4_layers",
        "qwen.model.ffn_gate_q6_layers",  "qwen.model.ffn_down_q6_layers",
        "qwen.model.attention_q6_layers", "qwen38.attention.cache_capacity",
        "qwen38.attention.pool_capacity", "qwen.model.hidden_capacity_bytes"};
    r.builder.operation = r.operation;
    r.builder.kind = r.kind;
    r.builder.supports = [](const Registration &, const InvocationRequest &,
                            std::string *) { return true; };
    r.builder.build = [](const Registration &, BuildContext &, Value *) {
      return loomc_ok_status();
    };
    Registry registry;
    std::string error;
    EXPECT_TRUE(registry.Add(r, &error)) << error;
    return registry;
  }
  int64_t Integer(loomc_module_t *module, const char *key) {
    loomc_module_interop_view_t view{};
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_get_interop_view(
        module, loomc_allocator_system(), &view, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result));
    if (result)
      loomc_result_release(result);
    for (size_t i = 0; i < view.module->symbols.count; ++i) {
      const auto &s = view.module->symbols.entries[i];
      auto name = loom_string_table_get(&view.module->strings, s.name_id);
      if (std::string(name.data, name.size) == key &&
          loom_config_def_isa(s.defining_op)) {
        const auto value = loom_config_def_value(s.defining_op);
        EXPECT_EQ(value.kind, LOOM_ATTR_I64);
        return value.i64;
      }
    }
    ADD_FAILURE() << "missing config " << key;
    return -1;
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST(QwenNativeGeometry, LogicalCacheAndPhysicalPoolRemainDistinct) {
  auto facts = Facts();
  auto options = Options();
  QwenNativeGeometry g;
  std::string error;
  ASSERT_TRUE(DeriveQwenNativeGeometry(facts, options, &g, &error)) << error;
  EXPECT_EQ(g.residual_bytes, 2097152u);
  EXPECT_EQ(g.gdn_history_bytes, 73728u);
  EXPECT_EQ(g.gdn_recurrent_bytes, 1048576u);
  EXPECT_EQ(g.gdn_layer_bytes, 1122304u);
  EXPECT_EQ(g.gdn_total_bytes, 20201472u);
  EXPECT_EQ(g.gdn_slot_origin, 256u);
  EXPECT_EQ(g.private_row_stride, 20204032u);
  EXPECT_EQ(g.kv_plane_count, 12u);
  EXPECT_EQ(g.kv_plane_stride, 131072u);
  EXPECT_EQ(g.kv_pool_bytes, 1572864u);
  EXPECT_EQ(g.map_origin, 2048u);
  options.token_capacity = 512;
  ASSERT_TRUE(DeriveQwenNativeGeometry(facts, options, &g, &error)) << error;
  EXPECT_EQ(g.residual_bytes, 2097152u);
}
TEST(QwenNativeGeometry, InvalidProfileRolesAndByteRangesLeaveOutputUnchanged) {
  for (int invalid = 0; invalid < 9; ++invalid) {
    auto facts = Facts();
    auto options = Options();
    if (invalid == 0)
      options.token_capacity = 513;
    if (invalid == 1)
      options.pool_capacity = 65;
    if (invalid == 2)
      facts.embedding_width = 3072;
    if (invalid == 3)
      facts.layers[0].recurrent_ordinal = 1;
    if (invalid == 4)
      facts.layers[0].tensors[2].tensor.byte_size -= 1;
    if (invalid == 5)
      facts.layers[0].tensors[3].tensor.name = "wrong.weight";
    if (invalid == 6)
      facts.globals[2].tensor.absolute_offset += 1;
    if (invalid == 7)
      facts.layers[0].tensors[0].tensor.absolute_offset = UINT64_MAX;
    if (invalid == 8)
      facts.layers[0].tensors.push_back(facts.layers[0].tensors[0]);
    QwenNativeGeometry g;
    g.residual_bytes = 123;
    std::string error;
    EXPECT_FALSE(DeriveQwenNativeGeometry(facts, options, &g, &error))
        << invalid;
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(g.residual_bytes, 123u);
  }
}
TEST_F(QwenConfigurationTest,
       LoaderDerivesActualEncodingSelectorsAndOneCapacityAuthority) {
  auto facts = Facts();
  auto options = Options();
  auto registry = Witness();
  for (int variant = 0; variant < 2; ++variant) {
    if (variant) {
      for (auto &t : facts.layers[0].tensors)
        if (t.role == "ffn_gate" || t.role == "ffn_up") {
          t.tensor.storage_type = 14;
          t.tensor.byte_size = 1024 / 256 * 3584 * 210;
        }
    }
    QwenNativeGeometry g;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(LoadQwenNative(context, workspace, registry,
                                  {Id::RmsNorm, Id::RmsNorm}, facts, options,
                                  root, &g, &module, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result));
    EXPECT_EQ(Integer(module, "qwen.model.ffn_gate_q4_layers"),
              2396745 - variant);
    EXPECT_EQ(Integer(module, "qwen.model.ffn_gate_q6_layers"),
              9586980 + variant);
    EXPECT_EQ(Integer(module, "qwen.model.ffn_down_q6_layers"), 11184810);
    EXPECT_EQ(Integer(module, "qwen.model.attention_q6_layers"), 8947848);
    EXPECT_EQ(Integer(module, "qwen.model.ffn_gate_q5_weight_bytes"), 2523136);
    EXPECT_EQ(Integer(module, "qwen38.attention.cache_capacity"), 65);
    EXPECT_EQ(Integer(module, "qwen38.attention.pool_capacity"), 128);
    EXPECT_EQ(Integer(module, "qwen.model.hidden_capacity_bytes"), 2097152);
    EXPECT_EQ(g.kv_plane_stride, 131072u);
    loomc_module_release(module);
    loomc_result_release(result);
  }
}
TEST_F(QwenConfigurationTest, OwnedSharedClosuresLoadWithoutSeedProviders) {
  const auto sources = QwenSharedSources();
  ASSERT_FALSE(sources.empty());
  for (const auto &source : sources)
    for (const auto &file : source.provider_files)
      EXPECT_EQ(file.find("loom-pipeline"), std::string::npos);
  auto registry = Witness();
  for (const auto &source : sources) {
    // This is an actual native root/closure witness, not a Qwen layer stub.
    auto entry = registry.Implementations(Id::RmsNorm).front();
    entry.symbol = source.symbol;
    entry.kind = source.kind;
    entry.interface_file = source.interface_file;
    entry.provider_files = source.provider_files;
    entry.required_config = source.required_config;
    entry.interface_symbol = source.interface_symbol;
    entry.builder.kind = source.kind;
    entry.builder.required_definitions = source.required_definitions;
    Registry only;
    std::string error;
    ASSERT_TRUE(only.Add(entry, &error)) << error;
    QwenNativeGeometry geometry;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(
        Ok(LoadQwenNative(context, workspace, only, {Id::RmsNorm}, Facts(),
                          Options(), root, &geometry, &module, &result)))
        << source.symbol;
    if (result && !loomc_result_succeeded(result)) {
      for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
        const auto *d = loomc_result_diagnostic_at(result, i);
        ADD_FAILURE() << source.symbol << ": "
                      << std::string(d->message.data, d->message.size);
      }
    }
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << source.symbol;
    loomc_module_release(module);
    loomc_result_release(result);
  }
}
TEST_F(QwenConfigurationTest,
       CommonRegistryLoadsActualEmbeddingAndSelectedHead) {
  Registry registry;
  std::string error;
  for (const auto &entry : QwenCommonRegistrations())
    ASSERT_TRUE(registry.Add(entry, &error)) << error;
  ASSERT_NE(registry.Find(Id::Embedding), nullptr);
  ASSERT_NE(registry.Find(Id::QwenEpochHead), nullptr);
  EXPECT_EQ(registry.Find(Id::QwenGdnLayer), nullptr);
  EXPECT_EQ(registry.Find(Id::QwenAttentionLayer), nullptr);
  QwenNativeGeometry geometry;
  loomc_module_t *loaded = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(LoadQwenNative(context, workspace, registry,
                                {Id::Embedding, Id::QwenEpochHead}, Facts(),
                                Options(), root, &geometry, &loaded, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, loaded, &module)));
  loomc_module_release(loaded);
  loomc_result_release(result);
  KernelRef embedding;
  ProgramRef head;
  ASSERT_TRUE(
      Ok(module->Kernel(registry.Find(Id::Embedding)->symbol, &embedding)));
  ASSERT_TRUE(
      Ok(module->Program(registry.Find(Id::QwenEpochHead)->symbol, &head)));
  const auto i = module->Scalar(LOOM_SCALAR_TYPE_INDEX), b = module->Buffer();
  EXPECT_TRUE(Ok(module->CheckKernelInterface(embedding, {i}, {b, b, b})));
  EXPECT_TRUE(Ok(module->CheckProgramInterface(head, {i, i}, {b, b, b, b})));
}
TEST_F(QwenConfigurationTest,
       EpochStateMappingsAreOwnedAndValidatedBeforeStaging) {
  using namespace operations;
  for (uint64_t cache : {65u, 96u}) {
    auto options = Options();
    options.cache_capacity = cache;
    auto registry = Witness();
    QwenNativeGeometry geometry;
    loomc_module_t *loaded = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(
        Ok(LoadQwenNative(context, workspace, registry, {Id::RmsNorm}, Facts(),
                          options, root, &geometry, &loaded, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result));
    std::shared_ptr<const qwen::QwenStructure> owner;
    std::string error;
    ASSERT_TRUE(qwen::BuildQwenStructure(Facts(), {128, 1, 1, cache, "test"},
                                         &owner, &error))
        << error;
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, loaded, &module)));
    loomc_module_release(loaded);
    loomc_result_release(result);
    const auto &graph = owner->graph();
    const auto graph_count = graph.invocations().size();
    for (int invalid = -1; invalid < 9; ++invalid) {
      std::unique_ptr<CommandProgramBuilder> program, foreign;
      ASSERT_TRUE(Ok(module->DefineProgram(
          "epoch_" + std::to_string(invalid + 1), {}, 3, &program)));
      ASSERT_TRUE(Ok(module->DefineProgram(
          "foreign_" + std::to_string(invalid + 1), {}, 1, &foreign)));
      NativeEmissionSession session(registry, graph, *module, *program);
      NativeQwenEpoch epoch;
      epoch.logical_epoch = owner->inputs().epoch;
      epoch.logical_rows = owner->inputs().rows;
      epoch.epoch = program->Binding(0);
      epoch.rows = program->Binding(1);
      epoch.resident_arena = program->Binding(2);
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
      const auto good = epoch;
      if (invalid == 0)
        epoch.logical_epoch = owner->inputs().tokens;
      if (invalid == 1)
        epoch.epoch = foreign->Binding(0);
      if (invalid == 2)
        epoch.rows = epoch.epoch;
      if (invalid == 3)
        epoch.options.token_capacity = 64;
      if (invalid == 4) {
        epoch.options.cache_capacity = 66;
      }
      if (invalid == 5)
        epoch.states[0].ordinal = 1;
      if (invalid == 6)
        epoch.states[0].original_layer_index = 1;
      if (invalid == 7)
        epoch.states[0].value = owner->inputs().residual;
      if (invalid == 8)
        epoch.states.pop_back();
      auto status = session.QwenEpoch(epoch);
      EXPECT_EQ(loomc_status_is_ok(status), invalid == -1) << invalid;
      loomc_status_free(status);
      if (invalid != -1) {
        ASSERT_TRUE(Ok(session.QwenEpoch(good))) << invalid;
      }
      for (const auto &state : good.states) {
        Value native;
        ASSERT_TRUE(Ok(session.Native(state.value, &native)));
        ASSERT_TRUE(Ok(program->InspectNative(
            {native, good.resident_arena},
            [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
              return ids[0] == ids[1]
                         ? iree_ok_status()
                         : iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
            })));
        auto bind = session.Bind(state.value, program->Binding(0));
        EXPECT_FALSE(loomc_status_is_ok(bind));
        loomc_status_free(bind);
      }
      auto duplicate = session.QwenEpoch(good);
      EXPECT_FALSE(loomc_status_is_ok(duplicate));
      loomc_status_free(duplicate);
      EXPECT_EQ(graph.invocations().size(), graph_count);
    }
  }
}
// Transport witness only: it deliberately performs no layer arithmetic.
TEST_F(QwenConfigurationTest, MultipleStateWritesRetainExplicitResidualResult) {
  using namespace operations;
  auto witness = Witness();
  QwenNativeGeometry geometry;
  loomc_module_t *loaded = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(
      Ok(LoadQwenNative(context, workspace, witness, {Id::RmsNorm}, Facts(),
                        Options(), root, &geometry, &loaded, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result));
  std::shared_ptr<const qwen::QwenStructure> owner;
  std::string error;
  ASSERT_TRUE(qwen::BuildQwenStructure(Facts(), {128, 1, 1, 65, "test"}, &owner,
                                       &error))
      << error;
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, loaded, &module)));
  loomc_module_release(loaded);
  loomc_result_release(result);
  auto entry = witness.Implementations(Id::RmsNorm).front();
  entry.id = Id::QwenGdnLayer;
  entry.operation = entry.builder.operation = OpCode::QwenGdnLayer;
  entry.operands.clear();
  for (const auto &port :
       owner->graph().definition(OpCode::QwenGdnLayer)->ports())
    entry.operands.push_back({port.port(), port.access()});
  entry.builder.operands = entry.operands;
  entry.builder.build = [](const Registration &, BuildContext &context,
                           Value *output) {
    EXPECT_NE(context.qwen_epoch, nullptr);
    EXPECT_TRUE(context.has_output);
    const auto *residual = context.Operand(Port::Residual);
    const auto *history = context.Operand(Port::ConvolutionHistory);
    EXPECT_NE(residual, nullptr);
    EXPECT_NE(history, nullptr);
    auto status = context.program.InspectNative(
        {context.output, residual->native, history->native},
        [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
          return ids[0] == ids[1] && ids[0] != ids[2]
                     ? iree_ok_status()
                     : iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
        });
    if (loomc_status_is_ok(status))
      *output = context.output;
    return status;
  };
  Registry registry;
  ASSERT_TRUE(registry.Add(entry, &error)) << error;
  auto failing = entry;
  failing.variant = "staged_failure";
  failing.builder.build = [](const Registration &, BuildContext &context,
                             Value *) {
    context.staged = true;
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "test staged failure"));
  };
  ASSERT_TRUE(registry.Add(failing, &error)) << error;
  for (bool fail : {false, true}) {
    std::unique_ptr<CommandProgramBuilder> program;
    ASSERT_TRUE(Ok(module->DefineProgram(
        fail ? "transport_failure" : "transport_alias", {}, 6, &program)));
    NativeEmissionSession session(registry, owner->graph(), *module, *program);
    NativeQwenEpoch epoch{owner->inputs().epoch, owner->inputs().rows,
                          program->Binding(2),   program->Binding(3),
                          program->Binding(4),   Options()};
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
    ASSERT_TRUE(Ok(session.QwenEpoch(epoch)));
    ASSERT_TRUE(
        Ok(session.Bind(owner->inputs().residual, program->Binding(1))));
    std::vector<ValueId> seen;
    for (const auto &association : owner->parameters()) {
      if (std::find(seen.begin(), seen.end(), association.value) != seen.end())
        continue;
      ASSERT_TRUE(Ok(session.Parameter(
          association.value, program->Binding(0),
          owner->parameter(association.coordinate)->tensor.name)));
      seen.push_back(association.value);
    }
    const auto count = owner->graph().invocations().size();
    auto status =
        session.EmitExisting(Id::QwenGdnLayer, owner->graph().invocations()[1],
                             fail ? "staged_failure" : "default");
    EXPECT_EQ(loomc_status_is_ok(status), !fail);
    loomc_status_free(status);
    Value native;
    auto get = session.Native(owner->inputs().residual, &native);
    EXPECT_EQ(loomc_status_is_ok(get), !fail);
    loomc_status_free(get);
    if (fail) {
      auto retry = session.EmitExisting(Id::QwenGdnLayer,
                                        owner->graph().invocations()[1]);
      EXPECT_FALSE(loomc_status_is_ok(retry));
      loomc_status_free(retry);
    }
    EXPECT_EQ(owner->graph().invocations().size(), count);
  }
}
TEST_F(QwenConfigurationTest, LayerSelectorsMustMatchLoadedConfiguration) {
  using namespace operations;
  auto registry = Witness();
  QwenNativeGeometry geometry;
  loomc_module_t *loaded = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(
      Ok(LoadQwenNative(context, workspace, registry, {Id::RmsNorm}, Facts(),
                        Options(), root, &geometry, &loaded, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, loaded, &module)));
  loomc_module_release(loaded);
  loomc_result_release(result);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("selector_check", {}, 0, &program)));
  CommandValues values(*program);
  for (uint32_t layer : {0u, 3u}) {
    for (int mismatch : {-1, 0, 1, 2}) {
      if (layer == 0 && mismatch == 2)
        continue;
      OperationRegistry operations;
      OperationGraph graph(operations);
      std::string error;
      auto value = [&](Encoding e) {
        auto type = TensorType::matrix(256, 256, ElementType::F32);
        type.setEncoding(e).setLayout(Layout::GgmlWeightRows);
        return graph.tensor(type, &error);
      };
      const auto gate = mismatch == 0 ? Encoding::Q5_K : Encoding::Q4_K;
      const auto down =
          (layer == 0) != (mismatch == 1) ? Encoding::Q5_K : Encoding::Q6_K;
      const auto qk = mismatch == 2 ? Encoding::Q5_K : Encoding::Q6_K;
      std::vector<shapes::Operand> logical{{Port::GateWeight, value(gate)},
                                           {Port::UpWeight, value(gate)},
                                           {Port::DownWeight, value(down)},
                                           {Port::QueryWeight, value(qk)},
                                           {Port::KeyWeight, value(qk)}};
      QwenGdnLayerAttrs gdn;
      gdn.layer_index = layer;
      QwenAttentionLayerAttrs attention;
      attention.layer_index = layer;
      OperationAttrs attributes =
          layer == 0 ? OperationAttrs(gdn) : OperationAttrs(attention);
      shapes::Result shape{TensorType::vector(1, ElementType::F32)};
      InvocationRequest request{graph, attributes, logical, shape};
      std::vector<NativeOperand> native;
      BuildContext build{request, *module, *program, values, native, {}, false};
      auto status = CheckQwenLayerConfiguration(build);
      EXPECT_EQ(loomc_status_is_ok(status), mismatch == -1)
          << layer << ":" << mismatch;
      loomc_status_free(status);
      EXPECT_FALSE(build.staged);
    }
  }
}
TEST_F(QwenConfigurationTest, LayerParametersRequireExactKeysAndOneScopedRoot) {
  using namespace operations;
  auto registry = Witness();
  QwenNativeGeometry geometry;
  loomc_module_t *loaded = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(
      Ok(LoadQwenNative(context, workspace, registry, {Id::RmsNorm}, Facts(),
                        Options(), root, &geometry, &loaded, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, loaded, &module)));
  loomc_module_release(loaded);
  loomc_result_release(result);
  std::unique_ptr<CommandProgramBuilder> program, foreign;
  ASSERT_TRUE(Ok(module->DefineProgram("parameter_check", {}, 2, &program)));
  ASSERT_TRUE(Ok(module->DefineProgram("foreign_check", {}, 1, &foreign)));
  CommandValues values(*program);
  OperationRegistry operations;
  OperationGraph graph(operations);
  OperationAttrs attributes = RmsNormAttrs{};
  std::vector<shapes::Operand> logical;
  shapes::Result shape{TensorType::vector(1, ElementType::F32)};
  InvocationRequest request{graph, attributes, logical, shape};
  std::vector<NativeOperand> inputs{
      {Port::Gamma, {}, program->Binding(0), "blk.0.attn_norm.weight"},
      {Port::Weight, {}, program->Binding(0), "blk.0.attn_q.weight"}};
  BuildContext build{request, *module, *program, values, inputs, {}, false};
  const std::vector<std::pair<Port, std::string>> roles{
      {Port::Gamma, "blk.0.attn_norm.weight"},
      {Port::Weight, "blk.0.attn_q.weight"}};
  Value output = program->Binding(1);
  ASSERT_TRUE(Ok(CheckQwenLayerParameters(build, roles, &output)));
  auto reject = [&](auto contract) {
    output = program->Binding(1);
    auto status = CheckQwenLayerParameters(build, contract, &output);
    EXPECT_FALSE(loomc_status_is_ok(status));
    loomc_status_free(status);
    EXPECT_FALSE(build.staged);
    EXPECT_TRUE(Ok(program->InspectNative(
        {output, program->Binding(1)},
        [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
          return ids[0] == ids[1]
                     ? iree_ok_status()
                     : iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
        })));
  };
  inputs[1].parameter = "blk.1.attn_q.weight";
  reject(roles);
  inputs[1].parameter = roles[1].second;
  inputs[1].native = program->Binding(1);
  reject(roles);
  inputs[1].native = foreign->Binding(0);
  reject(roles);
  inputs[1].native = program->Binding(0);
  reject(std::vector<std::pair<Port, std::string>>{roles[0]});
  reject(std::vector<std::pair<Port, std::string>>{roles[0], roles[0]});
  inputs.push_back(inputs[1]);
  reject(roles);
  inputs.pop_back();
  ASSERT_TRUE(Ok(CheckQwenLayerParameters(build, roles, &output)));
  EXPECT_FALSE(build.staged);
}
TEST_F(QwenConfigurationTest,
       MissingRegistrationAndInvalidDescriptorNeverPublishGeometry) {
  auto facts = Facts();
  auto options = Options();
  auto registry = Witness();
  for (bool invalid_descriptor : {false, true}) {
    if (invalid_descriptor)
      facts.rms_epsilon = 1e-5;
    QwenNativeGeometry g;
    g.residual_bytes = 123;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status =
        LoadQwenNative(context, workspace, registry,
                       {invalid_descriptor ? Id::RmsNorm : Id::QwenGdnLayer},
                       facts, options, root, &g, &module, &result);
    EXPECT_FALSE(loomc_status_is_ok(status));
    loomc_status_free(status);
    EXPECT_EQ(module, nullptr);
    EXPECT_EQ(g.residual_bytes, 123u);
    if (result)
      loomc_result_release(result);
  }
}
} // namespace
} // namespace hrx::dispatch
