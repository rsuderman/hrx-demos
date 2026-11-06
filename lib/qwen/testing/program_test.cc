// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/qwen/program.h"
#include "loomc/compile.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/model/qwen_configuration.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
namespace hrx::qwen {
namespace {
using namespace loomify_gguf;
::testing::AssertionResult Ok(loomc_status_t status) {
  auto message = loomc_status_message(status);
  std::string copy(message.data ? message.data : "", message.size);
  bool ok = loomc_status_is_ok(status);
  loomc_status_free(status);
  return ok ? ::testing::AssertionSuccess()
            : ::testing::AssertionFailure() << copy;
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  if (!result)
    return "no result";
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->message.data, d->message.size);
    text += '\n';
  }
  return text;
}
void Compile(loomc_context_t *context, loomc_workspace_t *workspace,
             loomc_target_environment_t *environment, loomc_module_t *module,
             const char *name) {
  std::vector<loomc_request_t *> requests;
  loomc_target_profile_t *profile = nullptr;
  ASSERT_TRUE(Ok(loomc_target_profile_select(
      environment, loomc_make_cstring_view("amdgpu:gfx1100"),
      loomc_allocator_system(), &profile)));
  auto root = loomc_make_cstring_view(name);
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  options.root_symbols = &root;
  options.root_symbol_count = 1;
  options.kernel_target_profile = profile;
  options.request_sink.publish = [](void *user, loomc_request_t *request) {
    static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
    return loomc_ok_status();
  };
  options.request_sink.user_data = &requests;
  loomc_product_t *product = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(loomc_cmd_program_product_build_module(
      workspace, module, &options, loomc_allocator_system(), &product,
      &result)));
  EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  if (result)
    loomc_result_release(result);
  if (product)
    loomc_product_release(product);
  EXPECT_FALSE(requests.empty());
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
    EXPECT_TRUE(
        Ok(loomc_compile_artifact(compiler, workspace, nullptr, leaf, &compile,
                                  loomc_allocator_system(), &result)));
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

class QwenProgramTest : public ::testing::Test {
protected:
  void SetUp() override {
    const auto *path = std::getenv("LOOMIFY_QWEN_TEST_GGUF");
    ASSERT_NE(path, nullptr) << "explicit immutable Q5_K_XL fixture required";
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                       &std::fclose);
    ASSERT_TRUE(file);
    ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
    std::string error;
    ASSERT_TRUE(model::ExtractQwenConfiguration(file.get(), &facts, &error))
        << error;
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
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
           std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    if (workspace)
      loomc_workspace_release(workspace);
    if (context)
      loomc_context_release(context);
    if (environment)
      loomc_target_environment_release(environment);
  }
  void Load(const dispatch::Registry &registry,
            const dispatch::QwenNativeOptions &options,
            std::unique_ptr<ModuleBuilder> *module,
            dispatch::QwenNativeGeometry *geometry) {
    loomc_module_t *definitions = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(dispatch::LoadQwenNative(
        context, workspace, registry, QwenProgramDispatches(), facts, options,
        root, geometry, &definitions, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, module)));
    loomc_module_release(definitions);
  }
  model::ModelDescriptor facts;
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_F(QwenProgramTest, CompleteRegistryContainsOnlyRealFamilyContracts) {
  auto registry = dispatch::Registry::Qwen08B();
  dispatch::Registry checked;
  std::string error;
  for (auto id : QwenProgramDispatches()) {
    auto implementations = registry.Implementations(id);
    ASSERT_FALSE(implementations.empty());
    for (const auto &entry : implementations) {
      EXPECT_FALSE(entry.symbol.empty());
      EXPECT_FALSE(entry.provider_files.empty());
      EXPECT_TRUE(checked.Add(entry, &error)) << error;
    }
  }
  ASSERT_NE(checked.Find(dispatch::Id::QwenGdnLayer), nullptr);
  ASSERT_NE(checked.Find(dispatch::Id::QwenAttentionLayer), nullptr);
  EXPECT_EQ(checked.Find(dispatch::Id::QwenGdnLayer)->kind,
            dispatch::Kind::CommandProgram);
  EXPECT_EQ(checked.Find(dispatch::Id::QwenAttentionLayer)->kind,
            dispatch::Kind::CommandProgram);
}
TEST_F(QwenProgramTest, CompleteRealGgufRootReloadsAndCompilesBothFormats) {
  auto registry = dispatch::Registry::Qwen08B();
  QwenProgramSettings settings;
  settings.native = {128, 1, 1, 96, 128, 1};
  settings.source_asset = "known-qwen";
  std::unique_ptr<ModuleBuilder> module;
  dispatch::QwenNativeGeometry geometry;
  Load(registry, settings.native, &module, &geometry);
  ASSERT_TRUE(module);
  QwenProgram output;
  std::string error;
  ASSERT_TRUE(DefineQwenProgram(facts, settings, registry, *module,
                                "generated.qwen_epoch", &output, &error))
      << error;
  ASSERT_EQ(output.buffers.size(), 7u);
  EXPECT_EQ(output.buffers[1].minimum_bytes, 2097152u);
  EXPECT_EQ(output.buffers[2].minimum_bytes, 4192u);
  EXPECT_EQ(output.buffers[3].minimum_bytes, 2056u);
  EXPECT_EQ(output.buffers[5].minimum_bytes, 2048u);
  EXPECT_EQ(output.buffers[6].minimum_bytes, 4u);
  EXPECT_EQ(output.structure->graph().invocations().size(), 26u);
  EXPECT_EQ(output.geometry.gdn_layer_count, 18u);
  EXPECT_EQ(output.geometry.attention_layer_count, 6u);
  EXPECT_EQ(output.geometry.kv_plane_stride, 128 * 1024u);
  EXPECT_EQ(output.buffers[4].minimum_bytes,
            geometry.private_row_stride + geometry.kv_pool_bytes);
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({output.program}, &linked, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t options{};
    options.format = format;
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
        linked, &options, loomc_allocator_system(), &source)));
    loomc_module_t *loaded = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(), &loaded,
        &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    loomc_source_release(source);
    Compile(context, workspace, environment, loaded, "generated.qwen_epoch");
    loomc_module_release(loaded);
  }
  loomc_module_release(linked);
}
TEST_F(QwenProgramTest, SingleTokenRootPreservesActualInputViewFootprint) {
  auto registry = dispatch::Registry::Qwen08B();
  QwenProgramSettings settings;
  settings.native = {1, 1, 1, 96, 128, 1};
  settings.source_asset = "known-qwen";
  std::unique_ptr<ModuleBuilder> module;
  dispatch::QwenNativeGeometry geometry;
  Load(registry, settings.native, &module, &geometry);
  ASSERT_TRUE(module);
  QwenProgram output;
  std::string error;
  ASSERT_TRUE(DefineQwenProgram(facts, settings, registry, *module, "one_token",
                                &output, &error))
      << error;
  ASSERT_EQ(output.buffers.size(), 7u);
  EXPECT_EQ(output.structure->graph()
                .value(output.structure->inputs().tokens)
                ->tensorType()
                ->dimension(0),
            1u);
  EXPECT_EQ(output.buffers[5].minimum_bytes, 2048u);
}
TEST_F(QwenProgramTest, LoadedOptionsMismatchDoesNotPublishOutput) {
  auto registry = dispatch::Registry::Qwen08B();
  QwenProgramSettings settings;
  settings.native = {128, 1, 1, 96, 128, 1};
  settings.source_asset = "known-qwen";
  for (auto field : {"pool", "cache", "tokens"}) {
    std::unique_ptr<ModuleBuilder> module;
    dispatch::QwenNativeGeometry geometry;
    Load(registry, settings.native, &module, &geometry);
    ASSERT_TRUE(module);
    auto mismatch = settings;
    if (std::string(field) == "pool") {
      mismatch.native.cache_capacity = 129;
      mismatch.native.pool_capacity = 192;
    }
    if (std::string(field) == "cache")
      mismatch.native.cache_capacity = 128;
    if (std::string(field) == "tokens")
      mismatch.native.token_capacity = 64;
    dispatch::QwenNativeGeometry admitted;
    std::string admission_error;
    ASSERT_TRUE(dispatch::DeriveQwenNativeGeometry(facts, mismatch.native,
                                                   &admitted, &admission_error))
        << admission_error;
    QwenProgram output;
    output.buffers = {{"sentinel", 42}};
    std::string error;
    EXPECT_FALSE(DefineQwenProgram(facts, mismatch, registry, *module,
                                   "mismatch", &output, &error))
        << field;
    ASSERT_EQ(output.buffers.size(), 1u);
    EXPECT_EQ(output.buffers[0].name, "sentinel");
    EXPECT_EQ(output.buffers[0].minimum_bytes, 42u);
    EXPECT_FALSE(output.structure);
  }
}
TEST_F(QwenProgramTest, SameGeometryDifferentEncodingSelectorsDoNotPublish) {
  auto registry = dispatch::Registry::Qwen08B();
  QwenProgramSettings settings;
  settings.native = {128, 1, 1, 96, 128, 1};
  settings.source_asset = "different-encoding-asset";
  for (auto family : {"gate", "down", "attention"}) {
    std::unique_ptr<ModuleBuilder> module;
    dispatch::QwenNativeGeometry geometry;
    Load(registry, settings.native, &module, &geometry);
    ASSERT_TRUE(module);
    auto alternate = facts;
    for (auto &layer : alternate.layers)
      for (auto &binding : layer.tensors) {
        bool gate = std::string(family) == "gate" && layer.index == 0 &&
                    (binding.role == "ffn_gate" || binding.role == "ffn_up");
        bool down = std::string(family) == "down" && layer.index == 0 &&
                    binding.role == "ffn_down";
        bool attention = std::string(family) == "attention" &&
                         layer.index == 3 &&
                         (binding.role == "query" || binding.role == "key");
        if (gate || down || attention) {
          auto &tensor = binding.tensor;
          tensor.storage_type = gate ? 12 : 13;
          tensor.byte_size = uint64_t(tensor.normalized_extents[0]) / 256 *
                             tensor.normalized_extents[1] * (gate ? 144 : 176);
        }
      }
    dispatch::QwenNativeGeometry admitted;
    std::string error;
    ASSERT_TRUE(dispatch::DeriveQwenNativeGeometry(alternate, settings.native,
                                                   &admitted, &error))
        << error;
    QwenProgram output;
    output.buffers = {{"sentinel", 42}};
    EXPECT_FALSE(DefineQwenProgram(alternate, settings, registry, *module,
                                   "selectors", &output, &error))
        << family;
    ASSERT_EQ(output.buffers.size(), 1u);
    EXPECT_EQ(output.buffers[0].name, "sentinel");
    EXPECT_EQ(output.buffers[0].minimum_bytes, 42u);
    EXPECT_FALSE(output.structure);
  }
}
} // namespace
} // namespace hrx::qwen
