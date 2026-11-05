// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/program.h"
#include "lib/llama/testing/model_fixture.h"
#include "loomc/compile.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/model/llama_configuration.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
namespace hrx::llama {
namespace {
using namespace operations;
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

class ModelProgramTest : public ::testing::Test {
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
    if (workspace)
      loomc_workspace_release(workspace);
    if (context)
      loomc_context_release(context);
    if (environment)
      loomc_target_environment_release(environment);
  }
  void Emit(const model::ModelDescriptor &facts, bool compile,
            uint64_t pool = 0) {
    auto registry = dispatch::Registry::Llama3B();
    loomc_module_t *definitions = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(dispatch::LoadLlamaNative(
        context, workspace, registry, LlamaProgramDispatches(), root,
        {uint64_t(facts.layers.size()), 512, pool}, &definitions, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    loomc_module_release(definitions);
    std::vector<RootRef> roots;
    std::vector<LlamaProgram> retained;
    std::string error;
    for (auto phase : {Phase::Prefill, Phase::Decode}) {
      LlamaProgramSettings settings;
      settings.phase = phase;
      settings.row_capacity = phase == Phase::Decode ? 1 : 8;
      settings.cache_capacity = 512;
      settings.pool_capacity = pool;
      settings.map_origin = pool ? 2048 : 0;
      settings.cache_base = 512;
      settings.source_asset = "fixture";
      LlamaProgram program;
      ASSERT_TRUE(DefineLlamaProgram(facts, settings, registry, *module,
                                     phase == Phase::Decode ? "model.decode"
                                                            : "model.prefill",
                                     &program, &error))
          << error;
      ASSERT_EQ(program.buffers.size(), 7u);
      EXPECT_EQ(program.buffers[5].minimum_bytes, 516u);
      EXPECT_EQ(program.buffers[4].minimum_bytes,
                512 + 4 * facts.layers.size() * (pool ? pool : 512) * 8 * 128);
      EXPECT_EQ(program.structure->graph().invocations().size(),
                1 + 11 * facts.layers.size() + 2);
      roots.push_back(program.program);
      retained.push_back(std::move(program));
    }
    loomc_module_t *linked = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(module->LinkAndVerify(roots, &linked, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    for (auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options{};
      options.format = format;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          linked, &options, loomc_allocator_system(), &source)));
      loomc_module_t *loaded = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context, workspace, source, nullptr, loomc_allocator_system(),
          &loaded, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      loomc_source_release(source);
      if (compile) {
        Compile(context, workspace, environment, loaded, "model.prefill");
        Compile(context, workspace, environment, loaded, "model.decode");
      }
      loomc_module_release(loaded);
    }
    loomc_module_release(linked);
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_F(ModelProgramTest, CompleteRootsReloadAndCompile) {
  for (size_t layers : {1, 2, 28})
    Emit(testing::ModelFacts(layers), true);
}
TEST_F(ModelProgramTest, PooledMapOriginsAndPhysicalTokenCapacity) {
  Emit(testing::ModelFacts(2), true, 1024);
}
TEST_F(ModelProgramTest,
       RejectsMismatchedLoadedConfigurationWithoutPublishing) {
  auto registry = dispatch::Registry::Llama3B();
  auto facts = testing::ModelFacts(2);
  for (uint64_t loaded_pool : {0, 1024}) {
    loomc_module_t *definitions = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(dispatch::LoadLlamaNative(
        context, workspace, registry, LlamaProgramDispatches(), root,
        {2, 512, loaded_pool}, &definitions, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    for (auto mismatch : {"pool", "cache", "layers"}) {
      std::unique_ptr<ModuleBuilder> module;
      ASSERT_TRUE(
          Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
      LlamaProgramSettings settings;
      settings.row_capacity = 8;
      settings.cache_capacity = 512;
      settings.pool_capacity = loaded_pool;
      settings.source_asset = "fixture";
      auto configured_facts = facts;
      if (std::string(mismatch) == "pool")
        settings.pool_capacity = loaded_pool ? 0 : 1024;
      if (std::string(mismatch) == "cache")
        settings.cache_capacity = 256;
      if (std::string(mismatch) == "layers")
        configured_facts.layers.pop_back();
      LlamaProgram output;
      output.buffers = {{"unchanged", 42}};
      std::string error;
      EXPECT_FALSE(DefineLlamaProgram(configured_facts, settings, registry,
                                      *module, "mismatch", &output, &error));
      EXPECT_NE(error.find("loaded attention configuration"), std::string::npos)
          << error;
      ASSERT_EQ(output.buffers.size(), 1u);
      EXPECT_EQ(output.buffers[0].name, "unchanged");
      EXPECT_EQ(output.buffers[0].minimum_bytes, 42u);
      EXPECT_FALSE(output.structure);
    }
    if (loaded_pool) {
      std::unique_ptr<ModuleBuilder> module;
      ASSERT_TRUE(
          Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
      LlamaProgramSettings settings;
      settings.row_capacity = 8;
      settings.cache_capacity = 512;
      settings.pool_capacity = 64;
      settings.source_asset = "fixture";
      LlamaProgram output;
      output.buffers = {{"unchanged", 42}};
      std::string error;
      EXPECT_FALSE(DefineLlamaProgram(facts, settings, registry, *module,
                                      "undersized", &output, &error));
      EXPECT_NE(error.find("loaded attention configuration"), std::string::npos)
          << error;
      ASSERT_EQ(output.buffers.size(), 1u);
      EXPECT_EQ(output.buffers[0].minimum_bytes, 42u);
    }
    loomc_module_release(definitions);
  }
}
TEST_F(ModelProgramTest, KnownGgufUsesSameFactoryAndLowerer) {
  const auto *path = std::getenv("LOOMIFY_TEST_GGUF");
  if (!path)
    GTEST_SKIP() << "explicit immutable GGUF gate";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     &std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::ModelDescriptor facts;
  std::string error;
  ASSERT_TRUE(model::ExtractLlamaConfiguration(file.get(), &facts, &error))
      << error;
  file.reset();
  ASSERT_EQ(facts.layers.size(), 28u);
  Emit(facts, true);
}
} // namespace
} // namespace hrx::llama
