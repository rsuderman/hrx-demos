// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/target/amdgpu.h"
#include "tools/loomify-gguf/serving/epoch.h"
#include "tools/loomify-gguf/serving/package.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
namespace hrx::loomify_gguf {
namespace {
::testing::AssertionResult Ok(loomc_status_t s) {
  if (loomc_status_is_ok(s)) {
    loomc_status_free(s);
    return ::testing::AssertionSuccess();
  }
  auto m = loomc_status_message(s);
  std::string text(m.data, m.size);
  loomc_status_free(s);
  return ::testing::AssertionFailure() << text;
}
std::string Diagnostics(loomc_result_t *r) {
  std::string text;
  if (r)
    for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
      auto d = loomc_result_diagnostic_at(r, i);
      text.append(d->message.data, d->message.size);
      text += '\n';
    }
  return text;
}
std::string Read(const std::filesystem::path &p) {
  std::ifstream in(p);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}
class ServingTest : public ::testing::Test {
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
    directory = std::filesystem::path(std::getenv("TEST_TMPDIR")) /
                ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::create_directory(directory);
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
  }
  bool Load(const std::filesystem::path &p, loomc_module_t **out) {
    loomc_result_t *r = nullptr;
    auto s = loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(p.c_str()), nullptr,
        loomc_allocator_system(), out, &r);
    bool ok = loomc_status_is_ok(s) && r && loomc_result_succeeded(r);
    if (!ok)
      ADD_FAILURE() << Diagnostics(r);
    loomc_status_free(s);
    loomc_result_release(r);
    return ok;
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::filesystem::path directory;
};
TEST_F(ServingTest, EpochConstructionAndSerialization) {
  loomc_module_t *definitions = nullptr;
  ASSERT_TRUE(Load("tools/loomify-gguf/serving/epoch.loom", &definitions));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> prefill;
  ASSERT_TRUE(Ok(module->DefineProgram("test.prefill", {}, 7, &prefill)));
  auto step = prefill->Ref();
  ASSERT_TRUE(Ok(prefill->Finish()));
  prefill.reset();
  ProgramRef epoch;
  ASSERT_TRUE(Ok(DefineLlamaEpoch(*module, step, 8, "test.epoch", &epoch)));
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({epoch}, &linked, &result)));
  ASSERT_TRUE(loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  loomc_module_serialize_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  auto path = directory / "model.loombc";
  ASSERT_TRUE(Ok(loomc_module_serialize_to_path(
      linked, &options, loomc_make_cstring_view(path.c_str()),
      loomc_allocator_system())));
  loomc_module_release(linked);
  ASSERT_TRUE(Load(path, &linked));
  loomc_module_release(linked);
}
TEST_F(ServingTest, PackageSourcesAndExternalAssets) {
  loomc_module_t *definitions = nullptr;
  ASSERT_TRUE(Load("tools/loomify-gguf/serving/epoch.loom", &definitions));
  model::ModelDescriptor model;
  model.embedding_width = 3072;
  model.key_head_width = 128;
  model.layers.resize(28);
  model.layers.front().kv_heads = 8;
  model.layers.front().feed_forward_width = 8192;
  auto weights = directory / "external weights.gguf";
  std::ofstream(weights) << "external";
  auto tokenizer = directory / "tokenizer.json";
  std::ofstream(tokenizer) << "{}";
  ModelPackageOptions o;
  o.source_root = ".";
  o.output_directory = (directory / "package").string();
  o.gguf = weights;
  o.tokenizer = tokenizer;
  o.prefill_capacity = 8;
  std::string error;
  ASSERT_TRUE(WriteLlamaPackage(model, o, definitions, &error)) << error;
  auto prepare =
      Read(std::filesystem::path(o.output_directory) / "prepare.loom");
  EXPECT_EQ(prepare.find("{{"), std::string::npos);
  EXPECT_NE(prepare.find("%history_capacity = scalar.constant 129"),
            std::string::npos);
  EXPECT_NE(prepare.find("%attention_origin = scalar.constant 512"),
            std::string::npos);
  EXPECT_NE(prepare.find("%one, %attention_base, %attention_bytes"),
            std::string::npos);
  auto manifest =
      Read(std::filesystem::path(o.output_directory) / "model.json");
  EXPECT_NE(manifest.find("external weights.gguf"), std::string::npos);
  EXPECT_NE(manifest.find("\"sources\": [\"model.loombc\"]"),
            std::string::npos);
  for (const char *file : {"prepare.loom", "control.loom", "weights.loom"}) {
    loomc_module_t *provider = nullptr;
    ASSERT_TRUE(
        Load(std::filesystem::path(o.output_directory) / file, &provider))
        << file;
    loomc_module_release(provider);
  }
  EXPECT_FALSE(WriteLlamaPackage(model, o, definitions, &error));
  EXPECT_TRUE(std::filesystem::exists(
      std::filesystem::path(o.output_directory) / "model.json"));
  o.output_directory = (directory / "missing-tokenizer").string();
  o.tokenizer = (directory / "absent.json").string();
  EXPECT_FALSE(WriteLlamaPackage(model, o, definitions, &error));
  EXPECT_FALSE(std::filesystem::exists(o.output_directory));
  loomc_module_release(definitions);
}
} // namespace
} // namespace hrx::loomify_gguf
