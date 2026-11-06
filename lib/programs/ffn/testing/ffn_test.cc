// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model-config/configuration.h"
#include "tools/loomify-gguf/authoring/model-config/providers.h"
#include "lib/llama/testing/model_fixture.h"
#include "lib/qwen/testing/model_fixture.h"
#include "loomc/compile.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <string>
#include <vector>
namespace hrx::loomify_gguf {
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


// Synthetic descriptor metadata exercises both native provider closures. Actual
// checkpoint bytes and active-row numerics are a separate bounded manual gate.
model::ModelDescriptor Facts(bool llama) {
  if (!llama) return qwen::testing::ModelFacts(3);
  auto facts = llama::testing::ModelFacts(2);
  for (size_t i = 0; i < facts.layers.size(); ++i) {
    auto &layer = facts.layers[i];
    const auto old = "blk." + std::to_string(layer.index) + ".";
    const auto name = "blk." + std::to_string(i) + ".";
    layer.index = i;
    for (auto &binding : layer.tensors) {
      binding.tensor.name.replace(0, old.size(), name);
      // Mixed encodings keep the lookup result nonuniform across layers.
      // Uniform metadata alone can hide premature family selection.
      if (i == 1 && binding.role == "ffn_down") {
        binding.tensor.storage_type = 14;
        binding.tensor.byte_size = 8192 / 256 * 3072 * 210;
      }
    }
  }
  return facts;
}
class ConfiguredFfnTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" + std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
  }
  void Link(bool relocatable, const std::string &suffix,
            loomc_module_t **module) {
    const bool llama = GetParam();
    const std::string family = llama ? "llama" : "qwen";
    auto facts = Facts(llama);
    LinkedConfigurationSettings settings;
    settings.profile = llama ? LinkedProfile::Llama3BModel : LinkedProfile::Qwen08BFfn;
    settings.token_capacity = 4;
    LinkedConfiguration config;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(CreateLinkedConfiguration(context, workspace, facts,
        "synthetic-descriptor", settings, &config, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    auto request = llama ? MakeLlamaLinkedProviders(root, relocatable)
                         : MakeQwenFfnLinkedProviders(root, relocatable);
    request.providers.push_back({"lib/programs/ffn/ffn.loom"});
    request.providers.push_back({"lib/" + family + "/programs/ffn_adapter.loom"});
    request.providers.push_back({"lib/programs/ffn/testing/" + family + "_roots.loom"});
    request.roots.push_back({"test." + family + ".ffn." + suffix,
        ConfiguredCallableKind::CommandProgram, {},
        {loom_type_buffer(), loom_type_buffer(), loom_type_buffer()}});
    request.required_configuration = {"loomify.ffn.profile"};
    request.required_definitions = {"loomify_ffn"};
    result = nullptr;
    ASSERT_TRUE(Ok(LinkConfiguredModules(context, workspace, config.module.get(),
        request, module, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    ASSERT_NE(*module, nullptr);
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_P(ConfiguredFfnTest, SourceAndRelocatableProfilesMaterializeAndCompile) {
  const std::string family = GetParam() ? "llama" : "qwen";
  for (bool relocatable : {false, true}) {
    for (const std::string suffix : {"decode", "prefill"}) {
      SCOPED_TRACE(family + (relocatable ? " bytecode " : " source ") + suffix);
      loomc_module_t *module = nullptr;
      Link(relocatable, suffix, &module);
      ASSERT_NE(module, nullptr);
      const auto symbol = "test." + family + ".ffn." + suffix;
      Compile(context, workspace, environment, module, symbol.c_str());
      loomc_module_release(module);
    }
  }
}
TEST_P(ConfiguredFfnTest, DecodeRejectsMoreThanOneRow) {
  loomc_module_t *module = nullptr;
  Link(false, "bad_decode_rows", &module);
  ASSERT_NE(module, nullptr);
  const std::string symbol = std::string("test.") + (GetParam() ? "llama" : "qwen") + ".ffn.bad_decode_rows";
  auto root_symbol = loomc_make_cstring_view(symbol.c_str());
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  loomc_target_profile_t *profile = nullptr;
  ASSERT_TRUE(Ok(loomc_target_profile_select(environment,
      loomc_make_cstring_view("amdgpu:gfx1100"), loomc_allocator_system(), &profile)));
  options.kernel_target_profile = profile;
  options.root_symbols = &root_symbol;
  options.root_symbol_count = 1;
  loomc_product_t *product = nullptr;
  loomc_result_t *result = nullptr;
  const auto status = loomc_cmd_program_product_build_module(workspace, module,
      &options, loomc_allocator_system(), &product, &result);
  const bool failed_status = !loomc_status_is_ok(status);
  loomc_status_free(status);
  EXPECT_TRUE(failed_status || (result && !loomc_result_succeeded(result)));
  EXPECT_EQ(product, nullptr);
  EXPECT_NE(Diagnostics(result).find("loomify.ffn.decode_rows"), std::string::npos);
  loomc_target_profile_release(profile);
  if (result) loomc_result_release(result);
  if (product) loomc_product_release(product);
  loomc_module_release(module);
}
INSTANTIATE_TEST_SUITE_P(Profiles, ConfiguredFfnTest, ::testing::Bool());
} // namespace
} // namespace hrx::loomify_gguf
