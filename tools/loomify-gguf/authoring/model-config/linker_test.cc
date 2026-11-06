// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "tools/loomify-gguf/authoring/model-config/linker.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>

namespace hrx::loomify_gguf {
namespace {
using Module = std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)>;
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto *d = loomc_result_diagnostic_at(result, i);
      text.append(d->message.data, d->message.size);
      text += '\n';
    }
  return text;
}
::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return ::testing::AssertionSuccess();
  auto text = loomc_status_message(status);
  std::string message(text.data, text.size);
  loomc_status_free(status);
  return ::testing::AssertionFailure() << message;
}
class ConfiguredLinkTest : public ::testing::Test {
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
    scratch = std::filesystem::path(std::getenv("TEST_TMPDIR")) /
              ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::create_directories(scratch);
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
  }
  Module Parse(const char *text) {
    loomc_source_options_t options{};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier = loomc_make_cstring_view("generated.loom");
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.contents = loomc_make_byte_span(text, std::strlen(text));
    loomc_source_t *source = nullptr;
    EXPECT_TRUE(
        Ok(loomc_source_create(&options, loomc_allocator_system(), &source)));
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(), &module,
        &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    loomc_source_release(source);
    return Module(module, loomc_module_release);
  }
  Module Config() {
    return Parse("config.def @llama.model.hidden_size = 768 : index\n"
                 "config.def @llama.model.rms_epsilon = 0.00002 : f32\n");
  }
  ConfiguredLinkRequest Request(bool bytecode = false) {
    ConfiguredLinkRequest request;
    request.source_root = root;
    if (bytecode) {
      request.providers = {{"tools/loomify-gguf/authoring/model-config/"
                            "test_norm_provider.loombc",
                            LOOMC_LINK_PROVIDER_ROLE_LIBRARY}};
    } else {
      request.providers = {
          {"lib/kernels/llama/norm/norm.loom", LOOMC_LINK_PROVIDER_ROLE_INPUT},
          {"lib/kernels/llama/active_rows.loom",
           LOOMC_LINK_PROVIDER_ROLE_LIBRARY},
          {"lib/kernels/normalization/rmsnorm_row.loom",
           LOOMC_LINK_PROVIDER_ROLE_LIBRARY}};
    }
    auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    auto buffer = loom_type_buffer();
    request.roots = {{"llama32_rmsnorm_f32",
                      ConfiguredCallableKind::Kernel,
                      {index},
                      {index, buffer, buffer, buffer}}};
    request.required_configuration = {"llama.model.hidden_size",
                                      "llama.model.rms_epsilon"};
    request.required_definitions = {"llm_rmsnorm_row"};
    return request;
  }
  std::string Text(const loomc_module_t *module) {
    loomc_module_serialize_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
    options.structure_size = sizeof(options);
    options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
    loomc_source_t *source = nullptr;
    EXPECT_TRUE(Ok(loomc_module_serialize_text_to_source(
        module, &options, loomc_allocator_system(), &source)));
    auto bytes = loomc_source_contents(source);
    std::string text(reinterpret_cast<const char *>(bytes.data),
                     bytes.data_length);
    loomc_source_release(source);
    return text;
  }
  Module Link(const loomc_module_t *config,
              const ConfiguredLinkRequest &request) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(LinkConfiguredModules(context, workspace, config, request,
                                         &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    return Module(module, loomc_module_release);
  }
  void Rejected(const loomc_module_t *config,
                const ConfiguredLinkRequest &request,
                const std::string &message) {
    loomc_module_t *module = reinterpret_cast<loomc_module_t *>(1);
    loomc_result_t *result = nullptr;
    auto status = LinkConfiguredModules(context, workspace, config, request,
                                        &module, &result);
    std::string text = Diagnostics(result);
    if (!loomc_status_is_ok(status)) {
      auto view = loomc_status_message(status);
      text.append(view.data, view.size);
    }
    EXPECT_TRUE(!loomc_status_is_ok(status) ||
                (result && !loomc_result_succeeded(result)))
        << text;
    EXPECT_EQ(module, nullptr);
    EXPECT_NE(text.find(message), std::string::npos) << text;
    loomc_status_free(status);
    if (result)
      loomc_result_release(result);
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
  std::filesystem::path scratch;
};
TEST_F(ConfiguredLinkTest,
       ActualSourceAndRelocatableProvidersPreserveConfigAndLifetime) {
  for (bool bytecode : {false, true}) {
    auto config = Config();
    auto before = Text(config.get());
    auto module = Link(config.get(), Request(bytecode));
    ASSERT_NE(module, nullptr);
    EXPECT_EQ(before, Text(config.get()));
    config.reset();
    auto text = Text(module.get());
    EXPECT_NE(text.find("768"), std::string::npos);
    EXPECT_NE(text.find("llama32_rmsnorm_f32"), std::string::npos);
    EXPECT_NE(text.find("llm_rmsnorm_row"), std::string::npos);
    EXPECT_EQ(text.find("llama32_rmsnorm_active_f32"), std::string::npos);
    // Both round trips must be independent of the released input config.
    auto parsed = Parse(text.c_str());
    ASSERT_NE(parsed, nullptr);
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_bytecode_to_source(
        module.get(), nullptr, loomc_allocator_system(), &source)));
    loomc_module_t *reloaded = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(),
        &reloaded, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    loomc_source_release(source);
    Module reload(reloaded, loomc_module_release);
    ASSERT_NE(reload, nullptr);
    EXPECT_NE(Text(reload.get()).find("768"), std::string::npos);
  }
}
TEST_F(ConfiguredLinkTest, DeduplicatesInputsAndRejectsContradictoryRoles) {
  auto config = Config();
  auto request = Request();
  request.providers.push_back(request.providers.front());
  ASSERT_NE(Link(config.get(), request), nullptr);
  request.providers.back().role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
  Rejected(config.get(), request, "contradictory provider roles");
}
TEST_F(ConfiguredLinkTest,
       RejectsMissingInputsContractsAndWrongRootInterfaces) {
  auto config = Config();
  auto request = Request();
  request.providers[0].path = "missing.loom";
  Rejected(config.get(), request, "missing.loom");
  request = Request();
  request.required_configuration.push_back("missing.config");
  Rejected(config.get(), request, "missing.config");
  request = Request();
  request.required_definitions.push_back("missing.helper");
  Rejected(config.get(), request, "missing.helper");
  request = Request();
  request.roots[0].symbol = "missing.root";
  Rejected(config.get(), request, "missing.root");
  request = Request();
  request.roots[0].kind = ConfiguredCallableKind::CommandProgram;
  Rejected(config.get(), request, "callable kind");
  request = Request();
  request.roots[0].arguments.pop_back();
  Rejected(config.get(), request, "operand group count");
  request = Request();
  request.roots[0].arguments[0] = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  Rejected(config.get(), request, "operand type");
  request = Request();
  request.providers[0].path = "../outside.loom";
  Rejected(config.get(), request, "escapes source root");
}
TEST_F(ConfiguredLinkTest, RejectsStaticDefaultsEvenForEqualValues) {
  auto config = Parse("config.def @llama.model.hidden_size = 3072 : index\n"
                      "config.def @llama.model.rms_epsilon = 0.00001 : f32\n");
  auto request = Request();
  request.providers.push_back(
      {"lib/kernels/llama/config.loom", LOOMC_LINK_PROVIDER_ROLE_LIBRARY});
  Rejected(config.get(), request, "competes with generated configuration");

  // Inspect embedded definitions too, independently of source filenames.
  loomc_source_t *source = nullptr;
  ASSERT_TRUE(Ok(loomc_module_serialize_bytecode_to_source(
      config.get(), nullptr, loomc_allocator_system(), &source)));
  auto bytes = loomc_source_contents(source);
  auto relative = std::filesystem::path("embedded-defaults.loombc");
  std::ofstream file(scratch / relative, std::ios::binary);
  file.write(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
  file.close();
  loomc_source_release(source);
  request = Request();
  for (const auto &provider : request.providers) {
    auto destination = scratch / provider.path;
    std::filesystem::create_directories(destination.parent_path());
    std::filesystem::copy_file(
        std::filesystem::path(root) / provider.path, destination,
        std::filesystem::copy_options::overwrite_existing);
  }
  request.source_root = scratch.string();
  request.providers.push_back(
      {relative.generic_string(), LOOMC_LINK_PROVIDER_ROLE_LIBRARY});
  Rejected(config.get(), request, "embedded-defaults.loombc");
}
TEST_F(ConfiguredLinkTest, UnresolvedNativeDependencyFailsWithNullOutput) {
  auto config = Config();
  auto request = Request();
  // Family selection can remain open until specialization, so remove an
  // ordinary function required by an exported active-row norm instead.
  request.providers.erase(request.providers.begin() + 1);
  request.roots[0].symbol = "llama32_rmsnorm_active_f32";
  auto index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  auto offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  auto buffer = loom_type_buffer();
  request.roots[0].arguments = {index,  index,  offset, buffer,
                                buffer, buffer, buffer};
  Rejected(config.get(), request, "llama_active_row");
}
TEST_F(ConfiguredLinkTest, DeclaredRunfilesBundleWorksOutsideCheckout) {
  auto request = Request(true);
  auto target = scratch / "provider.loombc";
  std::filesystem::copy_file(
      std::filesystem::path(root) / request.providers[0].path, target,
      std::filesystem::copy_options::overwrite_existing);
  request.source_root = scratch.string();
  request.providers[0].path = "provider.loombc";
  auto config = Config();
  auto previous = std::filesystem::current_path();
  std::filesystem::current_path(scratch);
  auto module = Link(config.get(), request);
  std::filesystem::current_path(previous);
  ASSERT_NE(module, nullptr);
}
} // namespace
} // namespace hrx::loomify_gguf
