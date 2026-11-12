// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Native authored-target seam only; no claim of GPU numerical execution.
#include "lib/loomify/linking.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/product.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "loomc/workspace.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
namespace {
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto text = loomc_status_message(status);
    ADD_FAILURE() << std::string(text.data, text.size);
  }
  loomc_status_free(status);
  return ok;
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  for (size_t i = 0; result && i < loomc_result_diagnostic_count(result); ++i) {
    const auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->code.data, d->code.size)
        .append(": ")
        .append(d->message.data, d->message.size)
        .append("\n");
  }
  return text;
}
std::string Runfile(const std::string &file) {
  return std::string(std::getenv("TEST_SRCDIR")) + "/" +
         std::getenv("TEST_WORKSPACE") + "/lib/loomify/testing/platform/" +
         file;
}
class PlatformProviderTest : public ::testing::TestWithParam<bool> {
protected:
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment{nullptr, loomc_target_environment_release};
  Owner<loomc_context_t, loomc_context_release> context{nullptr,
                                                        loomc_context_release};
  Owner<loomc_workspace_t, loomc_workspace_release> workspace{
      nullptr, loomc_workspace_release};
  void SetUp() override {
    loomc_target_environment_t *e = nullptr;
    ASSERT_TRUE(Ok(
        loomc_target_environment_create_amdgpu(loomc_allocator_system(), &e)));
    environment.reset(e);
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.structure_size = sizeof(target);
    target.target_environment = e;
    loomc_context_options_t options{};
    options.next = &target;
    loomc_context_t *c = nullptr;
    ASSERT_TRUE(
        Ok(loomc_context_create(&options, loomc_allocator_system(), &c)));
    context.reset(c);
    loomc_workspace_t *w = nullptr;
    ASSERT_TRUE(
        Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w)));
    workspace.reset(w);
  }
  bool Link(bool provider, loomc_module_t **out, std::string *error,
            bool missing_file = false) {
    std::string extension = GetParam() ? ".loombc" : ".loom";
    std::string format = GetParam() ? "bc" : "text";
    std::vector<std::string> inputs{Runfile("root_" + format + extension)};
    if (provider)
      inputs.push_back(
          Runfile((missing_file ? "missing_targets_" : "targets_") + format +
                  extension));
    return hrx::loomify_gguf::LinkConfiguredProducts(
        context.get(), workspace.get(), "", inputs, {"platform_test"}, true,
        out, error);
  }
  void Compile(loomc_module_t *module, const char *selector, bool compatible,
               bool require_authored_wave = true) {
    loomc_target_profile_t *p = nullptr;
    ASSERT_TRUE(Ok(loomc_target_profile_select(
        environment.get(), loomc_make_cstring_view(selector),
        loomc_allocator_system(), &p)));
    Owner<loomc_target_profile_t, loomc_target_profile_release> profile(
        p, loomc_target_profile_release);
    auto root = loomc_make_cstring_view("platform_test");
    std::vector<loomc_request_t *> requests;
    loomc_cmd_program_module_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
    options.structure_size = sizeof(options);
    options.root_symbols = &root;
    options.root_symbol_count = 1;
    options.target_profile = p;
    options.request_sink.publish = [](void *user, loomc_request_t *request) {
      static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
      return loomc_ok_status();
    };
    options.request_sink.user_data = &requests;
    loomc_product_t *raw_product = nullptr;
    loomc_result_t *raw_result = nullptr;
    auto status = loomc_cmd_program_product_build_module(
        workspace.get(), module, &options, loomc_allocator_system(),
        &raw_product, &raw_result);
    Owner<loomc_product_t, loomc_product_release> product(
        raw_product, loomc_product_release);
    Owner<loomc_result_t, loomc_result_release> result(raw_result,
                                                       loomc_result_release);
    const bool status_ok = loomc_status_is_ok(status);
    loomc_status_free(status);
    const bool succeeded =
        status_ok && result && loomc_result_succeeded(result.get());
    if (!compatible) {
      EXPECT_FALSE(succeeded);
      const auto diagnostic = Diagnostics(result.get());
      EXPECT_NE(diagnostic.find("TARGET/052"), std::string::npos) << diagnostic;
      EXPECT_NE(diagnostic.find("gfx942"), std::string::npos) << diagnostic;
      EXPECT_NE(diagnostic.find("platform_test_wave"), std::string::npos)
          << diagnostic;
      for (auto *request : requests)
        loomc_request_release(request);
      return;
    }
    ASSERT_TRUE(succeeded) << Diagnostics(result.get());
    ASSERT_EQ(requests.size(), 2u);
    loomc_compiler_t *raw_compiler = nullptr;
    ASSERT_TRUE(Ok(loomc_compiler_create(
        context.get(), nullptr, loomc_allocator_system(), &raw_compiler)));
    Owner<loomc_compiler_t, loomc_compiler_release> compiler(
        raw_compiler, loomc_compiler_release);
    bool saw32 = false, saw64 = false;
    for (auto *raw_request : requests) {
      Owner<loomc_request_t, loomc_request_release> request(
          raw_request, loomc_request_release);
      ASSERT_EQ(loomc_request_root_count(request.get()), 1u);
      loomc_module_t *leaf = nullptr;
      raw_result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context.get(), workspace.get(), loomc_request_source(request.get()),
          nullptr, loomc_allocator_system(), &leaf, &raw_result)));
      Owner<loomc_module_t, loomc_module_release> kernel(leaf,
                                                         loomc_module_release);
      result.reset(raw_result);
      ASSERT_TRUE(result && loomc_result_succeeded(result.get()))
          << Diagnostics(result.get());
      loomc_module_serialize_options_t text_options{};
      text_options.format = LOOMC_SOURCE_FORMAT_TEXT;
      loomc_source_t *raw_text = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          kernel.get(), &text_options, loomc_allocator_system(), &raw_text)));
      Owner<loomc_source_t, loomc_source_release> text_source(
          raw_text, loomc_source_release);
      const auto bytes = loomc_source_contents(raw_text);
      const std::string request_text(reinterpret_cast<const char *>(bytes.data),
                                     bytes.data_length);
      const bool is32 =
          request_text.find("@platform_test_kernel32") != std::string::npos;
      const bool is64 =
          request_text.find("@platform_test_kernel64") != std::string::npos;
      ASSERT_NE(is32, is64) << request_text;
      if (require_authored_wave) {
        EXPECT_NE(request_text.find(is32 ? "subgroup_size = 32"
                                         : "subgroup_size = 64"),
                  std::string::npos)
            << request_text;
      } else {
        EXPECT_NE(request_text.find("amdgpu.target<gfx1151>"),
                  std::string::npos)
            << request_text;
        EXPECT_NE(request_text.find("subgroup_size = 32"), std::string::npos)
            << request_text;
        EXPECT_EQ(request_text.find("subgroup_size = 64"), std::string::npos)
            << request_text;
        std::cout << "DECLARATION-ONLY REQUEST " << selector << " kernel"
                  << (is32 ? 32 : 64) << "\n"
                  << request_text << "\n";
      }
      if (is32) {
        EXPECT_FALSE(saw32);
        saw32 = true;
      } else {
        EXPECT_FALSE(saw64);
        saw64 = true;
      }
      loomc_compile_artifact_options_t compile{};
      compile.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
      compile.structure_size = sizeof(compile);
      compile.target_profile = p;
      raw_result = nullptr;
      ASSERT_TRUE(Ok(loomc_compile_artifact(
          compiler.get(), workspace.get(), nullptr, kernel.get(), &compile,
          loomc_allocator_system(), &raw_result)));
      result.reset(raw_result);
      ASSERT_TRUE(result && loomc_result_succeeded(result.get()))
          << Diagnostics(result.get());
      EXPECT_GT(loomc_result_artifact_count(result.get()), 0u);
    }
    EXPECT_TRUE(saw32);
    EXPECT_TRUE(saw64);
  }
};
TEST_P(PlatformProviderTest,
       SeparateAuthoredDefinitionsSurviveSerializationAndCompile) {
  loomc_module_t *raw = nullptr;
  std::string error;
  ASSERT_TRUE(Link(true, &raw, &error)) << error;
  Owner<loomc_module_t, loomc_module_release> module(raw, loomc_module_release);
  // Both roundtrips compile actual request artifacts, not just successful LINK.
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t serialization{};
    serialization.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
    serialization.structure_size = sizeof(serialization);
    serialization.format = format;
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
        module.get(), &serialization, loomc_allocator_system(), &source)));
    Owner<loomc_source_t, loomc_source_release> serialized(
        source, loomc_source_release);
    if (format == LOOMC_SOURCE_FORMAT_TEXT) {
      const auto bytes = loomc_source_contents(source);
      const std::string text(reinterpret_cast<const char *>(bytes.data),
                             bytes.data_length);
      EXPECT_EQ(text.find("target.decl"), std::string::npos);
      size_t concrete = 0, position = 0;
      while ((position = text.find("amdgpu.target<", position)) !=
             std::string::npos) {
        ++concrete;
        ++position;
      }
      EXPECT_EQ(concrete, 2u) << text;
      EXPECT_NE(text.find("subgroup_size = 32"), std::string::npos) << text;
      EXPECT_NE(text.find("subgroup_size = 64"), std::string::npos) << text;
      EXPECT_NE(
          text.find(
              "kernel.def target(@platform_test_wave32) @platform_test_kernel32"),
          std::string::npos)
          << text;
      EXPECT_NE(
          text.find(
              "kernel.def target(@platform_test_wave64) @platform_test_kernel64"),
          std::string::npos)
          << text;
    }
    loomc_module_t *roundtrip = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context.get(), workspace.get(), serialized.get(), nullptr,
        loomc_allocator_system(), &roundtrip, &result)));
    Owner<loomc_module_t, loomc_module_release> copy(roundtrip,
                                                     loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> parsed(result,
                                                       loomc_result_release);
    ASSERT_TRUE(parsed && loomc_result_succeeded(parsed.get()))
        << Diagnostics(parsed.get());
    Compile(copy.get(), "amdgpu:gfx1100", true);
    Compile(copy.get(), "amdgpu:gfx1151", true);
    Compile(copy.get(), "amdgpu:gfx942", false);
  }
}
// Diagnostic followup to A002: LINK accepts declarations; establish whether a
// supplied native profile can specialize/compile them without authored bounds.
// This is not acceptance of publishing an incompletely configured model.
TEST_P(PlatformProviderTest, DeclarationOnlyTargetsSpecializeToProfileDefault) {
  loomc_module_t *raw = nullptr;
  std::string error;
  ASSERT_TRUE(Link(false, &raw, &error)) << error;
  Owner<loomc_module_t, loomc_module_release> module(raw, loomc_module_release);
  loomc_module_serialize_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  loomc_source_t *raw_source = nullptr;
  ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
      module.get(), &options, loomc_allocator_system(), &raw_source)));
  Owner<loomc_source_t, loomc_source_release> source(raw_source,
                                                     loomc_source_release);
  const auto bytes = loomc_source_contents(source.get());
  const std::string text(reinterpret_cast<const char *>(bytes.data),
                         bytes.data_length);
  EXPECT_NE(text.find("target.decl @platform_test_wave32"), std::string::npos)
      << text;
  EXPECT_NE(text.find("target.decl @platform_test_wave64"), std::string::npos)
      << text;
  EXPECT_EQ(text.find("amdgpu.target<"), std::string::npos) << text;
  std::cout << "DECLARATION-ONLY LINK\n" << text << "\n";
  Compile(module.get(), "amdgpu:gfx1151", true, false);
}
// Loading the selected BUILD-owned provider is the construction boundary.
// Abstract declarations alone are legal native input, as the test above proves.
TEST_P(PlatformProviderTest, MissingSelectedProviderFileIsRejected) {
  loomc_module_t *raw = nullptr;
  std::string error;
  EXPECT_FALSE(Link(true, &raw, &error, true));
  Owner<loomc_module_t, loomc_module_release> module(raw, loomc_module_release);
  EXPECT_EQ(module.get(), nullptr);
  EXPECT_NE(error.find("missing_targets_"), std::string::npos) << error;
}
// A pre-target optional inliner must not turn this launched device write into
// an ordinary effect in the command body. Exercise the production normalizer.
TEST_P(PlatformProviderTest, NormalizationPreservesLaunchedSnapshotEffects) {
  const std::string format = GetParam() ? "bc" : "text";
  const std::string extension = GetParam() ? ".loombc" : ".loom";
  loomc_module_t *raw_module = nullptr;
  std::string error;
  ASSERT_TRUE(hrx::loomify_gguf::LinkConfiguredProducts(
      context.get(), workspace.get(), "",
      {Runfile("count_command_" + format + extension)}, {"count_command_test"},
      true, &raw_module, &error))
      << error;
  Owner<loomc_module_t, loomc_module_release> module(raw_module,
                                                     loomc_module_release);
  loomc_target_profile_t *raw_profile = nullptr;
  ASSERT_TRUE(Ok(loomc_target_profile_select(
      environment.get(), loomc_make_cstring_view("amdgpu:gfx1151"),
      loomc_allocator_system(), &raw_profile)));
  Owner<loomc_target_profile_t, loomc_target_profile_release> profile(
      raw_profile, loomc_target_profile_release);
  struct Requests {
    std::vector<loomc_request_t *> values;
    ~Requests() {
      for (auto *request : values)
        loomc_request_release(request);
    }
  } requests;
  auto root = loomc_make_cstring_view("count_command_test");
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  options.root_symbols = &root;
  options.root_symbol_count = 1;
  options.target_profile = profile.get();
  options.request_sink.publish = [](void *user, loomc_request_t *request) {
    static_cast<Requests *>(user)->values.push_back(request);
    return loomc_ok_status();
  };
  options.request_sink.user_data = &requests;
  loomc_product_t *raw_product = nullptr;
  loomc_result_t *raw_result = nullptr;
  auto status = loomc_cmd_program_product_build_module(
      workspace.get(), module.get(), &options, loomc_allocator_system(),
      &raw_product, &raw_result);
  Owner<loomc_product_t, loomc_product_release> product(raw_product,
                                                        loomc_product_release);
  Owner<loomc_result_t, loomc_result_release> result(raw_result,
                                                     loomc_result_release);
  ASSERT_TRUE(Ok(status)) << Diagnostics(result.get());
  ASSERT_TRUE(result && loomc_result_succeeded(result.get()))
      << Diagnostics(result.get());
  EXPECT_EQ(requests.values.size(), 1u);
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecodeProducts, PlatformProviderTest,
                         ::testing::Bool());
} // namespace
