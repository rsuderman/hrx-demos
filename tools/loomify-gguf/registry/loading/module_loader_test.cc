// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/registry/loading/module_loader.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "loomc/source.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf {
namespace {

template <typename T, void (*Release)(T *)> struct Releaser {
  void operator()(T *value) const { Release(value); }
};
using Context =
    std::unique_ptr<loomc_context_t,
                    Releaser<loomc_context_t, loomc_context_release>>;
using Workspace =
    std::unique_ptr<loomc_workspace_t,
                    Releaser<loomc_workspace_t, loomc_workspace_release>>;
using Module = std::unique_ptr<loomc_module_t,
                               Releaser<loomc_module_t, loomc_module_release>>;
using Result = std::unique_ptr<loomc_result_t,
                               Releaser<loomc_result_t, loomc_result_release>>;
using Source = std::unique_ptr<loomc_source_t,
                               Releaser<loomc_source_t, loomc_source_release>>;

::testing::AssertionResult IsOk(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return ::testing::AssertionSuccess();
  const auto message = loomc_status_message(status);
  const std::string copied =
      message.size ? std::string(message.data, message.size)
                   : loomc_status_code_string(loomc_status_code(status));
  loomc_status_free(status);
  return ::testing::AssertionFailure() << copied;
}

std::string Diagnostics(const loomc_result_t *result) {
  std::string text;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const auto *diagnostic = loomc_result_diagnostic_at(result, i);
    text.append(diagnostic->message.data, diagnostic->message.size);
    text.push_back('\n');
  }
  return text;
}

class ModuleLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    loomc_context_t *context = nullptr;
    ASSERT_TRUE(IsOk(
        loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    context_.reset(context);
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(IsOk(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    workspace_.reset(workspace);
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
    ASSERT_NE(std::getenv("TEST_TMPDIR"), nullptr);
  }

  std::string Fixture(const char *name) const {
    return std::string(std::getenv("TEST_SRCDIR")) + "/" +
           std::getenv("TEST_WORKSPACE") +
           "/tools/loomify-gguf/registry/loading/testdata/" + name + ".loombc";
  }

  Module Load(const std::string &path) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    const auto status =
        LoadBytecode(context_.get(), workspace_.get(),
                     loomc_make_cstring_view(path.c_str()), &module, &result);
    Result result_owner(result);
    if (!IsOk(status)) {
      ADD_FAILURE() << "Bytecode API load failed: " << path;
      return Module(module);
    }
    EXPECT_NE(result, nullptr);
    if (result)
      EXPECT_TRUE(loomc_result_succeeded(result)) << Diagnostics(result);
    return Module(module);
  }

  std::string Text(const loomc_module_t *module) {
    std::string text;
    EXPECT_TRUE(IsOk(DumpText(module, &text)));
    return text;
  }

  void ExpectRejected(const std::string &path) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status =
        LoadBytecode(context_.get(), workspace_.get(),
                     loomc_make_cstring_view(path.c_str()), &module, &result);
    Module module_owner(module);
    Result result_owner(result);
    EXPECT_EQ(module, nullptr);
    if (loomc_status_is_ok(status)) {
      ASSERT_NE(result, nullptr);
      EXPECT_FALSE(loomc_result_succeeded(result));
      // The loader has already released the input source; diagnostics must
      // still be readable through the owned result.
      EXPECT_FALSE(Diagnostics(result).empty());
    } else {
      EXPECT_EQ(result, nullptr);
      EXPECT_NE(loomc_status_code(status), LOOMC_STATUS_OK);
      loomc_status_free(status);
    }
  }

  std::string Write(const char *name, const std::string &contents) {
    const auto path = std::string(std::getenv("TEST_TMPDIR")) + "/" + name;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(contents.data(), contents.size());
    file.close();
    EXPECT_TRUE(file.good());
    return path;
  }

  Context context_;
  Workspace workspace_;
};

TEST_F(ModuleLoaderTest, LoadsDistinctRealKernelAndMotifFiles) {
  auto kernel = Load(Fixture("decode_control_provider"));
  auto motif = Load(Fixture("qk_common_provider"));
  ASSERT_NE(kernel, nullptr);
  ASSERT_NE(motif, nullptr);
  const auto kernel_text = Text(kernel.get());
  const auto motif_text = Text(motif.get());
  EXPECT_NE(kernel_text.find("@qwen38_decode_commit_token"), std::string::npos);
  EXPECT_EQ(kernel_text.find("@ggml_qk_scale_min"), std::string::npos);
  EXPECT_NE(motif_text.find("@ggml.qk.scale_min"), std::string::npos);
  EXPECT_NE(motif_text.find("@ggml_qk_scale_min"), std::string::npos);
  EXPECT_NE(motif_text.find("template.def"), std::string::npos);
  EXPECT_EQ(motif_text.find("@qwen38_decode_commit_token"), std::string::npos);
  loomc_module_function_t function = {};
  ASSERT_TRUE(loomc_module_try_lookup_function(
      kernel.get(), loomc_make_cstring_view("qwen38_decode_commit_token"),
      &function));
  EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
}

TEST_F(ModuleLoaderTest, RealProvidersRoundtripInFreshContexts) {
  for (const auto *name : {"decode_control_provider", "qk_common_provider"}) {
    auto module = Load(Fixture(name));
    ASSERT_NE(module, nullptr);
    const auto original_text = Text(module.get());
    for (const auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options = {};
      options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
      options.structure_size = sizeof(options);
      options.format = format;
      options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(IsOk(loomc_module_serialize_to_source(
          module.get(), &options, loomc_allocator_system(), &source)));
      Source source_owner(source);
      loomc_context_t *context = nullptr;
      ASSERT_TRUE(IsOk(
          loomc_context_create(nullptr, loomc_allocator_system(), &context)));
      Context fresh_context(context);
      loomc_workspace_t *workspace = nullptr;
      ASSERT_TRUE(IsOk(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                              &workspace)));
      Workspace fresh_workspace(workspace);
      loomc_module_t *roundtrip = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(IsOk(loomc_module_deserialize_from_source(
          context, workspace, source, nullptr, loomc_allocator_system(),
          &roundtrip, &result)));
      Module roundtrip_owner(roundtrip);
      Result result_owner(result);
      ASSERT_NE(result, nullptr);
      ASSERT_TRUE(loomc_result_succeeded(result)) << Diagnostics(result);
      source_owner.reset();
      EXPECT_EQ(Text(roundtrip), original_text);
    }
  }
}

TEST_F(ModuleLoaderTest, DumpOutlivesInputSourceWorkspaceAndModule) {
  auto module = Load(Fixture("qk_common_provider"));
  ASSERT_NE(module, nullptr);
  workspace_.reset();
  context_.reset();
  const auto text = Text(module.get());
  module.reset();
  EXPECT_NE(text.find("@ggml_qk_scale_min"), std::string::npos);
}

TEST_F(ModuleLoaderTest, RejectsMalformedFilesAndRecovers) {
  std::ifstream file(Fixture("decode_control_provider"), std::ios::binary);
  ASSERT_TRUE(file.good());
  const std::string bytes((std::istreambuf_iterator<char>(file)), {});
  ASSERT_GT(bytes.size(), 16u);
  std::string corrupt = bytes;
  corrupt[0] = '!';
  const std::vector<std::string> paths = {
      std::string(std::getenv("TEST_TMPDIR")) + "/missing.loombc",
      Write("empty.loombc", ""),
      Write("truncated.loombc", bytes.substr(0, bytes.size() / 2)),
      Write("corrupt.loombc", corrupt),
      Write("renamed.loombc", "func.def public @text() { func.return }\n"),
  };
  for (const auto &path : paths) {
    SCOPED_TRACE(path);
    ExpectRejected(path);
    auto valid = Load(Fixture("decode_control_provider"));
    ASSERT_NE(valid, nullptr);
    EXPECT_NE(Text(valid.get()).find("@qwen38_decode_commit_token"),
              std::string::npos);
  }
}

TEST_F(ModuleLoaderTest, InvalidArgumentsClearOutputs) {
  loomc_module_t *module = reinterpret_cast<loomc_module_t *>(1);
  loomc_result_t *result = reinterpret_cast<loomc_result_t *>(1);
  EXPECT_EQ(loomc_status_consume_code(LoadBytecode(
                nullptr, workspace_.get(), loomc_make_cstring_view("unused"),
                &module, &result)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(loomc_status_consume_code(LoadBytecode(
                context_.get(), workspace_.get(),
                loomc_make_cstring_view("unused"), nullptr, &result)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  std::string text = "previous output";
  EXPECT_EQ(loomc_status_consume_code(DumpText(nullptr, &text)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_TRUE(text.empty());
  EXPECT_EQ(loomc_status_consume_code(DumpText(nullptr, nullptr)),
            LOOMC_STATUS_INVALID_ARGUMENT);
}

} // namespace
} // namespace hrx::loomify_gguf
