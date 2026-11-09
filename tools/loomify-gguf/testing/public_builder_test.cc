// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Ordinary client: only the supported builder header exposes product APIs.
#include "tools/loomify-gguf/gguf_model_builder.h"
#include "gtest/gtest.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <csignal>
#include <sys/resource.h>
namespace hrx::loomify_gguf {
namespace {
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Context(loomc_context_t **context) {
  loomc_target_environment_t *environment = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(), &environment))) return false;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t options{};
  options.next = &target;
  auto status = loomc_context_create(&options, loomc_allocator_system(), context);
  loomc_target_environment_release(environment);
  return Ok(status);
}
TEST(PublicGgufBuilder, InvalidOpenPreservesExistingOutput) {
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  EXPECT_FALSE(GgufModelBuilder::Open("", &builder, &error));
  EXPECT_FALSE(builder);
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(GgufModelBuilder::Open("/nonexistent-loomify-test.gguf", &builder, nullptr));
}
TEST(PublicGgufBuilder, RealModulesNeedNoServeCompanionsAndOwnTheirLifetime) {
  const auto *llama = std::getenv("LOOMIFY_TEST_GGUF");
  const auto *qwen = std::getenv("LOOMIFY_TEST_QWEN_GGUF");
  if (!llama || !qwen) GTEST_SKIP() << "explicit immutable real-model gate";
  const auto *runfiles = std::getenv("TEST_SRCDIR");
  const auto *workspace_name = std::getenv("TEST_WORKSPACE");
  const auto *temporary = std::getenv("TEST_TMPDIR");
  ASSERT_TRUE(runfiles && workspace_name && temporary);
  const auto root = std::filesystem::path(runfiles) / workspace_name;
  for (const char *asset : {llama, qwen}) {
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error = "stale";
    ASSERT_TRUE(GgufModelBuilder::Open(asset, &builder, &error)) << error;
    EXPECT_TRUE(error.empty());
    auto *previous = builder.get();
    EXPECT_FALSE(GgufModelBuilder::Open("/nonexistent-loomify-test.gguf", &builder, &error));
    EXPECT_EQ(builder.get(), previous);
    const std::string architecture = builder->descriptor().family == "llama" ? "llama" : "qwen35";
    auto directory = std::filesystem::path(temporary) / architecture;
    std::filesystem::create_directories(directory / "lib/architectures" / architecture);
    // Deliberately copy only the actual numerical product, with all Serve-only
    // preparation/control/weight-policy companions absent.
    std::filesystem::copy_file(root / "lib/architectures" / architecture / "implementation.loombc",
        directory / "lib/architectures" / architecture / "implementation.loombc");
    loomc_context_t *context = nullptr;
    ASSERT_TRUE(Context(&context));
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    GgufBuildOptions options;
    options.library_root = directory.string();
    options.context_capacity = 513;
    std::unique_ptr<BuiltGgufModule> first, second;
    ASSERT_TRUE(builder->BuildModule(context, workspace, options, &first, &error)) << error;
    EXPECT_TRUE(error.empty());
    const auto configuration = first->configuration_source();
    EXPECT_FALSE(configuration.empty());
    options.prefill_capacity = 1;
    ASSERT_TRUE(builder->BuildModule(context, workspace, options, &second, &error)) << error;
    EXPECT_NE(second->configuration_source(), configuration);
    EXPECT_EQ(first->configuration_source(), configuration);
    auto *saved = first.get();
    options.prefill_capacity = 0;
    EXPECT_FALSE(builder->BuildModule(context, workspace, options, &first, &error));
    EXPECT_EQ(first.get(), saved);
    options.prefill_capacity = 128;
    options.library_root = (directory / "missing-library").string();
    EXPECT_FALSE(builder->BuildModule(context, workspace, options, &first, &error));
    EXPECT_EQ(first.get(), saved);
    builder.reset();
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    ASSERT_TRUE(first->module());
    const auto text_path = directory / "model.loom";
    const auto bytecode_path = directory / "model.loombc";
    ASSERT_TRUE(first->WriteModule(text_path.string(), LOOMC_SOURCE_FORMAT_TEXT, &error)) << error;
    ASSERT_TRUE(first->WriteModule(bytecode_path.string(), LOOMC_SOURCE_FORMAT_BYTECODE, &error)) << error;
    const auto before = std::filesystem::file_size(bytecode_path);
    EXPECT_FALSE(second->WriteModule(bytecode_path.string(), LOOMC_SOURCE_FORMAT_BYTECODE, &error));
    EXPECT_EQ(std::filesystem::file_size(bytecode_path), before);
    EXPECT_FALSE(first->WriteModule(asset, LOOMC_SOURCE_FORMAT_TEXT, &error));
    auto alias = directory / "weights-alias";
    std::filesystem::create_symlink(asset, alias);
    EXPECT_FALSE(first->WriteModule(alias.string(), LOOMC_SOURCE_FORMAT_BYTECODE, &error));
    EXPECT_TRUE(std::filesystem::is_symlink(alias));
    auto invalid = directory / "invalid";
    EXPECT_FALSE(first->WriteModule(invalid.string(), LOOMC_SOURCE_FORMAT_UNKNOWN, &error));
    EXPECT_FALSE(std::filesystem::exists(invalid));
    // Force an actual write/close failure without touching shared disk state.
    // Restore the process limit and handler before retrying the same new path.
    struct rlimit original{};
    ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &original), 0);
    auto limited = original;
    limited.rlim_cur = 0;
    auto handler = std::signal(SIGXFSZ, SIG_IGN);
    ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limited), 0);
    const auto failed = directory / "retry.loombc";
    bool wrote = first->WriteModule(failed.string(), LOOMC_SOURCE_FORMAT_BYTECODE, &error);
    int restored = setrlimit(RLIMIT_FSIZE, &original);
    std::signal(SIGXFSZ, handler);
    ASSERT_EQ(restored, 0);
    EXPECT_FALSE(wrote);
    EXPECT_FALSE(std::filesystem::exists(failed));
    ASSERT_TRUE(first->WriteModule(failed.string(), LOOMC_SOURCE_FORMAT_BYTECODE, &error)) << error;
    // Reload both formats in an independent fresh context/workspace after the
    // producing builder and all caller-owned native references were released.
    ASSERT_TRUE(Context(&context));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    for (const auto &path : {text_path, bytecode_path}) {
      loomc_module_t *module = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(context, workspace,
          loomc_make_cstring_view(path.c_str()), nullptr, loomc_allocator_system(), &module, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result));
      loomc_result_release(result);
      loomc_module_release(module);
    }
    loomc_workspace_release(workspace);
    loomc_context_release(context);
  }
}
} // namespace
} // namespace hrx::loomify_gguf
