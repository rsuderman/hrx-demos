// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/target/amdgpu.h"
#include "tools/loomify-gguf/serving/qwen/package.h"
#include "tools/loomify-gguf/serving/qwen/testing/fixture.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
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
TEST(QwenPackage, RendererPreservesNativeFootprintsAndPublicClosure) {
  ModelPackageOptions options;
  options.source_root = ".";
  options.prefill_capacity = 128;
  options.context_capacity = 512;
  detail::CompanionSources sources;
  std::string error;
  ASSERT_TRUE(
      RenderQwenCompanions(options, testing::Geometry(), &sources, &error))
      << error;
  EXPECT_EQ(sources.prepare.find("{{"), std::string::npos);
  EXPECT_NE(
      sources.prepare.find("%residual_bytes_index = index.constant 2097152"),
      std::string::npos);
  EXPECT_NE(sources.prepare.find("%gdn_bytes_index = index.constant 20201472"),
            std::string::npos);
  EXPECT_NE(sources.prepare.find("%row_stride = scalar.constant 20204032"),
            std::string::npos);
  EXPECT_NE(sources.control.find(
                "%token_bytes = buffer.length %tokens\n  buffer.fill %zero32, "
                "%tokens[%origin], %token_bytes : i32"),
            std::string::npos);
  loomc_target_environment_t *environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &environment)));
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t opts{};
  opts.next = &target;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  ASSERT_TRUE(
      Ok(loomc_context_create(&opts, loomc_allocator_system(), &context)));
  ASSERT_TRUE(Ok(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  const auto directory = std::filesystem::path(std::getenv("TEST_TMPDIR"));
  for (const auto &entry : {std::make_pair("prepare.loom", sources.prepare),
                            std::make_pair("control.loom", sources.control),
                            std::make_pair("weights.loom", sources.weights)}) {
    const auto path = directory / entry.first;
    std::ofstream(path) << entry.second;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
        loomc_allocator_system(), &module, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << entry.first << "\n"
        << Diagnostics(result);
    loomc_result_release(result);
    loomc_module_release(module);
  }
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
}
TEST(QwenPackage, GeometryMismatchDoesNotReplaceCompanions) {
  ModelPackageOptions options;
  options.source_root = ".";
  options.prefill_capacity = 128;
  options.context_capacity = 512;
  detail::CompanionSources sources{"old prepare", "old control", "old weights"};
  std::string error;
  EXPECT_FALSE(
      RenderQwenCompanions(options, testing::Geometry(1024), &sources, &error));
  EXPECT_EQ(sources.prepare, "old prepare");
  EXPECT_EQ(sources.control, "old control");
  options.prefill_capacity = 513;
  EXPECT_FALSE(
      RenderQwenCompanions(options, testing::Geometry(), &sources, &error));
  EXPECT_EQ(sources.weights, "old weights");
}
} // namespace
} // namespace hrx::loomify_gguf
