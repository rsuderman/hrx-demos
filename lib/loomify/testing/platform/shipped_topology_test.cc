// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Compile actual shipped exports and invoke their emitted launch contract;
// device execution and numerical expectations are qualified separately.
#include "lib/loomify/linking.h"
#include "loomc/loomc.h"
#include "loomc/target/configured.h"
#include "loomc/target/amdgpu.h"
#include "loomc/launch_config.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
namespace {
template <class T, void (*Release)(T*)>
using Owner = std::unique_ptr<T, decltype(Release)>;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
std::string Diagnostics(loomc_result_t* result) {
  std::string text;
  for (size_t i = 0; result && i < loomc_result_diagnostic_count(result); ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(result, i);
    const auto message = diagnostic->formatted_text.size ? diagnostic->formatted_text : diagnostic->message;
    text.append(message.data, message.size).append("\n");
  }
  return text;
}
class ShippedTopology : public ::testing::TestWithParam<bool> {};
TEST_P(ShippedTopology, VocabularyProjectionPreservesBoundedLaunchContract) {
  const auto allocator = loomc_allocator_system();
  loomc_target_environment_t* raw_environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_configured(allocator, &raw_environment)));
  Owner<loomc_target_environment_t, loomc_target_environment_release> environment(
      raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.structure_size = sizeof(target); target.target_environment = environment.get();
  loomc_context_options_t context_options{};
  context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
  context_options.structure_size = sizeof(context_options); context_options.next = &target;
  loomc_context_t* raw_context = nullptr;
  ASSERT_TRUE(Ok(loomc_context_create(&context_options, allocator, &raw_context)));
  Owner<loomc_context_t, loomc_context_release> context(raw_context, loomc_context_release);
  loomc_workspace_t* raw_workspace = nullptr;
  ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, allocator, &raw_workspace)));
  Owner<loomc_workspace_t, loomc_workspace_release> workspace(raw_workspace, loomc_workspace_release);
  ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
  ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
  const std::string input = std::string(std::getenv("TEST_SRCDIR")) + "/" +
      std::getenv("TEST_WORKSPACE") + "/lib/loomify/testing/platform/qwen_projection_" +
      (GetParam() ? "bc.loombc" : "text.loom");
  const std::vector<const char*> names = {
      "qwen38_output_projection_q6k_m8", "qwen38_output_projection_q6k_m32"};
  loomc_module_t* raw_module = nullptr;
  std::string error;
  ASSERT_TRUE(hrx::loomify_gguf::LinkConfiguredProducts(context.get(), workspace.get(),
      "", {input}, names, false, &raw_module, &error)) << error;
  Owner<loomc_module_t, loomc_module_release> module(raw_module, loomc_module_release);
  loomc_target_profile_t* raw_profile = nullptr;
  ASSERT_TRUE(Ok(loomc_target_profile_select(environment.get(),
      loomc_make_cstring_view("amdgpu:gfx1151"), allocator, &raw_profile)));
  Owner<loomc_target_profile_t, loomc_target_profile_release> profile(raw_profile, loomc_target_profile_release);
  loomc_compiler_t* raw_compiler = nullptr;
  ASSERT_TRUE(Ok(loomc_compiler_create(context.get(), nullptr, allocator, &raw_compiler)));
  Owner<loomc_compiler_t, loomc_compiler_release> compiler(raw_compiler, loomc_compiler_release);
  std::vector<loomc_string_view_t> roots;
  for (const char* name : names) roots.push_back(loomc_make_cstring_view(name));
  loomc_emit_options_t emit{};
  emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS; emit.structure_size = sizeof(emit);
  emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  loomc_compile_artifact_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
  options.structure_size = sizeof(options); options.roots = roots.data(); options.root_count = roots.size();
  options.target_profile = profile.get(); options.emit_options = &emit;
  options.artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG;
  loomc_result_t* raw_result = nullptr;
  const auto status = loomc_compile_artifact(compiler.get(), workspace.get(), nullptr,
      module.get(), &options, allocator, &raw_result);
  Owner<loomc_result_t, loomc_result_release> result(raw_result, loomc_result_release);
  ASSERT_TRUE(Ok(status)) << Diagnostics(result.get());
  ASSERT_TRUE(result && loomc_result_succeeded(result.get())) << Diagnostics(result.get());
  size_t checked = 0;
  for (size_t i = 0; i < loomc_result_artifact_count(result.get()); ++i) {
    const auto* artifact = loomc_result_artifact_at(result.get(), i);
    if (artifact->kind != LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) continue;
    loomc_launch_config_program_t* raw_launch = nullptr;
    ASSERT_TRUE(Ok(loomc_launch_config_program_load(artifact, allocator, &raw_launch)));
    Owner<loomc_launch_config_program_t, loomc_launch_config_program_release> launch(
        raw_launch, loomc_launch_config_program_release);
    for (const char* name : names) {
      SCOPED_TRACE(name);
      auto function = loomc_launch_config_function_invalid();
      ASSERT_TRUE(Ok(loomc_launch_config_program_lookup_function(launch.get(),
          loomc_make_cstring_view(name), &function)));
      const uint64_t workload = 248320;
      loomc_launch_config_t config{};
      config.type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG; config.structure_size = sizeof(config);
      ASSERT_TRUE(Ok(loomc_launch_config_program_invoke(launch.get(), function, &workload, 1, &config)));
      EXPECT_EQ(config.workgroup_count.x, 248320u);
      EXPECT_EQ(config.workgroup_count.y, 1u); EXPECT_EQ(config.workgroup_count.z, 1u);
      EXPECT_EQ(config.workgroup_size.x, 64u);
      EXPECT_EQ(config.workgroup_size.y, 1u); EXPECT_EQ(config.workgroup_size.z, 1u);
      ++checked;
    }
  }
  EXPECT_EQ(checked, names.size());
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, ShippedTopology, ::testing::Values(false, true));
} // namespace
