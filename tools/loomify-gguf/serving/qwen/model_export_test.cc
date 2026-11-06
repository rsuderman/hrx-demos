// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/loomify-gguf/serving/qwen/model_export.h"
#include "tools/model/qwen_configuration.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status)) {
    loomc_status_free(status);
    return ::testing::AssertionSuccess();
  }
  auto m = loomc_status_message(status);
  std::string text(m.data, m.size);
  loomc_status_free(status);
  return ::testing::AssertionFailure() << text;
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto d = loomc_result_diagnostic_at(result, i);
      text.append(d->message.data, d->message.size);
      text += '\n';
    }
  return text;
}
TEST(QwenServeExport, RealDescriptorProducesOneEpochAndRelocatablePackage) {
  const auto *gguf = std::getenv("LOOMIFY_TEST_QWEN_GGUF");
  const auto *tokenizer = std::getenv("LOOMIFY_TEST_QWEN_TOKENIZER");
  ASSERT_NE(gguf, nullptr) << "manual gate requires immutable Qwen GGUF";
  ASSERT_NE(tokenizer, nullptr)
      << "manual gate requires matching tokenizer.json";
  std::unique_ptr<FILE, decltype(&std::fclose)> input(std::fopen(gguf, "rb"),
                                                      &std::fclose);
  ASSERT_TRUE(input);
  ASSERT_EQ(std::setvbuf(input.get(), nullptr, _IONBF, 0), 0);
  model::ModelDescriptor descriptor;
  std::string error;
  ASSERT_TRUE(model::ExtractQwenConfiguration(input.get(), &descriptor, &error))
      << error;
  input.reset();
  ASSERT_EQ(descriptor.layers.size(), 24u);
  loomc_target_environment_t *environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &environment)));
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t opts{};
  opts.next = &target;
  loomc_context_t *context = nullptr;
  ASSERT_TRUE(
      Ok(loomc_context_create(&opts, loomc_allocator_system(), &context)));
  loomc_workspace_t *workspace = nullptr;
  ASSERT_TRUE(Ok(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  ModelPackageOptions options;
  options.source_root = ".";
  options.gguf = gguf;
  options.tokenizer = tokenizer;
  options.context_capacity = 512;
  options.prefill_capacity = 128;
  loomc_module_t *module = nullptr;
  dispatch::QwenNativeGeometry geometry;
  ASSERT_TRUE(BuildQwenServeModule(descriptor, options, context, workspace,
                                   &module, &geometry, &error))
      << error;
  EXPECT_EQ(geometry.gdn_total_bytes, 20201472u);
  EXPECT_EQ(geometry.private_row_stride, 20204032u);
  EXPECT_EQ(geometry.map_origin, 2048u);
  EXPECT_EQ(geometry.residual_bytes, 2097152u);
  const auto directory = std::filesystem::path(std::getenv("TEST_TMPDIR"));
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    const bool text = format == LOOMC_SOURCE_FORMAT_TEXT;
    options.format = format;
    options.output_directory =
        (directory / (text ? "source-package" : "bytecode-package")).string();
    ASSERT_TRUE(WriteQwenPackage(descriptor, options, geometry, module, &error))
        << error;
    auto relocated =
        directory / (text ? "relocated-source" : "relocated-bytecode");
    std::filesystem::rename(options.output_directory, relocated);
    for (const char *provider :
         {"prepare.loom", "control.loom", "weights.loom"})
      ASSERT_TRUE(std::filesystem::is_regular_file(relocated / provider));
    auto numerical = relocated / (text ? "model.loom" : "model.loombc");
    loomc_module_t *loaded = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(numerical.c_str()), nullptr,
        loomc_allocator_system(), &loaded, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    auto root = loomc_make_cstring_view(kQwenEpochRoot);
    loomc_cmd_program_module_options_t compile{};
    compile.root_symbols = &root;
    compile.root_symbol_count = 1;
    loomc_product_t *product = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_cmd_program_product_build_module(
        workspace, loaded, &compile, loomc_allocator_system(), &product,
        &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(loomc_cmd_program_product_program_count(product), 1u);
    loomc_cmd_program_t epoch{};
    ASSERT_TRUE(loomc_cmd_program_product_program_at(product, 0, &epoch));
    EXPECT_GT(loomc_byte_sequence_length(epoch.artifact.contents), 0u);
    loomc_product_release(product);
    loomc_result_release(result);
    loomc_module_release(loaded);
    EXPECT_TRUE(std::filesystem::is_regular_file(relocated / "model.json"));
    EXPECT_FALSE(std::filesystem::exists(relocated / "model.json.tmp"));
  }
  loomc_module_release(module);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
}
} // namespace
} // namespace hrx::loomify_gguf
