// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/loomify-gguf/serving/model_export.h"
#include "tools/model/llama_configuration.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
::testing::AssertionResult Ok(loomc_status_t s) {
  if (loomc_status_is_ok(s)) {
    loomc_status_free(s);
    return ::testing::AssertionSuccess();
  }
  auto message = loomc_status_message(s);
  std::string text(message.data, message.size);
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
TEST(LlamaServeExport, KnownGgufCompleteCommandProductReload) {
  const char *path = std::getenv("LOOMIFY_TEST_GGUF");
  ASSERT_NE(path, nullptr) << "manual gate requires LOOMIFY_TEST_GGUF";
  const char *tokenizer = std::getenv("LOOMIFY_TEST_TOKENIZER");
  ASSERT_NE(tokenizer, nullptr)
      << "manual gate requires LOOMIFY_TEST_TOKENIZER";
  std::unique_ptr<FILE, decltype(&std::fclose)> input(std::fopen(path, "rb"),
                                                      &std::fclose);
  ASSERT_TRUE(input);
  ASSERT_EQ(std::setvbuf(input.get(), nullptr, _IONBF, 0), 0);
  model::ModelDescriptor model;
  std::string error;
  ASSERT_TRUE(model::ExtractLlamaConfiguration(input.get(), &model, &error))
      << error;
  input.reset();
  loomc_target_environment_t *environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &environment)));
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t context_options{};
  context_options.next = &target;
  loomc_context_t *context = nullptr;
  ASSERT_TRUE(Ok(loomc_context_create(&context_options,
                                      loomc_allocator_system(), &context)));
  loomc_workspace_t *workspace = nullptr;
  ASSERT_TRUE(Ok(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  ModelPackageOptions options;
  options.source_root = ".";
  options.gguf = path;
  options.tokenizer = tokenizer;
  options.prefill_capacity = 128;
  options.context_capacity = 512;
  ConfiguredLlamaModel built;
  ASSERT_TRUE(
      BuildLlamaServeModule(model, options, context, workspace, &built, &error))
      << error;
  EXPECT_EQ(built.layout.token_capacity, options.prefill_capacity);
  EXPECT_EQ(built.layout.context_capacity, options.context_capacity);
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    options.format = format;
    const std::string suffix =
        format == LOOMC_SOURCE_FORMAT_TEXT ? "loom" : "bc";
    const auto destination = std::filesystem::path(std::getenv("TEST_TMPDIR")) /
                             ("package-" + suffix);
    options.output_directory = destination.string();
    ASSERT_TRUE(WriteLlamaPackage(model, options, built, &error)) << error;
    const auto relocated = destination.string() + "-relocated";
    std::filesystem::rename(destination, relocated);
    auto load = [&](const std::filesystem::path &provider,
                    loomc_module_t **out) {
      loomc_result_t *result = nullptr;
      auto status = loomc_module_deserialize_from_path(
          context, workspace, loomc_make_cstring_view(provider.c_str()),
          nullptr, loomc_allocator_system(), out, &result);
      const bool success = loomc_status_is_ok(status) && result &&
                           loomc_result_succeeded(result);
      EXPECT_TRUE(Ok(status)) << provider;
      EXPECT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      return success;
    };
    for (const char *name : {"prepare.loom", "control.loom", "weights.loom"}) {
      loomc_module_t *provider = nullptr;
      ASSERT_TRUE(load(std::filesystem::path(relocated) / name, &provider));
      loomc_module_release(provider);
    }
    loomc_module_t *loaded = nullptr;
    ASSERT_TRUE(load(std::filesystem::path(relocated) /
                         (format == LOOMC_SOURCE_FORMAT_TEXT ? "model.loom"
                                                             : "model.loombc"),
                     &loaded));
    loomc_result_t *r = nullptr;
    loomc_string_view_t roots[] = {loomc_make_cstring_view(kLlamaPrefillRoot),
                                   loomc_make_cstring_view(kLlamaDecodeRoot),
                                   loomc_make_cstring_view(kLlamaEpochRoot)};
    loomc_cmd_program_module_options_t compile{};
    compile.root_symbols = roots;
    compile.root_symbol_count = 3;
    loomc_product_t *product = nullptr;
    r = nullptr;
    ASSERT_TRUE(Ok(loomc_cmd_program_product_build_module(
        workspace, loaded, &compile, loomc_allocator_system(), &product, &r)));
    ASSERT_TRUE(r && loomc_result_succeeded(r)) << Diagnostics(r);
    ASSERT_NE(product, nullptr);
    EXPECT_EQ(loomc_cmd_program_product_program_count(product), 3u);
    for (unsigned i = 0; i < 3; ++i) {
      loomc_cmd_program_t command{};
      ASSERT_TRUE(loomc_cmd_program_product_program_at(product, i, &command));
      EXPECT_GT(loomc_byte_sequence_length(command.artifact.contents), 0u);
    }
    loomc_product_release(product);
    loomc_result_release(r);
    loomc_module_release(loaded);
  }
  built.module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
}
} // namespace
} // namespace hrx::loomify_gguf
