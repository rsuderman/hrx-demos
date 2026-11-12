// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Ordinary client: only the supported builder header exposes product APIs.
#include "lib/loomify/gguf_model_builder.h"
#include "gtest/gtest.h"
#include "loomc/target/amdgpu.h"
#include "loomc/source.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
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
// A header-only sparse fixture changes chat metadata without copying weights.
// It is linked for this API test and never executed as a numerical model.
bool HeaderWithUnlistedTemplate(const char *asset,
                                const std::filesystem::path &path) {
  std::unique_ptr<FILE, decltype(&std::fclose)> input(std::fopen(asset, "rb"),
                                                      std::fclose);
  if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0) != 0)
    return false;
  model::GgufCatalog catalog;
  std::string error;
  if (!model::ReadGgufCatalog(input.get(), &catalog, &error))
    return false;
  std::string header(catalog.header_end, '\0');
  if (fseeko(input.get(), 0, SEEK_SET) != 0 ||
      std::fread(header.data(), 1, header.size(), input.get()) != header.size())
    return false;
  auto number = [](std::string &out, uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
      out.push_back(char(value >> (8 * i)));
  };
  std::string field;
  const std::string key = "tokenizer.chat_template.s1_unlisted_fixture",
                    source = "unlisted-source";
  number(field, key.size(), 8);
  field += key;
  number(field, 8, 4);
  number(field, source.size(), 8);
  field += source;
  header.insert(24, field);
  const uint64_t count = catalog.metadata.size() + 1;
  for (unsigned i = 0; i < 8; ++i)
    header[16 + i] = char(count >> (8 * i));
  while (header.size() % catalog.alignment)
    header.push_back(0);
  const uint64_t size = header.size() + catalog.file_size - catalog.data_offset;
  std::unique_ptr<FILE, decltype(&std::fclose)> output(
      std::fopen(path.c_str(), "wbx"), std::fclose);
  if (!output ||
      std::fwrite(header.data(), 1, header.size(), output.get()) !=
          header.size() ||
      size > INT64_MAX ||
      fseeko(output.get(), int64_t(size - 1), SEEK_SET) != 0 ||
      std::fputc(0, output.get()) == EOF)
    return false;
  return std::fclose(output.release()) == 0;
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
  const auto *gemma = std::getenv("LOOMIFY_TEST_GEMMA_GGUF");
  const auto *plain = std::getenv("LOOMIFY_TEST_HYPERCLOVAX_GGUF");
  if (!llama || !qwen || !gemma) GTEST_SKIP() << "explicit immutable three-family gate";
  const auto *runfiles = std::getenv("TEST_SRCDIR");
  const auto *workspace_name = std::getenv("TEST_WORKSPACE");
  const auto *temporary = std::getenv("TEST_TMPDIR");
  ASSERT_TRUE(runfiles && workspace_name && temporary);
  const auto root = std::filesystem::path(runfiles) / workspace_name;
  std::vector<const char *> assets{llama, qwen, gemma};
  if (plain) assets.push_back(plain);
  size_t profile = 0;
  for (const char *asset : assets) {
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error = "stale";
    ASSERT_TRUE(GgufModelBuilder::Open(asset, &builder, &error)) << error;
    EXPECT_TRUE(error.empty());
    const auto text_snapshot = builder->text_metadata();
    ASSERT_TRUE(text_snapshot.vocabulary);
    EXPECT_FALSE(text_snapshot.templates.empty());
    EXPECT_TRUE(text_snapshot.template_issues.empty());
    auto *previous = builder.get();
    EXPECT_FALSE(GgufModelBuilder::Open("/nonexistent-loomify-test.gguf", &builder, &error));
    EXPECT_EQ(builder.get(), previous);
    const std::string architecture = builder->descriptor().family;
    auto directory = std::filesystem::path(temporary) /
        (architecture + "-" + std::to_string(profile++));
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
    // Configuration/selector admission is product-independent and preserves out.
    std::unique_ptr<GgufConfiguration> selected;
    ASSERT_TRUE(builder->Configure(options, nullptr, &selected, &error)) << error;
    EXPECT_EQ(selected->platform(), "amdgpu:gfx11-generic");
    const auto default_source = selected->configuration_source();
    options.platform = "amdgpu:gfx11-generic";
    ASSERT_TRUE(builder->Configure(options, nullptr, &selected, &error)) << error;
    EXPECT_EQ(selected->configuration_source(), default_source);
    auto* saved_selection = selected.get();
    options.platform = "amdgpu:gfx942";
    EXPECT_FALSE(builder->Configure(options, nullptr, &selected, &error));
    EXPECT_EQ(selected.get(), saved_selection);
    EXPECT_NE(error.find("unsupported generation platform: amdgpu:gfx942"), std::string::npos);
    options.platform = "amdgpu:gfx11-generic";
    std::unique_ptr<BuiltGgufModule> no_provider;
    EXPECT_FALSE(selected->BuildModule(context, workspace, &no_provider, &error));
    EXPECT_FALSE(no_provider);
    EXPECT_NE(error.find("targets.loombc"), std::string::npos) << error;
    const auto target_path = std::filesystem::path("lib/platforms/amdgpu/gfx11/targets.loombc");
    std::filesystem::create_directories((directory / target_path).parent_path());
    std::filesystem::copy_file(root / target_path, directory / target_path);
    const auto schedule_path = std::filesystem::path("lib/platforms/amdgpu/gfx11/attention_h256_implementation.loombc");
    EXPECT_FALSE(std::filesystem::exists(directory / schedule_path));
    if (architecture == "qwen35") {
      std::unique_ptr<BuiltGgufModule> missing_schedule;
      EXPECT_FALSE(selected->BuildModule(context, workspace, &missing_schedule, &error));
      EXPECT_FALSE(missing_schedule);
      EXPECT_NE(error.find(schedule_path.string()), std::string::npos) << error;
      std::filesystem::copy_file(root / schedule_path, directory / schedule_path);
    }
    // This sparse root has only this family product and its chosen targets;
    // Llama/Gemma neither load Qwen schedule nor require its resource.
    if (architecture != "qwen35") {
      EXPECT_FALSE(std::filesystem::exists(directory / "lib/architectures/qwen35"));
      std::unique_ptr<GgufModelBuilder> qwen_builder;
      ASSERT_TRUE(GgufModelBuilder::Open(qwen, &qwen_builder, &error)) << error;
      std::unique_ptr<GgufConfiguration> qwen_configuration;
      ASSERT_TRUE(qwen_builder->Configure(options, nullptr, &qwen_configuration, &error)) << error;
      std::unique_ptr<BuiltGgufModule> missing_qwen;
      EXPECT_FALSE(qwen_configuration->BuildModule(context, workspace, &missing_qwen, &error));
      EXPECT_FALSE(missing_qwen);
      EXPECT_NE(error.find("lib/architectures/qwen35/implementation.loombc"), std::string::npos) << error;
    }
    const auto malformed_path = directory / "unlisted-template.gguf";
    ASSERT_TRUE(HeaderWithUnlistedTemplate(asset, malformed_path));
    std::unique_ptr<GgufModelBuilder> malformed;
    ASSERT_TRUE(
        GgufModelBuilder::Open(malformed_path.string(), &malformed, &error))
        << error;
    EXPECT_FALSE(malformed->text_metadata().template_issues.empty());
    EXPECT_TRUE(malformed->text_metadata().token_issues.empty());
    std::unique_ptr<GgufConfiguration> malformed_configuration;
    ASSERT_TRUE(malformed->Configure(options, nullptr, &malformed_configuration, &error)) << error;
    malformed.reset();
    std::unique_ptr<BuiltGgufModule> malformed_module;
    ASSERT_TRUE(malformed_configuration->BuildModule(context, workspace, &malformed_module, &error)) << error;
    std::unique_ptr<GgufConfiguration> prepared, other, missing;
    ASSERT_TRUE(builder->Configure(options, nullptr, &prepared, &error)) << error;
    EXPECT_FALSE(prepared->has_package_configuration());
    EXPECT_TRUE(prepared->package_data_source().empty());
    const auto configuration = prepared->configuration_source();
    EXPECT_FALSE(configuration.empty());
    options.prefill_capacity = 1;
    ASSERT_TRUE(builder->Configure(options, nullptr, &other, &error)) << error;
    EXPECT_NE(other->configuration_source(), configuration);
    auto* saved_configuration = prepared.get();
    options.prefill_capacity = 0;
    EXPECT_FALSE(builder->Configure(options, nullptr, &prepared, &error));
    EXPECT_EQ(prepared.get(), saved_configuration);
    options.prefill_capacity = 128;
    options.library_root = (directory / "missing-library").string();
    // Preparation owns its root and needs no product files. A native error later
    // must not destroy either the inspected configuration or an earlier module.
    ASSERT_TRUE(builder->Configure(options, nullptr, &missing, &error)) << error;
    EXPECT_EQ(missing->configuration_source(), configuration);
    builder.reset();
    std::unique_ptr<BuiltGgufModule> first, second;
    ASSERT_TRUE(prepared->BuildModule(context, workspace, &first, &error)) << error;
    EXPECT_EQ(first->configuration_source(), configuration);
    EXPECT_EQ(first->platform(), "amdgpu:gfx11-generic");
    ASSERT_TRUE(other->BuildModule(context, workspace, &second, &error)) << error;
    auto* saved = first.get();
    EXPECT_FALSE(missing->BuildModule(context, workspace, &first, &error));
    EXPECT_EQ(first.get(), saved);
    EXPECT_FALSE(prepared->WritePackage(context, workspace, &error));
    EXPECT_EQ(prepared->configuration_source(), configuration);
    prepared.reset();
    other.reset();
    EXPECT_FALSE(text_snapshot.templates.front().text.empty());
    EXPECT_FALSE(text_snapshot.token_pieces.empty());
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
      loomc_module_serialize_options_t inspect{}; inspect.format = LOOMC_SOURCE_FORMAT_TEXT;
      loomc_source_t* source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(module, &inspect, loomc_allocator_system(), &source)));
      const auto contents = loomc_source_contents(source);
      const std::string text(reinterpret_cast<const char*>(contents.data), contents.data_length);
      EXPECT_EQ(text.find("target.decl"), std::string::npos) << path;
      EXPECT_NE(text.find("subgroup_size = 32"), std::string::npos) << path;
      EXPECT_NE(text.find("subgroup_size = 64"), std::string::npos) << path;
      EXPECT_NE(text.find("target(@llm.platform.selection.greedy.wave64)"), std::string::npos) << path;
      EXPECT_NE(text.find("amdgpu.target<gfx11-generic> @llm.platform.selection.greedy.wave64"), std::string::npos) << path;
      loomc_source_release(source);
      loomc_module_release(module);
    }
    loomc_workspace_release(workspace);
    loomc_context_release(context);
  }
}
} // namespace
} // namespace hrx::loomify_gguf
