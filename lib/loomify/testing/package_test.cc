// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/source.h"
#include "lib/loomify/gguf_model_builder.h"
#include "gtest/gtest.h"
#include "lib/loomify/testing/package_buffer_vm.h"
#include "loomc/target/amdgpu.h"
#include <array>
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <vector>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace hrx::loomify_gguf {
namespace {
namespace fs = std::filesystem;
std::string Read(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : value) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}
bool HeaderVariant(const fs::path &asset, const std::string &before,
                   const std::string &after, const fs::path &output,
                   std::string *error) {
  std::unique_ptr<FILE, decltype(&std::fclose)> input(
      std::fopen(asset.c_str(), "rb"), std::fclose);
  if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0))
    return false;
  model::GgufCatalog catalog;
  if (!model::ReadGgufCatalog(input.get(), &catalog, error))
    return false;
  std::string header(catalog.data_offset, '\0');
  if (std::fseek(input.get(), 0, SEEK_SET) ||
      std::fread(header.data(), 1, header.size(), input.get()) != header.size())
    return false;
  const auto position = header.find(before);
  if (position == std::string::npos || position < 8 ||
      header.find(before, position + before.size()) != std::string::npos)
    return false;
  uint64_t old_length = 0;
  for (unsigned i = 0; i < 8; ++i)
    old_length |= uint64_t(static_cast<unsigned char>(header[position - 8 + i]))
                  << (8 * i);
  if (old_length != before.size())
    return false;
  header.replace(position, before.size(), after);
  for (unsigned i = 0; i < 8; ++i)
    header[position - 8 + i] = char((uint64_t(after.size()) >> (8 * i)) & 255);
  const auto delta = int64_t(after.size()) - int64_t(before.size());
  const auto end = uint64_t(int64_t(catalog.header_end) + delta);
  const auto offset =
      (end + catalog.alignment - 1) / catalog.alignment * catalog.alignment;
  header.resize(offset, '\0');
  std::ofstream file(output, std::ios::binary);
  file.write(header.data(), header.size());
  file.close();
  if (!file)
    return false;
  // Keep valid tensor ranges as sparse holes; no checkpoint payload is copied
  // or read, and no inference is performed against this metadata fixture.
  std::error_code ec;
  fs::resize_file(output, catalog.file_size + offset - catalog.data_offset, ec);
  return !ec;
}
class GgufPackageTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *srcdir = std::getenv("TEST_SRCDIR");
    const char *workspace = std::getenv("TEST_WORKSPACE");
    const char *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(srcdir && workspace && temporary);
    root_ = fs::path(srcdir) / workspace;
    scratch_ = fs::path(temporary) /
               ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::create_directories(scratch_);
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment_)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment_;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(Ok(
        loomc_context_create(&options, loomc_allocator_system(), &context_)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                          &workspace_)));
  }
  void TearDown() override {
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    loomc_target_environment_release(environment_);
  }
  void Reload(const fs::path &path) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
        context_, workspace_, loomc_make_cstring_view(path.c_str()), nullptr,
        loomc_allocator_system(), &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << path;
    EXPECT_NE(module, nullptr);
    if (module && path.filename() == "prepare.loom") {
      loomc_module_function_t function{};
      EXPECT_TRUE(loomc_module_try_lookup_function(
          module, loomc_make_cstring_view("prepare"), &function));
      EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION);
      EXPECT_TRUE(function.flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC);
    }
    if (module && path.filename() == "control.loom") {
      const char *hooks[] = {
          "encode_state",    "encode_epoch",  "publish_epoch",
          "epoch",           "step",          "request_defaults",
          "request_options", "bind_request",  "chat_begin",
          "chat_message",    "chat_end",      "prepare_input",
          "output_next",     "complete_text", "render_tool",
          "parse_tools",     "model_name",    "chat_image_limits",
          "chat_reasoning",  "resize_state"};
      for (const char *name : hooks) {
        loomc_module_function_t function{};
        ASSERT_TRUE(loomc_module_try_lookup_function(
            module, loomc_make_cstring_view(name), &function))
            << name;
        EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION) << name;
        EXPECT_TRUE(function.flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC) << name;
      }
      for (const char *name : {"complete_chat"}) {
        loomc_module_function_t function{};
        EXPECT_FALSE(loomc_module_try_lookup_function(
            module, loomc_make_cstring_view(name), &function))
            << name;
      }
    }
    if (module && (path.filename() == "model.loom" ||
                   path.filename() == "model.loombc")) {
      loomc_module_serialize_options_t inspect{};
      inspect.format = LOOMC_SOURCE_FORMAT_TEXT;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          module, &inspect, loomc_allocator_system(), &source)));
      const auto contents = loomc_source_contents(source);
      const std::string text(reinterpret_cast<const char *>(contents.data),
                             contents.data_length);
      EXPECT_EQ(text.find("target.decl"), std::string::npos) << path;
      EXPECT_NE(text.find("subgroup_size = 32"), std::string::npos) << path;
      EXPECT_NE(text.find("subgroup_size = 64"), std::string::npos) << path;
      EXPECT_NE(text.find("target(@llm.platform.selection.greedy.wave64)"),
                std::string::npos)
          << path;
      EXPECT_NE(
          text.find(
              "amdgpu.target<gfx11-generic> @llm.platform.selection.greedy.wave64"),
          std::string::npos)
          << path;
      loomc_source_release(source);
    }
    if (result)
      loomc_result_release(result);
    if (module)
      loomc_module_release(module);
  }
  void CheckPackageData(const fs::path &directory, const char *model_name,
                        const char *terminal) {
    for (bool bytecode : {false, true}) {
      std::string value, error;
      ASSERT_TRUE(
          testing::InvokePackageBuffer((directory / "control.loom").string(),
                                       "model_name", bytecode, &value, &error))
          << error;
      EXPECT_EQ(value, model_name);
      ASSERT_TRUE(testing::InvokePackageBuffer(
          (directory / "prepare.loom").string(),
          "loomify.tokens.terminal_marker", bytecode, &value, &error))
          << error;
      EXPECT_EQ(value, terminal);
    }
  }
  std::vector<fs::path> CopyRuntimeProducts(const fs::path &destination) {
    std::vector<fs::path> copied;
    for (const auto &entry : fs::recursive_directory_iterator(
             root_ / "lib", fs::directory_options::follow_directory_symlink)) {
      if (!entry.is_regular_file())
        continue;
      const auto relative = entry.path().lexically_relative(root_);
      if (entry.path().extension() != ".loombc" &&
          relative != fs::path("lib/serving/weights.loom") &&
          relative != fs::path("lib/serving/control_interface.loom") &&
          std::find(RawTextProducts().begin(), RawTextProducts().end(),
                    relative) == RawTextProducts().end())
        continue;
      fs::create_directories((destination / relative).parent_path());
      fs::copy_file(entry.path(), destination / relative);
      copied.push_back(relative);
    }
    return copied;
  }
  const std::vector<fs::path> &RawTextProducts() {
    static const std::vector<fs::path> paths = {
        "lib/text/common/strings.loom",
        "lib/text/common/json.loom",
        "lib/text/common/jinja.loom",
        "lib/text/common/identity.loom",
        "lib/text/generated_input/environment.loom",
        "lib/text/generated_input/request.loom",
        "lib/text/generated_input/host.loom",
        "lib/text/generated_input/prepare.loom",
        "lib/text/output/stream.loom",
        "lib/text/output/json_calls.loom",
        "lib/text/output/schema.loom",
        "lib/text/output/tagged_calls.loom"};
    return paths;
  }
  const char *Asset(bool qwen) {
    return std::getenv(qwen ? "LOOMIFY_TEST_QWEN_GGUF" : "LOOMIFY_TEST_GGUF");
  }
  const char *Tokenizer(bool qwen) {
    return std::getenv(qwen ? "LOOMIFY_TEST_QWEN_TOKENIZER"
                            : "LOOMIFY_TEST_TOKENIZER");
  }
  GgufBuildOptions BuildOptions() {
    GgufBuildOptions result;
    result.library_root = root_.string();
    result.context_capacity =
        576; // Serve fixed backing requires page-aligned context
    return result;
  }
  bool Publish(const GgufModelBuilder &builder, const GgufBuildOptions &build,
               const GgufPackageOptions &package, std::string *error) {
    std::unique_ptr<GgufConfiguration> prepared;
    return builder.Configure(build, &package, &prepared, error) &&
           prepared->WritePackage(context_, workspace_, error);
  }
  bool Build(const GgufModelBuilder &builder, const GgufBuildOptions &options,
             std::unique_ptr<BuiltGgufModule> *out, std::string *error) {
    std::unique_ptr<GgufConfiguration> prepared;
    return builder.Configure(options, nullptr, &prepared, error) &&
           prepared->BuildModule(context_, workspace_, out, error);
  }
  int RunCli(const fs::path &executable, const fs::path &runfiles,
             std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), executable.string());
    std::vector<char *> argv;
    for (auto &argument : arguments)
      argv.push_back(argument.data());
    argv.push_back(nullptr);
    const pid_t child = fork();
    if (child < 0)
      return -1;
    if (child == 0) {
      if (setenv("RUNFILES_DIR", runfiles.c_str(), 1) ||
          chdir(scratch_.c_str()))
        _exit(126);
      unsetenv("RUNFILES_MANIFEST_FILE");
      execv(executable.c_str(), argv.data());
      _exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
      return -1;
    return WEXITSTATUS(status);
  }
  fs::path root_, scratch_;
  loomc_target_environment_t *environment_ = nullptr;
  loomc_context_t *context_ = nullptr;
  loomc_workspace_t *workspace_ = nullptr;
};
TEST_F(GgufPackageTest, ConfigurationOwnsPackageInputsAndExactNumericalSource) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Tokenizer(false), nullptr);
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(Asset(false), &builder, &error)) << error;
  auto build = BuildOptions();
  const auto destination = scratch_ / "owned-package";
  GgufPackageOptions package{destination.string(), Tokenizer(false)};
  std::unique_ptr<GgufConfiguration> prepared;
  ASSERT_TRUE(builder->Configure(build, &package, &prepared, &error)) << error;
  const auto numerical = prepared->configuration_source();
  std::unique_ptr<GgufConfiguration> numerical_only, delegated_numerical;
  ASSERT_TRUE(builder->ConfigureNumerical(build, &numerical_only, &error))
      << error;
  ASSERT_TRUE(builder->Configure(build, nullptr, &delegated_numerical, &error))
      << error;
  EXPECT_EQ(numerical_only->configuration_source(), numerical);
  EXPECT_EQ(delegated_numerical->configuration_source(), numerical);
  EXPECT_FALSE(numerical_only->has_package_configuration());
  EXPECT_TRUE(numerical_only->package_data_source().empty());
  const auto data = prepared->package_data_source();
  EXPECT_TRUE(prepared->has_package_configuration());
  EXPECT_NE(numerical.find("@llama.publication.end_turn_id"),
            std::string::npos);
  EXPECT_EQ(data.find("@llama.publication.end_turn_id"), std::string::npos);
  EXPECT_NE(data.find("@loomify.text.model_name"), std::string::npos);
  EXPECT_NE(data.find("@jinja.evaluate"), std::string::npos);
  auto *saved = prepared.get();
  auto bad = package;
  bad.tokenizer = (scratch_ / "absent.json").string();
  EXPECT_FALSE(builder->Configure(build, &bad, &prepared, &error));
  EXPECT_EQ(prepared.get(), saved);
  // Text Configure owns fully prepared control and therefore requires helpers;
  // an output destination is only needed when publishing.
  auto inspection_build = build;
  auto inspection_package = package;
  inspection_package.output_directory.clear();
  std::unique_ptr<GgufConfiguration> inspection;
  ASSERT_TRUE(builder->Configure(inspection_build, &inspection_package,
                                 &inspection, &error))
      << error;
  EXPECT_EQ(inspection->configuration_source(), numerical);
  EXPECT_EQ(inspection->package_data_source(), data);
  inspection_build.library_root.clear();
  auto *inspection_saved = inspection.get();
  EXPECT_FALSE(builder->Configure(inspection_build, &inspection_package,
                                  &inspection, &error));
  EXPECT_EQ(inspection.get(), inspection_saved);
  build.library_root = "invalid-after-configuration";
  build.prefill_capacity = 0;
  package.output_directory = "invalid-after-configuration";
  package.tokenizer.clear();
  builder.reset();
  std::unique_ptr<BuiltGgufModule> module;
  ASSERT_TRUE(prepared->BuildModule(context_, workspace_, &module, &error))
      << error;
  EXPECT_EQ(module->configuration_source(), numerical);
  ASSERT_TRUE(prepared->WritePackage(context_, workspace_, &error)) << error;
  EXPECT_EQ(prepared->configuration_source(), numerical);
  EXPECT_EQ(prepared->package_data_source(), data);
  EXPECT_TRUE(fs::is_regular_file(destination / "model.json"));
  CheckPackageData(destination, "loomify-llama", "<|eot_id|>");
}
TEST_F(GgufPackageTest, ConfigurationCliKeepsNumericalInspectionResourceFree) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Tokenizer(false), nullptr);
  const auto executable = scratch_ / "converter";
  fs::copy_file(root_ / "tools/loomify-gguf/loomify-gguf", executable);
  const auto absent_runfiles = scratch_ / "no-products";
  const auto numeric = scratch_ / "inspect-numerical";
  const auto text = scratch_ / "inspect-text";
  const std::string input = "--gguf=" + std::string(Asset(false));
  EXPECT_EQ(RunCli(executable, absent_runfiles,
                   {input, "--configuration-output=" + numeric.string()}),
            0);
  EXPECT_EQ(
      std::distance(fs::directory_iterator(numeric), fs::directory_iterator()),
      1);
  EXPECT_FALSE(fs::exists(numeric / "serve-data.loom"));
  const auto numeric_source = Read(numeric / "numeric.loom");
  EXPECT_NE(numeric_source.find("@llama.publication.end_turn_id"),
            std::string::npos);
  EXPECT_NE(numeric_source.find("@loomify.config.layer_ffn"),
            std::string::npos);
  EXPECT_NE(RunCli(executable, absent_runfiles,
                   {input, "--tokenizer=" + std::string(Tokenizer(false)),
                    "--configuration-output=" + text.string()}),
            0);
  EXPECT_FALSE(fs::exists(text));
  const auto runfiles = fs::path(executable.string() + ".runfiles");
  ASSERT_FALSE(CopyRuntimeProducts(runfiles / "_main").empty());
  ASSERT_EQ(RunCli(executable, runfiles,
                   {input, "--tokenizer=" + std::string(Tokenizer(false)),
                    "--configuration-output=" + text.string()}),
            0);
  EXPECT_EQ(Read(text / "numeric.loom"), numeric_source);
  EXPECT_EQ(
      std::distance(fs::directory_iterator(text), fs::directory_iterator()), 2);
  EXPECT_NE(Read(text / "serve-data.loom").find("@loomify.text.model_name"),
            std::string::npos);
  const auto removed_option = scratch_ / "removed-option";
  EXPECT_NE(RunCli(executable, runfiles,
                   {input, "--tokenizer=" + std::string(Tokenizer(false)),
                    "--text-protocol=llama-headers",
                    "--configuration-output=" + removed_option.string()}),
            0);
  EXPECT_FALSE(fs::exists(removed_option));
  const auto retained = scratch_ / "retained-after-failure";
  const auto missing_native = scratch_ / "model.loombc";
  EXPECT_NE(RunCli(executable, absent_runfiles,
                   {input, "--configuration-output=" + retained.string(),
                    "--output=" + missing_native.string()}),
            0);
  EXPECT_EQ(Read(retained / "numeric.loom"), numeric_source);
  EXPECT_FALSE(fs::exists(missing_native));
  EXPECT_NE(RunCli(executable, absent_runfiles,
                   {input, "--configuration-output=" + numeric.string()}),
            0);
  EXPECT_EQ(Read(numeric / "numeric.loom"), numeric_source);
  const auto same = scratch_ / "same-destination";
  EXPECT_NE(RunCli(executable, absent_runfiles,
                   {input, "--configuration-output=" + same.string(),
                    "--output=" + same.string()}),
            0);
  EXPECT_FALSE(fs::exists(same));
  for (const auto &flags : std::vector<std::vector<std::string>>{
           {input},
           {input, "--output="},
           {input, "--configuration-output="},
           {input, "--output=x", "--output-dir=y", "--tokenizer=z"},
           {input, "--output=x", "--tokenizer=z"},
           {input, "--output=x", "--platform=amdgpu:gfx942"},
           {input, "--configuration-output=x", "--chat-template=default"},
           {input, "--configuration-output=x", "--text-protocol=llama-headers"},
           {input, "--output=x", "--text-protocol=llama-headers"},
           {input, "--tokenizer=" + std::string(Tokenizer(false)),
            "--configuration-output=x", "--text-protocol=unknown-protocol"}})
    EXPECT_NE(RunCli(executable, absent_runfiles, flags), 0);
}
TEST_F(GgufPackageTest, EditedGgufSourceGeneratesControlAndSourceDefaults) {
  struct Fixture {
    const char *gguf;
    const char *tokenizer;
    const char *name;
    const char *before;
    const char *after;
    const char *literal;
  };
  const Fixture fixtures[] = {
      {Asset(false), Tokenizer(false), "llama-headers", "\"26 Jul 2024\"",
       "('10 Nov 2026' if true else '26 Jul 2024')", "10 Nov 2026"},
      {Asset(true), Tokenizer(true), "chatml-inline-history",
       "'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>'",
       "'<|im_start|>' + message.role + '\\nSOURCE_PREFIX ' + content + '<|im_end|>'",
       "SOURCE_PREFIX "},
      {std::getenv("LOOMIFY_TEST_QWEN_27_GGUF"),
       std::getenv("LOOMIFY_TEST_QWEN_27_TOKENIZER"),
       "chatml-explicit-reasoning", "reasoning_effort|default('xhigh')",
       "reasoning_effort|default('low')", "low"},
      {std::getenv("LOOMIFY_TEST_HYPERCLOVAX_GGUF"),
       std::getenv("LOOMIFY_TEST_HYPERCLOVAX_TOKENIZER"), "chatml-plain",
       "+ message['content'] +",
       "+ ('SOURCE_PREFIX ' if message['role'] == 'user' else '') + message['content'] +",
       "SOURCE_PREFIX "},
      {std::getenv("LOOMIFY_TEST_QWEN25_GGUF"),
       std::getenv("LOOMIFY_TEST_QWEN25_TOKENIZER"),
       "chatml-hermes-non-thinking",
       "'<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n'",
       "'<|im_start|>system\\nSOURCE_PREFIX You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n'",
       "SOURCE_PREFIX "},
      {std::getenv("LOOMIFY_TEST_QWEN3_GGUF"),
       std::getenv("LOOMIFY_TEST_QWEN3_TOKENIZER"), "chatml-hermes",
       "'<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>'",
       "'<|im_start|>' + message.role + '\\nSOURCE_PREFIX ' + message.content + '<|im_end|>'",
       "SOURCE_PREFIX "},
      {std::getenv("LOOMIFY_TEST_GEMMA_GGUF"),
       std::getenv("LOOMIFY_TEST_GEMMA_TOKENIZER"), "gemma-turns",
       "{{ message['content'] | trim }}",
       "{{ 'SOURCE_PREFIX ' + (message['content'] | trim) }}",
       "SOURCE_PREFIX "},
      {std::getenv("LOOMIFY_TEST_LFM3_GGUF"),
       std::getenv("LOOMIFY_TEST_LFM3_TOKENIZER"), "chatml-compact-reasoning",
       "{{- parse_content(message[\"content\"]) -}}",
       "{{- \"SOURCE_PREFIX \" + parse_content(message[\"content\"]) -}}",
       "SOURCE_PREFIX "},
      {std::getenv("LOOMIFY_TEST_SPARK_GGUF"),
       std::getenv("LOOMIFY_TEST_SPARK_TOKENIZER"), "spark-glm",
       "reasoning_effort|default('high')", "reasoning_effort|default('low')",
       "low"}};
  const char *external_output = std::getenv("LOOMIFY_TEST_MUTATION_OUTPUT");
  const auto mutation_root =
      external_output ? fs::path(external_output) : scratch_;
  ASSERT_TRUE(mutation_root.is_absolute());
  fs::create_directories(mutation_root);
  for (const auto &fixture_case : fixtures) {
    SCOPED_TRACE(fixture_case.name);
    ASSERT_NE(fixture_case.gguf, nullptr);
    ASSERT_NE(fixture_case.tokenizer, nullptr);
    std::unique_ptr<GgufModelBuilder> original;
    std::string error;
    ASSERT_TRUE(GgufModelBuilder::Open(fixture_case.gguf, &original, &error))
        << error;
    ASSERT_EQ(original->text_metadata().templates.size(), 1u);
    const auto &source = original->text_metadata().templates.front().text;
    auto changed = source;
    const auto offset = changed.find(fixture_case.before);
    ASSERT_NE(offset, std::string::npos);
    ASSERT_EQ(
        changed.find(fixture_case.before,
                     offset + std::string_view(fixture_case.before).size()),
        std::string::npos);
    changed.replace(offset, std::string_view(fixture_case.before).size(),
                    fixture_case.after);
    const auto fixture =
        mutation_root / (std::string(fixture_case.name) + ".gguf");
    ASSERT_TRUE(
        HeaderVariant(fixture_case.gguf, source, changed, fixture, &error))
        << error;
    std::unique_ptr<GgufModelBuilder> edited;
    ASSERT_TRUE(GgufModelBuilder::Open(fixture.string(), &edited, &error))
        << error;
    GgufPackageOptions package{(fixture.string() + ".package"),
                               fixture_case.tokenizer};
    std::unique_ptr<GgufConfiguration> configured;
    ASSERT_TRUE(
        original->Configure(BuildOptions(), &package, &configured, &error))
        << error;
    ASSERT_TRUE(
        edited->Configure(BuildOptions(), &package, &configured, &error))
        << error;
    const auto data = configured->package_data_source();
    EXPECT_NE(data.find("@jinja.evaluate"), std::string::npos);
    EXPECT_NE(data.find(Hex(fixture_case.literal)), std::string::npos);
    EXPECT_FALSE(fs::exists(package.output_directory));
    if (std::string_view(fixture_case.name) == "chatml-explicit-reasoning" ||
        std::string_view(fixture_case.name) == "spark-glm")
      EXPECT_NE(data.find("%default = scalar.constant 2 : i32"),
                std::string::npos);
    // Sparse metadata fixtures qualify generated host control only. Their
    // tensor payloads must never be used as a model-serving fixture.
    ASSERT_TRUE(configured->WritePackage(context_, workspace_, &error))
        << error;
    Reload(fs::path(package.output_directory) / "prepare.loom");
    Reload(fs::path(package.output_directory) / "control.loom");
    const auto template_path =
        mutation_root / (std::string(fixture_case.name) + ".jinja");
    std::ofstream template_output(template_path, std::ios::binary);
    template_output.write(changed.data(), changed.size());
    template_output.close();
    ASSERT_TRUE(template_output);
    std::ofstream manifest(mutation_root / "mutation-packages.jsonl",
                           std::ios::app);
    manifest
        << "{\"fixture\":" << std::quoted(fixture_case.name)
        << ",\"metadata_only\":true,\"gguf\":" << std::quoted(fixture.string())
        << ",\"template\":" << std::quoted(template_path.string())
        << ",\"package_control\":"
        << std::quoted(
               (fs::path(package.output_directory) / "control.loom").string())
        << ",\"tokenizer\":" << std::quoted(std::string(fixture_case.tokenizer))
        << "}\n";
    manifest.close();
    ASSERT_TRUE(manifest);
    const auto published_document =
        Read(fs::path(package.output_directory) / "model.json");
    package.template_name = "unlisted-template";
    auto *saved = configured.get();
    EXPECT_FALSE(
        edited->Configure(BuildOptions(), &package, &configured, &error));
    EXPECT_EQ(configured.get(), saved);
    EXPECT_NE(error.find("name"), std::string::npos);
    EXPECT_TRUE(
        fs::is_regular_file(fs::path(package.output_directory) / "model.json"));
    EXPECT_EQ(Read(fs::path(package.output_directory) / "model.json"),
              published_document);
  }
}
TEST_F(GgufPackageTest, EffortFactAndCompilerErrorsPublishNothing) {
  const char *gguf = std::getenv("LOOMIFY_TEST_QWEN_27_GGUF");
  const char *tokenizer = std::getenv("LOOMIFY_TEST_QWEN_27_TOKENIZER");
  ASSERT_NE(gguf, nullptr);
  ASSERT_NE(tokenizer, nullptr);
  std::unique_ptr<GgufModelBuilder> original;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(gguf, &original, &error)) << error;
  const auto &source = original->text_metadata().templates.front().text;
  const std::string before = "reasoning_effort|default('xhigh')";
  const auto offset = source.find(before);
  ASSERT_NE(offset, std::string::npos);
  size_t index = 0;
  for (
      const auto &[replacement, diagnostic] :
      std::vector<std::pair<std::string, std::string>>{
          {"reasoning_effort|default('none')",
           "source reasoning effort default/domain cannot be advertised by native discovery"},
          {"reasoning_effort|default(other)", "STRING literal fallback"},
          {"reasoning_effort", "no literal default declaration"},
          {"unknown_call()", "tokenizer.chat_template"}}) {
    auto changed = source;
    changed.replace(offset, before.size(), replacement);
    const auto fixture =
        scratch_ / ("bad-source-" + std::to_string(index++) + ".gguf");
    ASSERT_TRUE(HeaderVariant(gguf, source, changed, fixture, &error)) << error;
    std::unique_ptr<GgufModelBuilder> edited;
    ASSERT_TRUE(GgufModelBuilder::Open(fixture.string(), &edited, &error))
        << error;
    GgufPackageOptions package{fixture.string() + ".package", tokenizer};
    EXPECT_FALSE(Publish(*edited, BuildOptions(), package, &error));
    EXPECT_NE(error.find(diagnostic), std::string::npos) << error;
    EXPECT_FALSE(fs::exists(package.output_directory));
  }
}
TEST_F(GgufPackageTest, NumericalCliExportsBothFormatsUsingInspectedSource) {
  ASSERT_NE(Asset(false), nullptr);
  const auto executable = scratch_ / "converter";
  fs::copy_file(root_ / "tools/loomify-gguf/loomify-gguf", executable);
  const auto runfiles = fs::path(executable.string() + ".runfiles");
  const auto relative =
      fs::path("lib/architectures/llama/implementation.loombc");
  fs::create_directories((runfiles / "_main" / relative).parent_path());
  fs::copy_file(root_ / relative, runfiles / "_main" / relative);
  const auto missing_output = scratch_ / "missing-platform.loombc";
  EXPECT_NE(RunCli(executable, runfiles,
                   {"--gguf=" + std::string(Asset(false)),
                    "--output=" + missing_output.string()}),
            0);
  EXPECT_FALSE(fs::exists(missing_output));
  const auto target_path =
      fs::path("lib/platforms/amdgpu/gfx11/targets.loombc");
  fs::create_directories((runfiles / "_main" / target_path).parent_path());
  fs::copy_file(root_ / target_path, runfiles / "_main" / target_path);
  EXPECT_FALSE(fs::exists(runfiles / "_main/lib/text"));
  EXPECT_FALSE(fs::exists(runfiles / "_main/lib/architectures/qwen35"));
  for (const char *format : {"loom", "bc"}) {
    const auto output = scratch_ / (std::string("numerical.") + format);
    const auto inspection = scratch_ / (std::string("inspection-") + format);
    ASSERT_EQ(RunCli(executable, runfiles,
                     {"--gguf=" + std::string(Asset(false)),
                      "--output=" + output.string(),
                      std::string("--format=") + format,
                      "--configuration-output=" + inspection.string()}),
              0);
    EXPECT_TRUE(fs::is_regular_file(inspection / "numeric.loom"));
    EXPECT_FALSE(fs::exists(inspection / "serve-data.loom"));
    Reload(output);
  }
}
TEST_F(GgufPackageTest,
       UnalignedContextKeepsNumericalModuleButCannotPublishServePackage) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Tokenizer(false), nullptr);
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(Asset(false), &builder, &error)) << error;
  auto build = BuildOptions();
  build.context_capacity = 513;
  std::unique_ptr<BuiltGgufModule> numerical;
  ASSERT_TRUE(Build(*builder, build, &numerical, &error)) << error;
  ASSERT_NE(numerical->module(), nullptr);
  for (const auto format :
       {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    GgufPackageOptions package{
        (scratch_ / ("unaligned-" + std::to_string(format))).string(),
        Tokenizer(false), format};
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_EQ(error, "Serve package context_capacity must be a multiple of 64");
    EXPECT_FALSE(fs::exists(package.output_directory));
  }
}
TEST_F(GgufPackageTest, GgufAliasPreservesFormatSuffixAndRejectsRebinding) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Tokenizer(false), nullptr);
  // Header-only GGUF variants already underpin package admission fixtures.
  // Retain that exact descriptor/template under an extensionless target.
  std::unique_ptr<GgufModelBuilder> original;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(Asset(false), &original, &error)) << error;
  const auto &text = original->text_metadata().templates.front().text;
  const auto target = scratch_ / "checkpoint-blob";
  ASSERT_TRUE(HeaderVariant(Asset(false), text, text, target, &error)) << error;
  const auto alias = scratch_ / "checkpoint.gguf";
  fs::create_symlink(target, alias);
  std::unique_ptr<GgufModelBuilder> builder;
  ASSERT_TRUE(GgufModelBuilder::Open(alias.string(), &builder, &error))
      << error;
  EXPECT_EQ(builder->descriptor().family, original->descriptor().family);
  std::unique_ptr<GgufConfiguration> configured;
  GgufPackageOptions package{(scratch_ / "aliased-package").string(),
                             Tokenizer(false)};
  ASSERT_TRUE(builder->Configure(BuildOptions(), &package, &configured, &error))
      << error;
  // Opening the canonical target produces the same numerical configuration.
  std::unique_ptr<GgufModelBuilder> direct;
  ASSERT_TRUE(GgufModelBuilder::Open(target.string(), &direct, &error))
      << error;
  std::unique_ptr<GgufConfiguration> direct_configuration;
  ASSERT_TRUE(
      direct->Configure(BuildOptions(), nullptr, &direct_configuration, &error))
      << error;
  EXPECT_EQ(configured->configuration_source(),
            direct_configuration->configuration_source());
  ASSERT_TRUE(configured->WritePackage(context_, workspace_, &error)) << error;
  const auto document = Read(fs::path(package.output_directory) / "model.json");
  // Ordinary generated models do not promise restoration of arbitrary private
  // history from native prefix/session checkpoints.
  EXPECT_NE(document.find("\"session_cache_capacity\": 0"), std::string::npos);
  EXPECT_NE(document.find("\"prefix_cache_capacity\": 0"), std::string::npos);
  EXPECT_NE(document.find(alias.string()), std::string::npos);
  EXPECT_EQ(document.find(target.string()), std::string::npos);
  // A directory symlink followed by '..' must retain filesystem semantics.
  const auto parent = scratch_ / "real-parent";
  fs::create_directories(parent / "child");
  fs::create_symlink(target, parent / "parent-checkpoint.gguf");
  fs::create_directory_symlink(parent / "child", scratch_ / "parent-link");
  const auto parent_alias = scratch_ / "parent-link/../parent-checkpoint.gguf";
  std::unique_ptr<GgufModelBuilder> parent_builder;
  ASSERT_TRUE(
      GgufModelBuilder::Open(parent_alias.string(), &parent_builder, &error))
      << error;
  package.output_directory = (scratch_ / "parent-alias-package").string();
  ASSERT_TRUE(Publish(*parent_builder, BuildOptions(), package, &error))
      << error;
  EXPECT_NE(Read(fs::path(package.output_directory) / "model.json")
                .find(parent_alias.string()),
            std::string::npos);
  package.output_directory = (scratch_ / "rebound-package").string();
  ASSERT_TRUE(builder->Configure(BuildOptions(), &package, &configured, &error))
      << error;
  fs::remove(alias);
  fs::create_symlink(fs::canonical(Tokenizer(false)), alias);
  EXPECT_FALSE(configured->WritePackage(context_, workspace_, &error));
  EXPECT_EQ(error, "GGUF input alias no longer resolves to the opened source");
  EXPECT_FALSE(fs::exists(package.output_directory));
}
TEST_F(GgufPackageTest, RealModelsPublishRelocatableSourceAndBytecode) {
  for (bool qwen : {false, true}) {
    ASSERT_NE(Asset(qwen), nullptr) << "immutable real-model fixture required";
    ASSERT_NE(Tokenizer(qwen), nullptr)
        << "matching external tokenizer required";
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error;
    ASSERT_TRUE(GgufModelBuilder::Open(Asset(qwen), &builder, &error)) << error;
    const auto build = BuildOptions();
    for (const auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      const auto directory =
          scratch_ /
          (std::string(qwen ? "qwen-" : "llama-") + std::to_string(format));
      GgufPackageOptions package{directory.string(), Tokenizer(qwen), format};
      error = "stale";
      ASSERT_TRUE(Publish(*builder, build, package, &error)) << error;
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(std::distance(fs::directory_iterator(directory),
                              fs::directory_iterator()),
                5);
      EXPECT_FALSE(fs::exists(directory / "model.json.tmp"));
      const auto document = Read(directory / "model.json");
      EXPECT_NE(document.find(qwen ? "loomify-qwen35" : "loomify-llama"),
                std::string::npos);
      EXPECT_NE(document.find("\"context_capacity\": 576"), std::string::npos);
      EXPECT_EQ(document.find("\"pool_capacity\""), std::string::npos);
      EXPECT_NE(document.find("\"prefill_capacity\": 128"), std::string::npos);
      EXPECT_NE(document.find("\"rows\": 1"), std::string::npos);
      EXPECT_NE(document.find("\"max_sessions\": 1"), std::string::npos);
      const auto control = Read(directory / "control.loom");
      EXPECT_NE(
          control.find("// GGUF template \"tokenizer.chat_template\" bytes "),
          std::string::npos);
      EXPECT_EQ(control.find("Text protocol"), std::string::npos);
      EXPECT_NE(control.find("@jinja.evaluate"), std::string::npos);
      CheckPackageData(directory, qwen ? "loomify-qwen35" : "loomify-llama",
                       qwen ? "<|im_end|>" : "<|eot_id|>");
      EXPECT_EQ(Read(directory / "weights.loom"),
                Read(root_ / "lib/serving/weights.loom"));
      const auto prepared = Read(directory / "prepare.loom");
      for (const char *scope : {"input", "json", "prepare"})
        EXPECT_NE(prepared.find(std::string("import(\"") + scope + "\")"),
                  std::string::npos);
      EXPECT_EQ(prepared.find("{{"), std::string::npos);
      EXPECT_EQ(prepared.find("jinja."), std::string::npos);
      EXPECT_EQ(prepared.find("text.input_"), std::string::npos);
      const auto relocated = fs::path(directory.string() + "-relocated");
      fs::rename(directory, relocated);
      for (const char *companion :
           {"prepare.loom", "control.loom", "weights.loom"})
        Reload(relocated / companion);
      Reload(relocated / (format == LOOMC_SOURCE_FORMAT_TEXT ? "model.loom"
                                                             : "model.loombc"));
      const auto before = Read(relocated / "model.json");
      package.output_directory = relocated.string();
      EXPECT_FALSE(Publish(*builder, build, package, &error));
      EXPECT_EQ(Read(relocated / "model.json"), before);
    }
  }
}
TEST_F(GgufPackageTest,
       ModelCliUsesOnlyDeclaredBytecodeRunfilesOutsideCheckout) {
  const auto executable = scratch_ / "converter";
  fs::copy_file(root_ / "tools/loomify-gguf/loomify-gguf", executable);
  const auto runfiles = fs::path(executable.string() + ".runfiles");
  const auto library = runfiles / "_main";
  const auto files = CopyRuntimeProducts(library);
  ASSERT_FALSE(files.empty());
  // Copy the executable and exactly its runtime product inventory. Neither its
  // original runfiles nor another test's transitive source fixtures can mask a
  // stale source-path sentinel in the CLI resolver.
  EXPECT_FALSE(fs::exists(library / "lib/architectures/llama/model.loom"));
  EXPECT_FALSE(fs::exists(library / "lib/architectures/qwen35/model.loom"));
  for (bool qwen : {true, false})
    for (const char *format : {"bc", "loom"}) {
      ASSERT_NE(Asset(qwen), nullptr);
      ASSERT_NE(Tokenizer(qwen), nullptr);
      const auto output =
          scratch_ / (std::string(qwen ? "qwen-" : "llama-") + format);
      std::vector<std::string> arguments = {
          executable.string(),
          "--gguf=" + std::string(Asset(qwen)),
          "--tokenizer=" + std::string(Tokenizer(qwen)),
          "--output-dir=" + output.string(),
          "--format=" + std::string(format),
          "--context-capacity=512",
          "--prefill-capacity=128"};
      std::vector<char *> argv;
      for (auto &argument : arguments)
        argv.push_back(argument.data());
      argv.push_back(nullptr);
      const pid_t child = fork();
      ASSERT_GE(child, 0);
      if (child == 0) {
        if (setenv("RUNFILES_DIR", runfiles.c_str(), 1) ||
            chdir(scratch_.c_str()))
          _exit(126);
        unsetenv("RUNFILES_MANIFEST_FILE");
        execv(executable.c_str(), argv.data());
        _exit(127);
      }
      int status = 0;
      ASSERT_EQ(waitpid(child, &status, 0), child);
      ASSERT_TRUE(WIFEXITED(status));
      ASSERT_EQ(WEXITSTATUS(status), 0)
          << (qwen ? "qwen" : "llama") << " " << format;
      EXPECT_EQ(std::distance(fs::directory_iterator(output),
                              fs::directory_iterator()),
                5);
      EXPECT_TRUE(fs::is_regular_file(output / "model.json"));
      Reload(output / "prepare.loom");
      Reload(output /
             (std::string(format) == "bc" ? "model.loombc" : "model.loom"));
    }
}
TEST_F(GgufPackageTest, MissingCompanionsAndBadInputsPublishNothing) {
  for (bool qwen : {false, true}) {
    ASSERT_NE(Asset(qwen), nullptr);
    ASSERT_NE(Tokenizer(qwen), nullptr);
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error;
    ASSERT_TRUE(GgufModelBuilder::Open(Asset(qwen), &builder, &error)) << error;
    auto build = BuildOptions();
    const auto profile = std::string(qwen ? "qwen35" : "llama");
    const auto library = scratch_ / profile;
    const auto copied = CopyRuntimeProducts(library);
    ASSERT_FALSE(copied.empty());
    std::vector<fs::path> files = {
        fs::path("lib/platforms/amdgpu/gfx11/targets.loombc"),
        fs::path("lib/architectures") / profile / "implementation.loombc",
        fs::path("lib/architectures") / profile / "prepare_library.loombc",
        fs::path("lib/architectures") / profile / "execution.loombc",
        fs::path("lib/serving/greedy.loombc"),
        fs::path("lib/serving/control_interface.loom"),
        fs::path("lib/serving/weights.loom")};
    for (const auto &helper : RawTextProducts())
      if (qwen || helper != fs::path("lib/text/output/tagged_calls.loom"))
        files.push_back(helper);
    build.library_root = library.string();
    GgufPackageOptions package{(scratch_ / (profile + "-output")).string(),
                               Tokenizer(qwen)};
    for (const auto &file : files) {
      const auto saved = fs::path((library / file).string() + ".saved");
      fs::rename(library / file, saved);
      EXPECT_FALSE(Publish(*builder, build, package, &error)) << file;
      EXPECT_FALSE(fs::exists(package.output_directory));
      fs::rename(saved, library / file);
    }
    package.template_name = "unlisted-template";
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_NE(error.find("name"), std::string::npos);
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.template_name.reset();
    const auto valid_tokenizer = package.tokenizer;
    // Another real, valid native tokenizer is not the tokenizer for this GGUF.
    package.tokenizer = Tokenizer(!qwen);
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.tokenizer = valid_tokenizer;
    package.tokenizer = (scratch_ / "missing-tokenizer.json").string();
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.tokenizer = valid_tokenizer;
    package.format = static_cast<loomc_source_format_t>(-1);
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.format = LOOMC_SOURCE_FORMAT_BYTECODE;
    build.prefill_capacity = 0;
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    build.prefill_capacity = 128;
    const auto asset_size = fs::file_size(Asset(qwen));
    package.output_directory = Asset(qwen);
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_EQ(fs::file_size(Asset(qwen)), asset_size);
    const auto tokenizer_before = Read(Tokenizer(qwen));
    package.output_directory = Tokenizer(qwen);
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_EQ(Read(Tokenizer(qwen)), tokenizer_before);
    const auto sentinel = scratch_ / (profile + "-sentinel");
    std::ofstream(sentinel) << "preserved output parent";
    package.output_directory = (sentinel / "child").string();
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_EQ(Read(sentinel), "preserved output parent");
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.output_directory = std::string("unwanted\0suffix", 15);
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_FALSE(fs::exists(scratch_ / "unwanted"));
  }
}
TEST_F(GgufPackageTest, ChatAdmissionIsSeparateFromNumericalConstruction) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Asset(true), nullptr);
  std::unique_ptr<GgufModelBuilder> originals[2];
  std::string error;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(GgufModelBuilder::Open(Asset(i), &originals[i], &error))
        << error;
    ASSERT_EQ(originals[i]->text_metadata().templates.size(), 1u);
  }
  for (int i = 0; i < 2; ++i) {
    const auto &own_template = originals[i]->text_metadata().templates[0].text;
    for (bool incompatible : {false, true}) {
      const auto fixture =
          scratch_ / (std::to_string(i) +
                      (incompatible ? "-wrong-family.gguf" : "-unknown.gguf"));
      const auto replacement =
          incompatible ? originals[1 - i]->text_metadata().templates[0].text
                       : std::string(own_template.size(), '?');
      ASSERT_TRUE(
          HeaderVariant(Asset(i), own_template, replacement, fixture, &error))
          << error;
      std::unique_ptr<GgufModelBuilder> builder;
      ASSERT_TRUE(GgufModelBuilder::Open(fixture.string(), &builder, &error))
          << error;
      std::unique_ptr<BuiltGgufModule> numerical;
      std::unique_ptr<GgufConfiguration> numerical_configuration;
      ASSERT_TRUE(builder->ConfigureNumerical(BuildOptions(),
                                              &numerical_configuration, &error))
          << error;
      ASSERT_TRUE(numerical_configuration->BuildModule(context_, workspace_,
                                                       &numerical, &error))
          << error;
      ASSERT_TRUE(numerical->module());
      GgufPackageOptions package{(fixture.string() + ".output"), Tokenizer(i)};
      EXPECT_FALSE(Publish(*builder, BuildOptions(), package, &error));
      EXPECT_NE(error.find(incompatible ? "terminal" : "assistant"),
                std::string::npos)
          << error;
      EXPECT_FALSE(fs::exists(package.output_directory));
    }
  }
}
TEST_F(GgufPackageTest,
       WriteAndSerializationFailuresCleanOwnedDirectoryAndAllowRetry) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Tokenizer(false), nullptr);
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(Asset(false), &builder, &error)) << error;
  const auto build = BuildOptions();
  GgufPackageOptions package{(scratch_ / "preflight").string(),
                             Tokenizer(false), LOOMC_SOURCE_FORMAT_TEXT};
  ASSERT_TRUE(Publish(*builder, build, package, &error)) << error;
  uintmax_t companion_limit = 0;
  for (const char *name : {"prepare.loom", "control.loom", "weights.loom"})
    companion_limit =
        std::max(companion_limit,
                 fs::file_size(fs::path(package.output_directory) / name));
  // SOURCE is the late-write fixture: its measured numerical product is larger
  // than every companion. Numerical BC positive coverage is independent of
  // this explicit fixture precondition, and makes no relative-size promise.
  ASSERT_GT(fs::file_size(fs::path(package.output_directory) / "model.loom"),
            companion_limit);
  // First fail the initial companion write; then allow every companion through
  // and fail native numerical serialization. Both calls must remove only their
  // newly owned directory and leave no advertised document or retry obstacle.
  for (rlim_t limit : {rlim_t(0), rlim_t(companion_limit)}) {
    SCOPED_TRACE(limit);
    package.output_directory =
        (scratch_ / ("retry-" + std::to_string(limit))).string();
    std::unique_ptr<GgufConfiguration> configured;
    ASSERT_TRUE(builder->Configure(build, &package, &configured, &error))
        << error;
    EXPECT_FALSE(fs::exists(package.output_directory));
    struct rlimit original {};
    ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &original), 0);
    auto limited = original;
    limited.rlim_cur = limit;
    auto handler = std::signal(SIGXFSZ, SIG_IGN);
    const int installed = setrlimit(RLIMIT_FSIZE, &limited);
    if (installed != 0)
      std::signal(SIGXFSZ, handler);
    ASSERT_EQ(installed, 0);
    // Only publication is faulted. Configure's native analysis/probe scratch
    // and preparation have already completed with the normal process limits.
    const bool wrote = configured->WritePackage(context_, workspace_, &error);
    const int restored = setrlimit(RLIMIT_FSIZE, &original);
    std::signal(SIGXFSZ, handler);
    ASSERT_EQ(restored, 0);
    EXPECT_FALSE(wrote);
    EXPECT_FALSE(error.empty());
    if (limit == 0)
      EXPECT_NE(error.find("cannot write serving provider"), std::string::npos);
    else
      EXPECT_EQ(error.find("cannot write serving provider"), std::string::npos);
    EXPECT_FALSE(fs::exists(package.output_directory));
    ASSERT_TRUE(configured->WritePackage(context_, workspace_, &error))
        << error;
    EXPECT_TRUE(
        fs::is_regular_file(fs::path(package.output_directory) / "model.json"));
  }
}
// This case is released into the joined native gate only after S4 qualifies the
// real 27B numerical products. It never substitutes an 0.8B descriptor or
// skips.
TEST_F(GgufPackageTest, Qualified27ModelPublishesSourceAndBytecode) {
  const char *gguf = std::getenv("LOOMIFY_TEST_QWEN_27_GGUF");
  const char *tokenizer = std::getenv("LOOMIFY_TEST_QWEN_27_TOKENIZER");
  ASSERT_NE(gguf, nullptr) << "immutable qualified 27B fixture required";
  ASSERT_NE(tokenizer, nullptr) << "matching adjacent 27B tokenizer required";
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(gguf, &builder, &error)) << error;
  for (const auto format :
       {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    const auto directory = scratch_ / ("qwen27-" + std::to_string(format));
    GgufPackageOptions package{directory.string(), tokenizer, format};
    ASSERT_TRUE(Publish(*builder, BuildOptions(), package, &error)) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(std::distance(fs::directory_iterator(directory),
                            fs::directory_iterator()),
              5);
    EXPECT_FALSE(fs::exists(directory / "model.json.tmp"));
    const auto document = Read(directory / "model.json");
    EXPECT_NE(document.find("loomify-qwen35"), std::string::npos);
    EXPECT_EQ(document.find("loomify-qwen-08b"), std::string::npos);
    EXPECT_NE(document.find(fs::absolute(gguf).string()), std::string::npos);
    EXPECT_NE(document.find(fs::canonical(tokenizer).string()),
              std::string::npos);
    const auto control = Read(directory / "control.loom");
    EXPECT_NE(
        control.find("// GGUF template \"tokenizer.chat_template\" bytes "),
        std::string::npos);
    EXPECT_EQ(control.find("Text protocol"), std::string::npos);
    EXPECT_NE(control.find("@jinja.evaluate"), std::string::npos);
    CheckPackageData(directory, "loomify-qwen35", "<|im_end|>");
    Reload(directory / "prepare.loom");
    Reload(directory / "control.loom");
    Reload(directory / (format == LOOMC_SOURCE_FORMAT_TEXT ? "model.loom"
                                                           : "model.loombc"));
  }
}
TEST_F(GgufPackageTest, GemmaModelPublishesRelocatableSourceAndBytecode) {
  const char *gguf = std::getenv("LOOMIFY_TEST_GEMMA_GGUF");
  const char *tokenizer = std::getenv("LOOMIFY_TEST_GEMMA_TOKENIZER");
  ASSERT_NE(gguf, nullptr) << "immutable Gemma fixture required";
  ASSERT_NE(tokenizer, nullptr) << "matching Gemma tokenizer required";
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(gguf, &builder, &error)) << error;
  EXPECT_EQ(builder->descriptor().family, "gemma3");
  auto build = BuildOptions();
  build.context_capacity = 1024;
  for (const auto format :
       {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    const auto directory = scratch_ / ("gemma-" + std::to_string(format));
    GgufPackageOptions package{directory.string(), tokenizer, format};
    ASSERT_TRUE(Publish(*builder, build, package, &error)) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(std::distance(fs::directory_iterator(directory),
                            fs::directory_iterator()),
              5);
    EXPECT_FALSE(fs::exists(directory / "model.json.tmp"));
    const auto document = Read(directory / "model.json");
    EXPECT_NE(document.find("loomify-gemma3"), std::string::npos);
    EXPECT_NE(document.find(fs::absolute(gguf).string()), std::string::npos);
    EXPECT_NE(document.find(fs::canonical(tokenizer).string()),
              std::string::npos);
    EXPECT_NE(document.find("\"context_capacity\": 1024"), std::string::npos);
    const auto relocated = scratch_ / ("relocated-" + std::to_string(format));
    fs::rename(directory, relocated);
    EXPECT_EQ(Read(relocated / "model.json"), document);
    Reload(relocated / "prepare.loom");
    Reload(relocated / "control.loom");
    Reload(relocated / (format == LOOMC_SOURCE_FORMAT_TEXT ? "model.loom"
                                                           : "model.loombc"));
    CheckPackageData(relocated, "loomify-gemma3", "<end_of_turn>");
    package.output_directory = relocated.string();
    EXPECT_FALSE(Publish(*builder, build, package, &error));
    EXPECT_EQ(Read(relocated / "model.json"), document);
  }
}
TEST_F(GgufPackageTest,
       GemmaCliUsesOnlySelectedDeclaredProductsOutsideCheckout) {
  const char *gguf = std::getenv("LOOMIFY_TEST_GEMMA_GGUF");
  const char *tokenizer = std::getenv("LOOMIFY_TEST_GEMMA_TOKENIZER");
  ASSERT_NE(gguf, nullptr);
  ASSERT_NE(tokenizer, nullptr);
  const auto executable = scratch_ / "converter";
  fs::copy_file(root_ / "tools/loomify-gguf/loomify-gguf", executable);
  const auto runfiles = fs::path(executable.string() + ".runfiles");
  const auto library = runfiles / "_main";
  std::vector<fs::path> products = {
      "lib/platforms/amdgpu/gfx11/targets.loombc",
      "lib/architectures/gemma3/implementation.loombc",
      "lib/architectures/gemma3/prepare_library.loombc",
      "lib/architectures/gemma3/execution.loombc",
      "lib/serving/control_interface.loom",
      "lib/serving/greedy.loombc",
      "lib/serving/weights.loom"};
  products.insert(products.end(), RawTextProducts().begin(),
                  RawTextProducts().end());
  for (const auto &relative : products) {
    fs::create_directories((library / relative).parent_path());
    fs::copy_file(root_ / relative, library / relative);
  }
  EXPECT_FALSE(fs::exists(library / "lib/architectures/gemma3/model.loom"));
  EXPECT_FALSE(fs::exists(library / "lib/architectures/llama"));
  EXPECT_FALSE(fs::exists(library / "lib/architectures/qwen35"));
  for (const char *format : {"bc", "loom"}) {
    const auto directory = scratch_ / (std::string("gemma-") + format);
    ASSERT_EQ(RunCli(executable, runfiles,
                     {"--gguf=" + std::string(gguf),
                      "--tokenizer=" + std::string(tokenizer),
                      "--output-dir=" + directory.string(),
                      std::string("--format=") + format,
                      "--context-capacity=1024", "--prefill-capacity=128"}),
              0);
    EXPECT_EQ(std::distance(fs::directory_iterator(directory),
                            fs::directory_iterator()),
              5);
    Reload(directory / "prepare.loom");
    Reload(directory / "control.loom");
    Reload(directory /
           (std::string(format) == "bc" ? "model.loombc" : "model.loom"));
    CheckPackageData(directory, "loomify-gemma3", "<end_of_turn>");
  }
}
} // namespace
} // namespace hrx::loomify_gguf
