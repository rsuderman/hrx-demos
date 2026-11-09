// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/gguf_model_builder.h"
#include "gtest/gtest.h"
#include "loomc/target/amdgpu.h"
#include <array>
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace hrx::loomify_gguf {
namespace {
namespace fs = std::filesystem;
std::string Read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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
class GgufPackageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* srcdir = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    const char* temporary = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(srcdir && workspace && temporary);
    root_ = fs::path(srcdir) / workspace;
    scratch_ = fs::path(temporary) / ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::create_directories(scratch_);
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(), &environment_)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment_;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(), &context_)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace_)));
  }
  void TearDown() override {
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    loomc_target_environment_release(environment_);
  }
  void Reload(const fs::path& path) {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(context_, workspace_,
        loomc_make_cstring_view(path.c_str()), nullptr, loomc_allocator_system(), &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << path;
    EXPECT_NE(module, nullptr);
    if (module && path.filename() == "prepare.loom") {
      loomc_module_function_t function{};
      EXPECT_TRUE(loomc_module_try_lookup_function(module, loomc_make_cstring_view("prepare"), &function));
      EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION);
      EXPECT_TRUE(function.flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC);
    }
    if (result) loomc_result_release(result);
    if (module) loomc_module_release(module);
  }
  const char* Asset(bool qwen) { return std::getenv(qwen ? "LOOMIFY_TEST_QWEN_GGUF" : "LOOMIFY_TEST_GGUF"); }
  const char* Tokenizer(bool qwen) { return std::getenv(qwen ? "LOOMIFY_TEST_QWEN_TOKENIZER" : "LOOMIFY_TEST_TOKENIZER"); }
  GgufBuildOptions BuildOptions() {
    GgufBuildOptions result;
    result.library_root = root_.string();
    result.context_capacity = 513; // exercise final config-linked unaligned capacity
    return result;
  }
  fs::path root_, scratch_;
  loomc_target_environment_t* environment_ = nullptr;
  loomc_context_t* context_ = nullptr;
  loomc_workspace_t* workspace_ = nullptr;
};
TEST_F(GgufPackageTest, RealProfilesPublishRelocatableSourceAndBytecode) {
  for (bool qwen : {false,true}) {
    ASSERT_NE(Asset(qwen), nullptr) << "immutable real-model fixture required";
    ASSERT_NE(Tokenizer(qwen), nullptr) << "matching external tokenizer required";
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error;
    ASSERT_TRUE(GgufModelBuilder::Open(Asset(qwen), &builder, &error)) << error;
    const auto build = BuildOptions();
    for (const auto format : {LOOMC_SOURCE_FORMAT_TEXT,LOOMC_SOURCE_FORMAT_BYTECODE}) {
      const auto directory = scratch_ / (std::string(qwen ? "qwen-" : "llama-") + std::to_string(format));
      GgufPackageOptions package{directory.string(), Tokenizer(qwen), format};
      error = "stale";
      ASSERT_TRUE(builder->WritePackage(context_, workspace_, build, package, &error)) << error;
      EXPECT_TRUE(error.empty());
      EXPECT_EQ(std::distance(fs::directory_iterator(directory), fs::directory_iterator()), 5);
      EXPECT_FALSE(fs::exists(directory / "model.json.tmp"));
      const auto document = Read(directory / "model.json");
      EXPECT_NE(document.find(qwen ? "loomify-qwen-08b" : "loomify-llama"), std::string::npos);
      EXPECT_NE(document.find("\"context_capacity\": 513"), std::string::npos);
      EXPECT_NE(document.find("\"pool_capacity\": 576"), std::string::npos);
      EXPECT_NE(document.find("\"prefill_capacity\": 128"), std::string::npos);
      EXPECT_NE(document.find("\"rows\": 1"), std::string::npos);
      EXPECT_EQ(Read(directory / "control.loom"), Read(root_ / "lib/architectures" / (qwen ? "qwen35" : "llama") / "control.loom"));
      EXPECT_EQ(Read(directory / "weights.loom"), Read(root_ / "lib/serving/weights.loom"));
      const auto prepared = Read(directory / "prepare.loom");
      for (const char* scope : {"input","json","prepare"})
        EXPECT_NE(prepared.find(std::string("import(\"") + scope + "\")"), std::string::npos);
      EXPECT_EQ(prepared.find("{{"), std::string::npos);
      const auto relocated = fs::path(directory.string() + "-relocated");
      fs::rename(directory, relocated);
      for (const char* companion : {"prepare.loom","control.loom","weights.loom"}) Reload(relocated / companion);
      Reload(relocated / (format == LOOMC_SOURCE_FORMAT_TEXT ? "model.loom" : "model.loombc"));
      const auto before = Read(relocated / "model.json");
      package.output_directory = relocated.string();
      EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
      EXPECT_EQ(Read(relocated / "model.json"), before);
    }
  }
}
TEST_F(GgufPackageTest, ModelCliUsesOnlyDeclaredBytecodeRunfilesOutsideCheckout) {
  const auto executable = scratch_ / "converter";
  fs::copy_file(root_ / "tools/loomify-gguf/loomify-gguf", executable);
  const auto runfiles = fs::path(executable.string() + ".runfiles");
  const auto library = runfiles / "_main";
  const std::array<fs::path,7> files = {
    "lib/architectures/llama/implementation.loombc",
    "lib/architectures/llama/prepare_library.loombc",
    "lib/architectures/llama/control.loom",
    "lib/architectures/qwen35/implementation.loombc",
    "lib/architectures/qwen35/prepare_library.loombc",
    "lib/architectures/qwen35/control.loom",
    "lib/serving/weights.loom"};
  for (const auto& file : files) {
    fs::create_directories((library / file).parent_path());
    fs::copy_file(root_ / file, library / file);
  }
  // Copy the executable and exactly its runtime product inventory. Neither its
  // original runfiles nor another test's transitive source fixtures can mask a
  // stale source-path sentinel in the CLI resolver.
  EXPECT_FALSE(fs::exists(library / "lib/architectures/llama/model.loom"));
  EXPECT_FALSE(fs::exists(library / "lib/architectures/qwen35/model.loom"));
  for (bool qwen : {true,false}) for (const char* format : {"bc","loom"}) {
    ASSERT_NE(Asset(qwen), nullptr);
    ASSERT_NE(Tokenizer(qwen), nullptr);
    const auto output = scratch_ / (std::string(qwen ? "qwen-" : "llama-") + format);
    std::vector<std::string> arguments = {executable.string(), "--stage=model",
      "--gguf=" + std::string(Asset(qwen)), "--tokenizer=" + std::string(Tokenizer(qwen)),
      "--output-dir=" + output.string(), "--format=" + std::string(format),
      "--context-capacity=512", "--prefill-capacity=128"};
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      if (setenv("RUNFILES_DIR", runfiles.c_str(), 1) || chdir(scratch_.c_str())) _exit(126);
      unsetenv("RUNFILES_MANIFEST_FILE");
      execv(executable.c_str(), argv.data());
      _exit(127);
    }
    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0) << (qwen ? "qwen" : "llama") << " " << format;
    EXPECT_EQ(std::distance(fs::directory_iterator(output), fs::directory_iterator()), 5);
    EXPECT_TRUE(fs::is_regular_file(output / "model.json"));
    Reload(output / "prepare.loom");
    Reload(output / (std::string(format) == "bc" ? "model.loombc" : "model.loom"));
  }
}
TEST_F(GgufPackageTest, MissingCompanionsAndBadInputsPublishNothing) {
  for (bool qwen : {false,true}) {
    ASSERT_NE(Asset(qwen), nullptr);
    ASSERT_NE(Tokenizer(qwen), nullptr);
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error;
    ASSERT_TRUE(GgufModelBuilder::Open(Asset(qwen), &builder, &error)) << error;
    auto build = BuildOptions();
    const auto profile = std::string(qwen ? "qwen35" : "llama");
    const auto library = scratch_ / profile;
    const std::array<fs::path,4> files = {
      fs::path("lib/architectures") / profile / "implementation.loombc",
      fs::path("lib/architectures") / profile / "prepare_library.loombc",
      fs::path("lib/architectures") / profile / "control.loom",
      fs::path("lib/serving/weights.loom")};
    for (const auto& file : files) {
      fs::create_directories((library / file).parent_path());
      fs::copy_file(root_ / file, library / file);
    }
    build.library_root = library.string();
    GgufPackageOptions package{(scratch_ / (profile + "-output")).string(), Tokenizer(qwen)};
    for (const auto& file : files) {
      const auto saved = fs::path((library / file).string() + ".saved");
      fs::rename(library / file, saved);
      EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error)) << file;
      EXPECT_FALSE(fs::exists(package.output_directory));
      fs::rename(saved, library / file);
    }
    const auto valid_tokenizer = package.tokenizer;
    package.tokenizer = (scratch_ / "missing-tokenizer.json").string();
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.tokenizer = valid_tokenizer;
    package.format = static_cast<loomc_source_format_t>(-1);
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.format = LOOMC_SOURCE_FORMAT_BYTECODE;
    build.prefill_capacity = 0;
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    build.prefill_capacity = 128;
    const auto asset_size = fs::file_size(Asset(qwen));
    package.output_directory = Asset(qwen);
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_EQ(fs::file_size(Asset(qwen)), asset_size);
    const auto tokenizer_before = Read(Tokenizer(qwen));
    package.output_directory = Tokenizer(qwen);
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_EQ(Read(Tokenizer(qwen)), tokenizer_before);
    const auto sentinel = scratch_ / (profile + "-sentinel");
    std::ofstream(sentinel) << "preserved output parent";
    package.output_directory = (sentinel / "child").string();
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_EQ(Read(sentinel), "preserved output parent");
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.output_directory = std::string("unwanted\0suffix", 15);
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_FALSE(fs::exists(scratch_ / "unwanted"));
  }
}
TEST_F(GgufPackageTest, WriteAndSerializationFailuresCleanOwnedDirectoryAndAllowRetry) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Tokenizer(false), nullptr);
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(Asset(false), &builder, &error)) << error;
  const auto build = BuildOptions();
  GgufPackageOptions package{(scratch_ / "preflight").string(), Tokenizer(false)};
  ASSERT_TRUE(builder->WritePackage(context_, workspace_, build, package, &error)) << error;
  uintmax_t companion_limit = 0;
  for (const char* name : {"prepare.loom", "control.loom", "weights.loom"})
    companion_limit = std::max(companion_limit, fs::file_size(fs::path(package.output_directory) / name));
  ASSERT_GT(fs::file_size(fs::path(package.output_directory) / "model.loombc"), companion_limit);
  // First fail the initial companion write; then allow every companion through
  // and fail native numerical serialization. Both calls must remove only their
  // newly owned directory and leave no advertised document or retry obstacle.
  for (rlim_t limit : {rlim_t(0), rlim_t(companion_limit)}) {
    SCOPED_TRACE(limit);
    package.output_directory = (scratch_ / ("retry-" + std::to_string(limit))).string();
    struct rlimit original{};
    ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &original), 0);
    auto limited = original; limited.rlim_cur = limit;
    auto handler = std::signal(SIGXFSZ, SIG_IGN);
    const int installed = setrlimit(RLIMIT_FSIZE, &limited);
    if (installed != 0) std::signal(SIGXFSZ, handler);
    ASSERT_EQ(installed, 0);
    const bool wrote = builder->WritePackage(context_, workspace_, build, package, &error);
    const int restored = setrlimit(RLIMIT_FSIZE, &original);
    std::signal(SIGXFSZ, handler);
    ASSERT_EQ(restored, 0);
    EXPECT_FALSE(wrote);
    EXPECT_FALSE(error.empty());
    if (limit == 0) EXPECT_NE(error.find("cannot write serving provider"), std::string::npos);
    else EXPECT_EQ(error.find("cannot write serving provider"), std::string::npos);
    EXPECT_FALSE(fs::exists(package.output_directory));
    ASSERT_TRUE(builder->WritePackage(context_, workspace_, build, package, &error)) << error;
    EXPECT_TRUE(fs::is_regular_file(fs::path(package.output_directory) / "model.json"));
  }
}
} // namespace
} // namespace hrx::loomify_gguf
