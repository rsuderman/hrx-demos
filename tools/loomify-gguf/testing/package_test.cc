// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/gguf_model_builder.h"
#include "gtest/gtest.h"
#include "tools/loomify-gguf/testing/package_buffer_vm.h"
#include "loomc/target/amdgpu.h"
#include <array>
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
bool HeaderVariant(const fs::path& asset, const std::string& before,
                   const std::string& after, const fs::path& output,
                   std::string* error) {
  std::unique_ptr<FILE, decltype(&std::fclose)> input(std::fopen(asset.c_str(), "rb"), std::fclose);
  if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0)) return false;
  model::GgufCatalog catalog;
  if (!model::ReadGgufCatalog(input.get(), &catalog, error)) return false;
  std::string header(catalog.data_offset, '\0');
  if (std::fseek(input.get(), 0, SEEK_SET) ||
      std::fread(header.data(), 1, header.size(), input.get()) != header.size()) return false;
  const auto position = header.find(before);
  if (position == std::string::npos || position < 8 ||
      header.find(before, position + before.size()) != std::string::npos) return false;
  uint64_t old_length = 0;
  for (unsigned i = 0; i < 8; ++i)
    old_length |= uint64_t(static_cast<unsigned char>(header[position - 8 + i])) << (8 * i);
  if (old_length != before.size()) return false;
  header.replace(position, before.size(), after);
  for (unsigned i = 0; i < 8; ++i)
    header[position - 8 + i] = char((uint64_t(after.size()) >> (8 * i)) & 255);
  const auto delta = int64_t(after.size()) - int64_t(before.size());
  const auto end = uint64_t(int64_t(catalog.header_end) + delta);
  const auto offset = (end + catalog.alignment - 1) / catalog.alignment * catalog.alignment;
  header.resize(offset, '\0');
  std::ofstream file(output, std::ios::binary);
  file.write(header.data(), header.size());
  file.close();
  if (!file) return false;
  // Keep valid tensor ranges as sparse holes; no checkpoint payload is copied
  // or read, and no inference is performed against this metadata fixture.
  std::error_code ec;
  fs::resize_file(output, catalog.file_size + offset - catalog.data_offset, ec);
  return !ec;
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
    if (module && path.filename() == "control.loom") {
      for (const char* name : {"model_name", "output_next", "parse_tools"}) {
        loomc_module_function_t function{};
        ASSERT_TRUE(loomc_module_try_lookup_function(module, loomc_make_cstring_view(name), &function)) << name;
        EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION) << name;
        EXPECT_TRUE(function.flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC) << name;
      }
    }
    if (result) loomc_result_release(result);
    if (module) loomc_module_release(module);
  }
  void CheckPackageData(const fs::path& directory, const char* model_name,
                        const char* terminal) {
    for (bool bytecode : {false, true}) {
      std::string value, error;
      ASSERT_TRUE(testing::InvokePackageBuffer((directory / "control.loom").string(),
          "model_name", bytecode, &value, &error)) << error;
      EXPECT_EQ(value, model_name);
      ASSERT_TRUE(testing::InvokePackageBuffer((directory / "prepare.loom").string(),
          "loomify.tokens.terminal_marker", bytecode, &value, &error)) << error;
      EXPECT_EQ(value, terminal);
    }
  }
  std::vector<fs::path> CopyRuntimeProducts(const fs::path& destination) {
    std::vector<fs::path> copied;
    for (const auto& entry : fs::recursive_directory_iterator(root_ / "lib",
             fs::directory_options::follow_directory_symlink)) {
      if (!entry.is_regular_file()) continue;
      const auto relative = entry.path().lexically_relative(root_);
      if (entry.path().extension() != ".loombc" && relative != fs::path("lib/serving/weights.loom")) continue;
      fs::create_directories((destination / relative).parent_path());
      fs::copy_file(entry.path(), destination / relative);
      copied.push_back(relative);
    }
    return copied;
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
      EXPECT_NE(document.find(qwen ? "loomify-qwen35" : "loomify-llama"), std::string::npos);
      EXPECT_NE(document.find("\"context_capacity\": 513"), std::string::npos);
      EXPECT_NE(document.find("\"pool_capacity\": 576"), std::string::npos);
      EXPECT_NE(document.find("\"prefill_capacity\": 128"), std::string::npos);
      EXPECT_NE(document.find("\"rows\": 1"), std::string::npos);
      const auto control = Read(directory / "control.loom");
      EXPECT_NE(control.find(qwen ? "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67" : "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"), std::string::npos);
      CheckPackageData(directory, qwen ? "loomify-qwen35" : "loomify-llama",
          qwen ? "<|im_end|>" : "<|eot_id|>");
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
  const auto files = CopyRuntimeProducts(library);
  ASSERT_FALSE(files.empty());
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
    const auto copied = CopyRuntimeProducts(library);
    ASSERT_FALSE(copied.empty());
    const std::array<fs::path,7> files = {
      fs::path("lib/architectures") / profile / "implementation.loombc",
      fs::path("lib/architectures") / profile / "prepare_library.loombc",
      fs::path("lib/architectures") / profile / "execution.loombc",
      fs::path("lib/text/common/implementation.loombc"),
      fs::path("lib/serving/greedy.loombc"),
      qwen ? fs::path("lib/text/protocols/chatml_thinking/inline_history.loombc")
           : fs::path("lib/text/protocols/llama_headers/implementation.loombc"),
      fs::path("lib/serving/weights.loom")};
    build.library_root = library.string();
    GgufPackageOptions package{(scratch_ / (profile + "-output")).string(), Tokenizer(qwen)};
    for (const auto& file : files) {
      const auto saved = fs::path((library / file).string() + ".saved");
      fs::rename(library / file, saved);
      EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error)) << file;
      EXPECT_FALSE(fs::exists(package.output_directory));
      fs::rename(saved, library / file);
    }
    package.template_name = "unlisted-template";
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_NE(error.find("name"), std::string::npos);
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.template_name.reset();
    const auto valid_tokenizer = package.tokenizer;
    // Another real, valid native tokenizer is not the tokenizer for this GGUF.
    package.tokenizer = Tokenizer(!qwen);
    EXPECT_FALSE(builder->WritePackage(context_, workspace_, build, package, &error));
    EXPECT_FALSE(fs::exists(package.output_directory));
    package.tokenizer = valid_tokenizer;
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
TEST_F(GgufPackageTest, ChatAdmissionIsSeparateFromNumericalConstruction) {
  ASSERT_NE(Asset(false), nullptr);
  ASSERT_NE(Asset(true), nullptr);
  std::unique_ptr<GgufModelBuilder> originals[2];
  std::string error;
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(GgufModelBuilder::Open(Asset(i), &originals[i], &error)) << error;
    ASSERT_EQ(originals[i]->text_metadata().templates.size(), 1u);
  }
  for (int i = 0; i < 2; ++i) {
    const auto& own_template = originals[i]->text_metadata().templates[0].text;
    for (bool incompatible : {false, true}) {
      const auto fixture = scratch_ / (std::to_string(i) + (incompatible ? "-wrong-family.gguf" : "-unknown.gguf"));
      const auto replacement = incompatible ? originals[1-i]->text_metadata().templates[0].text
                                           : std::string(own_template.size(), '?');
      ASSERT_TRUE(HeaderVariant(Asset(i), own_template, replacement, fixture, &error)) << error;
      std::unique_ptr<GgufModelBuilder> builder;
      ASSERT_TRUE(GgufModelBuilder::Open(fixture.string(), &builder, &error)) << error;
      std::unique_ptr<BuiltGgufModule> numerical;
      ASSERT_TRUE(builder->BuildModule(context_, workspace_, BuildOptions(), &numerical, &error)) << error;
      ASSERT_TRUE(numerical->module());
      GgufPackageOptions package{(fixture.string() + ".output"), Tokenizer(i)};
      EXPECT_FALSE(builder->WritePackage(context_, workspace_, BuildOptions(), package, &error));
      EXPECT_NE(error.find(incompatible ? "unsupported tokenizer model/pre for selected text provider" : "unsupported"), std::string::npos) << error;
      EXPECT_FALSE(fs::exists(package.output_directory));
    }
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
// This case is released into the joined native gate only after S4 qualifies the
// real 27B numerical products. It never substitutes an 0.8B descriptor or skips.
TEST_F(GgufPackageTest, Qualified27ProfilePublishesSourceAndBytecode) {
  const char* gguf = std::getenv("LOOMIFY_TEST_QWEN_27_GGUF");
  const char* tokenizer = std::getenv("LOOMIFY_TEST_QWEN_27_TOKENIZER");
  ASSERT_NE(gguf, nullptr) << "immutable qualified 27B fixture required";
  ASSERT_NE(tokenizer, nullptr) << "matching adjacent 27B tokenizer required";
  std::unique_ptr<GgufModelBuilder> builder;
  std::string error;
  ASSERT_TRUE(GgufModelBuilder::Open(gguf, &builder, &error)) << error;
  for (const auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    const auto directory = scratch_ / ("qwen27-" + std::to_string(format));
    GgufPackageOptions package{directory.string(), tokenizer, format};
    ASSERT_TRUE(builder->WritePackage(context_, workspace_, BuildOptions(), package, &error)) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(std::distance(fs::directory_iterator(directory), fs::directory_iterator()), 5);
    EXPECT_FALSE(fs::exists(directory / "model.json.tmp"));
    const auto document = Read(directory / "model.json");
    EXPECT_NE(document.find("loomify-qwen35"), std::string::npos);
    EXPECT_EQ(document.find("loomify-qwen-08b"), std::string::npos);
    EXPECT_NE(document.find(fs::canonical(gguf).string()), std::string::npos);
    EXPECT_NE(document.find(fs::canonical(tokenizer).string()), std::string::npos);
    const auto control = Read(directory / "control.loom");
    EXPECT_NE(control.find("c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"), std::string::npos);
    CheckPackageData(directory, "loomify-qwen35", "<|im_end|>");
    Reload(directory / "prepare.loom");
    Reload(directory / "control.loom");
    Reload(directory / (format == LOOMC_SOURCE_FORMAT_TEXT ? "model.loom" : "model.loombc"));
  }
}
} // namespace
} // namespace hrx::loomify_gguf
