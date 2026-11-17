// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/gguf_model_builder.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/configuration_fixture.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/token_policy.h"
#include "lib/loomify/template_probe_runner.h"
#include "lib/text/testing/selected_policy.h"
#include "gtest/gtest.h"
#include "loomc/compile.h"
#include "loomc/target/vm.h"
#include <memory>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace hrx::loomify_gguf {
namespace {
namespace fs = std::filesystem;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto m = loomc_status_message(status);
    ADD_FAILURE() << std::string(m.data, m.size);
  }
  loomc_status_free(status);
  return ok;
}
class AssemblyContract : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
    ASSERT_NE(std::getenv("TEST_TMPDIR"), nullptr);
    root_ =
        fs::path(std::getenv("TEST_SRCDIR")) / std::getenv("TEST_WORKSPACE");
    scratch_ = fs::path(std::getenv("TEST_TMPDIR")) /
               ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::create_directories(scratch_);
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context_)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                          &workspace_)));
  }
  void TearDown() override {
    if (workspace_)
      loomc_workspace_release(workspace_);
    if (context_)
      loomc_context_release(context_);
  }
  std::string Read(const std::string &name) {
    std::ifstream input(root_ / name, std::ios::binary);
    EXPECT_TRUE(input) << name;
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }
  std::string Write(const std::string &name, const std::string &contents) {
    const auto path = scratch_ / name;
    std::ofstream output(path, std::ios::binary);
    output << contents;
    output.close();
    EXPECT_TRUE(output);
    return path.string();
  }
  void SelectGemma() {
    const char *gguf = std::getenv("LOOMIFY_TEST_GEMMA_GGUF");
    const char *tokenizer = std::getenv("LOOMIFY_TEST_GEMMA_TOKENIZER");
    ASSERT_NE(gguf, nullptr) << "real Gemma GGUF fixture required";
    ASSERT_NE(tokenizer, nullptr) << "adjacent real tokenizer.json required";
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error;
    ASSERT_TRUE(GgufModelBuilder::Open(gguf, &builder, &error)) << error;
    ASSERT_EQ(builder->descriptor().family, "gemma3");
    GgufBuildOptions build;
    build.library_root = root_.string();
    build.prefill_capacity = 8;
    build.context_capacity = 64;
    GgufPackageOptions package;
    package.tokenizer = tokenizer;
    std::unique_ptr<GgufConfiguration> configuration;
    ASSERT_TRUE(builder->Configure(build, &package, &configuration, &error))
        << error;
    ASSERT_TRUE(configuration->has_package_configuration());
    gemma_source_ = configuration->configuration_source() +
                    configuration->package_data_source();
    gemma_prepare_source_ = configuration->configuration_source() +
                            EmitTerminalMarkerSource("<end_of_turn>");
    products_ = {(root_ / "lib/architectures/gemma3/execution.loombc").string(),
                 (root_ / "lib/serving/greedy.loombc").string()};
    for (const char *product :
         {"lib/text/common/strings.loom", "lib/text/common/json.loom",
          "lib/text/common/jinja.loom",
          "lib/text/generated_input/environment.loom",
          "lib/text/generated_input/request.loom",
          "lib/text/generated_input/host.loom",
          "lib/text/generated_input/prepare.loom",
          "lib/text/output/stream.loom", "lib/text/output/json_calls.loom",
          "lib/text/output/schema.loom", "lib/serving/control_interface.loom",
          "lib/text/common/identity.loom"})
      products_.push_back((root_ / product).string());
    roots_ = {"step",
              "epoch",
              "encode_epoch",
              "publish_epoch",
              "encode_state",
              "resize_state",
              "request_defaults",
              "request_options",
              "bind_request",
              "model_name",
              "chat_image_limits",
              "chat_begin",
              "chat_message",
              "chat_end",
              "complete_text",
              "render_tool",
              "output_next",
              "parse_tools",
              "chat_reasoning",
              "prepare_input",
              "loomify.text.model_name"};
  }
  void Select(int revision) {
    gemma_source_.clear();
    gemma_prepare_source_.clear();
    if (revision == 3) {
      SelectGemma();
      return;
    }
    const char *templates[] = {"lib/text/llama/templates/5816fce1.jinja",
                               "lib/text/qwen35/templates/7f0e5290.jinja",
                               "lib/text/qwen35/templates/c3cf9e34.jinja"};
    std::string error;
    jinja::TextFormatBindings globals;
    if (revision == 0)
      globals.strings["bos_token"] = "<|begin_of_text|>";
    ASSERT_TRUE(testing::BuildSelectedText(
        templates[revision], globals,
        revision == 0 ? "<|eot_id|>" : "<|im_end|>", &generated_, &error))
        << error;
    products_.clear();
    for (const auto &product : generated_.helper_requirements)
      products_.push_back((root_ / product).string());
    // Host contract fixture only: these publication IDs stand in for the
    // already derived numerical configuration. No weights or numerical kernels
    // execute.
    numerical_ =
        revision == 0
            ? "config.def @llama.publication.end_text_id = 128001 : i32\nconfig.def @llama.publication.end_message_id = 128008 : i32\nconfig.def @llama.publication.end_turn_id = 128009 : i32\n"
            : "";
    marker_ =
        EmitTerminalMarkerSource(revision == 0 ? "<|eot_id|>" : "<|im_end|>");
    marker_.clear(); // BuildSelectedText owns its independently bound terminal.
    identity_ =
        "global.rodata.def public @loomify.text.model_name = bytes(\"617373656d626c792d636f6e7472616374\")\n";
    products_.push_back(
        (root_ / "lib/serving/control_interface.loom").string());
    products_.push_back((root_ / "lib/text/common/identity.loom").string());
    roots_ = {"chat_begin",        "chat_message",
              "chat_end",          "complete_text",
              "render_tool",       "output_next",
              "parse_tools",       "model_name",
              "chat_image_limits", "loomify.text.model_name"};
  }
  std::string Configuration() const {
    return gemma_source_.empty()
               ? numerical_ + marker_ + generated_.source + identity_
               : gemma_source_;
  }
  std::string WithoutDefinition(std::string source, const std::string &prefix,
                                bool function) {
    const auto begin = source.find(prefix);
    EXPECT_NE(begin, std::string::npos) << prefix;
    if (begin == std::string::npos)
      return source;
    const auto end = source.find(function ? "\n}" : "\n", begin);
    EXPECT_NE(end, std::string::npos) << prefix;
    if (end != std::string::npos)
      source.erase(begin, end - begin + (function ? 2 : 1));
    return source;
  }
  bool Link(const std::string &config, const std::vector<std::string> &products,
            loomc_module_t **out, std::string *error) {
    return LinkConfiguredProducts(context_, workspace_, config, products,
                                  roots_, false, out, error);
  }
  void AssertParses(const std::string &path) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
        context_, workspace_, loomc_make_cstring_view(path.c_str()), nullptr,
        loomc_allocator_system(), &module, &result)));
    ASSERT_NE(result, nullptr);
    const bool succeeded = loomc_result_succeeded(result);
    loomc_result_release(result);
    if (module)
      loomc_module_release(module);
    ASSERT_TRUE(succeeded) << path;
  }
  void Reject(const std::string &config,
              const std::vector<std::string> &products,
              const char *missing = nullptr) {
    loomc_module_t *output = nullptr;
    std::string error;
    EXPECT_FALSE(Link(config, products, &output, &error));
    EXPECT_EQ(output, nullptr);
    EXPECT_FALSE(error.empty());
    if (missing)
      EXPECT_NE(error.find(missing), std::string::npos) << error;
    if (output)
      loomc_module_release(output);
  }
  void RequireGlobalData(const std::string &missing,
                         const std::string &expected) {
    auto product = [&](const std::string &configuration) {
      jinja::GeneratedInputRenderer generated;
      generated.entrypoint = "test.required_globals";
      for (const auto &path : products_)
        generated.helper_requirements.push_back(
            fs::path(path).lexically_relative(root_).string());
      generated.source = configuration + R"(
func.def public @test.required_globals(%unused: buffer) -> (buffer) {
  %globals = func.call @text.template_globals() : () -> (buffer)
  func.return %globals : buffer
}
)";
      return generated;
    };
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      std::unique_ptr<jinja::PreparedTemplate> complete;
      std::string error, actual;
      ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
          product(Configuration()), root_.string(), scratch_.string(), bytecode,
          &complete, &error))
          << error;
      ASSERT_TRUE(complete->Render("{}", &actual, &error)) << error;
      EXPECT_EQ(actual, expected);
      std::unique_ptr<jinja::PreparedTemplate> incomplete;
      if (jinja::PreparedTemplate::Prepare(product(missing), root_.string(),
                                           scratch_.string(), bytecode,
                                           &incomplete, &error)) {
        actual = "prior output";
        EXPECT_FALSE(incomplete->Render("{}", &actual, &error));
        EXPECT_EQ(actual, "prior output");
      } else {
        EXPECT_EQ(incomplete, nullptr);
      }
      EXPECT_FALSE(error.empty());
    }
  }
  jinja::GeneratedInputRenderer generated_;
  std::vector<std::string> products_;
  std::vector<const char *> roots_;
  std::string numerical_, marker_, identity_, gemma_source_,
      gemma_prepare_source_;
  fs::path root_, scratch_;
  loomc_context_t *context_ = nullptr;
  loomc_workspace_t *workspace_ = nullptr;
};
TEST_F(AssemblyContract, ShippedQwenFullGenerationRootsCompileForVm) {
  // Keep the actual Serve root cohort: isolated renderer/helper compilation
  // does not prove that the maintained shared call boundaries converge.
  const std::vector<const char *> roots = {"step",
                                           "epoch",
                                           "encode_epoch",
                                           "publish_epoch",
                                           "encode_state",
                                           "resize_state",
                                           "prepare_input",
                                           "output_next",
                                           "complete_text",
                                           "render_tool",
                                           "model_name",
                                           "chat_reasoning",
                                           "chat_image_limits",
                                           "chat_begin",
                                           "chat_message",
                                           "chat_end",
                                           "parse_tools",
                                           "request_defaults",
                                           "request_options",
                                           "bind_request"};
  std::vector<std::string> inputs;
  for (const char *path :
       {"llm/qwen/qwen_08b/control.loom", "llm/qwen/qwen_08b/config.loom",
        "llm/common/generation_control.loom",
        "llm/common/fixed_epoch_abi.loom"})
    inputs.push_back((root_ / "loom-pipeline" / path).string());
  loomc_target_environment_t *raw_environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_vm(loomc_allocator_system(),
                                                    &raw_environment)));
  std::unique_ptr<loomc_target_environment_t,
                  decltype(&loomc_target_environment_release)>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target_options{};
  target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target_options.structure_size = sizeof(target_options);
  target_options.target_environment = environment.get();
  loomc_context_options_t context_options{};
  context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
  context_options.structure_size = sizeof(context_options);
  context_options.next = &target_options;
  loomc_context_t *raw_context = nullptr;
  ASSERT_TRUE(Ok(loomc_context_create(&context_options,
                                      loomc_allocator_system(), &raw_context)));
  std::unique_ptr<loomc_context_t, decltype(&loomc_context_release)> vm_context(
      raw_context, loomc_context_release);
  loomc_module_t *raw_module = nullptr;
  std::string error;
  ASSERT_TRUE(LinkConfiguredProducts(vm_context.get(), workspace_, "", inputs,
                                     roots, false, &raw_module, &error))
      << error;
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module(
      raw_module, loomc_module_release);
  loomc_target_profile_t *raw_profile = nullptr;
  ASSERT_TRUE(Ok(loomc_target_profile_select(
      environment.get(), loomc_make_cstring_view("vm:core"),
      loomc_allocator_system(), &raw_profile)));
  std::unique_ptr<loomc_target_profile_t,
                  decltype(&loomc_target_profile_release)>
      profile(raw_profile, loomc_target_profile_release);
  loomc_compiler_t *raw_compiler = nullptr;
  ASSERT_TRUE(Ok(loomc_compiler_create(
      vm_context.get(), nullptr, loomc_allocator_system(), &raw_compiler)));
  std::unique_ptr<loomc_compiler_t, decltype(&loomc_compiler_release)> compiler(
      raw_compiler, loomc_compiler_release);
  std::vector<loomc_string_view_t> root_views;
  for (const char *root : roots)
    root_views.push_back(loomc_make_cstring_view(root));
  loomc_emit_options_t emit{};
  emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
  emit.structure_size = sizeof(emit);
  emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
  loomc_compile_artifact_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
  options.structure_size = sizeof(options);
  options.roots = root_views.data();
  options.root_count = root_views.size();
  options.target_profile = profile.get();
  options.emit_options = &emit;
  loomc_result_t *raw_result = nullptr;
  const auto status =
      loomc_compile_artifact(compiler.get(), workspace_, nullptr, module.get(),
                             &options, loomc_allocator_system(), &raw_result);
  std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)> result(
      raw_result, loomc_result_release);
  std::string diagnostics;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result.get()); ++i) {
      const auto *diagnostic = loomc_result_diagnostic_at(result.get(), i);
      diagnostics.append(diagnostic->message.data, diagnostic->message.size)
          .append("\n");
    }
  ASSERT_TRUE(Ok(status)) << diagnostics;
  ASSERT_TRUE(result && loomc_result_succeeded(result.get())) << diagnostics;
  EXPECT_GT(loomc_result_artifact_count(result.get()), 0u);
}
TEST_F(AssemblyContract,
       GeneratedComponentsAndActualPackageResolvePublicHooks) {
  for (int revision = 0; revision < 4; ++revision) {
    SCOPED_TRACE(revision);
    Select(revision);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(roots_.size(), revision == 3 ? 21u : 10u);
    ASSERT_EQ(std::set<std::string>(roots_.begin(), roots_.end()).size(),
              roots_.size());
    loomc_module_t *module = nullptr;
    std::string error;
    ASSERT_TRUE(Link(Configuration(), products_, &module, &error)) << error;
    for (const char *name : roots_) {
      if (std::string_view(name) == "loomify.text.model_name")
        continue;
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
          module, loomc_make_cstring_view(name), &function));
    }
    loomc_module_release(module);
  }
}
TEST_F(AssemblyContract, MissingActualProviderIsRejected) {
  for (int revision = 0; revision < 4; ++revision) {
    Select(revision);
    ASSERT_FALSE(HasFatalFailure());
    for (size_t omitted = 0; omitted < products_.size(); ++omitted) {
      const auto filename = fs::path(products_[omitted]).filename();
      if (filename != "host.loom" && filename != "stream.loom" &&
          filename != "identity.loom")
        continue;
      SCOPED_TRACE(std::to_string(revision) + ":" + std::to_string(omitted));
      auto incomplete = products_;
      incomplete.erase(incomplete.begin() + omitted);
      Reject(Configuration(), incomplete);
    }
  }
}
TEST_F(AssemblyContract, MissingGeneratedPackageDataIsRejected) {
  for (int revision = 0; revision < 3; ++revision) {
    SCOPED_TRACE(revision);
    Select(revision);
    ASSERT_FALSE(HasFatalFailure());
    Reject(numerical_ + marker_ + generated_.source, products_);
    // Llama's generated environment consumes its bound BOS. Qwen control
    // uses protocol framing constants; all weight-preparation terminal-marker
    // dependencies are checked separately below.
    if (revision == 0) {
      const auto no_bos =
          std::string("global.rodata.decl @request_adapters.globals\n") +
          WithoutDefinition(Configuration(),
                            "global.rodata.def @request_adapters.globals",
                            false);
      ASSERT_FALSE(HasFailure());
      EXPECT_NE(no_bos.find("global.load @request_adapters.globals"),
                std::string::npos);
      AssertParses(Write("no_bos_global.loom", no_bos));
      ASSERT_FALSE(HasFatalFailure());
      RequireGlobalData(no_bos, "{\"bos_token\":\"<|begin_of_text|>\"}");
    }
    Reject(numerical_ + marker_ + identity_, products_);
  }
}
TEST_F(AssemblyContract, ActualPreparationRequiresGeneratedTerminalMarker) {
  for (int fixture = 0; fixture < 3; ++fixture) {
    SCOPED_TRACE(fixture);
    const bool llama = fixture == 0;
    ConfigurationFacts facts;
    std::string error;
    const auto model = llama ? testing::LlamaModelFacts()
                             : testing::QwenModelFacts(fixture == 2);
    ASSERT_TRUE(
        DeriveConfiguration(model, "fixture.gguf", {128, 512}, &facts, &error))
        << error;
    const std::vector<std::string> providers = {
        (root_ / (llama ? "lib/architectures/llama/prepare_library.loombc"
                        : "lib/architectures/qwen35/prepare_library.loombc"))
            .string()};
    const auto marker =
        EmitTerminalMarkerSource(llama ? "<|eot_id|>" : "<|im_end|>");
    loomc_module_t *module = nullptr;
    ASSERT_TRUE(LinkConfiguredProducts(context_, workspace_,
                                       facts.source + marker, providers,
                                       {"prepare"}, false, &module, &error))
        << error;
    loomc_module_function_t prepare{};
    ASSERT_TRUE(loomc_module_try_lookup_function(
        module, loomc_make_cstring_view("prepare"), &prepare));
    EXPECT_EQ(prepare.kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION);
    EXPECT_TRUE(prepare.flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC);
    loomc_module_release(module);
    module = nullptr;
    EXPECT_FALSE(LinkConfiguredProducts(context_, workspace_, facts.source,
                                        providers, {"prepare"}, false, &module,
                                        &error));
    EXPECT_EQ(module, nullptr);
    EXPECT_FALSE(error.empty());
    if (module)
      loomc_module_release(module);
  }
}
TEST_F(AssemblyContract, GemmaGeneratedDataIsRequiredByActualConsumers) {
  Select(3);
  ASSERT_FALSE(HasFatalFailure());
  const auto no_identity = WithoutDefinition(
      Configuration(), "global.rodata.def public @loomify.text.model_name",
      false);
  ASSERT_FALSE(HasFailure());
  Reject(no_identity, products_, "loomify.text.model_name");
  for (const char *symbol :
       {"request_adapters.globals", "loomify.tokens.vocabulary_size"}) {
    SCOPED_TRACE(symbol);
    auto missing = WithoutDefinition(
        Configuration(),
        (std::string_view(symbol) == "request_adapters.globals"
             ? "global.rodata.def @"
             : "func.def public @") +
            std::string(symbol),
        std::string_view(symbol) != "request_adapters.globals");
    if (std::string_view(symbol) == "request_adapters.globals")
      missing = "global.rodata.decl @" + std::string(symbol) + "\n" + missing;
    ASSERT_FALSE(HasFailure());
    if (std::string_view(symbol) == "request_adapters.globals")
      EXPECT_NE(missing.find("global.load @request_adapters.globals"),
                std::string::npos);
    AssertParses(Write(std::string(symbol) + ".loom", missing));
    ASSERT_FALSE(HasFatalFailure());
    if (std::string_view(symbol) == "request_adapters.globals")
      RequireGlobalData(missing, "{\"bos_token\":\"<bos>\"}");
    else
      Reject(missing, products_, symbol);
  }
}
TEST_F(AssemblyContract,
       GemmaActualPreparationRequiresGeneratedTerminalMarker) {
  Select(3);
  ASSERT_FALSE(HasFatalFailure());
  const std::vector<std::string> providers = {
      (root_ / "lib/architectures/gemma3/prepare_library.loombc").string()};
  loomc_module_t *module = nullptr;
  std::string error;
  ASSERT_TRUE(LinkConfiguredProducts(context_, workspace_,
                                     gemma_prepare_source_, providers,
                                     {"prepare"}, false, &module, &error))
      << error;
  loomc_module_function_t prepare{};
  EXPECT_TRUE(loomc_module_try_lookup_function(
      module, loomc_make_cstring_view("prepare"), &prepare));
  EXPECT_EQ(prepare.kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION);
  EXPECT_TRUE(prepare.flags & LOOMC_MODULE_FUNCTION_FLAG_PUBLIC);
  loomc_module_release(module);
  module = nullptr;
  const auto missing = WithoutDefinition(
      gemma_prepare_source_, "func.def public @loomify.tokens.terminal_marker",
      true);
  ASSERT_FALSE(HasFailure());
  AssertParses(Write("gemma_no_terminal.loom", missing));
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_FALSE(LinkConfiguredProducts(context_, workspace_, missing, providers,
                                      {"prepare"}, false, &module, &error));
  EXPECT_EQ(module, nullptr);
  EXPECT_NE(error.find("loomify.tokens.terminal_marker"), std::string::npos)
      << error;
  if (module)
    loomc_module_release(module);
}
TEST_F(AssemblyContract, DuplicateActualHookProviderIsRejected) {
  for (int revision : {1, 3}) {
    SCOPED_TRACE(revision);
    Select(revision);
    ASSERT_FALSE(HasFatalFailure());
    auto duplicate = products_;
    duplicate.push_back(Write("duplicate_identity.loom",
                              Read("lib/text/common/identity.loom")));
    Reject(Configuration(), duplicate);
  }
}
TEST_F(AssemblyContract, WrongHookAbiIsRejectedAgainstActualTypedContract) {
  for (int revision : {1, 3}) {
    SCOPED_TRACE(revision);
    Select(revision);
    ASSERT_FALSE(HasFatalFailure());
    // Mutate only the actual identity provider, leaving the maintained 20-hook
    // declaration product and every other real implementation unchanged.
    auto wrong = Read("lib/text/common/identity.loom");
    const auto start = wrong.find("func.def public @model_name()");
    const auto end = wrong.find("\n}", start);
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    wrong.replace(
        start, end + 2 - start,
        "func.def public @model_name() -> (i64) {\n  %name = scalar.constant 1 : i64\n  func.return %name : i64\n}");
    auto products = products_;
    products.pop_back(); // The sole raw identity implementation is last.
    const auto wrong_path = Write("wrong_identity.loom", wrong);
    AssertParses(wrong_path);
    ASSERT_FALSE(HasFatalFailure());
    products.push_back(wrong_path);
    Reject(Configuration(), products);
  }
}
} // namespace
} // namespace hrx::loomify_gguf
