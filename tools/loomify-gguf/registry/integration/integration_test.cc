// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/workspace.h"
#include "tools/loomify-gguf/registry/emission/consumer_emitter.h"
#include "tools/loomify-gguf/registry/loading/module_loader.h"
#include "tools/loomify-gguf/registry/providers/provider_registry.h"

namespace hrx::loomify_gguf {
namespace {

template <typename T, void (*Release)(T*)>
struct Deleter {
  void operator()(T* value) const { Release(value); }
};
template <typename T, void (*Release)(T*)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
using Context = Owner<loomc_context_t, loomc_context_release>;
using Workspace = Owner<loomc_workspace_t, loomc_workspace_release>;
using Module = Owner<loomc_module_t, loomc_module_release>;
using Result = Owner<loomc_result_t, loomc_result_release>;
using Source = Owner<loomc_source_t, loomc_source_release>;
using PassProgram = Owner<loomc_pass_program_t, loomc_pass_program_release>;

std::string String(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status)) return ::testing::AssertionSuccess();
  const std::string message = String(loomc_status_message(status));
  const auto code = loomc_status_consume_code(status);
  return ::testing::AssertionFailure() << "API status " << code << ": " << message;
}

::testing::AssertionResult Succeeded(const loomc_result_t* result) {
  if (result && loomc_result_succeeded(result)) {
    return ::testing::AssertionSuccess();
  }
  std::string diagnostics;
  if (result) {
    for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      diagnostics += String(loomc_result_diagnostic_at(result, i)->message) + "\n";
    }
  }
  return ::testing::AssertionFailure() << "semantic operation failed: " << diagnostics;
}

std::string FixturePath(const char* name) {
  const char* root = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (!root || !workspace) return {};
  return std::string(root) + "/" + workspace +
         "/tools/loomify-gguf/registry/loading/testdata/" + name + ".loombc";
}

void Verify(loomc_module_t* module) {
  loomc_module_verify_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS;
  options.structure_size = sizeof(options);
  options.scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL;
  loomc_result_t* raw = nullptr;
  const auto status = loomc_module_verify(module, &options, loomc_allocator_system(), &raw);
  Result result(raw);
  ASSERT_TRUE(Ok(status));
  ASSERT_TRUE(Succeeded(result.get()));
}

Result SelectTemplates(loomc_context_t* context, loomc_workspace_t* workspace,
                       loomc_module_t* module) {
  loomc_pass_program_t* raw_program = nullptr;
  loomc_result_t* raw_prepared = nullptr;
  const auto prepare_status = loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view("select-templates{mode=final}"), nullptr,
      loomc_allocator_system(), &raw_program, &raw_prepared);
  PassProgram program(raw_program);
  Result prepared(raw_prepared);
  EXPECT_TRUE(Ok(prepare_status));
  EXPECT_TRUE(Succeeded(prepared.get()));
  if (!program) return {};
  loomc_result_t* raw_result = nullptr;
  EXPECT_TRUE(Ok(loomc_pass_program_run(program.get(), workspace, module, nullptr,
                                       loomc_allocator_system(), &raw_result)));
  return Result(raw_result);
}

void ExpectProbeAndRealClosure(loomc_module_t* module) {
  loomc_module_function_t function = {};
  ASSERT_TRUE(loomc_module_try_lookup_function(
      module, loomc_make_cstring_view("registry_scale_min_probe"), &function));
  EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
  std::string text;
  ASSERT_TRUE(Ok(DumpText(module, &text)));
  const auto start = text.find("@registry_scale_min_probe");
  ASSERT_NE(start, std::string::npos);
  const auto end = text.find("kernel.return", start);
  ASSERT_NE(end, std::string::npos);
  const auto body = text.substr(start, end - start);
  EXPECT_NE(body.find("view<2xf32>"), std::string::npos);
  EXPECT_TRUE(std::regex_search(body, std::regex(R"(view\.store[^\n]*\[0\])")));
  EXPECT_TRUE(std::regex_search(body, std::regex(R"(view\.store[^\n]*\[1\])")));
  // These operations are in the actual scale/min provider body, not the probe
  // or a template declaration. They must survive in its dependency closure.
  EXPECT_NE(text.find("view<12xi8>"), std::string::npos);
  EXPECT_NE(text.find("scalar.extf"), std::string::npos);
  EXPECT_NE(text.find("scalar.mulf"), std::string::npos);
  EXPECT_NE(text.find("ggml_qk_scale_min"), std::string::npos);
}

struct Serialized {
  loomc_source_format_t format;
  std::vector<uint8_t> bytes;
};

Serialized Serialize(loomc_module_t* module, loomc_source_format_t format) {
  loomc_module_serialize_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
  options.structure_size = sizeof(options);
  options.format = format;
  options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
  loomc_source_t* raw = nullptr;
  const auto status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &raw);
  Source source(raw);
  EXPECT_TRUE(Ok(status));
  if (!source) return {format, {}};
  const auto bytes = loomc_source_contents(source.get());
  return {format, std::vector<uint8_t>(bytes.data, bytes.data + bytes.data_length)};
}

std::vector<std::string> Inventory(const FrozenProviders& providers) {
  std::vector<std::string> result;
  for (const auto& symbol : providers.EnumerateSymbols()) {
    result.push_back(symbol.provider_name + ":" + symbol.module_name + ":" +
                     symbol.name + ":" + std::to_string(symbol.ordinal) + ":" +
                     std::to_string(symbol.flags));
  }
  return result;
}

class ProviderEmissionIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_context_t* context = nullptr;
    ASSERT_TRUE(Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    context_.reset(context);
  }

  Workspace NewWorkspace() {
    loomc_workspace_t* workspace = nullptr;
    EXPECT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    return Workspace(workspace);
  }

  std::unique_ptr<FrozenProviders> LoadRegistry(bool include_motif) {
    std::unique_ptr<ProviderRegistryBuilder> builder;
    EXPECT_TRUE(Ok(ProviderRegistryBuilder::Create(context_.get(), &builder)));
    if (!builder) return nullptr;
    auto workspace = NewWorkspace();
    if (!workspace) return nullptr;
    for (const char* name : {"decode_control_provider", "qk_common_provider"}) {
      if (!include_motif && std::string(name) == "qk_common_provider") continue;
      const std::string path = FixturePath(name);
      EXPECT_FALSE(path.empty()) << "Bazel runfiles environment is required";
      loomc_module_t* raw_module = nullptr;
      loomc_result_t* raw_result = nullptr;
      const auto status = LoadBytecode(context_.get(), workspace.get(),
                                      loomc_make_cstring_view(path.c_str()),
                                      &raw_module, &raw_result);
      Module module(raw_module);
      Result result(raw_result);
      EXPECT_TRUE(Ok(status));
      EXPECT_TRUE(Succeeded(result.get()));
      if (!module) return nullptr;
      EXPECT_TRUE(Ok(builder->AddModule(loomc_make_cstring_view(name), module.get())));
      // Module and operation result release here, before Freeze.
    }
    loomc_workspace_trim(workspace.get());
    workspace.reset();
    std::unique_ptr<FrozenProviders> frozen;
    loomc_result_t* raw_result = nullptr;
    const auto status = builder->Freeze(&frozen, &raw_result);
    Result result(raw_result);
    EXPECT_TRUE(Ok(status));
    EXPECT_TRUE(Succeeded(result.get()));
    builder.reset();
    return frozen;
  }

  Module Emit(loomc_workspace_t* workspace) {
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    const auto status = EmitScaleMinProbe(context_.get(), workspace, &raw_module,
                                         &raw_result);
    Module module(raw_module);
    Result result(raw_result);
    EXPECT_TRUE(Ok(status));
    EXPECT_TRUE(Succeeded(result.get()));
    return module;
  }

  Module Link(const FrozenProviders& providers, loomc_workspace_t* workspace,
              const loomc_module_t* probe) {
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    const auto status = LinkConsumer(
        providers.context(), workspace, providers.native_index(), probe,
        loomc_make_cstring_view("registry_scale_min_probe"), &raw_module, &raw_result);
    Module module(raw_module);
    Result result(raw_result);
    EXPECT_TRUE(Ok(status));
    EXPECT_TRUE(Succeeded(result.get()));
    return module;
  }

  Context context_;
};

TEST_F(ProviderEmissionIntegrationTest, RealProvidersProduceIndependentReloadableConsumers) {
  auto providers = LoadRegistry(true);
  ASSERT_NE(providers, nullptr);
  const auto before = Inventory(*providers);
  const auto kernel = providers->FindCandidates(
      loomc_make_cstring_view("qwen38_decode_commit_token"));
  ASSERT_EQ(kernel.size(), 1u);
  EXPECT_EQ(kernel.front().function_kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
  ASSERT_TRUE(kernel.front().export_symbol.has_value());
  EXPECT_EQ(*kernel.front().export_symbol, "qwen38_decode_commit_token");
  const auto symbols = providers->EnumerateSymbols();
  EXPECT_TRUE(std::any_of(symbols.begin(), symbols.end(), [](const auto& symbol) {
    return symbol.name == "ggml.qk.scale_min";
  }));
  EXPECT_TRUE(std::any_of(symbols.begin(), symbols.end(), [](const auto& symbol) {
    return symbol.name == "ggml_qk_scale_min";
  }));

  std::vector<Serialized> artifacts;
  for (int invocation = 0; invocation < 2; ++invocation) {
    auto workspace = NewWorkspace();
    ASSERT_NE(workspace, nullptr);
    auto probe = Emit(workspace.get());
    ASSERT_NE(probe, nullptr);
    auto linked = Link(*providers, workspace.get(), probe.get());
    ASSERT_NE(linked, nullptr);
    Result selected = SelectTemplates(context_.get(), workspace.get(), linked.get());
    ASSERT_TRUE(Succeeded(selected.get()));
    Verify(linked.get());
    ExpectProbeAndRealClosure(linked.get());
    artifacts.push_back(Serialize(linked.get(), LOOMC_SOURCE_FORMAT_TEXT));
    artifacts.push_back(Serialize(linked.get(), LOOMC_SOURCE_FORMAT_BYTECODE));
    EXPECT_EQ(Inventory(*providers), before);
  }
  providers.reset();
  context_.reset();

  for (const auto& artifact : artifacts) {
    ASSERT_FALSE(artifact.bytes.empty());
    loomc_context_t* raw_context = nullptr;
    ASSERT_TRUE(Ok(loomc_context_create(nullptr, loomc_allocator_system(), &raw_context)));
    Context fresh_context(raw_context);
    auto workspace = NewWorkspace();
    ASSERT_NE(workspace, nullptr);
    loomc_source_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
    options.structure_size = sizeof(options);
    options.format = artifact.format;
    options.identifier = loomc_make_cstring_view("standalone_consumer");
    options.contents = loomc_make_byte_span(artifact.bytes.data(), artifact.bytes.size());
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t* raw_source = nullptr;
    ASSERT_TRUE(Ok(loomc_source_create(&options, loomc_allocator_system(), &raw_source)));
    Source source(raw_source);
    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_result = nullptr;
    const auto status = loomc_module_deserialize_from_source(
        fresh_context.get(), workspace.get(), source.get(), nullptr,
        loomc_allocator_system(), &raw_module, &raw_result);
    Module module(raw_module);
    Result result(raw_result);
    ASSERT_TRUE(Ok(status));
    ASSERT_TRUE(Succeeded(result.get()));
    ASSERT_NE(module, nullptr);
    source.reset();
    result.reset();
    Verify(module.get());
    ExpectProbeAndRealClosure(module.get());
  }
}

TEST_F(ProviderEmissionIntegrationTest, MissingRealMotifProviderFailsAndValidRetrySucceeds) {
  auto incomplete = LoadRegistry(false);
  ASSERT_NE(incomplete, nullptr);
  auto workspace = NewWorkspace();
  ASSERT_NE(workspace, nullptr);
  auto probe = Emit(workspace.get());
  ASSERT_NE(probe, nullptr);
  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  const auto status = LinkConsumer(
      incomplete->context(), workspace.get(), incomplete->native_index(), probe.get(),
      loomc_make_cstring_view("registry_scale_min_probe"), &raw_module, &raw_result);
  Module unexpected(raw_module);
  Result result(raw_result);
  ASSERT_TRUE(Ok(status));
  ASSERT_NE(result, nullptr);
  if (loomc_result_succeeded(result.get())) {
    // Native linkage preserves open template applications. Final selection is
    // the public canonical stage that rejects their missing implementations.
    ASSERT_NE(unexpected, nullptr);
    Result selected = SelectTemplates(context_.get(), workspace.get(), unexpected.get());
    ASSERT_NE(selected, nullptr);
    EXPECT_FALSE(loomc_result_succeeded(selected.get()));
    EXPECT_GT(loomc_result_diagnostic_count(selected.get()), 0u);
  } else {
    EXPECT_GT(loomc_result_diagnostic_count(result.get()), 0u);
    EXPECT_EQ(unexpected, nullptr);
  }
  auto complete = LoadRegistry(true);
  ASSERT_NE(complete, nullptr);
  auto linked = Link(*complete, workspace.get(), probe.get());
  ASSERT_NE(linked, nullptr);
  Result selected = SelectTemplates(context_.get(), workspace.get(), linked.get());
  ASSERT_TRUE(Succeeded(selected.get()));
  Verify(linked.get());
  ExpectProbeAndRealClosure(linked.get());
}

}  // namespace
}  // namespace hrx::loomify_gguf
