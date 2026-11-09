// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT

#include <memory>
#include <vector>
#include <string>
#include <string_view>

#include "loomc/module.h"
#include "loomc/target/amdgpu.h"
#include "loomcxx-pipeline/testing/consumer_modules.h"
#include "loomc/link.h"
#include "loomcxx-pipeline/testing/consumer_wrappers.h"

#include "gtest/gtest.h"

namespace hrx::loomify_gguf {
namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
using ModuleOwner = Owner<loomc_module_t, loomc_module_release>;

::testing::AssertionResult Ok(loomc_status_t status) {
  auto message = loomc_status_message(status);
  std::string text(message.data ? message.data : "", message.size);
  auto code = loomc_status_consume_code(status);
  return code == LOOMC_STATUS_OK ? ::testing::AssertionSuccess()
                                 : ::testing::AssertionFailure() << text;
}
::testing::AssertionResult Succeeded(const loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return ::testing::AssertionSuccess();
  std::string text;
  if (result) {
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto *diagnostic = loomc_result_diagnostic_at(result, i);
      text.append(diagnostic->message.data, diagnostic->message.size);
      text.push_back('\n');
    }
  }
  return ::testing::AssertionFailure() << text;
}

class LoomifyConsumerTest : public ::testing::Test {
protected:
  void SetUp() override {
    loomc_target_environment_t *environment = nullptr;
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment)));
    environment_.reset(environment);
    loomc_context_target_options_t target = {};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment;
    loomc_context_options_t options = {};
    options.next = &target;
    loomc_context_t *context = nullptr;
    ASSERT_TRUE(
        Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
    context_.reset(context);
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    workspace_.reset(workspace);
  }

  ModuleOwner Load(const char *name) {
    const auto *data = consumer_modules_create();
    for (size_t i = 0; i < consumer_modules_size(); ++i) {
      if (std::string_view(data[i].name) != name)
        continue;
      loomc_source_options_t options = {};
      options.identifier = loomc_make_cstring_view(name);
      options.contents = loomc_make_byte_span(data[i].data, data[i].size);
      options.storage = LOOMC_SOURCE_STORAGE_COPY;
      loomc_source_t *source = nullptr;
      if (!Ok(loomc_source_create(&options, loomc_allocator_system(), &source)))
        return {};
      Owner<loomc_source_t, loomc_source_release> source_owner(source);
      loomc_module_t *module = nullptr;
      loomc_result_t *result = nullptr;
      EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(
          context_.get(), workspace_.get(), source, nullptr,
          loomc_allocator_system(), &module, &result)));
      Owner<loomc_result_t, loomc_result_release> result_owner(result);
      EXPECT_TRUE(Succeeded(result));
      return ModuleOwner(module);
    }
    ADD_FAILURE() << "missing embedded module " << name;
    return {};
  }

  ModuleOwner Link(const char *provider_name, const char *wrapper_name,
                   const std::vector<const char *> &root_names,
                   const std::string &replacement = {}) {
    auto provider = Load(provider_name);
    if (!provider) return {};
    const auto *wrappers = consumer_wrappers_create();
    std::string text;
    for (size_t i = 0; i < consumer_wrappers_size(); ++i)
      if (std::string_view(wrappers[i].name) == wrapper_name)
        text.assign(reinterpret_cast<const char *>(wrappers[i].data), wrappers[i].size);
    EXPECT_FALSE(text.empty());
    if (!replacement.empty()) text = replacement;
    loomc_source_options_t source_options{};
    source_options.contents = loomc_make_byte_span(text.data(), text.size());
    source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t *source = nullptr;
    EXPECT_TRUE(Ok(loomc_source_create(&source_options, loomc_allocator_system(), &source)));
    Owner<loomc_source_t, loomc_source_release> source_owner(source);
    loomc_module_t *wrapper = nullptr;
    loomc_result_t *result = nullptr;
    auto parse = loomc_module_deserialize_from_source(context_.get(), workspace_.get(), source,
        nullptr, loomc_allocator_system(), &wrapper, &result);
    ModuleOwner wrapper_owner(wrapper);
    EXPECT_TRUE(Ok(parse));
    EXPECT_TRUE(Succeeded(result));
    if (!result || !loomc_result_succeeded(result)) {
      if (result) loomc_result_release(result);
      return {};
    }
    loomc_result_release(result);
    loomc_linker_t *linker = nullptr;
    EXPECT_TRUE(Ok(loomc_linker_create(context_.get(), nullptr, loomc_allocator_system(), &linker)));
    Owner<loomc_linker_t, loomc_linker_release> linker_owner(linker);
    loomc_link_module_provider_t providers[] = {
        {wrapper, {}, LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {provider.get(), {}, LOOMC_LINK_PROVIDER_ROLE_LIBRARY}};
    std::vector<loomc_string_view_t> roots;
    for (auto name : root_names) roots.push_back(loomc_make_cstring_view(name));
    loomc_link_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    options.structure_size = sizeof(options);
    options.mode = LOOMC_LINK_MODE_LINK;
    options.module_providers = providers;
    options.module_provider_count = 2;
    options.root_symbols = roots.data();
    options.root_symbol_count = roots.size();
    loomc_module_t *linked = nullptr;
    result = nullptr;
    auto status = loomc_link_module(linker, workspace_.get(), &options, &linked, &result);
    Owner<loomc_result_t, loomc_result_release> result_owner(result);
    if (replacement.empty()) {
      EXPECT_TRUE(Ok(status));
      EXPECT_TRUE(Succeeded(result));
    } else {
      EXPECT_TRUE(!loomc_status_is_ok(status) || !result || !loomc_result_succeeded(result));
      loomc_status_free(status);
    }
    return ModuleOwner(linked);
  }

  // The output belongs to Loom, independent of the source module and mutable
  // authoring owner. Check that it remains a verified ordinary Loom module.
  std::string Roundtrip(const loomc_module_t *module) {
    loomc_source_t *text = nullptr;
    EXPECT_TRUE(Ok(loomc_module_serialize_text_to_source(
        module, nullptr, loomc_allocator_system(), &text)));
    Owner<loomc_source_t, loomc_source_release> text_owner(text);
    if (!text)
      return {};
    auto bytes = loomc_source_contents(text);
    std::string ir(reinterpret_cast<const char *>(bytes.data),
                   bytes.data_length);
    loomc_module_t *restored = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_text_from_source(
        context_.get(), workspace_.get(), text, nullptr,
        loomc_allocator_system(), &restored, &result)));
    ModuleOwner restored_owner(restored);
    Owner<loomc_result_t, loomc_result_release> result_owner(result);
    EXPECT_TRUE(Succeeded(result));
    return ir;
  }

  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment_;
  Owner<loomc_context_t, loomc_context_release> context_;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_;
};

TEST_F(LoomifyConsumerTest, ImportedKernelLinksMaintainedCommandsWithOwnedClosure) {
  auto linked = Link("normalization.loombc", "consumer_commands.loom",
                     {"consumer.first", "consumer.second"});
  ASSERT_TRUE(linked);
  const auto text = Roundtrip(linked.get());
  EXPECT_NE(text.find("@consumer.first"), std::string::npos);
  EXPECT_NE(text.find("@consumer.second"), std::string::npos);
  EXPECT_NE(text.find("kernel.launch @qwen38_rmsnorm_f32_prefill"), std::string::npos);
  EXPECT_NE(text.find("scalar.rsqrtf"), std::string::npos);
  EXPECT_NE(text.find("kernel.workgroup.reduce"), std::string::npos);
}
TEST_F(LoomifyConsumerTest, NativeLinkRejectsMismatchedImportedKernelAbi) {
  // Every negative wrapper is internally well typed. Rejection therefore
  // checks its declared ABI against the actual imported numerical provider.
  for (const char *wrapper : {
      R"loom(kernel.decl @qwen38_rmsnorm_f32_prefill() launch(%input: buffer, %weight: buffer, %output: buffer)
command.program.def public @consumer.first() launch(%input: buffer, %weight: buffer, %output: buffer) {
  kernel.launch @qwen38_rmsnorm_f32_prefill[](%input, %weight, %output) : [](buffer, buffer, buffer)
  command.return
})loom",
      R"loom(kernel.decl @qwen38_rmsnorm_f32_prefill(%n: f32) launch(%rows: index, %epsilon: f32, %input: buffer, %weight: buffer, %output: buffer)
command.program.def public @consumer.first(%rows: index, %n: f32, %epsilon: f32) launch(%input: buffer, %weight: buffer, %output: buffer) {
  kernel.launch @qwen38_rmsnorm_f32_prefill[%n](%rows, %epsilon, %input, %weight, %output) : [f32](index, f32, buffer, buffer, buffer)
  command.return
})loom",
      R"loom(kernel.decl @qwen38_rmsnorm_f32_prefill(%rows: index) launch(%rows: index, %epsilon: index, %input: buffer, %weight: buffer, %output: buffer)
command.program.def public @consumer.first(%rows: index, %epsilon: index) launch(%input: buffer, %weight: buffer, %output: buffer) {
  kernel.launch @qwen38_rmsnorm_f32_prefill[%rows](%rows, %epsilon, %input, %weight, %output) : [index](index, index, buffer, buffer, buffer)
  command.return
})loom"}) {
    auto linked = Link("normalization.loombc", "consumer_commands.loom", {"consumer.first"}, wrapper);
    EXPECT_FALSE(linked);
  }
}
TEST_F(LoomifyConsumerTest, ProvidersOnlyFeedMaintainedModelRmsNormAdapter) {
  auto linked = Link("normalization_providers.loombc", "consumer_model_norm.loom", {"consumer.model_norm"});
  ASSERT_TRUE(linked);
  const auto text = Roundtrip(linked.get());
  EXPECT_NE(text.find("kernel.launch @consumer.llama_norm"), std::string::npos);
  EXPECT_NE(text.find("scalar.rsqrtf"), std::string::npos);
  EXPECT_NE(text.find("kernel.workgroup.reduce"), std::string::npos);
  EXPECT_EQ(text.find("qwen38_rmsnorm_f32_prefill"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf
