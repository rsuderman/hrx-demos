// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT

#include <memory>
#include <string>
#include <string_view>

#include "loomc/module.h"
#include "loomc/target/amdgpu.h"
#include "loomcxx-pipeline/testing/consumer_modules.h"
#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "tools/loomify-gguf/authoring/model-full/llama_step.h"
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

TEST_F(LoomifyConsumerTest, ImportedKernelRefAuthorsCommandsWithOwnedClosure) {
  auto definitions = Load("normalization.loombc");
  ASSERT_NE(definitions, nullptr);
  std::unique_ptr<ModuleBuilder> module, other;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                       definitions.get(), &module)));
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                       definitions.get(), &other)));
  definitions.reset();
  KernelRef kernel, foreign;
  ASSERT_TRUE(Ok(module->Kernel("qwen38_rmsnorm_f32_prefill", &kernel)));
  ASSERT_TRUE(Ok(other->Kernel("qwen38_rmsnorm_f32_prefill", &foreign)));
  const auto index = module->Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto f32 = module->Scalar(LOOM_SCALAR_TYPE_F32);
  const auto buffer = module->Buffer();
  ASSERT_TRUE(Ok(module->CheckKernelInterface(
      kernel, {index}, {index, f32, buffer, buffer, buffer})));
  EXPECT_EQ(loomc_status_consume_code(module->CheckKernelInterface(
                kernel, {}, {buffer, buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(module->CheckKernelInterface(
                kernel, {f32}, {index, f32, buffer, buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(module->CheckKernelInterface(
                kernel, {index}, {index, index, buffer, buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(module->CheckKernelInterface(
                foreign, {index}, {index, f32, buffer, buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  std::unique_ptr<CommandProgramBuilder> first, second;
  ASSERT_TRUE(Ok(module->DefineProgram(
      "consumer.first", {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_F32}, 3,
      &first)));
  ASSERT_TRUE(Ok(module->DefineProgram(
      "consumer.second", {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_F32}, 3,
      &second)));
  for (auto *program : {first.get(), second.get()}) {
    ASSERT_TRUE(Ok(program->Invoke(
        kernel, {program->Specialization(0)},
        {program->Specialization(0), program->Specialization(1),
         program->Binding(0), program->Binding(1), program->Binding(2)})));
    ASSERT_TRUE(Ok(program->Finish()));
  }
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(
      module->LinkAndVerify({first->Ref(), second->Ref()}, &linked, &result)));
  ModuleOwner linked_owner(linked);
  Owner<loomc_result_t, loomc_result_release> result_owner(result);
  ASSERT_TRUE(Succeeded(result));
  first.reset();
  second.reset();
  module.reset();
  const auto text = Roundtrip(linked);
  EXPECT_NE(text.find("@consumer.first"), std::string::npos);
  EXPECT_NE(text.find("@consumer.second"), std::string::npos);
  EXPECT_NE(text.find("kernel.launch @qwen38_rmsnorm_f32_prefill"),
            std::string::npos);
  EXPECT_NE(text.find("scalar.rsqrtf"), std::string::npos);
  EXPECT_NE(text.find("kernel.workgroup.reduce"), std::string::npos);
}

TEST_F(LoomifyConsumerTest, ProvidersOnlyFeedCurrentModelRmsNormAdapter) {
  auto definitions = Load("normalization_providers.loombc");
  ASSERT_NE(definitions, nullptr);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                       definitions.get(), &module)));
  definitions.reset();
  KernelRef kernel;
  ASSERT_TRUE(Ok(DefineLlamaRmsNorm(*module, "consumer.llama_norm", &kernel)));
  const auto index = module->Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer = module->Buffer();
  ASSERT_TRUE(Ok(module->CheckKernelInterface(
      kernel, {index}, {index, buffer, buffer, buffer})));
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("consumer.model_norm",
                                       {LOOM_SCALAR_TYPE_INDEX}, 3, &program)));
  ASSERT_TRUE(
      Ok(program->Invoke(kernel, {program->Specialization(0)},
                         {program->Specialization(0), program->Binding(0),
                          program->Binding(1), program->Binding(2)})));
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
  ModuleOwner linked_owner(linked);
  Owner<loomc_result_t, loomc_result_release> result_owner(result);
  ASSERT_TRUE(Succeeded(result));
  program.reset();
  module.reset();
  const auto text = Roundtrip(linked);
  EXPECT_NE(text.find("kernel.launch @consumer.llama_norm"), std::string::npos);
  EXPECT_NE(text.find("scalar.rsqrtf"), std::string::npos);
  EXPECT_NE(text.find("kernel.workgroup.reduce"), std::string::npos);
  EXPECT_EQ(text.find("qwen38_rmsnorm_f32_prefill"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf
