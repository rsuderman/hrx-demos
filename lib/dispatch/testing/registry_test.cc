// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/registry.h"
#include "loomc/compile.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <memory>
namespace hrx::dispatch {
namespace {
std::string Message(loomc_status_t status) {
  auto view = loomc_status_message(status);
  std::string text(view.data ? view.data : "", view.size);
  loomc_status_free(status);
  return text;
}
::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << Message(status);
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  if (!result)
    return "no result";
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->message.data, d->message.size);
    text += '\n';
  }
  return text;
}
loomc_status_t Dump(const loomc_module_t *module, std::string *text) {
  loomc_module_serialize_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
  options.structure_size = sizeof(options);
  options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
  loomc_source_t *source = nullptr;
  auto status = loomc_module_serialize_text_to_source(
      module, &options, loomc_allocator_system(), &source);
  if (loomc_status_is_ok(status)) {
    auto bytes = loomc_source_contents(source);
    text->assign(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
    loomc_source_release(source);
  }
  return status;
}
class RegistryTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(
        Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
           std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
  }
  loomc_module_t *LoadEntry(const Registration &entry) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status = Load(context, workspace, entry, root, &module, &result);
    EXPECT_TRUE(Ok(status));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    return module;
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_F(RegistryTest, NormLoadsPrintsAndStrictLinksActualDefinition) {
  auto registry = Registry::Llama3B();
  const auto *entry = registry.Find(Id::RmsNorm);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->operation, operations::OpCode::RmsNorm);
  ASSERT_EQ(entry->operands.size(), 3u);
  EXPECT_EQ(entry->operands.back().access, operations::Access::Write);
  auto *module = LoadEntry(*entry);
  ASSERT_NE(module, nullptr);
  std::string text;
  ASSERT_TRUE(Ok(Dump(module, &text)));
  EXPECT_NE(text.find("llm_rmsnorm_row"), std::string::npos);
  EXPECT_NE(text.find("scalar.rsqrtf"), std::string::npos);
  std::unique_ptr<loomify_gguf::ModuleBuilder> builder;
  ASSERT_TRUE(Ok(loomify_gguf::ModuleBuilder::Create(context, workspace, module,
                                                     &builder)));
  loomc_module_release(module);
  loomify_gguf::KernelRef kernel;
  ASSERT_TRUE(Ok(builder->Kernel(entry->symbol, &kernel)));
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(builder->LinkAndVerify({kernel}, &linked, &result)));
  EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  if (linked)
    loomc_module_release(linked);
  if (result)
    loomc_result_release(result);
}
TEST_F(RegistryTest, WholeFamiliesLoadOwnedClosureAndDependentViews) {
  auto registry = Registry::Llama3B();
  for (auto id :
       {Id::GateUpPrefill, Id::GateUpDecode, Id::DownPrefill, Id::DownDecode}) {
    const auto *entry = registry.Find(id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->kind, Kind::CommandFamily);
    auto *module = LoadEntry(*entry);
    ASSERT_NE(module, nullptr);
    std::string text;
    ASSERT_TRUE(Ok(Dump(module, &text)));
    EXPECT_NE(text.find(entry->symbol), std::string::npos);
    EXPECT_NE(text.find("ggml_linear_q4k_q8_1_x4_compat"), std::string::npos);
    EXPECT_NE(text.find("ggml_linear_q6k_q8_1_x4_compat"), std::string::npos);
    EXPECT_NE(text.find("view<["), std::string::npos);
    EXPECT_EQ(text.find("qwen38_linear"), std::string::npos);
    loomc_module_release(module);
  }
}
TEST_F(RegistryTest, WholeFamilyApplicationsPublishAndTargetCompileKernels) {
  auto registry = Registry::Llama3B();
  auto entry = *registry.Find(Id::GateUpPrefill);
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/ffn_commands.loom");
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/ffn_config.loom");
  auto *module = LoadEntry(entry);
  ASSERT_NE(module, nullptr);
  std::vector<loomc_request_t *> requests;
  auto collect = [](void *user, loomc_request_t *request) -> loomc_status_t {
    static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
    return loomc_ok_status();
  };
  loomc_target_profile_t *profile = nullptr;
  ASSERT_TRUE(Ok(loomc_target_profile_select(
      environment, loomc_make_cstring_view("amdgpu:gfx1100"),
      loomc_allocator_system(), &profile)));
  std::vector<loomc_string_view_t> roots;
  for (const char *name :
       {"registry_gate_up_prefill", "registry_gate_up_decode",
        "registry_down_prefill", "registry_down_decode",
        "registry_down_q4_prefill", "registry_down_q4_decode"})
    roots.push_back(loomc_make_cstring_view(name));
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  options.root_symbols = roots.data();
  options.root_symbol_count = roots.size();
  options.kernel_target_profile = profile;
  options.request_sink.publish = collect;
  options.request_sink.user_data = &requests;
  loomc_product_t *product = nullptr;
  loomc_result_t *result = nullptr;
  auto status = loomc_cmd_program_product_build_module(
      workspace, module, &options, loomc_allocator_system(), &product, &result);
  EXPECT_TRUE(Ok(status));
  EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  if (result)
    loomc_result_release(result);
  if (product)
    loomc_product_release(product);
  loomc_module_release(module);
  EXPECT_GE(requests.size(), 4u);
  loomc_compiler_t *compiler = nullptr;
  ASSERT_TRUE(Ok(loomc_compiler_create(context, nullptr,
                                       loomc_allocator_system(), &compiler)));
  for (auto *request : requests) {
    loomc_module_t *leaf = nullptr;
    result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, loomc_request_source(request), nullptr,
        loomc_allocator_system(), &leaf, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    result = nullptr;
    if (leaf) {
      loomc_compile_artifact_options_t compile{};
      compile.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
      compile.structure_size = sizeof(compile);
      compile.target_profile = profile;
      EXPECT_TRUE(Ok(loomc_compile_artifact(compiler, workspace, nullptr, leaf,
                                            &compile, loomc_allocator_system(),
                                            &result)));
      EXPECT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      if (result && loomc_result_succeeded(result)) {
        EXPECT_GT(loomc_result_artifact_count(result), 0u);
      }
      if (result)
        loomc_result_release(result);
      loomc_module_release(leaf);
    }
    loomc_request_release(request);
  }
  loomc_compiler_release(compiler);
  loomc_target_profile_release(profile);
}
TEST_F(RegistryTest, PublicFamilyOverridesAreSelected) {
  auto registry = Registry::Llama3B();
  auto entry = *registry.Find(Id::GateUpPrefill);
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/ffn_commands.loom");
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/ffn_config.loom");
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/family_override.loom");
  auto *module = LoadEntry(entry);
  ASSERT_NE(module, nullptr);
  std::vector<loomc_request_t *> requests;
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  std::vector<loomc_string_view_t> roots;
  for (const char *name :
       {"registry_gate_up_prefill", "registry_gate_up_decode",
        "registry_down_prefill", "registry_down_decode"})
    roots.push_back(loomc_make_cstring_view(name));
  options.root_symbols = roots.data();
  options.root_symbol_count = roots.size();
  options.request_sink.publish =
      [](void *state, loomc_request_t *request) -> loomc_status_t {
    static_cast<std::vector<loomc_request_t *> *>(state)->push_back(request);
    return loomc_ok_status();
  };
  options.request_sink.user_data = &requests;
  loomc_product_t *product = nullptr;
  loomc_result_t *result = nullptr;
  EXPECT_TRUE(Ok(loomc_cmd_program_product_build_module(
      workspace, module, &options, loomc_allocator_system(), &product,
      &result)));
  EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  EXPECT_FALSE(requests.empty());
  for (auto *request : requests) {
    loomc_request_root_t root{};
    ASSERT_TRUE(loomc_request_root_at(request, 0, &root));
    // Only the selected high-priority family's marker can be published.
    auto bytes = loomc_source_contents(loomc_request_source(request));
    EXPECT_GT(bytes.data_length, 0u);
    loomc_module_t *selected = nullptr;
    loomc_result_t *parsed = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, loomc_request_source(request), nullptr,
        loomc_allocator_system(), &selected, &parsed)));
    EXPECT_TRUE(parsed && loomc_result_succeeded(parsed))
        << Diagnostics(parsed);
    if (selected) {
      std::string text;
      EXPECT_TRUE(Ok(Dump(selected, &text)));
      EXPECT_NE(text.find("registry_override_marker"), std::string::npos);
      EXPECT_EQ(text.find("ggml_linear_q4k"), std::string::npos);
      loomc_module_release(selected);
    }
    if (parsed)
      loomc_result_release(parsed);
    loomc_request_release(request);
  }
  if (result)
    loomc_result_release(result);
  if (product)
    loomc_product_release(product);
  loomc_module_release(module);
}
TEST_F(RegistryTest, RejectsUnsupportedFlagsAtNativeFamilySelection) {
  auto registry = Registry::Llama3B();
  auto entry = *registry.Find(Id::GateUpDecode);
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/unsupported_ffn_call.loom");
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/ffn_config.loom");
  auto *module = LoadEntry(entry);
  ASSERT_NE(module, nullptr);
  auto selected = loomc_make_cstring_view("registry_gate_up_decode");
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  options.root_symbols = &selected;
  options.root_symbol_count = 1;
  loomc_product_t *product = nullptr;
  loomc_result_t *result = nullptr;
  auto status = loomc_cmd_program_product_build_module(
      workspace, module, &options, loomc_allocator_system(), &product, &result);
  EXPECT_TRUE(loomc_status_is_ok(status));
  EXPECT_TRUE(result && !loomc_result_succeeded(result)) << Diagnostics(result);
  EXPECT_EQ(product, nullptr);
  loomc_status_free(status);
  if (result)
    loomc_result_release(result);
  if (product)
    loomc_product_release(product);
  loomc_module_release(module);
}
TEST_F(RegistryTest, RejectsMissingFilesSymbolsAndWrongKind) {
  auto registry = Registry::Llama3B();
  auto base = *registry.Find(Id::RmsNorm);
  for (int variant = 0; variant < 3; ++variant) {
    auto entry = base;
    if (variant == 0)
      entry.provider_files.push_back("missing_provider.loom");
    if (variant == 1)
      entry.symbol = "missing_symbol";
    if (variant == 2)
      entry.kind = Kind::CommandFamily;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status = Load(context, workspace, entry, root, &module, &result);
    EXPECT_TRUE(!loomc_status_is_ok(status) ||
                (result && !loomc_result_succeeded(result)));
    loomc_status_free(status);
    EXPECT_EQ(module, nullptr);
    if (result)
      loomc_result_release(result);
  }
}
TEST_F(RegistryTest,
       RejectsIncompatibleInterfaceMissingConfigAndTransitiveMotif) {
  auto registry = Registry::Llama3B();
  for (int variant = 0; variant < 4; ++variant) {
    auto entry = *registry.Find(variant == 3 ? Id::GateUpPrefill : Id::RmsNorm);
    if (variant == 0)
      entry.interface_file =
          "lib/dispatch/testing/testdata/bad_norm_interface.loom";
    if (variant == 1)
      entry.provider_files.pop_back();
    if (variant == 2)
      entry.provider_files.erase(entry.provider_files.begin() + 1);
    if (variant == 3)
      entry.interface_file =
          "lib/dispatch/testing/testdata/bad_ffn_interface.loom";
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status = Load(context, workspace, entry, root, &module, &result);
    EXPECT_TRUE(!loomc_status_is_ok(status) ||
                (result && !loomc_result_succeeded(result)))
        << variant;
    EXPECT_EQ(module, nullptr);
    loomc_status_free(status);
    if (module)
      loomc_module_release(module);
    if (result)
      loomc_result_release(result);
  }
}
TEST(RegistryContracts, DuplicateAndProfileAdmission) {
  auto registry = Registry::Llama3B();
  std::string error;
  EXPECT_FALSE(registry.Add(*registry.Find(Id::RmsNorm), &error));
  const auto &gate = *registry.Find(Id::GateUpPrefill);
  operations::OperationRegistry ops;
  operations::OperationGraph graph(ops);
  auto test = [&](uint64_t hidden, operations::Encoding encoding) {
    auto x = graph.tensor(
        operations::TensorType::matrix(1, hidden, operations::ElementType::F32),
        &error);
    auto wt = operations::TensorType::matrix(hidden, 8192,
                                             operations::ElementType::F32);
    wt.setLayout(operations::Layout::GgmlWeightRows).setEncoding(encoding);
    auto w = graph.tensor(wt, &error);
    std::vector<shapes::Operand> inputs{{operations::Port::Input, x},
                                        {operations::Port::GateWeight, w},
                                        {operations::Port::UpWeight, w}};
    shapes::Result result{
        operations::TensorType::matrix(1, 8192, operations::ElementType::F32)};
    operations::OperationAttrs attrs =
        operations::GatedMatmulAttrs{hidden, 8192};
    return gate.builder.supports(
        gate, InvocationRequest{graph, attrs, inputs, result}, &error);
  };
  EXPECT_TRUE(test(3072, operations::Encoding::Q4_K));
  EXPECT_FALSE(test(3072, operations::Encoding::Q6_K));
  EXPECT_FALSE(test(4096, operations::Encoding::Q4_K));
}

TEST_F(RegistryTest, NativeCommandProgramUsesItsActualDeclarationAndCallAbi) {
  NativeSource source{"test.registered_program", Kind::CommandProgram,
      "lib/dispatch/testing/testdata/program_interface.loom",
      {"lib/dispatch/testing/testdata/program_provider.loom"}, {},
      "test.registered_program", {"test.registered_program"}};
  loomc_module_t *loaded = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(LoadNative(context, workspace, {source}, root, &loaded, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::unique_ptr<loomify_gguf::ModuleBuilder> module;
  ASSERT_TRUE(Ok(loomify_gguf::ModuleBuilder::Create(context, workspace, loaded, &module)));
  loomc_module_release(loaded);
  loomify_gguf::ProgramRef callee;
  ASSERT_TRUE(Ok(module->Program(source.symbol, &callee)));
  ASSERT_TRUE(Ok(module->CheckProgramInterface(callee,
      {module->Scalar(LOOM_SCALAR_TYPE_INDEX)}, {module->Buffer(), module->Buffer()})));
  std::unique_ptr<loomify_gguf::CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("caller", {LOOM_SCALAR_TYPE_INDEX}, 2, &program)));
  ASSERT_TRUE(Ok(program->Call(callee, {program->Specialization(0)},
                             {program->Binding(0), program->Binding(1)})));
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_module_release(linked);
  loomc_result_release(result);
  for (int invalid = 0; invalid < 4; ++invalid) {
    auto bad = source;
    if (invalid == 0) bad.kind = Kind::CommandFamily;
    if (invalid == 1) bad.kind = static_cast<Kind>(99);
    if (invalid == 2) bad.provider_files = {bad.interface_file};
    if (invalid == 3) bad.provider_files = {"lib/dispatch/testing/testdata/program_bad_provider.loom"};
    loaded = nullptr; result = nullptr;
    auto status = LoadNative(context, workspace, {bad}, root, &loaded, &result);
    EXPECT_TRUE(!loomc_status_is_ok(status) || !result || !loomc_result_succeeded(result));
    EXPECT_EQ(loaded, nullptr);
    loomc_status_free(status);
    if (result) loomc_result_release(result);
  }
}
} // namespace
} // namespace hrx::dispatch
