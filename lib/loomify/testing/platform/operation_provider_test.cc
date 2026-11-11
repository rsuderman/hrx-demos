// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/platforms.h"
#include "loomc/context.h"
#include <iostream>
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
namespace {
using namespace hrx::loomify_gguf;
template <class T, void (*Release)(T*)> using Owner = std::unique_ptr<T, decltype(Release)>;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) { auto message = loomc_status_message(status); ADD_FAILURE() << std::string(message.data, message.size); }
  loomc_status_free(status); return ok;
}
std::string Root() {
  return std::string(std::getenv("TEST_SRCDIR")) + "/" + std::getenv("TEST_WORKSPACE") + "/lib/loomify/testing/platform";
}
class OperationProviderTest : public ::testing::TestWithParam<bool> {
 protected:
  Owner<loomc_target_environment_t, loomc_target_environment_release> environment{nullptr, loomc_target_environment_release};
  Owner<loomc_context_t, loomc_context_release> context{nullptr, loomc_context_release};
  Owner<loomc_workspace_t, loomc_workspace_release> workspace{nullptr, loomc_workspace_release};
  void SetUp() override {
    loomc_target_environment_t* e = nullptr; ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(), &e))); environment.reset(e);
    loomc_context_target_options_t target{}; target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS; target.structure_size = sizeof(target); target.target_environment = e;
    loomc_context_options_t options{}; options.next = &target;
    loomc_context_t* c = nullptr; ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(), &c))); context.reset(c);
    loomc_workspace_t* w = nullptr; ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w))); workspace.reset(w);
  }
  std::string File(const char* member) { return std::string(member) + (GetParam() ? "_bc.loombc" : "_text.loom"); }
  ArchitectureFiles Architecture() {
    return {"probe", nullptr, "", "", "", "", {"selected_probe"}, {RequiredOperation::kScaledF16ProjectionV1}};
  }
  PlatformFiles Platform() { return {"test-only", "amdgpu", nullptr, {}}; }
};
TEST_P(OperationProviderTest, SelectsSeparateConcreteKernelThroughProductionComposition) {
  for (const char* name : {"a", "b"}) {
    const auto root = File("operation_root"), targets = File("targets"), body = File((std::string("operation_")+name).c_str());
    auto architecture = Architecture(); architecture.implementation = root.c_str();
    auto platform = Platform(); platform.targets = targets.c_str(); platform.implementations = {{RequiredOperation::kScaledF16ProjectionV1, body.c_str()}};
    loomc_module_t* raw = nullptr; std::string error;
    ASSERT_TRUE(LinkNumericalProducts(context.get(), workspace.get(), Root(), "", architecture, platform, &raw, &error)) << error;
    Owner<loomc_module_t, loomc_module_release> module(raw, loomc_module_release);
    for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options{}; options.format = format;
      loomc_source_t* source = nullptr; ASSERT_TRUE(Ok(loomc_module_serialize_to_source(module.get(), &options, loomc_allocator_system(), &source)));
      Owner<loomc_source_t, loomc_source_release> serialized(source, loomc_source_release);
      if (format == LOOMC_SOURCE_FORMAT_TEXT) {
        auto bytes = loomc_source_contents(source); const std::string text(reinterpret_cast<const char*>(bytes.data), bytes.data_length);
        EXPECT_NE(text.find(std::string("scalar.constant ") + (std::string(name)=="a" ? "17" : "29") + " : i32"), std::string::npos) << text;
        EXPECT_NE(text.find("kernel.def target(@platform_test_wave32) @selected_operation"), std::string::npos) << text;
      }
      if (const char* output = std::getenv("LOOMIFY_PROVIDER_PROBE_DIR")) {
        std::filesystem::create_directories(output);
        const auto path = (std::filesystem::path(output) / (std::string(name) + (GetParam() ? "_bc_input" : "_text_input") + (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc"))).string();
        const auto bytes = loomc_source_contents(source);
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data), bytes.data_length);
        ASSERT_TRUE(stream.good());
      }
      loomc_module_t* roundtrip = nullptr; loomc_result_t* result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(context.get(), workspace.get(), source, nullptr, loomc_allocator_system(), &roundtrip, &result)));
      Owner<loomc_module_t, loomc_module_release> copy(roundtrip, loomc_module_release);
      Owner<loomc_result_t, loomc_result_release> parsed(result, loomc_result_release);
      ASSERT_TRUE(result && loomc_result_succeeded(result));
    }
  }
}
TEST_P(OperationProviderTest, RejectsMissingDuplicateAndUnavailableSelectedProducts) {
  const auto root = File("operation_root"), targets = File("targets"), body = File("operation_a");
  auto architecture = Architecture(); architecture.implementation = root.c_str();
  auto platform = Platform(); platform.targets = targets.c_str();
  loomc_module_t* raw = nullptr; std::string error;
  EXPECT_FALSE(LinkNumericalProducts(context.get(), workspace.get(), Root(), "", architecture, platform, &raw, &error)); EXPECT_EQ(raw, nullptr); EXPECT_NE(error.find("missing platform operation"), std::string::npos);
  platform.implementations = {{RequiredOperation::kScaledF16ProjectionV1, body.c_str()}, {RequiredOperation::kScaledF16ProjectionV1, body.c_str()}};
  EXPECT_FALSE(LinkNumericalProducts(context.get(), workspace.get(), Root(), "", architecture, platform, &raw, &error)); EXPECT_EQ(raw, nullptr); EXPECT_NE(error.find("duplicate platform operation"), std::string::npos);
  platform.implementations = {{RequiredOperation::kScaledF16ProjectionV1, "missing_selected_operation.loombc"}};
  EXPECT_FALSE(LinkNumericalProducts(context.get(), workspace.get(), Root(), "", architecture, platform, &raw, &error)); EXPECT_EQ(raw, nullptr); EXPECT_NE(error.find("missing_selected_operation"), std::string::npos);
}
// Malformed-provider fixtures characterize native boundaries. Production uses
// only mandatory, trusted BUILD products; it does not certify arbitrary IR.
TEST_P(OperationProviderTest, NativeRejectsWrongWorkloadAbi) {
  const auto root = File("operation_root"), targets = File("targets"), body = File("operation_wrong_abi");
  auto architecture = Architecture(); architecture.implementation = root.c_str();
  auto platform = Platform(); platform.targets = targets.c_str();
  platform.implementations = {{RequiredOperation::kScaledF16ProjectionV1, body.c_str()}};
  loomc_module_t* raw = nullptr; std::string error;
  EXPECT_FALSE(LinkNumericalProducts(context.get(), workspace.get(), Root(), "", architecture, platform, &raw, &error));
  Owner<loomc_module_t, loomc_module_release> module(raw, loomc_module_release);
  EXPECT_EQ(raw, nullptr); EXPECT_NE(error.find("kernel workload args"), std::string::npos) << error;
  std::cout << "NATIVE ABI: " << error << "\n";
}
TEST_P(OperationProviderTest, ConcreteDefinitionsFollowNativeSelectionPrecedence) {
  const auto root = File("operation_conflicting_root"), targets = File("targets"), body = File("operation_b");
  auto architecture = Architecture(); architecture.implementation = root.c_str();
  auto platform = Platform(); platform.targets = targets.c_str();
  platform.implementations = {{RequiredOperation::kScaledF16ProjectionV1, body.c_str()}};
  loomc_module_t* raw = nullptr; std::string error;
  // Native LINK accepts compatible concrete definitions across providers and
  // selects one. Operation metadata must not promise arbitrary IR conflicts fail.
  ASSERT_TRUE(LinkNumericalProducts(context.get(), workspace.get(), Root(), "", architecture, platform, &raw, &error)) << error;
  Owner<loomc_module_t, loomc_module_release> module(raw, loomc_module_release);
  EXPECT_TRUE(error.empty());
  loomc_module_serialize_options_t options{}; options.format = LOOMC_SOURCE_FORMAT_TEXT;
  loomc_source_t* source = nullptr;
  ASSERT_TRUE(Ok(loomc_module_serialize_to_source(module.get(), &options, loomc_allocator_system(), &source)));
  Owner<loomc_source_t, loomc_source_release> serialized(source, loomc_source_release);
  const auto bytes = loomc_source_contents(source);
  const std::string text(reinterpret_cast<const char*>(bytes.data), bytes.data_length);
  const bool a = text.find("scalar.constant 17 : i32") != std::string::npos;
  const bool b = text.find("scalar.constant 29 : i32") != std::string::npos;
  EXPECT_NE(a, b) << text;
  std::cout << "CONCRETE PRECEDENCE SELECTED: " << (a ? 17 : (b ? 29 : -1)) << "\n";
}
TEST_P(OperationProviderTest, DeclarationOnlySelectedKernelRejectsAtExactLinkBoundary) {
  const auto root = File("operation_root"), targets = File("targets"), body = File("operation_empty");
  auto architecture = Architecture(); architecture.implementation = root.c_str();
  auto platform = Platform(); platform.targets = targets.c_str();
  platform.implementations = {{RequiredOperation::kScaledF16ProjectionV1, body.c_str()}};
  loomc_module_t* raw = nullptr; std::string error;
  EXPECT_FALSE(LinkNumericalProducts(context.get(), workspace.get(), Root(), "", architecture, platform, &raw, &error));
  Owner<loomc_module_t, loomc_module_release> module(raw, loomc_module_release);
  EXPECT_EQ(raw, nullptr); EXPECT_NE(error.find("unresolved exact declaration"), std::string::npos) << error;
  std::cout << "DECLARATION ONLY SELECTED KERNEL: " << error << "\n";
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, OperationProviderTest, ::testing::Bool());
}
