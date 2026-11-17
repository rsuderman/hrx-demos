// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/gguf_model_builder.h"
#include "lib/loomify/publication.h"
#include "loomc/target/amdgpu.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

namespace hrx::loomify_gguf {
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
TEST(NumericalBuilder, BuildsSeparateDecodeAndPrefillWithoutPackageCode) {
  const char *srcdir = std::getenv("TEST_SRCDIR");
  const char *workspace_name = std::getenv("TEST_WORKSPACE");
  ASSERT_TRUE(srcdir && workspace_name);
  const auto library_root =
      (std::filesystem::path(srcdir) / workspace_name).string();
  loomc_target_environment_t *raw_environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &raw_environment)));
  std::unique_ptr<loomc_target_environment_t,
                  decltype(&loomc_target_environment_release)>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t context_options{};
  context_options.next = &target;
  loomc_context_t *raw_context = nullptr;
  ASSERT_TRUE(Ok(loomc_context_create(&context_options,
                                      loomc_allocator_system(), &raw_context)));
  std::unique_ptr<loomc_context_t, decltype(&loomc_context_release)> context(
      raw_context, loomc_context_release);
  loomc_workspace_t *raw_workspace = nullptr;
  ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                        &raw_workspace)));
  std::unique_ptr<loomc_workspace_t, decltype(&loomc_workspace_release)>
      workspace(raw_workspace, loomc_workspace_release);
  for (const char *variable : {"LOOMIFY_TEST_GGUF", "HRX_QWEN3_GGUF"}) {
    const char *asset = std::getenv(variable);
    ASSERT_TRUE(asset && *asset)
        << "explicit immutable asset required: " << variable;
    std::string error;
    std::unique_ptr<GgufModelBuilder> builder;
    ASSERT_TRUE(GgufModelBuilder::Open(asset, &builder, &error)) << error;
    const auto family = builder->descriptor().family;
    for (uint64_t tokens : {1u, 8u}) {
      GgufBuildOptions options;
      options.library_root = library_root;
      options.prefill_capacity = tokens;
      std::unique_ptr<GgufConfiguration> configuration;
      ASSERT_TRUE(builder->ConfigureNumerical(options, &configuration, &error))
          << error;
      EXPECT_FALSE(configuration->has_package_configuration());
      EXPECT_NE(configuration->configuration_source().find(
                    "@runner." +
                    (family == "llama" ? std::string("llama32") : family) +
                    ".prefill_token_count = " + std::to_string(tokens) +
                    " : index"),
                std::string::npos);
      std::unique_ptr<BuiltGgufModule> module;
      ASSERT_TRUE(configuration->BuildModule(context.get(), workspace.get(),
                                             &module, &error))
          << error;
      EXPECT_NE(module->module(), nullptr);
      EXPECT_EQ(module->configuration_source(),
                configuration->configuration_source());
      const auto *saved = configuration.get();
      options.platform = "unsupported";
      EXPECT_FALSE(
          builder->ConfigureNumerical(options, &configuration, &error));
      EXPECT_EQ(configuration.get(), saved);
      EXPECT_NE(error.find("unsupported generation platform"),
                std::string::npos);
    }
  }
}
TEST(NumericalBuilder, KeepsRequiredSourcePublicationValidation) {
  std::optional<PublicationTokenFacts> publication;
  std::string error;
  EXPECT_FALSE(
      DerivePublicationTokenFacts(nullptr, "llama", {}, &publication, &error));
  EXPECT_EQ(error, "publication requires GGUF vocabulary");
  EXPECT_TRUE(
      DerivePublicationTokenFacts(nullptr, "qwen35", {}, &publication, &error));
  EXPECT_FALSE(publication);
  EXPECT_TRUE(error.empty());
}
} // namespace
} // namespace hrx::loomify_gguf
