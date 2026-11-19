// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/testing/qwen3moe_fixture.h"
#include "gtest/gtest.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

namespace hrx::loomify_gguf {
namespace {
bool Ok(loomc_status_t status) {
  const bool success = loomc_status_is_ok(status);
  if (!success) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return success;
}

TEST(Qwen3MoeMaterialization, PublicEpochConsumesExpertsAndActualOutputPolicy) {
  const char *runfiles = std::getenv("TEST_SRCDIR");
  const char *workspace_name = std::getenv("TEST_WORKSPACE");
  ASSERT_TRUE(runfiles && workspace_name);
  const auto root = std::filesystem::path(runfiles) / workspace_name;
  loomc_target_environment_t *raw_environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &raw_environment)));
  std::unique_ptr<loomc_target_environment_t,
                  decltype(&loomc_target_environment_release)>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *raw_context = nullptr;
  ASSERT_TRUE(Ok(
      loomc_context_create(&options, loomc_allocator_system(), &raw_context)));
  std::unique_ptr<loomc_context_t, decltype(&loomc_context_release)> context(
      raw_context, loomc_context_release);
  loomc_workspace_t *raw_workspace = nullptr;
  ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                        &raw_workspace)));
  std::unique_ptr<loomc_workspace_t, decltype(&loomc_workspace_release)>
      workspace(raw_workspace, loomc_workspace_release);
  for (bool q8 : {false, true}) {
    SCOPED_TRACE(q8);
    model::ModelDescriptor descriptor;
    std::string error;
    ASSERT_TRUE(model::LoadModelDescriptor(testing::qwen3moe::Catalog(2, q8),
                                           &descriptor, &error))
        << error;
    ASSERT_EQ(descriptor.vocabulary_size, 151936u);
    for (uint64_t rows : {1u, 8u, 17u}) {
      SCOPED_TRACE(rows);
      ConfigurationFacts facts;
      ASSERT_TRUE(DeriveConfiguration(descriptor, "materialization-fixture",
                                      {rows, 128}, &facts, &error))
          << error;
      const auto configuration =
          facts.source +
          "config.def @qwen3moe.publication.end_text_id = 2 : i32\n"
          "config.def @qwen3moe.publication.end_turn_id = 4 : i32\n";
      for (bool bytecode : {false, true}) {
        SCOPED_TRACE(bytecode);
        const auto product =
            root / "lib/architectures/qwen3moe" /
            (bytecode ? "implementation.loombc" : "implementation_text.loom");
        const auto projection =
            root / "lib/platforms/amdgpu/gfx11/projection" /
            (bytecode ? "implementation.loombc" : "implementation_text.loom");
        const auto targets = root / "lib/platforms/amdgpu/gfx11" /
                             (bytecode ? "targets.loombc" : "targets.loom");
        loomc_module_t *raw_module = nullptr;
        ASSERT_TRUE(LinkConfiguredProducts(
            context.get(), workspace.get(), configuration,
            {product.string(), targets.string(), projection.string()},
            {"generated.qwen3moe_epoch"}, true, &raw_module, &error))
            << error;
        std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module(
            raw_module, loomc_module_release);
        loomc_source_t *raw_text = nullptr;
        ASSERT_TRUE(Ok(loomc_module_serialize_text_to_source(
            module.get(), nullptr, loomc_allocator_system(), &raw_text)));
        std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)>
            printed(raw_text, loomc_source_release);
        const auto bytes = loomc_source_contents(printed.get());
        const std::string text(reinterpret_cast<const char *>(bytes.data),
                               bytes.data_length);
        EXPECT_NE(text.find("@generated.qwen3moe_epoch"), std::string::npos);
        EXPECT_EQ(text.find("@generated.qwen3_epoch"), std::string::npos);
        // These names are existing routed-operation launch contracts. Strict
        // normalization must consume the selected provider to reach them.
        for (const char *kernel :
             {"llm.routing.softmax_top8", "llm.experts.pack_split_q8",
              "llm.experts.indexed_swiglu_q4", "llm.experts.indexed_down_q4",
              "llm.routing.weighted_top8"})
          EXPECT_NE(text.find(std::string("kernel.launch @") + kernel + "["),
                    std::string::npos)
              << kernel;
        if (!q8)
          EXPECT_NE(text.find("kernel.launch @llm.experts.indexed_down_q6["),
                    std::string::npos);
        EXPECT_EQ(text.find(".ffn_gate.weight\""), std::string::npos);
        EXPECT_EQ(text.find(".ffn_up.weight\""), std::string::npos);
        EXPECT_EQ(text.find(".ffn_down.weight\""), std::string::npos);
        EXPECT_NE(text.find(".ffn_gate_exps.weight\""), std::string::npos);
        EXPECT_NE(text.find(".ffn_up_exps.weight\""), std::string::npos);
        EXPECT_NE(text.find(".ffn_down_exps.weight\""), std::string::npos);
        // Q6 consumes policy0's compatibility kernel; Q8 consumes policy1's
        // canonical kernel. Follow the final head weight operand as well, so
        // a tag-only change or embedding-as-head shortcut cannot pass.
        const auto launch =
            text.rfind(q8 ? "kernel.launch @ggml_linear_q8_0_q8_1_x4["
                          : "kernel.launch @ggml_linear_q6k_q8_1_x4_compat[");
        ASSERT_NE(launch, std::string::npos);
        const auto args = text.find("](", launch);
        ASSERT_NE(args, std::string::npos);
        const auto end = text.find(')', args);
        ASSERT_NE(end, std::string::npos);
        std::istringstream arguments(text.substr(args + 2, end - args - 2));
        std::string weight;
        for (int i = 0; i <= (q8 ? 4 : 8); ++i)
          ASSERT_TRUE(bool(std::getline(arguments, weight, ',')));
        weight.erase(0, weight.find_first_not_of(" \t"));
        weight.erase(weight.find_last_not_of(" \t") + 1);
        const auto definition =
            text.rfind(weight + " = command.parameter ", launch);
        ASSERT_NE(definition, std::string::npos) << weight;
        const auto parameter =
            text.substr(definition, text.find('\n', definition) - definition);
        EXPECT_NE(parameter.find(", \"output.weight\""), std::string::npos)
            << parameter;
      }
    }
  }
}
} // namespace
} // namespace hrx::loomify_gguf
