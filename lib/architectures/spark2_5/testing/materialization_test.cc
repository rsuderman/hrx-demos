// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/testing/spark2_5_fixture.h"
#include "tools/model/model_descriptor.h"
#include "gtest/gtest.h"
#include "loomc/target/amdgpu.h"
#include <cstdlib>
#include <filesystem>
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
TEST(Spark25Materialization,
     LocalGlobalBlocksAndTiedUntiedHeadsUseGeneratedShapesAndTags) {
  const char *runfiles = std::getenv("TEST_SRCDIR");
  const char *workspace_name = std::getenv("TEST_WORKSPACE");
  const char *temporary = std::getenv("TEST_TMPDIR");
  ASSERT_TRUE(runfiles && workspace_name && temporary);
  const auto root = std::filesystem::path(runfiles) / workspace_name;
  loomc_target_environment_t *environment = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &environment)));
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *context = nullptr;
  ASSERT_TRUE(
      Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
  loomc_target_environment_release(environment);
  loomc_workspace_t *workspace = nullptr;
  ASSERT_TRUE(Ok(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  for (uint64_t hidden : {1024, 2560}) {
    for (uint64_t rows : {8, 17}) {
      for (uint32_t gate_tag : {0u, 6u, 8u, 12u, 13u, 14u}) {
        auto catalog =
            testing::Spark25Catalog(hidden, 3, 16, 4, 10240, hidden == 2560);
        model::ModelDescriptor descriptor;
        std::string error;
        ASSERT_TRUE(model::LoadModelDescriptor(catalog, &descriptor, &error))
            << error;
        // Two distinct local profiles and an independent full-history profile.
        descriptor.layers[0].rope_base = 10000;
        descriptor.layers[0].rope_width = 256;
        descriptor.layers[1].rope_base = 100000;
        descriptor.layers[1].rope_width = 128;
        descriptor.layers[2].attention_window = 0;
        descriptor.layers[2].rope_base = 2000000;
        descriptor.layers[2].rope_width = 64;
        if (gate_tag != 0) {
          for (auto &binding : descriptor.layers.at(1).tensors) {
            if (binding.role != "attention_gate")
              continue;
            binding.tensor.storage_type = gate_tag;
            ASSERT_TRUE(model::EncodedTensorByteSize(
                gate_tag, hidden, descriptor.layers.at(1).query_heads,
                &binding.tensor.byte_size));
          }
        }
        ConfigurationFacts configuration;
        ASSERT_TRUE(DeriveConfiguration(descriptor, "fixture.gguf",
                                        {rows, 1024}, &configuration, &error))
            << error;
        loomc_module_t *module = nullptr;
        ASSERT_TRUE(LinkConfiguredProducts(
            context, workspace,
            configuration.source +
                "config.def @spark2_5.publication.eos_id = 1337 : i32\n",
            {(root / "lib/architectures/spark2_5/layer_fixture.loombc")
                 .string(),
             (root / "lib/platforms/amdgpu/gfx11/targets.loombc").string()},
            {"validated.spark2_5.layer0", "validated.spark2_5.layer1",
             "generated.spark2_5_epoch"},
            true, &module, &error))
            << "hidden=" << hidden << ", rows=" << rows << ": " << error;
        ASSERT_NE(module, nullptr);
        for (auto format :
             {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
          loomc_module_serialize_options_t serialization{};
          serialization.format = format;
          const auto path =
              std::filesystem::path(temporary) /
              ("spark2_5-layer-" + std::to_string(hidden) + "-" +
               std::to_string(rows) + "-q" + std::to_string(gate_tag) +
               (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc"));
          EXPECT_TRUE(Ok(loomc_module_serialize_to_path(
              module, &serialization, loomc_make_cstring_view(path.c_str()),
              loomc_allocator_system())));
          EXPECT_GT(std::filesystem::file_size(path), 0u);
        }
        loomc_module_release(module);
      }
    }
  }
  loomc_workspace_release(workspace);
  loomc_context_release(context);
}
} // namespace
} // namespace hrx::loomify_gguf
