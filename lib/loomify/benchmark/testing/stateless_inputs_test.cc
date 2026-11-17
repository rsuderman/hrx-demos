// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
class StatelessInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *directory = std::getenv("TEST_TMPDIR");
    ASSERT_NE(directory, nullptr);
    path = std::filesystem::path(directory) /
           ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directories(path));
    plan.source_asset = (path / "weights.gguf").string();
    plan.model.embedding_width = 4;
    plan.workload.seed = 7;
    model::TensorBinding gamma;
    gamma.tensor.name = "blk.0.attn_norm.weight";
    gamma.tensor.storage_type = 0;
    gamma.tensor.byte_size = 16;
    gamma.tensor.absolute_offset = 8;
    plan.model.globals.push_back(gamma);
    Program program;
    program.parameters.push_back({gamma.tensor.name, 3, 1000, 16, 4});
    plan.programs.push_back(program);
    invocation.kernel_symbol = "qwen38_rmsnorm_f32_prefill";
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = 5;
    invocation.command.payload.dispatch.workgroup_count[0] = 1;
    invocation.command.payload.dispatch.workgroup_count[1] = 3;
    invocation.command.payload.dispatch.workgroup_count[2] = 1;
    invocation.arguments.resize(5);
    invocation.arguments[0].kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
    invocation.arguments[0].value.scalar_bits = 8;
    invocation.arguments[1].kind = LOOMC_CMD_ARGUMENT_KIND_B32;
    const float epsilon = 0.00001f;
    uint32_t bits;
    std::memcpy(&bits, &epsilon, sizeof(bits));
    invocation.arguments[1].value.scalar_bits = bits;
    for (unsigned i = 2; i < 5; ++i)
      invocation.arguments[i].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[2].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            64, 128};
    invocation.arguments[3].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 3,
                                            1000, 16};
    invocation.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            256, 128};
    WriteGamma({0x3F800000, 0x40000000, 0x3F000000, 0xBF800000});
  }
  void WriteGamma(const std::vector<uint32_t> &values) {
    std::ofstream file(plan.source_asset, std::ios::binary | std::ios::trunc);
    file.write("metadata", 8);
    for (uint32_t value : values)
      for (unsigned lane = 0; lane < 4; ++lane)
        file.put(static_cast<char>(value >> (lane * 8)));
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  std::filesystem::path path;
  Plan plan;
  Invocation invocation;
};

TEST_F(StatelessInputsTest,
       UsesActualParameterPlacementAndPreservesRootAliases) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareStatelessCase(plan, invocation, 512, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.setup_recipe_id, "rms_f32_prefill/v1");
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3}));
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 8);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 128);
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "blk.0.attn_norm.weight");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 8);
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 16);
  ASSERT_EQ(prepared.roots.size(), 2);
  const auto *activations =
      FindInputRoot(prepared.roots, invocation.arguments[2].value.buffer);
  ASSERT_NE(activations, nullptr);
  EXPECT_EQ(activations->original_byte_offset, 64);
  EXPECT_EQ(activations->bytes.size(), 320);
  EXPECT_EQ(activations, FindInputRoot(prepared.roots,
                                       invocation.arguments[4].value.buffer));
  const auto *gamma =
      FindInputRoot(prepared.roots, invocation.arguments[3].value.buffer);
  ASSERT_NE(gamma, nullptr);
  EXPECT_EQ(gamma->original_byte_offset, 1000);
  EXPECT_EQ(gamma->bytes,
            (std::vector<uint8_t>{0, 0, 0x80, 0x3F, 0, 0, 0, 0x40, 0, 0, 0,
                                  0x3F, 0, 0, 0x80, 0xBF}));
}

TEST_F(StatelessInputsTest, UnsupportedOrMismatchedRecipesFailExplicitly) {
  PreparedCase prepared;
  std::string error;
  auto changed = invocation;
  changed.kernel_symbol = "unknown.operation";
  EXPECT_EQ(PrepareStatelessCase(plan, changed, 512, &prepared, &error),
            PreparationStatus::kUnsupported);
  changed = invocation;
  changed.arguments[1].kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
  EXPECT_EQ(PrepareStatelessCase(plan, changed, 512, &prepared, &error),
            PreparationStatus::kFailure);
  changed = invocation;
  changed.arguments[4].value.buffer = changed.arguments[2].value.buffer;
  EXPECT_EQ(PrepareStatelessCase(plan, changed, 512, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("disjoint"), std::string::npos);
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 335, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
}

TEST_F(StatelessInputsTest, NonfiniteOrMissingGammaIsNotDeclaredSafe) {
  PreparedCase prepared;
  std::string error;
  WriteGamma({0x7F800000, 0, 0, 0});
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 512, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
  plan.programs[0].parameters[0].key = "other.weight";
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 512, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("exact GGUF F32"), std::string::npos);
}

TEST_F(StatelessInputsTest, LogicalGridRowsDoNotReplaceRuntimeCapacity) {
  PreparedCase prepared;
  std::string error;
  invocation.arguments[0].value.scalar_bits = 512;
  invocation.command.payload.dispatch.workgroup_count[1] = 1;
  invocation.arguments[2].value.buffer.byte_length = 8192;
  invocation.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 1,
                                          0, 8192};
  ASSERT_EQ(PrepareStatelessCase(plan, invocation, 20000, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{1}));
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 8192);
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 512);
  invocation.command.payload.dispatch.workgroup_count[1] = 513;
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 20000, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("grid/capacity"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
