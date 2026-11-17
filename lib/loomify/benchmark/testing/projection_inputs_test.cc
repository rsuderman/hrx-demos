// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
uint16_t HalfBits(const std::vector<uint8_t> &bytes, size_t offset) {
  return bytes[offset] | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}
float DecodeHalf(uint16_t bits) {
  const unsigned exponent = (bits >> 10) & 31;
  const unsigned fraction = bits & 1023;
  const float magnitude =
      exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
               : std::ldexp(float(fraction), -24);
  return bits & 0x8000 ? -magnitude : magnitude;
}
class ProjectionInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary, nullptr);
    directory = std::filesystem::path(temporary) /
                ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    plan.source_asset = (directory / "weights.gguf").string();
    plan.workload.seed = 5;
    plan.model.family = "unrelated-model-name";
    model::TensorBinding binding;
    binding.tensor.name = "selected.matrix.weight";
    binding.tensor.storage_type = 12;
    binding.tensor.rank = 2;
    binding.tensor.normalized_extents = {256, 3, 1, 1};
    binding.tensor.byte_size = 432;
    binding.tensor.absolute_offset = 8;
    model::LayerDescriptor layer;
    layer.tensors.push_back(binding);
    plan.model.layers.push_back(layer);
    Program program;
    program.parameters.push_back({binding.tensor.name, 3, 1000, 432, 4});
    plan.programs.push_back(program);
    invocation.kernel_symbol = "ggml_linear_q4k_q8_1_x4";
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = 6;
    invocation.command.payload.dispatch.workgroup_count[0] = 1;
    invocation.command.payload.dispatch.workgroup_count[1] = 1;
    invocation.command.payload.dispatch.workgroup_count[2] = 1;
    invocation.arguments.resize(6);
    for (size_t i = 0; i < 6; ++i)
      invocation.arguments[i].kind = i < 3 ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                                           : LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[0].value.scalar_bits = 4;
    invocation.arguments[1].value.scalar_bits = 256;
    invocation.arguments[2].value.scalar_bits = 3;
    invocation.arguments[3].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            64, 1152};
    invocation.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 3,
                                            1000, 432};
    invocation.arguments[5].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            1408, 48};
    weights.assign(432, 0x33);
    for (size_t block = 0; block < weights.size(); block += 144) {
      weights[block] = 0;
      weights[block + 1] = 0x3C; // F16 d=1.
      weights[block + 2] = 0;
      weights[block + 3] = 0x38; // F16 dmin=.5.
    }
    WriteWeights();
  }
  void WriteWeights() {
    std::ofstream file(plan.source_asset, std::ios::binary | std::ios::trunc);
    file.write("metadata", 8);
    file.write(reinterpret_cast<const char *>(weights.data()), weights.size());
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
  }
  std::filesystem::path directory;
  Plan plan;
  Invocation invocation;
  std::vector<uint8_t> weights;
};

TEST_F(ProjectionInputsTest, UsesExactTensorAndPreservesAliasesAndCapacity) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{1, 256, 3}));
  EXPECT_EQ(prepared.source_argument_types,
            (std::vector<std::string>{"index", "index", "index", "buffer",
                                      "buffer", "buffer"}));
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 4);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 1152);
  EXPECT_EQ(prepared.buffer_extents[1].byte_length, 432);
  EXPECT_EQ(prepared.buffer_extents[2].byte_length, 48);
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "selected.matrix.weight");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 8);
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 432);
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  ASSERT_EQ(prepared.representative_weight_spans.size(), 1);
  EXPECT_EQ(prepared.representative_weight_spans[0].binding.byte_offset, 1000);
  const auto *packed =
      FindInputRoot(prepared.roots, invocation.arguments[3].value.buffer);
  EXPECT_EQ(packed, FindInputRoot(prepared.roots,
                                  invocation.arguments[5].value.buffer));
  ASSERT_NE(packed, nullptr);
  EXPECT_EQ(packed->original_byte_offset, 64);
  EXPECT_EQ(packed->bytes.size(), 1392);
  const auto *weight =
      FindInputRoot(prepared.roots, invocation.arguments[4].value.buffer);
  ASSERT_NE(weight, nullptr);
  EXPECT_EQ(weight->bytes, weights);
}

TEST_F(ProjectionInputsTest, PackedBlocksHaveExactScaleSumAndSignedPayload) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  const auto &bytes =
      FindInputRoot(prepared.roots, invocation.arguments[3].value.buffer)
          ->bytes;
  bool distinct = false;
  for (size_t group = 0; group < 8; ++group)
    for (size_t block = 0; block < 4; ++block) {
      const size_t base = group * 144;
      EXPECT_EQ(HalfBits(bytes, base + block * 4), 0x2000);
      int sum = 0;
      for (size_t element = 0; element < 32; ++element) {
        const auto byte = bytes[base + 16 + block * 32 + element];
        const int value = byte < 128 ? byte : int(byte) - 256;
        sum += value;
        if (element == 0)
          EXPECT_EQ(value, 127);
        else if (element == 1)
          EXPECT_EQ(value, -127);
        else
          EXPECT_LE(std::abs(value), 3);
      }
      EXPECT_FLOAT_EQ(DecodeHalf(HalfBits(bytes, base + block * 4 + 2)),
                      static_cast<float>(sum) / 128.0f);
      distinct |= bytes[base + 18] != bytes[18];
    }
  EXPECT_TRUE(distinct);
}

TEST_F(ProjectionInputsTest, SeedChangesValidFixtureButNotNativeArguments) {
  PreparedCase a, b;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &a, &error),
            PreparationStatus::kReady)
      << error;
  ++plan.workload.seed;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &b, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_NE(
      FindInputRoot(a.roots, invocation.arguments[3].value.buffer)->bytes,
      FindInputRoot(b.roots, invocation.arguments[3].value.buffer)->bytes);
  EXPECT_EQ(a.source_workloads, b.source_workloads);
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 4);
}

TEST_F(ProjectionInputsTest, FullRuntimeCapacityRemainsSeparateFromGridRows) {
  invocation.arguments[0].value.scalar_bits = 512;
  invocation.arguments[3].value.buffer.byte_length = 147456;
  invocation.arguments[5].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 1,
                                          0, 6144};
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 200000, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads.front(), 1);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 147456);
  EXPECT_EQ(prepared.buffer_extents[2].byte_length, 6144);
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 512);
}

TEST_F(ProjectionInputsTest, RejectsMalformedOrOverlappingBindingsAtomically) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "untouched";
  std::string error;
  auto changed = invocation;
  changed.kernel_symbol += "_compat";
  error = "not claimed";
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 4096, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "not claimed");
  changed = invocation;
  changed.arguments[2].kind = LOOMC_CMD_ARGUMENT_KIND_B64;
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  changed = invocation;
  changed.arguments[5].value.buffer = changed.arguments[3].value.buffer;
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("disjoint"), std::string::npos);
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 1800, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
}

TEST_F(ProjectionInputsTest, BudgetExclusionPrecedesIoButNotRangeValidation) {
  ASSERT_TRUE(std::filesystem::remove(plan.source_asset));
  PreparedCase prepared;
  prepared.setup_recipe_id = "untouched";
  std::string error;
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 1800, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  auto changed = invocation;
  changed.arguments[3].value.buffer.byte_length = 1;
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 1, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
}

TEST_F(ProjectionInputsTest, TensorEncodingShapeRangeAndScaleMustBeExact) {
  PreparedCase prepared;
  std::string error;
  auto changed = plan;
  changed.model.layers[0].tensors[0].tensor.storage_type = 14;
  EXPECT_EQ(PrepareProjectionCase(changed, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  changed = plan;
  changed.model.layers[0].tensors[0].tensor.normalized_extents[1] = 4;
  EXPECT_EQ(PrepareProjectionCase(changed, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  changed = plan;
  changed.programs[0].parameters[0].byte_length += 144;
  EXPECT_EQ(PrepareProjectionCase(changed, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  weights[1] = 0x7C;
  WriteWeights();
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
}

TEST_F(ProjectionInputsTest, TruncatedWeightFileDoesNotProducePartialCase) {
  weights.resize(100);
  WriteWeights();
  PreparedCase prepared;
  prepared.setup_recipe_id = "untouched";
  std::string error;
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
