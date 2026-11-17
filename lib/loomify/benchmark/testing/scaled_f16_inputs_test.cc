// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
class ScaledF16InputsTest : public ::testing::Test {
protected:
  static constexpr uint64_t kK = 256, kN = 64;
  void SetUp() override {
    const char *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary, nullptr);
    directory = std::filesystem::path(temporary) /
                ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    plan.source_asset = (directory / "weights.bin").string();
    plan.programs.resize(1);
    plan.workload.active_tokens = 65;
    plan.workload.token_capacity = 128;
    plan.workload.seed = 7;
    Weights(false);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
  }
  static loomc_cmd_argument_t Scalar(uint64_t bits, bool offset = false) {
    loomc_cmd_argument_t argument{};
    argument.kind =
        offset ? LOOMC_CMD_ARGUMENT_KIND_OFFSET : LOOMC_CMD_ARGUMENT_KIND_INDEX;
    argument.value.scalar_bits = bits;
    return argument;
  }
  static loomc_cmd_argument_t Buffer(unsigned root, uint64_t offset,
                                     uint64_t length, bool fixed = false) {
    loomc_cmd_argument_t argument{};
    argument.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    argument.value.buffer = {fixed ? LOOMC_CMD_BUFFER_ROLE_FIXED
                                   : LOOMC_CMD_BUFFER_ROLE_REBINDABLE,
                             root, offset, length};
    return argument;
  }
  void Call(std::string name, std::vector<loomc_cmd_argument_t> arguments,
            uint32_t x, uint32_t y = 1) {
    invocation = {};
    invocation.kernel_symbol = std::move(name);
    invocation.arguments = std::move(arguments);
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = invocation.arguments.size();
    invocation.command.payload.dispatch.workgroup_count[0] = x;
    invocation.command.payload.dispatch.workgroup_count[1] = y;
    invocation.command.payload.dispatch.workgroup_count[2] = 1;
  }
  void Weights(bool q6) {
    const uint64_t block = q6 ? 210 : 144;
    weight_bytes = kN * block;
    plan.model.globals.clear();
    plan.programs[0].parameters.clear();
    std::ofstream stream(plan.source_asset, std::ios::binary | std::ios::trunc);
    stream.write("metadata", 8);
    for (unsigned matrix = 0; matrix < 2; ++matrix) {
      std::vector<uint8_t> bytes(weight_bytes, matrix ? 0x22 : 0x11);
      for (uint64_t row = 0; row < kN; ++row) {
        const auto d = row * block + (q6 ? 208 : 0);
        bytes[d] = 0;
        bytes[d + 1] = 0x3c;
        if (!q6) {
          bytes[d + 2] = 0;
          bytes[d + 3] = 0;
        }
      }
      stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      model::TensorBinding binding;
      binding.tensor.name = matrix ? "second.weight" : "first.weight";
      binding.tensor.storage_type = q6 ? 14 : 12;
      binding.tensor.rank = 2;
      binding.tensor.normalized_extents = {kK, kN, 1, 1};
      binding.tensor.byte_size = bytes.size();
      binding.tensor.absolute_offset = 8 + matrix * weight_bytes;
      plan.model.globals.push_back(binding);
      plan.programs[0].parameters.push_back({binding.tensor.name, 7,
                                             64 + matrix * (weight_bytes + 64),
                                             weight_bytes, 4});
    }
  }
  void PrepareCall() {
    Call("llm.projection.prepare_scaled_f16",
         {Scalar(65), Scalar(128), Scalar(kK), Scalar(20, true),
          Buffer(0, 32, 32), Buffer(1, 64, 65 * kK * 4), Buffer(2, 16, 128 * 8),
          Buffer(3, 128, 128 * kK * 2)},
         128);
  }
  void MatrixCall(bool q6 = false, bool fused = false, uint64_t rows = 128) {
    std::vector<loomc_cmd_argument_t> arguments = {
        Scalar(rows),
        Scalar(kK),
        Scalar(kN),
        Buffer(0, 32, 12),
        Buffer(1, 64, rows * kK * 2),
        Buffer(2, 16, rows * 8),
        Buffer(7, 64, weight_bytes, true)};
    if (fused)
      arguments.push_back(Buffer(7, 128 + weight_bytes, weight_bytes, true));
    arguments.push_back(Buffer(3, 128, rows * kN * 4));
    Call(fused ? "llm.platform.q4k_scaled_f16_swiglu"
         : q6  ? "llm.platform.q6k_scaled_f16_store"
               : "llm.platform.q4k_scaled_f16_store",
         std::move(arguments), kN / 64, rows / 64);
  }
  uint32_t Word(size_t argument, uint64_t relative = 0, unsigned width = 4) {
    const auto ref = invocation.arguments[argument].value.buffer;
    const auto *root = FindInputRoot(prepared.roots, ref);
    if (!root) {
      ADD_FAILURE() << "missing fixture root";
      return UINT32_MAX;
    }
    const auto begin = ref.byte_offset - root->original_byte_offset + relative;
    if (begin + width > root->bytes.size()) {
      ADD_FAILURE() << "missing fixture bytes";
      return UINT32_MAX;
    }
    uint32_t result = 0;
    for (unsigned lane = 0; lane < width; ++lane)
      result |= uint32_t(root->bytes[begin + lane]) << (8 * lane);
    return result;
  }
  PreparationStatus Prepare(uint64_t budget = 8 * 1024 * 1024) {
    return PrepareScaledF16Case(plan, invocation, budget, &prepared, &error);
  }
  std::filesystem::path directory;
  Plan plan;
  Invocation invocation;
  PreparedCase prepared;
  std::string error;
  uint64_t weight_bytes = 0;
};

TEST_F(ScaledF16InputsTest, PreparationPreservesCapacityPaddingAndCountOrigin) {
  plan.workload.active_tokens = 63;
  PrepareCall();
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{65, 128, kK}));
  EXPECT_EQ(prepared.source_argument_types[3], "offset");
  EXPECT_EQ(prepared.buffer_extents[1].byte_length, 65 * kK * 4);
  EXPECT_EQ(prepared.buffer_extents[2].byte_length, 128 * 8);
  EXPECT_EQ(prepared.buffer_extents[3].byte_length, 128 * kK * 2);
  EXPECT_EQ(Word(4, 20), 63);
  EXPECT_TRUE(prepared.weight_sources.empty());
  EXPECT_TRUE(prepared.resets.empty());
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
}

TEST_F(ScaledF16InputsTest, Q4AndQ6MatricesUseFiniteProducerReachableRows) {
  for (bool q6 : {false, true}) {
    Weights(q6);
    for (uint64_t active : {63u, 65u, 128u}) {
      plan.workload.active_tokens = active;
      MatrixCall(q6);
      ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
      EXPECT_EQ(prepared.source_workloads,
                (std::vector<uint64_t>{128, kK, kN}));
      EXPECT_EQ(Word(3), active);
      EXPECT_EQ(Word(5), 0x3f800000u);
      EXPECT_EQ(Word(5, (active - 1) * 8 + 4), 0x3f800000u);
      const auto half = Word(4, 0, 2);
      EXPECT_NE(half & 0x7fff, 0);
      EXPECT_NE(half & 0x7c00, 0x7c00);
      if (active < 128) {
        EXPECT_EQ(Word(4, active * kK * 2, 2), 0);
        EXPECT_EQ(Word(5, active * 8), 0);
        EXPECT_EQ(Word(5, active * 8 + 4), 0);
      }
      ASSERT_EQ(prepared.weight_sources.size(), 1);
      EXPECT_EQ(prepared.weight_sources[0].key, "first.weight");
      EXPECT_EQ(prepared.weight_sources[0].source_offset, 8);
      EXPECT_EQ(prepared.weight_sources[0].byte_length, weight_bytes);
      EXPECT_EQ(Word(6, q6 ? 208 : 0, 2), 0x3c00u);
      EXPECT_EQ(prepared.arithmetic_check, "unchecked");
      EXPECT_TRUE(prepared.resets.empty());
    }
  }
}

TEST_F(ScaledF16InputsTest, FusedOperationRetainsTwoOrderedRealWeightSources) {
  MatrixCall(false, true);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  ASSERT_EQ(prepared.weight_sources.size(), 2);
  EXPECT_EQ(prepared.weight_sources[0].key, "first.weight");
  EXPECT_EQ(prepared.weight_sources[1].key, "second.weight");
  EXPECT_EQ(prepared.weight_sources[1].source_offset, 8 + weight_bytes);
  EXPECT_EQ(prepared.representative_weight_spans.size(), 2);
  EXPECT_EQ(Word(6, 16, 1), 0x11);
  EXPECT_EQ(Word(7, 16, 1), 0x22);
  EXPECT_EQ(
      FindInputRoot(prepared.roots, invocation.arguments[6].value.buffer),
      FindInputRoot(prepared.roots, invocation.arguments[7].value.buffer));
}

TEST_F(ScaledF16InputsTest, MatrixSourceBoundsAreNotNarrowedTo512Rows) {
  MatrixCall(false, false, 1024);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.source_workloads[0], 1024);
}

TEST_F(ScaledF16InputsTest,
       OverlappingDistinctReadsAreFixturePolicyExclusions) {
  for (const size_t argument : {3u, 5u, 4u}) {
    MatrixCall();
    invocation.arguments[argument].value.buffer =
        invocation.arguments[argument == 4 ? 6 : 4].value.buffer;
    if (argument == 4)
      invocation.arguments[argument].value.buffer.byte_length = UINT64_MAX;
    prepared.setup_recipe_id = "preserved";
    EXPECT_EQ(Prepare(), PreparationStatus::kUnsupported);
    EXPECT_EQ(prepared.setup_recipe_id, "preserved");
    EXPECT_NE(error.find("fixture policy"), std::string::npos);
  }
  PrepareCall();
  invocation.arguments[4].value.buffer = invocation.arguments[5].value.buffer;
  EXPECT_EQ(Prepare(), PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

TEST_F(ScaledF16InputsTest, RepeatedActualWeightOperandHasConsistentBytes) {
  MatrixCall(false, true);
  invocation.arguments[7].value.buffer = invocation.arguments[6].value.buffer;
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  ASSERT_EQ(prepared.weight_sources.size(), 2);
  EXPECT_EQ(prepared.weight_sources[0].key, "first.weight");
  EXPECT_EQ(prepared.weight_sources[1].key, "first.weight");
  EXPECT_EQ(Word(6, 16, 1), Word(7, 16, 1));
}

TEST_F(ScaledF16InputsTest, ExactBudgetAndMalformedSmallBudgetAreDistinct) {
  MatrixCall(false, true);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  uint64_t total = 0;
  for (const auto &root : prepared.roots)
    total += root.bytes.size();
  ASSERT_EQ(Prepare(total), PreparationStatus::kReady) << error;
  prepared.setup_recipe_id = "preserved";
  EXPECT_EQ(Prepare(total - 1), PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  EXPECT_NE(error.find("budget"), std::string::npos);
  invocation.arguments[8].value.buffer.byte_length = 1;
  EXPECT_EQ(Prepare(1), PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

TEST_F(ScaledF16InputsTest, BadAbiGridCountsAndWriteAliasesFailAtomically) {
  PrepareCall();
  prepared.setup_recipe_id = "preserved";
  --invocation.command.argument_count;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  PrepareCall();
  invocation.arguments[3].kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  PrepareCall();
  invocation.command.payload.dispatch.workgroup_count[0] = 64;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  PrepareCall();
  plan.workload.active_tokens = 66;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  MatrixCall();
  invocation.arguments[7].value.buffer = invocation.arguments[4].value.buffer;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

TEST_F(ScaledF16InputsTest, WeightShapeMissingPairAndEncodedScaleAreChecked) {
  MatrixCall(false, true);
  plan.programs[0].parameters.pop_back();
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  Weights(false);
  plan.model.globals[1].tensor.storage_type = 14;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  Weights(false);
  {
    std::fstream stream(plan.source_asset,
                        std::ios::binary | std::ios::in | std::ios::out);
    stream.seekp(8 + weight_bytes);
    const char infinity[] = {0, 0x7c};
    stream.write(infinity, 2);
  }
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
  std::filesystem::resize_file(plan.source_asset, 16);
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  EXPECT_NE(error.find("truncated"), std::string::npos);
}

TEST_F(ScaledF16InputsTest, UnknownFamilyPreservesPriorReasonAndOutput) {
  invocation.kernel_symbol = "unknown.operation";
  error = "prior reason";
  prepared.setup_recipe_id = "preserved";
  EXPECT_EQ(Prepare(), PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "prior reason");
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
