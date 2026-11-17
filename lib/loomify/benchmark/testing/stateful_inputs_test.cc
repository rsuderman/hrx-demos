// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace hrx::loomify_gguf::benchmark {
namespace {
loomc_cmd_buffer_ref_t
Ref(uint32_t root, uint64_t offset, uint64_t bytes,
    loomc_cmd_buffer_role_t role = LOOMC_CMD_BUFFER_ROLE_REBINDABLE) {
  return {role, root, offset, bytes};
}
uint32_t Word(const std::vector<uint8_t> &bytes, size_t offset) {
  uint32_t result = 0;
  for (unsigned i = 0; i < 4; ++i)
    result |= static_cast<uint32_t>(bytes[offset + i]) << (8 * i);
  return result;
}
TEST(StatefulInputs, ResetMergesAliasesWithoutIncludingGapsOrGuards) {
  std::vector<PreparedRoot> roots;
  std::string error;
  const auto root_ref = Ref(3, 100, 40);
  ASSERT_TRUE(AllocateInputRoots({{root_ref, 40, 1}}, 40, &roots, &error));
  std::vector<uint8_t> values(40);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<uint8_t>(i + 7);
  ASSERT_TRUE(SetInputBytes(root_ref, values, &roots, &error));
  std::vector<StateReset> resets;
  ASSERT_TRUE(CaptureStateResets(
      {{Ref(3, 105, 5), 5, 1}, {Ref(3, 108, 5), 5, 1}, {Ref(3, 120, 2), 2, 1}},
      roots, &resets, &error))
      << error;
  ASSERT_EQ(resets.size(), 2);
  EXPECT_EQ(resets[0].target.byte_offset, 105);
  EXPECT_EQ(resets[0].target.byte_length, 8);
  EXPECT_EQ(resets[0].seed.bytes,
            (std::vector<uint8_t>(values.begin() + 5, values.begin() + 13)));
  EXPECT_EQ(resets[1].target.byte_offset, 120);
  EXPECT_EQ(resets[1].target.byte_length, 2);
  EXPECT_EQ(resets[0].seed.role, LOOMC_CMD_BUFFER_ROLE_FIXED);
  // The seed owns bytes rather than retaining an alias of the live vector.
  std::fill(roots[0].bytes.begin(), roots[0].bytes.end(), 0);
  EXPECT_NE(resets[0].seed.bytes[0], 0);
}
TEST(StatefulInputs,
     ResetRejectsUninitializedAndOutOfRootWithoutChangingOutput) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots({{Ref(2, 0, 32), 32, 1}}, 32, &roots, &error));
  ASSERT_TRUE(
      SetInputBytes(Ref(2, 0, 8), std::vector<uint8_t>(8, 3), &roots, &error));
  ASSERT_TRUE(
      SetInputBytes(Ref(2, 12, 8), std::vector<uint8_t>(8, 5), &roots, &error));
  std::vector<StateReset> resets(1);
  resets[0].target.byte_offset = 999;
  EXPECT_FALSE(
      CaptureStateResets({{Ref(2, 4, 12), 12, 1}}, roots, &resets, &error));
  ASSERT_EQ(resets.size(), 1);
  EXPECT_EQ(resets[0].target.byte_offset, 999);
  EXPECT_FALSE(
      CaptureStateResets({{Ref(2, 31, 2), 2, 1}}, roots, &resets, &error));
  EXPECT_FALSE(CaptureStateResets({{Ref(2, UINT64_MAX, UINT64_MAX), 2, 1}},
                                  roots, &resets, &error));
  EXPECT_FALSE(
      CaptureStateResets({{Ref(2, 0, 4, LOOMC_CMD_BUFFER_ROLE_FIXED), 4, 1}},
                         roots, &resets, &error));
}
TEST(StatefulInputs, ResetKeepsRolesAndRootsDistinct) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots({{Ref(0, 0, 4), 4, 1}, {Ref(1, 0, 4), 4, 1}},
                                 8, &roots, &error));
  ASSERT_TRUE(SetInputBytes(Ref(0, 0, 4), {1, 2, 3, 4}, &roots, &error));
  ASSERT_TRUE(SetInputBytes(Ref(1, 0, 4), {5, 6, 7, 8}, &roots, &error));
  std::vector<StateReset> resets;
  ASSERT_TRUE(CaptureStateResets({{Ref(0, 0, 4), 4, 1}, {Ref(1, 0, 4), 4, 1}},
                                 roots, &resets, &error));
  ASSERT_EQ(resets.size(), 2);
  EXPECT_EQ(resets[0].target.root_index, 0);
  EXPECT_EQ(resets[1].target.root_index, 1);
  EXPECT_NE(resets[0].seed.bytes, resets[1].seed.bytes);
}

class ShortconvInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *tmp = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(tmp && *tmp);
    source = std::filesystem::path(tmp) / "shortconv-input-weights.bin";
    std::ofstream file(source, std::ios::binary);
    const char prefix[16] = {};
    file.write(prefix, sizeof(prefix));
    for (size_t i = 0; i < 256 * 7; ++i) {
      const float value = static_cast<float>(static_cast<int>(i % 7) - 3) / 16;
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(value));
      for (unsigned lane = 0; lane < 4; ++lane)
        file.put(static_cast<char>(bits >> (lane * 8)));
    }
    file.close();
    ASSERT_TRUE(file);
    plan.source_asset = source.string();
    plan.model.embedding_width = 256;
    plan.model.short_convolution = model::ShortConvolutionDescriptor{256, 7, 6};
    model::TensorRecord taps;
    taps.name = "blk.1.shortconv.conv.weight";
    taps.storage_type = 0;
    taps.byte_size = 256 * 7 * 4;
    taps.absolute_offset = 16;
    taps.rank = 2;
    taps.normalized_extents = {7, 256, 1, 1};
    model::LayerDescriptor layer;
    layer.tensors.push_back({"convolution", taps});
    plan.model.layers.push_back(layer);
    plan.workload.token_capacity = 8;
    plan.workload.active_tokens = 5;
    plan.workload.history_tokens = 17;
    plan.workload.context_capacity = 128;
    plan.programs.resize(1);
    plan.programs[0].parameters.push_back(
        {taps.name, 0, 128, taps.byte_size, 4});
    invocation.kernel_symbol = "llm.convolution.shortconv7";
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.arguments.resize(10);
    for (size_t i = 0; i < 4; ++i)
      invocation.arguments[i].kind = i < 2 ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                                           : LOOMC_CMD_ARGUMENT_KIND_OFFSET;
    invocation.arguments[0].value.scalar_bits = 8;
    invocation.arguments[1].value.scalar_bits = 256;
    invocation.arguments[2].value.scalar_bits = 32;
    invocation.arguments[3].value.scalar_bits = 12;
    for (size_t i = 4; i < 10; ++i) {
      invocation.arguments[i].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
      invocation.arguments[i].value.buffer =
          Ref(static_cast<uint32_t>(i), 64, UINT64_MAX);
    }
    invocation.arguments[7].value.buffer =
        Ref(0, 128, 256 * 7 * 4, LOOMC_CMD_BUFFER_ROLE_FIXED);
  }
  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove(source, ignored);
  }
  Plan plan;
  Invocation invocation;
  std::filesystem::path source;
};
TEST_F(ShortconvInputsTest, UsesExactTapsAndNonzeroHistoryWithPartialRows) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.measurement_subject, "reset_plus_kernel");
  EXPECT_EQ(prepared.reset_policy, "copy_seed_each_invocation");
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "blk.1.shortconv.conv.weight");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 16);
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 256 * 7 * 4);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 256}));
  EXPECT_EQ(prepared.source_argument_types[2], "offset");
  ASSERT_EQ(prepared.representative_weight_spans.size(), 1);
  EXPECT_EQ(prepared.representative_weight_spans[0].byte_length, 256 * 7 * 4);
  const auto count = invocation.arguments[4].value.buffer;
  const auto epoch = invocation.arguments[5].value.buffer;
  const auto *count_root = FindInputRoot(prepared.roots, count);
  const auto *epoch_root = FindInputRoot(prepared.roots, epoch);
  ASSERT_TRUE(count_root && epoch_root);
  EXPECT_EQ(Word(count_root->bytes,
                 count.byte_offset - count_root->original_byte_offset),
            5);
  const auto epoch_base =
      epoch.byte_offset - epoch_root->original_byte_offset + 12;
  EXPECT_EQ(Word(epoch_root->bytes, epoch_base), 5);
  EXPECT_EQ(Word(epoch_root->bytes, epoch_base + 4), 17);
  ASSERT_EQ(prepared.resets.size(), 1);
  const auto &reset = prepared.resets[0];
  EXPECT_EQ(reset.target.root_index,
            invocation.arguments[8].value.buffer.root_index);
  EXPECT_EQ(reset.target.byte_offset, 64 + 32);
  EXPECT_EQ(reset.target.byte_length, 6 * 256 * 4);
  EXPECT_EQ(reset.seed.bytes.size(), 6 * 256 * 4);
  EXPECT_TRUE(std::any_of(reset.seed.bytes.begin(), reset.seed.bytes.end(),
                          [](uint8_t v) { return v != 0; }));
  const auto *weight =
      FindInputRoot(prepared.roots, invocation.arguments[7].value.buffer);
  ASSERT_TRUE(weight);
  // First actual tap is -3/16 (0xbe400000), read at source offset16.
  EXPECT_EQ(Word(weight->bytes, 0), 0xbe400000u);
}
TEST_F(ShortconvInputsTest, FreshAndFirstContinuedHistoryUseExactResetSeeds) {
  for (const uint64_t history : {0u, 1u}) {
    SCOPED_TRACE(history);
    plan.workload.history_tokens = history;
    // Exercise the exact active+history context boundary as well.
    plan.workload.context_capacity = plan.workload.active_tokens + history;
    PreparedCase prepared;
    std::string error;
    ASSERT_EQ(
        PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
        PreparationStatus::kReady)
        << error;
    ASSERT_EQ(prepared.resets.size(), 1);
    const auto &reset = prepared.resets[0];
    EXPECT_EQ(reset.target.byte_offset, 64 + 32);
    EXPECT_EQ(reset.target.byte_length, 6 * 256 * 4);
    ASSERT_EQ(reset.seed.bytes.size(), 6 * 256 * 4);
    const bool all_zero =
        std::all_of(reset.seed.bytes.begin(), reset.seed.bytes.end(),
                    [](uint8_t byte) { return byte == 0; });
    EXPECT_EQ(all_zero, history == 0);
    const auto epoch = invocation.arguments[5].value.buffer;
    const auto *root = FindInputRoot(prepared.roots, epoch);
    ASSERT_TRUE(root);
    EXPECT_EQ(Word(root->bytes,
                   epoch.byte_offset - root->original_byte_offset + 12 + 4),
              history);
    const auto *live = FindInputRoot(prepared.roots, reset.target);
    ASSERT_TRUE(live);
    const auto begin = reset.target.byte_offset - live->original_byte_offset;
    EXPECT_TRUE(std::equal(reset.seed.bytes.begin(), reset.seed.bytes.end(),
                           live->bytes.begin() + begin));
    EXPECT_EQ(prepared.reset_policy, "copy_seed_each_invocation");
  }
}
TEST_F(ShortconvInputsTest, RejectsInvalidHistoryAndGeometryAtomically) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  plan.workload.history_tokens = 127;
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  plan.workload.history_tokens = 17;
  invocation.arguments[1].value.scalar_bits = 257;
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
}
TEST_F(ShortconvInputsTest, RejectsAliasesTruncatedWeightsAndBudget) {
  PreparedCase prepared;
  std::string error;
  EXPECT_EQ(PrepareStatefulCase(plan, invocation, 8, &prepared, &error),
            PreparationStatus::kUnsupported);
  invocation.arguments[9].value.buffer = invocation.arguments[6].value.buffer;
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
  invocation.arguments[9].value.buffer = Ref(9, 64, UINT64_MAX);
  std::filesystem::resize_file(source, 32);
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("truncated"), std::string::npos);
}
TEST_F(ShortconvInputsTest, RejectsNonfiniteRealWeightsAndNegativeOrigins) {
  std::fstream file(source, std::ios::binary | std::ios::in | std::ios::out);
  file.seekp(16);
  const char infinity[] = {0, 0, static_cast<char>(0x80), 0x7f};
  file.write(infinity, sizeof(infinity));
  file.close();
  PreparedCase prepared;
  std::string error;
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
  invocation.arguments[2].value.scalar_bits = UINT64_MAX;
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("history"), std::string::npos);
}
TEST_F(ShortconvInputsTest, UnsupportedOperationDoesNotMaskRecognizedFailures) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  invocation.kernel_symbol = "another.operation";
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  invocation.kernel_symbol = "llm.convolution.shortconv7";
  invocation.arguments.pop_back();
  EXPECT_EQ(
      PrepareStatefulCase(plan, invocation, 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
