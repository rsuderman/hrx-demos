// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include "gtest/gtest.h"
#include <algorithm>

namespace hrx::loomify_gguf::benchmark {
namespace {
loomc_cmd_buffer_ref_t Ref(uint32_t root, uint64_t offset = 0,
                           uint64_t bytes = UINT64_MAX) {
  return {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, root, offset, bytes};
}
uint32_t Word(const PreparedRoot &root, uint64_t absolute) {
  const uint64_t offset = absolute - root.original_byte_offset;
  uint32_t value = 0;
  for (unsigned lane = 0; lane < 4; ++lane)
    value |= static_cast<uint32_t>(root.bytes[offset + lane]) << (lane * 8);
  return value;
}
Invocation Make(bool publish, uint64_t head_size = 128, bool gated = false) {
  Invocation invocation;
  invocation.kernel_symbol =
      publish ? "llm.decoder.publish_kv_h" + std::to_string(head_size)
      : gated ? "llm.attention.gated_window_rows_h" + std::to_string(head_size)
              : "llm.decoder.attention_rows_h128";
  invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
  invocation.command.payload.dispatch.workgroup_count[0] = publish ? 2 : 4;
  invocation.command.payload.dispatch.workgroup_count[1] = 8;
  invocation.command.payload.dispatch.workgroup_count[2] = 1;
  const size_t n = publish ? 7 : gated ? 10 : 9;
  const size_t offsets = publish ? 4 : gated ? 7 : 6;
  invocation.arguments.resize(n + (gated ? 6 : 5));
  invocation.command.argument_count = invocation.arguments.size();
  for (size_t i = 0; i < n; ++i)
    invocation.arguments[i].kind = i >= offsets ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                                   : !publish && i == 5
                                       ? LOOMC_CMD_ARGUMENT_KIND_B32
                                       : LOOMC_CMD_ARGUMENT_KIND_INDEX;
  const std::vector<uint64_t> scalars =
      publish ? std::vector<uint64_t>{1, 3, 128, 192, 16, 12, 8}
      : gated
          ? std::vector<uint64_t>{2, 1, 3, 128, 192, 0x3d800000, 64, 16, 12, 8}
          : std::vector<uint64_t>{2, 1, 3, 128, 192, 0x3d800000, 16, 12, 8};
  for (size_t i = 0; i < n; ++i)
    invocation.arguments[i].value.scalar_bits = scalars[i];
  for (size_t i = n; i < invocation.arguments.size(); ++i) {
    invocation.arguments[i].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[i].value.buffer =
        Ref(static_cast<uint32_t>(i - n), 64 * (i - n + 1));
  }
  return invocation;
}
Plan Profile() {
  Plan plan;
  plan.workload.token_capacity = 8;
  plan.workload.active_tokens = 5;
  plan.workload.history_tokens = 65;
  plan.workload.context_capacity = 128;
  plan.workload.seed = 7;
  return plan;
}
constexpr uint64_t kBudget = 16 * 1024 * 1024;

TEST(AttentionInputs, PublisherUsesShiftedEpochAndNonidentityPages) {
  const auto invocation = Make(true);
  auto plan = Profile();
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 2}));
  EXPECT_EQ(prepared.source_argument_types.size(), 12);
  EXPECT_EQ(prepared.source_argument_types[4], "offset");
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  EXPECT_TRUE(prepared.resets.empty());
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  EXPECT_TRUE(prepared.weight_sources.empty());
  const auto epoch = invocation.arguments[7].value.buffer;
  const auto map = invocation.arguments[8].value.buffer;
  const auto *epoch_root = FindInputRoot(prepared.roots, epoch);
  const auto *map_root = FindInputRoot(prepared.roots, map);
  ASSERT_TRUE(epoch_root && map_root);
  EXPECT_EQ(Word(*epoch_root, epoch.byte_offset + 12), 5);
  EXPECT_EQ(Word(*epoch_root, epoch.byte_offset + 16), 65);
  EXPECT_EQ(Word(*map_root, map.byte_offset + 8), 1);
  EXPECT_EQ(Word(*map_root, map.byte_offset + 12), 2);
  const auto cache = invocation.arguments[11].value.buffer;
  const auto *cache_root = FindInputRoot(prepared.roots, cache);
  ASSERT_TRUE(cache_root);
  const uint64_t plane = 192 * 2 * 128 * 2;
  EXPECT_EQ(cache_root->bytes.size(), 16 + 4 * plane);
  const size_t selected_begin = 16 + 2 * plane;
  EXPECT_TRUE(std::any_of(cache_root->bytes.begin() + selected_begin,
                          cache_root->bytes.end(),
                          [](uint8_t byte) { return byte != 0; }));
  EXPECT_TRUE(std::all_of(cache_root->bytes.begin(),
                          cache_root->bytes.begin() + selected_begin,
                          [](uint8_t byte) { return byte == 0; }));
}

TEST(AttentionInputs, CausalAndGatedAttentionUseExactHeadGeometry) {
  const auto plan = Profile();
  for (const auto &invocation :
       {Make(false), Make(false, 128, true), Make(false, 256, true)}) {
    PreparedCase prepared;
    std::string error;
    ASSERT_EQ(
        PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
        PreparationStatus::kReady)
        << error;
    EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 4, 2}));
    EXPECT_EQ(prepared.source_argument_types[5], "f32");
    EXPECT_TRUE(prepared.resets.empty());
    EXPECT_EQ(prepared.reset_policy, "none_overwrite_only");
    EXPECT_EQ(prepared.buffer_extents.back().byte_length,
              8 * 4 *
                  (invocation.kernel_symbol.find("h256") != std::string::npos
                       ? 256
                       : 128) *
                  4);
  }
}

TEST(AttentionInputs, DenseCacheDoesNotReadOrAllocateMapOrigin) {
  auto invocation = Make(false);
  invocation.arguments[4].value.scalar_bits = 0;
  invocation.arguments[8].value.scalar_bits = UINT64_MAX;
  const auto map = invocation.arguments[10].value.buffer;
  invocation.arguments[10].value.buffer.byte_length = 4;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  const auto *root = FindInputRoot(prepared.roots, map);
  ASSERT_TRUE(root);
  EXPECT_EQ(root->bytes.size(), 4);
  EXPECT_TRUE(root->initialized_ranges.empty());
}

TEST(AttentionInputs, ReadOnlyAliasesAgreeAndWriteAliasesStayUnsupported) {
  auto invocation = Make(false, 128, true);
  invocation.arguments[13].value.buffer = invocation.arguments[12].value.buffer;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  invocation.arguments[15].value.buffer = invocation.arguments[12].value.buffer;
  prepared.setup_recipe_id = "preserved";
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  EXPECT_NE(error.find("disjoint read/write"), std::string::npos);
}

TEST(AttentionInputs, RejectsBadGeometryScaleRangesAndBudgetAtomically) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  auto invocation = Make(false);
  --invocation.command.argument_count;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  invocation = Make(false);
  invocation.arguments[5].value.scalar_bits |= uint64_t{1} << 32;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  invocation = Make(false);
  invocation.arguments[0].value.scalar_bits = 3;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[5].value.scalar_bits = 0x7fc00000;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[6].value.scalar_bits = UINT64_MAX;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[8].value.scalar_bits = 9;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[9].value.buffer.byte_length = 16;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  EXPECT_EQ(PrepareAttentionCase(Profile(), invocation, 1, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

TEST(AttentionInputs, InsufficientLivePagePoolAndUnknownOperationAreExplicit) {
  auto invocation = Make(true, 256);
  invocation.arguments[3].value.scalar_bits = 64;
  PreparedCase prepared;
  std::string error;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kUnsupported);
  invocation = Make(true, 256);
  invocation.kernel_symbol = "unrecognized.attention";
  error = "prior provider diagnostic";
  prepared.setup_recipe_id = "preserved";
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "prior provider diagnostic");
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
