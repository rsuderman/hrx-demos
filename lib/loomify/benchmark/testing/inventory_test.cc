// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/inventory.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
Plan MakePlan(size_t count = 3) {
  Plan plan;
  plan.selection_id = "selection";
  plan.workload_id = "workload";
  plan.implementations.push_back({"implementation", {}, {}});
  for (size_t i = 0; i < count; ++i) {
    Invocation invocation;
    invocation.invocation_id = "i" + std::to_string(i);
    invocation.command_ordinal = i;
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = 3;
    for (auto &count : invocation.command.payload.dispatch.workgroup_count)
      count = 1;
    loomc_cmd_argument_t scalar{};
    scalar.kind = LOOMC_CMD_ARGUMENT_KIND_B32;
    scalar.value.scalar_bits = 1;
    loomc_cmd_argument_t weight{};
    weight.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    weight.value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 0, 100 + i * 8, 4};
    loomc_cmd_argument_t live{};
    live.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    live.value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0, 0, 4};
    invocation.arguments = {scalar, weight, live};
    plan.invocations.push_back(std::move(invocation));
  }
  return plan;
}
PreparationStatus Prepare(const Plan &, const Invocation &invocation, uint64_t,
                          PreparedCase *out, std::string *error) {
  if (invocation.kernel_symbol == "unsupported") {
    *error = "operation has no maintained input recipe";
    return PreparationStatus::kUnsupported;
  }
  if (invocation.kernel_symbol == "io_failure") {
    *error = "GGUF source changed during preparation";
    return PreparationStatus::kFailure;
  }
  if (invocation.kernel_symbol == "unsupported_without_reason")
    return PreparationStatus::kUnsupported;
  PreparedCase prepared;
  prepared.setup_recipe_id = "test.operation/v1";
  const auto &weight = invocation.arguments[1].value.buffer;
  PreparedRoot weight_root;
  weight_root.role = weight.role;
  weight_root.root_index = weight.root_index;
  weight_root.original_byte_offset = weight.byte_offset;
  weight_root.minimum_alignment = 4;
  weight_root.bytes = {uint8_t(invocation.command_ordinal == 1 ? 2 : 1), 0, 0,
                       0};
  weight_root.initialized_ranges = {{0, 4}};
  PreparedRoot live;
  live.minimum_alignment = 4;
  live.bytes.assign(4, 0xA5);
  live.initialized_ranges = {{0, 4}};
  prepared.roots = {weight_root, live};
  prepared.buffer_extents = {{weight, 4, 4},
                             {invocation.arguments[2].value.buffer, 4, 4}};
  if (invocation.kernel_symbol == "uninitialized_weight")
    prepared.roots[0].initialized_ranges.clear();
  prepared.weight_sources = {
      {"weight." + std::to_string(invocation.command_ordinal),
       1000 + invocation.command_ordinal * 4, 4}};
  if (invocation.kernel_symbol == "multiweight")
    prepared.weight_sources.push_back(
        {"second", 9000 + invocation.command_ordinal * 4, 4});
  if (invocation.kernel_symbol != "undeclared_weights")
    prepared.representative_weight_spans.push_back({weight, 4, 4});
  if (invocation.kernel_symbol == "bad_weight_extent")
    prepared.representative_weight_spans[0].byte_length = 8;
  if (invocation.kernel_symbol == "rebindable_weight")
    prepared.representative_weight_spans = {
        {invocation.arguments[2].value.buffer, 4, 4}};
  if (invocation.kernel_symbol == "reset_state" ||
      invocation.kernel_symbol == "reset_overlaps_weight") {
    StateReset reset;
    reset.target = invocation.kernel_symbol == "reset_state"
                       ? invocation.arguments[2].value.buffer
                       : weight;
    reset.seed.minimum_alignment = 4;
    reset.seed.bytes.assign(4, invocation.command_ordinal);
    prepared.resets.push_back(reset);
    prepared.measurement_subject = "reset_plus_kernel";
    prepared.reset_policy = "copy_seed_each_invocation";
  }
  *out = std::move(prepared);
  return PreparationStatus::kReady;
}
std::vector<EmittedCase> Emissions(const CaseInventory &inventory) {
  std::vector<EmittedCase> emitted;
  for (const auto &group : inventory.groups) {
    EmittedCase record;
    record.case_id = group.identity.case_id;
    record.benchmark_symbol =
        "actual_emitted_" + std::to_string(emitted.size());
    emitted.push_back(std::move(record));
  }
  return emitted;
}
TEST(CaseInventory, RepresentativeWeightsRetainEveryOriginAndExactFirstCase) {
  const auto plan = MakePlan();
  CaseInventory inventory;
  std::string error;
  ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
      << error;
  ASSERT_EQ(inventory.groups.size(), 1u);
  ASSERT_EQ(inventory.prepared_cases.size(), 1u);
  EXPECT_EQ(inventory.prepared_cases[0].invocation_index, 0u);
  ASSERT_EQ(inventory.origins.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(inventory.origins[i].invocation_index, i);
    EXPECT_EQ(inventory.origins[i].group_index, 0u);
    ASSERT_EQ(inventory.origins[i].weight_sources.size(), 1u);
    EXPECT_EQ(inventory.origins[i].weight_sources[0].key,
              "weight." + std::to_string(i));
    EXPECT_EQ(inventory.origins[i].weight_sources[0].source_offset,
              1000 + i * 4);
  }
  EXPECT_FALSE(inventory.origins[0].representative_weights);
  EXPECT_TRUE(inventory.origins[1].representative_weights);
  EXPECT_FALSE(inventory.origins[2].representative_weights);
  EXPECT_EQ(inventory.prepared_cases[0].inputs.roots[0].bytes[0], 1);
  EXPECT_EQ(inventory.retained_input_bytes, 8u);
  EXPECT_EQ(inventory.unique_fixture_payload_bytes, 8u);
  EXPECT_EQ(inventory.unique_fixture_count, 2u);
  const auto report = ProjectReportInventory(
      plan, inventory, Emissions(inventory), "actual-final-id");
  EXPECT_EQ(report.manifest_id, "actual-final-id");
  EXPECT_NE(report.manifest_id, plan.selection_id);
  EXPECT_EQ(report.cases[0].benchmark_symbol, "actual_emitted_0");
  BenchmarkReport result;
  ASSERT_TRUE(BuildBenchmarkReport(report, {}, {}, &result, &error)) << error;
  EXPECT_EQ(result.coverage.selected_invocations, 3u);
  EXPECT_EQ(result.coverage.runnable_invocations, 3u);
  ASSERT_EQ(result.cases.size(), 1u);
  EXPECT_EQ(result.cases[0].invocation_count, 3u);
  EXPECT_EQ(result.cases[0].definition.implementation_id, "implementation");
}
TEST(CaseInventory, PreservesEveryWeightSourceForRepresentativeOrigins) {
  auto plan = MakePlan(2);
  for (auto &invocation : plan.invocations)
    invocation.kernel_symbol = "multiweight";
  CaseInventory inventory;
  std::string error;
  ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
      << error;
  ASSERT_EQ(inventory.groups.size(), 1u);
  ASSERT_EQ(inventory.origins.size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(inventory.origins[i].weight_sources.size(), 2u);
    EXPECT_EQ(inventory.origins[i].weight_sources[1].key, "second");
    EXPECT_EQ(inventory.origins[i].weight_sources[1].source_offset,
              9000 + i * 4);
    EXPECT_EQ(inventory.origins[i].weight_sources[1].byte_length, 4u);
  }
  EXPECT_TRUE(inventory.origins[1].representative_weights);
}
TEST(CaseInventory, AllWeightsKeepsDifferentDataAndDeduplicatesExactData) {
  auto plan = MakePlan();
  plan.workload.weight_policy = "all";
  CaseInventory inventory;
  std::string error;
  ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
      << error;
  ASSERT_EQ(inventory.groups.size(), 2u);
  EXPECT_EQ(inventory.origins[0].group_index, inventory.origins[2].group_index);
  EXPECT_NE(inventory.origins[0].group_index, inventory.origins[1].group_index);
  for (const auto &origin : inventory.origins)
    EXPECT_FALSE(origin.representative_weights);
  EXPECT_EQ(inventory.retained_input_bytes, 16u);
  EXPECT_EQ(inventory.unique_fixture_payload_bytes, 12u);
  EXPECT_EQ(inventory.unique_fixture_count, 3u);
}
TEST(CaseInventory, NeverInfersWeightRangesFromNamesOrFixedRoots) {
  auto plan = MakePlan(2);
  for (auto &invocation : plan.invocations)
    invocation.kernel_symbol = "undeclared_weights";
  CaseInventory inventory;
  std::string error;
  ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
      << error;
  EXPECT_EQ(inventory.groups.size(), 2u);
}
TEST(CaseInventory, ExactDedupNeverGrantsUndeclaredWeightSubstitution) {
  auto plan = MakePlan(2);
  plan.invocations[1].command_ordinal = 0; // Same exact bytes as first case.
  plan.invocations[1].kernel_symbol = "undeclared_weights";
  CaseInventory inventory;
  std::string error;
  ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
      << error;
  EXPECT_EQ(inventory.groups.size(), 1u);
  EXPECT_FALSE(inventory.origins[1].representative_weights);
  plan = MakePlan(3);
  plan.invocations[2].command_ordinal = 1; // Same as substituted origin1.
  plan.invocations[2].kernel_symbol = "undeclared_weights";
  ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
      << error;
  EXPECT_EQ(inventory.groups.size(), 2u);
  EXPECT_TRUE(inventory.origins[1].representative_weights);
  EXPECT_FALSE(inventory.origins[2].representative_weights);
  EXPECT_NE(inventory.origins[0].group_index, inventory.origins[2].group_index);
}
TEST(CaseInventory, ScalarsPhysicalGridAndResetSeedsRemainDistinct) {
  for (int mutation = 0; mutation < 3; ++mutation) {
    auto plan = MakePlan(2);
    if (mutation == 0)
      plan.invocations[1].arguments[0].value.scalar_bits = 2;
    if (mutation == 1)
      plan.invocations[1].command.payload.dispatch.workgroup_count[0] = 2;
    if (mutation == 2)
      for (auto &invocation : plan.invocations)
        invocation.kernel_symbol = "reset_state";
    CaseInventory inventory;
    std::string error;
    ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
        << error;
    EXPECT_EQ(inventory.groups.size(), 2u);
  }
}
TEST(CaseInventory, UnsupportedIsAccountedWhileIoFailureAlwaysAborts) {
  auto plan = MakePlan();
  plan.invocations[1].kernel_symbol = "unsupported";
  CaseInventory inventory;
  std::string error;
  ASSERT_TRUE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error))
      << error;
  ASSERT_EQ(inventory.origins.size(), 3u);
  EXPECT_FALSE(inventory.origins[1].group_index);
  EXPECT_EQ(inventory.origins[1].unsupported_reason,
            "operation has no maintained input recipe");
  const auto report =
      ProjectReportInventory(plan, inventory, Emissions(inventory), "final");
  EXPECT_EQ(report.invocations[1].unsupported_reason,
            inventory.origins[1].unsupported_reason);
  plan.invocations[2].kernel_symbol = "io_failure";
  EXPECT_FALSE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error));
  EXPECT_EQ(inventory.origins.size(), 3u);
  EXPECT_EQ(inventory.origins[1].unsupported_reason,
            "operation has no maintained input recipe");
  EXPECT_NE(error.find("GGUF source changed"), std::string::npos);
}
TEST(CaseInventory, RejectsUnsafeWeightMasksAndInvalidAdmissionResults) {
  for (const char *mode :
       {"bad_weight_extent", "uninitialized_weight", "rebindable_weight",
        "reset_overlaps_weight", "unsupported_without_reason"}) {
    auto plan = MakePlan(1);
    plan.invocations[0].kernel_symbol = mode;
    CaseInventory inventory;
    std::string error;
    EXPECT_FALSE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(inventory.origins.empty());
  }
  CaseInventory too_large;
  std::string budget_error;
  EXPECT_FALSE(
      BuildCaseInventory(MakePlan(1), 4, Prepare, &too_large, &budget_error));
  EXPECT_NE(budget_error.find("input budget"), std::string::npos);
  auto plan = MakePlan(2);
  plan.invocations[1].invocation_id = plan.invocations[0].invocation_id;
  CaseInventory inventory;
  std::string error;
  EXPECT_FALSE(BuildCaseInventory(plan, 1024, Prepare, &inventory, &error));
  EXPECT_NE(error.find("invocation identities"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
