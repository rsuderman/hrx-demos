// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/report.h"
#include "gtest/gtest.h"
#include <limits>

namespace hrx::loomify_gguf::benchmark {
namespace {

ReportInventory Inventory(uint32_t commands = 4) {
  ReportInventory inventory;
  inventory.manifest_id = "manifest";
  inventory.workload_id = "decode";
  inventory.phase = "decode";
  inventory.cases = {
      {"leaf", "leaf-inputs", "kernel", "unchecked", "impl", "leaf"},
      {"pair", "pair-inputs", "sequence", "unchecked", "pair-impl", "pair",
       "none_overwrite_only"}};
  inventory.program_command_counts = {commands};
  for (uint32_t i = 0; i < commands; ++i)
    inventory.invocations.push_back(
        {"p0/c" + std::to_string(i), "leaf", "", 0, i});
  inventory.sequence_occurrences = {
      {"first", "pair", 0, 0, 2, {"p0/c0", "p0/c1"}}};
  if (commands >= 4)
    inventory.sequence_occurrences.push_back(
        {"second", "pair", 0, 2, 4, {"p0/c2", "p0/c3"}});
  return inventory;
}

// Synthetic arithmetic fixture only: native provenance/quality determination is
// separately exercised by report_test and the actual central runner gate.
BenchmarkReport Report(const ReportInventory &inventory) {
  BenchmarkReport report;
  report.manifest_id = inventory.manifest_id;
  report.workload_id = inventory.workload_id;
  report.phase = inventory.phase;
  report.coverage.measured_invocations = 123;
  for (const auto &item : inventory.cases)
    report.cases.push_back({item, 0, {}});
  ReportMeasurement measurement;
  measurement.result_id = "score";
  measurement.p50_ns = 100;
  measurement.quality = "usable";
  measurement.benchmark_identity_state = "verified_native_baseline_group";
  measurement.correctness_state = "unchecked";
  measurement.producer_identity = "optimized-tool";
  measurement.device_json = "{\"device_id\":\"one\"}";
  measurement.cache_policy = "binding_ring";
  measurement.binding_ring_count = 1;
  measurement.logical_operations_per_batch = 8;
  report.cases[1].measurements.push_back(measurement);
  return report;
}

TEST(SequencePartitionTest, OneTwoDispatchSubjectContributesOnce) {
  const auto inventory = Inventory(2);
  auto report = Report(inventory);
  std::string error;
  ASSERT_TRUE(AddSequencePartitionReport(inventory, {{"first", "score"}},
                                         &report, &error))
      << error;
  ASSERT_TRUE(report.sequence_partition);
  const auto &partition = *report.sequence_partition;
  EXPECT_EQ(partition.qualified_bundle_subtotal_ns, 100);
  EXPECT_EQ(partition.partition_total_ns, 100);
  EXPECT_EQ(partition.qualified_commands, 2);
  EXPECT_EQ(partition.qualified_dispatch_origins, 2);
  EXPECT_TRUE(partition.complete);
  EXPECT_EQ(report.coverage.measured_invocations, 123);
  EXPECT_NE(BenchmarkReportJson(report).find(
                "\"approximate_numerical_phase_ns\":100"),
            std::string::npos);
}

TEST(SequencePartitionTest,
     ExplicitRepeatedOccurrencesAreNotDispatchMultipliers) {
  const auto inventory = Inventory();
  auto report = Report(inventory);
  std::string error;
  ASSERT_TRUE(AddSequencePartitionReport(
      inventory, {{"first", "score"}, {"second", "score"}}, &report, &error))
      << error;
  EXPECT_EQ(report.sequence_partition->partition_total_ns, 200);
  EXPECT_EQ(report.sequence_partition->qualified_dispatch_origins, 4);
  EXPECT_TRUE(report.sequence_partition->complete);
}

TEST(SequencePartitionTest, PartialSubtotalNeverClaimsCompletePhase) {
  const auto inventory = Inventory(5);
  auto report = Report(inventory);
  std::string error;
  ASSERT_TRUE(AddSequencePartitionReport(
      inventory, {{"first", "score"}, {"second", "score"}}, &report, &error))
      << error;
  EXPECT_EQ(report.sequence_partition->partition_total_ns, 200);
  EXPECT_FALSE(report.sequence_partition->complete);
  ASSERT_EQ(report.sequence_partition->uncovered_commands.size(), 1);
  EXPECT_EQ(report.sequence_partition->uncovered_commands[0].command_begin, 4);
  EXPECT_EQ(report.sequence_partition->uncovered_commands[0].command_end, 5);
  EXPECT_NE(BenchmarkReportJson(report).find(
                "\"approximate_numerical_phase_ns\":null"),
            std::string::npos);
}

TEST(SequencePartitionTest, MissingNoisyAndUnverifiedAreNotZeroContributions) {
  for (int variation = 0; variation < 4; ++variation) {
    const auto inventory = Inventory(2);
    auto report = Report(inventory);
    auto &measurement = report.cases[1].measurements[0];
    if (variation == 0)
      measurement.p50_ns.reset();
    else if (variation == 1)
      measurement.quality = "noisy";
    else if (variation == 2)
      measurement.benchmark_identity_state = "unverified_no_native_name";
    else
      measurement.correctness_state = "failed";
    std::string error;
    ASSERT_TRUE(AddSequencePartitionReport(inventory, {{"first", "score"}},
                                           &report, &error))
        << error;
    EXPECT_FALSE(report.sequence_partition->partition_total_ns);
    EXPECT_FALSE(report.sequence_partition->qualified_bundle_subtotal_ns);
    EXPECT_FALSE(report.sequence_partition->occurrences[0].contribution_ns);
    EXPECT_FALSE(report.sequence_partition->complete);
    EXPECT_EQ(report.sequence_partition->qualified_commands, 0);
  }
}

TEST(SequencePartitionTest, MissingExplicitResultRetainsGapReason) {
  const auto inventory = Inventory(2);
  auto report = Report(inventory);
  std::string error;
  ASSERT_TRUE(AddSequencePartitionReport(inventory, {{"first", "absent"}},
                                         &report, &error))
      << error;
  EXPECT_EQ(report.sequence_partition->occurrences[0].reason,
            "missing_selected_sequence_measurement");
}

TEST(SequencePartitionTest, AlternativeOverlapAllowedOnlyUntilSelected) {
  auto inventory = Inventory();
  inventory.sequence_occurrences.push_back(
      {"alternative", "pair", 0, 1, 3, {"p0/c1", "p0/c2"}});
  auto report = Report(inventory);
  std::string error;
  ASSERT_TRUE(AddSequencePartitionReport(inventory, {{"first", "score"}},
                                         &report, &error))
      << error;
  EXPECT_FALSE(AddSequencePartitionReport(
      inventory, {{"first", "score"}, {"alternative", "score"}}, &report,
      &error));
  EXPECT_EQ(report.sequence_partition->partition_total_ns, 100);
}

TEST(SequencePartitionTest, DuplicateUnknownAndInvalidOriginMetadataRejects) {
  for (int variation = 0; variation < 4; ++variation) {
    auto inventory = Inventory();
    auto report = Report(inventory);
    std::vector<SequenceCostSelection> selections{{"first", "score"}};
    if (variation == 0)
      selections.push_back(selections[0]);
    else if (variation == 1)
      selections[0].occurrence_id = "unknown";
    else if (variation == 2)
      inventory.sequence_occurrences[0].covered_invocation_ids = {"p0/c1",
                                                                  "p0/c0"};
    else
      inventory.sequence_occurrences[0].command_end = 5;
    std::string error;
    EXPECT_FALSE(
        AddSequencePartitionReport(inventory, selections, &report, &error));
    EXPECT_FALSE(report.sequence_partition);
  }
}

TEST(SequencePartitionTest, ResetCompositeCannotStandForPureSequence) {
  auto inventory = Inventory(2);
  inventory.cases[1].reset_policy = "copy_seed_each_invocation";
  auto report = Report(inventory);
  std::string error;
  EXPECT_FALSE(AddSequencePartitionReport(inventory, {{"first", "score"}},
                                          &report, &error));
}

TEST(SequencePartitionTest, IncompatiblePolicyProducesOnlyKnownSubtotal) {
  const auto inventory = Inventory();
  auto report = Report(inventory);
  auto other = report.cases[1].measurements[0];
  other.result_id = "other";
  other.binding_ring_count = 2;
  report.cases[1].measurements.push_back(other);
  std::string error;
  ASSERT_TRUE(AddSequencePartitionReport(
      inventory, {{"first", "score"}, {"second", "other"}}, &report, &error))
      << error;
  EXPECT_EQ(report.sequence_partition->qualified_bundle_subtotal_ns, 100);
  EXPECT_FALSE(report.sequence_partition->partition_total_ns);
  EXPECT_FALSE(report.sequence_partition->complete);
}

TEST(SequencePartitionTest, TimingOverflowAndWrongIdentityFailAtomically) {
  for (bool wrong_identity : {false, true}) {
    const auto inventory = Inventory();
    auto report = Report(inventory);
    if (wrong_identity)
      report.workload_id = "different";
    else
      report.cases[1].measurements[0].p50_ns =
          std::numeric_limits<double>::max();
    std::string error;
    EXPECT_FALSE(AddSequencePartitionReport(
        inventory, {{"first", "score"}, {"second", "score"}}, &report, &error));
    EXPECT_FALSE(report.sequence_partition);
  }
}

} // namespace
} // namespace hrx::loomify_gguf::benchmark
