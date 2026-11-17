// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/report.h"
#include "iree/base/internal/json.h"
#include "gtest/gtest.h"
#include <limits>
#include <string>
#include <utility>

namespace hrx::loomify_gguf::benchmark {
namespace {

ReportInventory Inventory() {
  ReportInventory result;
  result.manifest_id = "manifest-A";
  result.workload_id = "decode-context128";
  result.phase = "decode";
  result.cases = {
      {"projection", "inputs-A", "sequence", "independent_reference"},
      {"attention", "inputs-B", "reset_plus_kernel", "unchecked"}};
  result.cases[0].implementation_id = "implementation-projection";
  result.cases[1].implementation_id = "implementation-attention";
  result.invocations = {{"layer0/projection", "projection", ""},
                        {"layer1/projection", "projection", ""},
                        {"layer0/attention", "attention", ""},
                        {"layer0/unknown", "", "missing_input_adapter"}};
  return result;
}

// Minimized pinned a401 snapshot shape, retaining the normalized host score,
// actual nine-dispatch/batch64 boundary, cache policy, and correctness
// evidence.
std::string Window(const std::string &p50, const std::string &state = "ok",
                   const std::string &correctness = "passed",
                   const std::string &warnings = "[]") {
  return "{\"state\":\"" + state +
         "\",\"policy\":{\"measure\":\"dispatch_complete\"},"
         "\"correctness\":{\"state\":\"" +
         correctness +
         "\",\"sample_count\":1},"
         "\"data_cache\":{\"cache_policy\":\"binding_ring\",\"binding_ring_count\":1,"
         "\"binding_range_set_bytes\":85260,\"ring_byte_target_met\":false},"
         "\"measurement\":{\"logical_operations_per_batch\":64,"
         "\"physical_dispatches_per_logical_operation\":9,"
         "\"measured_logical_operation_count\":36352,"
         "\"stop_reason\":\"minimum_reached\",\"operation_timing_ns\":{\"count\":568,\"p50\":" +
         p50 +
         "},"
         "\"timing_interpretation\":{\"score\":\"operation_timing_ns\","
         "\"score_time_domain\":\"host_wall\",\"score_unit\":\"logical_operation\","
         "\"warnings\":" +
         warnings + "}}}";
}

std::string AddFields(std::string object, const std::string &fields) {
  object.pop_back();
  return object + "," + fields + "}";
}

std::string Prefix() {
  return "{\"tool\":\"loom-benchmark\",\"output_format\":\"snapshot\","
         "\"dry_run\":false,\"device\":{\"driver\":\"amdgpu\",\"device_id\":\"rdna\"},";
}

ReportResultInput Ordinary(const std::string &p50 = "13773") {
  ReportResultInput result;
  result.manifest_id = "manifest-A";
  result.result_id = "run-1";
  result.producer_identity = "a401/optimized-tool";
  result.exclusive_device = true;
  ResultBinding binding;
  binding.case_id = "projection";
  binding.input_policy_digest = "inputs-A";
  binding.candidate_id = "c0";
  binding.benchmark = "bench_projection";
  binding.benchmark_sample_index = 0;
  result.bindings = {binding};
  result.snapshot_json =
      Prefix() +
      "\"benchmarks\":[{\"candidate_id\":\"c0\",\"benchmark\":\"bench_projection\","
      "\"benchmark_sample_index\":0,\"work_item_index\":17}],\"work_items\":[" +
      AddFields(Window(p50),
                "\"work_item_index\":17,\"benchmark_sample_index\":0") +
      "]}";
  return result;
}

ReportResultInput Interleaved(const std::vector<std::string> &scores = {
                                  "13740", "13773", "13798", "13843"}) {
  auto result = Ordinary();
  result.bindings[0].benchmark.clear();
  result.bindings[0].comparison_group = "baseline";
  std::string rows;
  for (size_t i = 0; i < scores.size(); ++i) {
    if (i)
      rows += ',';
    rows += AddFields(
        Window(scores[i]),
        "\"candidate_id\":\"c0\",\"comparison_group\":\"baseline\","
        "\"order_index\":" +
            std::to_string(i * 2) +
            ",\"benchmark_sample_index\":0,"
            "\"configuration_ordinal\":0,\"trial_index\":0,\"trial_ordinal\":0");
  }
  result.snapshot_json = Prefix() + "\"repetitions\":[" + rows + "]}";
  return result;
}

void Replace(std::string *text, const std::string &from,
             const std::string &to) {
  size_t offset = 0;
  while ((offset = text->find(from, offset)) != std::string::npos) {
    text->replace(offset, from.size(), to);
    offset += to.size();
  }
}

bool Build(const std::vector<ReportResultInput> &inputs,
           BenchmarkReport *report, std::string *error) {
  return BuildBenchmarkReport(Inventory(), inputs, {}, report, error);
}

TEST(BenchmarkReportTest, OrdinaryScoreKeepsNativeBoundaryAndMissingCoverage) {
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({Ordinary()}, &report, &error)) << error;
  ASSERT_EQ(report.cases.size(), 2);
  const auto &score = report.cases[0].measurements[0];
  EXPECT_EQ(score.p50_ns, 13773);
  EXPECT_FALSE(score.standalone_work_proxy_ns);
  EXPECT_EQ(score.logical_operations_per_batch, 64);
  EXPECT_EQ(score.physical_dispatches_per_logical_operation, 9);
  EXPECT_EQ(score.work_item_index, 17);
  EXPECT_EQ(score.quality, "quality_unassessed");
  EXPECT_EQ(score.reason, "insufficient_repeated_windows");
  EXPECT_EQ(report.coverage.selected_invocations, 4);
  EXPECT_EQ(report.coverage.runnable_invocations, 3);
  EXPECT_EQ(report.coverage.measured_invocations, 2);
  EXPECT_EQ(report.coverage.qualified_invocations, 0);
  EXPECT_EQ(report.cases[1].measurements[0].reason, "missing_measurement");
  ASSERT_EQ(report.unsupported_invocations.size(), 1);
  EXPECT_EQ(report.unsupported_invocations[0].unsupported_reason,
            "missing_input_adapter");
}

TEST(BenchmarkReportTest, EmptyResultsRetainAllInventoryGaps) {
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({}, &report, &error)) << error;
  EXPECT_EQ(report.coverage.measured_invocations, 0);
  EXPECT_EQ(report.coverage.selected_invocations, 4);
  EXPECT_FALSE(report.cases[0].measurements[0].p50_ns);
}

TEST(BenchmarkReportTest, EarlyNativeFailureKeepsDiagnosticsAndMissingCost) {
  auto input = Ordinary();
  input.snapshot_json =
      Prefix() +
      "\"failures\":[{\"stage\":\"planning\",\"message\":\"unresolved root\"}]}";
  auto second = input.bindings.front();
  second.case_id = "attention";
  second.input_policy_digest = "inputs-B";
  input.bindings.push_back(second);
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({input, input}, &report, &error)) << error;
  EXPECT_EQ(report.coverage.measured_invocations, 0);
  EXPECT_EQ(report.run_diagnostics.size(), 1);
  EXPECT_EQ(report.cases[0].measurements[0].reason,
            "unattributed_native_run_failure");
  EXPECT_FALSE(report.cases[0].measurements[0].p50_ns);
  EXPECT_EQ(report.cases[1].measurements[0].reason,
            "unattributed_native_run_failure");
}

TEST(BenchmarkReportTest,
     InterleavedSnapshotUsesNearestRankAndNativeAggregate) {
  auto input = Interleaved();
  input.snapshot_json = AddFields(
      input.snapshot_json,
      "\"comparisons\":[{\"comparison_group\":\"baseline\",\"baseline_candidate_id\":\"c0\","
      "\"candidate_id\":\"c1\",\"p50_ns\":{\"baseline\":13773,\"candidate\":10271,"
      "\"ratio\":0.745734,\"speedup\":1.340960}}]");
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({input}, &report, &error)) << error;
  const auto &score = report.cases[0].measurements[0];
  EXPECT_EQ(score.p50_ns, 13773);
  ASSERT_EQ(score.window_p50_ns.size(), 4);
  EXPECT_EQ(score.quality, "usable");
  EXPECT_EQ(report.coverage.qualified_invocations, 2);
  ASSERT_TRUE(score.window_spread);
  EXPECT_NEAR(*score.window_spread, 103.0 / 13773, 1e-12);
}

TEST(BenchmarkReportTest, NoisyNumbersRemainVisibleWithoutQualifiedCoverage) {
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({Interleaved({"100", "101", "150"})}, &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].measurements[0].quality, "noisy");
  EXPECT_EQ(report.cases[0].measurements[0].p50_ns, 101);
  EXPECT_EQ(report.coverage.measured_invocations, 2);
  EXPECT_EQ(report.coverage.qualified_invocations, 0);
}

TEST(BenchmarkReportTest,
     BaselineGroupProvesExistingNativeNameWithoutInventedField) {
  auto inventory = Inventory();
  inventory.cases[0].benchmark_symbol = "bench_projection";
  auto input = Interleaved();
  Replace(&input.snapshot_json, "\"candidate_id\":\"c0\"",
          "\"candidate_id\":\"c0\",\"baseline_candidate_id\":\"c0\"");
  Replace(&input.snapshot_json, "\"comparison_group\":\"baseline\"",
          "\"comparison_group\":\"bench_projection\"");
  input.bindings[0].comparison_group = "bench_projection";
  const auto without_aggregate = input;
  input.snapshot_json =
      AddFields(input.snapshot_json,
                "\"comparisons\":[{\"comparison_group\":\"bench_projection\","
                "\"baseline_candidate_id\":\"c0\",\"candidate_id\":\"c1\","
                "\"p50_ns\":{\"baseline\":13773,\"candidate\":13773}}]");
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(BuildBenchmarkReport(inventory, {input}, {}, &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].measurements[0].benchmark_identity_state,
            "verified_native_baseline_group");
  EXPECT_EQ(report.cases[0].measurements[0].quality, "usable");
  EXPECT_EQ(report.coverage.qualified_invocations, 2);
  ASSERT_TRUE(
      BuildBenchmarkReport(inventory, {without_aggregate}, {}, &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].measurements[0].benchmark_identity_state,
            "unverified_no_native_name");
  auto not_baseline = input;
  Replace(&not_baseline.snapshot_json, "\"baseline_candidate_id\":\"c0\"",
          "\"baseline_candidate_id\":\"c1\"");
  ASSERT_TRUE(
      BuildBenchmarkReport(inventory, {not_baseline}, {}, &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].measurements[0].quality, "quality_unassessed");
}

TEST(BenchmarkReportTest, BaselineGroupMismatchRejectsInsteadOfRelabelingCase) {
  auto inventory = Inventory();
  inventory.cases[0].benchmark_symbol = "different_case";
  auto input = Interleaved();
  Replace(&input.snapshot_json, "\"candidate_id\":\"c0\"",
          "\"candidate_id\":\"c0\",\"baseline_candidate_id\":\"c0\"");
  BenchmarkReport report;
  std::string error;
  EXPECT_FALSE(BuildBenchmarkReport(inventory, {input}, {}, &report, &error));
  EXPECT_NE(error.find("baseline group disagrees"), std::string::npos);
}

TEST(BenchmarkReportTest, ManifestNamesAndInterleavedIdentityAreExplicit) {
  auto inventory = Inventory();
  inventory.cases[0].benchmark_symbol = "bench_projection";
  auto ordinary = Ordinary();
  ordinary.bindings[0].benchmark.clear();
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(BuildBenchmarkReport(inventory, {ordinary}, {}, &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].measurements[0].benchmark_identity_state,
            "verified_native_name");
  auto interleaved = Interleaved();
  ASSERT_TRUE(
      BuildBenchmarkReport(inventory, {interleaved}, {}, &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].measurements[0].benchmark_identity_state,
            "unverified_no_native_name");
  EXPECT_EQ(report.cases[0].measurements[0].quality, "quality_unassessed");
  EXPECT_EQ(report.coverage.qualified_invocations, 0);
  Replace(&interleaved.snapshot_json, "\"comparison_group\":\"baseline\"",
          "\"benchmark\":\"wrong\",\"comparison_group\":\"baseline\"");
  EXPECT_FALSE(
      BuildBenchmarkReport(inventory, {interleaved}, {}, &report, &error));
}

TEST(BenchmarkReportTest, UncheckedMathAndResetCompositeAreExplicit) {
  auto input = Interleaved();
  input.bindings[0].case_id = "attention";
  input.bindings[0].input_policy_digest = "inputs-B";
  Replace(&input.snapshot_json, "\"state\":\"passed\"",
          "\"state\":\"unchecked\"");
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({input}, &report, &error)) << error;
  EXPECT_EQ(report.cases[1].definition.measurement_subject,
            "reset_plus_kernel");
  EXPECT_EQ(report.cases[1].definition.arithmetic_check, "unchecked");
  EXPECT_EQ(report.cases[1].measurements[0].correctness_state, "unchecked");
  EXPECT_EQ(report.coverage.qualified_invocations, 1);
  EXPECT_NE(BenchmarkReportJson(report).find(
                "\"approximate_numerical_phase_ns\":null"),
            std::string::npos);
}

TEST(BenchmarkReportTest,
     WarningsAndFailedCorrectnessExcludeQualityNotRawScore) {
  for (bool failure : {false, true}) {
    auto input = Interleaved();
    if (failure)
      Replace(&input.snapshot_json, "\"state\":\"passed\"",
              "\"state\":\"failed\"");
    else
      Replace(&input.snapshot_json, "\"warnings\":[]",
              "\"warnings\":[\"noisy timer\"]");
    BenchmarkReport report;
    std::string error;
    ASSERT_TRUE(Build({input}, &report, &error)) << error;
    EXPECT_TRUE(report.cases[0].measurements[0].p50_ns);
    EXPECT_EQ(report.cases[0].measurements[0].quality, "unusable");
    EXPECT_EQ(report.coverage.qualified_invocations, 0);
  }
}

TEST(BenchmarkReportTest,
     DeviceExclusionAndProducerEvidenceAreRequiredForQuality) {
  for (bool no_producer : {false, true}) {
    auto input = Interleaved();
    if (no_producer)
      input.producer_identity.clear();
    else
      input.exclusive_device = false;
    BenchmarkReport report;
    std::string error;
    ASSERT_TRUE(Build({input}, &report, &error)) << error;
    EXPECT_EQ(report.cases[0].measurements[0].quality, "quality_unassessed");
    EXPECT_EQ(report.coverage.qualified_invocations, 0);
  }
}

TEST(BenchmarkReportTest, ZeroIsAValueAndNullOrInvalidScoreIsMissing) {
  for (const std::string value : {"0", "null", "-1", "1e9999"}) {
    BenchmarkReport report;
    std::string error;
    ASSERT_TRUE(Build({Ordinary(value)}, &report, &error)) << error << value;
    const auto &score = report.cases[0].measurements[0];
    EXPECT_EQ(score.p50_ns.has_value(), value == "0");
    if (value == "0")
      EXPECT_EQ(score.p50_ns, 0);
    else
      EXPECT_EQ(score.reason, "missing_or_invalid_score");
  }
}

TEST(BenchmarkReportTest, SkippedAndDryRunAreNotMeasuredZero) {
  for (bool dry_run : {false, true}) {
    auto input = Ordinary();
    if (dry_run)
      Replace(&input.snapshot_json, "\"dry_run\":false", "\"dry_run\":true");
    else
      Replace(&input.snapshot_json, "\"state\":\"ok\"",
              "\"state\":\"skipped\"");
    BenchmarkReport report;
    std::string error;
    ASSERT_TRUE(Build({input}, &report, &error)) << error;
    EXPECT_FALSE(report.cases[0].measurements[0].p50_ns);
    EXPECT_EQ(report.coverage.measured_invocations, 0);
    EXPECT_EQ(report.cases[0].measurements[0].reason,
              dry_run ? "dry_run" : "skipped");
  }
}

TEST(BenchmarkReportTest, DifferentTimingBoundaryAndMissingCountsAreExcluded) {
  for (bool boundary : {false, true}) {
    auto input = Ordinary();
    if (boundary)
      Replace(&input.snapshot_json, "dispatch_complete", "case_end_to_end");
    else
      Replace(&input.snapshot_json, "\"logical_operations_per_batch\":64",
              "\"logical_operations_per_batch\":0");
    BenchmarkReport report;
    std::string error;
    ASSERT_TRUE(Build({input}, &report, &error)) << error;
    EXPECT_FALSE(report.cases[0].measurements[0].p50_ns);
    EXPECT_EQ(report.cases[0].measurements[0].reason,
              boundary ? "unsupported_measurement_boundary"
                       : "missing_operation_counts");
  }
}

TEST(BenchmarkReportTest, RepeatedIdenticalInputIsIdempotent) {
  BenchmarkReport report;
  std::string error;
  const auto input = Ordinary();
  ASSERT_TRUE(Build({input, input}, &report, &error)) << error;
  EXPECT_EQ(report.cases[0].measurements.size(), 1);
  EXPECT_EQ(report.coverage.measured_invocations, 2);
}

TEST(BenchmarkReportTest, PhysicalAliasesDoNotInflateInvocationCounts) {
  auto input = Ordinary();
  Replace(&input.snapshot_json, "\"work_item_index\":17}],\"work_items\"",
          "\"work_item_index\":17},{\"candidate_id\":\"c1\","
          "\"benchmark\":\"bench_projection\",\"benchmark_sample_index\":0,"
          "\"work_item_index\":17}],\"work_items\"");
  auto alias = input.bindings.front();
  alias.candidate_id = "c1";
  input.bindings.push_back(alias);
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({input}, &report, &error)) << error;
  EXPECT_EQ(report.coverage.measured_invocations, 2);
  EXPECT_EQ(report.cases[0].measurements.size(), 1);
}

TEST(BenchmarkReportTest, MultiplicityIsDerivedOnceFromAllOrigins) {
  auto inventory = Inventory();
  inventory.cases[0].measurement_subject = "kernel";
  auto input = Ordinary();
  Replace(&input.snapshot_json,
          "\"physical_dispatches_per_logical_operation\":9",
          "\"physical_dispatches_per_logical_operation\":1");
  inventory.invocations.clear();
  for (int i = 0; i < 32; ++i)
    inventory.invocations.push_back(
        {"layer" + std::to_string(i), "projection", ""});
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(BuildBenchmarkReport(inventory, {input}, {}, &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].invocation_count, 32);
  EXPECT_EQ(report.coverage.measured_invocations, 32);
  EXPECT_EQ(report.cases[0].measurements[0].standalone_work_proxy_ns,
            13773 * 32);
}

TEST(BenchmarkReportTest, SequenceOriginCountsAreNotLogicalOccurrenceCounts) {
  for (int sequences : {1, 2}) {
    auto inventory = Inventory();
    inventory.invocations.clear();
    for (int i = 0; i < 9 * sequences; ++i)
      inventory.invocations.push_back(
          {"dispatch" + std::to_string(i), "projection", ""});
    BenchmarkReport report;
    std::string error;
    ASSERT_TRUE(
        BuildBenchmarkReport(inventory, {Ordinary()}, {}, &report, &error))
        << error;
    EXPECT_EQ(report.coverage.measured_invocations, 9 * sequences);
    EXPECT_EQ(report.cases[0].measurements[0].p50_ns, 13773);
    EXPECT_FALSE(report.cases[0].measurements[0].standalone_work_proxy_ns);
  }
}

TEST(BenchmarkReportTest, SameInputsCannotAliasDifferentImplementations) {
  auto inventory = Inventory();
  inventory.cases[1].input_policy_digest = "inputs-A";
  inventory.cases[1].measurement_subject = "sequence";
  auto input = Ordinary();
  auto alias = input.bindings.front();
  alias.case_id = "attention";
  input.bindings.push_back(alias);
  BenchmarkReport report;
  std::string error;
  EXPECT_FALSE(BuildBenchmarkReport(inventory, {input}, {}, &report, &error));
  EXPECT_NE(error.find("incompatible case identities"), std::string::npos);
  // Equal implementations and fixture policies still do not establish runtime
  // argument/count equivalence across distinct exported cases.
  inventory.cases[1].implementation_id = inventory.cases[0].implementation_id;
  EXPECT_FALSE(BuildBenchmarkReport(inventory, {input}, {}, &report, &error));
}

TEST(BenchmarkReportTest, SeparateRunsAreNeverImplicitlyAveraged) {
  auto second = Ordinary("15000");
  second.result_id = "run-2";
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build({Ordinary(), second, Ordinary()}, &report, &error))
      << error;
  ASSERT_EQ(report.cases[0].measurements.size(), 2);
  EXPECT_EQ(report.cases[0].measurements[0].p50_ns, 13773);
  EXPECT_EQ(report.cases[0].measurements[1].p50_ns, 15000);
}

TEST(BenchmarkReportTest, ConflictingIdentityAndPolicyFailAtomically) {
  for (int variation = 0; variation < 4; ++variation) {
    auto first = Ordinary();
    auto second = first;
    if (variation == 0)
      second.snapshot_json += " ";
    else if (variation == 1)
      second.manifest_id = "other";
    else if (variation == 2)
      second.bindings[0].input_policy_digest = "other";
    else
      second.bindings[0].candidate_id = "unknown";
    BenchmarkReport report;
    report.manifest_id = "untouched";
    std::string error;
    EXPECT_FALSE(Build({first, second}, &report, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(report.manifest_id, "untouched");
  }
}

TEST(BenchmarkReportTest, InterleavedGroupCoordinatesAndPolicyCannotMix) {
  for (int variation = 0; variation < 3; ++variation) {
    auto input = Interleaved();
    if (variation == 0)
      input.snapshot_json.replace(
          input.snapshot_json.find("\"trial_ordinal\":0"),
          std::string("\"trial_ordinal\":0").size(), "\"trial_ordinal\":1");
    else if (variation == 1)
      input.snapshot_json.replace(
          input.snapshot_json.find("\"binding_ring_count\":1"),
          std::string("\"binding_ring_count\":1").size(),
          "\"binding_ring_count\":2");
    else {
      input.bindings[0].comparison_group.clear();
      input.snapshot_json.replace(
          input.snapshot_json.find("\"comparison_group\":\"baseline\""),
          std::string("\"comparison_group\":\"baseline\"").size(),
          "\"comparison_group\":\"other\"");
    }
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(Build({input}, &report, &error));
  }
}

TEST(BenchmarkReportTest,
     AggregateWithoutWindowsOrWithWrongNumbersCannotCertify) {
  for (bool no_windows : {false, true}) {
    auto input = Interleaved();
    if (no_windows)
      input.snapshot_json = Prefix() + "\"repetitions\":[],\"comparisons\":[]}";
    else
      input.snapshot_json = AddFields(
          input.snapshot_json,
          "\"comparisons\":[{\"comparison_group\":\"baseline\",\"baseline_candidate_id\":\"c0\","
          "\"candidate_id\":\"c1\",\"p50_ns\":{\"baseline\":1,\"candidate\":2}}]");
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(Build({input}, &report, &error));
  }
}

TEST(BenchmarkReportTest, DuplicateInvocationsAndUnknownCasesAreRejected) {
  for (bool duplicate : {false, true}) {
    auto inventory = Inventory();
    if (duplicate)
      inventory.invocations.push_back(inventory.invocations.front());
    else
      inventory.invocations[0].case_id = "not-a-case";
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(BuildBenchmarkReport(inventory, {}, {}, &report, &error));
  }
}

TEST(BenchmarkReportTest, JsonlTrailingGarbageAndDuplicateKeysAreRejected) {
  for (const std::string suffix : {"\n{}", "trailing"}) {
    auto input = Ordinary();
    input.snapshot_json += suffix;
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(Build({input}, &report, &error));
  }
  auto input = Ordinary();
  input.snapshot_json =
      AddFields(input.snapshot_json, "\"tool\":\"loom-benchmark\"");
  BenchmarkReport report;
  std::string error;
  EXPECT_FALSE(Build({input}, &report, &error));
}

TEST(BenchmarkReportTest, SerializationIsValidJsonAndRetainsLimitsAndEvidence) {
  BenchmarkReport report;
  std::string error;
  auto input = Ordinary();
  input.producer_identity = "tool\"\nidentity";
  ASSERT_TRUE(Build({input}, &report, &error)) << error;
  auto json = BenchmarkReportJson(report);
  auto cursor = iree_make_string_view(json.data(), json.size());
  iree_string_view_t object{};
  auto status = iree_json_consume_object(&cursor, &object);
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
  EXPECT_NE(json.find("\"approximate_numerical_phase_ns\":null"),
            std::string::npos);
  EXPECT_NE(json.find("\"native_windows\":[{"), std::string::npos);
  EXPECT_NE(json.find("\"binding_ring_count\":1"), std::string::npos);
  EXPECT_NE(json.find("\"maximum_window_spread\":"), std::string::npos);
  EXPECT_DOUBLE_EQ(report.quality_policy.maximum_window_spread, 0.03);
  EXPECT_EQ(json, BenchmarkReportJson(report));
}

} // namespace
} // namespace hrx::loomify_gguf::benchmark
