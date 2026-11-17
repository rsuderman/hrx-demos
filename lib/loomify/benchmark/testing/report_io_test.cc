// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/report_io.h"
#include "loom/util/digest.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace hrx::loomify_gguf::benchmark {
namespace {

// The export manifest's actual root schema, with exporter-only metadata mixed
// into the same case/origin rows. This is not a second report-only manifest.
std::string Manifest() {
  return R"({"format":"loomify.benchmark.bundle.v1",
    "manifest_id":"manifest-A","selection_id":"selection-A",
    "workload_id":"workload-A","phase":"decode",
    "artifacts":[{"path":"benchmark.loombc","sha256":"export-owned"}],
    "cases":[{"case_id":"case-A","implementation_id":"impl-A",
      "input_policy_digest":"inputs-A","measurement_subject":"kernel",
      "arithmetic_check":"unchecked","benchmark_symbol":"bench_A",
      "setup_recipe_id":"selected_norm","weight_policy":"representative"}],
    "invocations":[
      {"invocation_id":"p0/c1","case_id":"case-A","unsupported_reason":"","command_ordinal":1},
      {"invocation_id":"p0/c2","case_id":"case-A","unsupported_reason":"","command_ordinal":2},
      {"invocation_id":"p0/c3","case_id":"","unsupported_reason":"missing_input_adapter"}]})";
}

// A minimized pinned native ordinary snapshot. Illustrative scores below are
// test data, not a measured performance claim; real exporter/native roundtrips
// belong to the central consumer gate.
std::string Snapshot() {
  return R"({"tool":"loom-benchmark","output_format":"snapshot","dry_run":false,
    "device":{"driver":"amdgpu","device_id":"rdna"},
    "benchmarks":[{"candidate_id":"c0","benchmark":"bench_A",
      "benchmark_sample_index":0,"configuration_ordinal":0,"trial_index":0,
      "trial_ordinal":0,"work_item_index":7}],
    "work_items":[{"work_item_index":7,"state":"ok","benchmark_sample_index":0,
      "configuration_ordinal":0,"trial_index":0,"trial_ordinal":0,
      "policy":{"measure":"dispatch_complete"},
      "correctness":{"state":"unchecked","sample_count":0},
      "data_cache":{"cache_policy":"binding_ring","binding_ring_count":1},
      "measurement":{"logical_operations_per_batch":1,
        "physical_dispatches_per_logical_operation":1,
        "measured_logical_operation_count":10,"stop_reason":"minimum_reached",
        "operation_timing_ns":{"count":10,"p50":123943},
        "timing_interpretation":{"score":"operation_timing_ns","score_unit":"logical_operation",
          "score_time_domain":"host_wall","warnings":[]}}}]})";
}

std::string Hash(const std::string &bytes) {
  loom_sha256_t digest;
  loom_sha256_compute(iree_make_const_byte_span(bytes.data(), bytes.size()),
                      &digest);
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : digest.bytes) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}

std::string Index(const std::string &snapshot) {
  return R"({"format":"loomify.benchmark.result-index.v1","manifest_id":"manifest-A",
    "results":[{"file":"results/native score.json","sha256":")" +
         Hash(snapshot) +
         R"(","producer_identity":"a401/optimized-test-tool","exclusive_device":false,
      "bindings":[{"case_id":"case-A","input_policy_digest":"inputs-A",
        "candidate_id":"c0","benchmark":"bench_A","benchmark_sample_index":0,
        "configuration_ordinal":0,"trial_index":0,"trial_ordinal":0}]}]})";
}

void Replace(std::string *text, const std::string &before,
             const std::string &after) {
  const auto position = text->find(before);
  ASSERT_NE(position, std::string::npos);
  text->replace(position, before.size(), after);
}

class ReportIoTest : public ::testing::Test {
protected:
  void SetUp() override {
    const auto *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary, nullptr);
    root_ = std::filesystem::path(temporary) /
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directory(root_));
    owns_root_ = true;
    std::filesystem::create_directories(root_ / "bundle/results");
    Write("bundle/manifest.json", Manifest());
    Write("bundle/results/native score.json", Snapshot());
    Write("bundle/index.json", Index(Snapshot()));
  }
  void TearDown() override {
    if (owns_root_) {
      std::error_code error;
      std::filesystem::remove_all(root_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }
  void Write(const std::string &relative, const std::string &bytes) {
    std::ofstream out(root_ / relative, std::ios::binary);
    ASSERT_TRUE(out.good());
    out.write(bytes.data(), bytes.size());
    out.close();
    ASSERT_TRUE(out.good());
  }
  bool Build(BenchmarkReport *report, std::string *error) {
    return BuildBenchmarkReportFromFiles(root_ / "bundle/manifest.json",
                                         root_ / "bundle/index.json", {},
                                         report, error);
  }
  std::filesystem::path root_;
  bool owns_root_ = false;
};

TEST_F(ReportIoTest, MapsActualSchemaAndExplicitNativeCoordinates) {
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build(&report, &error)) << error;
  EXPECT_EQ(report.manifest_id, "manifest-A");
  EXPECT_EQ(report.coverage.selected_invocations, 3);
  EXPECT_EQ(report.coverage.runnable_invocations, 2);
  EXPECT_EQ(report.coverage.measured_invocations, 2);
  EXPECT_EQ(report.coverage.qualified_invocations, 0);
  ASSERT_EQ(report.cases.size(), 1);
  const auto &score = report.cases[0].measurements[0];
  EXPECT_EQ(score.p50_ns, 123943);
  EXPECT_EQ(score.work_item_index, 7);
  EXPECT_EQ(score.result_id, "sha256:" + Hash(Snapshot()));
  EXPECT_EQ(score.quality, "quality_unassessed");
  EXPECT_EQ(score.reason, "device_exclusion_unverified");
  EXPECT_EQ(score.correctness_state, "unchecked");
  EXPECT_NE(BenchmarkReportJson(report).find(
                "\"approximate_numerical_phase_ns\":null"),
            std::string::npos);
}

TEST_F(ReportIoTest, RelativeResultPathsSurviveBundleRelocation) {
  std::filesystem::rename(root_ / "bundle", root_ / "relocated");
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(BuildBenchmarkReportFromFiles(root_ / "relocated/manifest.json",
                                            root_ / "relocated/index.json", {},
                                            &report, &error))
      << error;
  EXPECT_EQ(report.cases[0].measurements[0].p50_ns, 123943);
}

TEST_F(ReportIoTest,
       ManifestBenchmarkDefaultsSelectorAndRejectsWrongCaseMapping) {
  auto index = Index(Snapshot());
  Replace(&index, "\"benchmark\":\"bench_A\",", "");
  Write("bundle/index.json", index);
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build(&report, &error)) << error;
  EXPECT_EQ(report.cases[0].measurements[0].benchmark_identity_state,
            "verified_native_name");
  auto other_snapshot = Snapshot();
  Replace(&other_snapshot, "\"benchmark\":\"bench_A\"",
          "\"benchmark\":\"bench_B\"");
  Write("bundle/results/native score.json", other_snapshot);
  index = Index(other_snapshot);
  Replace(&index, "\"benchmark\":\"bench_A\",", "");
  Write("bundle/index.json", index);
  EXPECT_FALSE(Build(&report, &error));
  index = Index(other_snapshot);
  Replace(&index, "\"benchmark\":\"bench_A\"", "\"benchmark\":\"bench_B\"");
  Write("bundle/index.json", index);
  EXPECT_FALSE(Build(&report, &error));
  EXPECT_NE(error.find("disagrees with manifest case"), std::string::npos);
}

TEST_F(ReportIoTest, ExactResultByteMismatchFailsWithoutChangingOutput) {
  Write("bundle/results/native score.json", Snapshot() + "\n");
  BenchmarkReport report;
  report.manifest_id = "unchanged";
  std::string error;
  EXPECT_FALSE(Build(&report, &error));
  EXPECT_NE(error.find("SHA-256 mismatch"), std::string::npos);
  EXPECT_EQ(report.manifest_id, "unchanged");
}

TEST_F(ReportIoTest, WrongManifestOrCaseIdentityDoesNotAcquireCost) {
  for (bool wrong_manifest : {false, true}) {
    auto index = Index(Snapshot());
    Replace(&index, wrong_manifest ? "manifest-A" : "case-A", "other");
    Write("bundle/index.json", index);
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(Build(&report, &error));
    EXPECT_FALSE(error.empty());
  }
}

TEST_F(ReportIoTest, WrongInputPolicyOrCoordinateIsNotGuessed) {
  for (bool wrong_policy : {false, true}) {
    auto index = Index(Snapshot());
    Replace(&index, wrong_policy ? "inputs-A" : "\"trial_ordinal\":0",
            wrong_policy ? "other" : "\"trial_ordinal\":1");
    Write("bundle/index.json", index);
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(Build(&report, &error));
  }
}

TEST_F(ReportIoTest, EmptyResultListRetainsMissingCosts) {
  Write("bundle/index.json", R"({"format":"loomify.benchmark.result-index.v1",
     "manifest_id":"manifest-A","results":[]})");
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build(&report, &error)) << error;
  EXPECT_EQ(report.coverage.measured_invocations, 0);
  EXPECT_EQ(report.coverage.selected_invocations, 3);
  EXPECT_FALSE(report.cases[0].measurements[0].p50_ns);
  EXPECT_EQ(report.cases[0].measurements[0].reason, "missing_measurement");
}

TEST_F(ReportIoTest,
       OptionalSequenceMetadataAndExplicitPartitionRemainPartial) {
  auto manifest = Manifest();
  Replace(
      &manifest, "\"measurement_subject\":\"kernel\"",
      "\"measurement_subject\":\"sequence\",\"reset_policy\":\"none_overwrite_only\"");
  Replace(&manifest, "\"invocation_id\":\"p0/c1\"",
          "\"invocation_id\":\"p0/c1\",\"program_ordinal\":0");
  Replace(&manifest, "\"invocation_id\":\"p0/c2\"",
          "\"invocation_id\":\"p0/c2\",\"program_ordinal\":0");
  Replace(
      &manifest, "\"invocation_id\":\"p0/c3\"",
      "\"invocation_id\":\"p0/c3\",\"program_ordinal\":0,\"command_ordinal\":3");
  manifest.insert(
      manifest.size() - 1,
      ",\"programs\":[{\"commands\":[{},{},{},{}]}],"
      "\"sequence_occurrences\":[{\"occurrence_id\":\"pair\",\"case_id\":\"case-A\","
      "\"program_ordinal\":0,\"command_begin\":1,\"command_end\":3,"
      "\"covered_invocation_ids\":[\"p0/c1\",\"p0/c2\"]}]");
  Write("bundle/manifest.json", manifest);
  auto index = Index(Snapshot());
  index.insert(
      index.size() - 1,
      ",\"sequence_partition\":[{\"occurrence_id\":\"pair\",\"result_id\":\"sha256:" +
          Hash(Snapshot()) + "\"}]");
  Write("bundle/index.json", index);
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build(&report, &error)) << error;
  ASSERT_TRUE(report.sequence_partition);
  EXPECT_FALSE(report.sequence_partition->complete);
  EXPECT_FALSE(report.sequence_partition->qualified_bundle_subtotal_ns);
  EXPECT_EQ(report.sequence_partition->selected_commands, 2);
  EXPECT_EQ(report.sequence_partition->total_selected_commands, 4);
  EXPECT_EQ(report.cases[0].measurements[0].p50_ns, 123943);
  Replace(&index, "\"occurrence_id\":\"pair\"",
          "\"occurrence_id\":\"missing\"");
  Write("bundle/index.json", index);
  report.manifest_id = "unchanged";
  EXPECT_FALSE(Build(&report, &error));
  EXPECT_EQ(report.manifest_id, "unchanged");
}

TEST_F(ReportIoTest, MissingResultFileIsAnErrorNotAMeasuredZero) {
  std::filesystem::remove(root_ / "bundle/results/native score.json");
  BenchmarkReport report;
  std::string error;
  EXPECT_FALSE(Build(&report, &error));
  EXPECT_NE(error.find("cannot open report input"), std::string::npos);
}

TEST_F(ReportIoTest, InvalidCoordinatesAndBooleanDeclarationsAreRejected) {
  for (const std::string replacement : {"-1", "0.5", "\"0\"", "null"}) {
    auto index = Index(Snapshot());
    Replace(&index, "\"trial_ordinal\":0", "\"trial_ordinal\":" + replacement);
    Write("bundle/index.json", index);
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(Build(&report, &error));
  }
  auto index = Index(Snapshot());
  Replace(&index, "\"exclusive_device\":false",
          "\"exclusive_device\":\"true\"");
  Write("bundle/index.json", index);
  BenchmarkReport report;
  std::string error;
  EXPECT_FALSE(Build(&report, &error));
}

TEST_F(ReportIoTest, MissingCandidateAndConflictingDuplicateKeysAreRejected) {
  for (bool candidate : {false, true}) {
    auto index = Index(Snapshot());
    if (candidate)
      Replace(&index, "\"candidate_id\":\"c0\",", "");
    else
      Replace(&index, "\"manifest_id\":\"manifest-A\",",
              "\"manifest_id\":\"manifest-A\",\"manifest_id\":\"manifest-A\",");
    Write("bundle/index.json", index);
    BenchmarkReport report;
    std::string error;
    EXPECT_FALSE(Build(&report, &error));
  }
}

TEST_F(ReportIoTest, ShortWindowWarningsPreserveVisibleRawTiming) {
  auto snapshot = Snapshot();
  Replace(&snapshot, "\"warnings\":[]", "\"warnings\":[\"short window\"]");
  Write("bundle/results/native score.json", snapshot);
  Write("bundle/index.json", Index(snapshot));
  BenchmarkReport report;
  std::string error;
  ASSERT_TRUE(Build(&report, &error)) << error;
  EXPECT_EQ(report.cases[0].measurements[0].p50_ns, 123943);
  EXPECT_EQ(report.cases[0].measurements[0].quality, "unusable");
  EXPECT_EQ(report.coverage.qualified_invocations, 0);
}

TEST(ReportManifestTest, ParsesProjectionAndPreservesOutputOnInvalidInventory) {
  ReportInventory inventory;
  std::string error;
  ASSERT_TRUE(ParseBenchmarkReportManifest(Manifest(), &inventory, &error))
      << error;
  EXPECT_EQ(inventory.cases[0].implementation_id, "impl-A");
  EXPECT_EQ(inventory.invocations.size(), 3);
  auto invalid = Manifest();
  Replace(&invalid, "\"invocation_id\":\"p0/c2\"",
          "\"invocation_id\":\"p0/c1\"");
  inventory.manifest_id = "unchanged";
  EXPECT_FALSE(ParseBenchmarkReportManifest(invalid, &inventory, &error));
  EXPECT_EQ(inventory.manifest_id, "unchanged");
}

TEST(ReportManifestTest, UnknownFormatAndTrailingDocumentAreRejected) {
  for (bool trailing : {false, true}) {
    auto manifest = Manifest();
    if (trailing)
      manifest += "\n{}";
    else
      Replace(&manifest, "loomify.benchmark.bundle.v1", "future-format");
    ReportInventory inventory;
    std::string error;
    EXPECT_FALSE(ParseBenchmarkReportManifest(manifest, &inventory, &error));
  }
}

} // namespace
} // namespace hrx::loomify_gguf::benchmark
