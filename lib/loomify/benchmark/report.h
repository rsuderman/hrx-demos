// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_REPORT_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_REPORT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hrx::loomify_gguf::benchmark {

// A one-way projection of an export plan for reporting. This deliberately owns
// neither the compiler plan nor the fixture/manifest serialization schema.
struct ReportCase {
  std::string case_id;
  std::string input_policy_digest;
  std::string measurement_subject; // kernel, sequence, or reset_plus_kernel.
  std::string arithmetic_check = "unchecked";
  std::string implementation_id;
  // Optional for programmatic inventory; required by export-manifest IO.
  std::string benchmark_symbol;
};

struct ReportInvocation {
  std::string invocation_id;
  std::string
      case_id; // Empty for a selected occurrence without a runnable case.
  std::string unsupported_reason;
};

struct ReportInventory {
  std::string manifest_id;
  std::string workload_id;
  std::string phase;
  std::vector<ReportCase> cases;
  std::vector<ReportInvocation> invocations;
};

struct ResultBinding {
  std::string case_id;
  std::string input_policy_digest;
  std::string candidate_id;
  std::string benchmark; // Optional additional check on ordinary snapshots.
  std::string comparison_group; // Optional interleaved group selector.
  std::optional<uint64_t> benchmark_sample_index;
  std::optional<uint64_t> configuration_ordinal;
  std::optional<uint64_t> trial_index;
  std::optional<uint64_t> trial_ordinal;
};

struct ReportResultInput {
  std::string manifest_id;
  // Caller-owned identity for these exact bytes. Conflicting reuse is rejected;
  // callers loading files may use a content digest. No filename-based matching.
  std::string result_id;
  std::string producer_identity; // Exact tool/runtime build identity, if known.
  bool exclusive_device =
      false; // Caller records actual shared-device exclusion.
  std::string snapshot_json;
  std::vector<ResultBinding> bindings;
};

struct ReportOptions {
  double maximum_window_spread = 0.03;
  uint64_t minimum_windows = 3;
};

struct ReportMeasurement {
  std::string result_id;
  std::string candidate_id;
  std::optional<uint64_t> work_item_index;
  std::string state;
  std::string correctness_state = "unchecked";
  std::string quality = "unusable";
  std::string reason;
  std::string benchmark_identity_state = "not_requested";
  std::optional<double> p50_ns;
  std::optional<double> window_spread;
  std::vector<double> window_p50_ns;
  std::optional<uint64_t> logical_operations_per_batch;
  std::optional<uint64_t> physical_dispatches_per_logical_operation;
  // Native evidence is preserved independently of score/quality interpretation.
  std::string device_json;
  std::string data_cache_json;
  std::string timing_warnings_json;
  std::string stop_reason;
  std::string producer_identity;
  bool exclusive_device = false;
  bool device_identity_available = false;
  std::vector<std::string> native_windows_json;
  // Leaf-kernel diagnostic only, never predicted model/GPU latency. Sequence
  // and reset-composite dispatch-origin counts are not logical subject counts.
  std::optional<double> standalone_work_proxy_ns;
};

struct ReportCaseResult {
  ReportCase definition;
  uint64_t invocation_count = 0;
  std::vector<ReportMeasurement> measurements;
};

struct ReportCoverage {
  uint64_t selected_invocations = 0;
  uint64_t runnable_invocations = 0;
  uint64_t measured_invocations = 0;
  uint64_t qualified_invocations = 0;
};

struct BenchmarkReport {
  std::string manifest_id;
  std::string workload_id;
  std::string phase;
  ReportCoverage coverage;
  ReportOptions quality_policy;
  std::vector<ReportCaseResult> cases;
  std::vector<ReportInvocation> unsupported_invocations;
  std::vector<std::string> run_diagnostics;
};

// Consumes actual native ordinary or interleaved snapshot JSON, not JSONL.
// Missing case measurements are valid partial coverage; malformed or ambiguous
// associations fail atomically without replacing *report. No execution or IO.
// This checkpoint reports raw scores/coverage only, not phase-cost estimates.
bool BuildBenchmarkReport(const ReportInventory &inventory,
                          const std::vector<ReportResultInput> &inputs,
                          const ReportOptions &options, BenchmarkReport *report,
                          std::string *error);

std::string BenchmarkReportJson(const BenchmarkReport &report);

} // namespace hrx::loomify_gguf::benchmark
#endif
