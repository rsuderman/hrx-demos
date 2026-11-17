// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_EXPORT_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_EXPORT_H_

#include "lib/loomify/benchmark/plan.h"
#include "lib/loomify/benchmark/report.h"
#include <filesystem>

namespace hrx::loomify_gguf::benchmark {

struct ExportOptions {
  // A new JSON file for plan_only, otherwise a new bundle directory.
  std::filesystem::path output;
  bool plan_only = false;
  bool allow_partial = false;
  bool emit_source = true;
  // Optional caller resource budget; UINT64_MAX means no imposed per-case cap.
  uint64_t max_case_input_bytes = UINT64_MAX;
  // Identity of the generator build, not of a later benchmark runner/device.
  // Empty is explicitly unknown. The CLI supplies its executable SHA-256.
  std::string generator_identity;
};

struct ExportResult {
  std::string selection_id;
  std::string manifest_id; // Empty for plan_only; no fixture identity yet.
  uint64_t selected_invocations = 0;
  uint64_t exported_invocations = 0;
  uint64_t unsupported_invocations = 0;
  uint64_t case_count = 0;
  bool complete = false;
};

// CPU-only generation through the existing numerical builder/planner/recipes.
// Borrow context/workspace synchronously; do not share a workspace
// concurrently. Source file identity is captured before selection and rechecked
// before output publication. Real tensor fixtures are bound by their copied
// content hashes; a full checkpoint content hash is not implied.
//
// Output must not exist, including an empty directory or dangling symlink. A
// sibling staging path is published atomically without replacing a destination;
// unsupported no-replace hosts/filesystems fail instead of falling back to an
// overwriting rename. Failure leaves *out unchanged and removes owned partial
// outputs, reporting cleanup failures. Default export requires complete
// runnable coverage; explicit allow_partial preserves every unsupported
// origin/reason.
bool ExportGgufBenchmarks(const std::string &gguf_path,
                          const Workload &workload,
                          const std::string &library_root,
                          const ExportOptions &options,
                          loomc_context_t *context,
                          loomc_workspace_t *workspace, ExportResult *out,
                          std::string *error);

// Owned-plan inspection only. Does not read tensor bytes or prepare fixtures.
std::string BenchmarkPlanJson(const Plan &plan);

// Reports are computed separately by report_io. This only publishes their JSON
// to a new file using the same atomic no-replace policy as plan inspection.
bool WriteBenchmarkReportFile(const BenchmarkReport &report,
                              const std::filesystem::path &new_file,
                              std::string *error);

} // namespace hrx::loomify_gguf::benchmark
#endif
