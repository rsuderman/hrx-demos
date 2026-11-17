// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_REPORT_IO_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_REPORT_IO_H_

#include "lib/loomify/benchmark/report.h"
#include <filesystem>
#include <string_view>

namespace hrx::loomify_gguf::benchmark {

// Reads the report projection of the actual export manifest, not a second
// manifest schema. Export-only metadata is ignored. The exporter owns canonical
// manifest identity construction; this consumer validates the declared identity
// and report-required fields, not that export canonicalization algorithm.
bool ParseBenchmarkReportManifest(std::string_view manifest_json,
                                  ReportInventory *inventory,
                                  std::string *error);

// Reads a loomify.benchmark.result-index.v1 document and verifies each
// referenced snapshot's SHA-256 before passing explicit
// case/candidate/coordinate bindings to the existing native snapshot importer.
// Relative result paths resolve from the index's directory. No execution,
// fixture loading, or report output writes. Result timings remain provisional
// unless the native evidence and caller's explicit provenance/exclusion
// declaration satisfy the normal quality policy. Missing costs are retained,
// never replaced with zero or a phase estimate. Failure preserves *report.
bool BuildBenchmarkReportFromFiles(const std::filesystem::path &manifest_path,
                                   const std::filesystem::path &index_path,
                                   const ReportOptions &options,
                                   BenchmarkReport *report, std::string *error);

} // namespace hrx::loomify_gguf::benchmark
#endif
