// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_INVENTORY_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_INVENTORY_H_

#include "lib/loomify/benchmark/bundle.h"
#include "lib/loomify/benchmark/report.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace hrx::loomify_gguf::benchmark {

struct CaseGroup {
  CaseIdentity identity;
  std::string implementation_id;
  size_t representative_invocation_index = 0;
  std::string measurement_subject;
  std::string arithmetic_check;
};

struct OriginCoverage {
  size_t invocation_index = 0;
  // Index into groups/prepared_cases; empty only for an unsupported operation.
  std::optional<size_t> group_index;
  // This origin uses different real weight bytes than its group's first case.
  bool representative_weights = false;
  std::vector<WeightSource> weight_sources;
  std::string unsupported_reason;
};

struct CaseInventory {
  // Same order as groups. The caller can move this into the emitter while
  // retaining groups/origins for manifest and report publication.
  std::vector<PreparedInvocationCase> prepared_cases;
  std::vector<CaseGroup> groups;
  // One record per Plan::invocations entry, in selected command order.
  std::vector<OriginCoverage> origins;
  // Preflight before publication. Payload bytes exclude small NPY headers and
  // native module artifacts; prepared bytes include distinct physical roots.
  uint64_t retained_input_bytes = 0;
  uint64_t unique_fixture_payload_bytes = 0;
  uint64_t unique_fixture_count = 0;
};

// The operation provider owns admission and input setup. Only kUnsupported
// allows continuing with partial coverage. kFailure (including IO or changing
// source inputs) aborts atomically; error strings are never classified here.
using CasePreparer = PreparationStatus (*)(const Plan &, const Invocation &,
                                           uint64_t, PreparedCase *,
                                           std::string *);
bool BuildCaseInventory(const Plan &plan, uint64_t max_case_input_bytes,
                        CasePreparer prepare, CaseInventory *out,
                        std::string *error);

// Generation-time one-way projection after the facade has bound the actual
// emitted artifacts/fixtures into final_manifest_id. Does not read files or
// confuse Plan::selection_id with the published manifest identity.
ReportInventory ProjectReportInventory(const Plan &plan,
                                       const CaseInventory &inventory,
                                       const std::vector<EmittedCase> &emitted,
                                       const std::string &final_manifest_id);

} // namespace hrx::loomify_gguf::benchmark
#endif
