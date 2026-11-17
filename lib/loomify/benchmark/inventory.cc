// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/inventory.h"
#include <algorithm>
#include <array>
#include <limits>
#include "loom/util/digest.h"
#include <map>
#include <set>
#include <utility>

namespace hrx::loomify_gguf::benchmark {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
} // namespace

bool BuildCaseInventory(const Plan &plan, uint64_t max_case_input_bytes,
                        CasePreparer prepare, CaseInventory *out,
                        std::string *error) {
  if (!out || !prepare || !max_case_input_bytes)
    return Fail(error,
                "case inventory requires preparer, input budget and output");
  if (plan.workload.weight_policy != "representative" &&
      plan.workload.weight_policy != "all")
    return Fail(error,
                "case inventory weight policy must be representative or all");
  CaseInventory inventory;
  std::map<std::string, size_t> groups, exact_representatives;
  std::set<std::string> origins;
  std::set<std::array<uint8_t, 32>> fixture_contents;
  auto retain_payload = [&](const PreparedRoot &root) -> bool {
    const auto bytes = root.bytes.size();
    if (bytes >
        std::numeric_limits<uint64_t>::max() - inventory.retained_input_bytes)
      return Fail(error, "benchmark retained input byte count overflows");
    inventory.retained_input_bytes += bytes;
    loom_sha256_t digest;
    loom_sha256_compute(iree_make_const_byte_span(root.bytes.data(), bytes),
                        &digest);
    std::array<uint8_t, 32> identity;
    std::copy_n(digest.bytes, identity.size(), identity.begin());
    if (fixture_contents.insert(identity).second) {
      if (bytes > std::numeric_limits<uint64_t>::max() -
                      inventory.unique_fixture_payload_bytes)
        return Fail(error, "benchmark unique fixture byte count overflows");
      inventory.unique_fixture_payload_bytes += bytes;
      ++inventory.unique_fixture_count;
    }
    return true;
  };
  for (size_t i = 0; i < plan.invocations.size(); ++i) {
    const auto &invocation = plan.invocations[i];
    if (invocation.invocation_id.empty() ||
        !origins.insert(invocation.invocation_id).second ||
        invocation.implementation_index >= plan.implementations.size())
      return Fail(error, "case inventory has invalid invocation identities");
    OriginCoverage origin;
    origin.invocation_index = i;
    PreparedCase prepared;
    std::string preparation_error;
    const auto status = prepare(plan, invocation, max_case_input_bytes,
                                &prepared, &preparation_error);
    if (status == PreparationStatus::kFailure)
      return Fail(error, "cannot prepare " + invocation.invocation_id + ": " +
                             preparation_error);
    if (status == PreparationStatus::kUnsupported) {
      if (preparation_error.empty())
        return Fail(error, "unsupported case preparation requires a reason");
      origin.unsupported_reason = std::move(preparation_error);
      inventory.origins.push_back(std::move(origin));
      continue;
    }
    if (status != PreparationStatus::kReady)
      return Fail(error, "case preparation returned an invalid status");
    uint64_t case_bytes = 0;
    auto count_case_bytes = [&](const PreparedRoot &root) -> bool {
      if (root.bytes.size() > max_case_input_bytes - case_bytes)
        return false;
      case_bytes += root.bytes.size();
      return true;
    };
    for (const auto &root : prepared.roots)
      if (!count_case_bytes(root))
        return Fail(error, "prepared case exceeds its declared input budget");
    for (const auto &reset : prepared.resets)
      if (!count_case_bytes(reset.seed))
        return Fail(error, "prepared case reset seeds exceed its input budget");
    CaseIdentity identity;
    if (!BuildCaseIdentity(plan, invocation, prepared, &identity, error))
      return false;
    std::string grouping_key = "exact/" + identity.case_id;
    if (plan.workload.weight_policy == "representative" &&
        !identity.representative_group_key.empty())
      grouping_key = "representative/" + identity.representative_group_key;
    origin.weight_sources = prepared.weight_sources;
    const auto exact = exact_representatives.find(identity.case_id);
    const auto [group, inserted] =
        groups.emplace(grouping_key, exact != exact_representatives.end()
                                         ? exact->second
                                         : inventory.groups.size());
    const auto group_index =
        exact != exact_representatives.end() ? exact->second : group->second;
    origin.group_index = group_index;
    if (exact == exact_representatives.end() && inserted) {
      exact_representatives.emplace(identity.case_id, group_index);
      CaseGroup row;
      row.identity = std::move(identity);
      row.implementation_id =
          plan.implementations[invocation.implementation_index]
              .implementation_id;
      row.representative_invocation_index = i;
      row.measurement_subject = prepared.measurement_subject;
      row.arithmetic_check = prepared.arithmetic_check;
      for (const auto &root : prepared.roots)
        if (!retain_payload(root))
          return false;
      for (const auto &reset : prepared.resets)
        if (!retain_payload(reset.seed))
          return false;
      inventory.groups.push_back(std::move(row));
      inventory.prepared_cases.push_back({i, std::move(prepared)});
    } else {
      origin.representative_weights =
          identity.case_id != inventory.groups[group_index].identity.case_id;
    }
    inventory.origins.push_back(std::move(origin));
  }
  *out = std::move(inventory);
  if (error)
    error->clear();
  return true;
}

ReportInventory ProjectReportInventory(const Plan &plan,
                                       const CaseInventory &inventory,
                                       const std::vector<EmittedCase> &emitted,
                                       const std::string &final_manifest_id) {
  ReportInventory report;
  report.manifest_id = final_manifest_id;
  report.workload_id = plan.workload_id;
  report.phase = plan.workload.phase;
  std::map<std::string, std::string> benchmark_symbols;
  for (const auto &source : emitted)
    benchmark_symbols.emplace(source.case_id, source.benchmark_symbol);
  for (const auto &group : inventory.groups) {
    ReportCase row;
    row.case_id = group.identity.case_id;
    row.input_policy_digest = group.identity.input_policy_digest;
    row.measurement_subject = group.measurement_subject;
    row.arithmetic_check = group.arithmetic_check;
    row.implementation_id = group.implementation_id;
    const auto symbol = benchmark_symbols.find(row.case_id);
    if (symbol != benchmark_symbols.end())
      row.benchmark_symbol = symbol->second;
    report.cases.push_back(std::move(row));
  }
  for (const auto &origin : inventory.origins) {
    ReportInvocation row;
    row.invocation_id = plan.invocations[origin.invocation_index].invocation_id;
    if (origin.group_index)
      row.case_id = inventory.groups[*origin.group_index].identity.case_id;
    else
      row.unsupported_reason = origin.unsupported_reason;
    report.invocations.push_back(std::move(row));
  }
  return report;
}
} // namespace hrx::loomify_gguf::benchmark
