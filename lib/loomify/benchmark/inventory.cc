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
bool AccumulatePayload(const PreparedRoot &root, CaseInventory &inventory,
                       std::set<std::array<uint8_t, 32>> &fixture_contents,
                       std::string *error) {
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
  auto retain_payload = [&](const PreparedRoot &root) {
    return AccumulatePayload(root, inventory, fixture_contents, error);
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

bool AccountSequenceInputs(CaseInventory *inventory,
                           const std::vector<PreparedSequenceCase> &sequences,
                           std::string *error) {
  if (!inventory)
    return Fail(error, "input footprint requires inventory");
  CaseInventory counts;
  std::set<std::array<uint8_t, 32>> contents;
  auto add = [&](const PreparedCase &prepared) {
    for (const auto &root : prepared.roots)
      if (!AccumulatePayload(root, counts, contents, error))
        return false;
    for (const auto &reset : prepared.resets)
      if (!AccumulatePayload(reset.seed, counts, contents, error))
        return false;
    return true;
  };
  for (const auto &leaf : inventory->prepared_cases)
    if (!add(leaf.inputs))
      return false;
  for (const auto &sequence : sequences)
    if (!add(sequence.inputs))
      return false;
  inventory->retained_input_bytes = counts.retained_input_bytes;
  inventory->unique_fixture_payload_bytes = counts.unique_fixture_payload_bytes;
  inventory->unique_fixture_count = counts.unique_fixture_count;
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
  for (const auto &program : plan.programs)
    report.program_command_counts.push_back(program.commands.size());
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
    for (const auto &source : emitted)
      if (source.case_id == row.case_id)
        row.reset_policy = source.reset_policy;
    const auto symbol = benchmark_symbols.find(row.case_id);
    if (symbol != benchmark_symbols.end())
      row.benchmark_symbol = symbol->second;
    report.cases.push_back(std::move(row));
  }
  for (const auto &origin : inventory.origins) {
    ReportInvocation row;
    const auto &invocation = plan.invocations[origin.invocation_index];
    row.invocation_id = invocation.invocation_id;
    row.program_ordinal = invocation.program_ordinal;
    row.command_ordinal = invocation.command_ordinal;
    if (origin.group_index)
      row.case_id = inventory.groups[*origin.group_index].identity.case_id;
    else
      row.unsupported_reason = origin.unsupported_reason;
    report.invocations.push_back(std::move(row));
  }
  for (const auto &source : emitted) {
    if (source.measurement_subject != "sequence" ||
        source.covered_origin_ids.size() != 2)
      continue;
    const Invocation *first = nullptr;
    const Invocation *last = nullptr;
    for (const auto &invocation : plan.invocations) {
      if (invocation.invocation_id == source.covered_origin_ids.front())
        first = &invocation;
      if (invocation.invocation_id == source.covered_origin_ids.back())
        last = &invocation;
    }
    if (!first || !last || first->program_ordinal != last->program_ordinal ||
        last->command_ordinal != first->command_ordinal + 1)
      continue; // Only proven adjacent native-plan origins are reportable.
    if (!benchmark_symbols.count(source.case_id))
      continue;
    if (std::none_of(
            report.cases.begin(), report.cases.end(),
            [&](const auto &row) { return row.case_id == source.case_id; })) {
      ReportCase row;
      row.case_id = source.case_id;
      row.input_policy_digest = source.input_policy_digest;
      row.measurement_subject = source.measurement_subject;
      row.arithmetic_check = source.arithmetic_check;
      row.implementation_id =
          plan.implementations[first->implementation_index].implementation_id;
      row.benchmark_symbol = source.benchmark_symbol;
      row.reset_policy = source.reset_policy;
      report.cases.push_back(std::move(row));
    }
    SequenceOccurrence occurrence;
    occurrence.occurrence_id = "p" + std::to_string(first->program_ordinal) +
                               "/c" + std::to_string(first->command_ordinal) +
                               ":" + std::to_string(last->command_ordinal + 1);
    occurrence.case_id = source.case_id;
    occurrence.program_ordinal = first->program_ordinal;
    occurrence.command_begin = first->command_ordinal;
    occurrence.command_end = last->command_ordinal + 1;
    occurrence.covered_invocation_ids = source.covered_origin_ids;
    report.sequence_occurrences.push_back(std::move(occurrence));
  }
  return report;
}
} // namespace hrx::loomify_gguf::benchmark
