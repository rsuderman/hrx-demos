// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/report.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace hrx::loomify_gguf::benchmark {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}

bool Add(uint64_t value, uint64_t *total) {
  if (value > std::numeric_limits<uint64_t>::max() - *total)
    return false;
  *total += value;
  return true;
}

bool Compatible(const ReportMeasurement &a, const ReportMeasurement &b) {
  return a.producer_identity == b.producer_identity &&
         a.device_json == b.device_json && a.cache_policy == b.cache_policy &&
         a.binding_ring_count == b.binding_ring_count &&
         a.measurement_materialization == b.measurement_materialization &&
         a.logical_operations_per_batch == b.logical_operations_per_batch;
}
} // namespace

bool AddSequencePartitionReport(
    const ReportInventory &inventory,
    const std::vector<SequenceCostSelection> &selections,
    BenchmarkReport *report, std::string *error) {
  if (!report || report->manifest_id != inventory.manifest_id ||
      report->workload_id != inventory.workload_id ||
      report->phase != inventory.phase)
    return Fail(error, "sequence partition report/inventory identity mismatch");
  SequencePartitionReport partition;
  for (uint64_t count : inventory.program_command_counts) {
    if (count > UINT32_MAX || !Add(count, &partition.total_selected_commands))
      return Fail(error, "selected program command count overflow");
  }
  std::map<std::string, const ReportCase *> cases;
  for (const auto &item : inventory.cases)
    if (!cases.emplace(item.case_id, &item).second)
      return Fail(error, "duplicate sequence case identity");
  std::map<std::pair<uint32_t, uint32_t>, const ReportInvocation *> coordinates;
  if (!inventory.sequence_occurrences.empty()) {
    for (const auto &origin : inventory.invocations) {
      if (!origin.program_ordinal || !origin.command_ordinal ||
          *origin.program_ordinal >= inventory.program_command_counts.size() ||
          *origin.command_ordinal >=
              inventory.program_command_counts[*origin.program_ordinal] ||
          !coordinates
               .emplace(std::make_pair(*origin.program_ordinal,
                                       *origin.command_ordinal),
                        &origin)
               .second)
        return Fail(
            error,
            "sequence occurrence requires unique bounded origin coordinates");
    }
  }
  std::map<std::string, const SequenceOccurrence *> occurrences;
  for (const auto &occurrence : inventory.sequence_occurrences) {
    const auto item = cases.find(occurrence.case_id);
    if (occurrence.occurrence_id.empty() ||
        !occurrences.emplace(occurrence.occurrence_id, &occurrence).second ||
        item == cases.end() ||
        item->second->measurement_subject != "sequence" ||
        item->second->reset_policy != "none_overwrite_only" ||
        occurrence.program_ordinal >= inventory.program_command_counts.size() ||
        occurrence.command_begin >= occurrence.command_end ||
        occurrence.command_end >
            inventory.program_command_counts[occurrence.program_ordinal])
      return Fail(
          error,
          "invalid sequence occurrence identity, subject, or command span");
    std::vector<std::string> actual;
    auto it = coordinates.lower_bound(
        {occurrence.program_ordinal, occurrence.command_begin});
    for (; it != coordinates.end() &&
           it->first.first == occurrence.program_ordinal &&
           it->first.second < occurrence.command_end;
         ++it)
      actual.push_back(it->second->invocation_id);
    if (actual.size() < 2 || actual != occurrence.covered_invocation_ids)
      return Fail(error,
                  "sequence span does not match its ordered selected origins");
  }

  std::map<std::string, const ReportCaseResult *> results;
  for (const auto &item : report->cases)
    if (!results.emplace(item.definition.case_id, &item).second)
      return Fail(error, "duplicate case in sequence report");
  std::set<std::string> selected;
  std::vector<const SequenceOccurrence *> spans;
  for (const auto &selection : selections) {
    const auto found = occurrences.find(selection.occurrence_id);
    if (found == occurrences.end() || selection.result_id.empty() ||
        !selected.insert(selection.occurrence_id).second)
      return Fail(
          error,
          "unknown, duplicate, or unbound sequence occurrence selection");
    spans.push_back(found->second);
  }
  std::sort(spans.begin(), spans.end(), [](const auto *a, const auto *b) {
    return std::tie(a->program_ordinal, a->command_begin) <
           std::tie(b->program_ordinal, b->command_begin);
  });
  for (size_t i = 1; i < spans.size(); ++i)
    if (spans[i - 1]->program_ordinal == spans[i]->program_ordinal &&
        spans[i - 1]->command_end > spans[i]->command_begin)
      return Fail(
          error,
          "sequence partition has overlapping selected command intervals");

  std::vector<const SequenceOccurrence *> qualified_spans;
  const ReportMeasurement *policy = nullptr;
  bool all_eligible = !selections.empty();
  double subtotal = 0;
  for (const auto &selection : selections) {
    const auto &occurrence = *occurrences.at(selection.occurrence_id);
    if (!Add(occurrence.command_end - occurrence.command_begin,
             &partition.selected_commands))
      return Fail(error, "selected sequence command count overflow");
    SequenceContribution contribution;
    contribution.occurrence_id = selection.occurrence_id;
    contribution.case_id = occurrence.case_id;
    contribution.result_id = selection.result_id;
    const ReportMeasurement *measurement = nullptr;
    const auto item = results.find(occurrence.case_id);
    if (item != results.end()) {
      const auto *definition = cases.at(occurrence.case_id);
      if (item->second->definition.implementation_id !=
              definition->implementation_id ||
          item->second->definition.input_policy_digest !=
              definition->input_policy_digest ||
          item->second->definition.measurement_subject !=
              definition->measurement_subject ||
          item->second->definition.reset_policy != definition->reset_policy)
        return Fail(error, "sequence report case disagrees with inventory");
      for (const auto &candidate : item->second->measurements) {
        if (candidate.result_id != selection.result_id)
          continue;
        if (measurement)
          return Fail(error, "ambiguous selected sequence result");
        measurement = &candidate;
      }
    }
    if (!measurement || !measurement->p50_ns) {
      contribution.reason = "missing_selected_sequence_measurement";
    } else if (measurement->quality != "usable" ||
               measurement->correctness_state == "failed" ||
               (measurement->benchmark_identity_state !=
                    "verified_native_name" &&
                measurement->benchmark_identity_state !=
                    "verified_native_baseline_group")) {
      contribution.reason =
          "ineligible_sequence_measurement:" + measurement->quality;
    } else if (policy && !Compatible(*policy, *measurement)) {
      contribution.reason = "incompatible_sequence_measurement_policy";
    } else {
      policy = measurement;
      const double value = *measurement->p50_ns;
      if (!std::isfinite(value) || value < 0 ||
          !std::isfinite(subtotal + value))
        return Fail(error,
                    "sequence partition timing overflow or invalid score");
      subtotal += value;
      contribution.contribution_ns = value;
      qualified_spans.push_back(&occurrence);
      if (!Add(occurrence.command_end - occurrence.command_begin,
               &partition.qualified_commands) ||
          !Add(occurrence.covered_invocation_ids.size(),
               &partition.qualified_dispatch_origins))
        return Fail(error, "qualified sequence coverage overflow");
    }
    if (!contribution.contribution_ns)
      all_eligible = false;
    partition.occurrences.push_back(std::move(contribution));
  }
  if (!qualified_spans.empty())
    partition.qualified_bundle_subtotal_ns = subtotal;
  if (all_eligible)
    partition.partition_total_ns = subtotal;
  std::sort(qualified_spans.begin(), qualified_spans.end(),
            [](const auto *a, const auto *b) {
              return std::tie(a->program_ordinal, a->command_begin) <
                     std::tie(b->program_ordinal, b->command_begin);
            });
  for (uint32_t program = 0; program < inventory.program_command_counts.size();
       ++program) {
    uint64_t cursor = 0;
    for (const auto *span : qualified_spans) {
      if (span->program_ordinal != program)
        continue;
      if (cursor < span->command_begin)
        partition.uncovered_commands.push_back(
            {program, cursor, span->command_begin});
      cursor = span->command_end;
    }
    if (cursor < inventory.program_command_counts[program])
      partition.uncovered_commands.push_back(
          {program, cursor, inventory.program_command_counts[program]});
  }
  partition.complete = all_eligible &&
                       !inventory.program_command_counts.empty() &&
                       partition.uncovered_commands.empty();
  report->sequence_partition = std::move(partition);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::benchmark
