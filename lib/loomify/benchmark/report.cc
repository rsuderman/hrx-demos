// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/report.h"
#include "iree/base/internal/json.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace hrx::loomify_gguf::benchmark {
namespace {

bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}

bool StatusOk(iree_status_t status) {
  const bool ok = iree_status_is_ok(status);
  iree_status_ignore(status);
  return ok;
}

// A typed view built with the existing JSON reader; strings are decoded only
// where needed. The original input owns all raw spans for the duration of
// import.
struct Json {
  iree_json_value_type_t type = IREE_JSON_VALUE_TYPE_NULL;
  iree_string_view_t raw{};
  std::map<std::string, Json> fields;
  std::vector<Json> elements;

  const Json *Get(const std::string &key) const {
    const auto it = fields.find(key);
    return it == fields.end() ? nullptr : &it->second;
  }
  std::string Raw() const { return std::string(raw.data, raw.size); }
};

bool Decode(iree_string_view_t value, std::string *out) {
  out->resize(value.size);
  iree_host_size_t size = 0;
  if (!StatusOk(
          iree_json_unescape_string(value, out->size(), out->data(), &size)))
    return false;
  out->resize(size);
  return true;
}

bool ReadJson(iree_json_value_type_t type, iree_string_view_t raw, Json *out) {
  out->type = type;
  out->raw = raw;
  if (type == IREE_JSON_VALUE_TYPE_OBJECT) {
    const auto visitor = [](void *user, iree_string_view_t key,
                            iree_json_value_type_t type,
                            iree_string_view_t value) -> iree_status_t {
      auto &fields = static_cast<Json *>(user)->fields;
      std::string name;
      Json child;
      if (!Decode(key, &name) || !ReadJson(type, value, &child) ||
          !fields.emplace(std::move(name), std::move(child)).second)
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      return iree_ok_status();
    };
    return StatusOk(iree_json_enumerate_object_typed(raw, visitor, out));
  }
  if (type == IREE_JSON_VALUE_TYPE_ARRAY) {
    const auto visitor = [](void *user, iree_host_size_t,
                            iree_json_value_type_t type,
                            iree_string_view_t value) -> iree_status_t {
      Json child;
      if (!ReadJson(type, value, &child))
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      static_cast<Json *>(user)->elements.push_back(std::move(child));
      return iree_ok_status();
    };
    return StatusOk(iree_json_enumerate_array_typed(raw, visitor, out));
  }
  return true;
}

bool ParseSnapshot(std::string_view input, Json *out) {
  auto cursor = iree_make_string_view(input.data(), input.size());
  if (!StatusOk(iree_json_consume_insignificant(&cursor)) || !cursor.size ||
      cursor.data[0] != '{')
    return false;
  iree_string_view_t object{};
  return StatusOk(iree_json_consume_object(&cursor, &object)) &&
         StatusOk(iree_json_consume_insignificant(&cursor)) && !cursor.size &&
         ReadJson(IREE_JSON_VALUE_TYPE_OBJECT, object, out);
}

std::string String(const Json *value) {
  std::string result;
  if (value && value->type == IREE_JSON_VALUE_TYPE_STRING)
    Decode(value->raw, &result);
  return result;
}

std::optional<uint64_t> Uint(const Json *value) {
  uint64_t result = 0;
  if (!value || value->type != IREE_JSON_VALUE_TYPE_NUMBER ||
      !StatusOk(iree_json_parse_uint64(value->raw, &result)))
    return std::nullopt;
  return result;
}

std::optional<double> Number(const Json *value) {
  double result = 0;
  if (!value || value->type != IREE_JSON_VALUE_TYPE_NUMBER ||
      !StatusOk(iree_json_parse_double(value->raw, &result)) ||
      !std::isfinite(result) || result < 0)
    return std::nullopt;
  return result;
}

const Json *Field(const Json *value, const std::string &key) {
  return value ? value->Get(key) : nullptr;
}

std::string Raw(const Json *value) { return value ? value->Raw() : "null"; }

const std::vector<Json> *Array(const Json *value) {
  return value && value->type == IREE_JSON_VALUE_TYPE_ARRAY ? &value->elements
                                                            : nullptr;
}

bool CoordinateMatches(const Json &row, const ResultBinding &binding) {
  const std::pair<const char *, const std::optional<uint64_t> *> fields[] = {
      {"benchmark_sample_index", &binding.benchmark_sample_index},
      {"configuration_ordinal", &binding.configuration_ordinal},
      {"trial_index", &binding.trial_index},
      {"trial_ordinal", &binding.trial_ordinal}};
  for (const auto &field : fields) {
    if (*field.second && Uint(row.Get(field.first)) != *field.second)
      return false;
  }
  return true;
}

std::string Coordinates(const Json &row) {
  std::string result;
  for (const char *key : {"benchmark_sample_index", "configuration_ordinal",
                          "trial_index", "trial_ordinal"}) {
    result += std::string(key) + "=" + Raw(row.Get(key)) + ";";
  }
  return result;
}

struct Score {
  std::optional<double> value;
  std::string state;
  std::string correctness;
  std::string reason;
  const Json *row = nullptr;
};

Score ReadScore(const Json &row, bool dry_run) {
  Score score;
  score.row = &row;
  score.state = String(row.Get("state"));
  score.correctness = String(Field(row.Get("correctness"), "state"));
  if (score.correctness.empty())
    score.correctness = "unchecked";
  if (Uint(Field(row.Get("correctness"), "failed_sample_count")).value_or(0) !=
      0)
    score.correctness = "failed";
  const auto *measurement = row.Get("measurement");
  const auto *policy = row.Get("policy");
  const auto *interpretation = Field(measurement, "timing_interpretation");
  const auto *timing = Field(measurement, "operation_timing_ns");
  if (dry_run)
    score.reason = "dry_run";
  else if (score.correctness != "unchecked" && score.correctness != "passed" &&
           score.correctness != "failed")
    score.reason = "invalid_correctness_state";
  else if (score.state != "ok")
    score.reason = score.state.empty() ? "missing_state" : score.state;
  else if (String(Field(policy, "measure")) != "dispatch_complete")
    score.reason = "unsupported_measurement_boundary";
  else if (String(Field(interpretation, "score_time_domain")) != "host_wall" ||
           String(Field(interpretation, "score")) != "operation_timing_ns" ||
           String(Field(interpretation, "score_unit")) != "logical_operation")
    score.reason = "incompatible_timing_interpretation";
  else if (Uint(Field(measurement, "logical_operations_per_batch"))
                   .value_or(0) == 0 ||
           Uint(Field(measurement, "measured_logical_operation_count"))
                   .value_or(0) == 0 ||
           Uint(Field(timing, "count")).value_or(0) == 0)
    score.reason = "missing_operation_counts";
  else if (!(score.value = Number(Field(timing, "p50"))))
    score.reason = "missing_or_invalid_score";
  return score;
}

void CopyEvidence(const Json &row, ReportMeasurement *out) {
  const auto *measurement = row.Get("measurement");
  out->logical_operations_per_batch =
      Uint(Field(measurement, "logical_operations_per_batch"));
  out->physical_dispatches_per_logical_operation =
      Uint(Field(measurement, "physical_dispatches_per_logical_operation"));
  out->data_cache_json = Raw(row.Get("data_cache"));
  out->cache_policy = String(Field(row.Get("data_cache"), "cache_policy"));
  out->binding_ring_count =
      Uint(Field(row.Get("data_cache"), "binding_ring_count"));
  out->measurement_materialization =
      String(Field(row.Get("data_cache"), "measurement_materialization"));
  out->timing_warnings_json =
      Raw(Field(Field(measurement, "timing_interpretation"), "warnings"));
  out->stop_reason = String(Field(measurement, "stop_reason"));
}

bool Qualify(const std::vector<Score> &scores, const ReportOptions &options,
             ReportMeasurement *out, std::string *error) {
  if (scores.empty()) {
    out->state = "missing";
    out->reason = "missing_measurement";
    return true;
  }
  out->state = scores.front().state;
  out->correctness_state = scores.front().correctness;
  CopyEvidence(*scores.front().row, out);
  const auto *first_measurement = scores.front().row->Get("measurement");
  const std::string first_policy = Raw(scores.front().row->Get("policy"));
  const std::string coordinates = Coordinates(*scores.front().row);
  bool warnings = false;
  bool correctness_failed = false;
  for (const auto &score : scores) {
    const auto *row = score.row;
    out->native_windows_json.push_back(row->Raw());
    if (Coordinates(*row) != coordinates ||
        Raw(row->Get("policy")) != first_policy ||
        Raw(row->Get("data_cache")) != out->data_cache_json ||
        Raw(Field(row->Get("measurement"), "logical_operations_per_batch")) !=
            Raw(Field(first_measurement, "logical_operations_per_batch")) ||
        Raw(Field(row->Get("measurement"),
                  "physical_dispatches_per_logical_operation")) !=
            Raw(Field(first_measurement,
                      "physical_dispatches_per_logical_operation")))
      return Fail(
          error,
          "inconsistent coordinates or measurement policy across windows");
    if (!score.reason.empty()) {
      out->state = score.state;
      out->reason = score.reason;
      return true;
    }
    out->window_p50_ns.push_back(*score.value);
    correctness_failed |= score.correctness == "failed";
    if (score.correctness != "passed" && !correctness_failed)
      out->correctness_state = "unchecked";
    const auto *values = Array(Field(
        Field(row->Get("measurement"), "timing_interpretation"), "warnings"));
    warnings |= !values || !values->empty();
  }
  auto sorted = out->window_p50_ns;
  std::sort(sorted.begin(), sorted.end());
  // Match native nearest-rank p50, including the lower middle for even counts.
  out->p50_ns = sorted[(sorted.size() - 1) / 2];
  if (correctness_failed) {
    out->correctness_state = "failed";
    out->reason = "correctness_failed";
    return true;
  }
  if (warnings) {
    out->reason = "timing_warnings_or_missing_warning_evidence";
    return true;
  }
  out->quality = "quality_unassessed";
  const auto *cache = scores.front().row->Get("data_cache");
  if (out->producer_identity.empty() || !out->device_identity_available) {
    out->reason = "missing_producer_or_device_identity";
  } else if (String(Field(cache, "cache_policy")).empty() ||
             Uint(Field(cache, "binding_ring_count")).value_or(0) == 0) {
    out->reason = "missing_effective_cache_policy";
  } else if (!out->exclusive_device) {
    out->reason = "device_exclusion_unverified";
  } else if (sorted.size() < options.minimum_windows) {
    out->reason = "insufficient_repeated_windows";
  } else if (*out->p50_ns == 0) {
    out->reason = "zero_median_spread_undefined";
  } else {
    out->window_spread = (sorted.back() - sorted.front()) / *out->p50_ns;
    out->quality = *out->window_spread > options.maximum_window_spread
                       ? "noisy"
                       : "usable";
    if (out->quality == "noisy")
      out->reason = "window_spread_exceeds_policy";
  }
  return true;
}

bool Ordinary(const Json &snapshot, const ResultBinding &binding,
              const ReportOptions &options, ReportMeasurement *out,
              std::string *physical_key, std::string *error) {
  const auto *benchmarks = Array(snapshot.Get("benchmarks"));
  const auto *items = Array(snapshot.Get("work_items"));
  if (!benchmarks || !items)
    return Fail(error,
                "ordinary snapshot requires benchmarks and work_items arrays");
  const Json *selected = nullptr;
  for (const auto &row : *benchmarks) {
    if (String(row.Get("candidate_id")) != binding.candidate_id ||
        (!binding.benchmark.empty() &&
         String(row.Get("benchmark")) != binding.benchmark) ||
        !CoordinateMatches(row, binding))
      continue;
    if (selected)
      return Fail(error, "ambiguous ordinary result binding");
    selected = &row;
  }
  if (!selected)
    return Fail(error, "ordinary result binding selects no benchmark");
  if (!binding.benchmark.empty())
    out->benchmark_identity_state = "verified_native_name";
  out->work_item_index = Uint(selected->Get("work_item_index"));
  if (!out->work_item_index)
    return Fail(error, "ordinary result is missing work_item_index");
  const Json *item = nullptr;
  for (const auto &row : *items) {
    if (Uint(row.Get("work_item_index")) == out->work_item_index) {
      if (item)
        return Fail(error, "duplicate physical work_item_index");
      item = &row;
    }
  }
  if (!item || !CoordinateMatches(*item, binding))
    return Fail(error, "ordinary result has missing or inconsistent work item");
  *physical_key = "work:" + std::to_string(*out->work_item_index);
  const bool dry_run =
      snapshot.Get("dry_run") &&
      snapshot.Get("dry_run")->type == IREE_JSON_VALUE_TYPE_TRUE;
  return Qualify({ReadScore(*item, dry_run)}, options, out, error);
}

bool Interleaved(const Json &snapshot, const ResultBinding &binding,
                 const ReportOptions &options, ReportMeasurement *out,
                 std::string *physical_key, std::string *error) {
  const auto *repetitions = Array(snapshot.Get("repetitions"));
  if (!repetitions)
    return Fail(error, "interleaved snapshot requires repetitions array");
  const bool dry_run =
      snapshot.Get("dry_run") &&
      snapshot.Get("dry_run")->type == IREE_JSON_VALUE_TYPE_TRUE;
  std::vector<Score> scores;
  bool all_names_present = true;
  bool all_rows_are_baseline = true;
  std::string baseline_id;
  std::string group;
  std::set<uint64_t> order_indices;
  for (const auto &row : *repetitions) {
    if (String(row.Get("candidate_id")) != binding.candidate_id ||
        !CoordinateMatches(row, binding) ||
        (!binding.comparison_group.empty() &&
         String(row.Get("comparison_group")) != binding.comparison_group))
      continue;
    const auto current_group = String(row.Get("comparison_group"));
    if (!scores.empty() && current_group != group)
      return Fail(error, "ambiguous interleaved comparison group");
    group = current_group;
    const auto current_baseline = String(row.Get("baseline_candidate_id"));
    if (!baseline_id.empty() && !current_baseline.empty() &&
        baseline_id != current_baseline)
      return Fail(error, "inconsistent native baseline candidate identity");
    if (!current_baseline.empty())
      baseline_id = current_baseline;
    all_rows_are_baseline &=
        !current_baseline.empty() && current_baseline == binding.candidate_id;
    if (!binding.benchmark.empty() && !current_baseline.empty() &&
        current_baseline == binding.candidate_id && group != binding.benchmark)
      return Fail(error,
                  "native baseline group disagrees with manifest benchmark");
    if (!binding.benchmark.empty()) {
      const auto name = String(row.Get("benchmark"));
      if (!name.empty() && name != binding.benchmark)
        return Fail(error, "interleaved benchmark name disagrees with binding");
      all_names_present &= !name.empty();
    }
    const auto order = Uint(row.Get("order_index"));
    if (!order || !order_indices.insert(*order).second)
      return Fail(error, "missing or repeated interleaved order_index");
    scores.push_back(ReadScore(row, dry_run));
  }
  if (scores.empty())
    return Fail(error, "interleaved result binding selects no repetitions");
  *physical_key = "candidate:" + binding.candidate_id + ":" + group + ":" +
                  Coordinates(*scores.front().row);
  if (!Qualify(scores, options, out, error))
    return false;
  // Validate the snapshot aggregate association. It supplements, never
  // replaces, the selected windows and their correctness/policy evidence.
  const auto *comparisons = Array(snapshot.Get("comparisons"));
  bool verified_baseline_aggregate = false;
  if (comparisons && out->p50_ns) {
    std::optional<double> aggregate;
    for (const auto &row : *comparisons) {
      if (String(row.Get("comparison_group")) != group)
        continue;
      const bool baseline =
          String(row.Get("baseline_candidate_id")) == binding.candidate_id;
      const bool candidate =
          String(row.Get("candidate_id")) == binding.candidate_id;
      if (!baseline && !candidate)
        continue;
      const auto value =
          Number(Field(row.Get("p50_ns"), baseline ? "baseline" : "candidate"));
      if (!value || (aggregate && *aggregate != *value))
        return Fail(error,
                    "invalid or conflicting interleaved p50_ns aggregate");
      aggregate = value;
      verified_baseline_aggregate |= baseline;
    }
    if (aggregate && *aggregate != *out->p50_ns)
      return Fail(error,
                  "interleaved aggregate does not match selected windows");
  }
  if (!binding.benchmark.empty()) {
    const bool verified_group = all_rows_are_baseline &&
                                group == binding.benchmark &&
                                verified_baseline_aggregate;
    out->benchmark_identity_state = all_names_present ? "verified_native_name"
                                    : verified_group
                                        ? "verified_native_baseline_group"
                                        : "unverified_no_native_name";
    if (!all_names_present && !verified_group && out->quality == "usable") {
      out->quality = "quality_unassessed";
      out->reason = "benchmark_identity_unverified";
    }
  }
  return true;
}

std::string Quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 0x20)
      out << "\\u00" << "0123456789abcdef"[c >> 4]
          << "0123456789abcdef"[c & 15];
    else
      out << c;
  }
  out << '"';
  return out.str();
}

template <typename T>
void Optional(std::ostream &out, const std::optional<T> &value) {
  if (value)
    out << *value;
  else
    out << "null";
}

} // namespace

bool BuildBenchmarkReport(const ReportInventory &inventory,
                          const std::vector<ReportResultInput> &inputs,
                          const ReportOptions &options, BenchmarkReport *report,
                          std::string *error) {
  if (!report || inventory.manifest_id.empty() ||
      !std::isfinite(options.maximum_window_spread) ||
      options.maximum_window_spread < 0 || options.minimum_windows < 3)
    return Fail(error, "invalid report inventory, output, or quality policy");
  BenchmarkReport result;
  result.manifest_id = inventory.manifest_id;
  result.workload_id = inventory.workload_id;
  result.phase = inventory.phase;
  result.quality_policy = options;
  std::map<std::string, size_t> cases;
  for (const auto &item : inventory.cases) {
    if (item.case_id.empty() || item.input_policy_digest.empty() ||
        item.implementation_id.empty() ||
        (item.measurement_subject != "kernel" &&
         item.measurement_subject != "sequence" &&
         item.measurement_subject != "reset_plus_kernel") ||
        !cases.emplace(item.case_id, result.cases.size()).second)
      return Fail(error, "invalid or duplicate report case");
    result.cases.push_back({item, 0, {}});
  }
  std::set<std::string> invocations;
  for (const auto &invocation : inventory.invocations) {
    if (invocation.invocation_id.empty() ||
        !invocations.insert(invocation.invocation_id).second)
      return Fail(error, "empty or duplicate invocation identity");
    ++result.coverage.selected_invocations;
    const auto it = cases.find(invocation.case_id);
    if (invocation.case_id.empty()) {
      if (invocation.unsupported_reason.empty())
        return Fail(error, "unsupported invocation requires a coverage reason");
      result.unsupported_invocations.push_back(invocation);
    } else if (it == cases.end()) {
      return Fail(error, "invocation refers to unknown case");
    } else {
      ++result.cases[it->second].invocation_count;
      ++result.coverage.runnable_invocations;
    }
  }
  std::map<std::string, std::string> seen_inputs;
  std::map<std::string, std::string> physical_policies;
  std::map<std::pair<std::string, std::string>, std::string> bound_cases;
  for (const auto &input : inputs) {
    if (input.manifest_id != inventory.manifest_id || input.result_id.empty())
      return Fail(error, "result manifest mismatch or missing result identity");
    auto identity = input.snapshot_json + "\n" + input.producer_identity +
                    (input.exclusive_device ? "\nexclusive" : "\nunverified");
    auto prior = seen_inputs.find(input.result_id);
    if (prior != seen_inputs.end() && prior->second != identity)
      return Fail(error, "conflicting result identity");
    seen_inputs.emplace(input.result_id, std::move(identity));
    Json snapshot;
    if (!ParseSnapshot(input.snapshot_json, &snapshot) ||
        String(snapshot.Get("tool")) != "loom-benchmark" ||
        String(snapshot.Get("output_format")) != "snapshot")
      return Fail(error,
                  "expected one loom-benchmark snapshot object (not JSONL)");
    if (snapshot.Get("failures") && prior == seen_inputs.end())
      result.run_diagnostics.push_back(input.result_id + ": " +
                                       Raw(snapshot.Get("failures")));
    if (input.bindings.empty())
      return Fail(error, "result input requires explicit case bindings");
    for (auto binding : input.bindings) {
      const auto it = cases.find(binding.case_id);
      if (it == cases.end() || binding.candidate_id.empty())
        return Fail(error,
                    "result binding has unknown case or empty candidate");
      auto &item = result.cases[it->second];
      if (binding.input_policy_digest != item.definition.input_policy_digest)
        return Fail(error, "result binding input policy mismatch");
      if (!item.definition.benchmark_symbol.empty()) {
        if (!binding.benchmark.empty() &&
            binding.benchmark != item.definition.benchmark_symbol)
          return Fail(error,
                      "result binding benchmark disagrees with manifest case");
        binding.benchmark = item.definition.benchmark_symbol;
      }
      ReportMeasurement measurement;
      measurement.result_id = input.result_id;
      measurement.candidate_id = binding.candidate_id;
      measurement.producer_identity = input.producer_identity;
      measurement.exclusive_device = input.exclusive_device;
      measurement.device_json = Raw(snapshot.Get("device"));
      const auto *device = snapshot.Get("device");
      measurement.device_identity_available =
          !String(Field(device, "driver")).empty() &&
          !String(Field(device, "device_id")).empty();
      std::string physical_key;
      const bool interleaved = snapshot.Get("repetitions") != nullptr;
      const auto *failures = Array(snapshot.Get("failures"));
      const bool early_failure = failures && !failures->empty() &&
                                 !snapshot.Get("benchmarks") && !interleaved;
      if (early_failure) {
        measurement.state = "run_failed";
        measurement.reason = "unattributed_native_run_failure";
        physical_key = "failed:" + binding.case_id;
      } else if (!(interleaved ? Interleaved(snapshot, binding, options,
                                             &measurement, &physical_key, error)
                               : Ordinary(snapshot, binding, options,
                                          &measurement, &physical_key, error)))
        return false;
      physical_key = input.result_id + ":" + physical_key;
      // Exact exported cases already deduplicate equivalent execution. Fixture
      // policy or implementation equality alone cannot prove scalar/count/ABI
      // equivalence across different cases. Unattributed failures have no
      // physical measurement to alias in the first place.
      if (!early_failure) {
        const auto existing =
            physical_policies.emplace(physical_key, binding.case_id);
        if (!existing.second && existing.first->second != binding.case_id)
          return Fail(error,
                      "physical result aliases incompatible case identities");
      }
      const auto bound = std::make_pair(input.result_id, binding.case_id);
      const auto prior_binding = bound_cases.emplace(bound, physical_key);
      if (!prior_binding.second) {
        if (prior_binding.first->second != physical_key)
          return Fail(error, "conflicting duplicate case binding");
        continue;
      }
      if (measurement.p50_ns &&
          item.definition.measurement_subject == "kernel") {
        const double proxy = *measurement.p50_ns * item.invocation_count;
        if (!std::isfinite(proxy))
          return Fail(error, "standalone work proxy overflow");
        measurement.standalone_work_proxy_ns = proxy;
      }
      item.measurements.push_back(std::move(measurement));
    }
  }
  for (auto &item : result.cases) {
    if (item.measurements.empty()) {
      ReportMeasurement missing;
      missing.state = "missing";
      missing.reason = "missing_measurement";
      item.measurements.push_back(std::move(missing));
    }
    bool measured = false;
    bool qualified = false;
    for (const auto &measurement : item.measurements) {
      measured |= measurement.p50_ns.has_value();
      qualified |= measurement.quality == "usable";
    }
    if (measured)
      result.coverage.measured_invocations += item.invocation_count;
    if (qualified)
      result.coverage.qualified_invocations += item.invocation_count;
  }
  if (!inventory.sequence_occurrences.empty()) {
    if (!AddSequencePartitionReport(inventory, {}, &result, error))
      return false;
    result.sequence_partition.reset();
  }
  *report = std::move(result);
  if (error)
    error->clear();
  return true;
}

std::string BenchmarkReportJson(const BenchmarkReport &report) {
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\"manifest_id\":" << Quote(report.manifest_id)
      << ",\"workload_id\":" << Quote(report.workload_id)
      << ",\"phase\":" << Quote(report.phase)
      << ",\"coverage_unit\":\"invocation_count\",\"coverage\":{\"selected_invocations\":"
      << report.coverage.selected_invocations
      << ",\"runnable_invocations\":" << report.coverage.runnable_invocations
      << ",\"measured_invocations\":" << report.coverage.measured_invocations
      << ",\"qualified_invocations\":" << report.coverage.qualified_invocations
      << "},\"quality_policy\":{\"minimum_windows\":"
      << report.quality_policy.minimum_windows << ",\"maximum_window_spread\":"
      << report.quality_policy.maximum_window_spread
      << "},\"approximate_numerical_phase_ns\":";
  if (report.sequence_partition && report.sequence_partition->complete)
    Optional(out, report.sequence_partition->partition_total_ns);
  else
    out << "null";
  out << ','
      << "\"limits\":[\"standalone work proxies are not additive model or GPU latency\","
      << "\"sums of bundle medians are not model latency quantiles\","
      << "\"ring reuse does not establish cold or streaming behavior\"],\"cases\":[";
  bool first = true;
  for (const auto &item : report.cases) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"case_id\":" << Quote(item.definition.case_id)
        << ",\"implementation_id\":" << Quote(item.definition.implementation_id)
        << ",\"benchmark_symbol\":" << Quote(item.definition.benchmark_symbol)
        << ",\"measurement_subject\":"
        << Quote(item.definition.measurement_subject)
        << ",\"reset_policy\":" << Quote(item.definition.reset_policy)
        << ",\"arithmetic_check\":" << Quote(item.definition.arithmetic_check)
        << ",\"invocation_count\":" << item.invocation_count
        << ",\"measurements\":[";
    bool first_measurement = true;
    for (const auto &measurement : item.measurements) {
      if (!first_measurement)
        out << ',';
      first_measurement = false;
      out << "{\"result_id\":" << Quote(measurement.result_id)
          << ",\"candidate_id\":" << Quote(measurement.candidate_id)
          << ",\"state\":" << Quote(measurement.state)
          << ",\"correctness_state\":" << Quote(measurement.correctness_state)
          << ",\"quality\":" << Quote(measurement.quality)
          << ",\"reason\":" << Quote(measurement.reason)
          << ",\"benchmark_identity_state\":"
          << Quote(measurement.benchmark_identity_state) << ",\"p50_ns\":";
      Optional(out, measurement.p50_ns);
      out << ",\"standalone_work_proxy_ns\":";
      Optional(out, measurement.standalone_work_proxy_ns);
      out << ",\"window_spread\":";
      Optional(out, measurement.window_spread);
      out << ",\"window_p50_ns\":[";
      for (size_t i = 0; i < measurement.window_p50_ns.size(); ++i) {
        if (i)
          out << ',';
        out << measurement.window_p50_ns[i];
      }
      out << "],\"logical_operations_per_batch\":";
      Optional(out, measurement.logical_operations_per_batch);
      out << ",\"physical_dispatches_per_logical_operation\":";
      Optional(out, measurement.physical_dispatches_per_logical_operation);
      out << ",\"producer_identity\":" << Quote(measurement.producer_identity)
          << ",\"exclusive_device\":"
          << (measurement.exclusive_device ? "true" : "false") << ",\"device\":"
          << (measurement.device_json.empty() ? "null"
                                              : measurement.device_json)
          << ",\"data_cache\":"
          << (measurement.data_cache_json.empty() ? "null"
                                                  : measurement.data_cache_json)
          << ",\"timing_warnings\":"
          << (measurement.timing_warnings_json.empty()
                  ? "null"
                  : measurement.timing_warnings_json)
          << ",\"stop_reason\":" << Quote(measurement.stop_reason)
          << ",\"native_windows\":[";
      for (size_t i = 0; i < measurement.native_windows_json.size(); ++i) {
        if (i)
          out << ',';
        out << measurement.native_windows_json[i];
      }
      out << "]}";
    }
    out << "]}";
  }
  out << "],\"unsupported_invocations\":[";
  for (size_t i = 0; i < report.unsupported_invocations.size(); ++i) {
    if (i)
      out << ',';
    out << "{\"invocation_id\":"
        << Quote(report.unsupported_invocations[i].invocation_id)
        << ",\"reason\":"
        << Quote(report.unsupported_invocations[i].unsupported_reason) << '}';
  }
  out << "],\"run_diagnostics\":[";
  for (size_t i = 0; i < report.run_diagnostics.size(); ++i) {
    if (i)
      out << ',';
    out << Quote(report.run_diagnostics[i]);
  }
  out << ']';
  if (report.sequence_partition) {
    const auto &partition = *report.sequence_partition;
    out << ",\"sequence_partition\":{\"complete\":"
        << (partition.complete ? "true" : "false")
        << ",\"total_selected_commands\":" << partition.total_selected_commands
        << ",\"selected_commands\":" << partition.selected_commands
        << ",\"qualified_commands\":" << partition.qualified_commands
        << ",\"qualified_dispatch_origins\":"
        << partition.qualified_dispatch_origins
        << ",\"qualified_bundle_subtotal_ns\":";
    Optional(out, partition.qualified_bundle_subtotal_ns);
    out << ",\"partition_total_ns\":";
    Optional(out, partition.partition_total_ns);
    out << ",\"occurrences\":[";
    for (size_t i = 0; i < partition.occurrences.size(); ++i) {
      if (i)
        out << ',';
      const auto &item = partition.occurrences[i];
      out << "{\"occurrence_id\":" << Quote(item.occurrence_id)
          << ",\"case_id\":" << Quote(item.case_id)
          << ",\"result_id\":" << Quote(item.result_id)
          << ",\"contribution_ns\":";
      Optional(out, item.contribution_ns);
      out << ",\"reason\":" << Quote(item.reason) << '}';
    }
    out << "],\"uncovered_commands\":[";
    for (size_t i = 0; i < partition.uncovered_commands.size(); ++i) {
      if (i)
        out << ',';
      const auto &gap = partition.uncovered_commands[i];
      out << "{\"program_ordinal\":" << gap.program_ordinal
          << ",\"command_begin\":" << gap.command_begin
          << ",\"command_end\":" << gap.command_end << '}';
    }
    out << "]}";
  }
  out << "}\n";
  return out.str();
}

} // namespace hrx::loomify_gguf::benchmark
