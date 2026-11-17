// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/report_io.h"
#include "iree/base/internal/json.h"
#include "loom/util/digest.h"
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

bool Decode(iree_string_view_t value, std::string *out) {
  out->resize(value.size);
  iree_host_size_t size = 0;
  if (!StatusOk(
          iree_json_unescape_string(value, out->size(), out->data(), &size)))
    return false;
  out->resize(size);
  return true;
}

struct Value {
  iree_json_value_type_t type;
  iree_string_view_t bytes;
};
using Object = std::map<std::string, Value>;

bool ReadObject(iree_string_view_t bytes, Object *out) {
  const auto visitor = [](void *user, iree_string_view_t key,
                          iree_json_value_type_t type,
                          iree_string_view_t value) -> iree_status_t {
    std::string name;
    if (!Decode(key, &name) ||
        !static_cast<Object *>(user)->emplace(name, Value{type, value}).second)
      return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    return iree_ok_status();
  };
  return StatusOk(iree_json_enumerate_object_typed(bytes, visitor, out));
}

bool Document(std::string_view json, Object *out, std::string *error) {
  auto cursor = iree_make_string_view(json.data(), json.size());
  iree_string_view_t bytes{};
  if (!StatusOk(iree_json_consume_insignificant(&cursor)) || !cursor.size ||
      cursor.data[0] != '{' ||
      !StatusOk(iree_json_consume_object(&cursor, &bytes)) ||
      !StatusOk(iree_json_consume_insignificant(&cursor)) || cursor.size ||
      !ReadObject(bytes, out))
    return Fail(error, "expected one JSON object without duplicate fields");
  return true;
}

const Value *Field(const Object &object, const char *key) {
  const auto it = object.find(key);
  return it == object.end() ? nullptr : &it->second;
}

bool Text(const Object &object, const char *key, bool required,
          std::string *out, std::string *error) {
  const auto *value = Field(object, key);
  if (!value)
    return required ? Fail(error, std::string("missing field: ") + key) : true;
  if (value->type != IREE_JSON_VALUE_TYPE_STRING ||
      !Decode(value->bytes, out) || (required && out->empty()))
    return Fail(error, std::string("expected ") +
                           (required ? "nonempty " : "") +
                           "string field: " + key);
  return true;
}

bool Objects(const Object &object, const char *key, std::vector<Object> *out,
             std::string *error) {
  const auto *value = Field(object, key);
  if (!value || value->type != IREE_JSON_VALUE_TYPE_ARRAY)
    return Fail(error, std::string("expected array field: ") + key);
  const auto visitor = [](void *user, iree_host_size_t,
                          iree_json_value_type_t type,
                          iree_string_view_t bytes) -> iree_status_t {
    Object object;
    if (type != IREE_JSON_VALUE_TYPE_OBJECT || !ReadObject(bytes, &object))
      return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    static_cast<std::vector<Object> *>(user)->push_back(std::move(object));
    return iree_ok_status();
  };
  if (!StatusOk(iree_json_enumerate_array_typed(value->bytes, visitor, out)))
    return Fail(error, std::string("invalid object array: ") + key);
  return true;
}

bool Coordinate(const Object &object, const char *key,
                std::optional<uint64_t> *out, std::string *error) {
  const auto *value = Field(object, key);
  if (!value)
    return true;
  uint64_t number = 0;
  if (value->type != IREE_JSON_VALUE_TYPE_NUMBER ||
      !StatusOk(iree_json_parse_uint64(value->bytes, &number)))
    return Fail(error, std::string("expected unsigned integer field: ") + key);
  *out = number;
  return true;
}

bool Binding(const Object &object, ResultBinding *out, std::string *error) {
  return Text(object, "case_id", true, &out->case_id, error) &&
         Text(object, "input_policy_digest", true, &out->input_policy_digest,
              error) &&
         Text(object, "candidate_id", true, &out->candidate_id, error) &&
         Text(object, "benchmark", false, &out->benchmark, error) &&
         Text(object, "comparison_group", false, &out->comparison_group,
              error) &&
         Coordinate(object, "benchmark_sample_index",
                    &out->benchmark_sample_index, error) &&
         Coordinate(object, "configuration_ordinal",
                    &out->configuration_ordinal, error) &&
         Coordinate(object, "trial_index", &out->trial_index, error) &&
         Coordinate(object, "trial_ordinal", &out->trial_ordinal, error);
}

bool Read(const std::filesystem::path &path, std::string *out,
          std::string *error) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return Fail(error, "cannot open report input: " + path.string());
  std::ostringstream bytes;
  bytes << input.rdbuf();
  if (input.bad())
    return Fail(error, "cannot read report input: " + path.string());
  *out = bytes.str();
  return true;
}

std::string Sha256(std::string_view bytes) {
  loom_sha256_t digest;
  loom_sha256_compute(iree_make_const_byte_span(bytes.data(), bytes.size()),
                      &digest);
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : digest.bytes) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}

bool ReadResults(const Object &index, const ReportInventory &inventory,
                 const std::filesystem::path &directory,
                 std::vector<ReportResultInput> *out, std::string *error) {
  std::vector<Object> results;
  if (!Objects(index, "results", &results, error))
    return false;
  for (const auto &result : results) {
    std::string file;
    std::string sha256;
    ReportResultInput input;
    input.manifest_id = inventory.manifest_id;
    if (!Text(result, "file", true, &file, error) ||
        !Text(result, "sha256", true, &sha256, error) ||
        !Text(result, "producer_identity", false, &input.producer_identity,
              error))
      return false;
    if (const auto *exclusive = Field(result, "exclusive_device")) {
      if (exclusive->type != IREE_JSON_VALUE_TYPE_TRUE &&
          exclusive->type != IREE_JSON_VALUE_TYPE_FALSE)
        return Fail(error, "exclusive_device must be a boolean declaration");
      input.exclusive_device = exclusive->type == IREE_JSON_VALUE_TYPE_TRUE;
    }
    std::filesystem::path path(file);
    if (path.is_relative())
      path = directory / path;
    if (!Read(path, &input.snapshot_json, error))
      return false;
    if (sha256 != Sha256(input.snapshot_json))
      return Fail(error, "result SHA-256 mismatch: " + file);
    input.result_id = "sha256:" + sha256;
    std::vector<Object> bindings;
    if (!Objects(result, "bindings", &bindings, error))
      return false;
    for (const auto &binding : bindings) {
      ResultBinding parsed;
      if (!Binding(binding, &parsed, error))
        return false;
      input.bindings.push_back(std::move(parsed));
    }
    out->push_back(std::move(input));
  }
  return true;
}

} // namespace

bool ParseBenchmarkReportManifest(std::string_view manifest_json,
                                  ReportInventory *inventory,
                                  std::string *error) {
  if (!inventory)
    return Fail(error, "report inventory output is null");
  Object manifest;
  std::string format;
  ReportInventory parsed;
  if (!Document(manifest_json, &manifest, error) ||
      !Text(manifest, "format", true, &format, error))
    return false;
  if (format != "loomify.benchmark.bundle.v1")
    return Fail(error, "unsupported benchmark manifest format");
  if (!Text(manifest, "manifest_id", true, &parsed.manifest_id, error) ||
      !Text(manifest, "workload_id", true, &parsed.workload_id, error) ||
      !Text(manifest, "phase", true, &parsed.phase, error))
    return false;
  std::vector<Object> cases;
  std::vector<Object> invocations;
  if (!Objects(manifest, "cases", &cases, error) ||
      !Objects(manifest, "invocations", &invocations, error))
    return false;
  for (const auto &item : cases) {
    ReportCase result;
    if (!Text(item, "case_id", true, &result.case_id, error) ||
        !Text(item, "benchmark_symbol", true, &result.benchmark_symbol,
              error) ||
        !Text(item, "implementation_id", true, &result.implementation_id,
              error) ||
        !Text(item, "input_policy_digest", true, &result.input_policy_digest,
              error) ||
        !Text(item, "measurement_subject", true, &result.measurement_subject,
              error) ||
        !Text(item, "arithmetic_check", false, &result.arithmetic_check, error))
      return false;
    parsed.cases.push_back(std::move(result));
  }
  for (const auto &item : invocations) {
    ReportInvocation result;
    if (!Text(item, "invocation_id", true, &result.invocation_id, error) ||
        !Text(item, "case_id", false, &result.case_id, error) ||
        !Text(item, "unsupported_reason", false, &result.unsupported_reason,
              error))
      return false;
    parsed.invocations.push_back(std::move(result));
  }
  // Reuse the report's existing inventory/alias/coverage validation instead of
  // defining a second set of case/origin rules in this IO adapter.
  BenchmarkReport checked;
  if (!BuildBenchmarkReport(parsed, {}, {}, &checked, error))
    return false;
  *inventory = std::move(parsed);
  return true;
}

bool BuildBenchmarkReportFromFiles(const std::filesystem::path &manifest_path,
                                   const std::filesystem::path &index_path,
                                   const ReportOptions &options,
                                   BenchmarkReport *report,
                                   std::string *error) {
  std::string manifest_json;
  std::string index_json;
  if (!report)
    return Fail(error, "report output is null");
  if (!Read(manifest_path, &manifest_json, error) ||
      !Read(index_path, &index_json, error))
    return false;
  ReportInventory inventory;
  Object index;
  std::string format;
  std::string manifest_id;
  if (!ParseBenchmarkReportManifest(manifest_json, &inventory, error) ||
      !Document(index_json, &index, error) ||
      !Text(index, "format", true, &format, error) ||
      !Text(index, "manifest_id", true, &manifest_id, error))
    return false;
  if (format != "loomify.benchmark.result-index.v1")
    return Fail(error, "unsupported benchmark result-index format");
  if (manifest_id != inventory.manifest_id)
    return Fail(error, "result index manifest identity mismatch");
  std::vector<ReportResultInput> inputs;
  if (!ReadResults(index, inventory, index_path.parent_path(), &inputs, error))
    return false;
  return BuildBenchmarkReport(inventory, inputs, options, report, error);
}

} // namespace hrx::loomify_gguf::benchmark
