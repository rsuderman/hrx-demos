// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Test-only native SDK consumer. Generated code, its real helper closure and
// serialized SOURCE/bytecode all execute here; Jinja itself runs independently.
#include "lib/loomify/text/jinja/compiler.h"
#include "lib/loomify/linking.h"
#include "iree/base/internal/json.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/sync.h"
#include "iree/vm/buffer.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/json.h"
#include "loom/runtime/program/program.h"
#include "loomc/target/vm.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hrx::loomify_gguf::jinja::testing {
namespace {
iree_string_view_t View(std::string_view value) {
  return iree_make_string_view(value.data(), value.size());
}
std::string Copy(iree_string_view_t value) {
  return value.size ? std::string(value.data, value.size) : std::string{};
}
std::string StatusText(iree_status_t status) {
  char text[4096] = {};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(text), text, &length);
  iree_status_ignore(status);
  return text;
}
bool Ok(iree_status_t status) {
  if (iree_status_is_ok(status))
    return true;
  ADD_FAILURE() << StatusText(status);
  return false;
}
bool CompilerOk(loomc_status_t status) {
  if (loomc_status_is_ok(status)) {
    loomc_status_free(status);
    return true;
  }
  const auto message = loomc_status_message(status);
  ADD_FAILURE() << std::string(message.data, message.size);
  loomc_status_free(status);
  return false;
}
bool Read(const std::filesystem::path &path, std::string *out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    ADD_FAILURE() << "cannot read " << path;
    return false;
  }
  out->assign(std::istreambuf_iterator<char>(input), {});
  return !input.bad();
}
std::string RequiredEnv(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    ADD_FAILURE() << "required explicit fixture: " << name;
    return {};
  }
  return value;
}
bool Field(iree_string_view_t object, const char *key,
           iree_string_view_t *out) {
  return Ok(iree_json_try_lookup_object_value(
      object, iree_make_cstring_view(key), out));
}
bool StringField(iree_string_view_t object, const char *key, std::string *out) {
  iree_string_view_t raw{};
  if (!Field(object, key, &raw) || !raw.data)
    return false;
  out->resize(raw.size);
  iree_host_size_t size = 0;
  if (!Ok(iree_json_unescape_string(raw, out->size(), out->data(), &size)))
    return false;
  out->resize(size);
  return true;
}
iree_status_t CollectRows(void *data, iree_host_size_t,
                          iree_json_value_type_t type, iree_string_view_t row) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "row must be an object");
  static_cast<std::vector<iree_string_view_t> *>(data)->push_back(row);
  return iree_ok_status();
}
iree_status_t CollectToken(void *data, iree_host_size_t,
                           iree_string_view_t value) {
  int64_t token = -1;
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &token));
  if (token < 0 || token > std::numeric_limits<int32_t>::max())
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  static_cast<std::vector<int32_t> *>(data)->push_back(
      static_cast<int32_t>(token));
  return iree_ok_status();
}

bool MaskFloat(std::string *text, iree_string_view_t metadata, double *value) {
  std::string prefix, suffix;
  if (!StringField(metadata, "prefix", &prefix) ||
      !StringField(metadata, "suffix", &suffix) || prefix.empty() ||
      suffix.empty())
    return false;
  auto start = text->find(prefix);
  if (start == std::string::npos ||
      text->find(prefix, start + prefix.size()) != std::string::npos)
    return false;
  start += prefix.size();
  const auto end = text->find(suffix, start);
  if (end == std::string::npos)
    return false;
  const auto spelling = text->substr(start, end - start);
  if (!Ok(iree_json_parse_double(View(spelling), value)) ||
      !std::isfinite(*value))
    return false;
  text->replace(start, end - start, "[deferred floating spelling]");
  return true;
}

std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char byte : value) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

bool RecordDeferred(iree_string_view_t row, bool bytecode,
                    std::string_view actual, const std::vector<int32_t> &ids) {
  iree_string_view_t id{};
  if (!Field(row, "id", &id) || !id.data)
    return false;
  // The ID is copied as its original escaped JSON string, never interpolated.
  std::ofstream output(RequiredEnv("LOOMIFY_JINJA_DEFERRED_OUTPUT"),
                       std::ios::app);
  output << "{\"id\":\"" << Copy(id) << "\",\"form\":\""
         << (bytecode ? "bytecode" : "source") << "\",\"actual_text_hex\":\""
         << Hex(actual) << "\",\"token_ids\":[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i)
      output << ',';
    output << ids[i];
  }
  output << "]}\n";
  return static_cast<bool>(output);
}

class Evaluator {
public:
  ~Evaluator() {
    loom_runtime_program_free(program_);
    iree_vm_module_release(input_);
    iree_vm_module_release(json_);
    iree_vm_environment_free(environment_);
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    loomc_target_environment_release(target_);
  }
  bool Init(std::string_view text, bool bytecode, size_t ordinal) {
    GeneratedInputRenderer generated;
    Diagnostic diagnostic;
    if (!CompileChatTemplate(text, "oracle-template.jinja", &generated,
                             &diagnostic)) {
      ADD_FAILURE() << diagnostic.source_name << ':' << diagnostic.line << ':'
                    << diagnostic.column << ": " << diagnostic.message;
      return false;
    }
    if (!CompilerOk(loomc_target_environment_create_vm(loomc_allocator_system(),
                                                       &target_)))
      return false;
    loomc_context_target_options_t target_options{};
    target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.target_environment = target_;
    loomc_context_options_t context_options{};
    context_options.next = &target_options;
    if (!CompilerOk(loomc_context_create(
            &context_options, loomc_allocator_system(), &context_)) ||
        !CompilerOk(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                           &workspace_)))
      return false;
    const auto runfiles = std::filesystem::path(RequiredEnv("TEST_SRCDIR")) /
                          RequiredEnv("TEST_WORKSPACE");
    std::vector<std::string> helpers;
    for (const auto &helper : generated.helper_requirements)
      helpers.push_back((runfiles / helper).string());
    std::vector<const char *> roots{generated.entrypoint.c_str()};
    loomc_module_t *linked = nullptr;
    std::string error;
    if (!LinkConfiguredProducts(context_, workspace_, generated.source, helpers,
                                roots, false, &linked, &error)) {
      ADD_FAILURE() << error;
      return false;
    }
    loomc_module_serialize_options_t options{};
    options.format =
        bytecode ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
    loomc_source_t *serialized = nullptr;
    const bool serialized_ok = CompilerOk(loomc_module_serialize_to_source(
        linked, &options, loomc_allocator_system(), &serialized));
    loomc_module_release(linked);
    if (!serialized_ok)
      return false;
    const auto path = std::filesystem::path(RequiredEnv("TEST_TMPDIR")) /
                      ((bytecode ? "jinja-bc-" : "jinja-source-") +
                       std::to_string(ordinal) + ".loom");
    const auto bytes = loomc_source_contents(serialized);
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
    output.close();
    loomc_source_release(serialized);
    if (!output) {
      ADD_FAILURE() << "cannot write isolated product " << path;
      return false;
    }
    if (!Ok(iree_vm_environment_allocate(iree_allocator_system(),
                                         &environment_)) ||
        !Ok(iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                          environment_, IREE_SV("vm")),
                                      &types_)) ||
        !Ok(loom_runtime_json_module_create(environment_,
                                            iree_allocator_system(), &json_)) ||
        !Ok(loom_runtime_input_module_create(environment_, nullptr,
                                             iree_allocator_system(), &input_)))
      return false;
    iree_vm_module_t *libraries[] = {json_, input_};
    const iree_string_view_t runtime_root = View(generated.entrypoint);
    loom_runtime_sources_t sources{};
    loom_runtime_compilation_report_t report{};
    const auto compiled = loom_runtime_program_allocate(
        target_, environment_, &sources, View(path.string()), 1, &runtime_root,
        iree_vm_module_span_from_array(libraries), &report,
        iree_allocator_system(), &program_);
    if (!iree_status_is_ok(compiled))
      for (iree_host_size_t i = 0; i < report.diagnostic_count; ++i) {
        const auto &entry = report.diagnostics[i];
        ADD_FAILURE() << std::string(
            entry.message,
            std::min<uint64_t>(entry.message_length, sizeof(entry.message)));
      }
    if (!Ok(compiled))
      return false;
    return Ok(iree_vm_process_lookup_function(
        loom_runtime_program_process(program_), IREE_SV("model"), runtime_root,
        &entry_));
  }
  iree_status_t Evaluate(std::string_view environment, std::string *out) {
    iree_vm_buffer_t *input = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(environment.data(), environment.size()), 1,
        iree_allocator_system(), &input));
    iree_vm_variant_t argument =
        iree_vm_buffer_variant_from_ptr_move(&types_, &input);
    iree_vm_variant_t result = {0};
    auto status =
        iree_vm_invoke(loom_runtime_program_invocation(program_), entry_,
                       iree_vm_variant_span_from_ptr(&argument, 1),
                       iree_vm_variant_span_from_ptr(&result, 1));
    if (iree_status_is_ok(status)) {
      iree_vm_buffer_t *buffer = nullptr;
      status =
          iree_vm_buffer_ptr_from_variant_borrowed(&types_, result, &buffer);
      if (iree_status_is_ok(status)) {
        if (!buffer)
          status =
              iree_make_status(IREE_STATUS_INTERNAL, "null rendered buffer");
        else
          out->assign(
              static_cast<const char *>(iree_vm_buffer_const_data(buffer)),
              iree_vm_buffer_length(buffer));
      }
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_ptr(&result, 1));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_ptr(&argument, 1));
    return status;
  }

private:
  loomc_target_environment_t *target_ = nullptr;
  loomc_context_t *context_ = nullptr;
  loomc_workspace_t *workspace_ = nullptr;
  iree_vm_environment_t *environment_ = nullptr;
  iree_vm_module_t *json_ = nullptr;
  iree_vm_module_t *input_ = nullptr;
  loom_runtime_program_t *program_ = nullptr;
  iree_vm_ref_types_t types_{};
  iree_vm_function_t entry_ = iree_vm_function_null();
};

void CheckReferences(bool bytecode, bool capability_only) {
  std::string references, tokenizer_json, manifest;
  ASSERT_TRUE(Read(RequiredEnv("LOOMIFY_JINJA_REFERENCE"), &references));
  ASSERT_TRUE(Read(RequiredEnv("LOOMIFY_JINJA_TOKENIZER"), &tokenizer_json));
  ASSERT_TRUE(Read(RequiredEnv("LOOMIFY_JINJA_MANIFEST"), &manifest));
  auto expected_count = [&](const char *key, uint64_t *value) {
    iree_string_view_t raw{};
    return Field(View(manifest), key, &raw) && raw.data &&
           Ok(iree_json_parse_uint64(raw, value));
  };
  uint64_t expected_rows = 0, expected_positive = 0, expected_negative = 0,
           expected_unsupported = 0, expected_deferred = 0;
  ASSERT_TRUE(expected_count("rows", &expected_rows));
  ASSERT_TRUE(expected_count("exact_rows", &expected_positive));
  ASSERT_TRUE(expected_count("error_rows", &expected_negative));
  ASSERT_TRUE(expected_count("native_unsupported_rows", &expected_unsupported));
  ASSERT_TRUE(expected_count("deferred_float_rows", &expected_deferred));
  ASSERT_GT(expected_rows, 0u);
  ASSERT_EQ(expected_rows, expected_positive + expected_negative +
                               expected_unsupported + expected_deferred);
  std::vector<iree_string_view_t> rows;
  ASSERT_TRUE(Ok(
      iree_json_enumerate_array_typed(View(references), CollectRows, &rows)));
  ASSERT_EQ(rows.size(), expected_rows);
  iree_tokenizer_t *raw_tokenizer = nullptr;
  ASSERT_TRUE(Ok(iree_tokenizer_from_huggingface_json(
      View(tokenizer_json), iree_allocator_system(), &raw_tokenizer)));
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(
      raw_tokenizer, iree_tokenizer_free);
  std::map<std::string, std::unique_ptr<Evaluator>> evaluators;
  size_t positive = 0, negative = 0, unsupported = 0, deferred = 0,
         attempted = 0;
  for (auto row : rows) {
    iree_string_view_t capability{};
    ASSERT_TRUE(Field(row, "native_unsupported", &capability));
    if ((capability.data != nullptr) != capability_only)
      continue;
    ++attempted;
    // Fatal assertions stop only this row. A materialization or execution
    // failure must not prevent the remaining independent cases being attempted.
    const auto check_row = [&]() {
      std::string id, source;
      ASSERT_TRUE(StringField(row, "id", &id));
      SCOPED_TRACE(id);
      ASSERT_TRUE(StringField(row, "source", &source));
      auto &evaluator = evaluators[source];
      if (!evaluator) {
        auto candidate = std::make_unique<Evaluator>();
        ASSERT_TRUE(candidate->Init(source, bytecode, evaluators.size()));
        evaluator = std::move(candidate);
      }
      iree_string_view_t environment{}, error{}, native_error{};
      ASSERT_TRUE(Field(row, "environment", &environment));
      ASSERT_TRUE(Field(row, "error_contains", &error));
      ASSERT_TRUE(Field(row, "native_error_contains", &native_error));
      ASSERT_NE(environment.data, nullptr);
      iree_string_view_t raw_environment{};
      ASSERT_TRUE(Field(row, "environment_source", &raw_environment));
      std::string environment_text = Copy(environment);
      if (raw_environment.data)
        ASSERT_TRUE(StringField(row, "environment_source", &environment_text));
      std::string actual;
      auto status = evaluator->Evaluate(environment_text, &actual);
      if (capability.data) {
        const auto code = iree_status_code(status);
        const auto message = StatusText(status);
        ASSERT_EQ(code, IREE_STATUS_INVALID_ARGUMENT) << message;
        ++unsupported;
        if (native_error.data) {
          std::string required;
          ASSERT_TRUE(StringField(row, "native_error_contains", &required));
          EXPECT_NE(message.find(required), std::string::npos) << message;
        }
        return;
      }
      if (error.data) {
        ASSERT_FALSE(iree_status_is_ok(status));
        const auto message = StatusText(status);
        ++negative;
        // A source raise_exception message is oracle metadata, not a per-model
        // test-ID switch. Undefined/type diagnostic wrappers may differ.
        if (native_error.data) {
          std::string required;
          ASSERT_TRUE(StringField(row, "native_error_contains", &required));
          EXPECT_NE(message.find(required), std::string::npos) << message;
        }
        return;
      }
      ASSERT_TRUE(Ok(status));
      std::string expected;
      ASSERT_TRUE(StringField(row, "text", &expected));
      iree_string_view_t float_metadata{};
      ASSERT_TRUE(Field(row, "deferred_float", &float_metadata));
      if (float_metadata.data) {
        ++deferred;
        std::string masked_actual = actual, masked_expected = expected;
        double a = 0, e = 0;
        ASSERT_TRUE(MaskFloat(&masked_actual, float_metadata, &a));
        ASSERT_TRUE(MaskFloat(&masked_expected, float_metadata, &e));
        EXPECT_EQ(masked_actual, masked_expected);
        EXPECT_EQ(a, e);
        EXPECT_EQ(std::signbit(a), std::signbit(e));
      } else {
        ++positive;
        EXPECT_EQ(actual, expected);
      }
      iree_string_view_t ids{};
      ASSERT_TRUE(Field(row, "token_ids", &ids));
      std::vector<int32_t> expected_ids, actual_ids(actual.size() + 16);
      ASSERT_TRUE(
          Ok(iree_json_enumerate_array(ids, CollectToken, &expected_ids)));
      iree_host_size_t count = 0;
      ASSERT_TRUE(Ok(iree_tokenizer_encode(
          tokenizer.get(), View(actual), IREE_TOKENIZER_ENCODE_FLAG_NONE,
          iree_tokenizer_make_token_output(actual_ids.data(), nullptr, nullptr,
                                           actual_ids.size()),
          iree_allocator_system(), &count)));
      actual_ids.resize(count);
      if (!float_metadata.data || actual == expected)
        EXPECT_EQ(actual_ids, expected_ids);
      if (float_metadata.data)
        ASSERT_TRUE(RecordDeferred(row, bytecode, actual, actual_ids));
    };
    check_row();
  }
  EXPECT_EQ(attempted, capability_only ? expected_unsupported
                                       : expected_positive + expected_negative +
                                             expected_deferred);
  EXPECT_EQ(positive, capability_only ? 0u : expected_positive);
  EXPECT_EQ(negative, capability_only ? 0u : expected_negative);
  EXPECT_EQ(unsupported, capability_only ? expected_unsupported : 0u);
  EXPECT_EQ(deferred, capability_only ? 0u : expected_deferred);
}
class JinjaExecutionTest : public ::testing::TestWithParam<bool> {};
TEST_P(JinjaExecutionTest, IndependentOriginalTemplateAndSemanticCases) {
  CheckReferences(GetParam(), false);
}
TEST_P(JinjaExecutionTest, RejectsExplicitUnsupportedCapabilities) {
  CheckReferences(GetParam(), true);
}
INSTANTIATE_TEST_SUITE_P(SerializedForms, JinjaExecutionTest,
                         ::testing::Values(false, true));

TEST(JinjaCompilerTest, RejectsUnsupportedSyntaxAndPreservesOutput) {
  for (
      const char *source :
      {"{% include 'external' %}",
       "{{ unknown_call() }}",
       "{% if false %}{{ unknown_call() }}{% endif %}",
       "{{ 001 }}",
       "{% macro recur() %}{{ recur() }}{% endmacro %}{{ recur() }}",
       "{% macro f(x) %}{{ x }}{% endmacro %}{{ f(1, 2) }}",
       "{% macro f(x) %}{{ x }}{% endmacro %}{{ f(unknown=1) }}",
       "{{ value.items()|tojson }}",
       "{% for k,v in {'x': value.items()}|items %}{{ v|tojson }}{% endfor %}",
       "{{ ((1,2)+[3])|tojson }}",
       "{% if flag %}{% set value=(1,2) %}{% else %}{% set value=[1,2] %}{% endif %}{{ (value+[3])|tojson }}",
       "{% macro append(value) %}{{ (value+[3])|tojson }}{% endmacro %}{{ append((1,2)) }}",
       "{% set state=namespace(value=(1,2)) %}{{ state.value|length }}",
       "{{ value|list(1) }}",
       "{{ value|join(',',1) }}",
       "{{ value|join(attribute='x') }}",
       "{{ value|join(unknown='x') }}",
       "{% generation unexpected %}x{% endgeneration %}",
       "{% for pair in value.items() %}{{ (pair+[1])|tojson }}{% endfor %}",
       "{% set state = namespace(value=0) %}{{ [state]|tojson }}",
       "{{ value.split() }}",
       "{{ value.split(',', 1) }}",
       "{{ value.split(unknown=',') }}",
       "{{ values|first(1) }}",
       "{{ missing|default('x', true, 'extra') }}",
       "{{ missing|default(unknown='x') }}",
       "{{ missing|default(boolean=true) }}",
       "{{ value.get('x', 'fallback', 'extra') }}",
       "{% set state = namespace('positional') %}",
       "{% set state = namespace(value=1, value=2) %}",
       "{{ raise_exception(unexpected='x') }}",
       "{{ strftime_now(unexpected='%d %b %Y') }}",
       "{{ value|tojson(unsupported=4) }}",
       "{% if flag %}missing end",
       "{{ 'unterminated }}"}) {
    SCOPED_TRACE(source);
    GeneratedInputRenderer output{"sentinel", "entry", {"helper"}};
    Diagnostic diagnostic;
    EXPECT_FALSE(
        CompileChatTemplate(source, "negative.jinja", &output, &diagnostic));
    EXPECT_EQ(output.source, "sentinel");
    EXPECT_EQ(output.entrypoint, "entry");
    EXPECT_EQ(output.helper_requirements, std::vector<std::string>{"helper"});
    EXPECT_EQ(diagnostic.source_name, "negative.jinja");
    EXPECT_FALSE(diagnostic.message.empty());
    EXPECT_GE(diagnostic.line, 1u);
    EXPECT_GE(diagnostic.column, 1u);
    EXPECT_LE(diagnostic.offset, std::string_view(source).size());
  }
}
TEST(JinjaCompilerTest, SourceLocationUsesOriginalTemplateBytes) {
  GeneratedInputRenderer output;
  Diagnostic diagnostic;
  EXPECT_FALSE(CompileChatTemplate("literal\n{{ unknown_call() }}",
                                   "location.jinja", &output, &diagnostic));
  EXPECT_EQ(diagnostic.source_name, "location.jinja");
  EXPECT_EQ(diagnostic.line, 2u);
  EXPECT_GT(diagnostic.column, 1u);
}
} // namespace
} // namespace hrx::loomify_gguf::jinja::testing
