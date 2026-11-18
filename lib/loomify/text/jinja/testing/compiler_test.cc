// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Test-only native SDK consumer. Generated code, its real helper closure and
// serialized SOURCE/bytecode all execute here; Jinja itself runs independently.
#include "lib/loomify/text/jinja/compiler.h"
#include "lib/loomify/template_probe_runner.h"
#include "iree/base/internal/json.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
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
  bool Init(std::string_view text, bool bytecode, size_t) {
    GeneratedInputRenderer generated;
    Diagnostic diagnostic;
    if (!CompileChatTemplate(text, "oracle-template.jinja", &generated,
                             &diagnostic)) {
      ADD_FAILURE() << diagnostic.source_name << ':' << diagnostic.line << ':'
                    << diagnostic.column << ": " << diagnostic.message;
      return false;
    }
    const auto runfiles = std::filesystem::path(RequiredEnv("TEST_SRCDIR")) /
                          RequiredEnv("TEST_WORKSPACE");
    std::string error;
    if (!PreparedTemplate::Prepare(generated, runfiles.string(),
                                   RequiredEnv("TEST_TMPDIR"), bytecode,
                                   &prepared_, &error)) {
      ADD_FAILURE() << error;
      return false;
    }
    return true;
  }
  iree_status_t Evaluate(std::string_view environment, std::string *out) {
    std::string error;
    int32_t native_status_code = 0;
    if (prepared_->Render(environment, out, &error, &native_status_code))
      return iree_ok_status();
    return iree_make_status(static_cast<iree_status_code_t>(native_status_code),
                            "%s", error.c_str());
  }

private:
  std::unique_ptr<PreparedTemplate> prepared_;
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

TEST(JinjaCompilerTest, RejectsUnprovedMacroNamespaceShadowing) {
  for (
      const char *body : {
          "{% if flag %}{% set ns=namespace(value='local') %}{% endif %}{{ ns.value }}",
          "{{ ns.value }}{% if flag %}{% set ns=namespace(value='local') %}{% endif %}",
          "{% if flag %}{% set ns=namespace(value='local') %}{% else %}{{ ns.value }}{% endif %}",
          "{% if flag %}{% set ns=namespace(value='local') %}{% set alias=ns %}{% endif %}{% set alias.value='escaped' %}",
          "{% if flag %}{% set ns=namespace(value='one') %}{% else %}{% set ns=namespace(value='two') %}{% endif %}{{ ns.value }}",
          "{% if flag %}{% set ns=namespace(value=ns.value) %}{{ ns.value }}{% endif %}",
          "{% for item in [1] %}{% set ns=namespace(value=item) %}{% endfor %}{% if flag %}{% set ns=namespace(value='other') %}{% endif %}{{ ns.value }}",
      }) {
    const std::string source =
        "{% set ns=namespace(value='outer') %}{% macro render() %}" +
        std::string(body) + "{% endmacro %}{{ render() }}";
    SCOPED_TRACE(source);
    GeneratedInputRenderer output;
    output.source = "preserved";
    Diagnostic diagnostic;
    EXPECT_FALSE(
        CompileChatTemplate(source, "shadow.jinja", &output, &diagnostic));
    EXPECT_EQ(output.source, "preserved");
    EXPECT_NE(diagnostic.message.find("ambiguous identity"), std::string::npos);
    EXPECT_GT(diagnostic.offset, 0u);
  }
}

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
       "{{ value.replace('a') }}",
       "{{ value.replace(old='a', new='b') }}",
       "{{ value|replace('a') }}",
       "{{ value|replace('a','b',new='c') }}",
       "{{ value|replace('a','b',unknown=1) }}",
       "{{ value|reject('equalto',1)|join(',',1) }}",
       "{{ value|reject('equalto',1)|join(attribute='x') }}",
       "{{ value|reject('equalto',1)|list(1) }}",
       "{{ value|reject('odd')|list }}",
       "{{ value|reject('equalto',value=1)|list }}",
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
       "{{ value|tojson(ensure_ascii=True) }}",
       "{{ value|tojson(ensure_ascii=0) }}",
       "{{ value|tojson(ensure_ascii='false') }}",
       "{{ value|tojson(ensure_ascii=flag) }}",
       "{{ value|tojson(indent=width) }}",
       "{{ value|tojson(indent='  ') }}",
       "{{ value|tojson(indent=0) }}",
       "{{ value|tojson(indent=True) }}",
       "{{ value|tojson(2) }}",
       "{{ value|tojson(4) }}",
       "{{ value|tojson(2, indent=2) }}",
       "{{ value|tojson(2, 4) }}",
       "{{ value|tojson(False, 2, indent=2) }}",
       "{{ value|tojson(False, 2, 4) }}",
       "{{ value|tojson(indent=2, separators=(',', ':')) }}",
       "{{ value|tojson(separators=(',', ':'), indent=4) }}",
       "{{ value|tojson(separators=(',', '-'), ensure_ascii=False) }}",
       "{{ value|tojson(ensure_ascii=False, unsupported=2) }}",
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
TEST(JinjaCompilerTest, RejectIteratorValuesRemainUnsupported) {
  for (const char *source :
       {"{{ values|reject('equalto',1) }}",
        "{% set result=values|reject('equalto',1) %}{{ result|list }}",
        "{% if values|reject('equalto',1) %}yes{% endif %}",
        "{{ values|reject('equalto',1)|tojson }}"}) {
    SCOPED_TRACE(source);
    GeneratedInputRenderer output;
    Diagnostic diagnostic;
    EXPECT_FALSE(
        CompileChatTemplate(source, "reject.jinja", &output, &diagnostic));
    EXPECT_EQ(
        diagnostic.message,
        "reject iterator requires an immediate join or explicit list consumer");
    EXPECT_EQ(diagnostic.source_name, "reject.jinja");
    EXPECT_GT(diagnostic.offset, 0u);
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

TEST(JinjaCompilerTest, FilteredMetadataAndMalformedCapturesDiagnose) {
  for (
      const auto *source :
      {"{% for x in values if x %}{{ loop.index }}{% endfor %}",
       "{% for x in values if x %}{% set alias=loop %}{{ alias.index }}{% endfor %}",
       "{% for x in values if x %}{% set captured %}{{ loop.last }}{% endset %}{% endfor %}",
       "{% macro read(value) %}{{ value.index }}{% endmacro %}{% for x in values if x %}{{ read(loop) }}{% endfor %}",
       "{% for x in values if x %}{% for y in values if loop.first %}{{ y }}{% endfor %}{% endfor %}",
       "{% set captured %}missing endset",
       "{% set state.field %}text{% endset %}",
       "{% set captured|trim %}text{% endset %}",
       "{% set loop %}text{% endset %}",
       "{% for x in values if %}{{ x }}{% endfor %}"}) {
    SCOPED_TRACE(source);
    GeneratedInputRenderer output;
    Diagnostic diagnostic;
    EXPECT_FALSE(CompileChatTemplate(source, "syntax-negative.jinja", &output,
                                     &diagnostic));
    EXPECT_EQ(diagnostic.source_name, "syntax-negative.jinja");
    EXPECT_FALSE(diagnostic.message.empty());
  }
}

TEST(JinjaCompilerTest, MalformedLocationUsesOriginalBytesAndKeepsProduct) {
  // Six bytes before line 2, then two UTF-8 bytes and one standalone CR.
  const std::string source = u8"head\r\né\r{{";
  GeneratedInputRenderer output{"sentinel", "entry", {"helper"}};
  Diagnostic diagnostic{"old message", "old source", 999, 99, 99};
  EXPECT_FALSE(
      CompileChatTemplate(source, "malformed.jinja", &output, &diagnostic));
  EXPECT_EQ(output.source, "sentinel");
  EXPECT_EQ(output.entrypoint, "entry");
  EXPECT_EQ(output.helper_requirements, std::vector<std::string>{"helper"});
  EXPECT_EQ(diagnostic.message, "unterminated template tag");
  EXPECT_EQ(diagnostic.source_name, "malformed.jinja");
  EXPECT_EQ(diagnostic.offset, 9u);
  EXPECT_EQ(diagnostic.line, 2u);
  EXPECT_EQ(diagnostic.column, 4u);
  EXPECT_FALSE(CompileChatTemplate(source, "null.jinja", nullptr, &diagnostic));
  EXPECT_EQ(diagnostic.message,
            "chat template compilation needs an output slot");
  EXPECT_EQ(diagnostic.source_name, "null.jinja");
  EXPECT_EQ(diagnostic.offset, 0u);
  EXPECT_EQ(diagnostic.line, 1u);
  EXPECT_EQ(diagnostic.column, 1u);
  EXPECT_FALSE(
      CompileChatTemplate(source, "malformed.jinja", &output, nullptr));
  ASSERT_TRUE(
      CompileChatTemplate("{{ value }}", "valid.jinja", &output, &diagnostic));
  EXPECT_TRUE(diagnostic.message.empty());
  EXPECT_TRUE(diagnostic.source_name.empty());
  EXPECT_EQ(diagnostic.offset, 0u);
  EXPECT_EQ(diagnostic.line, 1u);
  EXPECT_EQ(diagnostic.column, 1u);
}

TEST(JinjaCompilerTest, DiagnosticLocatorClampsAndCanBeAppliedRepeatedly) {
  const std::string source = u8"a\r\né\r";
  Diagnostic diagnostic{"preserved", "old source", 999, 99, 99};
  for (int repeat = 0; repeat < 2; ++repeat) {
    LocateDiagnostic(source, "original.jinja", &diagnostic);
    EXPECT_EQ(diagnostic.message, "preserved");
    EXPECT_EQ(diagnostic.source_name, "original.jinja");
    EXPECT_EQ(diagnostic.offset, 6u);
    EXPECT_EQ(diagnostic.line, 2u);
    EXPECT_EQ(diagnostic.column, 4u);
  }
  LocateDiagnostic(source, "original.jinja", nullptr);
}

TEST(JinjaTemplateDefaultTest, ReadsActualEffortSources) {
  const auto runfiles = std::filesystem::path(RequiredEnv("TEST_SRCDIR")) /
                        RequiredEnv("TEST_WORKSPACE");
  for (const auto &[path, expected] :
       std::vector<std::pair<std::string, std::string>>{
           {"lib/text/qwen35/templates/c3cf9e34.jinja", "xhigh"},
           {"lib/text/spark/templates/93d6aa90.jinja", "high"}}) {
    SCOPED_TRACE(path);
    std::string source, value = "unchanged";
    ASSERT_TRUE(Read(runfiles / path, &source));
    Diagnostic diagnostic;
    ASSERT_TRUE(ReadTemplateStringDefault(source, path, "reasoning_effort",
                                          &value, &diagnostic))
        << diagnostic.message;
    EXPECT_EQ(value, expected);
    EXPECT_TRUE(diagnostic.message.empty());
  }
}

TEST(JinjaTemplateDefaultTest, ReadsLiteralFormsAndIgnoresDestinationName) {
  for (
      const auto &[source, expected] :
      std::vector<std::pair<std::string, std::string>>{
          {"{% set renamed = reasoning_effort|default('low') %}", "low"},
          {"{# comment #}{{ reasoning_effort | d(default_value=\"high\") }}",
           "high"},
          {"{{ reasoning_effort|default('caf\\u00e9') }}", u8"café"},
          {"{{ reasoning_effort|default('low') }}{{ reasoning_effort|d('low') }}",
           "low"}}) {
    SCOPED_TRACE(source);
    std::string value = "sentinel";
    Diagnostic diagnostic{"old diagnostic", "old source"};
    ASSERT_TRUE(ReadTemplateStringDefault(
        source, "default.jinja", "reasoning_effort", &value, &diagnostic))
        << diagnostic.message;
    EXPECT_EQ(value, expected);
    EXPECT_TRUE(diagnostic.message.empty());
  }
}

TEST(JinjaTemplateDefaultTest, RejectsUnavailableFactsAndPreservesOutput) {
  for (
      const char *source :
      {"plain", "{{ reasoning_effort|default(fallback) }}",
       "{{ reasoning_effort|default('low') }}{{ reasoning_effort|default('high') }}",
       "{% set reasoning_effort='local' %}{{ reasoning_effort|default('low') }}",
       "{% for reasoning_effort in values %}{{ reasoning_effort|default('low') }}{% endfor %}",
       "{% macro show(reasoning_effort) %}{{ reasoning_effort|default('low') }}{% endmacro %}",
       "{% macro reasoning_effort() %}x{% endmacro %}{{ reasoning_effort|default('low') }}",
       "{{ reasoning_effort|default('low', true) }}",
       "{{ reasoning_effort|default(default_value='low', boolean=true) }}",
       "{{ reasoning_effort|default(unknown='low') }}",
       "{{ reasoning_effort|default(['low']) }}",
       "{{ reasoning_effort|default('unterminated) }}"}) {
    SCOPED_TRACE(source);
    std::string value = "sentinel";
    Diagnostic diagnostic;
    EXPECT_FALSE(ReadTemplateStringDefault(source, "default-negative.jinja",
                                           "reasoning_effort", &value,
                                           &diagnostic));
    EXPECT_EQ(value, "sentinel");
    EXPECT_EQ(diagnostic.source_name, "default-negative.jinja");
    EXPECT_FALSE(diagnostic.message.empty());
    EXPECT_GE(diagnostic.line, 1u);
    EXPECT_GE(diagnostic.column, 1u);
    EXPECT_LE(diagnostic.offset, std::string_view(source).size());
  }
}

TEST(JinjaTemplateDefaultTest, RejectsNullOutputAndReportsOriginalLocation) {
  Diagnostic diagnostic;
  EXPECT_FALSE(ReadTemplateStringDefault(
      "{{ reasoning_effort|default('low') }}", "null.jinja", "reasoning_effort",
      nullptr, &diagnostic));
  EXPECT_FALSE(diagnostic.message.empty());
  std::string value = "sentinel";
  EXPECT_FALSE(ReadTemplateStringDefault(
      "literal\n{{ reasoning_effort|default(other) }}", "location.jinja",
      "reasoning_effort", &value, &diagnostic));
  EXPECT_EQ(value, "sentinel");
  EXPECT_EQ(diagnostic.source_name, "location.jinja");
  EXPECT_EQ(diagnostic.line, 2u);
  EXPECT_GT(diagnostic.column, 1u);
}
} // namespace
} // namespace hrx::loomify_gguf::jinja::testing
