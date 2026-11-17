// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/format_analysis.h"
#include "lib/loomify/template_probe_runner.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::jinja {
namespace {

std::filesystem::path SourceRoot() {
  if (const char *root = std::getenv("LOOMIFY_FORMAT_SOURCE_ROOT"))
    return root;
  const char *runfiles = std::getenv("TEST_SRCDIR");
  const char *workspace = std::getenv("TEST_WORKSPACE");
  EXPECT_NE(runfiles, nullptr);
  EXPECT_NE(workspace, nullptr);
  return std::filesystem::path(runfiles ? runfiles : "") /
         (workspace ? workspace : "");
}

std::string Quote(std::string_view text) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : text) {
    if (c == '\\' || c == '"') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 32) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += static_cast<char>(c);
    }
  }
  return result + '"';
}

std::string Literal(std::string_view text) {
  return "{{ " + Quote(text) + " }}";
}

std::string Frame(std::string_view body,
                  std::string_view generation = "{{ '<turn>assistant:' }}") {
  return "{% for m in messages %}{% if m.role == 'assistant' %}"
         "{{ '<turn>assistant:' }}" +
         std::string(body) +
         "{{ '</turn>' }}{% endif %}{% endfor %}"
         "{% if add_generation_prompt %}" +
         std::string(generation) + "{% endif %}";
}

std::string JsonCallsEncodedKeys(std::string_view name_key,
                                 std::string_view argument_key,
                                 std::string_view open,
                                 std::string_view close) {
  return "{% if m.tool_calls %}{% for c in m.tool_calls %}" +
         Literal(std::string(open) + "{" + std::string(name_key) + ":") +
         "{{ c.function.name | tojson }}" +
         Literal("," + std::string(argument_key) + ":") +
         "{{ c.function.arguments | tojson }}" +
         Literal("}" + std::string(close)) + "{% endfor %}{% endif %}";
}

std::string JsonCalls(std::string_view name_key = "name",
                      std::string_view argument_key = "arguments",
                      std::string_view open = "<call>",
                      std::string_view close = "</call>") {
  return JsonCallsEncodedKeys(Quote(name_key), Quote(argument_key), open,
                              close);
}

std::string TaggedCalls(bool named, std::string_view value) {
  return "{% if m.tool_calls %}{% for c in m.tool_calls %}" +
         Literal(named ? "<run><action=" : "<run>") + "{{ c.function.name }}" +
         (named ? Literal(">") : "") +
         "{% for k,v in c.function.arguments.items() %}" +
         Literal(named ? "<input=" : "<key>") + "{{ k }}" +
         Literal(named ? ">\n" : "</key><value>") + std::string(value) +
         Literal(named ? "\n</input>" : "</value>") + "{% endfor %}" +
         Literal(named ? "</action></run>" : "</run>") +
         "{% endfor %}{% endif %}";
}

std::unique_ptr<PreparedTemplate>
PrepareProduct(const GeneratedInputRenderer &generated, bool bytecode) {
  const char *scratch = std::getenv("TEST_TMPDIR");
  EXPECT_NE(scratch, nullptr);
  if (!scratch)
    return nullptr;
  std::unique_ptr<PreparedTemplate> result;
  std::string error;
  if (!PreparedTemplate::Prepare(generated, SourceRoot().string(), scratch,
                                 bytecode, &result, &error))
    ADD_FAILURE() << error;
  return result;
}

std::unique_ptr<PreparedTemplate>
PrepareSelection(std::string_view source, const Template &slice, bool bytecode,
                 TemplateSelectorKind kind = TemplateSelectorKind::Format) {
  GeneratedInputRenderer generated;
  Diagnostic diagnostic;
  if (!CompileTemplateSelector(source, "analysis-regression.jinja", slice,
                               &generated, &diagnostic, kind)) {
    ADD_FAILURE() << diagnostic.line << ':' << diagnostic.column << ':'
                  << diagnostic.message;
    return nullptr;
  }
  return PrepareProduct(generated, bytecode);
}

std::string Environment(std::string_view extra = {}) {
  return "{\"messages\":[{\"role\":\"user\",\"content\":\"question\"}],"
         "\"tools\":[],\"add_generation_prompt\":true" +
         std::string(extra) + "}";
}

bool Select(PreparedTemplate &prepared, std::string_view environment,
            int64_t *id, int64_t *mode) {
  std::string error;
  const bool selected = prepared.SelectFormat(environment, id, mode, &error);
  EXPECT_TRUE(selected) << error;
  return selected;
}

std::string WorkerEnvironment(const TextFormatAnalysis &analysis,
                              std::string_view base, std::string_view message,
                              size_t message_index, std::string_view call,
                              size_t call_index) {
  if (base.empty() || base.back() != '}') {
    ADD_FAILURE() << "worker fixture requires a complete request object";
    return {};
  }
  return std::string(base.substr(0, base.size() - 1)) + "," +
         Quote(analysis.adaptation_message_binding) + ":" +
         std::string(message) + "," +
         Quote(analysis.adaptation_message_index_binding) + ":" +
         std::to_string(message_index) + "," +
         Quote(analysis.adaptation_call_binding) + ":" + std::string(call) +
         "," + Quote(analysis.adaptation_call_index_binding) + ":" +
         std::to_string(call_index) + "}";
}

std::string ConditionalArgumentCalls(std::string_view condition) {
  return "{% set message_index = loop.index0 %}"
         "{% if m.tool_calls %}{% for c in m.tool_calls %}"
         "{{ '<call>{\"name\":' }}{{ c.function.name | tojson }}"
         "{{ ',\"arguments\":' }}{% if " +
         std::string(condition) +
         " %}{% for inspected_key, inspected_value in "
         "c.function.arguments.items() %}{% endfor %}"
         "{{ c.function.arguments | tojson }}{% else %}"
         "{% if c.function.arguments is string %}{{ c.function.arguments }}"
         "{% else %}{{ c.function.arguments | tojson }}{% endif %}"
         "{% endif %}{{ '}</call>' }}{% endfor %}{% endif %}";
}

bool HasLiteral(const std::vector<TextFormatPart> &parts,
                std::string_view fragment) {
  for (const auto &part : parts) {
    if ((part.capture == TextCapture::Literal &&
         part.literal.find(fragment) != std::string::npos) ||
        HasLiteral(part.body, fragment) || HasLiteral(part.otherwise, fragment))
      return true;
  }
  return false;
}

bool HasCapture(const std::vector<TextFormatPart> &parts, TextCapture capture) {
  for (const auto &part : parts)
    if (part.capture == capture || HasCapture(part.body, capture) ||
        HasCapture(part.otherwise, capture))
      return true;
  return false;
}

bool RefersTo(const Expr &expression, TextCapture capture,
              const TextFormatAnalysis &analysis) {
  const auto found = analysis.predicate_bindings.find(expression.text);
  if (expression.kind == Expr::Kind::Name &&
      found != analysis.predicate_bindings.end() && found->second == capture)
    return true;
  return std::any_of(
      expression.args.begin(), expression.args.end(),
      [&](const Expr &child) { return RefersTo(child, capture, analysis); });
}

bool HasChoiceFor(const std::vector<TextFormatPart> &parts, TextCapture capture,
                  const TextFormatAnalysis &analysis) {
  for (const auto &part : parts)
    if ((part.capture == TextCapture::Choice &&
         RefersTo(part.predicate, capture, analysis)) ||
        HasChoiceFor(part.body, capture, analysis) ||
        HasChoiceFor(part.otherwise, capture, analysis))
      return true;
  return false;
}

void ExpectJsonFields(const TextFormatAlternative &format,
                      std::string_view name, std::string_view arguments) {
  EXPECT_EQ(format.tools.kind, TextToolKind::JsonEnvelope);
  EXPECT_EQ(format.tools.name_field, name);
  EXPECT_EQ(format.tools.arguments_field, arguments);
  EXPECT_TRUE(format.tools.json_argument_wire.has_value());
}

std::string ReadSource(const char *path) {
  std::ifstream input(SourceRoot() / path);
  EXPECT_TRUE(input) << path;
  return std::string(std::istreambuf_iterator<char>(input), {});
}

TextFormatBindings Globals() {
  TextFormatBindings bindings;
  bindings.strings["bos_token"] = "<explicit-bos>";
  bindings.strings["eos_token"] = "<explicit-eos>";
  return bindings;
}

TEST(TextFormatAnalysisTest, PlainFrameComesFromStructure) {
  const char *source =
      "{% for item in messages %}{{ '<turn>' + item.role + ':' + "
      "item.content + '</turn>' }}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      AnalyzeTextFormat(source, "plain.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  EXPECT_EQ(result.alternatives[0].assistant_header, "<turn>assistant:");
  EXPECT_EQ(result.alternatives[0].frame_suffix, "</turn>");
  EXPECT_EQ(result.alternatives[0].tools.kind, TextToolKind::None);
  EXPECT_FALSE(result.selector_template.body.empty());
}

TEST(TextFormatAnalysisTest, FailurePreservesOwnedResult) {
  TextFormatAnalysis result;
  result.alternatives.push_back({});
  result.alternatives[0].assistant_header = "preserved";
  Diagnostic diagnostic;
  EXPECT_FALSE(AnalyzeTextFormat("{{ unknown }}", "unsupported.jinja", {},
                                 &result, &diagnostic));
  ASSERT_EQ(result.alternatives.size(), 1);
  EXPECT_EQ(result.alternatives[0].assistant_header, "preserved");
  EXPECT_EQ(diagnostic.source_name, "unsupported.jinja");
  EXPECT_FALSE(diagnostic.message.empty());
}

TEST(TextFormatAnalysisTest,
     DirectParseFailuresLocateOriginalBytesAndKeepProducts) {
  // CRLF and the standalone CR retain their original byte positions.
  const std::string source = u8"head\r\né\r{{";
  TextFormatAnalysis formats;
  formats.alternatives.emplace_back().assistant_header = "preserved format";
  TextUserContinuation continuation;
  continuation.text_binding = "preserved input";
  continuation.assistant_closure = "preserved closure";
  for (bool user : {false, true}) {
    SCOPED_TRACE(user ? "USER" : "format");
    Diagnostic diagnostic{"old message", "old source", 999, 99, 99};
    EXPECT_FALSE(user ? AnalyzeUserContinuation(source, "malformed.jinja", {},
                                                formats, &continuation,
                                                &diagnostic)
                      : AnalyzeTextFormat(source, "malformed.jinja", {},
                                          &formats, &diagnostic));
    EXPECT_EQ(diagnostic.message, "unterminated template tag");
    EXPECT_EQ(diagnostic.source_name, "malformed.jinja");
    EXPECT_EQ(diagnostic.offset, 9u);
    EXPECT_EQ(diagnostic.line, 2u);
    EXPECT_EQ(diagnostic.column, 4u);
    ASSERT_EQ(formats.alternatives.size(), 1u);
    EXPECT_EQ(formats.alternatives[0].assistant_header, "preserved format");
    EXPECT_EQ(continuation.text_binding, "preserved input");
    EXPECT_EQ(continuation.assistant_closure, "preserved closure");
  }
  EXPECT_FALSE(
      AnalyzeTextFormat(source, "malformed.jinja", {}, &formats, nullptr));
  EXPECT_FALSE(AnalyzeUserContinuation(source, "malformed.jinja", {}, formats,
                                       &continuation, nullptr));
}

TEST(TextFormatAnalysisTest,
     DirectBoundariesResetStaleErrorsAndClearOnSuccess) {
  const std::string source =
      "{% for m in messages %}"
      "{{ '<turn>' + m.role + ':' + m.content + '</turn>' }}"
      "{% endfor %}{% if add_generation_prompt %}"
      "{{ '<turn>assistant:' }}{% endif %}";
  TextFormatAnalysis formats;
  TextUserContinuation continuation;
  Diagnostic diagnostic{"old message", "old source", 999, 99, 99};
  for (bool user : {false, true}) {
    SCOPED_TRACE(user ? "USER" : "format");
    // Null output wins over the malformed source, with a fresh location.
    diagnostic = {"old message", "old source", 999, 99, 99};
    EXPECT_FALSE(
        user ? AnalyzeUserContinuation("{{", "null.jinja", {}, formats, nullptr,
                                       &diagnostic)
             : AnalyzeTextFormat("{{", "null.jinja", {}, nullptr, &diagnostic));
    EXPECT_EQ(diagnostic.message,
              user ? "continued USER analysis needs an output slot"
                   : "format analysis needs an output slot");
    EXPECT_EQ(diagnostic.source_name, "null.jinja");
    EXPECT_EQ(diagnostic.offset, 0u);
    EXPECT_EQ(diagnostic.line, 1u);
    EXPECT_EQ(diagnostic.column, 1u);
    ASSERT_TRUE(user ? AnalyzeUserContinuation(source, "valid.jinja", {},
                                               formats, &continuation,
                                               &diagnostic)
                     : AnalyzeTextFormat(source, "valid.jinja", {}, &formats,
                                         &diagnostic))
        << diagnostic.message;
    EXPECT_TRUE(diagnostic.message.empty());
    EXPECT_TRUE(diagnostic.source_name.empty());
    EXPECT_EQ(diagnostic.offset, 0u);
    EXPECT_EQ(diagnostic.line, 1u);
    EXPECT_EQ(diagnostic.column, 1u);
  }
}

TEST(TextFormatAnalysisTest, TaggedDescriptionsRetainBoundariesAndCodecs) {
  for (bool named : {false, true}) {
    const auto source = Frame(
        "{{ m.content }}" +
        TaggedCalls(named, named ? "{{ v | tojson }}"
                                 : "{{ v if v is string else v | tojson }}"));
    TextFormatAnalysis result;
    Diagnostic diagnostic;
    ASSERT_TRUE(AnalyzeTextFormat(source, "tagged-fields.jinja", {}, &result,
                                  &diagnostic))
        << diagnostic.message;
    ASSERT_FALSE(result.alternatives.empty());
    for (const auto &format : result.alternatives) {
      EXPECT_EQ(format.tools.kind, named ? TextToolKind::TaggedNamed
                                         : TextToolKind::TaggedKeyValue);
      const auto &tagged = format.tools.tagged;
      EXPECT_EQ(tagged.before_name, named ? "<run><action=" : "<run>");
      EXPECT_EQ(tagged.after_name, named ? ">" : "");
      EXPECT_EQ(tagged.after_arguments, named ? "</action></run>" : "</run>");
      EXPECT_EQ(tagged.before_key, named ? "<input=" : "<key>");
      EXPECT_EQ(tagged.after_key, named ? ">\n" : "</key><value>");
      EXPECT_EQ(tagged.after_value, named ? "\n</input>" : "</value>");
      EXPECT_EQ(tagged.value_codec.wire,
                named ? TextValueWire::Json : TextValueWire::StringElseJson);
      EXPECT_FALSE(tagged.value_codec.trim);
      EXPECT_FALSE(format.tools.call_body.empty());
    }
  }
}

TEST(TextFormatAnalysisTest,
     TaggedDescriptionUsesMergedBodyAndFinalBindingsForEveryGeneration) {
  // FormatChoices visits the second collected body first. Both bodies share
  // the same request route, so Merge must replace its initially absent tools
  // with the first loop's source pattern before finalizing tagged captures.
  const auto source =
      Frame("{{ m.content }}" +
                TaggedCalls(true, "{{ v if v is string else v | tojson }}"),
            "") +
      Frame("{{ m.content }}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "merged-tagged.jinja", {}, &result,
                                &diagnostic))
      << diagnostic.message;
  ASSERT_EQ(result.alternatives.size(), 2);
  EXPECT_TRUE(std::any_of(result.predicate_bindings.begin(),
                          result.predicate_bindings.end(),
                          [](const auto &entry) {
                            return entry.second == TextCapture::ArgumentValue;
                          }));
  bool generation = false, completed = false;
  for (const auto &format : result.alternatives) {
    generation |= !format.generation_prefix.empty();
    completed |= format.generation_prefix.empty();
    EXPECT_EQ(format.tools.kind, TextToolKind::TaggedNamed);
    EXPECT_TRUE(HasChoiceFor(format.tools.call_body, TextCapture::ArgumentValue,
                             result));
    EXPECT_EQ(format.tools.tagged.before_name, "<run><action=");
    EXPECT_EQ(format.tools.tagged.after_arguments, "</action></run>");
    EXPECT_EQ(format.tools.tagged.after_key, ">\n");
    EXPECT_EQ(format.tools.tagged.value_codec.wire,
              TextValueWire::StringElseJson);
    EXPECT_FALSE(format.tools.tagged.value_codec.trim);
  }
  EXPECT_TRUE(generation);
  EXPECT_TRUE(completed);
}

TEST(TextFormatAnalysisTest,
     JsonDescriptionUsesMergedBodyAndWholeChoiceForEveryGeneration) {
  for (bool choice : {false, true}) {
    std::string calls = JsonCalls();
    if (choice) {
      const std::string original = "c.function.arguments | tojson";
      const auto at = calls.find(original);
      ASSERT_NE(at, std::string::npos);
      calls.replace(at, original.size(),
                    "c.function.arguments if c.function.arguments is string "
                    "else c.function.arguments | tojson");
    }
    const auto source =
        Frame("{{ m.content }}" + calls, "") + Frame("{{ m.content }}");
    TextFormatAnalysis result;
    Diagnostic diagnostic;
    ASSERT_TRUE(AnalyzeTextFormat(source, "merged-json.jinja", {}, &result,
                                  &diagnostic))
        << diagnostic.message;
    ASSERT_EQ(result.alternatives.size(), 2u);
    bool generation = false, completed = false;
    for (const auto &format : result.alternatives) {
      generation |= !format.generation_prefix.empty();
      completed |= format.generation_prefix.empty();
      ExpectJsonFields(format, "name", "arguments");
      ASSERT_TRUE(format.tools.json_argument_wire);
      EXPECT_EQ(*format.tools.json_argument_wire,
                choice ? TextJsonArgumentWire::RawJson
                       : TextJsonArgumentWire::Json);
      if (choice)
        EXPECT_TRUE(HasChoiceFor(format.tools.call_body, TextCapture::Arguments,
                                 result));
      EXPECT_FALSE(format.tools.call_body.empty());
    }
    EXPECT_TRUE(generation);
    EXPECT_TRUE(completed);
  }
}

TEST(TextFormatAnalysisTest, JsonCodecRejectionKeepsPartOffsetAndOwnedResult) {
  auto source = "\n" + Frame("{{ m.content }}" + JsonCalls());
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      AnalyzeTextFormat(source, "json-codec.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  auto argument_offset =
      [&](auto &&self, const std::vector<TextFormatPart> &parts) -> size_t {
    for (const auto &part : parts) {
      if (part.capture == TextCapture::Arguments)
        return part.source.begin;
      for (const auto *children : {&part.body, &part.otherwise}) {
        const auto offset = self(self, *children);
        if (offset != std::string::npos)
          return offset;
      }
    }
    return std::string::npos;
  };
  const auto offset =
      argument_offset(argument_offset, result.alternatives[0].tools.call_body);
  ASSERT_NE(offset, std::string::npos);
  const auto count = result.alternatives.size();
  result.alternatives[0].assistant_header = "preserved";
  const std::string original = "c.function.arguments | tojson";
  const auto at = source.find(original);
  ASSERT_NE(at, std::string::npos);
  source.replace(at, original.size(), original + " | trim");
  EXPECT_FALSE(
      AnalyzeTextFormat(source, "json-codec.jinja", {}, &result, &diagnostic));
  EXPECT_EQ(diagnostic.message,
            "unsupported value codec filter or transform chain: trim");
  EXPECT_EQ(diagnostic.offset, offset);
  EXPECT_EQ(diagnostic.source_name, "json-codec.jinja");
  EXPECT_EQ(diagnostic.line, 2u);
  EXPECT_EQ(diagnostic.column, offset);
  ASSERT_EQ(result.alternatives.size(), count);
  EXPECT_EQ(result.alternatives[0].assistant_header, "preserved");
}

TEST(TextFormatAnalysisTest,
     OutputPrefixRejectionPreservesResultAndSourceEvidence) {
  auto source_for = [](std::string_view opener) {
    return "\n" + Frame("{{ '<reason>' + m.reasoning_content + '</reason>' + "
                        "m.content }}" +
                        JsonCalls("name", "arguments", opener));
  };
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  const auto valid = source_for("<invoke>");
  ASSERT_TRUE(
      AnalyzeTextFormat(valid, "prefix.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  ASSERT_FALSE(result.alternatives[0].evidence.empty());
  const auto offset = result.alternatives[0].evidence[0].begin;
  const auto count = result.alternatives.size();
  result.alternatives[0].assistant_header = "preserved";
  const auto invalid = source_for("<reason>");
  EXPECT_FALSE(
      AnalyzeTextFormat(invalid, "prefix.jinja", {}, &result, &diagnostic));
  EXPECT_EQ(diagnostic.message,
            "output marker prefixes require an unambiguous priority");
  EXPECT_EQ(diagnostic.source_name, "prefix.jinja");
  EXPECT_EQ(diagnostic.offset, offset);
  EXPECT_EQ(diagnostic.line, 2u);
  EXPECT_EQ(diagnostic.column, offset);
  ASSERT_EQ(result.alternatives.size(), count);
  EXPECT_EQ(result.alternatives[0].assistant_header, "preserved");
}

TEST(TextFormatAnalysisTest, TaggedRejectionPreservesResultAndSourceLocation) {
  const auto valid =
      "\n" + Frame("{{ m.content }}" + TaggedCalls(true, "{{ v }}"));
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      AnalyzeTextFormat(valid, "tagged-name.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  ASSERT_FALSE(result.alternatives[0].evidence.empty());
  const auto offset = result.alternatives[0].evidence[0].begin;
  const auto count = result.alternatives.size();
  const auto bindings = result.predicate_bindings;
  result.alternatives[0].assistant_header = "preserved";
  result.alternatives[0].tools.tagged.before_name = "preserved tagged shape";
  auto source = valid;
  const auto name = source.find("{{ c.function.name }}");
  ASSERT_NE(name, std::string::npos);
  source.replace(name, std::string("{{ c.function.name }}").size(),
                 "{{ c.function.name | tojson }}");
  EXPECT_FALSE(
      AnalyzeTextFormat(source, "tagged-name.jinja", {}, &result, &diagnostic));
  ASSERT_EQ(result.alternatives.size(), count);
  EXPECT_EQ(result.alternatives[0].assistant_header, "preserved");
  EXPECT_EQ(result.alternatives[0].tools.tagged.before_name,
            "preserved tagged shape");
  EXPECT_EQ(result.predicate_bindings, bindings);
  EXPECT_EQ(diagnostic.message, "tagged names require a direct source capture");
  EXPECT_EQ(diagnostic.source_name, "tagged-name.jinja");
  EXPECT_EQ(diagnostic.offset, offset);
  EXPECT_EQ(diagnostic.line,
            1u + std::count(source.begin(), source.begin() + offset, '\n'));
  const auto newline = source.rfind('\n', offset ? offset - 1 : 0);
  EXPECT_EQ(diagnostic.column,
            newline == std::string::npos ? offset + 1 : offset - newline);
}

TEST(TextFormatAnalysisTest, FunctionFieldSpellingIsNotAProfile) {
  const char *source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<turn>assistant:' + m.content }}"
      "{% for c in m.tool_calls %}"
      "{{ '<invoke>{\"operation\":\"' + c.function.name + "
      "'\",\"inputs\":' }}{{ c.function.arguments|tojson }}"
      "{{ '}</invoke>' }}{% endfor %}{{ '</turn>' }}"
      "{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      AnalyzeTextFormat(source, "renamed.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  const auto &tools = result.alternatives[0].tools;
  EXPECT_EQ(tools.kind, TextToolKind::JsonEnvelope);
  EXPECT_EQ(tools.open, "<invoke>");
  EXPECT_EQ(tools.close, "</invoke>");
  EXPECT_EQ(tools.name_field, "operation");
  EXPECT_EQ(tools.arguments_field, "inputs");
}

TEST(TextFormatAnalysisTest, Qwen3NoUserHistoryKeepsSourceStreamStartup) {
  const char *path = "lib/text/qwen3/templates/5da44855.jinja";
  const auto source = ReadSource(path);
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, path, Globals(), &result, &diagnostic))
      << diagnostic.line << ':' << diagnostic.column << ':'
      << diagnostic.message;
  GeneratedInputRenderer generated;
  ASSERT_TRUE(CompileChatTemplate(source, path, &generated, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto selector =
        PrepareSelection(source, result.selector_template, bytecode);
    auto renderer = PrepareProduct(generated, bytecode);
    ASSERT_NE(selector, nullptr);
    ASSERT_NE(renderer, nullptr);
    for (bool disabled : {false, true}) {
      SCOPED_TRACE(disabled ? "thinking explicitly false" : "thinking omitted");
      const auto environment =
          std::string("{\"messages\":[{\"role\":\"system\","
                      "\"content\":\"rules\"}],\"tools\":[],"
                      "\"add_generation_prompt\":true") +
          (disabled ? ",\"enable_thinking\":false}" : "}");
      const std::string expected_prefix =
          disabled ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                   : "<|im_start|>assistant\n";
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*selector, environment, &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      const auto &format = result.alternatives[id];
      EXPECT_EQ(format.generation_prefix, expected_prefix);
      EXPECT_EQ(mode, disabled ? 0 : 2);
      EXPECT_EQ(format.reasoning_open, "<think>");
      EXPECT_EQ(format.reasoning_close, "</think>");
      // No user was found, so the selected history-retention branch is plain.
      // Stream capability must not manufacture reasoning in that history.
      EXPECT_FALSE(
          HasCapture(format.assistant_pattern, TextCapture::Reasoning));
      EXPECT_FALSE(HasLiteral(format.assistant_pattern, "<think>"));
      EXPECT_FALSE(HasLiteral(format.assistant_pattern, "</think>"));
      std::string rendered, error;
      ASSERT_TRUE(renderer->Render(environment, &rendered, &error)) << error;
      EXPECT_EQ(rendered,
                "<|im_start|>system\nrules<|im_end|>\n" + expected_prefix);
    }
    std::string history, error;
    ASSERT_TRUE(renderer->Render(
        "{\"messages\":[{\"role\":\"system\",\"content\":\"rules\"},"
        "{\"role\":\"assistant\",\"content\":\"payload\","
        "\"reasoning_content\":\"must-not-be-replayed\"}],\"tools\":[],"
        "\"add_generation_prompt\":false}",
        &history, &error))
        << error;
    EXPECT_EQ(history, "<|im_start|>system\nrules<|im_end|>\n"
                       "<|im_start|>assistant\npayload<|im_end|>\n");
  }
}

TEST(TextFormatAnalysisTest, UnusedSetupBranchesDoNotMultiplyFormats) {
  const auto frame = Frame("{{ m.content }}");
  TextFormatAnalysis baseline, result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(frame, "plain-baseline.jinja", {}, &baseline,
                                &diagnostic))
      << diagnostic.message;
  std::string setup = "{% set x = '' %}";
  for (size_t i = 0; i < 12; ++i) {
    const auto suffix = std::to_string(i);
    setup += "{% if unused_option_" + suffix +
             " | default(false) %}"
             "{% set x = x + 'a' %}{% else %}"
             "{% set x = x + 'b' %}{% endif %}";
  }
  const auto source = setup + frame;
  ASSERT_TRUE(
      AnalyzeTextFormat(source, "unused-setup.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  // The shared x is never read by the frame. These accumulated branch values
  // affect no emitted byte or projection.
  EXPECT_EQ(result.alternatives.size(), baseline.alternatives.size());
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    auto projection =
        PrepareSelection(source, result.adaptation_selector_template, bytecode,
                         TemplateSelectorKind::Adaptation);
    ASSERT_NE(prepared, nullptr);
    ASSERT_NE(projection, nullptr);
    for (const char *option :
         {"", ",\"unused_option_0\":true", ",\"unused_option_11\":true"}) {
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared, Environment(option), &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      EXPECT_EQ(result.alternatives[id].assistant_header, "<turn>assistant:");
      EXPECT_EQ(result.alternatives[id].frame_suffix, "</turn>");
      EXPECT_EQ(mode, 0);
      int64_t mask = -1;
      ASSERT_TRUE(Select(
          *projection,
          WorkerEnvironment(result, Environment(option),
                            "{\"role\":\"user\",\"content\":\"question\"}", 0,
                            "null", 0),
          &mask, &mode));
      TextAdaptation facts;
      ASSERT_TRUE(DecodeTextAdaptation(mask, &facts));
      EXPECT_EQ(mode, 0);
      EXPECT_FALSE(facts.arguments_need_mapping);
      EXPECT_FALSE(facts.arguments_observe_string);
      EXPECT_FALSE(facts.content_observes_representation);
    }
  }
}

TEST(TextFormatAnalysisTest, HistoricalSystemValueDoesNotEnterAssistantFrame) {
  std::string source = "{% set x = '' %}";
  for (size_t i = 0; i < 12; ++i) {
    source += "{% if option_" + std::to_string(i) +
              " | default(false) %}"
              "{% set x = x ~ 'a' %}{% else %}"
              "{% set x = x ~ 'b' %}{% endif %}";
  }
  source += "{% for m in messages %}"
            "{% if m.role == 'system' %}{{ x }}"
            "{% elif m.role == 'assistant' %}{% if x %}"
            "{{ '<H>' + m.content + '<E>' }}{% endif %}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<H>' }}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "historical-system-value.jinja", {},
                                &result, &diagnostic))
      << diagnostic.line << ':' << diagnostic.column << ':'
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &format : result.alternatives) {
    EXPECT_EQ(format.assistant_header, "<H>");
    EXPECT_EQ(format.frame_suffix, "<E>");
    EXPECT_FALSE(format.evidence.empty());
  }
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    for (const char *option :
         {"", ",\"option_0\":true", ",\"option_11\":true"}) {
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared, Environment(option), &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      EXPECT_EQ(result.alternatives[id].assistant_header, "<H>");
      EXPECT_EQ(result.alternatives[id].frame_suffix, "<E>");
      EXPECT_EQ(mode, 0);
    }
  }
}

TEST(TextFormatAnalysisTest, RequestKwargSpellingIsNotFutureProvenance) {
  for (const char *keyword : {"__text_future_option", "__text_future_content",
                              "add_generation_prompt_extra"}) {
    SCOPED_TRACE(keyword);
    const auto source =
        Frame("{{ m.content }}{% if " + std::string(keyword) +
              " | default(false) %}" + JsonCalls("operation", "inputs") +
              "{% else %}" + JsonCalls() + "{% endif %}");
    TextFormatAnalysis result;
    Diagnostic diagnostic;
    ASSERT_TRUE(AnalyzeTextFormat(source, "fresh-request-name.jinja", {},
                                  &result, &diagnostic))
        << diagnostic.message;
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      auto prepared =
          PrepareSelection(source, result.selector_template, bytecode);
      ASSERT_NE(prepared, nullptr);
      int64_t plain_id = -1, alternate_id = -1, mode = -1;
      for (const char *value : {"false", "null"}) {
        ASSERT_TRUE(Select(*prepared,
                           Environment("," + Quote(keyword) + ":" + value),
                           &plain_id, &mode));
        ASSERT_GE(plain_id, 0);
        ASSERT_LT(static_cast<size_t>(plain_id), result.alternatives.size());
        ExpectJsonFields(result.alternatives[plain_id], "name", "arguments");
        EXPECT_EQ(mode, 0);
      }
      ASSERT_TRUE(Select(*prepared, Environment("," + Quote(keyword) + ":true"),
                         &alternate_id, &mode));
      ASSERT_GE(alternate_id, 0);
      ASSERT_LT(static_cast<size_t>(alternate_id), result.alternatives.size());
      ExpectJsonFields(result.alternatives[alternate_id], "operation",
                       "inputs");
      EXPECT_NE(plain_id, alternate_id);
      EXPECT_EQ(result.alternatives[plain_id].generation_prefix,
                result.alternatives[alternate_id].generation_prefix);
    }
  }
}

TEST(TextFormatAnalysisTest, HistoricalRoleRemainsActualRequestData) {
  const auto source =
      "{% set original_system = messages[0].role == 'system' %}" +
      Frame("{{ m.content }}{% if original_system %}" + JsonCalls() +
            "{% else %}" + JsonCalls("operation", "inputs") + "{% endif %}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      AnalyzeTextFormat(source, "history-role.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    const char *requests[] = {
        "{\"messages\":[{\"role\":\"system\",\"content\":\"rules\"},"
        "{\"role\":\"user\",\"content\":\"q\"}],"
        "\"add_generation_prompt\":true}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],"
        "\"add_generation_prompt\":true}",
    };
    for (size_t i = 0; i < 2; ++i) {
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared, requests[i], &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      ExpectJsonFields(result.alternatives[id], i ? "operation" : "name",
                       i ? "inputs" : "arguments");
      EXPECT_EQ(mode, 0);
    }
  }
}

std::string HistorySearchFrame(std::string_view after_search) {
  return "{% set seek = namespace(pending=true, position=messages|length - 1) %}"
         "{% for event in messages[::-1] %}"
         "{% set ordinal = (messages|length - 1) - loop.index0 %}"
         "{% if seek.pending and event.role == 'user' %}"
         "{% set seek.pending = false %}{% set seek.position = ordinal %}"
         "{% endif %}{% endfor %}" +
         std::string(after_search) +
         Frame("{{ m.content }}{% if loop.index0 > seek.position %}"
               "{{ '[AFTER]' }}{% else %}{{ '[BEFORE]' }}{% endif %}");
}

std::string SearchRequest(bool has_user, bool completed) {
  return "{\"messages\":[{\"role\":" + Quote(has_user ? "user" : "system") +
         ",\"content\":\"question\"}" +
         (completed ? ", {\"role\":\"assistant\",\"content\":\"payload\"}"
                    : "") +
         "],\"add_generation_prompt\":" + (completed ? "false}" : "true}");
}

TEST(TextFormatAnalysisTest,
     HistorySearchAliasOverwriteDoesNotUseStaleSummary) {
  const auto source = HistorySearchFrame(
      "{% set mirror = seek %}{% set mirror.position = 10000 %}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "history-alias-overwrite.jinja", {},
                                &result, &diagnostic))
      << diagnostic.line << ':' << diagnostic.column << ':'
      << diagnostic.message;
  GeneratedInputRenderer generated;
  ASSERT_TRUE(CompileChatTemplate(source, "history-alias-overwrite.jinja",
                                  &generated, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto selector =
        PrepareSelection(source, result.selector_template, bytecode);
    auto renderer = PrepareProduct(generated, bytecode);
    ASSERT_NE(selector, nullptr);
    ASSERT_NE(renderer, nullptr);
    for (bool has_user : {true, false, true}) {
      SCOPED_TRACE(has_user ? "user found" : "no user");
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(
          Select(*selector, SearchRequest(has_user, false), &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      EXPECT_EQ(result.alternatives[id].assistant_header, "<turn>assistant:");
      EXPECT_EQ(result.alternatives[id].frame_suffix, "[BEFORE]</turn>");
      EXPECT_EQ(mode, 0);
      std::string rendered, error;
      ASSERT_TRUE(
          renderer->Render(SearchRequest(has_user, true), &rendered, &error))
          << error;
      EXPECT_EQ(rendered, "<turn>assistant:payload[BEFORE]</turn>");
    }
  }
}

TEST(TextFormatAnalysisTest, HistorySearchNoUserBehaviorMatchesSourceGuard) {
  for (bool reject_no_user : {false, true}) {
    SCOPED_TRACE(reject_no_user ? "source rejects no user"
                                : "source permits no user");
    const auto source = HistorySearchFrame(
        reject_no_user
            ? "{% if seek.pending %}"
              "{{ raise_exception('a user message is required') }}{% endif %}"
            : "");
    TextFormatAnalysis result;
    Diagnostic diagnostic;
    ASSERT_TRUE(AnalyzeTextFormat(source, "history-no-user.jinja", {}, &result,
                                  &diagnostic))
        << diagnostic.message;
    GeneratedInputRenderer generated;
    ASSERT_TRUE(CompileChatTemplate(source, "history-no-user.jinja", &generated,
                                    &diagnostic))
        << diagnostic.message;
    for (bool bytecode : {false, true}) {
      auto selector =
          PrepareSelection(source, result.selector_template, bytecode);
      auto renderer = PrepareProduct(generated, bytecode);
      ASSERT_NE(selector, nullptr);
      ASSERT_NE(renderer, nullptr);
      for (bool has_user : {true, false}) {
        SCOPED_TRACE(has_user ? "user found" : "no user");
        int64_t id = 123, mode = 456;
        std::string rendered = "preserved", error;
        if (!has_user && reject_no_user) {
          EXPECT_FALSE(selector->SelectFormat(SearchRequest(false, false), &id,
                                              &mode, &error));
          EXPECT_EQ(id, 123);
          EXPECT_EQ(mode, 456);
          EXPECT_FALSE(error.empty());
          EXPECT_FALSE(
              renderer->Render(SearchRequest(false, true), &rendered, &error));
          EXPECT_EQ(rendered, "preserved");
          EXPECT_FALSE(error.empty());
          continue;
        }
        ASSERT_TRUE(
            Select(*selector, SearchRequest(has_user, false), &id, &mode));
        ASSERT_GE(id, 0);
        ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
        EXPECT_EQ(result.alternatives[id].assistant_header, "<turn>assistant:");
        EXPECT_EQ(result.alternatives[id].frame_suffix,
                  has_user ? "[AFTER]</turn>" : "[BEFORE]</turn>");
        EXPECT_EQ(mode, 0);
        ASSERT_TRUE(
            renderer->Render(SearchRequest(has_user, true), &rendered, &error))
            << error;
        EXPECT_EQ(rendered, has_user
                                ? "<turn>assistant:payload[AFTER]</turn>"
                                : "<turn>assistant:payload[BEFORE]</turn>");
      }
    }
  }
}

TEST(TextFormatAnalysisTest, AppendSensitiveMacroCannotUseLiveHistorySummary) {
  const auto source =
      std::string(
          "{% set aux = namespace(last=false) %}"
          "{% macro side(flag) %}{% set aux.last = flag %}{% endmacro %}"
          "{% set seek = namespace(pending=true, position=messages|length - 1) %}"
          "{% for event in messages %}"
          "{% set ignored = side(loop.last) %}"
          "{% if seek.pending and event.role == 'user' and aux.last %}"
          "{% set seek.pending = false %}"
          "{% set seek.position = loop.index0 %}"
          "{% endif %}{% endfor %}") +
      Frame("{{ m.content }}{% if loop.index0 > seek.position %}"
            "{{ '[AFTER]' }}{% else %}{{ '[BEFORE]' }}{% endif %}");
  const char *source_name = "append-sensitive-macro.jinja";
  GeneratedInputRenderer generated;
  Diagnostic diagnostic;
  ASSERT_TRUE(CompileChatTemplate(source, source_name, &generated, &diagnostic))
      << diagnostic.message;
  // The completed source sees user.last=false followed by assistant.last=true.
  // Its macro therefore prevents the user write; the final assistant is BEFORE.
  // Prove that fixture fact with the original renderer before testing analysis.
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto renderer = PrepareProduct(generated, bytecode);
    ASSERT_NE(renderer, nullptr);
    std::string rendered, error;
    ASSERT_TRUE(renderer->Render(SearchRequest(true, true), &rendered, &error))
        << error;
    EXPECT_EQ(rendered, "<turn>assistant:payload[BEFORE]</turn>");
  }

  TextFormatAnalysis result;
  if (!AnalyzeTextFormat(source, source_name, {}, &result, &diagnostic)) {
    EXPECT_EQ(diagnostic.source_name, source_name);
    EXPECT_LT(diagnostic.offset, source.size());
    EXPECT_GT(diagnostic.line, 0u);
    const bool causal_diagnostic =
        diagnostic.message.find("effect") != std::string::npos &&
        (diagnostic.message.find("macro") != std::string::npos ||
         diagnostic.message.find("append") != std::string::npos ||
         diagnostic.message.find("history") != std::string::npos);
    EXPECT_TRUE(causal_diagnostic) << diagnostic.message;
    return;
  }

  // Live [user] alone has last=true. Applying its paired-state summary to the
  // appended assistant would incorrectly choose AFTER; no such waiver is valid.
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto selector =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(selector, nullptr);
    int64_t id = -1, mode = -1;
    ASSERT_TRUE(Select(*selector, SearchRequest(true, false), &id, &mode));
    ASSERT_GE(id, 0);
    ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
    EXPECT_EQ(result.alternatives[id].assistant_header, "<turn>assistant:");
    EXPECT_EQ(result.alternatives[id].frame_suffix, "[BEFORE]</turn>");
    EXPECT_EQ(mode, 0);
  }
}

TEST(TextFormatAnalysisTest, SourceDefaultDoesNotOverwriteExplicitFalseOrNull) {
  const auto source =
      "{% if enable_thinking is not defined %}"
      "{% set enable_thinking = true %}{% endif %}" +
      Frame("{% if enable_thinking %}{{ '<think>' }}"
            "{{ m.reasoning_content | default('') }}{{ '</think>' }}"
            "{% endif %}{{ m.content }}",
            "{{ '<turn>assistant:' }}{% if enable_thinking %}"
            "{{ '<think>' }}{% endif %}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "thinking-default.jinja", {}, &result,
                                &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    const char *options[] = {"", ",\"enable_thinking\":false",
                             ",\"enable_thinking\":null",
                             ",\"enable_thinking\":true"};
    const int64_t expected[] = {1, 0, 0, 1};
    for (size_t i = 0; i < 4; ++i) {
      SCOPED_TRACE(options[i]);
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared, Environment(options[i]), &id, &mode));
      EXPECT_EQ(mode, expected[i]);
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      EXPECT_EQ(result.alternatives[id].generation_prefix,
                i == 0 || i == 3 ? "<turn>assistant:<think>"
                                 : "<turn>assistant:");
    }
  }
}

TEST(TextFormatAnalysisTest, ConditionalModuleFramesKeepTheirGuards) {
  const std::string source =
      "{% if alternate | default(false) %}{% set opening = '[ALT]' %}"
      "{% else %}{% set opening = '[BASE]' %}{% endif %}"
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ opening + m.content + '[END]' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ opening }}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      AnalyzeTextFormat(source, "module-frame.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    for (bool alternate : {false, true, false}) {
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared,
                         Environment(alternate ? ",\"alternate\":true"
                                               : ",\"alternate\":false"),
                         &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      const auto &format = result.alternatives[id];
      EXPECT_EQ(format.assistant_header, alternate ? "[ALT]" : "[BASE]");
      EXPECT_EQ(format.generation_prefix, alternate ? "[ALT]" : "[BASE]");
      EXPECT_EQ(format.frame_suffix, "[END]");
      EXPECT_EQ(mode, 0);
    }
  }
}

TEST(TextFormatAnalysisTest, TargetValidationDoesNotAdmitOrRejectLiveInput) {
  const std::string source =
      "{% for m in messages %}"
      "{% if (m.role == 'user') != (loop.index0 % 2 == 0) %}"
      "{{ raise_exception('roles must alternate') }}{% endif %}"
      "{% if m.role == 'assistant' %}"
      "{% if alternate|default(false) %}{{ '<ALT>' }}{% else %}{{ '<BASE>' }}{% endif %}"
      "{{ first_prefix if loop.first else '' }}{{ m.content + '<E>' }}"
      "{% else %}{{ '<U>' + m.content + '<E>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}"
      "{% if alternate|default(false) %}{{ '<ALT>' }}{% else %}{{ '<BASE>' }}{% endif %}"
      "{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "target-validation.jinja", {}, &result,
                                &diagnostic))
      << diagnostic.message;
  GeneratedInputRenderer generated;
  ASSERT_TRUE(CompileChatTemplate(source, "target-validation.jinja", &generated,
                                  &diagnostic))
      << diagnostic.message;
  const auto has_validation =
      [&](auto &&self, const std::vector<TextFormatPart> &parts) -> bool {
    for (const auto &part : parts)
      if (part.body_error.kind != Expr::Kind::None ||
          part.otherwise_error.kind != Expr::Kind::None ||
          self(self, part.body) || self(self, part.otherwise))
        return true;
    return false;
  };
  for (bool bytecode : {false, true}) {
    auto selector =
        PrepareSelection(source, result.selector_template, bytecode);
    auto renderer = PrepareProduct(generated, bytecode);
    ASSERT_NE(selector, nullptr);
    ASSERT_NE(renderer, nullptr);
    for (bool alternate : {false, true}) {
      const std::string header = alternate ? "<ALT>" : "<BASE>";
      for (bool generation : {false, true}) {
        const std::string options =
            ",\"first_prefix\":\"unavailable first-assistant prefix\",\"alternate\":" +
            std::string(alternate ? "true" : "false") +
            ",\"add_generation_prompt\":" + (generation ? "true" : "false") +
            "}";
        const auto live = "{\"messages\":[]" + options;
        std::string actual, error;
        ASSERT_TRUE(renderer->Render(live, &actual, &error)) << error;
        EXPECT_EQ(actual, generation ? header : "");
        int64_t id = -1, mode = -1;
        ASSERT_TRUE(Select(*selector, live, &id, &mode));
        ASSERT_GE(id, 0);
        ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
        EXPECT_EQ(result.alternatives[id].assistant_header, header);
        EXPECT_EQ(result.alternatives[id].generation_prefix,
                  generation ? header : "");
        EXPECT_EQ(mode, 0);
        EXPECT_TRUE(has_validation(has_validation,
                                   result.alternatives[id].assistant_pattern));
        actual = "prior output";
        EXPECT_FALSE(renderer->Render(
            "{\"messages\":[{\"role\":\"assistant\",\"content\":\"payload\"}]" +
                options,
            &actual, &error));
        EXPECT_EQ(actual, "prior output");
        EXPECT_FALSE(error.empty());
        ASSERT_TRUE(renderer->Render(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"question\"},"
            "{\"role\":\"assistant\",\"content\":\"payload\"}]" +
                options,
            &actual, &error))
            << error;
        EXPECT_EQ(actual, "<U>question<E>" + header + "payload<E>" +
                              (generation ? header : ""));
      }
    }
  }
}

TEST(TextFormatAnalysisTest, RedundantValidationKeepsOuterRuntimeFramingGuard) {
  const std::string source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{% if alternate %}"
      "{% if alternate %}{% else %}{{ raise_exception('alternate required') }}{% endif %}"
      "{{ '<ALT>' + m.content + '<E>' }}"
      "{% else %}"
      "{% if alternate %}{{ raise_exception('base required') }}{% else %}{% endif %}"
      "{{ '<BASE>' + m.content + '<E>' }}{% endif %}"
      "{% endif %}{% endfor %}{% if add_generation_prompt %}"
      "{% if alternate %}{{ '<ALT>' }}{% else %}{{ '<BASE>' }}{% endif %}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "redundant-validation.jinja", {},
                                &result, &diagnostic))
      << diagnostic.message;
  GeneratedInputRenderer generated;
  ASSERT_TRUE(CompileChatTemplate(source, "redundant-validation.jinja",
                                  &generated, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto selector =
        PrepareSelection(source, result.selector_template, bytecode);
    auto renderer = PrepareProduct(generated, bytecode);
    ASSERT_NE(selector, nullptr);
    ASSERT_NE(renderer, nullptr);
    for (bool alternate : {false, true, false}) {
      const auto environment = Environment(alternate ? ",\"alternate\":true"
                                                     : ",\"alternate\":false");
      const std::string header = alternate ? "<ALT>" : "<BASE>";
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*selector, environment, &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      EXPECT_EQ(result.alternatives[id].assistant_header, header);
      EXPECT_EQ(result.alternatives[id].generation_prefix, header);
      EXPECT_EQ(result.alternatives[id].frame_suffix, "<E>");
      EXPECT_EQ(mode, 0);
      std::string actual, error;
      ASSERT_TRUE(renderer->Render(environment, &actual, &error)) << error;
      EXPECT_EQ(actual, header);
    }
  }
}

TEST(TextFormatAnalysisTest, ConditionalTargetValidationKeepsPlainFraming) {
  const std::string source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{% if m.tool_calls %}{% if alternate %}{% else %}"
      "{{ raise_exception('calls require alternate') }}{% endif %}{% endif %}"
      "{% if alternate %}{{ '<ALT>' }}{% else %}{{ '<BASE>' }}{% endif %}"
      "{{ m.content + '<E>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}"
      "{% if alternate %}{{ '<ALT>' }}{% else %}{{ '<BASE>' }}{% endif %}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "conditional-validation.jinja", {},
                                &result, &diagnostic))
      << diagnostic.message;
  GeneratedInputRenderer generated;
  ASSERT_TRUE(CompileChatTemplate(source, "conditional-validation.jinja",
                                  &generated, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto selector =
        PrepareSelection(source, result.selector_template, bytecode);
    auto renderer = PrepareProduct(generated, bytecode);
    ASSERT_NE(selector, nullptr);
    ASSERT_NE(renderer, nullptr);
    for (bool alternate : {false, true, false}) {
      const auto environment = Environment(alternate ? ",\"alternate\":true"
                                                     : ",\"alternate\":false");
      const std::string header = alternate ? "<ALT>" : "<BASE>";
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*selector, environment, &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      EXPECT_EQ(result.alternatives[id].assistant_header, header);
      EXPECT_EQ(result.alternatives[id].generation_prefix, header);
      EXPECT_EQ(mode, 0);
      const auto completion =
          std::string(
              "{\"messages\":[{\"role\":\"assistant\",\"content\":\"payload\"}],") +
          "\"add_generation_prompt\":false,\"alternate\":" +
          (alternate ? "true" : "false") + "}";
      std::string actual, error;
      ASSERT_TRUE(renderer->Render(completion, &actual, &error)) << error;
      EXPECT_EQ(actual, header + "payload<E>");
    }
    std::string actual = "prior output", error;
    EXPECT_FALSE(renderer->Render(
        "{\"messages\":[{\"role\":\"assistant\",\"content\":\"payload\",\"tool_calls\":[{\"type\":\"function\",\"id\":\"c\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]}],"
        "\"add_generation_prompt\":false,\"alternate\":false}",
        &actual, &error));
    EXPECT_EQ(actual, "prior output");
    EXPECT_FALSE(error.empty());
  }
}

TEST(TextFormatAnalysisTest, UnmatchedRequestRouteRaisesAndPreservesOutputs) {
  const auto source = "{% if mode == 'supported' %}" +
                      Frame("{{ m.content }}") +
                      "{% else %}{{ raise_exception('unsupported mode') }}"
                      "{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "unmatched-route.jinja", {}, &result,
                                &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    int64_t id = -1, mode = -1;
    ASSERT_TRUE(
        Select(*prepared, Environment(",\"mode\":\"supported\""), &id, &mode));
    id = 123;
    mode = 456;
    std::string error;
    EXPECT_FALSE(prepared->SelectFormat(Environment(",\"mode\":\"other\""), &id,
                                        &mode, &error));
    EXPECT_EQ(id, 123);
    EXPECT_EQ(mode, 456);
    EXPECT_FALSE(error.empty());
  }
}

TEST(TextFormatAnalysisTest, MacroActualsResolveBeforeParameterShadowing) {
  const std::string prelude =
      "{% set ending = '[CALLER]' %}"
      "{% macro wrap(ending, value, closing='[DEFAULT]') %}"
      "{{ ending + value + closing }}{% endmacro %}";
  const auto source =
      prelude + Frame("{{ wrap('[OPEN]', m.content, closing=ending) }}",
                      "{{ '<turn>assistant:[OPEN]' }}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      AnalyzeTextFormat(source, "macro-shadow.jinja", {}, &result, &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &format : result.alternatives) {
    EXPECT_EQ(format.assistant_header, "<turn>assistant:[OPEN]");
    EXPECT_EQ(format.frame_suffix, "[CALLER]</turn>");
  }
  const auto defaults = prelude + Frame("{{ wrap('[OPEN]', m.content) }}",
                                        "{{ '<turn>assistant:[OPEN]' }}");
  ASSERT_TRUE(AnalyzeTextFormat(defaults, "macro-default.jinja", {}, &result,
                                &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &format : result.alternatives)
    EXPECT_EQ(format.frame_suffix, "[DEFAULT]</turn>");
}

TEST(TextFormatAnalysisTest, MacroFreeCaptureUsesModuleScope) {
  const std::string source =
      "{% set ending = '[MODULE]' %}"
      "{% macro emit(value) %}{{ value + ending }}{% endmacro %}" +
      Frame("{% set ending = '[LOCAL]' %}{{ emit(m.content) }}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "macro-module-capture.jinja", {},
                                &result, &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &format : result.alternatives)
    EXPECT_EQ(format.frame_suffix, "[MODULE]</turn>");
  GeneratedInputRenderer renderer;
  ASSERT_TRUE(CompileChatTemplate(source, "macro-module-capture.jinja",
                                  &renderer, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto prepared = PrepareProduct(renderer, bytecode);
    ASSERT_NE(prepared, nullptr);
    std::string rendered, error;
    ASSERT_TRUE(prepared->Render(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"question\"},"
        "{\"role\":\"assistant\",\"content\":\"payload\"}],"
        "\"add_generation_prompt\":false}",
        &rendered, &error))
        << error;
    EXPECT_EQ(rendered, "<turn>assistant:payload[MODULE]</turn>");
  }
}

TEST(TextFormatAnalysisTest, ToolOnlyNullAndMissingReasoningRemainChoices) {
  const auto source =
      Frame("{% if m.reasoning_content is defined %}"
            "{{ '<think>' + m.reasoning_content + '</think>' }}{% endif %}"
            "{% if m.content is not none %}{{ m.content }}{% endif %}" +
                JsonCalls(),
            "{{ '<turn>assistant:<think>' }}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "optional-assistant-fields.jinja", {},
                                &result, &diagnostic))
      << diagnostic.message;
  bool content_choice = false, reasoning_choice = false;
  for (const auto &format : result.alternatives) {
    content_choice |=
        HasChoiceFor(format.assistant_pattern, TextCapture::Content, result);
    reasoning_choice |=
        HasChoiceFor(format.assistant_pattern, TextCapture::Reasoning, result);
  }
  EXPECT_TRUE(content_choice);
  EXPECT_TRUE(reasoning_choice);
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    int64_t id = -1, mode = -1;
    ASSERT_TRUE(Select(*prepared, Environment(), &id, &mode));
    EXPECT_EQ(mode, 1);
  }
}

TEST(TextFormatAnalysisTest, JsonKeysAreDecodedAndMarkersNeedNotBeAngles) {
  struct Keys {
    std::string encoded_name, encoded_arguments, name, arguments;
  };
  const Keys cases[] = {
      {Quote("na\"me\\key"), Quote("in\"puts"), "na\"me\\key", "in\"puts"},
      {"\"na\\u006de\"", "\"ar\\u0067uments\"", "name", "arguments"},
  };
  for (const auto &keys : cases) {
    const auto source =
        Frame("{{ m.content }}" + JsonCallsEncodedKeys(keys.encoded_name,
                                                       keys.encoded_arguments,
                                                       "[CALL]", "[END]"));
    TextFormatAnalysis result;
    Diagnostic diagnostic;
    ASSERT_TRUE(AnalyzeTextFormat(source, "escaped-fields.jinja", {}, &result,
                                  &diagnostic))
        << diagnostic.message;
    bool found = false;
    for (const auto &format : result.alternatives) {
      if (format.tools.kind == TextToolKind::None)
        continue;
      found = true;
      ExpectJsonFields(format, keys.name, keys.arguments);
      EXPECT_EQ(format.tools.open, "[CALL]");
      EXPECT_EQ(format.tools.close, "[END]");
    }
    EXPECT_TRUE(found);
  }
}

TEST(TextFormatAnalysisTest, AdaptationSelectionIsPerOriginalMessage) {
  const auto source =
      Frame("{{ m.content }}" + ConditionalArgumentCalls("message_index == 1"));
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "per-message-projection.jinja", {},
                                &result, &diagnostic))
      << diagnostic.message;
  const std::string call =
      R"({"type":"function","function":{"name":"f","arguments":"{\"x\":1}"}})";
  const std::string message =
      "{\"role\":\"assistant\",\"content\":\"answer\",\"tool_calls\":[" + call +
      "]}";
  const std::string base =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"first\"}," + message +
      ",{\"role\":\"user\",\"content\":\"second\"}," + message +
      "],\"add_generation_prompt\":true}";
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.adaptation_selector_template, bytecode,
                         TemplateSelectorKind::Adaptation);
    ASSERT_NE(prepared, nullptr);
    for (size_t index : {size_t{1}, size_t{3}, size_t{1}}) {
      SCOPED_TRACE(index);
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(
          *prepared, WorkerEnvironment(result, base, message, index, call, 0),
          &id, &mode));
      ASSERT_GE(id, 0);
      TextAdaptation adaptation;
      ASSERT_TRUE(DecodeTextAdaptation(id, &adaptation));
      EXPECT_EQ(mode, 0);
      EXPECT_EQ(adaptation.arguments_need_mapping, index == 1);
      if (index == 3)
        EXPECT_TRUE(adaptation.arguments_observe_string);
    }
  }
}

TEST(TextFormatAnalysisTest, NameDependentJsonWireFailsBeforeEmission) {
  const auto source =
      Frame("{{ m.content }}" +
            ConditionalArgumentCalls("c.function.name == 'structured'"));
  TextFormatAnalysis result;
  result.alternatives.emplace_back().assistant_header = "preserved";
  Diagnostic diagnostic;
  EXPECT_FALSE(AnalyzeTextFormat(source, "name-dependent-wire.jinja", {},
                                 &result, &diagnostic));
  EXPECT_EQ(diagnostic.message,
            "value encoding must refer to exactly one semantic capture");
  EXPECT_EQ(diagnostic.source_name, "name-dependent-wire.jinja");
  // Statement source ranges begin just inside the opening tag delimiter.
  EXPECT_EQ(diagnostic.offset,
            source.find("{% if c.function.name == 'structured' %}") + 2);
  EXPECT_EQ(diagnostic.line, 1u);
  EXPECT_EQ(diagnostic.column, diagnostic.offset + 1);
  ASSERT_EQ(result.alternatives.size(), 1u);
  EXPECT_EQ(result.alternatives[0].assistant_header, "preserved");
}

TEST(TextFormatAnalysisTest, AdaptationSelectionIsPerOriginalToolCall) {
  // The name-dependent operation selects mapping demand, while the argument
  // wire has one source-type codec independent of the function name.
  const auto source = Frame(
      "{{ m.content }}{% if m.tool_calls %}{% for c in m.tool_calls %}"
      "{% if c.function.name == 'structured' %}"
      "{% for inspected_key, inspected_value in c.function.arguments.items() %}"
      "{% endfor %}{% endif %}"
      "{{ '<call>{\"name\":' }}{{ c.function.name | tojson }}"
      "{{ ',\"arguments\":' }}"
      "{{ c.function.arguments if c.function.arguments is string "
      "else c.function.arguments | tojson }}"
      "{{ '}</call>' }}{% endfor %}{% endif %}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "per-call-projection.jinja", {},
                                &result, &diagnostic))
      << diagnostic.message;
  const std::string structured =
      R"({"type":"function","function":{"name":"structured","arguments":"{\"x\":1}"}})";
  const std::string raw =
      R"({"type":"function","function":{"name":"raw","arguments":"{\"x\":2}"}})";
  const std::string message =
      "{\"role\":\"assistant\",\"content\":\"answer\",\"tool_calls\":[" +
      structured + "," + raw + "]}";
  const std::string base =
      "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}," + message +
      "],\"add_generation_prompt\":true}";
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.adaptation_selector_template, bytecode,
                         TemplateSelectorKind::Adaptation);
    ASSERT_NE(prepared, nullptr);
    for (size_t index : {size_t{0}, size_t{1}, size_t{0}}) {
      SCOPED_TRACE(index);
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(
          Select(*prepared,
                 WorkerEnvironment(result, base, message, 1,
                                   index == 0 ? structured : raw, index),
                 &id, &mode));
      ASSERT_GE(id, 0);
      TextAdaptation adaptation;
      ASSERT_TRUE(DecodeTextAdaptation(id, &adaptation));
      EXPECT_EQ(mode, 0);
      EXPECT_EQ(adaptation.arguments_need_mapping, index == 0);
      if (index == 1)
        EXPECT_TRUE(adaptation.arguments_observe_string);
    }
  }
}

void ExpectMappingProjection(std::string_view argument_body,
                             const char *source_name, bool raw_string_must_fail,
                             const char *analysis_error = nullptr) {
  const auto source = Frame(
      "{{ m.content }}{% if m.tool_calls %}{% for c in m.tool_calls %}"
      "{{ '<call>{\"name\":' }}{{ c.function.name | tojson }}"
      "{{ ',\"arguments\":' }}" +
      std::string(argument_body) + "{{ '}</call>' }}{% endfor %}{% endif %}");
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  if (analysis_error) {
    result.alternatives.emplace_back().assistant_header = "preserved";
    EXPECT_FALSE(
        AnalyzeTextFormat(source, source_name, {}, &result, &diagnostic));
    EXPECT_EQ(diagnostic.message, analysis_error);
    EXPECT_EQ(diagnostic.source_name, source_name);
    EXPECT_EQ(diagnostic.offset,
              source.find("{% if c.function.arguments is string %}") + 2);
    EXPECT_EQ(diagnostic.line, 1u);
    EXPECT_EQ(diagnostic.column, diagnostic.offset + 1);
    ASSERT_EQ(result.alternatives.size(), 1u);
    EXPECT_EQ(result.alternatives[0].assistant_header, "preserved");
  } else {
    ASSERT_TRUE(
        AnalyzeTextFormat(source, source_name, {}, &result, &diagnostic))
        << diagnostic.line << ':' << diagnostic.column << ':'
        << diagnostic.message;
  }
  GeneratedInputRenderer generated;
  ASSERT_TRUE(CompileChatTemplate(source, source_name, &generated, &diagnostic))
      << diagnostic.message;
  const std::string raw_call =
      R"({"type":"function","function":{"name":"f","arguments":"{\"x\":1}"}})";
  const std::string mapped_call =
      R"({"type":"function","function":{"name":"f","arguments":{"x":1}}})";
  const auto message = [](const std::string &call) {
    return "{\"role\":\"assistant\",\"content\":\"answer\","
           "\"tool_calls\":[" +
           call + "]}";
  };
  const auto request = [&](const std::string &call, bool generation) {
    return "{\"messages\":[{\"role\":\"user\",\"content\":\"question\"}," +
           message(call) +
           "],\"add_generation_prompt\":" + (generation ? "true}" : "false}");
  };
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    TextAdaptation adaptation;
    if (!analysis_error) {
      auto selector =
          PrepareSelection(source, result.adaptation_selector_template,
                           bytecode, TemplateSelectorKind::Adaptation);
      ASSERT_NE(selector, nullptr);
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*selector,
                         WorkerEnvironment(result, request(raw_call, true),
                                           message(raw_call), 1, raw_call, 0),
                         &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_TRUE(DecodeTextAdaptation(id, &adaptation));
      EXPECT_EQ(mode, 0);
      EXPECT_TRUE(adaptation.arguments_need_mapping);
      EXPECT_FALSE(adaptation.arguments_observe_string);
    }

    auto renderer = PrepareProduct(generated, bytecode);
    ASSERT_NE(renderer, nullptr);
    std::string rendered = "preserved", error;
    if (raw_string_must_fail) {
      EXPECT_FALSE(
          renderer->Render(request(raw_call, false), &rendered, &error));
      EXPECT_EQ(rendered, "preserved");
      EXPECT_FALSE(error.empty());
    }
    // Supported formats exercise the selected projection on native wire input.
    // For a rejected codec, mapped input proves only original renderer
    // behavior; no adaptation decision is claimed for an unavailable format
    // product.
    const auto &projected_call =
        analysis_error || adaptation.arguments_need_mapping ? mapped_call
                                                            : raw_call;
    ASSERT_TRUE(
        renderer->Render(request(projected_call, false), &rendered, &error))
        << error;
    EXPECT_EQ(rendered, "<turn>assistant:answer<call>{\"name\":\"f\","
                        "\"arguments\":{\"x\": 1}}</call></turn>");
  }
}

TEST(TextFormatAnalysisTest, ExplicitMappingRejectionEstablishesProjection) {
  ExpectMappingProjection(
      "{% if c.function.arguments is not mapping %}"
      "{{ raise_exception('arguments must be a mapping') }}{% endif %}"
      "{{ c.function.arguments | tojson }}",
      "mapping-rejection.jinja", true);
}

TEST(TextFormatAnalysisTest,
     RejectedStringEmissionFailsCodecButRetainsRendererSemantics) {
  ExpectMappingProjection(
      "{% if c.function.arguments is string %}"
      "{{ c.function.arguments }}"
      "{{ raise_exception('parse native STRING arguments first') }}"
      "{% endif %}"
      "{% if c.function.arguments is not mapping %}"
      "{{ raise_exception('arguments must be a mapping') }}{% endif %}"
      "{{ c.function.arguments | tojson }}",
      "rejected-string-emission.jinja", true,
      "source rejects this captured value type");
}

TEST(TextFormatAnalysisTest, EstablishedMappingMakesMemberGuardNoncyclic) {
  ExpectMappingProjection(
      "{% set member = c.function.arguments.x %}"
      "{% if member %}{% for key, value in c.function.arguments.items() %}"
      "{% endfor %}{% endif %}{{ c.function.arguments | tojson }}",
      "redundant-member-guard.jinja", false);
}

TEST(TextFormatAnalysisTest,
     RepeatedConditionalPrefixDropsPreserveNativeSelection) {
  constexpr size_t joins = 20;
  std::string source;
  for (size_t i = 0; i < joins; ++i)
    source +=
        "{% if drop_" + std::to_string(i) +
        " | default(false) %}{% set messages = messages[1:] %}{% endif %}";
  source += "{% for m in messages %}{% if m.role == 'assistant' %}"
            "{{ '<A>' + m.content }}{% if loop.length == 2 %}"
            "{{ '</two>' }}{% else %}{{ '</many>' }}{% endif %}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<A>' }}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "conditional-prefixes.jinja", {},
                                &result, &diagnostic))
      << diagnostic.message;
  std::string messages = "[";
  for (size_t i = 0; i < joins + 1; ++i) {
    if (i)
      messages += ',';
    messages += R"({"role":"user","content":"U"})";
  }
  messages += ']';
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    for (int pattern : {0, 1, 2}) {
      SCOPED_TRACE(pattern);
      std::string environment =
          "{\"messages\":" + messages + ",\"add_generation_prompt\":true";
      size_t dropped = 0;
      for (size_t i = 0; i < joins; ++i) {
        const bool drop = pattern == 1 || (pattern == 2 && i % 2 == 0);
        environment +=
            ",\"drop_" + std::to_string(i) + "\":" + (drop ? "true" : "false");
        dropped += drop;
      }
      environment += '}';
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared, environment, &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      // The completed sequence retains one real historical user even when
      // every optional drop executes, so the appended assistant survives.
      EXPECT_EQ(result.alternatives[id].frame_suffix,
                joins + 2 - dropped == 2 ? "</two>" : "</many>");
      EXPECT_EQ(result.alternatives[id].generation_prefix, "<A>");
      EXPECT_EQ(mode, 0);
    }
  }
}

TEST(TextFormatAnalysisTest, AffineCountOverflowDoesNotProveConstantRoutes) {
  // These are deliberately static-only. The analyzer must not use signed C++
  // overflow as a proof, independently of native integer arithmetic semantics.
  for (const char *condition : {"(total + 9223372036854775807 + 1) > total",
                                "(total - 9223372036854775807 - 2) < total"}) {
    SCOPED_TRACE(condition);
    const std::string branch = "{% if " + std::string(condition) + " %}";
    const std::string source =
        "{% set total = messages | length %}"
        "{% for m in messages %}{% if m.role == 'assistant' %}" +
        branch +
        "{{ '<HIGH>' + m.content + '</turn>' }}{% else %}"
        "{{ '<LOW>' + m.content + '</turn>' }}{% endif %}"
        "{% endif %}{% endfor %}{% if add_generation_prompt %}" +
        branch + "{{ '<HIGH>' }}{% else %}{{ '<LOW>' }}{% endif %}{% endif %}";
    TextFormatAnalysis result;
    result.alternatives.resize(1);
    result.alternatives.front().assistant_header = "preserved";
    Diagnostic diagnostic;
    EXPECT_FALSE(AnalyzeTextFormat(source, "count-overflow.jinja", {}, &result,
                                   &diagnostic));
    EXPECT_NE(diagnostic.message.find("unsupported cross-domain proof"),
              std::string::npos)
        << diagnostic.message;
    ASSERT_EQ(result.alternatives.size(), 1u);
    EXPECT_EQ(result.alternatives.front().assistant_header, "preserved");
  }
}

TEST(TextFormatAnalysisTest, CountPredicatesRemainRequestSelected) {
  const std::string source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<A>' + m.content }}{% if messages | length > 2 %}"
      "{{ '</many>' }}{% else %}{{ '</few>' }}{% endif %}"
      "{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<A>' }}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "request-count.jinja", {}, &result,
                                &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    for (size_t historical = 0; historical < 4; ++historical) {
      SCOPED_TRACE(historical);
      std::string messages = "[";
      for (size_t i = 0; i < historical; ++i) {
        if (i)
          messages += ',';
        messages += R"({"role":"user","content":"U"})";
      }
      messages += ']';
      const auto environment =
          "{\"messages\":" + messages + ",\"add_generation_prompt\":true}";
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared, environment, &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      // Selection sees the conceptual completed assistant's count, without
      // treating that count predicate as a future response-value choice.
      EXPECT_EQ(result.alternatives[id].frame_suffix,
                historical + 1 > 2 ? "</many>" : "</few>");
      EXPECT_EQ(result.alternatives[id].generation_prefix, "<A>");
      EXPECT_EQ(mode, 0);
    }
  }
}

TEST(TextFormatAnalysisTest, PrefixDropsRequireActualTargetSurvival) {
  for (size_t drops : {size_t{1}, size_t{2}}) {
    SCOPED_TRACE(drops);
    std::string source = "{% set work = messages %}";
    for (size_t i = 0; i < drops; ++i)
      source += "{% set work = work[1:] %}";
    source += "{% for m in work %}{% if m.role == 'assistant' %}"
              "{{ '<A>' + m.content }}"
              "{% if loop.length < 0 or loop.index0 < 0 %}{{ '</negative>' }}"
              "{% else %}{{ '</ok>' }}{% endif %}"
              "{% endif %}{% endfor %}"
              "{% if add_generation_prompt %}{{ '<A>' }}{% endif %}";
    TextFormatAnalysis result;
    Diagnostic diagnostic;
    ASSERT_TRUE(AnalyzeTextFormat(source, "prefix-survival.jinja", {}, &result,
                                  &diagnostic))
        << diagnostic.message;
    GeneratedInputRenderer generated;
    ASSERT_TRUE(CompileChatTemplate(source, "prefix-survival.jinja", &generated,
                                    &diagnostic))
        << diagnostic.message;
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      auto selector =
          PrepareSelection(source, result.selector_template, bytecode);
      auto renderer = PrepareProduct(generated, bytecode);
      ASSERT_NE(selector, nullptr);
      ASSERT_NE(renderer, nullptr);
      for (size_t historical = 0; historical <= drops + 1; ++historical) {
        SCOPED_TRACE(historical);
        std::string history;
        for (size_t i = 0; i < historical; ++i) {
          if (i)
            history += ',';
          history += R"({"role":"user","content":"U"})";
        }
        const auto live =
            "{\"messages\":[" + history + "],\"add_generation_prompt\":true}";
        const auto completed =
            "{\"messages\":[" + history + (historical ? "," : "") +
            "{\"role\":\"assistant\",\"content\":\"payload\"}],"
            "\"add_generation_prompt\":false}";
        const bool survives = historical + 1 > drops;
        std::string rendered, error;
        ASSERT_TRUE(renderer->Render(completed, &rendered, &error)) << error;
        EXPECT_EQ(rendered, survives ? "<A>payload</ok>" : "");
        ASSERT_TRUE(renderer->Render(live, &rendered, &error)) << error;
        EXPECT_EQ(rendered, "<A>");
        int64_t id = -41, mode = -42;
        if (survives) {
          ASSERT_TRUE(Select(*selector, live, &id, &mode));
          ASSERT_GE(id, 0);
          ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
          EXPECT_EQ(result.alternatives[id].frame_suffix, "</ok>");
        } else {
          // A valid live generation prefix does not prove that the completed
          // source would emit a target which a prefix drop has removed.
          EXPECT_FALSE(selector->SelectFormat(live, &id, &mode, &error));
          EXPECT_NE(error.find("no source-qualified text format"),
                    std::string::npos)
              << error;
          EXPECT_EQ(id, -41);
          EXPECT_EQ(mode, -42);
        }
      }
    }
  }
}

TEST(TextFormatAnalysisTest, ReasoningMarkersAreLiteralBuffers) {
  const std::string source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<A>[R]' + m.reasoning_content + '[/R]' + m.content + '<E>' }}"
      "{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<A>[R]' }}{% endif %}";
  TextFormatAnalysis result;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "bracket-reasoning.jinja", {}, &result,
                                &diagnostic))
      << diagnostic.message;
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &format : result.alternatives) {
    EXPECT_EQ(format.assistant_header, "<A>");
    EXPECT_EQ(format.reasoning_open, "[R]");
    EXPECT_EQ(format.reasoning_close, "[/R]");
    EXPECT_EQ(format.frame_suffix, "<E>");
  }
  for (bool bytecode : {false, true}) {
    auto prepared =
        PrepareSelection(source, result.selector_template, bytecode);
    ASSERT_NE(prepared, nullptr);
    for (bool generation : {false, true}) {
      const auto environment =
          "{\"messages\":[{\"role\":\"user\",\"content\":\"question\"}],"
          "\"add_generation_prompt\":" +
          std::string(generation ? "true}" : "false}");
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(Select(*prepared, environment, &id, &mode));
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.alternatives.size());
      EXPECT_EQ(result.alternatives[id].generation_prefix,
                generation ? "<A>[R]" : "");
      EXPECT_EQ(mode, generation ? 1 : 0);
    }
  }
}

} // namespace
} // namespace hrx::loomify_gguf::jinja
