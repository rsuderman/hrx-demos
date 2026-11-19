// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/template_probe_runner.h"
#include "lib/loomify/template_analysis.h"
#include "lib/loomify/text/jinja/format_analysis.h"
#include "lib/loomify/text/jinja/output_emitter.h"
#include "lib/loomify/text/jinja/request_emitter.h"
#include "lib/loomify/token_policy.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::jinja::testing {
namespace {
std::string Hex(std::string_view bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : bytes) {
    result += digits[c >> 4];
    result += digits[c & 15];
  }
  return result;
}
std::string Required(const char *key) {
  const char *value = std::getenv(key);
  EXPECT_TRUE(value && *value) << key;
  return value ? value : "";
}
constexpr std::string_view kPlain =
    "{% for m in messages %}"
    "{% if m.role == 'assistant' %}{{ '<assistant>' + m.content + '</turn>' }}"
    "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
constexpr std::string_view kTools =
    "{% for m in messages %}{% if m.role == 'assistant' %}"
    "{{ '<assistant>' }}{{ m.content }}"
    "{% if m.tool_calls %}{% for c in m.tool_calls %}"
    "{{ '<call>{\"name\":' }}{{ c.function.name | tojson }}"
    "{{ ',\"arguments\":' }}{{ c.function.arguments | tojson }}"
    "{{ '}</call>' }}{% endfor %}{% endif %}{{ '</turn>' }}"
    "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";

class CompletionTest : public ::testing::TestWithParam<bool> {
protected:
  // Every hook implementation comes from the actual owned module. The wrapper
  // only supplies invocation arguments and exposes its result through the
  // existing one-buffer PreparedTemplate seam.
  void Prepare(std::string_view source, std::string_view content,
               std::string_view reasoning = "", std::string_view calls = "",
               int candidate = 1, bool inspect_formatter = false,
               std::string_view response = "") {
    Diagnostic diagnostic;
    // Request/response wrappers can differ while the source-owned controls are
    // identical. Keep only this fixture's current product, not a shared cache.
    if (controls_.source.empty() || source_ != source) {
      TextFormatAnalysis analysis;
      GeneratedInputRenderer renderer, selector, adaptation, adapters, output,
          turn;
      ASSERT_TRUE(CompileChatTemplate(source, "completion.jinja", &renderer,
                                      &diagnostic))
          << diagnostic.message;
      ASSERT_TRUE(AnalyzeTextFormat(source, "completion.jinja", {}, &analysis,
                                    &diagnostic))
          << diagnostic.message;
      ASSERT_TRUE(CompileTemplateSelector(source, "completion.jinja",
                                          analysis.selector_template, &selector,
                                          &diagnostic))
          << diagnostic.message;
      ASSERT_TRUE(CompileTemplateSelector(
          source, "completion.jinja", analysis.adaptation_selector_template,
          &adaptation, &diagnostic, TemplateSelectorKind::Adaptation))
          << diagnostic.message;
      ASSERT_TRUE(EmitRequestAdapters("completion.jinja", analysis, adaptation,
                                      {}, {}, &adapters, &diagnostic))
          << diagnostic.message;
      ASSERT_TRUE(
          EmitOutputParser("completion.jinja", analysis, &output, &diagnostic))
          << diagnostic.message;
      ASSERT_TRUE(EmitTextTurnControls(source, "completion.jinja", analysis, {},
                                       &turn, &diagnostic))
          << diagnostic.message;
      GeneratedInputRenderer controls;
      ASSERT_TRUE(CombineGeneratedTemplates(
          {renderer, selector, adapters, output, turn}, renderer.entrypoint,
          &controls, &diagnostic))
          << diagnostic.message;
      controls_ = std::move(controls);
      source_ = source;
    }
    GeneratedInputRenderer wrapper;
    wrapper.entrypoint = "test.complete";
    wrapper.helper_requirements = {"lib/text/generated_input/host.loom"};
    wrapper.source = EmitTerminalMarkerSource("</turn>");
    for (const auto &[key, value] :
         std::vector<std::pair<std::string_view, std::string_view>>{
             {"content", content},
             {"reasoning", reasoning},
             {"calls", calls},
             {"response", response.empty() ? content : response}})
      wrapper.source += "global.rodata.def @test." + std::string(key) +
                        " = bytes(\"" + Hex(value) + "\")\n";
    wrapper.source += R"(
func.decl @chat_begin(%request: buffer, %tools: buffer) -> (buffer, i64)
func.decl import("input") @require(%condition: i32, %diagnostic: buffer)
func.def public @test.complete(%request: buffer) -> (buffer) {
  %content = global.load @test.content : buffer
  %reasoning = global.load @test.reasoning : buffer
  %calls = global.load @test.calls : buffer
  %response = global.load @test.response : buffer
)";
    if (inspect_formatter) {
      wrapper.source += R"(
  %ordinal = scalar.constant 0 : i64
  %name = buffer.constant utf8("lookup") : buffer
  %result, %ignored = func.call @render_tool(%name, %content, %ordinal) : (buffer, buffer, i64) -> (buffer, i64)
  func.return %result : buffer
}
)";
    } else {
      wrapper.source += R"(
  %prompt, %initial = func.call @chat_begin(%request, %calls) : (buffer, buffer) -> (buffer, i64)
  %result, %candidate = func.call @complete_text(%prompt, %response, %content, %reasoning, %calls, %initial, %initial) : (buffer, buffer, buffer, buffer, buffer, i64, i64) -> (buffer, i32)
)";
      wrapper.source += "  %expected = scalar.constant " +
                        std::to_string(candidate) + " : i32\n";
      wrapper.source += R"(
  %same = scalar.cmpi eq, %candidate, %expected : i32
  %flag = scalar.extui %same : i1 to i32
  %error = buffer.constant utf8("unexpected completion reuse candidate") : buffer
  func.call @require(%flag, %error) : (i32, buffer)
  func.return %result : buffer
}
)";
    }
    GeneratedInputRenderer combined;
    ASSERT_TRUE(CombineGeneratedTemplates(
        {controls_, wrapper}, wrapper.entrypoint, &combined, &diagnostic))
        << diagnostic.message;
    const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                       Required("TEST_WORKSPACE"))
                          .string();
    std::string error;
    ASSERT_TRUE(PreparedTemplate::Prepare(combined, root,
                                          Required("TEST_TMPDIR"), GetParam(),
                                          &prepared_, &error))
        << error;
  }
  std::string Render(std::string_view request) {
    std::string result, error;
    EXPECT_NE(prepared_, nullptr);
    if (prepared_)
      EXPECT_TRUE(prepared_->Render(request, &result, &error)) << error;
    return result;
  }
  std::string source_;
  GeneratedInputRenderer controls_;
  std::unique_ptr<PreparedTemplate> prepared_;
};

TEST_P(CompletionTest, RendersSameSourceForCompletionAndNextTurn) {
  Prepare(kPlain, "Hello");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"user","content":"Hi"}],"model_options":{"add_generation_prompt":true,"new_option":"kept"}})"),
      "<user>Hi</turn><assistant>Hello</turn>");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"},{"role":"user","content":"Again"}]})"),
      "<user>Hi</turn><assistant>Hello</turn><user>Again</turn><assistant>Hello</turn>");
}

TEST_P(CompletionTest, StructuredHistoryKeepsFutureNullLoopErrorLazy) {
  constexpr std::string_view source =
      "{% for m in messages %}{{ '<'+m.role+'>' }}"
      "{% if m.content is string %}{{ m.content }}{% else %}"
      "{% for piece in m.content %}{{ piece.text }}{% endfor %}"
      "{% endif %}{{ '</turn>' }}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  Prepare(source, "answer");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"user","content":[{"text":"question"}]}]})"),
      "<user>question</turn><assistant>answer</turn>");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"assistant","content":[{"text":"old"}]},{"role":"user","content":"again"}]})"),
      "<assistant>old</turn><user>again</turn><assistant>answer</turn>");
}

TEST_P(CompletionTest, HistoricalReplacementsLeaveConstructedAssistantRaw) {
  constexpr std::string_view source =
      "{% macro clean(t) %}{% if t is string %}"
      "{{ t|replace('x','')|replace('\\x00','')|replace('<A>','&lt;A>') }}"
      "{% else %}{{ t }}{% endif %}{% endmacro %}"
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<assistant>'+m.content+'</turn>' }}{% else %}"
      "{{ '<user>'+clean(m.content)+'</turn>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  const auto answer = std::string("x<A>") + '\0' + u8"é";
  Prepare(source, answer);
  EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"x<A>\u0000é"}]})"),
            std::string(u8"<user>&lt;A>é</turn><assistant>") + answer +
                "</turn>");
}

TEST_P(CompletionTest, SourceTrimDeterminesCanonicalAndRejectsCandidate) {
  const std::string source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<assistant>' + (m.content | trim) + '</turn>' }}"
      "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
      "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  Prepare(source, " Hello ", "", "", 0);
  EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
            "<user>Hi</turn><assistant>Hello</turn>");
}

TEST_P(CompletionTest, InertLastConditionsPreserveCompletionCandidate) {
  const std::vector<std::string> sources = {
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{% if loop.last %}{% set unused = 1 %}{% endif %}"
      "{{ '<assistant>' + m.content + '</turn>' }}"
      "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
      "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}",
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{% if loop.last %}{{ '<assistant>' + m.content + '</turn>' }}"
      "{% else %}{{ '<assistant>' + m.content + '</turn>' }}{% endif %}"
      "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
      "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"};
  for (const auto &source : sources) {
    SCOPED_TRACE(source);
    Prepare(source, "Hello");
    EXPECT_EQ(
        Render(R"({"messages":[{"role":"assistant","content":"Prior"}]})"),
        "<assistant>Prior</turn><assistant>Hello</turn>");
  }
}

TEST_P(CompletionTest, ReflowSnapshotPreservesOutputMacroEffects) {
  const std::string source =
      "{% set ns = namespace(ok=false) %}"
      "{% macro ready() %}{% set ns.ok = true %}{% endmacro %}"
      "{{ ready() }}{% if not ns.ok %}{{ raise_exception('lost effect') }}"
      "{% endif %}" +
      std::string(kPlain);
  Prepare(source, "Hello");
  EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
            "<user>Hi</turn><assistant>Hello</turn>");
}

TEST_P(CompletionTest, AppendRoleValidationDoesNotRejectValidLiveHistory) {
  constexpr std::string_view role_error =
      "Conversation roles must alternate user/assistant/user/assistant/...";
  const std::string source =
      "{% for m in messages %}"
      "{% if (m.role == 'user') != (loop.index0 % 2 == 0) %}"
      "{{ raise_exception('" +
      std::string(role_error) +
      "') }}{% endif %}"
      "{% if m.role == 'assistant' %}"
      "{{ '<assistant>' + m.content + '</turn>' }}"
      "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
      "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  Prepare(source, "Hello");
  ASSERT_NE(prepared_, nullptr);
  GeneratedInputRenderer original;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      CompileChatTemplate(source, "roles.jinja", &original, &diagnostic))
      << diagnostic.message;
  std::unique_ptr<PreparedTemplate> live;
  std::string error;
  const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                     Required("TEST_WORKSPACE"))
                        .string();
  ASSERT_TRUE(PreparedTemplate::Prepare(original, root, Required("TEST_TMPDIR"),
                                        GetParam(), &live, &error))
      << error;
  const std::vector<std::pair<std::string, std::string>> invalid_appends = {
      {R"({"messages":[],"add_generation_prompt":true})", "<assistant>"},
      {R"({"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Prior"}],"add_generation_prompt":true})",
       "<user>Hi</turn><assistant>Prior</turn><assistant>"}};
  for (const auto &[request, expected_live] : invalid_appends) {
    SCOPED_TRACE(request);
    std::string actual;
    ASSERT_TRUE(live->Render(request, &actual, &error)) << error;
    EXPECT_EQ(actual, expected_live);
    actual = "preserved";
    EXPECT_FALSE(prepared_->Render(request, &actual, &error));
    EXPECT_EQ(actual, "preserved");
    EXPECT_NE(error.find(role_error), std::string::npos) << error;
  }
  EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
            "<user>Hi</turn><assistant>Hello</turn>");
}

TEST(CompletionFactsTest, UnsupportedLastSensitiveRewritesFailAtTurnExport) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"{% set final = loop.last %}{% if final %}"
       "{{ '<assistant>' + m.content + '</turn>' }}{% else %}"
       "{{ '<assistant>OLD' + m.content + '</turn>' }}{% endif %}",
       "a401 completion cannot qualify this source history rewrite"},
      {"{{ '<assistant>' + ('NEW' if loop.last else 'OLD') + m.content + "
       "'</turn>' }}",
       "a401 completion cannot qualify a last-sensitive emitted expression"},
      {"{{ '<assistant>' }}{% if loop.last %}"
       "{% if enable_thinking | default(false) %}{{ '<think>' }}"
       "{{ m.reasoning_content | default('') }}{{ '</think>' }}{% endif %}"
       "{{ m.content }}{% else %}{{ m.content }}{% endif %}{{ '</turn>' }}",
       "a401 completion cannot qualify this source history rewrite"}};
  for (const auto &[assistant, expected_diagnostic] : cases) {
    const std::string source =
        "{% for m in messages %}{% if m.role == 'assistant' %}" + assistant +
        "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
    SCOPED_TRACE(source);
    TextFormatAnalysis analysis;
    Diagnostic diagnostic;
    ASSERT_TRUE(
        AnalyzeTextFormat(source, "history.jinja", {}, &analysis, &diagnostic))
        << diagnostic.message;
    GeneratedInputRenderer turn;
    turn.source = "preserved";
    EXPECT_FALSE(EmitTextTurnControls(source, "history.jinja", analysis, {},
                                      &turn, &diagnostic));
    EXPECT_NE(diagnostic.message.find(expected_diagnostic), std::string::npos)
        << diagnostic.message;
    EXPECT_EQ(diagnostic.source_name, "history.jinja");
    EXPECT_EQ(turn.source, "preserved");
  }
}

TEST_P(CompletionTest, OriginalKwargsSelectBothLiveAndCompletedBranches) {
  constexpr std::string_view source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{% if custom_style | default(false) %}"
      "{{ '<assistant>styled:' + m.content + '</turn>' }}"
      "{% else %}{{ '<assistant>' + m.content + '</turn>' }}{% endif %}"
      "{% else %}{{ '<' + m.role + '>' + m.content + '</turn>' }}{% endif %}"
      "{% endfor %}{% if add_generation_prompt %}"
      "{% if custom_style | default(false) %}{{ '<assistant>styled:' }}"
      "{% else %}{{ '<assistant>' }}{% endif %}{% endif %}";
  Prepare(source, "Hello");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"user","content":"Hi"}],"model_options":{"custom_style":true,"add_generation_prompt":true}})"),
      "<user>Hi</turn><assistant>styled:Hello</turn>");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"user","content":"Hi"}],"custom_style":false})"),
      "<user>Hi</turn><assistant>Hello</turn>");
}

TEST_P(CompletionTest, RawResponseAndReasoningMustMatchFiniteCandidateRule) {
  Prepare(kPlain, "Hello", "", "", 0, false, "different raw output");
  EXPECT_EQ(Render(R"({"messages":[]})"), "<assistant>Hello</turn>");
  Prepare(kPlain, "Hello", "thought", "", 0);
  EXPECT_EQ(Render(R"({"messages":[]})"), "<assistant>Hello</turn>");
}

TEST_P(CompletionTest, CanonicalReasoningFallbackPreservesHistoricalAliases) {
  for (const char *fallback : {
           "m.reasoning_content",
           "m.reasoning or m.reasoning_content",
           "m.thinking or m.reasoning or m.reasoning_content",
       }) {
    SCOPED_TRACE(fallback);
    const std::string source =
        "{% for m in messages %}{% if m.role == 'assistant' %}"
        "{{ '<assistant>' }}{% set r=" +
        std::string(fallback) +
        " %}{% set r=r if r is string else '' %}"
        "{% if r %}{{ '<think>'+r+'</think>' }}{% endif %}"
        "{{ m.content+'</turn>' }}{% else %}"
        "{{ '<'+m.role+'>'+m.content+'</turn>' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
    // Prepare invokes actual emitted turn_environment; its reasoning argument
    // creates only reasoning_content, rather than a hand-built all-alias
    // object.
    Prepare(source, "A", "R", "", 0);
    EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
              "<user>Hi</turn><assistant><think>R</think>A</turn>");
    const std::string old_reason =
        std::string(fallback) == "m.reasoning_content"                  ? "R"
        : std::string(fallback) == "m.reasoning or m.reasoning_content" ? "Q"
                                                                        : "T";
    EXPECT_EQ(
        Render(
            R"({"messages":[{"role":"assistant","content":"old","thinking":"T","reasoning":"Q","reasoning_content":"R"},{"role":"user","content":"Hi"}]})"),
        "<assistant><think>" + old_reason +
            "</think>old</turn><user>Hi</turn><assistant><think>R</think>A</turn>");
    Prepare(source, "A");
    EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
              "<user>Hi</turn><assistant>A</turn>");
  }
}

TEST_P(CompletionTest, CanonicalRecordOmitsNoncanonicalReasoningFields) {
  const std::string source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{% if m.thinking is defined or m.reasoning is defined %}"
      "{{ raise_exception('noncanonical reasoning fields constructed') }}"
      "{% endif %}{{ '<assistant>' }}"
      "{% set r=m.thinking or m.reasoning or m.reasoning_content %}"
      "{% if r %}{{ '<think>'+r+'</think>' }}{% endif %}"
      "{{ m.content+'</turn>' }}{% else %}"
      "{{ '<'+m.role+'>'+m.content+'</turn>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  Prepare(source, "A", "R", "", 0);
  EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
            "<user>Hi</turn><assistant><think>R</think>A</turn>");
}

TEST_P(CompletionTest, CanonicalReasoningMacroRetainsActualMessageArgument) {
  const std::string source =
      "{% macro show(record) %}"
      "{% set r=record.thinking or record.reasoning or record.reasoning_content %}"
      "{% if r is string and r %}{{ '<think>'+r+'</think>' }}{% endif %}"
      "{{ record.content }}{% endmacro %}"
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<assistant>'+show(m)+'</turn>' }}{% else %}"
      "{{ '<'+m.role+'>'+m.content+'</turn>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  Prepare(source, "A", "R", "", 0);
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"assistant","content":"old","thinking":"T","reasoning":"Q","reasoning_content":"R"},{"role":"user","content":"Hi"}]})"),
      "<assistant><think>T</think>old</turn><user>Hi</turn><assistant><think>R</think>A</turn>");
}

TEST_P(CompletionTest, MissingReasoningGetIsDefinedNoneAndDotIsUndefined) {
  const std::string source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<assistant>' }}"
      "{% if m.thinking is defined or m.reasoning is defined %}"
      "{{ raise_exception('missing dot was defined') }}{% endif %}"
      "{% if m.get('thinking') is undefined or m.get('reasoning') is undefined %}"
      "{{ raise_exception('missing get was undefined') }}{% endif %}"
      "{% if m.get('thinking') is not none or m.get('reasoning') is not none %}"
      "{{ raise_exception('missing get was not None') }}{% endif %}"
      "{{ m.content+'</turn>' }}{% else %}"
      "{{ '<'+m.role+'>'+m.content+'</turn>' }}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  Prepare(source, "A", "R", "", 0);
  EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
            "<user>Hi</turn><assistant>A</turn>");
}

TEST_P(CompletionTest,
       CompactOriginalUsesCanonicalReasonAndHistoricalPrecedence) {
  const auto path = std::filesystem::path(Required("TEST_SRCDIR")) /
                    Required("TEST_WORKSPACE") /
                    "lib/text/compact_chatml/templates/f45005c5.jinja";
  std::ifstream input(path);
  ASSERT_TRUE(input);
  const std::string source(std::istreambuf_iterator<char>(input), {});
  Prepare(source, "A", "R", "", 0);
  const auto actual = Render(
      R"({"messages":[{"role":"assistant","content":"old","thinking":"T","reasoning":"Q","reasoning_content":"historical-R"},{"role":"user","content":"Hi"}],"model_options":{"preserve_thinking":true}})");
  EXPECT_NE(actual.find("<think>T</think>old"), std::string::npos);
  const std::string tail =
      "<|im_start|>assistant\n<think>R</think>A<|im_end|>\n";
  ASSERT_GE(actual.size(), tail.size());
  EXPECT_EQ(actual.substr(actual.size() - tail.size()), tail);
  EXPECT_EQ(actual.find("<think>Q</think>"), std::string::npos);
  EXPECT_EQ(actual.find("<think>historical-R</think>"), std::string::npos);
}

TEST_P(CompletionTest, ToolCompletionUsesOrdinaryWireStringProjection) {
  // Generated tool-only content is canonical NULL; this source prints it as
  // Jinja's "None". Original history keeps its existing request projection.
  Prepare(kTools, "", "",
          R"(<call>{"name":"lookup","arguments":"{\"n\":1.2500e+02}"}</call>)",
          0);
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]})"),
      "<user>Hi</turn><assistant>None<call>{\"name\":\"lookup\",\"arguments\":\"{\\\"n\\\":1.2500e+02}\"}</call></turn>");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"request-assigned-7","type":"function","function":{"name":"lookup","arguments":"{\"n\":1.2500e+02}"}}]},{"role":"tool","tool_call_id":"request-assigned-7","content":"found"}]})"),
      "<assistant><call>{\"name\":\"lookup\",\"arguments\":\"{\\\"n\\\":1.2500e+02}\"}</call></turn><tool>found</turn><assistant>None<call>{\"name\":\"lookup\",\"arguments\":\"{\\\"n\\\":1.2500e+02}\"}</call></turn>");
}

TEST_P(CompletionTest, CanonicalToolOnlyContentReachesSourceNullLoopFailure) {
  std::string source(kTools);
  const auto content = source.find("{{ m.content }}");
  ASSERT_NE(content, std::string::npos);
  source.replace(
      content, std::string("{{ m.content }}").size(),
      "{% if m.content is string %}{{ m.content }}{% else %}"
      "{% for piece in m.content %}{{ piece.text }}{% endfor %}{% endif %}");
  // Obtain actual source-formatted bytes through the existing render_tool
  // wrapper, then pass that same fragment to the real complete_text hook.
  Prepare(source, "{}", "", "", 0, true);
  const auto calls = Render("{}");
  ASSERT_FALSE(calls.empty());
  constexpr std::string_view request =
      R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]})";
  for (const char *reasoning : {"", "thought"}) {
    SCOPED_TRACE(reasoning);
    Prepare(source, "", reasoning, calls, 0);
    ASSERT_NE(prepared_, nullptr);
    std::string actual = "preserved", error;
    EXPECT_FALSE(prepared_->Render(request, &actual, &error));
    EXPECT_EQ(actual, "preserved");
    EXPECT_FALSE(error.empty());
    Prepare(source, "answer", reasoning, calls, 0);
    EXPECT_EQ(Render(request),
              "<user>Hi</turn><assistant>answer" + calls + "</turn>");
  }
  // Empty content without generated calls is STRING. An explicit historical
  // empty STRING remains its original representation even while calls follow.
  Prepare(source, "", "", "", 0, false, "different raw response");
  EXPECT_EQ(Render(R"({"messages":[{"role":"user","content":"Hi"}]})"),
            "<user>Hi</turn><assistant></turn>");
  Prepare(source, "answer", "", calls, 0);
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]})"),
      "<assistant>" + calls + "</turn><user>Hi</turn><assistant>answer" +
          calls + "</turn>");
}

TEST_P(CompletionTest, NativeFormatterPreservesOriginalArgumentStringLexemes) {
  // a401 supplies name/argument STRING/ordinal, not an assigned client ID or
  // a whole request object. Its source codec must preserve the argument bytes.
  Prepare(kTools, R"({"n":1.2500e+02})", "", "", 0, true);
  EXPECT_EQ(
      Render("{}"),
      R"(<call>{"name":"lookup","arguments":"{\"n\":1.2500e+02}"}</call>)");
}

TEST_P(CompletionTest, FailurePreservesOutputAndSuccessfulRenderCanAliasInput) {
  Prepare(kPlain, "Hello");
  ASSERT_NE(prepared_, nullptr);
  std::string output = "preserved", error;
  EXPECT_FALSE(prepared_->Render(R"({"messages":{}})", &output, &error));
  EXPECT_EQ(output, "preserved");
  std::string request = R"({"messages":[{"role":"user","content":"Hi"}]})";
  EXPECT_TRUE(prepared_->Render(request, &request, &error)) << error;
  EXPECT_EQ(request, "<user>Hi</turn><assistant>Hello</turn>");
}

TEST_P(CompletionTest,
       RequestPrefixStaysInPromptAcrossFirstAssistantPlacement) {
  const std::string source =
      "{% set prefix %}{% if fail %}{{ raise_exception('prefix failure') }}{% endif %}"
      "{% if tools %}<tools>{% for tool in tools %}{{ tool.name }}{% endfor %}</tools>{% endif %}{% endset %}"
      "{% set state=namespace(pending=true) %}"
      "{% for m in messages %}"
      "{% if state.pending and m.role != 'system' %}{{ prefix }}{% set state.pending=false %}{% endif %}"
      "{% if m.role == 'assistant' %}<assistant>{{ m.content }}</turn>{% else %}<other>{{ m.content }}</turn>{% endif %}"
      "{% endfor %}"
      "{% if state.pending %}{{ prefix }}{% endif %}"
      "{% if add_generation_prompt %}<assistant>{% endif %}";
  Prepare(source, "Hello");
  for (
      const auto &[history, prior] :
      std::vector<std::pair<std::string, std::string>>{
          {"[]", ""},
          {R"([{"role":"system","content":"s"}])", "<other>s</turn>"},
          {R"([{"role":"system","content":"s"},{"role":"system","content":"t"}])",
           "<other>s</turn><other>t</turn>"}}) {
    for (const auto &[tools, prefix] :
         std::vector<std::pair<std::string, std::string>>{
             {"", ""},
             {",\"tools\":[]", ""},
             {R"(,"tools":[{"name":"x"},{"name":"y"}])", "<tools>xy</tools>"}})
      EXPECT_EQ(Render("{\"messages\":" + history + tools + "}"),
                prior + prefix + "<assistant>Hello</turn>");
  }
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"user","content":"u"}],"tools":[{"name":"x"}]})"),
      "<tools>x</tools><other>u</turn><assistant>Hello</turn>");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"assistant","content":"a"}],"tools":[{"name":"x"}]})"),
      "<tools>x</tools><assistant>a</turn><assistant>Hello</turn>");
  EXPECT_EQ(
      Render(
          R"({"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","content":"v"}],"tools":[{"name":"x"}]})"),
      "<other>s</turn><tools>x</tools><other>u</turn><assistant>a</turn><other>v</turn><assistant>Hello</turn>");
  ASSERT_NE(prepared_, nullptr);
  std::string output = "preserved", error;
  EXPECT_FALSE(
      prepared_->Render(R"({"messages":[],"fail":true})", &output, &error));
  EXPECT_EQ(output, "preserved");
  EXPECT_NE(error.find("prefix failure"), std::string::npos) << error;
}

INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, CompletionTest, ::testing::Bool());
} // namespace
} // namespace hrx::loomify_gguf::jinja::testing
