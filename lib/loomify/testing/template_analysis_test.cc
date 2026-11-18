// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/template_analysis.h"
#include "lib/loomify/template_probe_runner.h"
#include "lib/loomify/template_analysis_internal.h"
#include "lib/loomify/text/jinja/source_template.h"
#include "lib/loomify/testing/template_qualification.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::jinja {
namespace {

std::filesystem::path Root() {
  const char *runfiles = std::getenv("TEST_SRCDIR");
  const char *workspace = std::getenv("TEST_WORKSPACE");
  EXPECT_NE(runfiles, nullptr);
  EXPECT_NE(workspace, nullptr);
  return std::filesystem::path(runfiles ? runfiles : "") /
         (workspace ? workspace : "");
}

TemplateAnalysisOptions Options(bool bytecode) {
  const char *scratch = std::getenv("TEST_TMPDIR");
  EXPECT_NE(scratch, nullptr);
  return {Root().string(), scratch ? scratch : "", bytecode};
}

bool CompileAndAnalyze(std::string_view source, std::string_view name,
                       const TextFormatBindings &bindings,
                       const TemplateAnalysisOptions &options,
                       AnalyzedTemplate *out, Diagnostic *diagnostic) {
  GeneratedInputRenderer input;
  AnalyzedTemplate result;
  if (!CompileChatTemplate(source, name, &input, diagnostic) ||
      !AnalyzeTemplate(source, name, input, bindings, options, &result,
                       diagnostic))
    return false;
  *out = std::move(result);
  return true;
}

constexpr const char *kPlain =
    "{{ bos_token }}{% for item in messages %}"
    "{{ '<turn>' + item.role + ':' + item.content + '</turn>' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<turn>assistant:' }}{% endif %}";

std::string OriginalSpark() {
  std::ifstream stream(Root() / "lib/text/spark/templates/93d6aa90.jinja");
  EXPECT_TRUE(stream);
  return std::string(std::istreambuf_iterator<char>(stream), {});
}

TEST(TemplateAnalysisTest, SharedSourceKeepsModesFreshAndProductsOwned) {
  AnalyzedTemplate owned;
  {
    const std::string source =
        "{% set __text_raw_message = 'source value' %}"
        "{% set __text_raw_call = 'source call' %}"
        "{% set __text_target_missing = 0 %}"
        "{% set __text_continued_user_text = 'source text' %}"
        "{% macro echo(value) %}{{ value }}{% endmacro %}"
        "{% for item in messages %}"
        "{{ '<turn>' + item.role + ':' + echo(item.content) + '</turn>' }}"
        "{% endfor %}{% if add_generation_prompt %}"
        "{{ '<turn>assistant:' }}{% endif %}";
    Diagnostic diagnostic;
    std::unique_ptr<internal::SourceTemplate> unit;
    ASSERT_TRUE(internal::SourceTemplate::Parse(source, "shared.jinja", &unit,
                                                &diagnostic));
    const auto *macro = unit->analysis_macros().at("echo");
    const auto *call = unit->analysis_macro_calls().at(0);
    const auto names = unit->source_names();
    TextFormatAnalysis first, repeated, standalone;
    TextUserContinuation user;
    ASSERT_TRUE(internal::AnalyzeTextFormat(*unit, {}, &first, &diagnostic))
        << diagnostic.message;
    ASSERT_TRUE(
        internal::AnalyzeUserContinuation(*unit, {}, first, &user, &diagnostic))
        << diagnostic.message;
    ASSERT_TRUE(internal::AnalyzeTextFormat(*unit, {}, &repeated, &diagnostic))
        << diagnostic.message;
    ASSERT_TRUE(
        AnalyzeTextFormat(source, "shared.jinja", {}, &standalone, &diagnostic))
        << diagnostic.message;
    EXPECT_EQ(first.adaptation_message_binding, "__text_raw_message_");
    EXPECT_EQ(first.adaptation_call_binding, "__text_raw_call_");
    EXPECT_NE(user.text_binding, "__text_continued_user_text");
    EXPECT_EQ(first.predicate_bindings, repeated.predicate_bindings);
    EXPECT_EQ(first.alternatives.size(), repeated.alternatives.size());
    for (bool adaptation : {false, true}) {
      GeneratedInputRenderer before, after, separate;
      auto compile = [&](const TextFormatAnalysis &analysis,
                         GeneratedInputRenderer *out) {
        return CompileTemplateSelector(
            source, "shared.jinja",
            adaptation ? analysis.adaptation_selector_template
                       : analysis.selector_template,
            out, &diagnostic,
            adaptation ? TemplateSelectorKind::Adaptation
                       : TemplateSelectorKind::Format);
      };
      ASSERT_TRUE(compile(first, &before)) << diagnostic.message;
      ASSERT_TRUE(compile(repeated, &after)) << diagnostic.message;
      ASSERT_TRUE(compile(standalone, &separate)) << diagnostic.message;
      EXPECT_EQ(before.source, after.source);
      EXPECT_EQ(before.source, separate.source);
      EXPECT_EQ(before.external_bindings, after.external_bindings);
      EXPECT_EQ(before.protected_bindings, after.protected_bindings);
    }
    GeneratedInputRenderer renderer, separate_renderer;
    ASSERT_TRUE(internal::CompileChatTemplate(*unit, &renderer, &diagnostic))
        << diagnostic.message;
    ASSERT_TRUE(CompileChatTemplate(source, "shared.jinja", &separate_renderer,
                                    &diagnostic));
    EXPECT_EQ(renderer.source, separate_renderer.source);
    EXPECT_EQ(renderer.external_bindings, separate_renderer.external_bindings);
    EXPECT_EQ(renderer.protected_bindings,
              separate_renderer.protected_bindings);
    // This source must not prepare native programs during construction.
    const TemplateAnalysisOptions unused{"/unavailable", "/unavailable", false};
    ASSERT_TRUE(internal::AnalyzeTemplate(*unit, renderer, {}, unused, &owned,
                                          &diagnostic))
        << diagnostic.message;
    AnalyzedTemplate separate;
    ASSERT_TRUE(AnalyzeTemplate(source, "shared.jinja", separate_renderer, {},
                                unused, &separate, &diagnostic))
        << diagnostic.message;
    ASSERT_TRUE(owned.continuation.has_value())
        << owned.continuation_diagnostic.message;
    ASSERT_TRUE(separate.continuation.has_value())
        << separate.continuation_diagnostic.message;
    EXPECT_EQ(owned.selector.source, separate.selector.source);
    EXPECT_EQ(owned.adaptation_selector.source,
              separate.adaptation_selector.source);
    EXPECT_EQ(owned.continuation->renderer.source,
              separate.continuation->renderer.source);
    EXPECT_EQ(owned.continuation->text_binding, user.text_binding);
    EXPECT_EQ(owned.continuation->assistant_closure, user.assistant_closure);
    EXPECT_EQ(unit->analysis_macros().at("echo"), macro);
    EXPECT_EQ(unit->analysis_macro_calls().at(0), call);
    EXPECT_EQ(unit->source_names(), names);
  }
  // Both the caller's source storage and its indexed AST owner are gone before
  // any native preparation/execution of the retained generated products.
  for (bool bytecode : {false, true}) {
    const auto options = Options(bytecode);
    std::unique_ptr<PreparedTemplate> renderer, selector, adaptation, user;
    std::string error;
    for (const auto &[generated, prepared] :
         std::vector<std::pair<const GeneratedInputRenderer *,
                               std::unique_ptr<PreparedTemplate> *>>{
             {&owned.renderer, &renderer},
             {&owned.selector, &selector},
             {&owned.adaptation_selector, &adaptation},
             {&owned.continuation->renderer, &user}})
      ASSERT_TRUE(PreparedTemplate::Prepare(*generated, options.helper_root,
                                            options.scratch_parent, bytecode,
                                            prepared, &error))
          << error;
    constexpr std::string_view environment =
        R"({"messages":[{"role":"user","content":"U"}],"add_generation_prompt":true})";
    std::string rendered;
    ASSERT_TRUE(renderer->Render(environment, &rendered, &error)) << error;
    EXPECT_EQ(rendered, "<turn>user:U</turn><turn>assistant:");
    int64_t id = -1, mode = -1;
    ASSERT_TRUE(selector->SelectFormat(environment, &id, &mode, &error))
        << error;
    ASSERT_GE(id, 0);
    ASSERT_LT(static_cast<size_t>(id), owned.format.alternatives.size());
    const auto &format = owned.format.alternatives[id];
    EXPECT_EQ(format.generation_prefix, "<turn>assistant:");
    EXPECT_EQ(format.assistant_header, "<turn>assistant:");
    EXPECT_EQ(format.frame_suffix, "</turn>");
    EXPECT_EQ(mode, 0);
    ASSERT_TRUE(adaptation->SelectFormat(environment, &id, &mode, &error))
        << error;
    EXPECT_EQ(id, 0);
    EXPECT_EQ(mode, 0);
    ASSERT_TRUE(
        user->Render("{\"" + owned.continuation->text_binding + "\":\"NEXT\"}",
                     &rendered, &error))
        << error;
    EXPECT_EQ(rendered, "<turn>user:NEXT</turn><turn>assistant:");
  }
}

TEST(TemplateAnalysisTest, NoEffortDiscoveryStillBuildsUserWithoutNativeSetup) {
  std::ifstream stream(Root() / "lib/text/qwen3/templates/5da44855.jinja");
  ASSERT_TRUE(stream);
  const std::string reasoning_source(std::istreambuf_iterator<char>(stream),
                                     {});
  const auto effort_without_reasoning =
      std::string("{% set effort = reasoning_effort | default('high') %}") +
      kPlain;
  for (const auto &source :
       {std::string(kPlain), reasoning_source, effort_without_reasoning}) {
    GeneratedInputRenderer input;
    Diagnostic diagnostic;
    ASSERT_TRUE(
        CompileChatTemplate(source, "no-effort.jinja", &input, &diagnostic))
        << diagnostic.message;
    if (source == effort_without_reasoning)
      EXPECT_NE(std::find(input.external_bindings.begin(),
                          input.external_bindings.end(), "reasoning_effort"),
                input.external_bindings.end());
    for (bool bytecode : {false, true}) {
      auto options = Options(bytecode);
      options.helper_root += "/does-not-exist";
      options.scratch_parent += "/does-not-exist";
      AnalyzedTemplate result;
      ASSERT_TRUE(AnalyzeTemplate(source, "no-effort.jinja", input, {}, options,
                                  &result, &diagnostic))
          << diagnostic.message;
      EXPECT_EQ(result.reasoning.enabled, source == reasoning_source);
      EXPECT_EQ(result.reasoning.effort_mask, 0);
      EXPECT_EQ(result.reasoning.default_effort, 0);
      ASSERT_TRUE(result.continuation.has_value())
          << result.continuation_diagnostic.message;
      EXPECT_FALSE(result.continuation->renderer.source.empty());
    }
  }
}

TEST(TemplateAnalysisTest,
     QualificationRejectsInconsistentProductsAndKeepsReport) {
  GeneratedInputRenderer input;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      CompileChatTemplate(kPlain, "qualification.jinja", &input, &diagnostic));
  TextFormatBindings globals;
  globals.strings["bos_token"] = "GGUF-BOS";
  for (bool bytecode : {false, true}) {
    AnalyzedTemplate result;
    ASSERT_TRUE(AnalyzeTemplate(kPlain, "qualification.jinja", input, globals,
                                Options(bytecode), &result, &diagnostic))
        << diagnostic.message;
    for (auto &format : result.format.alternatives)
      format.generation_prefix = "not-the-original-generation-prefix";
    TemplateQualification report;
    report.validated_probe_count = 101;
    EXPECT_FALSE(QualifyTemplate(kPlain, "qualification.jinja", globals, result,
                                 Options(bytecode), &report, &diagnostic));
    EXPECT_NE(diagnostic.message.find("generation prefix/startup disagrees"),
              std::string::npos)
        << diagnostic.message;
    EXPECT_EQ(report.validated_probe_count, 101u);
    EXPECT_EQ(result.renderer.source, input.source);
  }
}

TEST(TemplateAnalysisTest, CompleteCoverageStillChecksRestOfRouteCandidate) {
  GeneratedInputRenderer input;
  Diagnostic diagnostic;
  ASSERT_TRUE(CompileChatTemplate(kPlain, "candidate-entry.jinja", &input,
                                  &diagnostic));
  TextFormatBindings globals;
  globals.strings["bos_token"] = "GGUF-BOS";
  for (bool bytecode : {false, true}) {
    AnalyzedTemplate result;
    ASSERT_TRUE(AnalyzeTemplate(kPlain, "candidate-entry.jinja", input, globals,
                                Options(bytecode), &result, &diagnostic))
        << diagnostic.message;
    ASSERT_EQ(result.format.alternatives.size(), 2u);
    const size_t completed =
        result.format.alternatives[0].generation_prefix.empty() ? 0 : 1;
    const size_t live = 1 - completed;
    const size_t extra = result.format.alternatives.size();
    // This test-only route is absent from the original source's independent
    // controls. Its first context completes coverage, but its next
    // tools-defined context deliberately returns an invalid route and must
    // still be checked.
    result.format.alternatives.push_back(result.format.alternatives[completed]);
    result.renderer.external_bindings.push_back("extra_route");
    Expr route;
    route.kind = Expr::Kind::Name;
    route.text = "extra_route";
    Expr prompt;
    prompt.kind = Expr::Kind::Name;
    prompt.text = "add_generation_prompt";
    Expr no_prompt;
    no_prompt.kind = Expr::Kind::Unary;
    no_prompt.text = "not";
    no_prompt.args.push_back(std::move(prompt));
    result.format.format_route_guards.push_back({route, no_prompt});
    const std::string selector_source =
        "{% if add_generation_prompt | default(true) %}{{ [" +
        std::to_string(live) +
        ",0] }}{% elif extra_route | default(false) %}"
        "{% if tools is defined %}{{ [" +
        std::to_string(extra + 1) + ",0] }}{% else %}{{ [" +
        std::to_string(extra) + ",0] }}{% endif %}{% else %}{{ [" +
        std::to_string(completed) + ",0] }}{% endif %}";
    Template selector;
    ASSERT_TRUE(ParseTemplate(selector_source, &selector, &diagnostic));
    ASSERT_TRUE(CompileTemplateSelector(selector_source,
                                        "candidate-entry.jinja", selector,
                                        &result.selector, &diagnostic))
        << diagnostic.message;
    TemplateQualification report;
    report.validated_probe_count = 101;
    EXPECT_FALSE(QualifyTemplate(kPlain, "candidate-entry.jinja", globals,
                                 result, Options(bytecode), &report,
                                 &diagnostic));
    EXPECT_NE(diagnostic.message.find("invalid format/startup"),
              std::string::npos)
        << diagnostic.message;
    EXPECT_EQ(report.validated_probe_count, 101u);
  }
}

TEST(TemplateAnalysisTest, OrderedContextDiscoveryStopsWithoutCoverage) {
  GeneratedInputRenderer input;
  Template original;
  TextFormatAnalysis analysis;
  TextFormatBindings globals;
  globals.strings["bos_token"] = "IMMUTABLE-GGUF-GLOBAL";
  Diagnostic diagnostic;
  ASSERT_TRUE(
      CompileChatTemplate(kPlain, "ordered.jinja", &input, &diagnostic));
  ASSERT_TRUE(ParseTemplate(kPlain, &original, &diagnostic));
  ASSERT_TRUE(AnalyzeTextFormat(kPlain, "ordered.jinja", globals, &analysis,
                                &diagnostic))
      << diagnostic.message;
  std::vector<std::pair<int, int>> observed;
  ASSERT_TRUE(internal::VisitTemplateInputs(
      original, input, analysis, globals,
      [&](const internal::TemplateInput &candidate) {
        EXPECT_EQ(candidate.candidate_index, 0u);
        EXPECT_FALSE(candidate.route_candidate);
        EXPECT_EQ(candidate.environment.at("bos_token"),
                  internal::Quote("IMMUTABLE-GGUF-GLOBAL"));
        observed.emplace_back(candidate.history_kind, candidate.tools_kind);
        return observed.size() == 5 ? internal::InputVisit::Stop
                                    : internal::InputVisit::Continue;
      }));
  EXPECT_EQ(observed, (std::vector<std::pair<int, int>>{
                          {0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}}));
}

TEST(TemplateAnalysisTest, FirstActiveEffortContextControlsSuccessOrFailure) {
  const auto original = OriginalSpark();
  ASSERT_FALSE(original.empty());
  for (bool reject_first : {false, true}) {
    // The default context has absent tools. Its full finite effort domain is
    // valid only in the success case; later tool contexts have the inverse
    // behavior, so searching past the first active context changes the result.
    const std::string prefix =
        std::string("{% if tools is ") +
        (reject_first ? "undefined" : "defined") +
        " and reasoning_effort is defined and reasoning_effort == 'low' %}"
        "{{ raise_exception('this context rejects low effort') }}{% endif %}";
    const auto source = prefix + original;
    GeneratedInputRenderer input;
    Diagnostic diagnostic;
    ASSERT_TRUE(
        CompileChatTemplate(source, "first-active.jinja", &input, &diagnostic))
        << diagnostic.message;
    for (bool bytecode : {false, true}) {
      AnalyzedTemplate result;
      result.renderer.source = "previous-product";
      const bool analyzed =
          AnalyzeTemplate(source, "first-active.jinja", input, {},
                          Options(bytecode), &result, &diagnostic);
      if (reject_first) {
        EXPECT_FALSE(analyzed);
        EXPECT_EQ(result.renderer.source, "previous-product");
        EXPECT_NE(diagnostic.message.find("effort admission disagrees"),
                  std::string::npos)
            << diagnostic.message;
      } else {
        ASSERT_TRUE(analyzed) << diagnostic.message;
        EXPECT_TRUE(result.reasoning.enabled);
        EXPECT_EQ(result.reasoning.effort_mask, 20);
        EXPECT_EQ(result.reasoning.default_effort, 4);
      }
    }
  }
}

TEST(TemplateAnalysisTest, EffortPreparationFailurePreservesProducts) {
  const auto source = OriginalSpark();
  GeneratedInputRenderer input;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      CompileChatTemplate(source, "effort-prepare.jinja", &input, &diagnostic));
  auto options = Options(false);
  options.helper_root += "/does-not-exist";
  AnalyzedTemplate result;
  result.renderer.source = "previous-product";
  EXPECT_FALSE(AnalyzeTemplate(source, "effort-prepare.jinja", input, {},
                               options, &result, &diagnostic));
  EXPECT_EQ(result.renderer.source, "previous-product");
  EXPECT_EQ(diagnostic.source_name, "effort-prepare.jinja");
  EXPECT_NE(diagnostic.message.find("prepare compiled template analysis"),
            std::string::npos)
      << diagnostic.message;
}

TEST(TemplateAnalysisTest, InactiveEffortGuardExhaustionDoesNotPublishFacts) {
  auto source = OriginalSpark();
  const std::string guard =
      "{%- if resolved_reasoning_effort not in ('high', 'low') %}";
  const auto begin = source.find(guard);
  ASSERT_NE(begin, std::string::npos);
  const std::string end_tag = "{%- endif %}";
  const auto end = source.find(end_tag, begin);
  ASSERT_NE(end, std::string::npos);
  source.insert(end + end_tag.size(), end_tag);
  source.insert(begin, "{%- if false %}");
  GeneratedInputRenderer input;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      CompileChatTemplate(source, "inactive-effort.jinja", &input, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    AnalyzedTemplate result;
    result.renderer.source = "previous-product";
    EXPECT_FALSE(AnalyzeTemplate(source, "inactive-effort.jinja", input, {},
                                 Options(bytecode), &result, &diagnostic));
    EXPECT_EQ(result.renderer.source, "previous-product");
    EXPECT_NE(diagnostic.message.find("no source-valid reasoning context"),
              std::string::npos)
        << diagnostic.message;
  }
}

TEST(TemplateAnalysisTest, ExecutesOwnedProductsAndCleansPreparedPrograms) {
  TextFormatBindings globals;
  globals.strings["bos_token"] = "EXPLICIT_GGUF_GLOBAL";
  GeneratedInputRenderer input;
  Diagnostic compile_diagnostic;
  ASSERT_TRUE(CompileChatTemplate(kPlain, "ordinary.jinja", &input,
                                  &compile_diagnostic));
  input.source += "\n// caller-owned compiled renderer\n";
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    const auto options = Options(bytecode);
    const auto count_children = [&] {
      return static_cast<size_t>(std::distance(
          std::filesystem::directory_iterator(options.scratch_parent),
          std::filesystem::directory_iterator()));
    };
    const auto before = count_children();
    AnalyzedTemplate result;
    TemplateQualification qualification;
    Diagnostic diagnostic;
    ASSERT_TRUE(AnalyzeTemplate(kPlain, "ordinary.jinja", input, globals,
                                options, &result, &diagnostic))
        << diagnostic.line << ':' << diagnostic.column << ':'
        << diagnostic.message;
    ASSERT_FALSE(result.format.alternatives.empty());
    EXPECT_EQ(result.format.alternatives[0].assistant_header,
              "<turn>assistant:");
    EXPECT_EQ(result.format.alternatives[0].frame_suffix, "</turn>");
    EXPECT_EQ(result.format.alternatives[0].tools.kind, TextToolKind::None);
    ASSERT_TRUE(QualifyTemplate(kPlain, "ordinary.jinja", globals, result,
                                options, &qualification, &diagnostic))
        << diagnostic.message;
    EXPECT_GE(qualification.validated_probe_count, 2u);
    EXPECT_FALSE(result.renderer.source.empty());
    EXPECT_EQ(result.renderer.source, input.source);
    EXPECT_FALSE(result.selector.source.empty());
    EXPECT_FALSE(result.adaptation_selector.source.empty());
    EXPECT_EQ(count_children(), before);
  }
}

TEST(TemplateAnalysisTest, PreparationFailurePreservesAllOwnedOutputs) {
  GeneratedInputRenderer input;
  AnalyzedTemplate result;
  Diagnostic diagnostic;
  ASSERT_TRUE(
      CompileChatTemplate(kPlain, "ordinary.jinja", &input, &diagnostic));
  ASSERT_TRUE(AnalyzeTemplate(kPlain, "ordinary.jinja", input, {},
                              Options(false), &result, &diagnostic))
      << diagnostic.message;
  const auto previous_source = result.renderer.source;
  TemplateQualification qualification;
  qualification.validated_probe_count = 101;
  auto options = Options(false);
  options.helper_root += "/missing-helper-root";
  EXPECT_FALSE(QualifyTemplate(kPlain, "ordinary.jinja", {}, result, options,
                               &qualification, &diagnostic));
  EXPECT_EQ(result.renderer.source, previous_source);
  EXPECT_EQ(qualification.validated_probe_count, 101u);
  EXPECT_EQ(diagnostic.source_name, "ordinary.jinja");
  EXPECT_NE(diagnostic.message.find("prepare compiled template analysis"),
            std::string::npos);
}

TEST(TemplateAnalysisTest, CompletionDoesNotReuseLiveGenerationTail) {
  const std::string source =
      "{% for item in messages %}"
      "{{ '<turn>' + item.role + ':' + item.content + '</turn>' }}"
      "{% endfor %}{% if add_generation_prompt %}"
      "{% if messages|length > 1 %}{{ '\\n' }}{% endif %}"
      "{{ '<turn>assistant:' }}{% endif %}";
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    AnalyzedTemplate result;
    Diagnostic diagnostic;
    ASSERT_TRUE(CompileAndAnalyze(source, "history-generation.jinja", {},
                                  Options(bytecode), &result, &diagnostic))
        << diagnostic.line << ':' << diagnostic.column << ':'
        << diagnostic.message;
    bool short_history = false, long_history = false;
    for (const auto &format : result.format.alternatives) {
      short_history |= format.generation_prefix == "<turn>assistant:";
      long_history |= format.generation_prefix == "\n<turn>assistant:";
    }
    EXPECT_TRUE(short_history);
    EXPECT_TRUE(long_history);
  }
}

TEST(TemplateAnalysisTest, UnsupportedSourceNeverPublishesPartialAnalysis) {
  AnalyzedTemplate result;
  result.renderer.source = "keep previous source";
  Diagnostic diagnostic;
  EXPECT_FALSE(CompileAndAnalyze("{{ unknown }}", "unsupported.jinja", {},
                                 Options(false), &result, &diagnostic));
  EXPECT_EQ(result.renderer.source, "keep previous source");
  EXPECT_FALSE(diagnostic.message.empty());
  EXPECT_EQ(diagnostic.source_name, "unsupported.jinja");
}

TEST(TemplateAnalysisTest, ColdReasoningTracksSourceDefaultsAndLiteralDomains) {
  const char *path = "lib/text/spark/templates/93d6aa90.jinja";
  std::ifstream stream(Root() / path);
  ASSERT_TRUE(stream);
  const std::string original((std::istreambuf_iterator<char>(stream)), {});
  struct Mutation {
    const char *before;
    const char *after;
    int64_t mask;
    int64_t default_effort;
  };
  const Mutation mutations[] = {
      {"reasoning_effort|default('high')", "reasoning_effort|default('low')",
       20, 2},
      {"('high', 'low')", "('high', 'medium')", 24, 4},
      {"enable_thinking | default(true)", "enable_thinking | default(false)",
       20, 4},
  };
  for (const auto &mutation : mutations) {
    SCOPED_TRACE(mutation.after);
    auto source = original;
    const auto position = source.find(mutation.before);
    ASSERT_NE(position, std::string::npos);
    source.replace(position, std::string(mutation.before).size(),
                   mutation.after);
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      AnalyzedTemplate result;
      Diagnostic diagnostic;
      ASSERT_TRUE(CompileAndAnalyze(source, "changed-source.jinja", {},
                                    Options(bytecode), &result, &diagnostic))
          << diagnostic.line << ':' << diagnostic.column << ':'
          << diagnostic.message;
      EXPECT_TRUE(result.reasoning.enabled);
      EXPECT_EQ(result.reasoning.effort_mask, mutation.mask);
      EXPECT_EQ(result.reasoning.default_effort, mutation.default_effort);
    }
  }
}

TEST(TemplateAnalysisTest, ColdReasoningLiteralNormalizationUsesInputPreimage) {
  const std::string guard =
      "{%- if resolved_reasoning_effort not in ('high', 'low') %}";
  auto original = OriginalSpark();
  const std::string old_default = "reasoning_effort|default('high')";
  const auto position = original.find(old_default);
  ASSERT_NE(position, std::string::npos);
  original.replace(position, old_default.size(),
                   "reasoning_effort|default('xhigh')");
  struct Case {
    const char *setup;
    const char *subject;
    const char *domain;
    int64_t mask;
  };
  for (const auto &test : {
           Case{"{% if resolved_reasoning_effort == 'high' %}"
                "{% set resolved_reasoning_effort='xhigh' %}{% endif %}",
                "resolved_reasoning_effort", "('xhigh','medium','low')", 60},
           Case{"{% set copied=resolved_reasoning_effort %}"
                "{% if resolved_reasoning_effort == 'high' %}"
                "{% set resolved_reasoning_effort='xhigh' %}{% endif %}",
                "copied", "('xhigh','medium','low')", 44},
           Case{"{% if resolved_reasoning_effort == 'high' %}"
                "{% set resolved_reasoning_effort='xhigh' %}{% endif %}"
                "{% set copied=resolved_reasoning_effort %}",
                "copied", "('xhigh','medium','low')", 60},
           Case{"{% if resolved_reasoning_effort == 'high' %}"
                "{% set resolved_reasoning_effort='none' %}{% endif %}",
                "resolved_reasoning_effort", "('xhigh','high','medium','low')",
                44},
           // A different assignment target does not normalize this binding.
           Case{"{% if resolved_reasoning_effort == 'high' %}"
                "{% set other='xhigh' %}{% endif %}",
                "resolved_reasoning_effort", "('xhigh','medium','low')", 44},
       }) {
    SCOPED_TRACE(test.setup);
    auto source = original;
    const auto start = source.find(guard);
    ASSERT_NE(start, std::string::npos);
    source.replace(start, guard.size(),
                   std::string(test.setup) + "{% if " + test.subject +
                       " not in " + test.domain + " %}");
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      AnalyzedTemplate result;
      Diagnostic diagnostic;
      ASSERT_TRUE(CompileAndAnalyze(source, "literal-normalization.jinja", {},
                                    Options(bytecode), &result, &diagnostic))
          << diagnostic.message;
      EXPECT_EQ(result.reasoning.effort_mask, test.mask);
      EXPECT_EQ(result.reasoning.default_effort, 5);
      std::unique_ptr<PreparedTemplate> renderer;
      std::string error;
      ASSERT_TRUE(PreparedTemplate::Prepare(
          result.renderer, Options(bytecode).helper_root,
          Options(bytecode).scratch_parent, bytecode, &renderer, &error))
          << error;
      for (const char *effort : {"\"low\"", "\"medium\"", "\"high\"",
                                 "\"xhigh\"", "\"unknown\"", "null", "false"}) {
        const std::string context =
            "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],"
            "\"add_generation_prompt\":true,\"reasoning_effort\":" +
            std::string(effort);
        const bool valid =
            std::string_view(effort) == "\"low\"" ||
            std::string_view(effort) == "\"medium\"" ||
            std::string_view(effort) == "\"xhigh\"" ||
            (std::string_view(effort) == "\"high\"" && test.mask == 60);
        std::string rendered;
        EXPECT_EQ(renderer->Render("{" + context + "}", &rendered, &error),
                  valid)
            << effort << ": " << error;
        EXPECT_TRUE(renderer->Render(
            "{\"enable_thinking\":false," + context + "}", &rendered, &error))
            << error;
      }
    }
  }
}

TEST(TemplateAnalysisTest, ColdReasoningUnprovedRewritesKillOnlyTheirBinding) {
  const std::string guard =
      "{%- if resolved_reasoning_effort not in ('high', 'low') %}";
  for (
      const char *setup : {
          "{% if resolved_reasoning_effort == 'low' %}"
          "{% set resolved_reasoning_effort=other %}{% endif %}",
          "{% if resolved_reasoning_effort == 'low' %}"
          "{% set resolved_reasoning_effort='high' %}{{ 'effect' }}{% endif %}",
          "{% if resolved_reasoning_effort == 'low' %}"
          "{% set resolved_reasoning_effort='high' %}{% else %}"
          "{% set resolved_reasoning_effort='low' %}{% endif %}",
          "{% if resolved_reasoning_effort == 'low' %}"
          "{% set resolved_reasoning_effort='high' %}{% endif %}"
          "{% set resolved_reasoning_effort='high' %}",
          "{% if resolved_reasoning_effort == 'low' %}"
          "{% set resolved_reasoning_effort='high' %}{% endif %}"
          "{% if resolved_reasoning_effort == 'high' %}"
          "{% set resolved_reasoning_effort='low' %}{% endif %}",
      }) {
    SCOPED_TRACE(setup);
    auto source = OriginalSpark();
    const auto start = source.find(guard);
    ASSERT_NE(start, std::string::npos);
    source.insert(start, setup);
    for (bool bytecode : {false, true}) {
      AnalyzedTemplate result;
      result.renderer.source = "preserved";
      Diagnostic diagnostic;
      EXPECT_FALSE(CompileAndAnalyze(source, "unproved-normalization.jinja", {},
                                     Options(bytecode), &result, &diagnostic));
      EXPECT_EQ(result.renderer.source, "preserved");
      EXPECT_NE(diagnostic.message.find("source-qualified finite literal"),
                std::string::npos)
          << diagnostic.message;
    }
  }
}

TEST(TemplateAnalysisTest, PlainCurrentSourceUsesExactNativeRecipe) {
  const char *path = "lib/text/plain_chatml/templates/58c1a1f0.jinja";
  std::ifstream stream(Root() / path);
  ASSERT_TRUE(stream);
  const std::string source((std::istreambuf_iterator<char>(stream)), {});
  TextFormatBindings globals;
  globals.strings["bos_token"] = "<probe-bos>";
  globals.strings["eos_token"] = "<probe-eos>";
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    AnalyzedTemplate result;
    TemplateQualification qualification;
    Diagnostic diagnostic;
    ASSERT_TRUE(CompileAndAnalyze(source, path, globals, Options(bytecode),
                                  &result, &diagnostic))
        << diagnostic.line << ':' << diagnostic.column << ':'
        << diagnostic.message;
    ASSERT_TRUE(QualifyTemplate(source, path, globals, result,
                                Options(bytecode), &qualification, &diagnostic))
        << diagnostic.message;
    ASSERT_FALSE(result.format.alternatives.empty());
    for (const auto &format : result.format.alternatives) {
      EXPECT_EQ(format.tools.kind, TextToolKind::None);
      EXPECT_EQ(format.assistant_header, "<|im_start|>assistant\n");
      EXPECT_EQ(format.frame_suffix, "<|im_end|>\n");
    }
    EXPECT_GE(qualification.validated_probe_count, 4u);
  }
}

TEST(TemplateAnalysisTest, ConditionalToolFieldsKeepIdenticalGenerationPrefix) {
  const std::string base_source =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{{ '<turn>assistant:' + m.content }}{% for c in m.tool_calls %}"
      "{% if alternate_fields | default(false) %}"
      "{{ '<invoke>{\"operation\":\"' + c.function.name + '\",\"inputs\":' }}"
      "{% else %}"
      "{{ '<invoke>{\"name\":\"' + c.function.name + '\",\"arguments\":' }}"
      "{% endif %}{{ c.function.arguments | tojson }}{{ '}</invoke>' }}"
      "{% endfor %}{{ '</turn>' }}{% else %}"
      "{{ '<turn>' + m.role + ':' + m.content + '</turn>' }}"
      "{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
  for (const char *guard : {"alternate_fields | default(false)",
                            "((primary_flag | default(false)) or "
                            "(secondary_flag | default(false))) and not "
                            "(primary_flag | default(false))"}) {
    SCOPED_TRACE(guard);
    std::string source = base_source;
    const std::string original_guard = "alternate_fields | default(false)";
    source.replace(source.find(original_guard), original_guard.size(), guard);
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      AnalyzedTemplate result;
      Diagnostic diagnostic;
      ASSERT_TRUE(CompileAndAnalyze(source, "conditional.jinja", {},
                                    Options(bytecode), &result, &diagnostic))
          << diagnostic.line << ':' << diagnostic.column << ':'
          << diagnostic.message;
      std::unique_ptr<PreparedTemplate> selector, renderer;
      std::string error;
      const auto options = Options(bytecode);
      ASSERT_TRUE(PreparedTemplate::Prepare(
          result.selector, options.helper_root, options.scratch_parent,
          bytecode, &selector, &error))
          << error;
      ASSERT_TRUE(PreparedTemplate::Prepare(
          result.renderer, options.helper_root, options.scratch_parent,
          bytecode, &renderer, &error))
          << error;
      std::string first_prompt;
      for (bool alternate : {false, true}) {
        const std::string environment =
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],"
            "\"add_generation_prompt\":true,\"primary_flag\":false,"
            "\"secondary_flag\":" +
            std::string(alternate ? "true" : "false") +
            ",\"alternate_fields\":" +
            std::string(alternate ? "true}" : "false}");
        int64_t id = -1, mode = -1;
        ASSERT_TRUE(selector->SelectFormat(environment, &id, &mode, &error))
            << error;
        ASSERT_GE(id, 0);
        ASSERT_LT(static_cast<size_t>(id), result.format.alternatives.size());
        const auto &format = result.format.alternatives[id];
        EXPECT_EQ(format.tools.kind, TextToolKind::JsonEnvelope);
        EXPECT_EQ(format.tools.name_field, alternate ? "operation" : "name");
        EXPECT_EQ(format.tools.arguments_field,
                  alternate ? "inputs" : "arguments");
        EXPECT_EQ(format.generation_prefix, "<turn>assistant:");
        EXPECT_EQ(mode, 0);
        std::string prompt;
        ASSERT_TRUE(renderer->Render(environment, &prompt, &error)) << error;
        if (alternate)
          EXPECT_EQ(prompt, first_prompt);
        else
          first_prompt = std::move(prompt);
      }
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(selector->SelectFormat(
          "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],"
          "\"add_generation_prompt\":false}",
          &id, &mode, &error))
          << error;
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.format.alternatives.size());
      EXPECT_TRUE(result.format.alternatives[id].generation_prefix.empty());
      EXPECT_EQ(mode, 0);
    }
  }
}

TEST(TemplateAnalysisTest, TargetMessageCountsPreserveSequenceAndSourcePoint) {
  struct CountCase {
    const char *name;
    const char *setup;
    const char *sequence;
    const char *count;
    const char *expected_suffix;
  };
  const CountCase cases[] = {
      {"direct", "", "messages", "messages | length", "</final>"},
      {"module-alias", "{% set total = messages | length %}", "messages",
       "total", "</final>"},
      {"sliced-count",
       "{% set work = messages[1:] %}"
       "{% set total = work | length %}",
       "work", "total", "</final>"},
      // The root count was cached before slicing. Treating every target as
      // index root_count-1, or re-evaluating the cached count after the slice,
      // would incorrectly choose </final> for this source.
      {"root-count-before-slice",
       "{% set total = messages | length %}"
       "{% set work = messages[1:] %}",
       "work", "total", "</earlier>"},
  };
  for (const auto &fixture : cases) {
    SCOPED_TRACE(fixture.name);
    const auto count = std::string("(") + fixture.count + ')';
    const std::string source =
        std::string(fixture.setup) + "{% for m in " + fixture.sequence +
        " %}{% if m.role == 'assistant' %}"
        "{{ '<turn>assistant:' + m.content }}"
        "{% if loop.index0 == " +
        count + " - 1 and loop.length == " + count +
        " %}{{ '</final>' }}{% else %}{{ '</earlier>' }}{% endif %}"
        "{% else %}{{ '<turn>' + m.role + ':' + m.content + '</user>' }}"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      AnalyzedTemplate result;
      Diagnostic diagnostic;
      ASSERT_TRUE(CompileAndAnalyze(source, fixture.name, {}, Options(bytecode),
                                    &result, &diagnostic))
          << diagnostic.line << ':' << diagnostic.column << ':'
          << diagnostic.message;
      std::unique_ptr<PreparedTemplate> selector;
      std::string error;
      const auto options = Options(bytecode);
      ASSERT_TRUE(PreparedTemplate::Prepare(
          result.selector, options.helper_root, options.scratch_parent,
          bytecode, &selector, &error))
          << error;
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(selector->SelectFormat(
          "{\"messages\":[{\"role\":\"system\",\"content\":\"S\"},"
          "{\"role\":\"user\",\"content\":\"U\"}],"
          "\"add_generation_prompt\":true}",
          &id, &mode, &error))
          << error;
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.format.alternatives.size());
      EXPECT_EQ(result.format.alternatives[id].frame_suffix,
                fixture.expected_suffix);
      EXPECT_EQ(mode, 0);
    }
  }
}

TEST(TemplateAnalysisTest, CompletedCountControlsDerivedDelimiter) {
  const char *source =
      "{% set total = messages | length %}"
      "{% if total > 3 %}{% set close = '</many>' %}"
      "{% else %}{% set close = '</few>' %}{% endif %}"
      "{% for m in messages %}"
      "{{ '<turn>' + m.role + ':' + m.content + close }}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    AnalyzedTemplate result;
    Diagnostic diagnostic;
    ASSERT_TRUE(CompileAndAnalyze(source, "count-delimiter.jinja", {},
                                  Options(bytecode), &result, &diagnostic))
        << diagnostic.line << ':' << diagnostic.column << ':'
        << diagnostic.message;
    std::unique_ptr<PreparedTemplate> selector;
    std::string error;
    const auto options = Options(bytecode);
    ASSERT_TRUE(PreparedTemplate::Prepare(result.selector, options.helper_root,
                                          options.scratch_parent, bytecode,
                                          &selector, &error))
        << error;
    for (const auto &[messages, expected] :
         {std::pair<const char *, const char *>{
              "[{\"role\":\"user\",\"content\":\"U\"}]", "</few>"},
          {"[{\"role\":\"user\",\"content\":\"U\"},"
           "{\"role\":\"assistant\",\"content\":\"A\"},"
           "{\"role\":\"user\",\"content\":\"U2\"}]",
           "</many>"}}) {
      // With three original messages, the conceptual completed count is four.
      // Using the raw selector count would choose </few> incorrectly.
      int64_t id = -1, mode = -1;
      ASSERT_TRUE(selector->SelectFormat(std::string("{\"messages\":") +
                                             messages +
                                             ",\"add_generation_prompt\":true}",
                                         &id, &mode, &error))
          << error;
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.format.alternatives.size());
      EXPECT_EQ(result.format.alternatives[id].frame_suffix, expected);
    }
  }
}

std::string ClosureSource(std::string_view setup) {
  return std::string(setup) +
         "{% for m in messages %}{% if m.role == 'assistant' %}"
         "{{ '<turn>assistant:' + m.content }}"
         "{% if alternate %}{{ '<alternate_end>' }}"
         "{% else %}{{ '<default_end>' }}{% endif %}"
         "{% else %}{{ '<turn>' + m.role + ':' + m.content + '</user>' }}"
         "{% endif %}{% endfor %}"
         "{% if add_generation_prompt %}{{ '<turn>assistant:' }}{% endif %}";
}

TEST(TemplateAnalysisTest,
     UserClosureUsesSourceDefaultsWithoutRestrictingChat) {
  for (bool alternate_default : {false, true}) {
    const auto source =
        ClosureSource(std::string("{% set alternate = style | default(") +
                      (alternate_default ? "true" : "false") + ") %}");
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(alternate_default ? "default true" : "default false");
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      AnalyzedTemplate result;
      Diagnostic diagnostic;
      ASSERT_TRUE(CompileAndAnalyze(source, "closure-default.jinja", {},
                                    Options(bytecode), &result, &diagnostic))
          << diagnostic.message;
      ASSERT_TRUE(result.continuation.has_value())
          << result.continuation_diagnostic.message;
      EXPECT_EQ(result.continuation->assistant_closure,
                alternate_default ? "<alternate_end>" : "<default_end>");
      const auto options = Options(bytecode);
      std::unique_ptr<PreparedTemplate> selector, original, continuation;
      std::string error;
      ASSERT_TRUE(PreparedTemplate::Prepare(
          result.selector, options.helper_root, options.scratch_parent,
          bytecode, &selector, &error))
          << error;
      ASSERT_TRUE(PreparedTemplate::Prepare(
          result.renderer, options.helper_root, options.scratch_parent,
          bytecode, &original, &error))
          << error;
      ASSERT_TRUE(PreparedTemplate::Prepare(
          result.continuation->renderer, options.helper_root,
          options.scratch_parent, bytecode, &continuation, &error))
          << error;
      std::string rendered;
      ASSERT_TRUE(continuation->Render(
          "{\"" + result.continuation->text_binding + "\":\"USER\"}", &rendered,
          &error))
          << error;
      EXPECT_EQ(rendered, "<turn>user:USER</user><turn>assistant:");
      for (bool alternate : {false, true}) {
        const std::string option = alternate ? "true" : "false";
        int64_t id = -1, mode = -1;
        ASSERT_TRUE(selector->SelectFormat(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"U\"}],"
            "\"add_generation_prompt\":true,\"style\":" +
                option + '}',
            &id, &mode, &error))
            << error;
        ASSERT_GE(id, 0);
        ASSERT_LT(static_cast<size_t>(id), result.format.alternatives.size());
        const std::string expected =
            alternate ? "<alternate_end>" : "<default_end>";
        EXPECT_EQ(result.format.alternatives[id].frame_suffix, expected);
        ASSERT_TRUE(original->Render(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"U\"},"
            "{\"role\":\"assistant\",\"content\":\"A\"}],"
            "\"add_generation_prompt\":false,\"style\":" +
                option + '}',
            &rendered, &error))
            << error;
        ASSERT_GE(rendered.size(), expected.size());
        EXPECT_EQ(rendered.substr(rendered.size() - expected.size()), expected);
      }
    }
  }
}

TEST(TemplateAnalysisTest,
     HistoryDependentClosureKeepsStructuredChatAvailable) {
  const auto source =
      ClosureSource("{% set alternate = messages[0].role == 'system' %}");
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    AnalyzedTemplate result;
    Diagnostic diagnostic;
    ASSERT_TRUE(CompileAndAnalyze(source, "history-closure.jinja", {},
                                  Options(bytecode), &result, &diagnostic))
        << diagnostic.message;
    EXPECT_FALSE(result.continuation.has_value());
    EXPECT_FALSE(result.continuation_diagnostic.message.empty());
    EXPECT_FALSE(result.renderer.source.empty());
    EXPECT_FALSE(result.selector.source.empty());
  }
}

void CheckContinuation(const AnalyzedTemplate &result, std::string_view name,
                       bool bytecode) {
  struct Expected {
    const char *name;
    const char *prefix;
    const char *suffix;
    const char *closure;
    bool trim;
  };
  // Independent legacy USER suffix expectations. The generated recipe and
  // original source are both checked against these bytes, not each other.
  const Expected expectations[] = {
      {"Llama", "<|start_header_id|>user<|end_header_id|>\n\n",
       "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
       "<|eot_id|>", true},
      {"Qwen08", "<|im_start|>user\n",
       "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
       "<|im_end|>\n", true},
      {"Qwen27", "<|im_start|>user\n",
       "<|im_end|>\n<|im_start|>assistant\n<think>\n", "<|im_end|>\n", true},
      {"Plain", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n",
       "<|im_end|>\n", false},
      {"Qwen25", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n",
       "<|im_end|>\n", false},
      {"Qwen3", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n",
       "<|im_end|>\n", false},
      {"Gemma", "<start_of_turn>user\n",
       "<end_of_turn>\n<start_of_turn>model\n", "<end_of_turn>\n", true},
      {"Compact", "<|im_start|>user\n",
       "<|im_end|>\n<|im_start|>assistant\n<think>", "<|im_end|>\n", false},
      {"Spark", "<｜start▁of▁sentence｜><|User|>",
       "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|><think>",
       "<｜end▁of▁sentence｜>", false},
  };
  const Expected *expected = nullptr;
  for (const auto &candidate : expectations)
    if (candidate.name == name)
      expected = &candidate;
  ASSERT_NE(expected, nullptr);
  ASSERT_TRUE(result.continuation.has_value())
      << result.continuation_diagnostic.line << ':'
      << result.continuation_diagnostic.column << ':'
      << result.continuation_diagnostic.message;
  const auto &product = *result.continuation;
  EXPECT_EQ(product.assistant_closure, expected->closure);
  const auto options = Options(bytecode);
  std::unique_ptr<PreparedTemplate> continuation, original;
  std::string error;
  ASSERT_TRUE(PreparedTemplate::Prepare(product.renderer, options.helper_root,
                                        options.scratch_parent, bytecode,
                                        &continuation, &error))
      << error;
  ASSERT_TRUE(PreparedTemplate::Prepare(result.renderer, options.helper_root,
                                        options.scratch_parent, bytecode,
                                        &original, &error))
      << error;
  struct Text {
    const char *json;
    const char *raw;
    const char *trimmed;
  };
  const Text texts[] = {
      {"\"  USER_PROBE é\\n \"", "  USER_PROBE é\n ", "USER_PROBE é"},
      {"\"\"", "", ""}};
  for (const auto &text : texts) {
    const std::string globals =
        "\"bos_token\":\"<probe-bos>\",\"eos_token\":\"<probe-eos>\",";
    const auto wanted = std::string(expected->prefix) +
                        (expected->trim ? text.trimmed : text.raw) +
                        expected->suffix;
    std::string actual;
    ASSERT_TRUE(continuation->Render(
        "{" + globals + '"' + product.text_binding + "\":" + text.json + '}',
        &actual, &error))
        << error;
    EXPECT_EQ(actual, wanted);
    EXPECT_EQ(actual.find("<probe-bos>"), std::string::npos);
    EXPECT_EQ(actual.find("PRIOR_SYSTEM"), std::string::npos);
    const auto environment =
        "{" + globals +
        "\"add_generation_prompt\":true,\"messages\":["
        "{\"role\":\"system\",\"content\":\"PRIOR_SYSTEM\"},"
        "{\"role\":\"user\",\"content\":\"old user\"},"
        "{\"role\":\"assistant\",\"content\":\"old assistant\"},"
        "{\"role\":\"user\",\"content\":" +
        text.json + "}]}";
    ASSERT_TRUE(original->Render(environment, &actual, &error)) << error;
    ASSERT_GE(actual.size(), wanted.size());
    EXPECT_EQ(actual.substr(actual.size() - wanted.size()), wanted);
  }
}

struct SourceCase {
  const char *name;
  const char *path;
  TextToolKind tools;
  int64_t default_startup;
  bool reasoning;
  int64_t effort_mask;
  int64_t effort_default;
  const char *header;
  const char *generation_prefix;
  const char *suffix;
  bool bare_json_tools = false;
};

bool HasLiteral(const std::vector<TextFormatPart> &parts,
                std::string_view fragment) {
  for (const auto &part : parts)
    if ((part.capture == TextCapture::Literal &&
         part.literal.find(fragment) != std::string::npos) ||
        HasLiteral(part.body, fragment) || HasLiteral(part.otherwise, fragment))
      return true;
  return false;
}

class CurrentTemplateAnalysisTest : public testing::TestWithParam<SourceCase> {
};

TEST_P(CurrentTemplateAnalysisTest, UsesActualCompiledProbes) {
  // These are source fixtures, not production selectors or inferred model IDs.
  // Explicit caller-provided globals demonstrate marker-binding independence.
  TextFormatBindings globals;
  globals.strings["bos_token"] = "<probe-bos>";
  globals.strings["eos_token"] = "<probe-eos>";
  const auto &fixture = GetParam();
  SCOPED_TRACE(fixture.path);
  std::ifstream stream(Root() / fixture.path);
  ASSERT_TRUE(stream);
  const std::string source((std::istreambuf_iterator<char>(stream)), {});
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    AnalyzedTemplate result;
    TemplateQualification qualification;
    Diagnostic diagnostic;
    const bool analyzed = CompileAndAnalyze(
        source, fixture.path, globals, Options(bytecode), &result, &diagnostic);
    EXPECT_TRUE(analyzed) << diagnostic.line << ':' << diagnostic.column << ':'
                          << diagnostic.message;
    if (!analyzed)
      continue;
    ASSERT_TRUE(QualifyTemplate(source, fixture.path, globals, result,
                                Options(bytecode), &qualification, &diagnostic))
        << diagnostic.message;
    EXPECT_GE(qualification.validated_probe_count, 2u);
    EXPECT_EQ(result.reasoning.enabled, fixture.reasoning);
    EXPECT_EQ(result.reasoning.effort_mask, fixture.effort_mask);
    EXPECT_EQ(result.reasoning.default_effort, fixture.effort_default);
    bool saw_header = false, saw_reasoning = false, saw_tools = false;
    for (const auto &format : result.format.alternatives) {
      SCOPED_TRACE(format.assistant_header);
      saw_header |= format.assistant_header == fixture.header;
      EXPECT_EQ(format.frame_suffix, fixture.suffix);
      EXPECT_FALSE(format.assistant_pattern.empty());
      EXPECT_FALSE(format.evidence.empty());
      if (!format.reasoning_open.empty() || !format.reasoning_close.empty()) {
        EXPECT_TRUE(fixture.reasoning);
        EXPECT_EQ(format.reasoning_open, "<think>");
        EXPECT_EQ(format.reasoning_close, "</think>");
        saw_reasoning = true;
      }
      if (format.tools.kind == TextToolKind::None)
        continue;
      saw_tools = true;
      EXPECT_EQ(format.tools.kind, fixture.tools);
      EXPECT_EQ(format.tools.open, "<tool_call>");
      EXPECT_EQ(format.tools.close, "</tool_call>");
      if (format.tools.kind == TextToolKind::JsonEnvelope) {
        EXPECT_EQ(format.tools.name_field, "name");
        EXPECT_EQ(format.tools.arguments_field, "arguments");
        EXPECT_TRUE(format.tools.json_argument_wire.has_value());
      } else if (format.tools.kind == TextToolKind::TaggedNamed) {
        EXPECT_TRUE(HasLiteral(format.tools.call_body, "<function="));
        EXPECT_TRUE(HasLiteral(format.tools.call_body, "<parameter="));
      } else {
        EXPECT_TRUE(HasLiteral(format.tools.call_body, "<arg_key>"));
        EXPECT_TRUE(HasLiteral(format.tools.call_body, "<arg_value>"));
      }
    }
    EXPECT_TRUE(saw_header) << fixture.header;
    EXPECT_EQ(saw_reasoning, fixture.reasoning);
    EXPECT_EQ(saw_tools, fixture.tools != TextToolKind::None);
    EXPECT_EQ(std::any_of(result.format.alternatives.begin(),
                          result.format.alternatives.end(),
                          [](const TextFormatAlternative &format) {
                            return format.ambiguous_bare_json_tools;
                          }),
              fixture.bare_json_tools);
    CheckContinuation(result, fixture.name, bytecode);
    std::unique_ptr<PreparedTemplate> selector;
    std::string error;
    ASSERT_TRUE(PreparedTemplate::Prepare(result.selector, Root().string(),
                                          Options(bytecode).scratch_parent,
                                          bytecode, &selector, &error))
        << error;
    for (bool generation : {false, true}) {
      SCOPED_TRACE(generation ? "generation" : "no generation");
      int64_t id = -1, mode = -1;
      const auto environment =
          "{\"bos_token\":\"<probe-bos>\",\"eos_token\":\"<probe-eos>\","
          "\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],"
          "\"tools\":[],\"add_generation_prompt\":" +
          std::string(generation ? "true}" : "false}");
      ASSERT_TRUE(selector->SelectFormat(environment, &id, &mode, &error))
          << error;
      ASSERT_GE(id, 0);
      ASSERT_LT(static_cast<size_t>(id), result.format.alternatives.size());
      const auto &format = result.format.alternatives[id];
      EXPECT_EQ(mode, generation ? fixture.default_startup : 0);
      EXPECT_EQ(format.generation_prefix,
                generation ? fixture.generation_prefix : "");
      EXPECT_EQ(format.tools.kind, fixture.tools);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CurrentSources, CurrentTemplateAnalysisTest,
    testing::Values(
        SourceCase{"Llama", "lib/text/llama/templates/5816fce1.jinja",
                   TextToolKind::None, 0, false, 0, 0,
                   "<|start_header_id|>assistant<|end_header_id|>\n\n",
                   "<|start_header_id|>assistant<|end_header_id|>\n\n",
                   "<|eot_id|>", true},
        SourceCase{
            "Qwen08", "lib/text/qwen35/templates/7f0e5290.jinja",
            TextToolKind::TaggedNamed, 0, true, 0, 0, "<|im_start|>assistant\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_end|>\n"},
        SourceCase{"Qwen27", "lib/text/qwen35/templates/c3cf9e34.jinja",
                   TextToolKind::TaggedNamed, 1, true, 44, 5,
                   "<|im_start|>assistant\n",
                   "<|im_start|>assistant\n<think>\n", "<|im_end|>\n"},
        SourceCase{"Plain", "lib/text/plain_chatml/templates/58c1a1f0.jinja",
                   TextToolKind::None, 0, false, 0, 0,
                   "<|im_start|>assistant\n", "<|im_start|>assistant\n",
                   "<|im_end|>\n"},
        SourceCase{"Qwen25", "lib/text/qwen25/templates/cd8e9439.jinja",
                   TextToolKind::JsonEnvelope, 0, false, 0, 0,
                   "<|im_start|>assistant", "<|im_start|>assistant\n",
                   "<|im_end|>\n"},
        SourceCase{"Qwen3", "lib/text/qwen3/templates/5da44855.jinja",
                   TextToolKind::JsonEnvelope, 2, true, 0, 0,
                   "<|im_start|>assistant\n", "<|im_start|>assistant\n",
                   "<|im_end|>\n"},
        SourceCase{"Gemma", "lib/text/gemma/templates/7de1c58e.jinja",
                   TextToolKind::None, 0, false, 0, 0, "<start_of_turn>model\n",
                   "<start_of_turn>model\n", "<end_of_turn>\n"},
        SourceCase{
            "Compact", "lib/text/compact_chatml/templates/f45005c5.jinja",
            TextToolKind::TaggedNamed, 1, true, 0, 0, "<|im_start|>assistant\n",
            "<|im_start|>assistant\n<think>", "<|im_end|>\n"},
        SourceCase{"Spark", "lib/text/spark/templates/93d6aa90.jinja",
                   TextToolKind::TaggedKeyValue, 1, true, 20, 4,
                   "<｜start▁of▁sentence｜><|Bot|>",
                   "<｜start▁of▁sentence｜><|Bot|><think>",
                   "<｜end▁of▁sentence｜>"}),
    [](const testing::TestParamInfo<SourceCase> &info) {
      return info.param.name;
    });

} // namespace
} // namespace hrx::loomify_gguf::jinja
