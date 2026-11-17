// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/compiler.h"
#include "lib/loomify/text/jinja/source_template.h"
#include "lib/loomify/text/jinja/parser.h"
#include "lib/loomify/template_probe_runner.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <set>
#include <type_traits>

namespace hrx::loomify_gguf::jinja::testing {
namespace {

std::string Required(const char *name) {
  const char *value = std::getenv(name);
  EXPECT_TRUE(value && *value) << name;
  return value ? value : "";
}

std::string HelperRoot() {
  return (std::filesystem::path(Required("TEST_SRCDIR")) /
          Required("TEST_WORKSPACE"))
      .string();
}

bool CompileSelector(std::string_view source, GeneratedInputRenderer *out,
                     Diagnostic *diagnostic) {
  Template ast;
  if (!ParseTemplate(source, &ast, diagnostic))
    return false;
  return CompileTemplateSelector(source, "selector-case.jinja", ast, out,
                                 diagnostic);
}

std::unique_ptr<PreparedTemplate>
PrepareSelector(const GeneratedInputRenderer &generated, bool bytecode) {
  std::unique_ptr<PreparedTemplate> prepared;
  std::string error;
  EXPECT_TRUE(PreparedTemplate::Prepare(generated, HelperRoot(),
                                        Required("TEST_TMPDIR"), bytecode,
                                        &prepared, &error))
      << error;
  return prepared;
}

void ExpectSelection(PreparedTemplate &prepared, std::string_view environment,
                     int64_t expected_id, int64_t expected_mode) {
  int64_t id = -1, mode = -1;
  std::string error;
  ASSERT_TRUE(prepared.SelectFormat(environment, &id, &mode, &error)) << error;
  EXPECT_EQ(id, expected_id);
  EXPECT_EQ(mode, expected_mode);
}

TEST(JinjaSourceTemplateTest, OwnsStableSyntaxAndExactAnalysisIndexes) {
  using internal::SourceTemplate;
  static_assert(!std::is_copy_constructible_v<SourceTemplate>);
  static_assert(!std::is_move_constructible_v<SourceTemplate>);
  const std::string expected =
      "literal"
      "{% macro chosen(x=default_only()) %}{{ body_only() }}{% endmacro %}"
      "{% if flag %}{% macro nested() %}{{ chosen() }}{% endmacro %}{% endif %}"
      "{{ chosen(nested()) }}{{ unknown() }}"
      "{% macro chosen() %}{{ replacement }}{% endmacro %}"
      "{% set ns.value = referenced %}";
  std::unique_ptr<SourceTemplate> original;
  Diagnostic diagnostic;
  {
    std::string source = expected, name = "owned-index.jinja";
    ASSERT_TRUE(SourceTemplate::Parse(source, name, &original, &diagnostic))
        << diagnostic.message;
    source.assign(source.size(), 'x');
    name.assign(name.size(), 'x');
  }
  const auto *ast = &original->ast();
  auto unit = std::move(original);
  EXPECT_EQ(original, nullptr);
  EXPECT_EQ(&unit->ast(), ast);
  EXPECT_EQ(unit->source(), expected);
  EXPECT_EQ(unit->source_name(), "owned-index.jinja");
  const auto &body = unit->ast().body;
  ASSERT_EQ(body.size(), 7u);
  ASSERT_EQ(unit->analysis_macros().size(), 2u);
  EXPECT_EQ(unit->analysis_macros().at("chosen"), &body[5]);
  EXPECT_EQ(unit->analysis_macros().at("nested"), &body[2].body[0]);
  ASSERT_EQ(unit->analysis_macro_calls().size(), 2u);
  EXPECT_EQ(unit->analysis_macro_calls()[0], &body[3].expression);
  EXPECT_EQ(unit->analysis_macro_calls()[1], &body[3].expression.args[1]);
  EXPECT_EQ(unit->source_names(),
            (std::set<std::string>{"", "literal", "chosen", "x", "default_only",
                                   "body_only", "nested", "flag", "unknown",
                                   "replacement", "ns.value", "referenced"}));
  // Parsing/indexing does not bypass the renderer's separate macro validation.
  GeneratedInputRenderer renderer{"unchanged", "root", {}};
  EXPECT_FALSE(internal::CompileChatTemplate(*unit, &renderer, &diagnostic));
  EXPECT_EQ(renderer.source, "unchanged");
  EXPECT_EQ(diagnostic.message, "macros require unique module-level names");
}

TEST(JinjaSourceTemplateTest, ParseFailureKeepsOwnerAndLocatesFreshDiagnostic) {
  using internal::SourceTemplate;
  std::unique_ptr<SourceTemplate> unit;
  Diagnostic diagnostic;
  ASSERT_TRUE(SourceTemplate::Parse("kept", "kept.jinja", &unit, &diagnostic));
  const auto *original = unit.get();
  diagnostic = {"stale message", "stale source", 999, 999, 999};
  EXPECT_FALSE(SourceTemplate::Parse(u8"é\r\n{{ missing(", "malformed.jinja",
                                     &unit, &diagnostic));
  EXPECT_EQ(unit.get(), original);
  EXPECT_EQ(unit->source(), "kept");
  EXPECT_EQ(diagnostic.source_name, "malformed.jinja");
  EXPECT_EQ(diagnostic.offset, 4u);
  EXPECT_EQ(diagnostic.line, 2u);
  EXPECT_EQ(diagnostic.column, 1u);
  EXPECT_NE(diagnostic.message, "stale message");
  EXPECT_FALSE(diagnostic.message.empty());
  EXPECT_FALSE(SourceTemplate::Parse("{{", "null.jinja", nullptr, nullptr));
  ASSERT_TRUE(
      SourceTemplate::Parse("replacement", "next.jinja", &unit, &diagnostic));
  EXPECT_EQ(unit->source(), "replacement");
  EXPECT_TRUE(diagnostic.message.empty());
}

TEST(TemplateSelectorTest, ExecutesMissingFalseNullAndChangedRequestBranch) {
  constexpr std::string_view slice =
      "{% if alternate_fields | default(false) %}{{ [1, 2] }}"
      "{% else %}{{ [0, 0] }}{% endif %}";
  GeneratedInputRenderer generated;
  Diagnostic diagnostic;
  ASSERT_TRUE(CompileSelector(slice, &generated, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto prepared = PrepareSelector(generated, bytecode);
    ASSERT_NE(prepared, nullptr);
    ExpectSelection(*prepared, "{}", 0, 0);
    ExpectSelection(*prepared, "{\"alternate_fields\":false}", 0, 0);
    ExpectSelection(*prepared, "{\"alternate_fields\":null}", 0, 0);
    ExpectSelection(*prepared, "{\"alternate_fields\":true}", 1, 2);
    ExpectSelection(*prepared, "{\"alternate_fields\":false}", 0, 0);
  }
}

TEST(TemplateSelectorTest,
     RetainsNamespaceHistoryLoopAndLocalDefaultSemantics) {
  constexpr std::string_view slice =
      "{% set scan = namespace(found=false) %}"
      "{% for message in messages %}"
      "{% if message.role == 'tool' %}{% set scan.found = true %}{% endif %}"
      "{% endfor %}"
      "{% set requested = enabled | default(scan.found) %}"
      "{% if requested %}{{ [3, 1] }}{% else %}{{ [0, 0] }}{% endif %}";
  GeneratedInputRenderer generated;
  Diagnostic diagnostic;
  ASSERT_TRUE(CompileSelector(slice, &generated, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto prepared = PrepareSelector(generated, bytecode);
    ASSERT_NE(prepared, nullptr);
    ExpectSelection(*prepared, "{\"messages\":[]}", 0, 0);
    ExpectSelection(*prepared,
                    "{\"messages\":[{\"role\":\"user\"},{\"role\":\"tool\"}]}",
                    3, 1);
    ExpectSelection(*prepared,
                    "{\"messages\":[{\"role\":\"tool\"}],\"enabled\":false}", 0,
                    0);
    ExpectSelection(*prepared, "{\"messages\":[],\"enabled\":true}", 3, 1);
  }
}

TEST(TemplateSelectorTest,
     LinksRendererAndSelectorMacrosWithoutSymbolCollision) {
  constexpr std::string_view render_source =
      "{% macro choose(value) %}{{ value }}{% endmacro %}"
      "{{ choose('prefix-A') }}";
  constexpr std::string_view selector_source =
      "{% macro choose(value) %}{{ value }}{% endmacro %}"
      "{% if choose(option | default('plain')) == 'alternate' %}"
      "{{ [1, 2] }}{% else %}{{ [0, 0] }}{% endif %}";
  constexpr std::string_view adaptation_source =
      "{% macro choose(value) %}{{ value }}{% endmacro %}"
      "{% if choose(option | default('plain')) == 'alternate' %}"
      "{{ [5, 0] }}{% else %}{{ [4, 0] }}{% endif %}";
  constexpr std::string_view recipe_source =
      "{% macro choose(value) %}{{ value }}{% endmacro %}"
      "{{ choose('recipe-B') }}";
  GeneratedInputRenderer renderer, selector, adaptation, recipe;
  Diagnostic diagnostic;
  ASSERT_TRUE(CompileChatTemplate(render_source, "renderer.jinja", &renderer,
                                  &diagnostic))
      << diagnostic.message;
  ASSERT_TRUE(CompileSelector(selector_source, &selector, &diagnostic))
      << diagnostic.message;
  Template adaptation_ast;
  ASSERT_TRUE(ParseTemplate(adaptation_source, &adaptation_ast, &diagnostic));
  ASSERT_TRUE(CompileTemplateSelector(adaptation_source, "adaptation.jinja",
                                      adaptation_ast, &adaptation, &diagnostic,
                                      TemplateSelectorKind::Adaptation))
      << diagnostic.message;
  Template recipe_ast;
  ASSERT_TRUE(ParseTemplate(recipe_source, &recipe_ast, &diagnostic));
  ASSERT_TRUE(CompileTemplateRenderer(recipe_source, "recipe.jinja", recipe_ast,
                                      &recipe, &diagnostic))
      << diagnostic.message;
  GeneratedInputRenderer combined;
  ASSERT_TRUE(
      CombineGeneratedTemplates({renderer, selector, adaptation, recipe},
                                selector.entrypoint, &combined, &diagnostic))
      << diagnostic.message;
  combined.entrypoint = "test.renderer_and_selector";
  combined.source += R"(
func.def public @test.renderer_and_selector(%environment: buffer) -> (i64, i64) {
  %rendered = func.call @jinja.evaluate(%environment) : (buffer) -> (buffer)
  %size = buffer.length %rendered
  %expected = index.constant 8 : offset
  %valid = index.cmp eq, %size, %expected : offset
  %flag = scalar.extui %valid : i1 to i32
  %diagnostic = buffer.constant utf8("renderer and selector symbol collision") : buffer
  func.call @require(%flag, %diagnostic) : (i32, buffer)
  %recipe = func.call @jinja.render_format_recipe(%environment) : (buffer) -> (buffer)
  %expected_recipe = buffer.constant utf8("recipe-B") : buffer
  %string = scalar.constant 0 : i64
  %recipe_matches = func.call @jinja.equal(%recipe, %string, %expected_recipe, %string) : (buffer, i64, buffer, i64) -> (i1)
  %recipe_flag = scalar.extui %recipe_matches : i1 to i32
  func.call @require(%recipe_flag, %diagnostic) : (i32, buffer)
  %id, %mode = func.call @jinja.select_format(%environment) : (buffer) -> (i64, i64)
  %adaptation, %unused = func.call @jinja.select_adaptation(%environment) : (buffer) -> (i64, i64)
  %total = scalar.addi %id, %adaptation : i64
  func.return %total, %mode : i64, i64
}
)";
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto prepared = PrepareSelector(combined, bytecode);
    ASSERT_NE(prepared, nullptr);
    ExpectSelection(*prepared, "{}", 4, 0);
    ExpectSelection(*prepared, "{\"option\":\"alternate\"}", 6, 2);
  }
}

TEST(TemplateSelectorTest, RecipeRendererPreservesExactSourceFilterKeywords) {
  constexpr std::string_view source =
      "{% macro render(value) %}{{ value | tojson(indent=4) }}{% endmacro %}"
      "{{ render(arguments) }}";
  Template ast;
  Diagnostic diagnostic;
  ASSERT_TRUE(ParseTemplate(source, &ast, &diagnostic));
  GeneratedInputRenderer recipe;
  ASSERT_TRUE(CompileTemplateRenderer(source, "recipe-keywords.jinja", ast,
                                      &recipe, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    auto prepared = PrepareSelector(recipe, bytecode);
    ASSERT_NE(prepared, nullptr);
    std::string result, error;
    ASSERT_TRUE(
        prepared->Render("{\"arguments\":{\"count\":3}}", &result, &error))
        << error;
    EXPECT_EQ(result, "{\n    \"count\": 3\n}");
    ASSERT_TRUE(prepared->Render("{\"arguments\":{\"name\":\"next\"}}", &result,
                                 &error))
        << error;
    EXPECT_EQ(result, "{\n    \"name\": \"next\"\n}");
  }
}

TEST(TemplateSelectorTest, FailurePreservesOutputsAndNextInvocationWorks) {
  constexpr std::string_view slice =
      "{% if invalid %}{{ raise_exception('unsupported active format') }}"
      "{% else %}{{ [4, 1] }}{% endif %}";
  GeneratedInputRenderer generated;
  Diagnostic diagnostic;
  ASSERT_TRUE(CompileSelector(slice, &generated, &diagnostic))
      << diagnostic.message;
  for (bool bytecode : {false, true}) {
    auto prepared = PrepareSelector(generated, bytecode);
    ASSERT_NE(prepared, nullptr);
    int64_t id = 9, mode = 8;
    std::string error;
    EXPECT_FALSE(
        prepared->SelectFormat("{\"invalid\":true}", &id, &mode, &error));
    EXPECT_EQ(id, 9);
    EXPECT_EQ(mode, 8);
    EXPECT_NE(error.find("unsupported active format"), std::string::npos);
    ExpectSelection(*prepared, "{\"invalid\":false}", 4, 1);
  }
}

TEST(TemplateSelectorTest, RejectsAmbiguousResultShapeBeforeProducingSource) {
  const std::vector<std::string_view> invalid = {
      "{% if option %}{{ [1, 0] }}{% endif %}",
      "{{ [0, 0] }}{{ [1, 0] }}",
      "{{ [0, 3] }}",
      "{{ [option, 0] }}",
      "literal{{ [0, 0] }}",
      "{% for item in items %}{{ [0, 0] }}{% endfor %}",
  };
  for (const auto source : invalid) {
    SCOPED_TRACE(source);
    GeneratedInputRenderer generated{"preserved", "preserved", {"preserved"}};
    Diagnostic diagnostic;
    EXPECT_FALSE(CompileSelector(source, &generated, &diagnostic));
    EXPECT_EQ(generated.source, "preserved");
    EXPECT_EQ(generated.entrypoint, "preserved");
    EXPECT_EQ(generated.helper_requirements,
              std::vector<std::string>{"preserved"});
    EXPECT_FALSE(diagnostic.message.empty());
    EXPECT_EQ(diagnostic.source_name, "selector-case.jinja");
    EXPECT_GE(diagnostic.line, 1u);
    EXPECT_GE(diagnostic.column, 1u);
  }
}

TEST(TemplateSelectorTest, AdaptationCannotReturnParserStartupState) {
  constexpr std::string_view source = "{{ [1, 2] }}";
  Template ast;
  Diagnostic diagnostic;
  ASSERT_TRUE(ParseTemplate(source, &ast, &diagnostic));
  GeneratedInputRenderer generated{"kept", "kept", {}};
  EXPECT_FALSE(CompileTemplateSelector(source, "adaptation.jinja", ast,
                                       &generated, &diagnostic,
                                       TemplateSelectorKind::Adaptation));
  EXPECT_EQ(generated.source, "kept");
  EXPECT_NE(diagnostic.message.find("adaptation"), std::string::npos);
}

TEST(TemplateSelectorTest,
     CompositionRejectsConflictingDeclarationsTransactionally) {
  const GeneratedInputRenderer first{
      "func.decl @shared(%value: buffer) -> (buffer)\n", "first", {"one"}};
  const GeneratedInputRenderer second{
      "func.decl @shared(%value: buffer) -> (i64)\n", "second", {"two"}};
  GeneratedInputRenderer result{"preserved", "preserved", {"preserved"}};
  Diagnostic diagnostic;
  EXPECT_FALSE(CombineGeneratedTemplates({first, second}, "first", &result,
                                         &diagnostic));
  EXPECT_EQ(result.source, "preserved");
  EXPECT_EQ(result.entrypoint, "preserved");
  EXPECT_EQ(result.helper_requirements, std::vector<std::string>{"preserved"});
  EXPECT_NE(diagnostic.message.find("@shared"), std::string::npos);
}

TEST(TemplateSelectorTest, DiscoversActualExternalReadsUsingCompilerScopes) {
  const auto names = [](std::string_view source) {
    GeneratedInputRenderer generated;
    Diagnostic diagnostic;
    EXPECT_TRUE(
        CompileChatTemplate(source, "bindings.jinja", &generated, &diagnostic))
        << diagnostic.message;
    return generated.external_bindings;
  };
  const auto has = [](const auto &values, const char *name) {
    return std::find(values.begin(), values.end(), name) != values.end();
  };
  EXPECT_TRUE(has(names("{{ bos_token | default('') }}"), "bos_token"));
  EXPECT_FALSE(
      has(names("{% set bos_token = 'local' %}{{ bos_token }}"), "bos_token"));
  EXPECT_FALSE(has(names("{% macro show(bos_token) %}{{ bos_token }}"
                         "{% endmacro %}{{ show('local') }}"),
                   "bos_token"));
  EXPECT_TRUE(has(names("{% macro show() %}{{ eos_token }}{% endmacro %}"
                        "{{ show() }}"),
                  "eos_token"));
  EXPECT_FALSE(has(names("{% macro show() %}{{ eos_token }}{% endmacro %}"
                         "{% set eos_token = 'local' %}{{ show() }}"),
                   "eos_token"));
  EXPECT_FALSE(has(names("{% macro unused() %}{{ eos_token }}{% endmacro %}"
                         "{{ 'ordinary' }}"),
                   "eos_token"));
  EXPECT_FALSE(has(names("{% if flag %}{% set bos_token = 'a' %}"
                         "{% else %}{% set bos_token = 'b' %}{% endif %}"
                         "{{ bos_token }}"),
                   "bos_token"));
  EXPECT_TRUE(has(names("{% if flag %}{% set bos_token = 'a' %}{% endif %}"
                        "{{ bos_token }}"),
                  "bos_token"));

  GeneratedInputRenderer generated;
  Diagnostic diagnostic;
  ASSERT_TRUE(CompileChatTemplate("{{ strftime_now('%d %b %Y') }}",
                                  "capability.jinja", &generated, &diagnostic))
      << diagnostic.message;
  EXPECT_EQ(generated.protected_bindings,
            std::vector<std::string>{"__loomify_strftime_now"});
}

TEST(TemplateSelectorTest,
     NamespaceConstructorsDoNotDiscoverDeadExternalFallbacks) {
  const auto reads = [](std::string_view source, const char *name) {
    GeneratedInputRenderer generated;
    Diagnostic diagnostic;
    EXPECT_TRUE(CompileChatTemplate(source, "namespace-bindings.jinja",
                                    &generated, &diagnostic))
        << diagnostic.message;
    return std::find(generated.external_bindings.begin(),
                     generated.external_bindings.end(),
                     name) != generated.external_bindings.end();
  };
  EXPECT_FALSE(reads(
      "{% set bos_token = namespace(value='local') %}"
      "{% for item in items %}{% set bos_token.value = item %}{% endfor %}"
      "{{ bos_token.value }}",
      "bos_token"));
  EXPECT_FALSE(reads(
      "{% macro show(items) %}{% set eos_token = namespace(value='local') %}"
      "{% for item in items %}{% set eos_token.value = item %}{% endfor %}"
      "{{ eos_token.value }}{% endmacro %}{{ show([]) }}",
      "eos_token"));
  EXPECT_FALSE(
      reads("{% set pad_token = namespace(value='local') %}"
            "{% macro show() %}{{ pad_token.value }}{% endmacro %}{{ show() }}",
            "pad_token"));
  // The untaken constructor path can genuinely read a supplied JSON binding.
  // Metadata must retain it rather than hiding every namespace fallback.
  EXPECT_TRUE(reads(
      "{% if flag %}{% set bos_token = namespace(value='local') %}{% endif %}"
      "{{ bos_token.value }}",
      "bos_token"));
}

} // namespace
} // namespace hrx::loomify_gguf::jinja::testing
