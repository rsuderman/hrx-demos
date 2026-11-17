// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/template_probe_runner.h"
#include "lib/loomify/text/jinja/format_analysis.h"
#include "lib/loomify/text/jinja/request_emitter.h"
#include "iree/base/internal/json.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::jinja::testing {
namespace {

std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : value) {
    result += digits[c >> 4];
    result += digits[c & 15];
  }
  return result;
}

std::optional<std::string> Field(std::string_view object, const char *key) {
  iree_string_view_t value{};
  auto status = iree_json_try_lookup_object_value(
      iree_make_string_view(object.data(), object.size()),
      iree_make_cstring_view(key), &value);
  const bool ok = iree_status_is_ok(status);
  EXPECT_TRUE(ok) << key;
  iree_status_ignore(status);
  if (!ok || !value.data)
    return std::nullopt;
  return std::string(value.data, value.size);
}

std::vector<std::string> Elements(std::string_view array) {
  std::vector<std::string> result;
  const auto append = [](void *user, iree_host_size_t, iree_json_value_type_t,
                         iree_string_view_t value) {
    static_cast<std::vector<std::string> *>(user)->emplace_back(value.data,
                                                                value.size);
    return iree_ok_status();
  };
  auto status = iree_json_enumerate_array_typed(
      iree_make_string_view(array.data(), array.size()), append, &result);
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
  return result;
}

std::optional<std::string> Arguments(std::string_view call) {
  const auto function = Field(call, "function");
  EXPECT_TRUE(function);
  return function ? Field(*function, "arguments") : std::nullopt;
}

std::string Required(const char *name) {
  const char *value = std::getenv(name);
  EXPECT_TRUE(value && *value) << name;
  return value ? value : "";
}

GeneratedInputRenderer Program(int operation, bool project, bool has_tools,
                               std::string_view globals,
                               std::string_view protected_names) {
  GeneratedInputRenderer result;
  result.entrypoint = "test.request_environment";
  result.helper_requirements = {"lib/text/common/strings.loom",
                                "lib/text/common/json.loom",
                                "lib/text/generated_input/environment.loom",
                                "lib/text/generated_input/request.loom"};
  result.source = "global.rodata.def @test.globals = bytes(\"" + Hex(globals) +
                  "\")\n" + "global.rodata.def @test.names = bytes(\"" +
                  Hex(protected_names) + "\")\n" +
                  R"(
func.decl @text.request_base_environment(%request: buffer, %globals: buffer, %names: buffer, %operation: i32) -> (buffer)
func.decl @text.request_apply_adaptation(%base: buffer, %messages: buffer, %tools: buffer, %present: i1) -> (buffer)
func.decl @text.join8(%a: buffer, %b: buffer, %c: buffer, %d: buffer, %e: buffer, %f: buffer, %g: buffer, %h: buffer) -> (buffer)
func.def public @test.request_environment(%request: buffer) -> (buffer) {
  %globals = global.load @test.globals : buffer
  %names = global.load @test.names : buffer
  %operation = scalar.constant )" +
                  std::to_string(operation) + R"( : i32
  %base = func.call @text.request_base_environment(%request, %globals, %names, %operation) : (buffer, buffer, buffer, i32) -> (buffer)
)";
  if (project) {
    result.source +=
        "  %messages = buffer.constant utf8(\"[{\\\"role\\\":\\\"assistant\\\",\\\"content\\\":null}]\") : buffer\n"
        "  %tools = buffer.constant utf8(\"null\") : buffer\n"
        "  %present = scalar.constant " +
        std::string(has_tools ? "true" : "false") + R"( : i1
  %projected = func.call @text.request_apply_adaptation(%base, %messages, %tools, %present) : (buffer, buffer, buffer, i1) -> (buffer)
  %prefix = buffer.constant utf8("{\"base\":") : buffer
  %middle = buffer.constant utf8(",\"projected\":") : buffer
  %close = buffer.constant utf8("}") : buffer
  %empty = buffer.constant utf8("") : buffer
  %result = func.call @text.join8(%prefix, %base, %middle, %projected, %close, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)
  func.return %result : buffer
}
)";
  } else {
    result.source += "  func.return %base : buffer\n}\n";
  }
  return result;
}

class InputEnvironmentTest : public ::testing::TestWithParam<bool> {
protected:
  void
  Prepare(int operation = 0, bool project = false, bool has_tools = false,
          std::string_view globals =
              R"({"bos_token":"","compiler_global":"immutable"})",
          std::string_view protected_names = R"(["compiler_capability"])") {
    const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                       Required("TEST_WORKSPACE"))
                          .string();
    std::string error;
    ASSERT_TRUE(PreparedTemplate::Prepare(
        Program(operation, project, has_tools, globals, protected_names), root,
        Required("TEST_TMPDIR"), GetParam(), &prepared_, &error))
        << error;
  }

  std::string Render(std::string_view request) {
    std::string result, error;
    EXPECT_NE(prepared_, nullptr);
    if (prepared_)
      EXPECT_TRUE(prepared_->Render(request, &result, &error)) << error;
    return result;
  }

  void PrepareAdapters(const TextFormatAnalysis &analysis,
                       const GeneratedInputRenderer &selector,
                       const TextFormatBindings &globals,
                       const std::vector<std::string> &protected_names,
                       bool raw_base = false, int operation = 0) {
    GeneratedInputRenderer generated;
    Diagnostic diagnostic;
    ASSERT_TRUE(EmitRequestAdapters("adapter-fixture.jinja", analysis, selector,
                                    globals, protected_names, &generated,
                                    &diagnostic))
        << diagnostic.message;
    generated.entrypoint = "test.request_adapters";
    generated.source += R"(
func.decl @text.request_base_environment(%request: buffer, %globals: buffer, %names: buffer, %operation: i32) -> (buffer)
func.decl @text.request_apply_adaptation(%base: buffer, %messages: buffer, %tools: buffer, %present: i1) -> (buffer)
func.decl @text.request_environment(%request: buffer, %operation: i32) -> (buffer)
func.decl @text.adapt_messages(%base: buffer) -> (buffer)
func.decl @text.adapt_tools(%base: buffer) -> (buffer, i1)
)";
    generated.source += "func.def public @test.request_adapters(" +
                        std::string(raw_base ? "%base" : "%request") +
                        ": buffer) -> (buffer) {\n";
    if (!raw_base) {
      generated.source += "  %operation = scalar.constant " +
                          std::to_string(operation) + R"( : i32
  %result = func.call @text.request_environment(%request, %operation) : (buffer, i32) -> (buffer)
  func.return %result : buffer
}
)";
    } else {
      generated.source += R"(
  %messages = func.call @text.adapt_messages(%base) : (buffer) -> (buffer)
  %tools, %present = func.call @text.adapt_tools(%base) : (buffer) -> (buffer, i1)
  %result = func.call @text.request_apply_adaptation(%base, %messages, %tools, %present) : (buffer, buffer, buffer, i1) -> (buffer)
  func.return %result : buffer
}
)";
    }
    const auto root = (std::filesystem::path(Required("TEST_SRCDIR")) /
                       Required("TEST_WORKSPACE"))
                          .string();
    std::string error;
    ASSERT_TRUE(PreparedTemplate::Prepare(generated, root,
                                          Required("TEST_TMPDIR"), GetParam(),
                                          &prepared_, &error))
        << error;
  }

  std::unique_ptr<PreparedTemplate> prepared_;
};

TEST_P(InputEnvironmentTest,
       CopiesOrdinaryKwargsAndPreservesExplicitPrecedence) {
  Prepare();
  const auto result = Render(
      R"({"messages":[{"role":"custom","content":"input","extra":9}],"new_style":"top","enable_thinking":false,"precision":1.2500e+02,"model_options":{"new_style":"compact","enable_thinking":true,"extra_kwarg":null}})");
  EXPECT_EQ(Field(result, "new_style"), "compact");
  EXPECT_EQ(Field(result, "enable_thinking"), "false");
  EXPECT_EQ(Field(result, "extra_kwarg"), "null");
  EXPECT_EQ(Field(result, "precision"), "1.2500e+02");
  EXPECT_EQ(
      Field(result, "model_options"),
      R"({"new_style":"compact","enable_thinking":true,"extra_kwarg":null})");
  EXPECT_EQ(Field(result, "messages"),
            R"([{"role":"custom","content":"input","extra":9}])");
  EXPECT_EQ(Field(result, "bos_token"), "");
  EXPECT_EQ(Field(result, "compiler_global"), "immutable");
  EXPECT_EQ(Field(result, "add_generation_prompt"), "true");
}

TEST_P(InputEnvironmentTest, DistinguishesMissingNullFalseAndEmptyTools) {
  Prepare();
  auto result = Render(R"({"messages":[]})");
  EXPECT_FALSE(Field(result, "enable_thinking"));
  EXPECT_FALSE(Field(result, "tools"));
  result = Render(R"({"messages":[],"tools":null,"enable_thinking":null})");
  EXPECT_EQ(Field(result, "tools"), "null");
  EXPECT_EQ(Field(result, "enable_thinking"), "null");
  result = Render(R"({"messages":[],"tools":[],"enable_thinking":false})");
  EXPECT_EQ(Field(result, "tools"), "[]");
  EXPECT_EQ(Field(result, "enable_thinking"), "false");
  result = Render(R"({"messages":[],"enable_thinking":"source-defined"})");
  EXPECT_EQ(Field(result, "enable_thinking"), "source-defined");
}

TEST_P(InputEnvironmentTest,
       CompletionOwnsGenerationFlagWithoutInventingDefaults) {
  Prepare(1);
  const auto result = Render(
      R"({"messages":[{"role":"assistant","content":null}],"add_generation_prompt":true,"model_options":{"add_generation_prompt":true}})");
  EXPECT_EQ(Field(result, "add_generation_prompt"), "false");
  EXPECT_EQ(Field(result, "model_options"),
            R"({"add_generation_prompt":true})");
  EXPECT_FALSE(Field(result, "enable_thinking"));
  EXPECT_FALSE(Field(result, "reasoning_effort"));
  Prepare();
  EXPECT_EQ(
      Field(
          Render(
              R"({"messages":[],"model_options":{"add_generation_prompt":false}})"),
          "add_generation_prompt"),
      "false");
}

TEST_P(InputEnvironmentTest, ProtectsActualBindingsWithoutAnOptionWhitelist) {
  Prepare();
  const auto ordinary = Render(
      R"({"messages":[],"model":"ordinary","format":"custom","namespace":"ordinary","eos_token":"unused","model_options":{"other_control":{"nested":true}}})");
  EXPECT_EQ(Field(ordinary, "model"), "ordinary");
  EXPECT_EQ(Field(ordinary, "namespace"), "ordinary");
  EXPECT_EQ(Field(ordinary, "format"), "custom");
  EXPECT_EQ(Field(ordinary, "eos_token"), "unused");
  EXPECT_EQ(Field(ordinary, "other_control"), R"({"nested":true})");
  const std::vector<std::pair<std::string_view, std::string_view>> rejected = {
      {R"({"messages":[],"bos_token":"override"})", "bos_token"},
      {R"({"messages":[],"compiler_capability":{}})", "compiler_capability"},
      {R"({"messages":[],"model_options":{"messages":[]}})", "messages"},
      {R"({"messages":[],"model_options":{"tools":[]}})", "tools"},
      {R"({"messages":[],"model_options":{"model_options":{}}})",
       "model_options"},
      {R"({"messages":[],"model_options":{"compiler_global":"override"}})",
       "compiler_global"},
      {R"({"messages":[],"model_options":{"compiler_capability":false}})",
       "compiler_capability"},
  };
  for (const auto &[request, key] : rejected) {
    SCOPED_TRACE(request);
    std::string output = "preserved", error;
    EXPECT_FALSE(prepared_->Render(request, &output, &error));
    EXPECT_EQ(output, "preserved");
    EXPECT_NE(
        error.find("request binding cannot override: " + std::string(key)),
        std::string::npos)
        << error;
  }
  EXPECT_EQ(
      Field(Render(R"({"messages":[],"ordinary":"after errors"})"), "ordinary"),
      "after errors");
}

TEST_P(InputEnvironmentTest, AppliesActualAdaptationsAfterTheImmutableBase) {
  const std::string request =
      R"({"messages":[{"role":"user","content":"raw"}],"tools":[{"original":true}],"model_options":{"custom":false}})";
  for (bool has_tools : {false, true}) {
    Prepare(0, true, has_tools);
    const auto result = Render(request);
    const auto base = Field(result, "base"),
               projected = Field(result, "projected");
    ASSERT_TRUE(base);
    ASSERT_TRUE(projected);
    EXPECT_EQ(Field(*base, "messages"), R"([{"role":"user","content":"raw"}])");
    EXPECT_EQ(Field(*base, "tools"), R"([{"original":true}])");
    EXPECT_EQ(Field(*projected, "messages"),
              R"([{"role":"assistant","content":null}])");
    if (has_tools)
      EXPECT_EQ(Field(*projected, "tools"), "null");
    else
      EXPECT_FALSE(Field(*projected, "tools"));
    EXPECT_EQ(Field(*projected, "custom"), "false");
    EXPECT_EQ(Field(*projected, "model_options"), R"({"custom":false})");
    EXPECT_EQ(Field(*projected, "compiler_global"), "immutable");
  }
  EXPECT_EQ(
      request,
      R"({"messages":[{"role":"user","content":"raw"}],"tools":[{"original":true}],"model_options":{"custom":false}})");
}

TEST_P(InputEnvironmentTest, RejectsMalformedNativeShapesAndRecovers) {
  Prepare();
  for (auto request : {"{}", "[]", "null", "{\"messages\":{}}",
                       "{\"messages\":[],\"model_options\":null}"}) {
    SCOPED_TRACE(request);
    std::string output = "preserved", error;
    EXPECT_FALSE(prepared_->Render(request, &output, &error));
    EXPECT_EQ(output, "preserved");
  }
  EXPECT_EQ(Field(Render(R"({"messages":[],"value":0})"), "value"), "0");
  Prepare(2);
  std::string output = "preserved", error;
  EXPECT_FALSE(prepared_->Render(R"({"messages":[]})", &output, &error));
  EXPECT_EQ(output, "preserved");
}

TEST_P(InputEnvironmentTest, EmitsAdaptersFromActualAnalyzedSourceAndBindings) {
  constexpr std::string_view source =
      "{{ bos_token | default('') }}{{ eos_token | default('') }}"
      "{% for message in messages %}{% if message.role == 'assistant' %}"
      "{{ '<assistant>' + message.content + '</assistant>' }}"
      "{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}";
  TextFormatBindings globals;
  std::string binding = "quote\" slash\\ line\n café";
  binding.push_back('\0');
  binding += "tail";
  globals.strings["bos_token"] = binding;
  const auto expect_binding = [&](std::string_view environment) {
    const auto encoded = Field(environment, "bos_token");
    ASSERT_TRUE(encoded);
    std::string decoded(encoded->size(), '\0');
    iree_host_size_t length = 0;
    const auto status = iree_json_unescape_string(
        iree_make_string_view(encoded->data(), encoded->size()), decoded.size(),
        decoded.data(), &length);
    const bool ok = iree_status_is_ok(status);
    iree_status_ignore(status);
    ASSERT_TRUE(ok);
    decoded.resize(length);
    EXPECT_EQ(decoded, binding);
  };
  TextFormatAnalysis analysis;
  Diagnostic diagnostic;
  ASSERT_TRUE(AnalyzeTextFormat(source, "plain-binding.jinja", globals,
                                &analysis, &diagnostic))
      << diagnostic.message;
  GeneratedInputRenderer selector;
  ASSERT_TRUE(CompileTemplateSelector(
      source, "plain-binding.jinja", analysis.adaptation_selector_template,
      &selector, &diagnostic, TemplateSelectorKind::Adaptation))
      << diagnostic.message;
  PrepareAdapters(analysis, selector, globals, {"eos_token"});
  const auto result = Render(
      R"({"messages":[{"role":"assistant","content":null,"extra":"preserved"}],"model_options":{"new_control":false}})");
  expect_binding(result);
  EXPECT_FALSE(Field(result, "eos_token"));
  EXPECT_FALSE(Field(result, "tools"));
  EXPECT_EQ(Field(result, "new_control"), "false");
  const auto messages = Field(result, "messages");
  ASSERT_TRUE(messages);
  const auto rows = Elements(*messages);
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(Field(rows[0], "content"), "");
  EXPECT_EQ(Field(rows[0], "extra"), "preserved");
  EXPECT_EQ(Field(Render(R"({"messages":[],"tools":null})"), "tools"), "null");
  EXPECT_EQ(Field(Render(R"({"messages":[],"tools":[]})"), "tools"), "[]");
  std::string output = "preserved", error;
  EXPECT_FALSE(prepared_->Render(
      R"({"messages":[],"eos_token":"cannot replace absent optional binding"})",
      &output, &error));
  EXPECT_NE(error.find("request binding cannot override: eos_token"),
            std::string::npos);
  EXPECT_EQ(output, "preserved");
  PrepareAdapters(analysis, selector, globals, {"eos_token"}, false, 1);
  const auto completed = Render(
      R"({"messages":[{"role":"assistant","content":null}],"add_generation_prompt":true,"model_options":{"add_generation_prompt":true}})");
  EXPECT_EQ(Field(completed, "add_generation_prompt"), "false");
  expect_binding(completed);
  EXPECT_FALSE(Field(completed, "eos_token"));
}

TEST_P(InputEnvironmentTest,
       AdapterConsumerUsesOriginalPerMessageAndPerCallViews) {
  // Explicit consumer-contract fixture, not a claim about source inference.
  // Fixed adaptation bits exercise the emitter independently of the analyzer
  // that computes these source-dependent predicates.
  TextFormatAnalysis analysis;
  const auto &message = analysis.adaptation_message_binding;
  const auto &message_index = analysis.adaptation_message_index_binding;
  const auto &call = analysis.adaptation_call_binding;
  const auto &call_index = analysis.adaptation_call_index_binding;
  const std::string selector_source =
      "{% if " + call + " is defined and " + message + ".mapping_enabled and " +
      call + ".function.name == 'decode' and " + message_index +
      " == " + message + ".expected_index and " + call_index + " == " + call +
      ".expected_index and messages[0].content is none %}"
      "{{ [1, 0] }}{% else %}{{ [6, 0] }}{% endif %}";
  Template ast;
  Diagnostic diagnostic;
  ASSERT_TRUE(ParseTemplate(selector_source, &ast, &diagnostic));
  GeneratedInputRenderer selector;
  ASSERT_TRUE(CompileTemplateSelector(selector_source, "adapter-consumer.jinja",
                                      ast, &selector, &diagnostic,
                                      TemplateSelectorKind::Adaptation))
      << diagnostic.message;
  PrepareAdapters(analysis, selector, {}, {}, true);
  const std::string request =
      R"({"messages":[)"
      R"({"role":"assistant","content":null,"mapping_enabled":true,"expected_index":0,"tool_calls":[{"id":"first","extra":9,"expected_index":0,"function":{"name":"decode","arguments":"{\"left\":1}"}}]},)"
      R"({"role":"assistant","content":null,"mapping_enabled":true,"expected_index":1,"tool_calls":[{"id":"mapped","expected_index":0,"function":{"name":"decode","arguments":"{\"next\":2}"}},{"id":"raw","expected_index":1,"function":{"name":"raw","arguments":"{\"keep\":3}"}}]},)"
      R"({"role":"custom","content":[{"type":"text","text":"typed"}],"mapping_enabled":false,"expected_index":2,"tool_calls":[{"id":"third","expected_index":0,"function":{"name":"decode","arguments":"{\"unchanged\":4}"}}]},)"
      R"({"role":"assistant","content":null,"mapping_enabled":true,"expected_index":3})"
      R"(],")" +
      call + R"(":{"expected_index":0,"function":{"name":"decode"}},")" +
      call_index + R"(":0})";
  const auto result = Render(request);
  const auto rows_value = Field(result, "messages");
  ASSERT_TRUE(rows_value);
  const auto rows = Elements(*rows_value);
  ASSERT_EQ(rows.size(), 4u);
  EXPECT_EQ(Field(rows[0], "content"), "");
  EXPECT_EQ(Field(rows[1], "content"), "null");
  EXPECT_EQ(Field(rows[2], "content"), R"([{"type":"text","text":"typed"}])");
  EXPECT_EQ(Field(rows[3], "content"), "null");
  const auto calls0_value = Field(rows[0], "tool_calls");
  const auto calls1_value = Field(rows[1], "tool_calls");
  const auto calls2_value = Field(rows[2], "tool_calls");
  ASSERT_TRUE(calls0_value);
  ASSERT_TRUE(calls1_value);
  ASSERT_TRUE(calls2_value);
  const auto calls0 = Elements(*calls0_value);
  const auto calls1 = Elements(*calls1_value);
  const auto calls2 = Elements(*calls2_value);
  ASSERT_EQ(calls0.size(), 1u);
  ASSERT_EQ(calls1.size(), 2u);
  ASSERT_EQ(calls2.size(), 1u);
  EXPECT_EQ(Field(calls0[0], "id"), "first");
  EXPECT_EQ(Field(calls0[0], "extra"), "9");
  EXPECT_EQ(Arguments(calls0[0]), R"({"left":1})");
  EXPECT_EQ(Arguments(calls1[0]), R"({"next":2})");
  EXPECT_EQ(Arguments(calls1[1]), R"({\"keep\":3})");
  EXPECT_EQ(Arguments(calls2[0]), R"({\"unchanged\":4})");
}

TEST_P(InputEnvironmentTest,
       GeneratedMappingRejectsInvalidWireArgumentsAndRecovers) {
  TextFormatAnalysis analysis;
  constexpr std::string_view source = "{{ [1, 0] }}";
  Template ast;
  Diagnostic diagnostic;
  ASSERT_TRUE(ParseTemplate(source, &ast, &diagnostic));
  GeneratedInputRenderer selector;
  ASSERT_TRUE(CompileTemplateSelector(source, "mapping-contract.jinja", ast,
                                      &selector, &diagnostic,
                                      TemplateSelectorKind::Adaptation))
      << diagnostic.message;
  PrepareAdapters(analysis, selector, {}, {});
  for (
      auto request :
      {R"({"messages":[{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":"[]"}}]}]})",
       R"({"messages":[{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":{}}}]}]})"}) {
    std::string output = "preserved", error;
    EXPECT_FALSE(prepared_->Render(request, &output, &error));
    EXPECT_EQ(output, "preserved");
  }
  const auto result = Render(
      R"({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"kept","type":"function","function":{"name":"f","arguments":"{}"}}]}]})");
  const auto messages = Field(result, "messages");
  ASSERT_TRUE(messages);
  const auto rows = Elements(*messages);
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(Field(rows[0], "content"), "");
  const auto calls_value = Field(rows[0], "tool_calls");
  ASSERT_TRUE(calls_value);
  const auto calls = Elements(*calls_value);
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_EQ(Field(calls[0], "id"), "kept");
  EXPECT_EQ(Arguments(calls[0]), "{}");
}

TEST_P(InputEnvironmentTest,
       RejectsUnknownAdaptationBitsWithoutPublishingOutput) {
  TextFormatAnalysis analysis;
  constexpr std::string_view source = "{{ [8, 0] }}";
  Template ast;
  Diagnostic diagnostic;
  ASSERT_TRUE(ParseTemplate(source, &ast, &diagnostic));
  GeneratedInputRenderer selector;
  ASSERT_TRUE(CompileTemplateSelector(source, "invalid-mask.jinja", ast,
                                      &selector, &diagnostic,
                                      TemplateSelectorKind::Adaptation));
  PrepareAdapters(analysis, selector, {}, {});
  std::string output = "preserved", error;
  EXPECT_FALSE(prepared_->Render(
      R"({"messages":[{"role":"user","content":"input"}]})", &output, &error));
  EXPECT_EQ(output, "preserved");
  EXPECT_FALSE(error.empty());
}

INSTANTIATE_TEST_SUITE_P(SerializedForms, InputEnvironmentTest,
                         ::testing::Bool());

} // namespace
} // namespace hrx::loomify_gguf::jinja::testing
