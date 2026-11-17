// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/template_analysis.h"
#include "lib/loomify/text/jinja/compiler.h"
#include "lib/loomify/text/jinja/output_emitter.h"
#include "lib/loomify/text/jinja/request_emitter.h"
#include "lib/loomify/text/jinja/source_template.h"
#include <memory>
#include <string_view>
#include <utility>

namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
std::string Hex(std::string_view text) {
  constexpr char digits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(text.size() * 2);
  for (unsigned char c : text) {
    encoded.push_back(digits[c >> 4]);
    encoded.push_back(digits[c & 15]);
  }
  return encoded;
}
bool TemplateError(const jinja::Diagnostic &diagnostic, std::string *error) {
  return Fail(error, diagnostic.source_name + ":" +
                         std::to_string(diagnostic.line) + ":" +
                         std::to_string(diagnostic.column) + ": " +
                         diagnostic.message);
}
void EmitInputBytes(std::string_view symbol, std::string_view value,
                    std::string *source) {
  *source += "global.rodata.def @" + std::string(symbol) + " = bytes(\"" +
             Hex(value) + "\")\n";
}
void EmitInputSeparator(std::string_view name, std::string_view value,
                        std::string *source) {
  EmitInputBytes(std::string(name) + ".bytes", value, source);
  *source += "func.def public @" + std::string(name) +
             "() -> (buffer) {\n  %value = global.load @" + std::string(name) +
             ".bytes : buffer\n"
             "  func.return %value : buffer\n}\n";
}
void EmitUnavailableInput(std::string_view name, bool text_argument,
                          std::string_view diagnostic, std::string *source) {
  EmitInputBytes(std::string(name) + ".diagnostic", diagnostic, source);
  *source += "func.def public @" + std::string(name) +
             (text_argument ? "(%text: buffer)" : "()") +
             " -> (buffer) {\n  %false = scalar.constant 0 : i32\n"
             "  %diagnostic = global.load @" +
             std::string(name) +
             ".diagnostic : buffer\n"
             "  func.call @require(%false, %diagnostic) : (i32, buffer)\n"
             "  %empty = buffer.constant utf8(\"\") : buffer\n"
             "  func.return %empty : buffer\n}\n";
}
jinja::GeneratedInputRenderer
EmitInputPreparation(const ChatTemplateSelection &selected,
                     const jinja::AnalyzedTemplate &analyzed,
                     const BoundTokenPolicy &tokens) {
  jinja::GeneratedInputRenderer generated;
  generated.helper_requirements = {"lib/text/generated_input/prepare.loom"};
  auto &source = generated.source;
  source = R"(
func.decl import("input") @require(%condition: i32, %diagnostic: buffer)
func.decl @text.json_quote(%value: buffer) -> (buffer)
func.decl @text.input_set_member(%object: buffer, %key: buffer, %value: buffer) -> (buffer)
)";
  if (analyzed.continuation) {
    const auto &continuation = *analyzed.continuation;
    EmitInputBytes("generated_input.text_binding", continuation.text_binding,
                   &source);
    source +=
        "func.def public @text.input_user_continuation(%text: buffer) -> (buffer) {\n"
        "  %globals = func.call @text.template_globals() : () -> (buffer)\n"
        "  %key = global.load @generated_input.text_binding : buffer\n"
        "  %quoted = func.call @text.json_quote(%text) : (buffer) -> (buffer)\n"
        "  %environment = func.call @text.input_set_member(%globals, %key, %quoted) : (buffer, buffer, buffer) -> (buffer)\n"
        "  %fragment = func.call @" +
        continuation.renderer.entrypoint +
        "(%environment) : (buffer) -> (buffer)\n"
        "  func.return %fragment : buffer\n}\n";
  } else {
    const auto &diagnostic = analyzed.continuation_diagnostic;
    const auto message =
        selected.source_key + ":" + std::to_string(diagnostic.line) + ":" +
        std::to_string(diagnostic.column) + ": " +
        (diagnostic.message.empty()
             ? "history-free continued USER recipe is unavailable"
             : diagnostic.message);
    EmitUnavailableInput("text.input_user_continuation", true, message,
                         &source);
  }

  // The default-qualified USER product supplies its actual closure. Without
  // it, agreement of every accepted format still proves one universal closure
  // for RENDERED continuations, independently of USER recipe availability.
  std::optional<std::string> closure;
  if (analyzed.continuation) {
    closure = analyzed.continuation->assistant_closure;
  } else if (!analyzed.format.alternatives.empty()) {
    closure = analyzed.format.alternatives.front().frame_suffix;
    for (const auto &format : analyzed.format.alternatives)
      if (format.frame_suffix != *closure) {
        closure.reset();
        break;
      }
  }
  const auto &terminal = tokens.terminal.piece;
  if (closure && !terminal.empty() &&
      closure->compare(0, terminal.size(), terminal) == 0 &&
      closure->find_first_not_of(" \t\r\n\f\v", terminal.size()) ==
          std::string::npos) {
    EmitInputSeparator("text.input_open_separator", *closure, &source);
    EmitInputSeparator("text.input_terminated_separator",
                       std::string_view(*closure).substr(terminal.size()),
                       &source);
  } else {
    const auto message =
        selected.source_key +
        ": continued input has no source-qualified terminal closure";
    EmitUnavailableInput("text.input_open_separator", false, message, &source);
    EmitUnavailableInput("text.input_terminated_separator", false, message,
                         &source);
  }
  return generated;
}
} // namespace
bool PrepareGeneratedText(
    const ChatTemplateSelection &selected, FILE *gguf,
    const model::TextMetadata &metadata,
    const std::optional<PublicationTokenFacts> &publication,
    std::string_view tokenizer_json,
    const jinja::TemplateAnalysisOptions &analysis_options,
    PreparedTextControl *out, std::string *error) {
  if (!out || !gguf)
    return Fail(error, "generated text preparation needs GGUF and output");

  jinja::Diagnostic diagnostic;
  std::unique_ptr<jinja::internal::SourceTemplate> source;
  jinja::GeneratedInputRenderer renderer;
  if (!jinja::internal::SourceTemplate::Parse(
          selected.source_text, selected.source_key, &source, &diagnostic) ||
      !jinja::internal::CompileChatTemplate(*source, &renderer, &diagnostic))
    return TemplateError(diagnostic, error);

  PreparedTokenBindings token_bindings;
  if (!PrepareTokenBindings(renderer.external_bindings, metadata,
                            tokenizer_json, &token_bindings, error))
    return false;
  jinja::TextFormatBindings globals;
  globals.strings = token_bindings.globals();
  jinja::AnalyzedTemplate analyzed;
  if (!jinja::internal::AnalyzeTemplate(
          *source, renderer, globals, analysis_options, &analyzed, &diagnostic))
    return TemplateError(diagnostic, error);

  std::vector<std::string> frame_suffixes;
  frame_suffixes.reserve(analyzed.format.alternatives.size());
  for (const auto &format : analyzed.format.alternatives)
    frame_suffixes.push_back(format.frame_suffix);
  BoundTokenPolicy tokens;
  // The output parsers consume marker bytes, not individual token IDs. They
  // impose no additional atomic-token requirements on template literals.
  if (!FinalizeTokenPolicy(gguf, token_bindings, metadata, publication,
                           frame_suffixes, {}, &tokens, error))
    return false;

  auto protected_names = analyzed.renderer.protected_bindings;
  protected_names.insert(protected_names.end(),
                         token_bindings.global_names().begin(),
                         token_bindings.global_names().end());
  jinja::GeneratedInputRenderer adapters, output, turn;
  if (!jinja::EmitRequestAdapters(selected.source_key, analyzed.format,
                                  analyzed.adaptation_selector, globals,
                                  protected_names, &adapters, &diagnostic) ||
      !jinja::EmitOutputParser(selected.source_key, analyzed.format, &output,
                               &diagnostic) ||
      !jinja::internal::EmitTextTurnControls(*source, analyzed.format, globals,
                                             &turn, &diagnostic)) {
    jinja::LocateDiagnostic(selected.source_text, selected.source_key,
                            &diagnostic);
    return TemplateError(diagnostic, error);
  }

  jinja::GeneratedInputRenderer host;
  host.helper_requirements = {"lib/text/generated_input/host.loom"};
  auto input = EmitInputPreparation(selected, analyzed, tokens);
  std::vector<jinja::GeneratedInputRenderer> products{
      analyzed.renderer, analyzed.selector, adapters, output, turn};
  if (analyzed.continuation)
    products.push_back(analyzed.continuation->renderer);
  products.push_back(std::move(input));
  products.push_back(std::move(host));
  jinja::GeneratedInputRenderer combined;
  if (!jinja::CombineGeneratedTemplates(products, analyzed.renderer.entrypoint,
                                        &combined, &diagnostic))
    return TemplateError(diagnostic, error);

  PreparedTextControl prepared;
  prepared.token_source = std::move(tokens.text_source);
  prepared.source = prepared.token_source + combined.source;
  prepared.source +=
      "func.def public @chat_reasoning() -> (i32, i32, i32) {\n"
      "  %enabled = scalar.constant " +
      std::to_string(analyzed.reasoning.enabled ? 1 : 0) +
      " : i32\n  %efforts = scalar.constant " +
      std::to_string(analyzed.reasoning.effort_mask) +
      " : i32\n  %default = scalar.constant " +
      std::to_string(analyzed.reasoning.default_effort) +
      " : i32\n  func.return %enabled, %efforts, %default : i32, i32, i32\n}\n";
  prepared.helper_requirements = std::move(combined.helper_requirements);
  prepared.roots = {"chat_begin",    "chat_message",   "chat_end",
                    "complete_text", "render_tool",    "output_next",
                    "parse_tools",   "chat_reasoning", "prepare_input"};
  *out = std::move(prepared);
  if (error)
    error->clear();
  return true;
}

bool SelectChatTemplateSource(const model::TextMetadata &metadata,
                              const std::optional<std::string> &template_name,
                              ChatTemplateSelection *out, std::string *error) {
  if (!out)
    return Fail(error, "chat template selection needs an output slot");
  if (template_name && template_name->empty())
    return Fail(error, "chat template name must not be empty");
  if (!metadata.template_issues.empty()) {
    const auto &issue = metadata.template_issues.front();
    return Fail(error, "invalid chat template metadata at " + issue.source_key +
                           ": " + issue.message);
  }
  const model::ChatTemplateSource *selected = nullptr;
  if (template_name) {
    for (const auto &candidate : metadata.templates) {
      if (!candidate.name || *candidate.name != *template_name)
        continue;
      if (!candidate.listed)
        return Fail(error, "chat template is not listed: " + *template_name);
      if (selected)
        return Fail(error,
                    "duplicate selected chat template: " + *template_name);
      selected = &candidate;
    }
    if (!selected)
      return Fail(error, "chat template name is absent: " + *template_name);
  } else {
    for (const auto &candidate : metadata.templates)
      if (!candidate.name) {
        if (selected)
          return Fail(error, "multiple scalar chat template defaults");
        selected = &candidate;
      }
    if (!selected) {
      for (const auto &candidate : metadata.templates)
        if (candidate.listed) {
          if (selected)
            return Fail(error,
                        "named chat templates require an explicit selection");
          selected = &candidate;
        }
    }
    if (!selected)
      return Fail(error, "GGUF has no selectable chat template");
  }
  ChatTemplateSelection result;
  result.source_key = selected->source_key;
  result.template_name = selected->name;
  result.source_text = selected->text;
  *out = std::move(result);
  if (error)
    error->clear();
  return true;
}

} // namespace hrx::loomify_gguf
