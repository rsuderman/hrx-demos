// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/testing/template_qualification.h"
#include "lib/loomify/template_analysis_internal.h"
#include "iree/base/internal/json.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::jinja::internal {
bool RecipeBuilder::Build(const TextFormatAlternative &format, Template *out,
                          std::string *error) {
  continuation_ = nullptr;
  if (analysis_.selector_prelude_size >
      analysis_.selector_template.body.size()) {
    *error = "invalid analyzed recipe dependency prefix";
    return false;
  }
  out->body.assign(analysis_.selector_template.body.begin(),
                   analysis_.selector_template.body.begin() +
                       analysis_.selector_prelude_size);
  if (!analysis_.target_missing_binding.empty()) {
    const auto &binding = analysis_.target_missing_binding;
    if (out->body.empty() || out->body.front().kind != Node::Kind::Set ||
        out->body.front().text != binding ||
        out->body.front().expression.kind != Expr::Kind::Integer ||
        out->body.front().expression.text != "1") {
      *error = "invalid analyzed target-domain initializer";
      return false;
    }
    for (size_t i = 1; i < out->body.size(); ++i)
      if (out->body[i].kind == Node::Kind::Set &&
          out->body[i].text == binding) {
        *error = "duplicate analyzed target-domain initializer";
        return false;
      }
    // The selector sees original history, while this recipe sees history
    // with the completed assistant already present. Change the compiler's
    // private domain initializer before any dependent snapshot executes.
    out->body.front().expression.text = "0";
  }
  Bind({}, &out->body);
  return Parts(format.assistant_pattern, {}, &out->body, error);
}
} // namespace hrx::loomify_gguf::jinja::internal

namespace hrx::loomify_gguf::jinja {
namespace {
using namespace internal;
ProbeMessage Assistant(int version, int shape) {
  const char suffix = version ? 'B' : 'A';
  ProbeMessage result =
      Message("assistant", std::string("PROBE_CONTENT_") + suffix);
  if (shape == 1 || shape == 5)
    result.reasoning = Quote(std::string("PROBE_REASONING_") + suffix);
  else if (shape == 2)
    result.reasoning = Quote("");
  if (shape >= 3) {
    if (shape == 4)
      result.content = "null";
    const int count = shape == 5 ? 2 : 1;
    for (int i = 0; i < count; ++i)
      result.calls.push_back(
          {std::string("probe_function_") + suffix + std::to_string(i),
           "probe_call_" + std::to_string(i),
           {{std::string("probe_key_") + suffix,
             Quote(std::string("PROBE_VALUE_") + suffix)},
            {"probe_boolean", version ? "false" : "true"},
            {"probe_integer", version ? "-7" : "13"},
            {"probe_null", "null"},
            {"probe_object", "{\"nested\":1}"},
            {"probe_array", "[1,null]"}},
           false});
  }
  return result;
}

bool EndsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

struct Coverage {
  bool route = false;
  std::set<int> shapes[2];
};

struct Recipe {
  std::string assistant_binding;
  std::unique_ptr<PreparedTemplate> program;
};

bool Decode(iree_string_view_t value, std::string *out) {
  out->resize(value.size);
  iree_host_size_t length = 0;
  const auto status =
      iree_json_unescape_string(value, out->size(), out->data(), &length);
  const bool ok = iree_status_is_ok(status);
  iree_status_ignore(status);
  if (ok)
    out->resize(length);
  return ok;
}

struct JsonCallFacts {
  const TextToolFormat *format;
  const ProbeCall *call;
  bool name = false;
  bool arguments = false;
};

iree_status_t ObserveJsonMember(void *user, iree_string_view_t key,
                                iree_json_value_type_t type,
                                iree_string_view_t value) {
  auto *facts = static_cast<JsonCallFacts *>(user);
  std::string decoded_key;
  if (!Decode(key, &decoded_key))
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  if (decoded_key == facts->format->name_field) {
    std::string name;
    facts->name = type == IREE_JSON_VALUE_TYPE_STRING && Decode(value, &name) &&
                  name == facts->call->name;
  }
  if (decoded_key == facts->format->arguments_field) {
    std::string decoded;
    if (type == IREE_JSON_VALUE_TYPE_STRING) {
      if (!Decode(value, &decoded))
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      value = iree_make_string_view(decoded.data(), decoded.size());
    } else if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
      return iree_ok_status();
    }
    // A known unique probe key must be a direct member of the actual argument
    // object. A same-spelled nested key or escaped quote cannot qualify it.
    for (const auto &[expected_key, expected_value] : facts->call->arguments) {
      if (expected_key.rfind("probe_key_", 0) != 0)
        continue;
      iree_string_view_t observed{};
      const auto status = iree_json_lookup_object_value(
          value,
          iree_make_string_view(expected_key.data(), expected_key.size()),
          &observed);
      const bool ok = iree_status_is_ok(status);
      iree_status_ignore(status);
      std::string content;
      facts->arguments =
          ok && Decode(observed, &content) && Quote(content) == expected_value;
    }
  }
  return iree_ok_status();
}

bool ObserveJsonCalls(std::string_view assistant, const TextToolFormat &format,
                      const std::vector<ProbeCall> &calls) {
  size_t cursor = 0;
  for (const auto &call : calls) {
    const auto open = assistant.find(format.open, cursor);
    if (open == std::string_view::npos || format.open.empty() ||
        format.close.empty())
      return false;
    const auto begin = open + format.open.size();
    const auto close = assistant.find(format.close, begin);
    if (close == std::string_view::npos)
      return false;
    // JSON object enumeration expects the opening brace at byte zero. Source
    // envelopes may put JSON whitespace around it; this view normalization
    // does not change the exact assistant recipe comparison above.
    auto object = assistant.substr(begin, close - begin);
    const auto first = object.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
      return false;
    const auto last = object.find_last_not_of(" \t\r\n");
    object = object.substr(first, last - first + 1);
    JsonCallFacts facts{&format, &call};
    const auto status = iree_json_enumerate_object_typed(
        iree_make_string_view(object.data(), object.size()), ObserveJsonMember,
        &facts);
    const bool ok = iree_status_is_ok(status);
    iree_status_ignore(status);
    if (!ok || !facts.name || !facts.arguments)
      return false;
    cursor = close + format.close.size();
  }
  return true;
}

bool ObserveStartup(std::string_view prompt,
                    const TextFormatAlternative &format, int64_t mode) {
  if (format.generation_prefix.empty())
    return mode == static_cast<int64_t>(TextStartupMode::Content);
  if (!EndsWith(prompt, format.generation_prefix))
    return false;
  const auto &prefix = format.generation_prefix;
  const auto open = format.reasoning_open.empty()
                        ? std::string::npos
                        : prefix.rfind(format.reasoning_open);
  const auto close = format.reasoning_close.empty()
                         ? std::string::npos
                         : prefix.rfind(format.reasoning_close);
  const auto expected =
      close != std::string::npos && (open == std::string::npos || close > open)
          ? TextStartupMode::Content
      : open != std::string::npos      ? TextStartupMode::ReasoningOpen
      : !format.reasoning_open.empty() ? TextStartupMode::ReasoningMayOpen
                                       : TextStartupMode::Content;
  return mode == static_cast<int64_t>(expected);
}

// Only a diagnostic search: if the declared projection rejects a constructed
// wire probe, check its known valid JSON argument objects in both wire-string
// and decoded forms. A successful alternative never qualifies coverage or
// becomes a fallback; it proves the declared projection missed a representable
// source case and therefore makes analysis fail.
bool HasRepresentableArguments(PreparedTemplate &renderer, const Object &base,
                               const std::vector<ProbeMessage> &messages) {
  std::vector<std::pair<size_t, size_t>> calls;
  for (size_t i = 0; i < messages.size(); ++i)
    for (size_t j = 0; j < messages[i].calls.size(); ++j)
      calls.emplace_back(i, j);
  std::vector<std::vector<ProbeMessage>> variants{messages};
  for (const auto &[message, call] : calls) {
    const auto count = variants.size();
    for (size_t i = 0; i < count; ++i) {
      variants[i][message].calls[call].mapped = false;
      auto mapped = variants[i];
      mapped[message].calls[call].mapped = true;
      variants.push_back(std::move(mapped));
    }
  }
  for (const auto &variant : variants) {
    auto environment = base;
    environment["messages"] = Messages(variant);
    std::string ignored, error;
    if (renderer.Render(Json(environment), &ignored, &error))
      return true;
  }
  return false;
}

bool HasCompletedSourceWitness(PreparedTemplate &renderer, const Object &base,
                               const std::vector<ProbeMessage> &history) {
  // A valid prompt may still reject every completed assistant (for example,
  // source-enforced role alternation after a system-only conversation). Check
  // the original compiled source directly, independently of inferred formats
  // and adaptation selectors, before requiring a future-output route.
  auto completed_base = base;
  completed_base["add_generation_prompt"] = "false";
  for (int version = 0; version < 2; ++version)
    for (int shape = 0; shape < 6; ++shape) {
      auto completed = history;
      completed.push_back(Assistant(version, shape));
      if (HasRepresentableArguments(renderer, completed_base, completed))
        return true;
    }
  return false;
}

bool Complete(const TextFormatAnalysis &analysis,
              const std::vector<Coverage> &coverage) {
  for (size_t i = 0; i < coverage.size(); ++i) {
    const auto &facts = analysis.alternatives[i];
    const auto &seen = coverage[i];
    if (!seen.route)
      return false;
    for (int version = 0; version < 2; ++version) {
      if (!seen.shapes[version].count(0) || !seen.shapes[version].count(2))
        return false;
      if (HasHistoryReasoning(facts) && !seen.shapes[version].count(1))
        return false;
      if (facts.tools.kind != TextToolKind::None)
        for (int shape : {3, 4, 5})
          if (!seen.shapes[version].count(shape))
            return false;
    }
  }
  return true;
}
} // namespace
using namespace internal;

bool QualifyTemplate(std::string_view source, std::string_view source_name,
                     const TextFormatBindings &bindings,
                     const AnalyzedTemplate &result,
                     const TemplateAnalysisOptions &options,
                     TemplateQualification *out, Diagnostic *diagnostic) {
  if (!out)
    return Fail(source, source_name, 0, "qualification output must not be null",
                diagnostic);
  TemplateQualification report;
  std::unique_ptr<PreparedTemplate> renderer, selector, adaptation_selector;
  std::string error;
  for (const auto &[generated, prepared] :
       std::vector<std::pair<const GeneratedInputRenderer *,
                             std::unique_ptr<PreparedTemplate> *>>{
           {&result.renderer, &renderer},
           {&result.selector, &selector},
           {&result.adaptation_selector, &adaptation_selector}})
    if (!PreparedTemplate::Prepare(*generated, options.helper_root,
                                   options.scratch_parent, options.bytecode,
                                   prepared, &error))
      return Fail(source, source_name, 0,
                  "cannot prepare compiled template analysis: " + error,
                  diagnostic);

  std::vector<Recipe> recipes;
  for (const auto &format : result.format.alternatives) {
    RecipeBuilder builder(result.format, source);
    Template ast;
    GeneratedInputRenderer generated;
    Recipe recipe;
    if (!builder.Build(format, &ast, &error))
      return Fail(source, source_name, Offset(format),
                  "cannot construct bounded format recipe: " + error,
                  diagnostic);
    if (!CompileTemplateRenderer(source, source_name, ast, &generated,
                                 diagnostic))
      return false;
    if (!PreparedTemplate::Prepare(generated, options.helper_root,
                                   options.scratch_parent, options.bytecode,
                                   &recipe.program, &error))
      return Fail(source, source_name, Offset(format),
                  "cannot prepare bounded format recipe: " + error, diagnostic);
    recipe.assistant_binding = builder.assistant_binding();
    recipes.push_back(std::move(recipe));
  }

  Template original;
  if (!ParseTemplate(source, &original, diagnostic))
    return false;
  std::vector<Coverage> coverage(result.format.alternatives.size());
  std::optional<size_t> current_candidate;
  std::string last_error;
  const auto Failure = [&](std::string_view text, std::string_view name,
                           size_t offset, const std::string &message,
                           Diagnostic *out_diagnostic) {
    Fail(text, name, offset, message, out_diagnostic);
    return InputVisit::Fail;
  };
  if (!VisitTemplateInputs(
          original, result.renderer, result.format, bindings,
          [&](const TemplateInput &candidate) {
            if (!current_candidate ||
                *current_candidate != candidate.candidate_index) {
              // Complete coverage must not skip the remaining history/tool
              // counterexamples belonging to this same candidate assignment.
              if (candidate.route_candidate &&
                  Complete(result.format, coverage))
                return InputVisit::Stop;
              current_candidate = candidate.candidate_index;
            }
            const auto &environment = candidate.environment;
            const auto &history = candidate.history;
            const int history_kind = candidate.history_kind;
            const int tools_kind = candidate.tools_kind;
            std::vector<ProbeMessage> normalized_history;
            if (!Normalize(environment, history, result.format,
                           *adaptation_selector, &normalized_history, &error)) {
              last_error = "live wire projection: " + error;
              if (HasRepresentableArguments(*renderer, environment, history))
                return Failure(
                    source, source_name, 0,
                    "projection selector rejected a source-representable "
                    "wire history: " +
                        error,
                    diagnostic);
              return InputVisit::Continue;
            }
            Object live_environment = environment;
            live_environment["messages"] = Messages(normalized_history);
            const auto live_json = Json(live_environment);
            std::string prompt;
            if (!renderer->Render(live_json, &prompt, &error)) {
              last_error = error;
              if (HasRepresentableArguments(*renderer, environment, history))
                return Failure(
                    source, source_name, 0,
                    "declared projection lost a source-representable "
                    "wire history: " +
                        error,
                    diagnostic);
              return InputVisit::Continue;
            }
            int64_t format_id = -1, startup = -1;
            if (!selector->SelectFormat(live_json, &format_id, &startup,
                                        &error)) {
              // This exclusion earns no route, adaptation or shape coverage. A
              // source-accepted completion still makes selector failure fatal.
              if (!HasCompletedSourceWitness(*renderer, environment, history))
                return InputVisit::Continue;
              return Failure(
                  source, source_name, 0,
                  "compiled selector rejected a source-valid environment "
                  "(history " +
                      std::to_string(history_kind) + ", tools " +
                      std::to_string(tools_kind) + "): " + error,
                  diagnostic);
            }
            if (format_id < 0 ||
                static_cast<size_t>(format_id) >= coverage.size() ||
                startup < 0 || startup > 2)
              return Failure(
                  source, source_name, 0,
                  "compiled selector returned an invalid format/startup",
                  diagnostic);
            const auto &facts = result.format.alternatives[format_id];
            if (!ObserveStartup(prompt, facts, startup))
              return Failure(
                  source, source_name, Offset(facts),
                  "compiled generation prefix/startup disagrees with "
                  "format route " +
                      std::to_string(format_id),
                  diagnostic);
            auto &seen = coverage[format_id];
            seen.route = true;
            for (int version = 0; version < 2; ++version) {
              for (int shape = 0; shape < 6; ++shape) {
                const bool calls = shape >= 3;
                if ((shape == 1 && facts.reasoning_open.empty() &&
                     !HasHistoryReasoning(facts)) ||
                    (calls && facts.tools.kind == TextToolKind::None))
                  continue;
                auto completed_history = history;
                completed_history.push_back(Assistant(version, shape));
                Object completed_base = environment;
                completed_base["add_generation_prompt"] = "false";
                completed_base["messages"] = Messages(completed_history, true);
                std::vector<ProbeMessage> normalized_completed;
                if (!Normalize(completed_base, completed_history, result.format,
                               *adaptation_selector, &normalized_completed,
                               &error)) {
                  last_error = "completed wire projection: " + error;
                  if (HasRepresentableArguments(*renderer, completed_base,
                                                completed_history))
                    return Failure(
                        source, source_name, Offset(facts),
                        "projection selector rejected source-representable "
                        "assistant shape " +
                            std::to_string(shape) + ": " + error,
                        diagnostic);
                  continue;
                }
                Object completed = completed_base;
                completed["messages"] = Messages(normalized_completed);
                std::string rendered;
                if (!renderer->Render(Json(completed), &rendered, &error)) {
                  last_error = "completed render: " + error;
                  if (HasRepresentableArguments(*renderer, completed_base,
                                                completed_history))
                    return Failure(
                        source, source_name, Offset(facts),
                        "declared projection lost source-representable "
                        "assistant shape " +
                            std::to_string(shape) + ": " + error,
                        diagnostic);
                  continue;
                }
                Object recipe_environment = completed;
                recipe_environment[recipes[format_id].assistant_binding] =
                    normalized_completed.back().Serialize();
                std::string expected;
                if (!recipes[format_id].program->Render(
                        Json(recipe_environment), &expected, &error))
                  return Failure(source, source_name, Offset(facts),
                                 "format recipe execution failed for route " +
                                     std::to_string(format_id) + ", shape " +
                                     std::to_string(shape) + ": " + error,
                                 diagnostic);
                ++report.validated_probe_count;
                const auto &suffix = expected;
                if (expected.empty() || !EndsWith(rendered, suffix))
                  return Failure(
                      source, source_name, Offset(facts),
                      "exact compiled assistant recipe mismatch for route " +
                          std::to_string(format_id) + ", shape " +
                          std::to_string(shape) + ", history " +
                          std::to_string(history_kind) + ", tools " +
                          std::to_string(tools_kind) + "; expected suffix (" +
                          std::to_string(suffix.size()) + " bytes) tail " +
                          Quote(std::string_view(suffix).substr(
                              suffix.size() > 512 ? suffix.size() - 512 : 0)) +
                          "; actual completed render (" +
                          std::to_string(rendered.size()) + " bytes) tail " +
                          Quote(std::string_view(rendered).substr(
                              rendered.size() > 512 ? rendered.size() - 512
                                                    : 0)),
                      diagnostic);
                if (calls && facts.tools.kind == TextToolKind::JsonEnvelope &&
                    !ObserveJsonCalls(expected, facts.tools,
                                      normalized_completed.back().calls))
                  return Failure(
                      source, source_name, Offset(facts),
                      "observed JSON member/capture disagrees with format "
                      "route " +
                          std::to_string(format_id),
                      diagnostic);
                seen.shapes[version].insert(shape);
              }
            }
            return InputVisit::Continue;
          }))
    return false;
  if (Complete(result.format, coverage)) {
    *out = std::move(report);
    if (diagnostic)
      *diagnostic = {};
    return true;
  }
  for (size_t i = 0; i < coverage.size(); ++i) {
    if (!coverage[i].route)
      return Fail(
          source, source_name, Offset(result.format.alternatives[i]),
          "cannot construct a valid compiled witness for format route " +
              std::to_string(i) + ": " + last_error,
          diagnostic);
    for (int version = 0; version < 2; ++version) {
      const auto &facts = result.format.alternatives[i];
      for (int shape : {0, 1, 2, 3, 4, 5}) {
        if ((shape == 1 && !HasHistoryReasoning(facts)) ||
            (shape >= 3 && facts.tools.kind == TextToolKind::None))
          continue;
        if (!coverage[i].shapes[version].count(shape))
          return Fail(source, source_name, Offset(facts),
                      "no valid compiled witness for format route " +
                          std::to_string(i) + ", semantic shape " +
                          std::to_string(shape) + ", value set " +
                          std::to_string(version) + ": " + last_error,
                      diagnostic);
      }
    }
  }
  return Fail(source, source_name, 0,
              "compiled probes did not cover every declared format, semantic "
              "capture: " +
                  last_error,
              diagnostic);
}

} // namespace hrx::loomify_gguf::jinja
