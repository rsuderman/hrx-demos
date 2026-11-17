// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/request_emitter.h"
#include "lib/loomify/text/jinja/emission.h"
#include "lib/loomify/text/jinja/format_analysis.h"

#include <array>
#include <set>
#include <utility>

namespace hrx::loomify_gguf::jinja {
namespace {
bool Fail(std::string_view name, std::string message, Diagnostic *diagnostic) {
  if (diagnostic) {
    *diagnostic = Diagnostic{};
    diagnostic->source_name = std::string(name);
    diagnostic->message = std::move(message);
  }
  return false;
}
} // namespace

bool EmitRequestAdapters(std::string_view source_name,
                         const TextFormatAnalysis &analysis,
                         const GeneratedInputRenderer &adaptation_selector,
                         const TextFormatBindings &globals,
                         const std::vector<std::string> &protected_names,
                         GeneratedInputRenderer *out, Diagnostic *diagnostic) {
  if (!out || adaptation_selector.entrypoint != "jinja.select_adaptation" ||
      adaptation_selector.source.empty())
    return Fail(
        source_name,
        "request adaptation needs a compiled adaptation selector and output",
        diagnostic);
  const std::array<std::string, 4> workers{
      analysis.adaptation_message_binding,
      analysis.adaptation_message_index_binding,
      analysis.adaptation_call_binding, analysis.adaptation_call_index_binding};
  std::set<std::string> worker_names;
  for (const auto &name : workers)
    if (name.empty() || !worker_names.insert(name).second)
      return Fail(source_name,
                  "adaptation worker bindings must be distinct nonempty names",
                  diagnostic);

  std::set<std::string> protected_bindings(protected_names.begin(),
                                           protected_names.end());
  protected_bindings.insert(workers.begin(), workers.end());
  protected_bindings.insert(adaptation_selector.protected_bindings.begin(),
                            adaptation_selector.protected_bindings.end());
  for (const auto &[name, value] : globals.strings) {
    if (worker_names.count(name))
      return Fail(source_name,
                  "template global collides with adaptation worker binding",
                  diagnostic);
    protected_bindings.insert(name);
  }

  GeneratedInputRenderer generated;
  generated.entrypoint = "text.adapt_messages";
  generated.helper_requirements = {"lib/text/common/strings.loom",
                                   "lib/text/common/json.loom",
                                   "lib/text/generated_input/environment.loom",
                                   "lib/text/generated_input/request.loom"};
  generated.protected_bindings.assign(protected_bindings.begin(),
                                      protected_bindings.end());
  auto &source = generated.source;
  for (size_t i = 0; i < workers.size(); ++i)
    internal::AppendBufferLiteral(
        "request_adapters.worker." + std::to_string(i), workers[i], &source);
  source += R"(
func.def public @text.request_worker_names() -> (buffer, buffer, buffer, buffer) {
  %message = global.load @request_adapters.worker.0 : buffer
  %message_index = global.load @request_adapters.worker.1 : buffer
  %call = global.load @request_adapters.worker.2 : buffer
  %call_index = global.load @request_adapters.worker.3 : buffer
  func.return %message, %message_index, %call, %call_index : buffer, buffer, buffer, buffer
}
)";

  std::string object = "{";
  for (const auto &[name, value] : globals.strings) {
    if (object.size() != 1)
      object += ',';
    object += internal::QuoteJsonString(name) + ':' +
              internal::QuoteJsonString(value);
  }
  object += '}';
  std::string array = "[";
  for (const auto &name : protected_bindings) {
    if (array.size() != 1)
      array += ',';
    array += internal::QuoteJsonString(name);
  }
  array += ']';
  internal::AppendBufferLiteral("request_adapters.globals", object, &source);
  internal::AppendBufferLiteral("request_adapters.protected_names", array,
                                &source);
  source += R"(
func.def public @text.template_globals() -> (buffer) {
  %globals = global.load @request_adapters.globals : buffer
  func.return %globals : buffer
}
func.def public @text.request_protected_names() -> (buffer) {
  %names = global.load @request_adapters.protected_names : buffer
  func.return %names : buffer
}
)";
  GeneratedInputRenderer result;
  Diagnostic error;
  if (!CombineGeneratedTemplates({adaptation_selector, generated},
                                 generated.entrypoint, &result, &error))
    return Fail(source_name, error.message, diagnostic);
  *out = std::move(result);
  if (diagnostic)
    *diagnostic = Diagnostic{};
  return true;
}

} // namespace hrx::loomify_gguf::jinja
