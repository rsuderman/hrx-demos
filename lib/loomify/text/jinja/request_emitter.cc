// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/request_emitter.h"
#include "lib/loomify/text/jinja/format_analysis.h"

#include <array>
#include <set>
#include <utility>

namespace hrx::loomify_gguf::jinja {
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

void Literal(std::string_view name, std::string_view value, std::string *out) {
  *out += "global.rodata.def @" + std::string(name) + " = bytes(\"" +
          Hex(value) + "\")\n";
}

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
  source += R"(
func.decl @text.json_quote(%value: buffer) -> (buffer)
func.decl @text.input_set_member(%object: buffer, %key: buffer, %value: buffer) -> (buffer)
func.decl @text.join8(%a: buffer, %b: buffer, %c: buffer, %d: buffer, %e: buffer, %f: buffer, %g: buffer, %h: buffer) -> (buffer)
)";
  for (size_t i = 0; i < workers.size(); ++i)
    Literal("request_adapters.worker." + std::to_string(i), workers[i],
            &source);
  source += R"(
func.def public @text.request_worker_names() -> (buffer, buffer, buffer, buffer) {
  %message = global.load @request_adapters.worker.0 : buffer
  %message_index = global.load @request_adapters.worker.1 : buffer
  %call = global.load @request_adapters.worker.2 : buffer
  %call_index = global.load @request_adapters.worker.3 : buffer
  func.return %message, %message_index, %call, %call_index : buffer, buffer, buffer, buffer
}
)";

  size_t index = 0;
  for (const auto &[name, value] : globals.strings) {
    Literal("request_adapters.global_key." + std::to_string(index), name,
            &source);
    Literal("request_adapters.global_value." + std::to_string(index), value,
            &source);
    ++index;
  }
  source += R"(
func.def public @text.template_globals() -> (buffer) {
  %initial = buffer.constant utf8("{}") : buffer
)";
  std::string object = "%initial";
  for (size_t i = 0; i < globals.strings.size(); ++i) {
    const auto suffix = std::to_string(i);
    source += "  %key" + suffix +
              " = global.load @request_adapters.global_key." + suffix +
              " : buffer\n";
    source += "  %value" + suffix +
              " = global.load @request_adapters.global_value." + suffix +
              " : buffer\n";
    source += "  %quoted" + suffix + " = func.call @text.json_quote(%value" +
              suffix + ") : (buffer) -> (buffer)\n";
    source += "  %object" + suffix + " = func.call @text.input_set_member(" +
              object + ", %key" + suffix + ", %quoted" + suffix +
              ") : (buffer, buffer, buffer) -> (buffer)\n";
    object = "%object" + suffix;
  }
  source += "  func.return " + object + " : buffer\n}\n";
  index = 0;
  for (const auto &name : protected_bindings)
    Literal("request_adapters.protected." + std::to_string(index++), name,
            &source);
  source += R"(
func.def public @text.request_protected_names() -> (buffer) {
  %initial = buffer.constant utf8("[") : buffer
  %empty = buffer.constant utf8("") : buffer
  %comma = buffer.constant utf8(",") : buffer
  %close = buffer.constant utf8("]") : buffer
)";
  std::string array = "%initial";
  for (size_t i = 0; i < protected_bindings.size(); ++i) {
    const auto suffix = std::to_string(i);
    source += "  %name" + suffix +
              " = global.load @request_adapters.protected." + suffix +
              " : buffer\n";
    source += "  %quoted" + suffix + " = func.call @text.json_quote(%name" +
              suffix + ") : (buffer) -> (buffer)\n";
    source +=
        "  %array" + suffix + " = func.call @text.join8(" + array + ", " +
        (i ? "%comma" : "%empty") + ", %quoted" + suffix +
        ", %empty, %empty, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)\n";
    array = "%array" + suffix;
  }
  source +=
      "  %result = func.call @text.join8(" + array +
      ", %close, %empty, %empty, %empty, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)\n";
  source += "  func.return %result : buffer\n}\n";
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
