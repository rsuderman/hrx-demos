// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/matching.h"
#include "lib/loomify/chat_templates.h"
#include <algorithm>

namespace hrx::loomify_gguf::text_policy {
std::string_view Fixture(const char *name) {
  const auto *files = chat_templates_create();
  for (size_t i = 0; i < chat_templates_size(); ++i)
    if (std::string_view(files[i].name) == name)
      return {files[i].data, files[i].size};
  return {};
}
bool Utf8(std::string_view value) {
  for (size_t i = 0; i < value.size();) {
    const auto c = static_cast<unsigned char>(value[i++]);
    if (c < 0x80)
      continue;
    unsigned n = 0, cp = 0, minimum = 0;
    if (c >= 0xc2 && c <= 0xdf) {
      n = 1;
      cp = c & 31;
      minimum = 0x80;
    } else if (c >= 0xe0 && c <= 0xef) {
      n = 2;
      cp = c & 15;
      minimum = 0x800;
    } else if (c >= 0xf0 && c <= 0xf4) {
      n = 3;
      cp = c & 7;
      minimum = 0x10000;
    } else
      return false;
    if (n > value.size() - i)
      return false;
    for (unsigned j = 0; j < n; ++j) {
      const auto next = static_cast<unsigned char>(value[i++]);
      if ((next & 0xc0) != 0x80)
        return false;
      cp = (cp << 6) | (next & 63);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
      return false;
  }
  return true;
}
bool Quoted(std::string_view source, size_t begin, std::string *value,
            size_t *end, bool multiline = false) {
  if (begin >= source.size() || (source[begin] != '\'' && source[begin] != '"'))
    return false;
  const char quote = source[begin];
  std::string decoded;
  for (size_t i = begin + 1; i < source.size(); ++i) {
    char c = source[i];
    if (c == quote) {
      if (!Utf8(decoded))
        return false;
      *value = std::move(decoded);
      *end = i + 1;
      return true;
    }
    if (static_cast<unsigned char>(c) < 32 && !(multiline && c == '\n'))
      return false;
    if (c == '\\') {
      if (++i == source.size())
        return false;
      switch (source[i]) {
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case '\\':
      case '\'':
      case '"':
        c = source[i];
        break;
      default:
        return false;
      }
    }
    if (decoded.size() == 65536)
      return false;
    decoded.push_back(c);
  }
  return false;
}
bool Tags(std::string_view value, std::vector<std::string> *tags) {
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '>')
      return false;
    if (value[i] != '<')
      continue;
    const auto end = value.find('>', i + 1);
    if (end == std::string_view::npos ||
        value.substr(i + 1, end - i - 1).find('<') != std::string_view::npos)
      return false;
    tags->emplace_back(value.substr(i, end - i + 1));
    i = end;
  }
  return true;
}
bool AddSlot(std::string_view source, const char *name, std::string_view anchor,
             size_t quote_offset, std::vector<Slot> *slots, bool multiline) {
  const auto found = source.find(anchor);
  if (found == std::string_view::npos ||
      source.find(anchor, found + 1) != std::string_view::npos)
    return false;
  Slot slot;
  slot.name = name;
  slot.begin = found + quote_offset;
  slot.multiline = multiline;
  if (!Quoted(source, slot.begin, &slot.original, &slot.end, multiline))
    return false;
  slots->push_back(std::move(slot));
  return true;
}
bool Capture(std::string_view expected, std::string_view actual,
             std::vector<Slot> slots, Captures *values) {
  std::sort(slots.begin(), slots.end(),
            [](const Slot &a, const Slot &b) { return a.begin < b.begin; });
  size_t prior = 0, cursor = 0;
  for (const auto &slot : slots) {
    const auto fixed = expected.substr(prior, slot.begin - prior);
    if (actual.substr(cursor, fixed.size()) != fixed)
      return false;
    cursor += fixed.size();
    if (cursor >= actual.size() || actual[cursor] != expected[slot.begin])
      return false;
    std::string value;
    size_t end = 0;
    if (!Quoted(actual, cursor, &value, &end, slot.multiline))
      return false;
    std::vector<std::string> original_tags, actual_tags;
    if (!Tags(slot.original, &original_tags) || !Tags(value, &actual_tags) ||
        original_tags != actual_tags)
      return false;
    if (!values->emplace(slot.name, std::move(value)).second)
      return false;
    cursor = end;
    prior = slot.end;
  }
  return actual.substr(cursor) == expected.substr(prior);
}
std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : value) {
    result += digits[c >> 4];
    result += digits[c & 15];
  }
  return result;
}
void Datum(ChatPolicySelection *policy, const char *name,
           std::string_view value) {
  policy->configuration_source += "global.rodata.def public @" +
                                  std::string(name) + " = bytes(\"" +
                                  Hex(value) + "\")\n";
  policy->data_roots.push_back(name);
}
namespace {
// Fixed implementation details of maintained products, not checkpoint options.
void FixedDatum(ChatPolicySelection *policy, const char *name,
                std::string_view hex) {
  policy->configuration_source += "global.rodata.def public @" +
                                  std::string(name) + " = bytes(\"" +
                                  std::string(hex) + "\")\n";
  policy->data_roots.push_back(name);
}
void Leaf(ChatPolicySelection *policy, const char *name, std::string_view args,
          std::string_view result, std::string_view body) {
  policy->configuration_source +=
      "func.def public @" + std::string(name) + "(" + std::string(args) +
      ") -> (" + std::string(result) + ") {\n" + std::string(body) + "\n}\n";
  policy->data_roots.push_back(name);
}
void Constant(ChatPolicySelection *policy, const char *name,
              std::string_view args, std::string_view type,
              std::string_view value) {
  Leaf(policy, name, args, type,
       "%v = scalar.constant " + std::string(value) + " : " +
           std::string(type) + "\nfunc.return %v : " + std::string(type));
}
void Identity(ChatPolicySelection *policy, const char *name,
              std::string_view args, std::string_view type,
              std::string_view values) {
  Leaf(policy, name, args, type,
       "func.return " + std::string(values) + " : " + std::string(type));
}
void Literal(ChatPolicySelection *policy, const char *name,
             std::string_view args, std::string_view hex) {
  const auto storage = "loomify.fixed." + std::string(name);
  policy->configuration_source += "global.rodata.def @" + storage +
                                  " = bytes(\"" + std::string(hex) + "\")\n";
  Leaf(policy, name, args, "buffer",
       "%v = global.load @" + storage + " : buffer\nfunc.return %v : buffer");
}
} // namespace
void EmitFixedTextDefaults(ChatRevision revision, ChatPolicySelection *policy) {
  switch (revision) {
  case ChatRevision::ChatmlCompactReasoning:
  case ChatRevision::ChatmlExplicitReasoning:
  case ChatRevision::ChatmlHermes:
  case ChatRevision::ChatmlHermesNonThinking:
  case ChatRevision::ChatmlInlineHistory:
  case ChatRevision::ChatmlPlain:
  case ChatRevision::SparkGlm:
    break;
  default:
    return;
  }
  const bool compact = revision == ChatRevision::ChatmlCompactReasoning;
  const bool explicit_reasoning =
      revision == ChatRevision::ChatmlExplicitReasoning;
  const bool hermes = revision == ChatRevision::ChatmlHermes;
  const bool non_thinking = revision == ChatRevision::ChatmlHermesNonThinking;
  const bool inline_history = revision == ChatRevision::ChatmlInlineHistory;
  const bool plain = revision == ChatRevision::ChatmlPlain;
  const bool spark = revision == ChatRevision::SparkGlm;

  if (hermes || non_thinking)
    Leaf(policy, "chatml.admit_request", "%request: buffer", "", "func.return");
  if (plain || inline_history || explicit_reasoning || hermes || non_thinking) {
    Constant(policy, "chatml.supports_preserve_thinking", "", "i1",
             explicit_reasoning ? "true" : "false");
    Constant(policy, "chatml.python_tool_literals", "", "i1",
             inline_history || hermes ? "true" : "false");
  }
  if (explicit_reasoning || inline_history)
    Constant(policy, "chatml.allow_later_system", "", "i1", "false");
  if (!(explicit_reasoning || inline_history))
    Constant(policy, "chatml.allow_later_system", "", "i1", "true");
  if (non_thinking || inline_history || plain)
    Constant(policy, "chatml.default_thinking", "", "i1", "false");
  if (!(inline_history || non_thinking || plain))
    Constant(policy, "chatml.default_thinking", "", "i1", "true");
  if (compact || explicit_reasoning)
    Constant(policy, "chatml.emit_empty_system", "", "i1", "false");
  if (!(compact || explicit_reasoning))
    Constant(policy, "chatml.emit_empty_system", "", "i1", "true");
  if (explicit_reasoning || inline_history)
    Constant(policy, "chatml.first_tool_header", "", "i1", "false");
  if (!(explicit_reasoning || inline_history))
    Constant(policy, "chatml.first_tool_header", "", "i1", "true");
  if (compact)
    Constant(policy, "chatml.group_tool_responses", "", "i1", "false");
  if (!compact)
    Constant(policy, "chatml.group_tool_responses", "", "i1", "true");
  if (!(explicit_reasoning || spark))
    Constant(policy, "chatml.has_effort", "", "i1", "false");
  if (explicit_reasoning || spark)
    Constant(policy, "chatml.has_effort", "", "i1", "true");
  if (non_thinking || plain || spark)
    Constant(policy, "chatml.history_state", "%last_user: i64, %count: i64",
             "i64", "0");
  if (non_thinking || plain)
    Constant(policy, "chatml.history_thought",
             "%state: i64, %ordinal: i64, %reason: buffer", "i1", "false");
  if (spark)
    Constant(policy, "chatml.history_thought",
             "%state: i64, %ordinal: i64, %reason: buffer", "i1", "true");
  if (!(hermes || inline_history))
    Constant(policy, "chatml.inline_reasoning", "", "i1", "false");
  if (hermes || inline_history)
    Constant(policy, "chatml.inline_reasoning", "", "i1", "true");
  if (!explicit_reasoning)
    Constant(policy, "chatml.keep_reasoning_default", "", "i1", "false");
  if (explicit_reasoning)
    Constant(policy, "chatml.keep_reasoning_default", "", "i1", "true");
  if (!(hermes || plain))
    Constant(policy, "chatml.model_emits_think_open", "", "i1", "false");
  if (hermes)
    Constant(policy, "chatml.model_emits_think_open", "", "i1", "true");
  if (compact || spark)
    Constant(policy, "chatml.no_user_index", "%last: i64", "i64", "-1");
  if (compact || spark)
    Constant(policy, "chatml.python_tool_literals", "", "i1", "false");
  if (non_thinking)
    Constant(policy, "chatml.raw_string_arguments", "", "i1", "false");
  if (hermes)
    Constant(policy, "chatml.raw_string_arguments", "", "i1", "true");
  if (!(explicit_reasoning || inline_history))
    Constant(policy, "chatml.require_user", "", "i1", "false");
  if (explicit_reasoning || inline_history)
    Constant(policy, "chatml.require_user", "", "i1", "true");
  if (compact || spark)
    Constant(policy, "chatml.strip_after_think_newlines", "", "i1", "false");
  if (!(compact || spark))
    Constant(policy, "chatml.strip_after_think_newlines", "", "i1", "true");
  if (spark)
    Constant(policy, "chatml.supports_preserve_thinking", "", "i1", "false");
  if (compact)
    Constant(policy, "chatml.supports_preserve_thinking", "", "i1", "true");
  if (non_thinking || plain)
    Constant(policy, "chatml.supports_thinking", "", "i1", "false");
  if (!(non_thinking || plain))
    Constant(policy, "chatml.supports_thinking", "", "i1", "true");
  if (explicit_reasoning || inline_history || spark)
    Constant(policy, "chatml.system_before_tools", "", "i1", "false");
  if (!(explicit_reasoning || inline_history || spark))
    Constant(policy, "chatml.system_before_tools", "", "i1", "true");
  if (!spark)
    FixedDatum(policy, "chatml.assistant_open",
               "3c7c696d5f73746172747c3e617373697374616e740a");
  if (spark)
    FixedDatum(
        policy, "chatml.assistant_open",
        "3cefbd9c7374617274e296816f66e2968173656e74656e6365efbd9c3e3c7c426f747c3e");
  if (!spark)
    FixedDatum(policy, "chatml.end", "3c7c696d5f656e647c3e0a");
  if (spark)
    FixedDatum(policy, "chatml.end",
               "3cefbd9c656e64e296816f66e2968173656e74656e6365efbd9c3e");
  if (!spark)
    FixedDatum(policy, "chatml.system_open",
               "3c7c696d5f73746172747c3e73797374656d0a");
  if (spark)
    FixedDatum(
        policy, "chatml.system_open",
        "3cefbd9c7374617274e296816f66e2968173656e74656e6365efbd9c3e3c7c53797374656d7c3e0a");
  if (!spark)
    FixedDatum(policy, "chatml.user_open",
               "3c7c696d5f73746172747c3e757365720a");
  if (spark)
    FixedDatum(
        policy, "chatml.user_open",
        "3cefbd9c7374617274e296816f66e2968173656e74656e6365efbd9c3e3c7c557365727c3e");
  if (!spark)
    FixedDatum(policy, "chatml_output.assistant",
               "3c7c696d5f73746172747c3e617373697374616e740a3c7468696e6b3e0a");
  if (spark)
    FixedDatum(
        policy, "chatml_output.assistant",
        "3cefbd9c7374617274e296816f66e2968173656e74656e6365efbd9c3e3c7c426f747c3e3c7468696e6b3e");
  if (!spark)
    FixedDatum(policy, "chatml_output.end", "3c7c696d5f656e647c3e0a");
  if (spark)
    FixedDatum(policy, "chatml_output.end",
               "3cefbd9c656e64e296816f66e2968173656e74656e6365efbd9c3e");
  if (!spark)
    FixedDatum(policy, "chatml_output.plain_assistant",
               "3c7c696d5f73746172747c3e617373697374616e740a");
  if (spark)
    FixedDatum(
        policy, "chatml_output.plain_assistant",
        "3cefbd9c7374617274e296816f66e2968173656e74656e6365efbd9c3e3c7c426f747c3e");
  if (!spark)
    FixedDatum(policy, "input_open_boundary", "3c7c696d5f656e647c3e0a");
  if (spark)
    FixedDatum(policy, "input_open_boundary",
               "3cefbd9c656e64e296816f66e2968173656e74656e6365efbd9c3e");
  if (spark)
    FixedDatum(policy, "input_terminated_boundary", "");
  if (!spark)
    FixedDatum(policy, "input_terminated_boundary", "0a");
  if (compact || explicit_reasoning || inline_history)
    FixedDatum(policy, "parameter_end", "0a3c2f706172616d657465723e0a");
  if (spark)
    FixedDatum(policy, "parameter_end", "3c2f6172675f76616c75653e");
  if (spark)
    FixedDatum(policy, "parameter_key_end",
               "3c2f6172675f6b65793e3c6172675f76616c75653e");
  if (compact || explicit_reasoning || inline_history)
    FixedDatum(policy, "parameter_key_end", "3e0a");
  if (spark)
    FixedDatum(policy, "parameter_open", "3c6172675f6b65793e");
  if (compact || explicit_reasoning || inline_history)
    FixedDatum(policy, "parameter_open", "3c706172616d657465723d");
  if (spark)
    FixedDatum(
        policy, "spark.tool_header",
        "3cefbd9c7374617274e296816f66e2968173656e74656e6365efbd9c3e3c7c546f6f6c7c3e");
  if (spark)
    FixedDatum(policy, "tag_end", "");
  if (compact || explicit_reasoning || inline_history)
    FixedDatum(policy, "tag_end", "3e0a");
  if (compact || explicit_reasoning || inline_history)
    FixedDatum(policy, "tool_close",
               "3c2f66756e6374696f6e3e0a3c2f746f6f6c5f63616c6c3e");
  if (spark)
    FixedDatum(policy, "tool_close", "3c2f746f6f6c5f63616c6c3e");
  if (spark)
    FixedDatum(policy, "tool_open", "3c746f6f6c5f63616c6c3e");
  if (compact || explicit_reasoning || inline_history)
    FixedDatum(policy, "tool_open",
               "3c746f6f6c5f63616c6c3e0a3c66756e6374696f6e3d");
  if (!(compact || spark))
    Identity(
        policy, "chatml.assistant_parts",
        "%raw: buffer,%text: buffer,%reason: buffer,%fields: i64,%retain: i1",
        "buffer,buffer", "%text,%reason");
  if (spark)
    Identity(policy, "chatml.history_calls", "%raw: buffer, %rendered: buffer",
             "buffer", "%rendered");
  if (compact || explicit_reasoning || inline_history)
    Identity(policy, "chatml.history_calls", "%raw: buffer,%rendered: buffer",
             "buffer", "%rendered");
  if (spark)
    Identity(policy, "chatml.is_last_query", "%user: i1, %wrapped: i1", "i1",
             "%user");
  if (compact)
    Identity(policy, "chatml.is_last_query", "%user: i1,%wrapped: i1", "i1",
             "%user");
  if (!(compact || spark))
    Identity(policy, "chatml.no_user_index", "%last: i64", "i64", "%last");
  if (!(explicit_reasoning || inline_history))
    Identity(policy, "chatml.normalize_content", "%text: buffer", "buffer",
             "%text");
  if (!(explicit_reasoning || hermes || inline_history))
    Identity(policy, "chatml.normalize_reason", "%text: buffer", "buffer",
             "%text");
  if (!hermes)
    Identity(policy, "chatml.retained_body", "%text: buffer", "buffer",
             "%text");
  if (!(compact || spark))
    Identity(policy, "chatml.schema_value", "%tool: buffer", "buffer", "%tool");
  if (compact || plain || spark)
    Literal(policy, "chatml.call_separator", "", "");
  if (hermes || non_thinking)
    Literal(policy, "chatml.call_separator", "", "0a");
  if (explicit_reasoning || inline_history)
    Literal(policy, "chatml.call_separator", "", "0a0a");
  if (!(compact || spark))
    Literal(policy, "chatml.completion_think_close", "",
            "0a3c2f7468696e6b3e0a0a");
  if (compact || spark)
    Literal(policy, "chatml.completion_think_close", "", "3c2f7468696e6b3e");
  if (compact || spark)
    Literal(policy, "chatml.completion_think_open", "", "3c7468696e6b3e");
  if (!(compact || spark))
    Literal(policy, "chatml.completion_think_open", "", "3c7468696e6b3e0a");
  if (non_thinking || plain)
    Literal(policy, "chatml.generation_thought", "%thinking: i1", "");
  if (plain)
    Literal(policy, "chatml.history_calls", "%raw: buffer, %rendered: buffer",
            "");
  if (!compact)
    Literal(policy, "chatml.prompt_prefix", "", "");
  if (spark)
    Literal(policy, "chatml.system_tools_separator", "%has: i1, %has_text: i1",
            "");
  if (plain)
    Literal(policy, "chatml.system_tools_separator",
            "%has_system: i1,%has_text: i1", "");
  if (spark)
    Literal(policy, "chatml.tool_call_gap", "", "");
  if (!(compact || spark))
    Literal(policy, "chatml.tool_call_gap", "", "0a");
  if (compact)
    Literal(policy, "chatml.tool_call_gap", "", "0a0a");
}

} // namespace hrx::loomify_gguf::text_policy
