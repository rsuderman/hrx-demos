// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/chat_templates.h"
#include "lib/serving/library_resources.h"
#include <string_view>
#include <algorithm>
#include <map>

namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}
const std::vector<const char*> kRoots = {
  "step", "epoch", "encode_epoch", "publish_epoch", "encode_state", "render_tool",
  "prepare_input", "output_next", "complete_text", "model_name", "chat_image_limits",
  "chat_begin", "chat_message", "chat_end", "parse_tools", "request_defaults",
  "request_options", "bind_request"};
struct Revision {
  const char* fixture;
  ChatRevision revision;
  const char* sha;
  const char* family;
  const char* product;
};
const Revision kRevisions[] = {
  {"5816fce1.jinja", ChatRevision::LlamaHeadersBaseline,
   "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
   "llama", resources::kLlamaText},
  {"7f0e5290.jinja", ChatRevision::ChatmlInlineHistory,
   "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67",
   "qwen35", resources::kChatmlInlineHistory},
  {"c3cf9e34.jinja", ChatRevision::ChatmlExplicitReasoning,
   "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
   "qwen35", resources::kChatmlExplicitReasoning},
};
std::string_view Fixture(const char* name) {
  const auto* files = chat_templates_create();
  for (size_t i = 0; i < chat_templates_size(); ++i)
    if (std::string_view(files[i].name) == name)
      return {files[i].data, files[i].size};
  return {};
}
// Only these quoted slots are data. Every intervening source byte, including
// whitespace controls, branches and filters, remains the reviewed skeleton.
struct Slot { std::string name; size_t begin, end; std::string original; };
bool Utf8(std::string_view value) {
  for (size_t i = 0; i < value.size();) {
    const auto c = static_cast<unsigned char>(value[i++]);
    if (c < 0x80) continue;
    unsigned n = 0, cp = 0, minimum = 0;
    if (c >= 0xc2 && c <= 0xdf) { n = 1; cp = c & 31; minimum = 0x80; }
    else if (c >= 0xe0 && c <= 0xef) { n = 2; cp = c & 15; minimum = 0x800; }
    else if (c >= 0xf0 && c <= 0xf4) { n = 3; cp = c & 7; minimum = 0x10000; }
    else return false;
    if (n > value.size() - i) return false;
    for (unsigned j = 0; j < n; ++j) {
      const auto next = static_cast<unsigned char>(value[i++]);
      if ((next & 0xc0) != 0x80) return false;
      cp = (cp << 6) | (next & 63);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
  }
  return true;
}
bool Quoted(std::string_view source, size_t begin, std::string* value, size_t* end) {
  if (begin >= source.size() || (source[begin] != '\'' && source[begin] != '"')) return false;
  const char quote = source[begin];
  std::string decoded;
  for (size_t i = begin + 1; i < source.size(); ++i) {
    char c = source[i];
    if (c == quote) {
      if (!Utf8(decoded)) return false;
      *value = std::move(decoded); *end = i + 1; return true;
    }
    if (static_cast<unsigned char>(c) < 32) return false;
    if (c == '\\') {
      if (++i == source.size()) return false;
      switch (source[i]) {
      case 'n': c = '\n'; break;
      case 'r': c = '\r'; break;
      case 't': c = '\t'; break;
      case '\\': case '\'': case '"': c = source[i]; break;
      default: return false;
      }
    }
    if (decoded.size() == 65536) return false;
    decoded.push_back(c);
  }
  return false;
}
bool Tags(std::string_view value, std::vector<std::string>* tags) {
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '>') return false;
    if (value[i] != '<') continue;
    const auto end = value.find('>', i + 1);
    if (end == std::string_view::npos || value.substr(i + 1, end - i - 1).find('<') != std::string_view::npos) return false;
    tags->emplace_back(value.substr(i, end - i + 1)); i = end;
  }
  return true;
}
bool AddSlot(std::string_view source, const char* name, std::string_view anchor,
             size_t quote_offset, std::vector<Slot>* slots) {
  const auto found = source.find(anchor);
  if (found == std::string_view::npos || source.find(anchor, found + 1) != std::string_view::npos) return false;
  Slot slot; slot.name = name; slot.begin = found + quote_offset;
  if (!Quoted(source, slot.begin, &slot.original, &slot.end)) return false;
  slots->push_back(std::move(slot)); return true;
}
bool Capture(const Revision& revision, std::string_view expected,
             std::string_view actual, std::map<std::string, std::string>* values) {
  std::vector<Slot> slots;
  if (revision.revision == ChatRevision::LlamaHeadersBaseline) {
    const std::string anchor = "{%- else %}\n        {%- set date_string = ";
    if (!AddSlot(expected, "date", anchor, anchor.size(), &slots)) return false;
  } else {
    if (!AddSlot(expected, "tools_prefix", "{{- \"# Tools", 4, &slots) ||
        !AddSlot(expected, "tools_instruction", "{{- '\\n\\nIf you choose", 4, &slots)) return false;
    if (revision.revision == ChatRevision::ChatmlExplicitReasoning) {
      const std::string effort = "reasoning_effort|default(";
      const std::string assignment = "{%- set reasoning_instructions = ";
      if (!AddSlot(expected, "default_effort", effort, effort.size(), &slots) ||
          !AddSlot(expected, "effort_xhigh", assignment + "'Reasoning effort is set to xhigh.", assignment.size(), &slots) ||
          !AddSlot(expected, "effort_low", assignment + "'Reasoning effort is set to low.", assignment.size(), &slots)) return false;
    }
  }
  std::sort(slots.begin(), slots.end(), [](const Slot& a, const Slot& b) { return a.begin < b.begin; });
  size_t prior = 0, cursor = 0;
  for (const auto& slot : slots) {
    const auto fixed = expected.substr(prior, slot.begin - prior);
    if (actual.substr(cursor, fixed.size()) != fixed) return false;
    cursor += fixed.size();
    if (cursor >= actual.size() || actual[cursor] != expected[slot.begin]) return false;
    std::string value; size_t end = 0;
    if (!Quoted(actual, cursor, &value, &end)) return false;
    std::vector<std::string> original_tags, actual_tags;
    if (!Tags(slot.original, &original_tags) || !Tags(value, &actual_tags) || original_tags != actual_tags) return false;
    if (slot.name == "default_effort" && value != "low" && value != "medium" && value != "xhigh") return false;
    if (!values->emplace(slot.name, std::move(value)).second) return false;
    cursor = end; prior = slot.end;
  }
  return actual.substr(cursor) == expected.substr(prior);
}
std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : value) { result += digits[c >> 4]; result += digits[c & 15]; }
  return result;
}
void Datum(ChatPolicySelection* policy, const char* name, std::string_view value) {
  policy->configuration_source += "global.rodata.def public @" + std::string(name) +
      " = bytes(\"" + Hex(value) + "\")\n";
  policy->data_roots.push_back(name);
}
void EmitData(const std::map<std::string, std::string>& slots, ChatPolicySelection* policy) {
  if (policy->preserved_baseline) {
    Datum(policy, "loomify.text.fallback_date", slots.at("date"));
    return;
  }
  Datum(policy, "loomify.text.tools_prefix", slots.at("tools_prefix"));
  Datum(policy, "loomify.text.tools_suffix", "\n</tools>" + slots.at("tools_instruction"));
  const auto effort = slots.find("default_effort");
  const std::string default_effort = effort == slots.end() ? "" : effort->second;
  const auto high = slots.find("effort_xhigh"), low = slots.find("effort_low");
  const std::string high_text = high == slots.end() ? "" : high->second;
  const std::string low_text = low == slots.end() ? "" : low->second;
  Datum(policy, "loomify.text.effort_xhigh", high_text);
  Datum(policy, "loomify.text.effort_low", low_text);
  Datum(policy, "loomify.text.default_effort", default_effort);
  Datum(policy, "loomify.text.default_effort_instruction",
        default_effort == "xhigh" ? high_text : default_effort == "low" ? low_text : "");
}
} // namespace
bool SelectChatPolicy(const model::TextMetadata& metadata,
                      const std::optional<std::string>& template_name,
                      ChatPolicySelection* out, std::string* error) {
  if (!out) return Fail(error, "chat policy needs an output slot");
  if (template_name && template_name->empty())
    return Fail(error, "chat template name must not be empty");
  if (!metadata.template_issues.empty()) {
    const auto& issue = metadata.template_issues.front();
    return Fail(error, "invalid chat template metadata at " + issue.source_key + ": " + issue.message);
  }
  const model::ChatTemplateSource* selected = nullptr;
  if (template_name) {
    for (const auto& candidate : metadata.templates) {
      if (!candidate.name || *candidate.name != *template_name) continue;
      if (!candidate.listed) return Fail(error, "chat template is not listed: " + *template_name);
      if (selected) return Fail(error, "duplicate selected chat template: " + *template_name);
      selected = &candidate;
    }
    if (!selected) return Fail(error, "chat template name is absent: " + *template_name);
  } else {
    for (const auto& candidate : metadata.templates) if (!candidate.name) {
      if (selected) return Fail(error, "multiple scalar chat template defaults");
      selected = &candidate;
    }
    if (!selected) {
      for (const auto& candidate : metadata.templates) if (candidate.listed) {
        if (selected) return Fail(error, "named chat templates require an explicit selection");
        selected = &candidate;
      }
    }
    if (!selected) return Fail(error, "GGUF has no selectable chat template");
  }
  for (const auto& revision : kRevisions) {
    const auto expected = Fixture(revision.fixture);
    if (expected.empty()) return Fail(error, "built-in chat template identity data is missing");
    std::map<std::string, std::string> slots;
    if (!Capture(revision, expected, selected->text, &slots)) continue;
    ChatPolicySelection result;
    result.revision = revision.revision;
    result.source_key = selected->source_key;
    result.template_name = selected->name;
    result.skeleton_sha256 = revision.sha;
    result.source_text = selected->text;
    result.family = revision.family;
    result.text_product = revision.product;
    result.roots = kRoots;
    result.preserved_baseline = revision.revision == ChatRevision::LlamaHeadersBaseline;
    if (result.preserved_baseline) {
      result.markers = {{"<|begin_of_text|>",true}, {"<|start_header_id|>",true},
        {"<|end_header_id|>",true}, {"<|eot_id|>",true}};
    } else {
      result.markers = {{"<|im_start|>",true}, {"<|im_end|>",true},
        {"<think>"}, {"</think>"}, {"<tool_call>"}, {"</tool_call>"},
        {"<function="}, {"</function>"}, {"<parameter="}, {"</parameter>"},
        {"<tool_response>"}, {"</tool_response>"}};

    }
    EmitData(slots, &result);
    *out = std::move(result);
    if (error) error->clear();
    return true;
  }
  return Fail(error, "unsupported exact chat template revision at " + selected->source_key);
}
} // namespace hrx::loomify_gguf
