// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/matching.h"
#include "lib/serving/library_resources.h"

namespace hrx::loomify_gguf::text_policy {
namespace {
bool AddToolSlots(std::string_view expected, std::string_view prefix_anchor,
                  std::string_view instruction_anchor,
                  std::vector<Slot> *slots) {
  return AddSlot(expected, "tools_prefix", prefix_anchor, 4, slots) &&
         AddSlot(expected, "tools_instruction", instruction_anchor, 4, slots);
}
void EmitData(const Captures &slots, std::string_view tools_suffix,
              ChatPolicySelection *policy) {
  Datum(policy, "loomify.text.tools_prefix", slots.at("tools_prefix"));
  Datum(policy, "loomify.text.tools_suffix", tools_suffix);
  const auto effort = slots.find("default_effort");
  const std::string default_effort =
      effort == slots.end() ? "" : effort->second;
  const auto high = slots.find("effort_xhigh"), low = slots.find("effort_low");
  const std::string high_text = high == slots.end() ? "" : high->second;
  const std::string low_text = low == slots.end() ? "" : low->second;
  Datum(policy, "loomify.text.effort_xhigh", high_text);
  Datum(policy, "loomify.text.effort_low", low_text);
  Datum(policy, "loomify.text.default_effort", default_effort);
  Datum(policy, "loomify.text.default_effort_instruction",
        default_effort == "xhigh" ? high_text
        : default_effort == "low" ? low_text
                                  : "");
}
bool ConfigureInlineHistory(std::string_view expected, std::string_view actual,
                            ChatPolicySelection *candidate) {
  std::vector<Slot> slots;
  Captures values;
  if (!AddToolSlots(expected, "{{- \"# Tools", "{{- '\\n\\nIf you choose",
                    &slots) ||
      !Capture(expected, actual, std::move(slots), &values))
    return false;
  EmitData(values, "\n</tools>" + values.at("tools_instruction"), candidate);
  return true;
}
bool ConfigureExplicitReasoning(std::string_view expected,
                                std::string_view actual,
                                ChatPolicySelection *candidate) {
  std::vector<Slot> slots;
  Captures values;
  const std::string effort = "reasoning_effort|default(";
  const std::string assignment = "{%- set reasoning_instructions = ";
  if (!AddToolSlots(expected, "{{- \"# Tools", "{{- '\\n\\nIf you choose",
                    &slots) ||
      !AddSlot(expected, "default_effort", effort, effort.size(), &slots) ||
      !AddSlot(expected, "effort_xhigh",
               assignment + "'Reasoning effort is set to xhigh.",
               assignment.size(), &slots) ||
      !AddSlot(expected, "effort_low",
               assignment + "'Reasoning effort is set to low.",
               assignment.size(), &slots) ||
      !Capture(expected, actual, std::move(slots), &values))
    return false;
  const auto &default_effort = values.at("default_effort");
  if (default_effort != "low" && default_effort != "medium" &&
      default_effort != "xhigh")
    return false;
  EmitData(values, "\n</tools>" + values.at("tools_instruction"), candidate);
  return true;
}
bool ConfigureHermes(std::string_view expected, std::string_view actual,
                     ChatPolicySelection *candidate) {
  std::vector<Slot> slots;
  Captures values;
  if (!AddToolSlots(expected, "{{- \"# Tools", "{{- \"\\n</tools>", &slots) ||
      !Capture(expected, actual, std::move(slots), &values))
    return false;
  // Shared ChatML framing adds the terminal exactly once.
  constexpr std::string_view end = "<|im_end|>\n";
  const auto &instruction = values.at("tools_instruction");
  if (instruction.size() < end.size() ||
      instruction.compare(instruction.size() - end.size(), end.size(), end))
    return false;
  EmitData(values, instruction.substr(0, instruction.size() - end.size()),
           candidate);
  return true;
}
bool ConfigureHermesNonThinking(std::string_view expected,
                                std::string_view actual,
                                ChatPolicySelection *candidate) {
  std::vector<Slot> slots;
  Captures values;
  if (!AddToolSlots(expected, "{{- \"\\n\\n# Tools", "{{- \"\\n</tools>",
                    &slots) ||
      !AddSlot(expected, "default_tools",
               "{{- 'You are Qwen, created by Alibaba Cloud.", 4, &slots) ||
      !AddSlot(
          expected, "default_frame",
          "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud.",
          4, &slots) ||
      !Capture(expected, actual, std::move(slots), &values))
    return false;
  constexpr std::string_view begin = "<|im_start|>system\n";
  constexpr std::string_view end = "<|im_end|>\n";
  const auto &instruction = values.at("tools_instruction");
  if (instruction.size() < end.size() ||
      instruction.compare(instruction.size() - end.size(), end.size(), end))
    return false;
  const auto &frame = values.at("default_frame");
  if (frame.size() < begin.size() + end.size() ||
      frame.compare(0, begin.size(), begin) ||
      frame.compare(frame.size() - end.size(), end.size(), end) ||
      frame.substr(begin.size(), frame.size() - begin.size() - end.size()) !=
          values.at("default_tools"))
    return false;
  EmitData(values, instruction.substr(0, instruction.size() - end.size()),
           candidate);
  Datum(candidate, "loomify.text.default_system", values.at("default_tools"));
  return true;
}
struct Revision {
  const char *fixture;
  ChatRevision revision;
  const char *sha;
  TextTokenRequirements tokens;
  const char *product;
  std::vector<ChatMarker> markers;
  bool (*capture_and_emit)(std::string_view expected, std::string_view actual,
                           ChatPolicySelection *candidate);
};
bool ConfigurePlain(std::string_view expected, std::string_view actual,
                    ChatPolicySelection *candidate) {
  Captures values;
  if (!Capture(expected, actual, {}, &values))
    return false;
  // Shared input declarations have no prose/default content in this protocol.
  for (const char *name :
       {"loomify.text.tools_prefix", "loomify.text.tools_suffix",
        "loomify.text.effort_xhigh", "loomify.text.effort_low",
        "loomify.text.default_effort",
        "loomify.text.default_effort_instruction"})
    Datum(candidate, name, "");
  return true;
}
bool ConfigureCompact(std::string_view expected, std::string_view actual,
                      ChatPolicySelection *candidate) {
  std::vector<Slot> slots;
  Captures values;
  if (!AddSlot(expected, "tools_prefix", "+ \"<tools>\\n\"", 2, &slots) ||
      !AddSlot(expected, "tools_suffix", "+ \"\\n</tools>\"", 2, &slots) ||
      !Capture(expected, actual, std::move(slots), &values))
    return false;
  auto &prefix = values.at("tools_prefix");
  if (prefix.empty() || prefix.back() != '\n')
    return false;
  // Shared traversal owns the newline before every schema, including the first.
  prefix.pop_back();
  EmitData(values, values.at("tools_suffix"), candidate);
  return true;
}
const Revision kRevisions[] = {
    {"f45005c5.jinja",
     ChatRevision::ChatmlCompactReasoning,
     "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818",
     {"gpt2", "lfm2", true, "", "<|im_end|>"},
     resources::kChatmlCompactReasoning,
     {{"<|im_start|>", true},
      {"<|im_end|>", true},
      {"<think>"},
      {"</think>"},
      {"<tool_call>"},
      {"</tool_call>"},
      {"<tool_response>"},
      {"</tool_response>"},
      {"<function="},
      {"</function>"},
      {"<parameter="},
      {"</parameter>"}},
     ConfigureCompact},
    {"58c1a1f0.jinja",
     ChatRevision::ChatmlPlain,
     "58c1a1f04baa7adaeaba1f90267c3d57d9396f0c1f0129bdde9e76d3c6784af7",
     {"gpt2", "dbrx", false, "", "<|im_end|>", false},
     resources::kChatmlPlain,
     {{"<|im_start|>", true}, {"<|im_end|>", true}},
     ConfigurePlain},
    {"7f0e5290.jinja",
     ChatRevision::ChatmlInlineHistory,
     "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67",
     {"gpt2", "qwen35", false, "", "<|im_end|>"},
     resources::kChatmlInlineHistory,
     {{"<|im_start|>", true},
      {"<|im_end|>", true},
      {"<think>"},
      {"</think>"},
      {"<tool_call>"},
      {"</tool_call>"},
      {"<tool_response>"},
      {"</tool_response>"},
      {"<function="},
      {"</function>"},
      {"<parameter="},
      {"</parameter>"}},
     ConfigureInlineHistory},
    {"c3cf9e34.jinja",
     ChatRevision::ChatmlExplicitReasoning,
     "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
     {"gpt2", "qwen35", false, "", "<|im_end|>"},
     resources::kChatmlExplicitReasoning,
     {{"<|im_start|>", true},
      {"<|im_end|>", true},
      {"<think>"},
      {"</think>"},
      {"<tool_call>"},
      {"</tool_call>"},
      {"<tool_response>"},
      {"</tool_response>"},
      {"<function="},
      {"</function>"},
      {"<parameter="},
      {"</parameter>"}},
     ConfigureExplicitReasoning},
    {"5da44855.jinja",
     ChatRevision::ChatmlHermes,
     "5da44855ab7e0641774d5ed0ff0b4f89671d01df7d316e54d06557a962211cb4",
     {"gpt2", "qwen2", false, "", "<|im_end|>"},
     resources::kChatmlHermes,
     {{"<|im_start|>", true},
      {"<|im_end|>", true},
      {"<think>"},
      {"</think>"},
      {"<tool_call>"},
      {"</tool_call>"},
      {"<tool_response>"},
      {"</tool_response>"}},
     ConfigureHermes},
    {"cd8e9439.jinja",
     ChatRevision::ChatmlHermesNonThinking,
     "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
     {"gpt2", "qwen2", false, "", "<|im_end|>"},
     resources::kChatmlHermesNonThinking,
     {{"<|im_start|>", true},
      {"<|im_end|>", true},
      {"<tool_call>"},
      {"</tool_call>"},
      {"<tool_response>"},
      {"</tool_response>"}},
     ConfigureHermesNonThinking},
};
} // namespace
Match SelectChatmlThinking(std::string_view actual, ChatPolicySelection *out) {
  for (const auto &revision : kRevisions) {
    const auto expected = Fixture(revision.fixture);
    if (expected.empty())
      return Match::InvalidBuiltin;
    ChatPolicySelection candidate;
    candidate.revision = revision.revision;
    candidate.skeleton_sha256 = revision.sha;
    candidate.tokens = revision.tokens;
    candidate.text_product = revision.product;
    candidate.markers = revision.markers;
    if (!revision.capture_and_emit(expected, actual, &candidate))
      continue;
    *out = std::move(candidate);
    return Match::Matched;
  }
  return Match::NoMatch;
}
} // namespace hrx::loomify_gguf::text_policy
