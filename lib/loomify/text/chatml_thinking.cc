// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/matching.h"
#include "lib/serving/library_resources.h"

namespace hrx::loomify_gguf::text_policy {
namespace {
struct Revision {
  const char *fixture;
  ChatRevision revision;
  const char *sha;
  TextTokenRequirements tokens;
  const char *product;
};
const Revision kRevisions[] = {
    {"7f0e5290.jinja",
     ChatRevision::ChatmlInlineHistory,
     "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67",
     {"gpt2", "qwen35", false, "", "<|im_end|>"},
     resources::kChatmlInlineHistory},
    {"c3cf9e34.jinja",
     ChatRevision::ChatmlExplicitReasoning,
     "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
     {"gpt2", "qwen35", false, "", "<|im_end|>"},
     resources::kChatmlExplicitReasoning},
    {"5da44855.jinja",
     ChatRevision::ChatmlHermes,
     "5da44855ab7e0641774d5ed0ff0b4f89671d01df7d316e54d06557a962211cb4",
     {"gpt2", "qwen2", false, "", "<|im_end|>"},
     resources::kChatmlHermes},
    {"cd8e9439.jinja",
     ChatRevision::ChatmlHermesNonThinking,
     "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f",
     {"gpt2", "qwen2", false, "", "<|im_end|>"},
     resources::kChatmlHermesNonThinking},
};
bool IsHermes(ChatRevision revision) {
  return revision == ChatRevision::ChatmlHermes ||
         revision == ChatRevision::ChatmlHermesNonThinking;
}
bool CaptureRevision(const Revision &revision, std::string_view expected,
                     std::string_view actual, Captures *values) {
  std::vector<Slot> slots;
  const bool non_thinking =
      revision.revision == ChatRevision::ChatmlHermesNonThinking;
  const char *tools_prefix =
      non_thinking ? "{{- \"\\n\\n# Tools" : "{{- \"# Tools";
  if (!AddSlot(expected, "tools_prefix", tools_prefix, 4, &slots) ||
      !AddSlot(expected, "tools_instruction",
               IsHermes(revision.revision) ? "{{- \"\\n</tools>"
                                           : "{{- '\\n\\nIf you choose",
               4, &slots))
    return false;
  if (revision.revision == ChatRevision::ChatmlExplicitReasoning) {
    const std::string effort = "reasoning_effort|default(";
    const std::string assignment = "{%- set reasoning_instructions = ";
    if (!AddSlot(expected, "default_effort", effort, effort.size(), &slots) ||
        !AddSlot(expected, "effort_xhigh",
                 assignment + "'Reasoning effort is set to xhigh.",
                 assignment.size(), &slots) ||
        !AddSlot(expected, "effort_low",
                 assignment + "'Reasoning effort is set to low.",
                 assignment.size(), &slots))
      return false;
  }
  if (non_thinking &&
      (!AddSlot(expected, "default_tools",
                "{{- 'You are Qwen, created by Alibaba Cloud.", 4, &slots) ||
       !AddSlot(
           expected, "default_frame",
           "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud.",
           4, &slots)))
    return false;
  if (!Capture(expected, actual, std::move(slots), values))
    return false;
  if (IsHermes(revision.revision)) {
    constexpr std::string_view end = "<|im_end|>\n";
    const auto &value = values->at("tools_instruction");
    if (value.size() < end.size() ||
        value.substr(value.size() - end.size()) != end)
      return false;
  }
  if (non_thinking) {
    constexpr std::string_view begin = "<|im_start|>system\n";
    constexpr std::string_view end = "<|im_end|>\n";
    const auto &frame = values->at("default_frame");
    if (frame.size() < begin.size() + end.size() ||
        frame.compare(0, begin.size(), begin) ||
        frame.compare(frame.size() - end.size(), end.size(), end) ||
        frame.substr(begin.size(), frame.size() - begin.size() - end.size()) !=
            values->at("default_tools"))
      return false;
  }
  const auto effort = values->find("default_effort");
  if (effort != values->end() && effort->second != "low" &&
      effort->second != "medium" && effort->second != "xhigh")
    return false;
  return true;
}
void EmitData(const Captures &slots, ChatPolicySelection *policy) {
  Datum(policy, "loomify.text.tools_prefix", slots.at("tools_prefix"));
  std::string suffix = slots.at("tools_instruction");
  if (IsHermes(policy->revision)) {
    // Shared ChatML framing adds the terminal exactly once.
    constexpr std::string_view end = "<|im_end|>\n";
    suffix.resize(suffix.size() - end.size());
  } else {
    suffix = "\n</tools>" + suffix;
  }
  Datum(policy, "loomify.text.tools_suffix", suffix);
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
  if (policy->revision == ChatRevision::ChatmlHermesNonThinking)
    Datum(policy, "loomify.text.default_system", slots.at("default_tools"));
}
} // namespace
Match SelectChatmlThinking(std::string_view actual, ChatPolicySelection *out) {
  for (const auto &revision : kRevisions) {
    const auto expected = Fixture(revision.fixture);
    if (expected.empty())
      return Match::InvalidBuiltin;
    Captures values;
    if (!CaptureRevision(revision, expected, actual, &values))
      continue;
    ChatPolicySelection result;
    result.revision = revision.revision;
    result.skeleton_sha256 = revision.sha;
    result.tokens = revision.tokens;
    result.text_product = revision.product;
    result.markers = {
        {"<|im_start|>", true}, {"<|im_end|>", true}, {"<think>"},
        {"</think>"},           {"<tool_call>"},      {"</tool_call>"},
        {"<tool_response>"},    {"</tool_response>"}};
    if (!IsHermes(revision.revision))
      result.markers.insert(
          result.markers.end(),
          {{"<function="}, {"</function>"}, {"<parameter="}, {"</parameter>"}});
    if (revision.revision == ChatRevision::ChatmlHermesNonThinking)
      result.markers.erase(result.markers.begin() + 2,
                           result.markers.begin() + 4);
    EmitData(values, &result);
    *out = std::move(result);
    return Match::Matched;
  }
  return Match::NoMatch;
}
} // namespace hrx::loomify_gguf::text_policy
