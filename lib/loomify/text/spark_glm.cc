// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/matching.h"
#include "lib/serving/library_resources.h"

namespace hrx::loomify_gguf::text_policy {
Match SelectSparkGlm(std::string_view actual, ChatPolicySelection *out) {
  const auto expected = Fixture("93d6aa90.jinja");
  if (expected.empty())
    return Match::InvalidBuiltin;
  std::vector<Slot> slots;
  Captures values;
  const std::string effort = "reasoning_effort|default(";
  const std::string assignment = "{%- set reasoning_instructions = ";
  const std::string lead = "{%- set default_system = ";
  const std::string high_branch =
      "{%- if resolved_reasoning_effort == 'high' %}";
  const std::string tools_title = "{{- '\n\n' + ";
  const std::string tools_intro =
      " + 'You have access to the following functions:'";
  if (!AddSlot(expected, "default_effort", effort, effort.size(), &slots) ||
      !AddSlot(expected, "effort_high", high_branch + assignment,
               high_branch.size() + assignment.size(), &slots, true) ||
      !AddSlot(
          expected, "effort_low",
          "{%- elif resolved_reasoning_effort == 'low' %}" + assignment,
          std::string("{%- elif resolved_reasoning_effort == 'low' %}").size() +
              assignment.size(),
          &slots, true) ||
      !AddSlot(expected, "system_lead", lead, lead.size(), &slots) ||
      !AddSlot(expected, "tools_title", tools_title + "'## Tools'",
               tools_title.size(), &slots) ||
      !AddSlot(expected, "tools_intro", tools_intro, 3, &slots) ||
      !Capture(expected, actual, std::move(slots), &values)) {
    return Match::NoMatch;
  }
  if (values.at("default_effort") != "high" &&
      values.at("default_effort") != "low")
    return Match::NoMatch;
  ChatPolicySelection result;
  result.revision = ChatRevision::SparkGlm;
  result.skeleton_sha256 =
      "93d6aa906ec72365efcab3c0109dba83148bdc9bec7c55848623d17483693eda";
  result.tokens = {"gpt2", "spark2_5", true, u8"<｜start▁of▁sentence｜>",
                   u8"<｜end▁of▁sentence｜>"};
  result.text_product = resources::kSparkGlm;
  result.markers = {{u8"<｜start▁of▁sentence｜>", true},
                    {u8"<｜end▁of▁sentence｜>", true},
                    {"<|System|>"},
                    {"<|User|>"},
                    {"<|Bot|>"},
                    {"<|Tool|>"},
                    {"<think>"},
                    {"</think>"},
                    {"<tool_call>"},
                    {"</tool_call>"},
                    {"<arg_key>"},
                    {"</arg_key>"},
                    {"<arg_value>"},
                    {"</arg_value>"},
                    {"<tool_response>"},
                    {"</tool_response>"}};
  Datum(&result, "spark.system_lead", values.at("system_lead"));
  Datum(&result, "loomify.text.default_effort", values.at("default_effort"));
  Datum(&result, "loomify.text.effort_xhigh", values.at("effort_high"));
  Datum(&result, "loomify.text.effort_low", values.at("effort_low"));
  Datum(&result, "loomify.text.default_effort_instruction",
        values.at("default_effort") == "high" ? values.at("effort_high")
                                              : values.at("effort_low"));
  Datum(&result, "loomify.text.tools_prefix",
        "\n\n" + values.at("tools_title") + "\n" + values.at("tools_intro") +
            "\n<tools>");
  Datum(&result, "loomify.text.tools_suffix", "\n</tools>");
  *out = std::move(result);
  return Match::Matched;
}
} // namespace hrx::loomify_gguf::text_policy
