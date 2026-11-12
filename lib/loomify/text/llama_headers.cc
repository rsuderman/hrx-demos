// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/matching.h"
#include "lib/serving/library_resources.h"

namespace hrx::loomify_gguf::text_policy {
Match SelectLlamaHeaders(std::string_view actual, ChatPolicySelection *out) {
  const auto expected = Fixture("5816fce1.jinja");
  if (expected.empty())
    return Match::InvalidBuiltin;
  const std::string anchor = "{%- else %}\n        {%- set date_string = ";
  std::vector<Slot> slots;
  Captures values;
  if (!AddSlot(expected, "date", anchor, anchor.size(), &slots) ||
      !Capture(expected, actual, std::move(slots), &values))
    return Match::NoMatch;
  ChatPolicySelection result;
  result.revision = ChatRevision::LlamaHeadersBaseline;
  result.skeleton_sha256 =
      "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4";
  result.tokens = {"gpt2", "llama-bpe", true, "<|begin_of_text|>",
                   "<|eot_id|>"};
  result.text_product = resources::kLlamaText;
  result.preserved_baseline = true;
  result.markers = {{"<|begin_of_text|>", true},
                    {"<|start_header_id|>", true},
                    {"<|end_header_id|>", true},
                    {"<|eot_id|>", true}};
  Datum(&result, "loomify.text.fallback_date", values.at("date"));
  *out = std::move(result);
  return Match::Matched;
}
} // namespace hrx::loomify_gguf::text_policy
