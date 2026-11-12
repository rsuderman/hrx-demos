// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/matching.h"
#include "lib/serving/library_resources.h"

namespace hrx::loomify_gguf::text_policy {
Match SelectGemmaTurns(std::string_view actual, ChatPolicySelection *out) {
  const auto expected = Fixture("7de1c58e.jinja");
  if (expected.empty())
    return Match::InvalidBuiltin;
  if (actual != expected)
    return Match::NoMatch;
  ChatPolicySelection result;
  result.revision = ChatRevision::GemmaTurns;
  result.skeleton_sha256 =
      "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4";
  result.tokens = {"llama", "default", true, "", "<end_of_turn>", false};
  result.text_product = resources::kGemmaTurns;
  result.markers = {{"<start_of_turn>", true}, {"<end_of_turn>", true}};
  *out = std::move(result);
  return Match::Matched;
}
} // namespace hrx::loomify_gguf::text_policy
