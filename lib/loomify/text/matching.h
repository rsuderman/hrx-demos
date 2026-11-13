// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_MATCHING_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_MATCHING_H_

#include "lib/loomify/chat_policy.h"
#include <map>
#include <string_view>

namespace hrx::loomify_gguf::text_policy {
// Private outcomes distinguish an ordinary mismatch from a missing build asset.
enum class Match { NoMatch, Matched, InvalidBuiltin };
struct Slot {
  std::string name;
  size_t begin, end;
  std::string original;
  bool multiline = false;
};
using Captures = std::map<std::string, std::string>;
std::string_view Fixture(const char *name);
bool AddSlot(std::string_view source, const char *name, std::string_view anchor,
             size_t quote_offset, std::vector<Slot> *slots,
             bool multiline = false);
// Match every byte outside the selected quoted slots, preserving tag
// identities.
bool Capture(std::string_view expected, std::string_view actual,
             std::vector<Slot> slots, Captures *values);
void Datum(ChatPolicySelection *policy, const char *name,
           std::string_view value);
Match SelectLlamaHeaders(std::string_view actual, ChatPolicySelection *out);
Match SelectChatmlThinking(std::string_view actual, ChatPolicySelection *out);
Match SelectGemmaTurns(std::string_view actual, ChatPolicySelection *out);
Match SelectSparkGlm(std::string_view actual, ChatPolicySelection *out);
} // namespace hrx::loomify_gguf::text_policy
#endif
