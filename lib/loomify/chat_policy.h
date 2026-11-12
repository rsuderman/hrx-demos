// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_CHAT_POLICY_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_CHAT_POLICY_H_
#include "tools/model/text_metadata.h"
#include "lib/loomify/token_policy.h"
#include <optional>
#include <string>
#include <vector>
namespace hrx::loomify_gguf {
// Private selection data, never a numerical model or dispatch graph.
enum class ChatRevision { LlamaHeadersBaseline, ChatmlInlineHistory, ChatmlExplicitReasoning, ChatmlHermes, ChatmlHermesNonThinking, GemmaTurns };
struct ChatMarker {
  std::string text;
  // Framing literals such as XML tags may encode as several ordinary tokens.
  bool require_single_token = false;
};
struct ChatPolicySelection {
  ChatRevision revision = ChatRevision::LlamaHeadersBaseline;
  std::string source_key;
  std::optional<std::string> template_name;
  // Identity of the maintained semantic skeleton, not mutated source bytes.
  std::string skeleton_sha256;
  std::string source_text;
  // Public definitions linked with the selected maintained text provider.
  std::string configuration_source;
  std::vector<const char*> data_roots;
  TextTokenRequirements tokens;
  std::string text_product;
  bool preserved_baseline = false;
  std::vector<ChatMarker> markers;
  std::vector<const char*> roots; // static literal names with process lifetime
};
// Known semantic skeletons admit only named literal/default slots.
// Scalar default wins without a name;
// otherwise exactly one listed candidate is unambiguous. Explicit names never
// select scalar defaults. Any failure preserves *out; success clears error.
bool SelectChatPolicy(const model::TextMetadata&,
                      const std::optional<std::string>& template_name,
                      ChatPolicySelection* out, std::string* error);
} // namespace hrx::loomify_gguf
#endif
