// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/token_policy.h"
#include "tools/model/text_metadata.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include <limits>
#include <memory>
#include <set>
#include <utility>
namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string* error, std::string message) {
  if (error) *error = std::move(message);
  return false;
}
const model::DeclaredToken* Role(const model::TextMetadata& text,
                                 std::string_view key) {
  for (const auto& token : text.tokens)
    if (token.source_key == key) return &token;
  return nullptr;
}
bool Identity(const model::TextMetadata& text, const model::DeclaredToken& role,
              TokenIdentity* out, std::string* error) {
  auto piece = text.token_pieces.find(role.id);
  if (role.id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      piece == text.token_pieces.end() || piece->second.empty())
    return Fail(error, "invalid token identity for " + role.source_key);
  *out = {static_cast<int32_t>(role.id), piece->second};
  return true;
}
std::string Hex(std::string_view bytes) {
  const char* digits = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    result += digits[byte >> 4]; result += digits[byte & 15];
  }
  return result;
}
bool CheckNative(const iree_tokenizer_vocab_t* vocab, const TokenIdentity& token,
                 std::string* error) {
  auto piece = iree_make_string_view(token.piece.data(), token.piece.size());
  if (iree_tokenizer_vocab_lookup(vocab, piece) != token.id ||
      !iree_string_view_equal(iree_tokenizer_vocab_token_text(vocab, token.id), piece))
    return Fail(error, "tokenizer identity mismatch for " + token.piece +
                       " at ID " + std::to_string(token.id));
  if (!(iree_tokenizer_vocab_token_attrs(vocab, token.id) &
        IREE_TOKENIZER_TOKEN_ATTR_SPECIAL))
    return Fail(error, "tokenizer does not recognize special piece " + token.piece);
  return true;
}
} // namespace
std::string EmitTerminalMarkerSource(std::string_view piece) {
  return "global.rodata.def @loomify.tokens.terminal = bytes(\"" +
      Hex(piece) + "\")\nfunc.def public @loomify.tokens.terminal_marker() -> (buffer) {\n"
      "  %value = global.load @loomify.tokens.terminal : buffer\n"
      "  func.return %value : buffer\n}\n";
}
bool DerivePublicationTokenFacts(FILE* file, std::string_view family,
    const model::TextMetadata& text, std::optional<PublicationTokenFacts>* out,
    std::string* error) {
  if (!out) return Fail(error, "publication token output is required");
  if (family == "qwen35") {
    *out = std::nullopt;
    if (error) error->clear();
    return true;
  }
  if (family != "llama") return Fail(error, "unsupported publication token provider");
  if (!file || !text.vocabulary)
    return Fail(error, "Llama publication requires GGUF vocabulary");
  PublicationTokenFacts facts;
  TokenIdentity* found[] = {&facts.end_text, &facts.end_message, &facts.end_turn};
  const std::string names[] = {"<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"};
  bool present[3] = {};
  if (!model::VisitGgufStringArray(file, *text.vocabulary,
      [&](uint64_t id, std::string_view piece, std::string* issue) {
        for (int i = 0; i != 3; ++i) {
          if (piece != names[i]) continue;
          if (present[i] || id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
            return Fail(issue, "invalid Llama publication piece " + names[i]);
          *found[i] = {static_cast<int32_t>(id), std::string(piece)};
          present[i] = true;
        }
        return true;
      }, error)) return false;
  for (int i = 0; i != 3; ++i)
    if (!present[i]) return Fail(error, "Llama publication requires " + names[i]);
  auto* eos = Role(text, "tokenizer.ggml.eos_token_id");
  if (!eos || eos->id != static_cast<uint64_t>(facts.end_turn.id))
    return Fail(error, "Llama publication requires declared EOS to be <|eot_id|>");
  const char* keys[] = {"end_text_id", "end_message_id", "end_turn_id"};
  for (int i = 0; i != 3; ++i)
    facts.configuration_source += "config.def @llama.publication." +
        std::string(keys[i]) + " = " + std::to_string(found[i]->id) + " : i32\n";
  *out = std::move(facts);
  if (error) error->clear();
  return true;
}
bool BindTokenPolicy(FILE* gguf, std::string_view text_family, const model::TextMetadata& text,
    const std::optional<PublicationTokenFacts>& publication,
    const std::vector<std::string>& required_atomic_markers,
    std::string_view tokenizer_json, BoundTokenPolicy* out, std::string* error) {
  if (!out) return Fail(error, "bound token policy output is required");
  if ((text_family != "llama" && text_family != "qwen35") ||
      text.tokenizer_model != std::optional<std::string>("gpt2") ||
      text.pre_tokenizer != std::optional<std::string>(text_family == "llama" ? "llama-bpe" : "qwen35"))
    return Fail(error, "unsupported tokenizer model/pre for selected text provider");
  // Only roles consumed by this finite text provider are admission requirements.
  // PAD/other unused roles remain source facts, not extra tokenizer obligations.
  std::vector<std::string> required_roles = {"tokenizer.ggml.eos_token_id"};
  if (text_family == "llama") required_roles.push_back("tokenizer.ggml.bos_token_id");
  for (const auto& issue : text.token_issues)
    for (const auto& key : required_roles)
      if (issue.source_key == key)
        return Fail(error, issue.source_key + ": " + issue.message);
  iree_tokenizer_t* raw = nullptr;
  auto status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(tokenizer_json.data(), tokenizer_json.size()),
      iree_allocator_system(), &raw);
  if (!iree_status_is_ok(status)) {
    char message[2048] = {}; iree_host_size_t length = 0;
    if (!iree_status_format(status, sizeof(message), message, &length))
      message[sizeof(message) - 1] = 0;
    iree_status_ignore(status);
    return Fail(error, "native tokenizer JSON: " + std::string(message));
  }
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(raw, iree_tokenizer_free);
  const auto* vocab = iree_tokenizer_vocab(tokenizer.get());
  BoundTokenPolicy policy;
  for (const auto& role : text.tokens) {
    TokenIdentity identity;
    if (Identity(text, role, &identity, nullptr))
      policy.roles.emplace(role.source_key, std::move(identity));
  }
  for (const auto& key : required_roles) {
    auto role = policy.roles.find(key);
    if (role == policy.roles.end()) return Fail(error, "text provider requires " + key);
    if (!CheckNative(vocab, role->second, error)) return false;
  }
  if (text_family == "llama" &&
      policy.roles.at("tokenizer.ggml.bos_token_id").piece != "<|begin_of_text|>")
    return Fail(error, "Llama text provider requires BOS <|begin_of_text|>");
  auto eos = policy.roles.find("tokenizer.ggml.eos_token_id");
  if (eos == policy.roles.end()) return Fail(error, "text provider requires declared EOS");
  policy.terminal = eos->second;
  if (text_family == "llama") {
    if (policy.terminal.piece != "<|eot_id|>")
      return Fail(error, "Llama text provider requires EOS <|eot_id|>");
  } else if (policy.terminal.piece != "<|im_end|>") {
    return Fail(error, "Qwen text provider requires EOS <|im_end|>");
  }
  // Publication facts describe the selected numerical producer. Their native
  // identities and terminal agreement matter regardless of the text strategy;
  // a text-only provider does not itself require a numerical stop mapper.
  if (publication) {
    for (const auto* token : {&publication->end_text, &publication->end_message, &publication->end_turn})
      if (!CheckNative(vocab, *token, error)) return false;
    if (policy.terminal.id != publication->end_turn.id)
      return Fail(error, "package terminal differs from numerical publication");
  }
  // Selected atomic markers may not have separately declared role IDs.
  // Recover only requested pieces from this same immutable GGUF, without a
  // retained full vocabulary or trusting external JSON to establish source IDs.
  std::map<std::string, TokenIdentity> markers;
  const std::set<std::string> requested(required_atomic_markers.begin(),
                                        required_atomic_markers.end());
  if (!required_atomic_markers.empty()) {
    if (!gguf || !text.vocabulary)
      return Fail(error, "required atomic markers need GGUF vocabulary");
    if (!model::VisitGgufStringArray(gguf, *text.vocabulary,
        [&](uint64_t id, std::string_view piece, std::string* issue) {
          for (const auto& marker : requested) {
            if (marker != piece) continue;
            if (id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
                markers.count(marker))
              return Fail(issue, "ambiguous GGUF atomic marker " + marker);
            markers.emplace(marker, TokenIdentity{static_cast<int32_t>(id), marker});
          }
          return true;
        }, error)) return false;
    for (const auto& marker : requested) {
      auto found = markers.find(marker);
      if (found == markers.end())
        return Fail(error, "GGUF missing required atomic marker: " + marker);
      if (!CheckNative(vocab, found->second, error)) return false;
    }
  }
  policy.text_source = EmitTerminalMarkerSource(policy.terminal.piece);
  *out = std::move(policy);
  if (error) error->clear();
  return true;
}
} // namespace hrx::loomify_gguf
