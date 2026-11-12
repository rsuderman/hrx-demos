// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/token_policy.h"
#include "lib/loomify/architectures.h"
#include "lib/loomify/publication_internal.h"
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
  const auto *architecture = SelectArchitecture(family);
  if (!architecture) return Fail(error, "unsupported publication token provider");
  return architecture->derive_publication(file, architecture->family, text, out, error);
}
bool BindTokenPolicy(FILE* gguf, const TextTokenRequirements& requirements, const model::TextMetadata& text,
    const std::optional<PublicationTokenFacts>& publication,
    const std::vector<std::string>& required_atomic_markers,
    std::string_view tokenizer_json, BoundTokenPolicy* out, std::string* error) {
  if (!out) return Fail(error, "bound token policy output is required");
  if (requirements.tokenizer_model.empty() || requirements.pre_tokenizer.empty() ||
      requirements.terminal_piece.empty() ||
      text.tokenizer_model != std::optional<std::string>(requirements.tokenizer_model) ||
      text.pre_tokenizer != std::optional<std::string>(requirements.pre_tokenizer))
    return Fail(error, "unsupported tokenizer model/pre for selected text provider");
  // Only roles consumed by this finite text provider are admission requirements.
  // PAD/other unused roles remain source facts, not extra tokenizer obligations.
  std::vector<std::string> required_roles = {"tokenizer.ggml.eos_token_id"};
  if (requirements.require_bos) required_roles.push_back("tokenizer.ggml.bos_token_id");
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
    if (internal::ReadTokenIdentity(text, role, &identity, nullptr))
      policy.roles.emplace(role.source_key, std::move(identity));
  }
  for (const auto& key : required_roles) {
    auto role = policy.roles.find(key);
    if (role == policy.roles.end()) return Fail(error, "text provider requires " + key);
    if (!CheckNative(vocab, role->second, error)) return false;
  }
  if (requirements.require_bos && !requirements.bos_piece.empty() &&
      policy.roles.at("tokenizer.ggml.bos_token_id").piece != requirements.bos_piece)
    return Fail(error, "text provider requires BOS " + requirements.bos_piece);
  const auto eos = policy.roles.find("tokenizer.ggml.eos_token_id");
  if (eos == policy.roles.end()) return Fail(error, "text provider requires declared EOS");
  policy.terminal = eos->second;
  if (requirements.terminal_is_eos && policy.terminal.piece != requirements.terminal_piece)
    return Fail(error, "selected text provider requires EOS " + requirements.terminal_piece);
  // Selected atomic markers may not have separately declared role IDs.
  // Recover only requested pieces from this same immutable GGUF, without a
  // retained full vocabulary or trusting external JSON to establish source IDs.
  std::map<std::string, TokenIdentity> markers;
  std::set<std::string> requested(required_atomic_markers.begin(), required_atomic_markers.end());
  // A protocol may finish turns with a source marker distinct from declared EOS.
  if (policy.terminal.piece != requirements.terminal_piece)
    requested.insert(requirements.terminal_piece);
  if (!requested.empty()) {
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
  if (policy.terminal.piece != requirements.terminal_piece)
    policy.terminal = markers.at(requirements.terminal_piece);
  if (publication) {
    if (publication->raw_stops.empty() || publication->terminal_index >= publication->raw_stops.size())
      return Fail(error, "invalid numerical publication terminal index");
    for (const auto& token : publication->raw_stops)
      if (!CheckNative(vocab, token, error)) return false;
    const auto& terminal = publication->raw_stops[publication->terminal_index];
    if (policy.terminal.id != terminal.id || policy.terminal.piece != terminal.piece)
      return Fail(error, "package terminal differs from numerical publication");
  }
  policy.text_source = EmitTerminalMarkerSource(policy.terminal.piece);
  if (requirements.require_bos && requirements.bos_piece.empty()) {
    if (!text.vocabulary || !text.vocabulary->count() ||
        text.vocabulary->count() > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
      return Fail(error, "source-bound text provider requires bounded GGUF vocabulary");
    policy.text_source += "func.def public @loomify.tokens.vocabulary_size() -> (i64) {\n"
        "  %value = scalar.constant " + std::to_string(text.vocabulary->count()) +
        " : i64\n  func.return %value : i64\n}\n";
    const auto& piece = policy.roles.at("tokenizer.ggml.bos_token_id").piece;
    policy.text_source += "global.rodata.def @loomify.tokens.bos = bytes(\"" + Hex(piece) +
        "\")\nfunc.def public @loomify.tokens.bos_marker() -> (buffer) {\n"
        "  %value = global.load @loomify.tokens.bos : buffer\n"
        "  func.return %value : buffer\n}\n";
  }
  *out = std::move(policy);
  if (error) error->clear();
  return true;
}
} // namespace hrx::loomify_gguf
