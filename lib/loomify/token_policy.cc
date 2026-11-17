// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/token_policy.h"
#include "lib/loomify/publication_internal.h"
#include "tools/model/text_metadata.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <utility>
namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string *error, std::string message) {
  if (error)
    *error = std::move(message);
  return false;
}
std::string Hex(std::string_view bytes) {
  const char *digits = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}
bool CheckNative(const iree_tokenizer_vocab_t *vocab,
                 const TokenIdentity &token, std::string *error) {
  auto piece = iree_make_string_view(token.piece.data(), token.piece.size());
  if (iree_tokenizer_vocab_lookup(vocab, piece) != token.id ||
      !iree_string_view_equal(iree_tokenizer_vocab_token_text(vocab, token.id),
                              piece))
    return Fail(error, "tokenizer identity mismatch for " + token.piece +
                           " at ID " + std::to_string(token.id));
  if (!(iree_tokenizer_vocab_token_attrs(vocab, token.id) &
        IREE_TOKENIZER_TOKEN_ATTR_SPECIAL))
    return Fail(error,
                "tokenizer does not recognize special piece " + token.piece);
  return true;
}
} // namespace
struct PreparedTokenBindings::Impl {
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer{
      nullptr, iree_tokenizer_free};
  NativeTokenFacts facts;
  std::map<std::string, TokenIdentity> roles;
  std::map<std::string, std::string> globals;
  std::string globals_json = "{}";
  std::vector<std::string> global_names;
};
PreparedTokenBindings::PreparedTokenBindings()
    : impl_(std::make_unique<Impl>()) {}
PreparedTokenBindings::~PreparedTokenBindings() = default;
PreparedTokenBindings::PreparedTokenBindings(
    PreparedTokenBindings &&) noexcept = default;
PreparedTokenBindings &
PreparedTokenBindings::operator=(PreparedTokenBindings &&) noexcept = default;
const std::map<std::string, std::string> &
PreparedTokenBindings::globals() const {
  return impl_->globals;
}
const std::string &PreparedTokenBindings::globals_json() const {
  return impl_->globals_json;
}
const std::vector<std::string> &PreparedTokenBindings::global_names() const {
  return impl_->global_names;
}
const NativeTokenFacts &PreparedTokenBindings::facts() const {
  return impl_->facts;
}
namespace {
std::string JsonString(std::string_view value) {
  const char *hex = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += c;
    } else if (c < 32) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += c;
    }
  }
  return result + '"';
}
std::string GlobalRole(std::string_view name) {
  if (name == "bos_token")
    return "tokenizer.ggml.bos_token_id";
  if (name == "eos_token")
    return "tokenizer.ggml.eos_token_id";
  if (name == "pad_token")
    return "tokenizer.ggml.padding_token_id";
  if (name == "unk_token")
    return "tokenizer.ggml.unknown_token_id";
  if (name == "sep_token")
    return "tokenizer.ggml.separator_token_id";
  if (name == "cls_token")
    return "tokenizer.ggml.cls_token_id";
  if (name == "mask_token")
    return "tokenizer.ggml.mask_token_id";
  return {};
}
bool OnlyFrameWhitespace(std::string_view bytes) {
  return bytes.find_first_not_of(" \t\r\n\f\v") == std::string_view::npos;
}
} // namespace
bool PrepareTokenBindings(const std::vector<std::string> &external_names,
                          const model::TextMetadata &text,
                          std::string_view tokenizer_json,
                          PreparedTokenBindings *out, std::string *error) {
  if (!out)
    return Fail(error, "prepared token bindings output is required");
  PreparedTokenBindings prepared;
  auto &result = *prepared.impl_;
  iree_tokenizer_t *raw = nullptr;
  auto status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(tokenizer_json.data(), tokenizer_json.size()),
      iree_allocator_system(), &raw);
  if (!iree_status_is_ok(status)) {
    char message[2048] = {};
    iree_host_size_t length = 0;
    if (!iree_status_format(status, sizeof(message), message, &length))
      message[sizeof(message) - 1] = 0;
    iree_status_ignore(status);
    return Fail(error, "native tokenizer JSON: " + std::string(message));
  }
  result.tokenizer.reset(raw);
  result.facts = {text.tokenizer_model, text.pre_tokenizer};
  for (const auto &role : text.tokens) {
    TokenIdentity identity;
    if (internal::ReadTokenIdentity(text, role, &identity, nullptr))
      result.roles.emplace(role.source_key, std::move(identity));
  }
  const auto *vocab = iree_tokenizer_vocab(result.tokenizer.get());
  for (const auto &name : external_names) {
    const auto key = GlobalRole(name);
    if (key.empty())
      continue;
    if (std::find(result.global_names.begin(), result.global_names.end(),
                  name) == result.global_names.end())
      result.global_names.push_back(name);
    for (const auto &issue : text.token_issues)
      if (issue.source_key == key)
        return Fail(error, key + ": " + issue.message);
    const auto found = result.roles.find(key);
    if (found == result.roles.end())
      continue;
    if (!CheckNative(vocab, found->second, error))
      return false;
    result.globals.emplace(name, found->second.piece);
  }
  result.globals_json = "{";
  for (const auto &[name, value] : result.globals) {
    if (result.globals_json.size() > 1)
      result.globals_json += ',';
    result.globals_json += JsonString(name) + ':' + JsonString(value);
  }
  result.globals_json += '}';
  *out = std::move(prepared);
  if (error)
    error->clear();
  return true;
}
bool FinalizeTokenPolicy(
    FILE *gguf, const PreparedTokenBindings &prepared,
    const model::TextMetadata &text,
    const std::optional<PublicationTokenFacts> &publication,
    const std::vector<std::string> &frame_suffixes,
    const std::vector<std::string> &required_atomic_markers,
    BoundTokenPolicy *out, std::string *error) {
  if (!out)
    return Fail(error, "bound token policy output is required");
  if (!prepared.impl_ || !prepared.impl_->tokenizer)
    return Fail(error,
                "native token bindings must be prepared before finalization");
  if (frame_suffixes.empty())
    return Fail(error, "analyzed assistant frame suffix is required");
  if (!gguf || !text.vocabulary)
    return Fail(error, "assistant terminal needs GGUF vocabulary");
  if (!text.vocabulary->count() ||
      text.vocabulary->count() >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
    return Fail(error, "generated text requires bounded GGUF vocabulary");
  const auto *vocab = iree_tokenizer_vocab(prepared.impl_->tokenizer.get());
  std::vector<std::optional<TokenIdentity>> terminals(frame_suffixes.size());
  std::map<std::string, TokenIdentity> markers;
  const std::set<std::string> requested(required_atomic_markers.begin(),
                                        required_atomic_markers.end());
  if (!model::VisitGgufStringArray(
          gguf, *text.vocabulary,
          [&](uint64_t id, std::string_view piece, std::string *issue) {
            if (piece.empty())
              return true;
            bool terminal_candidate = false;
            for (const auto &suffix : frame_suffixes)
              terminal_candidate |=
                  suffix.compare(0, piece.size(), piece) == 0 &&
                  OnlyFrameWhitespace(
                      std::string_view(suffix).substr(piece.size()));
            if (!terminal_candidate && !requested.count(std::string(piece)))
              return true;
            if (id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
              return Fail(issue, "GGUF marker ID exceeds native token range");
            TokenIdentity identity{static_cast<int32_t>(id),
                                   std::string(piece)};
            if (requested.count(identity.piece)) {
              if (!markers.emplace(identity.piece, identity).second)
                return Fail(issue,
                            "ambiguous GGUF atomic marker " + identity.piece);
              if (!CheckNative(vocab, identity, issue))
                return false;
            }
            if (!terminal_candidate)
              return true;
            // Ordinary delimiter tokens are not promoted to native stop tokens.
            const auto native_id = iree_tokenizer_vocab_lookup(
                vocab, iree_make_string_view(piece.data(), piece.size()));
            if (native_id < 0 ||
                !(iree_tokenizer_vocab_token_attrs(vocab, native_id) &
                  IREE_TOKENIZER_TOKEN_ATTR_SPECIAL))
              return true;
            if (!CheckNative(vocab, identity, issue))
              return false;
            for (size_t i = 0; i < frame_suffixes.size(); ++i) {
              const auto &suffix = frame_suffixes[i];
              if (suffix.compare(0, piece.size(), piece) != 0 ||
                  !OnlyFrameWhitespace(
                      std::string_view(suffix).substr(piece.size())))
                continue;
              auto &selected = terminals[i];
              if (selected && selected->piece.size() == piece.size())
                return Fail(issue, "ambiguous GGUF assistant terminal " +
                                       identity.piece);
              if (!selected || selected->piece.size() < piece.size())
                selected = identity;
            }
            return true;
          },
          error))
    return false;
  for (const auto &marker : requested)
    if (!markers.count(marker))
      return Fail(error, "GGUF missing required atomic marker: " + marker);
  BoundTokenPolicy policy;
  policy.roles = prepared.impl_->roles;
  for (const auto &terminal : terminals) {
    if (!terminal)
      return Fail(error,
                  "analyzed assistant frame has no native special terminal");
    if (policy.terminal.piece.empty())
      policy.terminal = *terminal;
    else if (policy.terminal.id != terminal->id ||
             policy.terminal.piece != terminal->piece)
      return Fail(error,
                  "analyzed assistant alternatives have different terminals");
  }
  if (publication) {
    if (publication->raw_stops.empty() ||
        publication->terminal_index >= publication->raw_stops.size())
      return Fail(error, "invalid numerical publication terminal index");
    for (const auto &token : publication->raw_stops)
      if (!CheckNative(vocab, token, error))
        return false;
    const auto &terminal = publication->raw_stops[publication->terminal_index];
    if (policy.terminal.id != terminal.id ||
        policy.terminal.piece != terminal.piece)
      return Fail(error, "package terminal differs from numerical publication");
  }
  policy.text_source = EmitTerminalMarkerSource(policy.terminal.piece);
  policy.text_source +=
      "func.def public @loomify.tokens.vocabulary_size() -> (i64) {\n"
      "  %value = scalar.constant " +
      std::to_string(text.vocabulary->count()) +
      " : i64\n  func.return %value : i64\n}\n";
  *out = std::move(policy);
  if (error)
    error->clear();
  return true;
}
std::string EmitTerminalMarkerSource(std::string_view piece) {
  return "global.rodata.def @loomify.tokens.terminal = bytes(\"" + Hex(piece) +
         "\")\nfunc.def public @loomify.tokens.terminal_marker() -> (buffer) {\n"
         "  %value = global.load @loomify.tokens.terminal : buffer\n"
         "  func.return %value : buffer\n}\n";
}
} // namespace hrx::loomify_gguf
