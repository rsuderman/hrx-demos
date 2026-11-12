// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/publication_internal.h"
#include <limits>
#include <utility>
namespace hrx::loomify_gguf::internal {
namespace {
bool Fail(std::string *error, std::string message) {
  if (error)
    *error = std::move(message);
  return false;
}
// Source identities and terminal semantics, without numerical config names.
struct SourceProfile {
  std::vector<std::string> pieces;
  size_t terminal_index;
  size_t eos_index;
};
bool SelectSourceProfile(PublicationLayout layout,
                         const model::TextMetadata &text,
                         SourceProfile *profile, std::string *error) {
  const auto *eos = DeclaredRole(text, "tokenizer.ggml.eos_token_id");
  TokenIdentity identity;
  if (!eos || !ReadTokenIdentity(text, *eos, &identity, error))
    return Fail(error, "publication requires declared EOS identity");
  switch (layout) {
  case PublicationLayout::kThreeStops:
    if (identity.piece == "<|eot_id|>") {
      *profile = {{"<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"}, 2, 2};
      return true;
    }
    if (identity.piece == "<|endofturn|>") {
      *profile = {{"<|endoftext|>", "<|im_end|>", "<|endofturn|>"}, 1, 2};
      return true;
    }
    break;
  case PublicationLayout::kTextAndTurn:
    if (identity.piece == "<|im_end|>") {
      *profile = {{"<|endoftext|>", "<|im_end|>"}, 1, 1};
      return true;
    }
    break;
  case PublicationLayout::kEosAndTurn:
    *profile = {{identity.piece, "<end_of_turn>"}, 1, 0};
    return true;
  default:
    return Fail(error, "unsupported numerical publication layout");
  }
  return Fail(
      error,
      "publication disagrees with declared EOS identity for selected layout");
}
bool ResolveSourceProfile(FILE *file, const model::TextMetadata &text,
                          const SourceProfile &profile,
                          PublicationTokenFacts *facts, std::string *error) {
  facts->terminal_index = profile.terminal_index;
  facts->raw_stops.resize(profile.pieces.size());
  std::vector<bool> present(profile.pieces.size());
  if (!model::VisitGgufStringArray(
          file, *text.vocabulary,
          [&](uint64_t id, std::string_view piece, std::string *issue) {
            for (size_t i = 0; i < profile.pieces.size(); ++i) {
              if (piece != profile.pieces[i])
                continue;
              if (present[i] || id > static_cast<uint64_t>(
                                         std::numeric_limits<int32_t>::max()))
                return Fail(issue,
                            "invalid publication piece " + profile.pieces[i]);
              facts->raw_stops[i] = {static_cast<int32_t>(id),
                                     std::string(piece)};
              present[i] = true;
            }
            return true;
          },
          error))
    return false;
  for (size_t i = 0; i < profile.pieces.size(); ++i)
    if (!present[i])
      return Fail(error, "publication requires " + profile.pieces[i]);
  const auto *eos = DeclaredRole(text, "tokenizer.ggml.eos_token_id");
  if (!eos ||
      eos->id != static_cast<uint64_t>(facts->raw_stops[profile.eos_index].id))
    return Fail(error, "publication disagrees with declared EOS identity");
  return true;
}
bool EmitSlotConfiguration(PublicationLayout layout, std::string_view prefix,
                           PublicationTokenFacts *facts, std::string *error) {
  std::vector<std::string_view> keys;
  switch (layout) {
  case PublicationLayout::kThreeStops:
    keys = {"end_text_id", "end_message_id", "end_turn_id"};
    break;
  case PublicationLayout::kTextAndTurn:
    keys = {"end_text_id", "end_turn_id"};
    break;
  case PublicationLayout::kEosAndTurn:
    keys = {"eos_id", "end_turn_id"};
    break;
  default:
    return Fail(error, "unsupported numerical publication layout");
  }
  if (facts->raw_stops.size() != keys.size() ||
      facts->terminal_index >= facts->raw_stops.size())
    return Fail(error,
                "source publication profile does not fit numerical slots");
  for (size_t i = 0; i < keys.size(); ++i)
    facts->configuration_source +=
        "config.def @" + std::string(prefix) + ".publication." +
        std::string(keys[i]) + " = " + std::to_string(facts->raw_stops[i].id) +
        " : i32\n";
  return true;
}
} // namespace
const model::DeclaredToken *DeclaredRole(const model::TextMetadata &text,
                                         std::string_view key) {
  for (const auto &token : text.tokens)
    if (token.source_key == key)
      return &token;
  return nullptr;
}
bool ReadTokenIdentity(const model::TextMetadata &text,
                       const model::DeclaredToken &role, TokenIdentity *out,
                       std::string *error) {
  auto piece = text.token_pieces.find(role.id);
  if (role.id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      piece == text.token_pieces.end() || piece->second.empty())
    return Fail(error, "invalid token identity for " + role.source_key);
  *out = {static_cast<int32_t>(role.id), piece->second};
  return true;
}
bool DeriveSourcePublication(FILE *file, PublicationLayout layout,
                             std::string_view config_prefix,
                             const model::TextMetadata &text,
                             std::optional<PublicationTokenFacts> *out,
                             std::string *error) {
  if (!out)
    return Fail(error, "publication token output is required");
  switch (layout) {
  case PublicationLayout::kHostManaged:
    *out = std::nullopt;
    if (error)
      error->clear();
    return true;
  case PublicationLayout::kThreeStops:
  case PublicationLayout::kTextAndTurn:
  case PublicationLayout::kEosAndTurn:
    break;
  default:
    return Fail(error, "unsupported numerical publication layout");
  }
  if (!file || !text.vocabulary)
    return Fail(error, "publication requires GGUF vocabulary");
  SourceProfile profile;
  PublicationTokenFacts facts;
  if (!SelectSourceProfile(layout, text, &profile, error) ||
      !ResolveSourceProfile(file, text, profile, &facts, error) ||
      !EmitSlotConfiguration(layout, config_prefix, &facts, error))
    return false;
  *out = std::move(facts);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::internal
