// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/publication_internal.h"
#include <limits>
#include <utility>
namespace hrx::loomify_gguf::internal {
namespace {
bool Fail(std::string *error, std::string message) {
  if (error) *error = std::move(message);
  return false;
}
} // namespace
const model::DeclaredToken* DeclaredRole(const model::TextMetadata& text,
                                 std::string_view key) {
  for (const auto& token : text.tokens)
    if (token.source_key == key) return &token;
  return nullptr;
}
bool ReadTokenIdentity(const model::TextMetadata& text, const model::DeclaredToken& role,
              TokenIdentity* out, std::string* error) {
  auto piece = text.token_pieces.find(role.id);
  if (role.id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      piece == text.token_pieces.end() || piece->second.empty())
    return Fail(error, "invalid token identity for " + role.source_key);
  *out = {static_cast<int32_t>(role.id), piece->second};
  return true;
}
bool DeriveNamedPublication(FILE *file, std::string_view config_prefix,
    const model::TextMetadata &text, const std::vector<std::string> &names,
    const std::vector<std::string> &keys, size_t terminal_index,
    size_t eos_index, std::optional<PublicationTokenFacts> *out,
    std::string *error) {
  if (!out) return Fail(error, "publication token output is required");
  if (!file || !text.vocabulary) return Fail(error, "publication requires GGUF vocabulary");
  PublicationTokenFacts facts;
  facts.terminal_index = terminal_index;
  facts.raw_stops.resize(names.size());
  std::vector<bool> present(names.size());
  if (!model::VisitGgufStringArray(file, *text.vocabulary,
      [&](uint64_t id, std::string_view piece, std::string* issue) {
        for (size_t i = 0; i < names.size(); ++i) {
          if (piece != names[i]) continue;
          if (present[i] || id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
            return Fail(issue, "invalid publication piece " + names[i]);
          facts.raw_stops[i] = {static_cast<int32_t>(id), std::string(piece)};
          present[i] = true;
        }
        return true;
      }, error)) return false;
  for (size_t i = 0; i < names.size(); ++i)
    if (!present[i]) return Fail(error, "publication requires " + names[i]);
  const auto* eos = DeclaredRole(text, "tokenizer.ggml.eos_token_id");
  if (!eos || eos->id != static_cast<uint64_t>(facts.raw_stops[eos_index].id))
    return Fail(error, "publication disagrees with declared EOS identity");
  const std::string prefix(config_prefix);
  for (size_t i = 0; i < keys.size(); ++i)
    facts.configuration_source += "config.def @" + prefix + ".publication." + keys[i] +
        " = " + std::to_string(facts.raw_stops[i].id) + " : i32\n";
  *out = std::move(facts);
  if (error) error->clear();
  return true;
}
bool DeriveChatmlPublication(FILE *file, std::string_view family,
    const model::TextMetadata &text, std::optional<PublicationTokenFacts> *out,
    std::string *error) {
  return DeriveNamedPublication(file, family, text,
      {"<|endoftext|>", "<|im_end|>"}, {"end_text_id", "end_turn_id"},
      1, 1, out, error);
}
} // namespace hrx::loomify_gguf::internal
