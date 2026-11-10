// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/text_metadata.h"
#include <algorithm>
#include <set>

namespace hrx::model {
namespace {
void Issue(std::vector<TextMetadataIssue> &issues, const std::string &key,
           const std::string &message) {
  issues.push_back({key, message});
}
const std::string *String(const MetadataRecord &field) {
  return field.source_type == 8 ? std::get_if<std::string>(&field.scalar)
                                : nullptr;
}
std::optional<uint64_t> TokenId(const MetadataRecord &field) {
  switch (field.source_type) {
  case 0:
  case 2:
  case 4:
  case 10:
    if (const auto *v = std::get_if<uint64_t>(&field.scalar))
      return *v;
    break;
  case 1:
  case 3:
  case 5:
  case 11:
    if (const auto *v = std::get_if<int64_t>(&field.scalar))
      if (*v >= 0)
        return uint64_t(*v);
    break;
  }
  return std::nullopt;
}
bool RoleKey(const std::string &key) {
  const std::string prefix = "tokenizer.ggml.", suffix = "_token_id";
  return key.size() > prefix.size() + suffix.size() &&
         key.compare(0, prefix.size(), prefix) == 0 &&
         key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0;
}
} // namespace
bool ReadTextMetadata(FILE *file, const GgufCatalog &catalog, TextMetadata *out,
                      std::string *error) {
  std::string ignored;
  if (!error)
    error = &ignored;
  if (!file || !out) {
    *error = "text metadata requires file and output";
    return false;
  }
  try {
    TextMetadata result;
    auto array = [&](const char *key,
                     std::optional<GgufStringArraySpan> &target) {
      if (const auto *field = FindMetadata(catalog, key)) {
        if (field->source_type == 9 && field->element_type == 8 &&
            field->string_array)
          target = field->string_array;
        else
          Issue(result.vocabulary_issues, key, "expected a GGUF string array");
      }
    };
    array("tokenizer.ggml.tokens", result.vocabulary);
    array("tokenizer.ggml.merges", result.merges);
    auto string = [&](const char *key, std::optional<std::string> &target) {
      if (const auto *field = FindMetadata(catalog, key)) {
        if (const auto *value = String(*field))
          target = *value;
        else
          Issue(result.token_issues, key, "expected a GGUF string");
      }
    };
    string("tokenizer.ggml.model", result.tokenizer_model);
    string("tokenizer.ggml.pre", result.pre_tokenizer);
    auto flag = [&](const char *key, std::optional<bool> &target) {
      if (const auto *field = FindMetadata(catalog, key)) {
        const auto *value = std::get_if<bool>(&field->scalar);
        if (field->source_type == 7 && value)
          target = *value;
        else
          Issue(result.token_issues, key, "expected a GGUF boolean");
      }
    };
    flag("tokenizer.ggml.add_bos_token", result.add_bos);
    flag("tokenizer.ggml.add_eos_token", result.add_eos);
    for (const auto &field : catalog.metadata)
      if (RoleKey(field.key)) {
        const auto id = TokenId(field);
        if (!id) {
          Issue(result.token_issues, field.key,
                "expected a nonnegative GGUF integer token ID");
          continue;
        }
        if (!result.vocabulary || *id >= result.vocabulary->count()) {
          Issue(result.token_issues, field.key,
                "token ID requires a vocabulary and must be smaller than its "
                "count");
          continue;
        }
        result.tokens.push_back({field.key, field.source_type, *id});
        result.token_pieces.emplace(*id, std::string{});
      }
    if (!result.token_pieces.empty()) {
      std::vector<uint64_t> ids;
      for (const auto &piece : result.token_pieces)
        ids.push_back(piece.first);
      std::vector<std::string> pieces;
      if (!ReadGgufTokenPieces(file, *result.vocabulary, ids, &pieces, error))
        return false;
      for (size_t i = 0; i < ids.size(); ++i)
        result.token_pieces[ids[i]] = std::move(pieces[i]);
    }
    constexpr const char *index_key = "tokenizer.chat_templates";
    std::set<std::string> indexed;
    if (const auto *index = FindMetadata(catalog, index_key)) {
      if (index->source_type != 9 || index->element_type != 8 ||
          !index->string_array)
        Issue(result.template_issues, index_key,
              "expected a GGUF string array of names");
      else if (index->string_array->count() > 4096)
        Issue(result.template_issues, index_key,
              "template names exceed bounded metadata count");
      else {
        if (!VisitGgufStringArray(
                file, *index->string_array,
                [&](uint64_t, std::string_view name, std::string *) {
                  result.named_template_order.emplace_back(name);
                  if (name.empty())
                    Issue(result.template_issues, index_key,
                          "template name must not be empty");
                  else if (!indexed.emplace(name).second)
                    Issue(result.template_issues, index_key,
                          "duplicate template name: " + std::string(name));
                  return true;
                },
                error))
          return false;
      }
    }
    const std::string default_key = "tokenizer.chat_template",
                      prefix = default_key + ".";
    std::set<std::string> defined;
    for (const auto &field : catalog.metadata) {
      const bool scalar = field.key == default_key;
      if (!scalar && field.key.compare(0, prefix.size(), prefix) != 0)
        continue;
      const auto *text = String(field);
      if (!text) {
        Issue(result.template_issues, field.key,
              "expected a GGUF template string");
        continue;
      }
      if (scalar)
        result.templates.push_back({field.key, std::nullopt, false, *text});
      else {
        const std::string name = field.key.substr(prefix.size());
        const bool listed = indexed.count(name) != 0;
        result.templates.push_back({field.key, name, listed, *text});
        defined.insert(name);
        if (name.empty())
          Issue(result.template_issues, field.key,
                "template name must not be empty");
        if (!listed)
          Issue(result.template_issues, field.key,
                "named template is absent from tokenizer.chat_templates");
      }
    }
    for (const auto &name : indexed)
      if (!defined.count(name))
        Issue(result.template_issues, prefix + name,
              "indexed template has no valid scalar source");
    *out = std::move(result);
    error->clear();
    return true;
  } catch (const std::exception &ex) {
    *error =
        std::string("unable to read owned GGUF text metadata: ") + ex.what();
    return false;
  }
}
} // namespace hrx::model
