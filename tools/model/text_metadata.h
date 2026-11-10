// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_MODEL_TEXT_METADATA_H_
#define HRX_SOLUTIONS_MODEL_TEXT_METADATA_H_
#include "tools/model/gguf_catalog.h"
#include <map>
#include <optional>
namespace hrx::model {
struct TextMetadataIssue {
  std::string source_key;
  std::string message;
};
struct DeclaredToken {
  std::string source_key;
  uint32_t source_type;
  uint64_t id;
};
struct ChatTemplateSource {
  std::string source_key;
  std::optional<std::string> name;
  bool listed = false;
  std::string text;
};
struct TextMetadata {
  std::optional<GgufStringArraySpan> vocabulary;
  std::optional<GgufStringArraySpan> merges;
  std::vector<TextMetadataIssue> vocabulary_issues;
  std::optional<std::string> tokenizer_model, pre_tokenizer;
  std::optional<bool> add_bos, add_eos;
  std::vector<DeclaredToken> tokens;
  std::map<uint64_t, std::string> token_pieces;
  std::vector<TextMetadataIssue> token_issues;
  std::vector<std::string> named_template_order;
  std::vector<ChatTemplateSource> templates;
  std::vector<TextMetadataIssue> template_issues;
};
// Copies owned facts from this catalog and the same immutable unbuffered file.
// Semantic errors are keyed issues in independent groups; I/O/boundary failure
// returns false and preserves *out. Reads no tensor payload; retains no full
// vocabulary/merges. Success clears error, including when issues are present.
bool ReadTextMetadata(FILE *file, const GgufCatalog &catalog, TextMetadata *out,
                      std::string *error);
} // namespace hrx::model
#endif
