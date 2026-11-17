// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TOKEN_POLICY_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TOKEN_POLICY_H_
#include "lib/loomify/publication.h"
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
namespace hrx::model {
struct TextMetadata;
}
namespace hrx::loomify_gguf {
struct BoundTokenPolicy {
  std::map<std::string, TokenIdentity> roles;
  TokenIdentity terminal;
  uint32_t encode_flags = 0; // Provider-rendered bytes already own insertion.
  std::string text_source;
};
// Extracted source facts, independent of the selected response format.
struct NativeTokenFacts {
  std::optional<std::string> tokenizer_model, pre_tokenizer;
};
// Internal package preparation. This owns one native tokenizer through analysis
// and finalization; no partially bound package is published.
class PreparedTokenBindings {
public:
  PreparedTokenBindings();
  ~PreparedTokenBindings();
  PreparedTokenBindings(PreparedTokenBindings &&) noexcept;
  PreparedTokenBindings &operator=(PreparedTokenBindings &&) noexcept;
  const std::map<std::string, std::string> &globals() const;
  const std::string &globals_json() const;
  // Includes consumed optional globals whose GGUF roles are absent.
  const std::vector<std::string> &global_names() const;
  const NativeTokenFacts &facts() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend bool PrepareTokenBindings(const std::vector<std::string> &,
                                   const model::TextMetadata &,
                                   std::string_view, PreparedTokenBindings *,
                                   std::string *);
  friend bool FinalizeTokenPolicy(FILE *, const PreparedTokenBindings &,
                                  const model::TextMetadata &,
                                  const std::optional<PublicationTokenFacts> &,
                                  const std::vector<std::string> &,
                                  const std::vector<std::string> &,
                                  BoundTokenPolicy *, std::string *);
};
// external_names comes from the existing compiler's lexical binding analysis.
// Referenced missing roles stay absent, retaining source guards and defaults.
// Native ID/piece/special mismatches for available consumed roles fail before
// analysis. Model/pre are retained as facts, without profile admission.
bool PrepareTokenBindings(const std::vector<std::string> &external_names,
                          const model::TextMetadata &,
                          std::string_view tokenizer_json,
                          PreparedTokenBindings *out, std::string *error);
// Resolves the longest GGUF/native special prefix of each analyzed assistant
// frame, permitting only trailing literal whitespace. Alternatives must agree
// on the terminal identity. Ordinary text delimiters add no atomic obligations;
// only actual native consumers contribute required_atomic_markers.
bool FinalizeTokenPolicy(
    FILE *gguf, const PreparedTokenBindings &, const model::TextMetadata &,
    const std::optional<PublicationTokenFacts> &,
    const std::vector<std::string> &frame_suffixes,
    const std::vector<std::string> &required_atomic_markers,
    BoundTokenPolicy *out, std::string *error);
// Data-only bytes emitter shared by package binding and focused native
// fixtures.
std::string EmitTerminalMarkerSource(std::string_view piece);
} // namespace hrx::loomify_gguf
#endif
