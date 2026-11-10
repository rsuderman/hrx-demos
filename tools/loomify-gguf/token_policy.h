// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_TOKEN_POLICY_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_TOKEN_POLICY_H_
#include <cstdio>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
namespace hrx::model { struct TextMetadata; }
namespace hrx::loomify_gguf {
struct TokenIdentity {
  int32_t id = 0;
  std::string piece;
};
// Only Llama's numerical publication consumes these identities. Qwen's
// numerical product has no tokenizer/EOS prerequisite.
struct PublicationTokenFacts {
  TokenIdentity end_text, end_message, end_turn;
  std::string configuration_source;
};
struct BoundTokenPolicy {
  std::map<std::string, TokenIdentity> roles;
  TokenIdentity terminal;
  uint32_t encode_flags = 0; // Provider-rendered bytes already own insertion.
  std::string text_source;
};
// Data-only bytes emitter shared by package binding and focused native fixtures.
std::string EmitTerminalMarkerSource(std::string_view piece);
// Called once during Open with the same immutable catalog FILE. An unsupported
// template or external tokenizer cannot affect this source-only operation.
// Failure preserves output; Qwen succeeds with an empty optional result.
bool DerivePublicationTokenFacts(FILE*, std::string_view family,
    const model::TextMetadata&, std::optional<PublicationTokenFacts>* out,
    std::string* error);
// Native HF JSON binding for a reviewed text provider, not exhaustive tokenizer
// equivalence. text_family selects token/text requirements, not the numerical ABI.
// Optional publication facts are numerical requirements checked independently.
// JSON bytes and metadata are borrowed; result owns its data.
// Required atomic pieces must exist in both GGUF and JSON at the same ID.
bool BindTokenPolicy(FILE* gguf, std::string_view text_family, const model::TextMetadata&,
    const std::optional<PublicationTokenFacts>&,
    const std::vector<std::string>& required_atomic_markers,
    std::string_view tokenizer_json, BoundTokenPolicy* out, std::string* error);
} // namespace hrx::loomify_gguf
#endif
