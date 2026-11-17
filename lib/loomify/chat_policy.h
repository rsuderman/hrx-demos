// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_CHAT_POLICY_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_CHAT_POLICY_H_
#include "tools/model/text_metadata.h"
#include "lib/loomify/token_policy.h"
#include <optional>
#include <string>
#include <string_view>
#include <vector>
namespace hrx::loomify_gguf {
namespace jinja {
struct TemplateAnalysisOptions;
}
struct ChatTemplateSelection {
  std::string source_key;
  std::optional<std::string> template_name;
  std::string source_text;
};

// Owned generated text products. No tokenizer, prepared VM, AST or borrowed
// source survives preparation. source includes token_source; preparation-only
// LINK consumes the latter without pulling in the text implementation.
// roots lists the hooks supplied by the generated host/output/input/discovery
// products; package construction adds its numerical and identity hooks
// separately.
struct PreparedTextControl {
  std::string source;
  std::string token_source;
  std::vector<std::string> helper_requirements;
  std::vector<std::string> roots;
};

// Compiles the selected source once, binds its actual GGUF globals, validates
// its native source-derived formats, then emits request/output/input control.
// The caller supplies BUILD-owned helper/scratch roots through
// analysis_options. GGUF and JSON inputs are borrowed synchronously; failure
// preserves *out. This prepares text products without selecting or publishing a
// package.
bool PrepareGeneratedText(
    const ChatTemplateSelection &, FILE *gguf, const model::TextMetadata &,
    const std::optional<PublicationTokenFacts> &,
    std::string_view tokenizer_json,
    const jinja::TemplateAnalysisOptions &analysis_options,
    PreparedTextControl *out, std::string *error);

// Source selection is independent of control compilation and numerical
// execution. On failure, preserve the caller's output.
bool SelectChatTemplateSource(const model::TextMetadata &,
                              const std::optional<std::string> &template_name,
                              ChatTemplateSelection *out, std::string *error);
} // namespace hrx::loomify_gguf
#endif
