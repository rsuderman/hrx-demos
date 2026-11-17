// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_ANALYSIS_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_ANALYSIS_H_

#include "lib/loomify/text/jinja/format_analysis.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace hrx::loomify_gguf::jinja {

struct TemplateAnalysisOptions {
  std::string helper_root;
  std::string scratch_parent;
  bool bytecode = false;
};

// Cold native discovery facts, not request-option admission rules. Effort bits
// and the default use the existing none/minimal/low/medium/high/xhigh/max ABI.
struct TextReasoningFacts {
  bool enabled = false;
  int64_t effort_mask = 0;
  int64_t default_effort = 0;
};

struct UserContinuationProduct {
  GeneratedInputRenderer renderer;
  std::string text_binding;
  std::string assistant_closure;
};

// Owned generation inputs. No prepared VM, borrowed source view or temporary
// probe storage survives this result. Package construction consumes these exact
// products rather than deriving text behavior a second time.
struct AnalyzedTemplate {
  TextFormatAnalysis format;
  GeneratedInputRenderer renderer;
  GeneratedInputRenderer selector;
  GeneratedInputRenderer adaptation_selector;
  TextReasoningFacts reasoning;
  // An unavailable bounded USER shortcut does not invalidate structured chat.
  // The diagnostic retains the exact source limitation for shortcut rejection.
  std::optional<UserContinuationProduct> continuation;
  Diagnostic continuation_diagnostic;
};

// Constructs the source-derived selectors, cold reasoning facts and actual USER
// product. Native execution is needed only for finite effort discovery,
// stopping on its first active context. Broad differential qualification is
// test-only. An unavailable USER shortcut preserves structured chat. Failure
// leaves *out unchanged; no VM or temporary context survives the owned result.
bool AnalyzeTemplate(std::string_view source, std::string_view source_name,
                     const GeneratedInputRenderer &input,
                     const TextFormatBindings &bindings,
                     const TemplateAnalysisOptions &options,
                     AnalyzedTemplate *out, Diagnostic *diagnostic);

// Lowers source-owned assistant/call recipes to the fixed a401 completion and
// formatter hooks. This fragment does not compose selectors or request
// adapters. A hook that needs unavailable request/call context is a source
// diagnostic.
bool EmitTextTurnControls(std::string_view source, std::string_view source_name,
                          const TextFormatAnalysis &analysis,
                          const TextFormatBindings &bindings,
                          GeneratedInputRenderer *out, Diagnostic *diagnostic);

namespace internal {
bool EmitTextTurnControls(const SourceTemplate &source,
                          const TextFormatAnalysis &analysis,
                          const TextFormatBindings &bindings,
                          GeneratedInputRenderer *out, Diagnostic *diagnostic);
// Package preparation shares the original source owner across all consumers.
// Returned products remain independent of that owner's lifetime.
bool AnalyzeTemplate(const SourceTemplate &source,
                     const GeneratedInputRenderer &input,
                     const TextFormatBindings &bindings,
                     const TemplateAnalysisOptions &options,
                     AnalyzedTemplate *out, Diagnostic *diagnostic);
} // namespace internal

} // namespace hrx::loomify_gguf::jinja

#endif // HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_ANALYSIS_H_
