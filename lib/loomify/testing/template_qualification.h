// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_TEMPLATE_QUALIFICATION_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TESTING_TEMPLATE_QUALIFICATION_H_
#include "lib/loomify/template_analysis.h"

namespace hrx::loomify_gguf::jinja {
struct TemplateQualification {
  size_t validated_probe_count = 0;
};
// Development-only differential qualification of exact production products.
// Failure leaves the caller's report unchanged; no package product is changed.
bool QualifyTemplate(std::string_view source, std::string_view source_name,
                     const TextFormatBindings &bindings,
                     const AnalyzedTemplate &analyzed,
                     const TemplateAnalysisOptions &options,
                     TemplateQualification *out, Diagnostic *diagnostic);
} // namespace hrx::loomify_gguf::jinja
#endif
