// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_REQUEST_EMITTER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_REQUEST_EMITTER_H_

#include "lib/loomify/text/jinja/compiler.h"

namespace hrx::loomify_gguf::jinja {

struct TextFormatAnalysis;
struct TextFormatBindings;

// Composes the accepted adaptation selector with shared request adaptation.
// The selector is the already-compiled product from template analysis, not a
// second lowering. Globals are actual supplied bindings; protected_names also
// includes source-consumed optional globals that must remain undefined when
// unavailable. The result owns source/helper/binding metadata and exposes
// text.adapt_messages, text.adapt_tools, text.template_globals and
// text.request_protected_names through its shared helper closure.
// Failure preserves *out. This does not switch package export or input hooks.
bool EmitRequestAdapters(std::string_view source_name,
                         const TextFormatAnalysis &analysis,
                         const GeneratedInputRenderer &adaptation_selector,
                         const TextFormatBindings &globals,
                         const std::vector<std::string> &protected_names,
                         GeneratedInputRenderer *out, Diagnostic *diagnostic);

} // namespace hrx::loomify_gguf::jinja
#endif
