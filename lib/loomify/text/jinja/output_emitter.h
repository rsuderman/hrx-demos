// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_OUTPUT_EMITTER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_OUTPUT_EMITTER_H_

#include "lib/loomify/text/jinja/compiler.h"

namespace hrx::loomify_gguf::jinja {
struct TextFormatAnalysis;

// Emits finalized AnalyzeTextFormat output over shared bounded mechanics. JSON
// wire facts must be present exactly for JSON envelopes; malformed producer
// facts are rejected without classifying source patterns again. The
// scanner reads the source-selected format from opaque state. The fixed a401
// tool hook has no request/state, so all alternatives must describe the same
// parser. Finite tagged captures lower to direct bounded operations and the
// native XML backend. Returns only this component; the package owns
// composition. Failure preserves *out and identifies conflicting source facts.
bool EmitOutputParser(std::string_view source_name,
                      const TextFormatAnalysis &analysis,
                      GeneratedInputRenderer *out, Diagnostic *diagnostic);

} // namespace hrx::loomify_gguf::jinja
#endif
