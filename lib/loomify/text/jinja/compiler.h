// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_COMPILER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_COMPILER_H_

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace hrx::loomify_gguf::jinja {

struct Template;

enum class TemplateSelectorKind { Format, Adaptation };

struct Diagnostic {
  std::string message;
  std::string source_name;
  size_t offset = 0;
  size_t line = 1;
  size_t column = 1;
};

// Attaches original-source name and one-based byte line/column, clamping offset
// to the source size. LF starts a line; other bytes, including CR, advance the
// column. Preserves the message, accepts null, and can be applied repeatedly.
void LocateDiagnostic(std::string_view source, std::string_view source_name,
                      Diagnostic *diagnostic);

// An owned, source-derived renderer. Helpers identify finite BUILD-owned
// inputs, never paths or implementations supplied by template text.
struct GeneratedInputRenderer {
  std::string source;
  std::string entrypoint;
  std::vector<std::string> helper_requirements;
  // Environment reads emitted by this compiler, including guarded/default
  // paths. These are binding-discovery facts, not mandatory-token requirements.
  std::vector<std::string> external_bindings;
  // Compiler runtime capabilities read through the environment; ordinary
  // source identifiers are not a guessed reserved-name blacklist.
  std::vector<std::string> protected_bindings;
};

// Compiles the supported template language to direct Loom control flow over an
// explicit JSON environment. Leaves *out unchanged on failure. This operation
// performs no file I/O, template-revision recognition or serving policy
// selection.
bool CompileChatTemplate(std::string_view source, std::string_view source_name,
                         GeneratedInputRenderer *out, Diagnostic *diagnostic);

// Lowers an analyzer-owned assistant recipe AST with the renderer's existing
// expressions, filters, scopes and macros. No source reparsing or recipe
// interpreter is involved. The buffer-returning recipe entrypoint and its
// private symbols are distinct from the prompt renderer and selectors, allowing
// them to share one linked product. Failure preserves *out.
bool CompileTemplateRenderer(std::string_view original_source,
                             std::string_view source_name,
                             const Template &renderer_slice,
                             GeneratedInputRenderer *out,
                             Diagnostic *diagnostic,
                             std::string_view symbol_suffix = {});

// Lowers an analyzer-owned request-only AST slice using the same expression,
// macro and namespace semantics as the renderer. Each execution path must emit
// exactly one [format_id, startup_mode] pair of nonnegative integer literals;
// startup_mode is CONTENT=0, REASONING_OPEN=1 or REASONING_MAY_OPEN=2. An
// adaptation selector instead returns [adaptation_route, 0]. Loops may
// compute selector dependencies but cannot emit a selection. Macro output is
// ordinary source computation. The generated entrypoint takes one environment
// buffer and returns two i64 values. Failure leaves *out unchanged.
bool CompileTemplateSelector(
    std::string_view original_source, std::string_view source_name,
    const Template &selector_slice, GeneratedInputRenderer *out,
    Diagnostic *diagnostic,
    TemplateSelectorKind kind = TemplateSelectorKind::Format);

// Composes owned compiler products before native LINK. Their one-line helper
// declarations are deduplicated only when identical; conflicting declarations
// fail. Definitions retain order and their compiler-assigned private names.
// The selected root is copied, and helper requirements retain first-use order.
// Failure preserves *out. This accepts compiler output, not arbitrary Loom.
bool CombineGeneratedTemplates(const std::vector<GeneratedInputRenderer> &parts,
                               std::string_view entrypoint,
                               GeneratedInputRenderer *out,
                               Diagnostic *diagnostic);

// Reads a declared string fallback for an unshadowed external name from the
// existing source AST. Accepts default/d with one literal fallback, positional
// or default_value=, and consistent repeated declarations. This does not
// execute or change template rendering. Failure preserves *out and reports its
// location.
bool ReadTemplateStringDefault(std::string_view source,
                               std::string_view source_name,
                               std::string_view variable_name, std::string *out,
                               Diagnostic *diagnostic);

namespace internal {
class SourceTemplate;
// Original-source consumers share ownership and parsing on the package path.
// Every lowering still validates its actual product AST independently.
bool CompileChatTemplate(const SourceTemplate &source,
                         GeneratedInputRenderer *out, Diagnostic *diagnostic);
bool ReadTemplateStringDefault(const SourceTemplate &source,
                               std::string_view variable_name, std::string *out,
                               Diagnostic *diagnostic);
} // namespace internal

} // namespace hrx::loomify_gguf::jinja

#endif
