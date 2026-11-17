// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_FORMAT_ANALYSIS_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_FORMAT_ANALYSIS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "lib/loomify/text/jinja/compiler.h"
#include "lib/loomify/text/jinja/parser.h"

namespace hrx::loomify_gguf::jinja {

struct SourceRange {
  size_t begin = 0;
  size_t end = 0;
};

// Semantic entry values. The output emitter alone translates these to its
// private scanner state; these are not bit patterns from an existing profile.
enum class TextStartupMode {
  Content = 0,
  ReasoningOpen = 1,
  ReasoningMayOpen = 2
};
enum class TextToolKind { None, JsonEnvelope, TaggedNamed, TaggedKeyValue };
enum class TextJsonArgumentWire { RawJson, Json };
enum class TextCapture {
  Literal,
  Content,
  Reasoning,
  FunctionName,
  Arguments,
  ArgumentKey,
  ArgumentValue,
  RepeatCalls,
  RepeatArguments,
  Choice,
  RepeatFirst,
  RepeatIndex,
  CallsValue,
  CallRecord,
  FunctionRecord,
  Expression,
  RepeatLast,
};
enum class TextValueWire {
  RawScalar,
  Json,
  StringElseJson,
  JsonContainersElseScalar,
  JsonContainersAndLiteralsElseScalar,
};
struct TextValueCodec {
  TextValueWire wire = TextValueWire::RawScalar;
  // Observed default-whitespace trimming of the raw scalar branch.
  bool trim = false;
};

// A bounded source pattern consumed by output emission. Repeats own the one
// repeated body, never an expanded probe count. This is private compiler data.
struct TextFormatPart {
  TextCapture capture = TextCapture::Literal;
  std::string literal;
  std::vector<TextFormatPart> body;
  // Choice retains a bounded future-value predicate over semantic capture
  // names, plus both actual source bodies. It is not an unknown codec bit.
  Expr predicate;
  // A source-error arm is not a successful empty output. Native recipe
  // validation emits this existing raise_exception expression for that arm.
  Expr body_error;
  Expr otherwise_error;
  // Exact supported capture transform, including filter arguments/keywords,
  // over predicate_bindings. Native probe validation lowers this with the
  // existing compiler rather than reimplementing JSON/string semantics.
  Expr value_expression;
  std::vector<TextFormatPart> otherwise;
  SourceRange source;
};

// Fully analyzed tagged wire structure. The source patterns remain separate
// for render recipes; output emission consumes these owned boundaries/codecs.
struct TextTaggedFormat {
  std::string before_name, after_name, after_arguments;
  std::string before_key, after_key, after_value;
  TextValueCodec value_codec;
};

struct TextToolFormat {
  TextToolKind kind = TextToolKind::None;
  std::string open;
  std::string close;
  std::string name_field;
  std::string arguments_field;
  std::string repeat_separator;
  std::vector<TextFormatPart> call_body;
  // Populated for TaggedNamed/TaggedKeyValue after routes and predicate
  // bindings are final. Other kinds do not use a tagged description.
  TextTaggedFormat tagged;
  // Present exactly for JsonEnvelope after successful format analysis. This is
  // the parser backend's argument wire, derived from the whole source capture.
  std::optional<TextJsonArgumentWire> json_argument_wire;
};

struct TextFormatAlternative {
  // Exact terminal mode emitted for this alternative by the format selector.
  TextStartupMode startup_mode = TextStartupMode::Content;
  std::string assistant_header;
  std::string generation_prefix;
  std::string frame_suffix;
  std::string reasoning_open;
  std::string reasoning_close;
  // Conversation-history framing around the reasoning payload. These are
  // separate from tag-only stream markers; consumers may qualify bounded
  // optional framing, never replace a marker by an entire history suffix.
  std::string reasoning_after_open;
  std::string reasoning_before_close;
  std::string reasoning_after_close;
  TextToolFormat tools;
  // Source history contains a bare JSON function record, which alone cannot
  // distinguish model-emitted calls from ordinary JSON content. The request
  // layer must reject that unqualified tool mode rather than enable/ignore it.
  bool ambiguous_bare_json_tools = false;
  // Source history directly formats one call rather than iterating calls.
  // This preserves its formatter without admitting ambiguous output parsing.
  bool historical_single_call = false;
  // Successful source validations discharged during completed-assistant
  // analysis. These simplify its retained error arms, not LIVE selector routes.
  std::vector<Expr> completion_guards;
  // Complete conceptual final-assistant emission, including its header and
  // conversation suffix, for compiled-render witness validation.
  std::vector<TextFormatPart> assistant_pattern;
  std::vector<SourceRange> evidence;
};

// History retention and generated stream syntax are separate: a source-proven
// retention sibling may qualify markers without adding a history capture.
bool HasHistoryReasoning(const TextFormatAlternative &format);

struct TextAdaptation {
  // Only a source operation requiring mapping arguments authorizes decoding
  // native wire STRINGs. Explicit source string branches remain observable.
  bool arguments_need_mapping = false;
  bool arguments_observe_string = false;
  // Retain native text projection only when no source branch observes the
  // content representation; otherwise source receives its original value.
  bool content_observes_representation = false;
  std::vector<SourceRange> evidence;
};

inline constexpr int64_t kTextAdaptationMappingArguments = 1;
inline constexpr int64_t kTextAdaptationArgumentStrings = 2;
inline constexpr int64_t kTextAdaptationOriginalContent = 4;

// The selector returns internal policy bits, not an accepted semantic route.
// STRING handling takes precedence over mapping. Failure preserves output.
bool DecodeTextAdaptation(int64_t mask, TextAdaptation *out);

struct TextFormatAnalysis {
  std::vector<TextFormatAlternative> alternatives;
  // Completion-only source provenance from this analyzer's existing resolved
  // assistant walk. Short-circuited/dead message-role paths contribute no
  // facts.
  std::set<size_t> completion_message_loops;
  std::map<size_t, Expr> completion_last_conditions;
  // Actual Output nodes whose newly emitted pattern contains Reasoning.
  // Capture source ranges may instead point to an alias definition.
  std::set<size_t> completion_reasoning_outputs;
  std::map<size_t, std::string> completion_rewrite_diagnostics;
  std::set<std::string> completion_output_dependencies;
  // Request/history computation only, with terminal Output List [id,mode].
  // CompileTemplateSelector lowers this using the existing compiler. Future
  // assistant values must never appear in this slice.
  Template selector_template;
  // Prefix shared by native witness recipes and request selectors. It retains
  // original request/history definitions but emits no conversation text.
  size_t selector_prelude_size = 0;
  // If present, the shared prelude starts with exactly one Set(name,1).
  // Completed-history recipes replace that initializer by 0 before execution.
  std::string target_missing_binding;
  // Computes fixed adaptation bits using the immutable PRE-adaptation
  // environment. Its second terminal integer is zero. Keeping this separate
  // prevents source type tests from observing the projection they select.
  Template adaptation_selector_template;
  // Generated predicate identifiers are collision-free with all source
  // bindings. Consumers resolve their actual semantic identity through this
  // map, never by recognizing an identifier prefix.
  std::map<std::string, TextCapture> predicate_bindings;
  // Guard snapshots retain their original source expressions for the probe
  // driver's external-binding discovery. Snapshot names are compiler locals,
  // never request options or semantic protocol IDs.
  std::map<std::string, Expr> selector_binding_expressions;
  // Exact retained guards are available to the native probe driver for
  // deterministic witness generation and coverage accounting. Unvisited
  // accepted routes are not validated merely because another probe succeeded.
  std::vector<std::vector<Expr>> format_route_guards;
  // The adaptation worker's explicit environment bindings. Its caller adds
  // these from immutable original history for EACH message/call. They cannot
  // come from request kwargs. Selecting once for an entire request is wrong.
  std::string adaptation_message_binding = "__text_raw_message";
  std::string adaptation_message_index_binding = "__text_message_index";
  std::string adaptation_call_binding = "__text_raw_call";
  std::string adaptation_call_index_binding = "__text_call_index";
};

struct TextFormatBindings {
  // Actual source-consumed GGUF globals, not a model name or protocol ID.
  // String markers are sufficient here; ordinary request values stay in the
  // generated selector's explicit environment.
  std::map<std::string, std::string> strings;
};

// A history-free continued USER fragment plus the actual default generation
// tail. The existing recipe builder lowers these source parts and definitions;
// text_binding is its sole runtime input, apart from GGUF globals.
struct TextUserContinuation {
  Template prelude;
  std::vector<TextFormatPart> pattern;
  std::map<std::string, TextCapture> predicate_bindings;
  std::string text_binding;
  std::string assistant_closure;
  std::vector<SourceRange> evidence;
};

// Qualifies only the USER shortcut. A diagnostic here does not invalidate the
// original structured-chat renderer. Existing format facts supply the closure;
// tokenizer terminal/whitespace qualification remains a separate step.
bool AnalyzeUserContinuation(std::string_view source,
                             std::string_view source_name,
                             const TextFormatBindings &bindings,
                             const TextFormatAnalysis &formats,
                             TextUserContinuation *out, Diagnostic *diagnostic);

// Classifies the exact source transform, not a separately stored annotation.
// Only the bounded capture encodings consumed by output generation qualify.
// Failure preserves output and describes the unsupported source operation.
bool ClassifyTextValueCodec(const TextFormatPart &part,
                            const TextFormatAnalysis &analysis,
                            TextValueCodec *out, std::string *error);

// Extracts finite source structure and finalizes output wire/marker admission
// after route merging and semantic bindings. It does not execute probe renders
// or certify learned model conventions. Differential compiled-render/selector
// witnesses qualify these facts in development tests, outside normal export.
// Failure leaves *out unchanged.
bool AnalyzeTextFormat(std::string_view source, std::string_view source_name,
                       const TextFormatBindings &bindings,
                       TextFormatAnalysis *out, Diagnostic *diagnostic);

namespace internal {
// Each analysis has fresh mode-specific state and borrows only original syntax
// and immutable lexical indexes from the source owner.
bool AnalyzeTextFormat(const SourceTemplate &source,
                       const TextFormatBindings &bindings,
                       TextFormatAnalysis *out, Diagnostic *diagnostic);
bool AnalyzeUserContinuation(const SourceTemplate &source,
                             const TextFormatBindings &bindings,
                             const TextFormatAnalysis &formats,
                             TextUserContinuation *out, Diagnostic *diagnostic);
} // namespace internal

} // namespace hrx::loomify_gguf::jinja
#endif
