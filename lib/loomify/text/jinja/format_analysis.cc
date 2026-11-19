// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/format_analysis.h"
#include "lib/loomify/text/jinja/source_template.h"

#include <algorithm>
#include <charconv>
#include <optional>
#include <set>
#include <utility>

#include "iree/base/internal/json.h"

namespace hrx::loomify_gguf::jinja {
namespace {
using Parts = std::vector<TextFormatPart>;
enum class Origin {
  Unknown,
  Request,
  Messages,
  Message,
  Content,
  Reasoning,
  Calls,
  Call,
  Function,
  FunctionName,
  Arguments,
  ArgumentKeys,
  ArgumentItems,
  ArgumentKey,
  ArgumentValue,
  Loop,
  RepeatFirst,
  RepeatIndex,
  RepeatLast,
};
struct Value {
  Origin origin = Origin::Unknown;
  Expr expression;
  std::optional<std::string> string;
  std::optional<bool> boolean;
  Parts expansion;
  std::optional<bool> defined;
  std::optional<bool> truth;
  bool string_result = false;
  // Source-proven historical retention predicate: write identity and the
  // predicate polarity that retains reasoning. Not an ordinary style option.
  std::optional<std::pair<size_t, bool>> retention;
  size_t sequence = 0;
  size_t count_root = 0;
  size_t prefix_drop = 0;
  bool append_preserving = true;
  bool target_ordinal = false;
  // Equality of this scalar before/after appending this Analysis mode's target.
  // Derived from source, never from Origin::Request or folded truth alone.
  bool append_invariant = false;
  bool root_index_identity = false;
  std::vector<Expr> target_survival;
  // Original source expression with only last-sensitive local aliases expanded.
  // This is provenance for compiler lowering, not a second evaluator.
  std::optional<Expr> last_expression;
};
std::optional<bool> Truth(const Value &value) {
  if (value.boolean)
    return value.boolean;
  if (value.string)
    return !value.string->empty();
  return value.truth;
}
using Bindings = std::map<std::string, Value>;
struct Path {
  struct Retention {
    size_t search = 0;
    bool selected = false;
    bool keeps = false;
    Expr predicate;
  };
  Bindings bindings;
  Bindings module_bindings;
  Parts parts;
  std::vector<Expr> guards;
  // Completed-target admission can prove a framing arm unreachable without
  // restricting valid prompt rendering. Keep those facts for inference only.
  std::vector<Expr> proof_guards;
  std::vector<std::pair<size_t, Expr>> retained_validations;
  TextAdaptation adaptation;
  bool rejected = false;
  // Only the original root's explicit empty-history rejection establishes this.
  bool source_nonempty = false;
  std::optional<Expr> rejection;
  std::optional<Template> rejection_statements;
  std::map<Origin, unsigned> domains;
  std::map<Origin, bool> future_truth;
  size_t emission = 0;
  std::map<size_t, Retention> retention;
};
struct Context {
  bool target = false;
  bool continued_user = false;
  Expr messages;
  std::vector<std::pair<Origin, Expr>> loops;
  size_t target_sequence = 0;
  size_t target_loop = 0;
  // Only ScanTop grants same-scope availability of this retained producer.
  std::optional<size_t> request_capture;
  // Granted only by a future Choice arm beginning with an actual For;
  // never inherited through another lexical owner or repetition.
  std::optional<size_t> failing_loop;
  bool allow_statement_errors = true;
  // Original target-loop Output, preserved through macro-body expansion.
  std::optional<size_t> source_output_owner;
};
Expr Name(std::string name, SourceRange range = {}) {
  Expr result;
  result.kind = Expr::Kind::Name;
  result.text = std::move(name);
  result.offset = range.begin;
  result.end = range.end;
  return result;
}
Expr String(std::string text) {
  Expr result;
  result.kind = Expr::Kind::String;
  result.text = std::move(text);
  return result;
}
Expr Integer(size_t number) {
  Expr result;
  result.kind = Expr::Kind::Integer;
  result.text = std::to_string(number);
  return result;
}
Expr Boolean(bool value) {
  Expr result;
  result.kind = Expr::Kind::Boolean;
  result.text = value ? "true" : "false";
  return result;
}
Expr Unary(std::string op, Expr value) {
  Expr result;
  result.kind = Expr::Kind::Unary;
  result.text = std::move(op);
  result.args.push_back(std::move(value));
  return result;
}
Expr Binary(std::string op, Expr a, Expr b) {
  Expr result;
  result.kind = Expr::Kind::Binary;
  result.text = std::move(op);
  result.args.push_back(std::move(a));
  result.args.push_back(std::move(b));
  return result;
}
Expr Length(Expr value) {
  Expr result;
  result.kind = Expr::Kind::Filter;
  result.text = "length";
  result.args.push_back(std::move(value));
  return result;
}
SourceRange Range(const Expr &expression) {
  return {expression.offset, expression.end};
}
TextFormatPart Literal(std::string bytes, SourceRange range) {
  TextFormatPart result;
  result.literal = std::move(bytes);
  result.source = range;
  return result;
}
void Append(Parts *parts, TextFormatPart part) {
  if (part.capture == TextCapture::Literal && part.literal.empty())
    return;
  if (!parts->empty() && parts->back().capture == TextCapture::Literal &&
      part.capture == TextCapture::Literal) {
    parts->back().literal += part.literal;
    parts->back().source.end = part.source.end;
  } else {
    parts->push_back(std::move(part));
  }
}
std::optional<TextCapture> Capture(Origin origin) {
  switch (origin) {
  case Origin::Content:
    return TextCapture::Content;
  case Origin::Reasoning:
    return TextCapture::Reasoning;
  case Origin::FunctionName:
    return TextCapture::FunctionName;
  case Origin::Function:
  case Origin::Call:
    return std::nullopt;
  case Origin::Arguments:
    return TextCapture::Arguments;
  case Origin::ArgumentKey:
    return TextCapture::ArgumentKey;
  case Origin::ArgumentValue:
    return TextCapture::ArgumentValue;
  default:
    return std::nullopt;
  }
}
bool IsFuture(Origin origin) {
  return origin == Origin::Content || origin == Origin::Reasoning ||
         origin == Origin::Calls || origin == Origin::Call ||
         origin == Origin::Function || origin == Origin::FunctionName ||
         origin == Origin::Arguments || origin == Origin::ArgumentKeys ||
         origin == Origin::ArgumentItems || origin == Origin::ArgumentKey ||
         origin == Origin::ArgumentValue;
}
bool ContainsName(const Expr &expression, std::string_view prefix) {
  if (expression.kind == Expr::Kind::Name && expression.text == prefix)
    return true;
  for (const auto &child : expression.args)
    if (ContainsName(child, prefix))
      return true;
  return false;
}
std::string SemanticName(Origin origin) {
  switch (origin) {
  case Origin::Content:
    return "__text_future_content";
  case Origin::Reasoning:
    return "__text_future_reasoning";
  case Origin::Calls:
    return "__text_future_calls";
  case Origin::Call:
    return "__text_future_call";
  case Origin::Function:
    return "__text_future_function";
  case Origin::FunctionName:
    return "__text_future_function_name";
  case Origin::Arguments:
    return "__text_future_arguments";
  case Origin::ArgumentKey:
    return "__text_future_argument_key";
  case Origin::ArgumentValue:
    return "__text_future_argument_value";
  case Origin::RepeatFirst:
    return "__text_future_repeat_first";
  case Origin::RepeatIndex:
    return "__text_future_repeat_index";
  case Origin::RepeatLast:
    return "__text_future_repeat_last";
  default:
    return "__text_future_unknown";
  }
}
Value Constant(std::string string, const Expr &source) {
  Value value;
  value.origin = Origin::Request;
  value.expression = String(string);
  value.expression.offset = source.offset;
  value.expression.end = source.end;
  value.string = std::move(string);
  value.defined = true;
  value.string_result = true;
  return value;
}
Value Constant(bool boolean, const Expr &source) {
  Value value;
  value.origin = Origin::Request;
  value.expression = Boolean(boolean);
  value.expression.offset = source.offset;
  value.expression.end = source.end;
  value.boolean = boolean;
  value.defined = true;
  return value;
}
Value Constant(const char *string, const Expr &source) {
  return Constant(std::string(string), source);
}
std::optional<std::string> MemberName(const Expr &expression) {
  if (expression.kind == Expr::Kind::GetAttr)
    return expression.text;
  if (expression.kind == Expr::Kind::GetItem && expression.args.size() == 2 &&
      expression.args[1].kind == Expr::Kind::String)
    return expression.args[1].text;
  return std::nullopt;
}
bool HasCapture(const Parts &parts, TextCapture capture) {
  for (const auto &part : parts) {
    if (part.capture == capture || HasCapture(part.body, capture) ||
        HasCapture(part.otherwise, capture))
      return true;
  }
  return false;
}
const TextFormatPart *FindCapture(const Parts &parts, TextCapture capture) {
  for (const auto &part : parts) {
    if (part.capture == capture)
      return &part;
    if (auto *found = FindCapture(part.body, capture))
      return found;
    if (auto *found = FindCapture(part.otherwise, capture))
      return found;
  }
  return nullptr;
}
bool SameExpression(const Expr &a, const Expr &b) {
  if (a.kind != b.kind || a.text != b.text || a.args.size() != b.args.size())
    return false;
  // Parsed positional children carry empty keyword entries; synthesized AST
  // nodes may omit those entries. They denote the same positional binding.
  for (size_t i = 0; i < std::max(a.keywords.size(), b.keywords.size()); ++i) {
    const std::string_view left = i < a.keywords.size()
                                      ? std::string_view(a.keywords[i])
                                      : std::string_view{};
    const std::string_view right = i < b.keywords.size()
                                       ? std::string_view(b.keywords[i])
                                       : std::string_view{};
    if (left != right)
      return false;
  }
  for (size_t i = 0; i < a.args.size(); ++i)
    if (!SameExpression(a.args[i], b.args[i]))
      return false;
  return true;
}
bool SameGuards(const std::vector<Expr> &a, const std::vector<Expr> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!SameExpression(a[i], b[i]))
      return false;
  return true;
}
bool HasStatementError(const Parts &parts) {
  for (const auto &part : parts)
    if (!part.body_error_statements.body.empty() ||
        !part.otherwise_error_statements.body.empty() ||
        HasStatementError(part.body) || HasStatementError(part.otherwise))
      return true;
  return false;
}
bool SameErrorStatements(const Template &a, const Template &b) {
  if (a.body.size() != b.body.size())
    return false;
  // Construction retains exactly one childless, unfiltered For header.
  for (size_t i = 0; i < a.body.size(); ++i)
    if (a.body[i].kind != b.body[i].kind || a.body[i].text != b.body[i].text ||
        !SameExpression(a.body[i].expression, b.body[i].expression))
      return false;
  return true;
}
bool SameErrorPattern(const Parts &a, const Parts &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].capture != b[i].capture || a[i].literal != b[i].literal ||
        !SameExpression(a[i].predicate, b[i].predicate) ||
        !SameExpression(a[i].value_expression, b[i].value_expression) ||
        !SameExpression(a[i].body_error, b[i].body_error) ||
        !SameExpression(a[i].otherwise_error, b[i].otherwise_error) ||
        !SameErrorStatements(a[i].body_error_statements,
                             b[i].body_error_statements) ||
        !SameErrorStatements(a[i].otherwise_error_statements,
                             b[i].otherwise_error_statements) ||
        !SameErrorPattern(a[i].body, b[i].body) ||
        !SameErrorPattern(a[i].otherwise, b[i].otherwise))
      return false;
  return true;
}
Expr Conjunction(const std::vector<Expr> &guards) {
  Expr result = Boolean(true);
  for (const auto &guard : guards)
    result = Binary("and", std::move(result), guard);
  return result;
}
Node Selection(size_t id, TextStartupMode mode) {
  Node node;
  node.kind = Node::Kind::Output;
  node.expression.kind = Expr::Kind::List;
  node.expression.args = {Integer(id), Integer(static_cast<size_t>(mode))};
  return node;
}
Node UnmatchedSelection() {
  Node node;
  node.kind = Node::Kind::Output;
  node.expression.kind = Expr::Kind::Call;
  node.expression.args = {
      Name("raise_exception"),
      String("active request has no source-qualified text format")};
  node.expression.keywords.resize(2);
  return node;
}
bool HasOutput(const std::vector<Node> &nodes) {
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Output || node.kind == Node::Kind::Text)
      return true;
    if (node.kind != Node::Kind::Macro &&
        (HasOutput(node.body) || HasOutput(node.otherwise)))
      return true;
  }
  return false;
}
bool Raises(const std::vector<Node> &nodes) {
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Output &&
        node.expression.kind == Expr::Kind::Call &&
        !node.expression.args.empty() &&
        node.expression.args[0].kind == Expr::Kind::Name &&
        node.expression.args[0].text == "raise_exception")
      return true;
    if (Raises(node.body) || Raises(node.otherwise))
      return true;
  }
  return false;
}

bool TaggedSpace(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
std::string_view TrimTaggedLeft(std::string_view value) {
  while (!value.empty() && TaggedSpace(value.front()))
    value.remove_prefix(1);
  return value;
}
std::string_view TrimTagged(std::string_view value) {
  value = TrimTaggedLeft(value);
  while (!value.empty() && TaggedSpace(value.back()))
    value.remove_suffix(1);
  return value;
}

// Temporary analysis work, never part of the emitted parser's interface.
struct TaggedStep {
  TextCapture capture;
  std::string literal;
  std::vector<TaggedStep> body;
  TextValueCodec codec;
};
void AppendTaggedStep(TaggedStep step, std::vector<TaggedStep> *out) {
  if (step.capture == TextCapture::Literal && step.literal.empty())
    return;
  if (step.capture == TextCapture::Literal && !out->empty() &&
      out->back().capture == TextCapture::Literal)
    out->back().literal += step.literal;
  else
    out->push_back(std::move(step));
}
bool SameTaggedSteps(const std::vector<TaggedStep> &a,
                     const std::vector<TaggedStep> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].capture != b[i].capture || a[i].literal != b[i].literal ||
        a[i].codec.wire != b[i].codec.wire ||
        a[i].codec.trim != b[i].codec.trim ||
        !SameTaggedSteps(a[i].body, b[i].body))
      return false;
  return true;
}

// Choices may collapse only when their complete wire structure is invariant.
// Value choices retain their exact expressions for the one codec classifier.
bool TaggedSteps(const Parts &parts, const TextFormatAnalysis &analysis,
                 std::string_view opener, std::vector<TaggedStep> *out,
                 std::string *error) {
  for (const auto &part : parts) {
    const auto contains = [&](TextCapture capture) {
      return part.capture == capture || HasCapture(part.body, capture) ||
             HasCapture(part.otherwise, capture);
    };
    const bool value = contains(TextCapture::ArgumentValue) &&
                       !contains(TextCapture::ArgumentKey) &&
                       !contains(TextCapture::FunctionName) &&
                       !contains(TextCapture::RepeatArguments);
    if (value) {
      TaggedStep step{TextCapture::ArgumentValue, {}, {}};
      if (!ClassifyTextValueCodec(part, analysis, &step.codec, error))
        return false;
      AppendTaggedStep(std::move(step), out);
    } else if (part.capture == TextCapture::Literal ||
               part.capture == TextCapture::FunctionName ||
               part.capture == TextCapture::ArgumentKey) {
      if (part.capture != TextCapture::Literal) {
        TextValueCodec codec;
        if (!ClassifyTextValueCodec(part, analysis, &codec, error))
          return false;
        if (codec.wire != TextValueWire::RawScalar || codec.trim) {
          *error = "tagged names require a direct source capture";
          return false;
        }
      }
      AppendTaggedStep({part.capture, part.literal, {}}, out);
    } else if (part.capture == TextCapture::RepeatArguments) {
      TaggedStep step{part.capture, {}, {}};
      if (!TaggedSteps(part.body, analysis, {}, &step.body, error))
        return false;
      AppendTaggedStep(std::move(step), out);
    } else if (part.capture == TextCapture::Expression) {
      if (!TaggedSteps(part.body, analysis, opener, out, error))
        return false;
    } else if (part.capture == TextCapture::Choice) {
      if (part.HasBodyError() || part.HasOtherwiseError()) {
        if (part.HasBodyError() && part.HasOtherwiseError()) {
          *error = "tagged source has no successful wire arm";
          return false;
        }
        const auto &success = !part.HasBodyError() ? part.body : part.otherwise;
        if (!TaggedSteps(success, analysis, opener, out, error))
          return false;
        continue;
      }
      std::vector<TaggedStep> yes, no;
      if (!TaggedSteps(part.body, analysis, opener, &yes, error) ||
          !TaggedSteps(part.otherwise, analysis, opener, &no, error))
        return false;
      auto outside_whitespace = [&](std::vector<TaggedStep> *steps) {
        if (opener.empty() || steps->empty() ||
            steps->front().capture != TextCapture::Literal)
          return;
        auto &literal = steps->front().literal;
        const auto stripped = TrimTaggedLeft(literal);
        if (stripped.empty() || stripped.substr(0, opener.size()) == opener)
          literal = std::string(stripped);
        if (literal.empty())
          steps->erase(steps->begin());
      };
      outside_whitespace(&yes);
      outside_whitespace(&no);
      // The repetition already includes its zero-argument case.
      if (yes.empty() && no.size() == 1 &&
          no[0].capture == TextCapture::RepeatArguments)
        yes = no;
      if (no.empty() && yes.size() == 1 &&
          yes[0].capture == TextCapture::RepeatArguments)
        no = yes;
      if (!SameTaggedSteps(yes, no)) {
        *error = "tagged response choices require an invariant wire structure";
        return false;
      }
      for (auto &step : yes)
        AppendTaggedStep(std::move(step), out);
    } else {
      *error = "tagged source contains an unsupported structural capture";
      return false;
    }
  }
  return true;
}

bool DescribeTagged(const TextToolFormat &tools,
                    const TextFormatAnalysis &analysis, TextTaggedFormat *out,
                    std::string *error) {
  std::vector<TaggedStep> call;
  if (!TaggedSteps(tools.call_body, analysis, tools.open, &call, error))
    return false;
  const TaggedStep *name = nullptr, *repeat = nullptr;
  int position = 0;
  for (const auto &step : call) {
    if (step.capture == TextCapture::FunctionName && position == 0) {
      name = &step;
      position = 1;
    } else if (step.capture == TextCapture::RepeatArguments && position == 1) {
      repeat = &step;
      position = 2;
    } else if (step.capture == TextCapture::Literal) {
      (position == 0   ? out->before_name
       : position == 1 ? out->after_name
                       : out->after_arguments) += step.literal;
    } else {
      *error =
          "tagged call requires one name followed by one argument repetition";
      return false;
    }
  }
  if (!name || !repeat || tools.open.empty() || tools.close.empty()) {
    *error = "tagged call requires bounded call and argument structure";
    return false;
  }
  out->before_name = std::string(TrimTaggedLeft(out->before_name));
  if (out->before_name.substr(0, tools.open.size()) != tools.open) {
    *error = "tagged call does not begin at its selected tool entry";
    return false;
  }
  position = 0;
  const TextValueCodec *value = nullptr;
  for (const auto &step : repeat->body) {
    if (step.capture == TextCapture::ArgumentKey && position == 0) {
      position = 1;
    } else if (step.capture == TextCapture::ArgumentValue && position == 1) {
      value = &step.codec;
      position = 2;
    } else if (step.capture == TextCapture::Literal) {
      (position == 0   ? out->before_key
       : position == 1 ? out->after_key
                       : out->after_value) += step.literal;
    } else {
      *error = "tagged argument requires one key and one bounded value";
      return false;
    }
  }
  if (!value || TrimTagged(out->before_key).empty() ||
      TrimTagged(out->after_key).empty() ||
      TrimTagged(out->after_value).empty() ||
      TrimTagged(out->after_arguments).empty()) {
    *error = "tagged captures require nonempty structural boundaries";
    return false;
  }
  out->value_codec = *value;
  return true;
}

class Analysis {
public:
  Analysis(const internal::SourceTemplate &source,
           const TextFormatBindings &globals, Diagnostic *diagnostic)
      : source_(source.source()), source_name_(source.source_name()),
        ast_(source.ast()), globals_(globals), diagnostic_(diagnostic),
        macros_(source.analysis_macros()), source_names_(source.source_names()),
        macro_calls_(source.analysis_macro_calls()) {
    GatherRecursiveDependencies();
  }

  bool Run(TextFormatAnalysis *out) {
    const auto &ast = ast_;
    GatherDependencies(ast.body, {}, false);
    CloseNamespaceAliasDependencies(ast.body);
    GatherHistorySearches(ast.body);
    if (failed_)
      return false;
    analysis_needed_ = NeededClosure(analysis_roots_);
    literal_needed_ = NeededClosure(literal_roots_, true);
    auto worker_name = [&](std::string name) {
      while (source_names_.count(name))
        name += '_';
      return name;
    };
    worker_message_ = worker_name("__text_raw_message");
    worker_message_index_ = worker_name("__text_message_index");
    worker_call_ = worker_name("__text_raw_call");
    worker_call_index_ = worker_name("__text_call_index");
    target_missing_ = FreshCountName("__text_target_missing");
    selector_binding_expressions_[target_missing_] = Integer(1);
    RegisterSequenceCount(0, 0, 0, Name("messages"), 0, false, true);
    Path initial;
    initial.bindings["messages"] = {Origin::Messages, Name("messages")};
    initial.bindings["messages"].truth = true;
    initial.bindings["messages"].root_index_identity = true;
    initial.bindings["messages"].defined = true;
    // The shared LIVE/COMPLETED environment always supplies the generation
    // control. Unlike ordinary kwargs, it is not an omitted source option.
    initial.bindings["add_generation_prompt"] = {Origin::Request,
                                                 Name("add_generation_prompt")};
    initial.bindings["add_generation_prompt"].defined = true;
    for (const auto &[name, bytes] : globals_.strings)
      initial.bindings[name] = Constant(bytes, Name(name));
    initial.module_bindings = initial.bindings;
    // Discover emitted assistant bodies; historical bookkeeping remains in the
    // selector slice, never contributes marker literals from system prose.
    ScanTop(ast.body, {initial});
    if (failed_)
      return false;
    ProvePromptPrefix();
    bodies_ = FormatChoices(bodies_, true);
    // Splitting a request guard can expose an arm which emits no assistant
    // bytes. It is not a format: preserve its non-admission through the guarded
    // selector's unmatched terminal, rather than assigning it a frame/stop.
    // Nonempty partial frames still undergo normal description/token checks.
    bodies_.erase(
        std::remove_if(bodies_.begin(), bodies_.end(),
                       [](const Path &path) { return path.parts.empty(); }),
        bodies_.end());
    generations_ = FormatChoices(generations_);
    if (bodies_.empty())
      return Fail(0, "no supported assistant content emission was found");
    projection_mode_ = true;
    ScanTop(ast.body, {initial});
    projection_mode_ = false;
    if (failed_)
      return false;
    std::set<std::string> selector_roots;
    auto part_roots = [&](auto &&self, const Parts &parts) -> void {
      for (const auto &part : parts) {
        References(part.predicate, &selector_roots);
        References(part.value_expression, &selector_roots);
        References(part.body_error, &selector_roots);
        References(part.otherwise_error, &selector_roots);
        NodeReferences(part.body_error_statements.body, &selector_roots);
        NodeReferences(part.otherwise_error_statements.body, &selector_roots);
        self(self, part.body);
        self(self, part.otherwise);
      }
    };
    for (const auto &path : bodies_) {
      part_roots(part_roots, path.parts);
      for (const auto &guard : path.guards)
        References(guard, &selector_roots);
    }
    for (const auto &path : generations_)
      for (const auto &guard : path.guards)
        References(guard, &selector_roots);
    for (const auto &rules : projection_rules_)
      for (const auto &guard : rules)
        References(guard, &selector_roots);
    for (const auto &guard : projection_raw_mapping_rules_)
      References(guard, &selector_roots);
    for (const auto &guard : projection_rejection_rules_)
      References(guard, &selector_roots);
    if (!recursive_target_outputs_.empty())
      selector_roots.insert(target_missing_);
    selector_needed_ = SelectorDependencies(std::move(selector_roots));
    std::vector<Node> prelude;
    if (selector_needed_.count(target_missing_)) {
      Node missing;
      missing.kind = Node::Kind::Set;
      missing.text = target_missing_;
      missing.expression = Integer(1);
      prelude.push_back(std::move(missing));
    }
    EmitCountDefinitions(0, false, true, &prelude);
    SlicePrelude(ast.body, &prelude);
    TextFormatAnalysis result;
    std::vector<std::vector<Expr>> routes;
    std::vector<const Path *> route_sources;
    for (const auto &body : bodies_) {
      TextFormatAlternative alternative;
      if (!Describe(body.parts, &alternative))
        return false;
      alternative.completion_guards = body.proof_guards;
      const auto routing = RuntimeGuards(body);
      auto it = std::find_if(routes.begin(), routes.end(), [&](const auto &g) {
        return SameGuards(g, routing);
      });
      if (it == routes.end()) {
        routes.push_back(routing);
        result.alternatives.push_back(std::move(alternative));
        route_sources.push_back(&body);
      } else {
        const size_t id = static_cast<size_t>(it - routes.begin());
        if (!Merge(alternative, &result.alternatives[id]))
          return false;
        route_sources[id] = nullptr;
      }
    }
    if (!QualifyRetentionStreams(&result.alternatives, route_sources))
      return false;
    // The generation suffix is a separate source path. Its guards combine with
    // request-qualified body routes, while payload conditions never enter here.
    if (generations_.empty())
      generations_.push_back(Path{});
    const bool counted_body =
        std::any_of(routes.begin(), routes.end(), [&](const auto &guards) {
          return std::any_of(
              guards.begin(), guards.end(),
              [&](const Expr &guard) { return CountDependent(guard); });
        });
    if (counted_body)
      for (const auto &generation : generations_)
        for (const auto &guard : generation.guards)
          if (CountDependent(guard))
            return Fail(
                guard.offset,
                "count-dependent body and generation routes need an unsupported cross-domain proof");
    std::vector<Node> choices;
    auto body_formats = std::move(result.alternatives);
    std::vector<std::vector<Expr>> selected_guards;
    for (size_t id = 0; id < routes.size(); ++id) {
      for (const auto &generation : generations_) {
        std::string prefix;
        if (!LiteralText(generation.parts, &prefix))
          return Fail(0, "generation prefix has a non-finite source value");
        auto guards = routes[id];
        bool compatible = true;
        for (const auto &guard : generation.guards)
          compatible &= AddGuard(&guards, guard);
        if (!compatible)
          continue;
        auto format = body_formats[id];
        TextStartupMode mode = TextStartupMode::Content;
        if (!prefix.empty() && !format.reasoning_open.empty()) {
          if (prefix.find(format.reasoning_close) != std::string::npos &&
              !format.reasoning_close.empty())
            mode = TextStartupMode::Content;
          else if (prefix.find(format.reasoning_open) != std::string::npos)
            mode = TextStartupMode::ReasoningOpen;
          else
            mode = TextStartupMode::ReasoningMayOpen;
        }
        // Entry literals belong to this source route. A different request's
        // prefix must never overwrite/freeze the first observed prefix.
        format.generation_prefix = prefix;
        format.startup_mode = mode;
        const size_t selected_id = result.alternatives.size();
        result.alternatives.push_back(std::move(format));
        selected_guards.push_back(guards);
        Node branch;
        branch.kind = Node::Kind::If;
        branch.expression = Conjunction(guards);
        branch.body.push_back(Selection(selected_id, mode));
        choices.push_back(std::move(branch));
      }
    }
    // Nest, rather than emit independent Ifs: exactly one terminal is executed.
    std::vector<Node> tail{UnmatchedSelection()};
    for (auto it = choices.rbegin(); it != choices.rend(); ++it) {
      it->otherwise = std::move(tail);
      std::vector<Node> moved;
      moved.push_back(std::move(*it));
      tail = std::move(moved);
    }
    result.selector_template.body = prelude;
    result.selector_prelude_size = prelude.size();
    if (selector_needed_.count(target_missing_))
      result.target_missing_binding = target_missing_;
    result.selector_template.body.insert(result.selector_template.body.end(),
                                         tail.begin(), tail.end());
    result.adaptation_selector_template.body = std::move(prelude);
    std::vector<Expr> projection_predicates;
    for (size_t feature = 0; feature < 3; ++feature)
      projection_predicates.push_back(ProjectionPredicate(feature));
    Expr raw_mapping = Boolean(false);
    for (const auto &rule : projection_raw_mapping_rules_)
      raw_mapping = Binary("or", std::move(raw_mapping), rule);
    raw_mapping = SimplifyBoolean(std::move(raw_mapping));
    projection_predicates[0] =
        SimplifyBoolean(Binary("or", projection_predicates[0], raw_mapping));
    Expr rejected_string = Boolean(false);
    for (const auto &rule : projection_rejection_rules_)
      rejected_string = Binary("or", std::move(rejected_string), rule);
    rejected_string = SimplifyBoolean(std::move(rejected_string));
    projection_predicates[1] = SimplifyBoolean(
        Binary("and", projection_predicates[1],
               Unary("not", Binary("or", raw_mapping, rejected_string))));
    result.adaptation_selector_template.body.push_back(
        ProjectionTree(0, projection_predicates, {}, {}));
    result.format_route_guards = std::move(selected_guards);
    result.predicate_bindings = predicate_bindings_;
    result.completion_message_loops = completion_message_loops_;
    result.completion_prompt_prefixes = completion_prompt_prefixes_;
    result.completion_last_conditions = completion_last_conditions_;
    result.completion_reasoning_outputs = completion_reasoning_outputs_;
    result.completion_rewrite_diagnostics = completion_rewrite_diagnostics_;
    result.completion_output_dependencies = analysis_needed_;
    result.selector_binding_expressions = selector_binding_expressions_;
    result.adaptation_message_binding = worker_message_;
    result.adaptation_message_index_binding = worker_message_index_;
    result.adaptation_call_binding = worker_call_;
    result.adaptation_call_index_binding = worker_call_index_;
    if (failed_)
      return false;
    if (!FinalizeOutputDescriptions(&result))
      return false;
    *out = std::move(result);
    if (diagnostic_)
      *diagnostic_ = {};
    return true;
  }

  bool RunContinuation(const TextFormatAnalysis &formats,
                       TextUserContinuation *out) {
    continuation_mode_ = true;
    const auto &ast = ast_;
    GatherDependencies(ast.body, {}, false);
    CloseNamespaceAliasDependencies(ast.body);
    analysis_needed_ = NeededClosure(analysis_roots_);
    literal_needed_ = NeededClosure(literal_roots_, true);
    target_missing_ = FreshCountName("__text_target_missing");
    user_ordinal_ = FreshCountName("__text_continued_user_ordinal");
    RegisterSequenceCount(0, 0, 0, Name("messages"), 0, false, true);
    Path initial;
    initial.bindings["messages"] = {Origin::Messages, Name("messages")};
    initial.bindings["messages"].defined = true;
    initial.bindings["messages"].truth = true;
    initial.bindings["messages"].root_index_identity = true;
    initial.bindings["add_generation_prompt"] =
        Constant(true, Name("add_generation_prompt"));
    for (const auto &[name, bytes] : globals_.strings)
      initial.bindings[name] = Constant(bytes, Name(name));
    initial.module_bindings = initial.bindings;
    auto default_paths = ScanTop(ast.body, {std::move(initial)});
    if (failed_)
      return false;
    if (bodies_.size() != 1 || !bodies_.front().guards.empty())
      return Fail(
          0,
          "continued USER output depends on unavailable history or source admission");
    TextUserContinuation result;
    result.pattern = std::move(bodies_.front().parts);
    for (auto &generation : generations_) {
      if (!generation.guards.empty())
        return Fail(
            0,
            "continued USER generation depends on unavailable history or options");
      for (auto &part : generation.parts)
        Append(&result.pattern, std::move(part));
    }
    std::set<std::string> roots;
    auto references = [&](auto &&self, const Parts &parts) -> void {
      for (const auto &part : parts) {
        References(part.predicate, &roots);
        References(part.value_expression, &roots);
        References(part.body_error, &roots);
        References(part.otherwise_error, &roots);
        NodeReferences(part.body_error_statements.body, &roots);
        NodeReferences(part.otherwise_error_statements.body, &roots);
        self(self, part.body);
        self(self, part.otherwise);
        result.evidence.push_back(part.source);
      }
    };
    references(references, result.pattern);
    selector_needed_ = SelectorDependencies(std::move(roots));
    for (const auto &name : selector_needed_)
      if (name == "messages" || name.compare(0, 9, "messages.") == 0 ||
          name == user_ordinal_ || count_target_names_.count(name) ||
          count_targets_.count(name))
        return Fail(0,
                    "continued USER output reads unavailable history: " + name);
    for (const auto &[name, capture] : predicate_bindings_) {
      if (!selector_needed_.count(name))
        continue;
      if (capture != TextCapture::Content)
        return Fail(
            0,
            "continued USER output requires an unavailable assistant or call value");
      result.predicate_bindings[name] = capture;
    }
    SlicePrelude(ast.body, &result.prelude.body);
    result.text_binding = FreshCountName("__text_continued_user_text");
    if (formats.format_route_guards.size() != formats.alternatives.size() ||
        default_paths.size() != 1)
      return Fail(0, "continued USER needs qualified default route facts");
    Path defaults = std::move(default_paths.front());
    for (const auto &[name, expression] : formats.selector_binding_expressions)
      defaults.bindings.emplace(name, Value{Origin::Request, Name(name)});
    for (const auto &[name, capture] : formats.predicate_bindings)
      defaults.bindings.emplace(name, Value{Origin::Request, Name(name)});
    for (size_t id = 0; id < formats.alternatives.size(); ++id) {
      bool inactive = false;
      for (const auto &guard : formats.format_route_guards[id]) {
        const auto expression = DefaultGuard(guard);
        const auto value = Resolve(expression, defaults, {});
        if (const auto known = Truth(value); known && !*known) {
          inactive = true;
          break;
        }
      }
      if (inactive)
        continue;
      const auto &format = formats.alternatives[id];
      if (format.frame_suffix.empty())
        return Fail(0,
                    "continued USER has no source-qualified assistant closure");
      if (result.assistant_closure.empty())
        result.assistant_closure = format.frame_suffix;
      else if (result.assistant_closure != format.frame_suffix)
        return Fail(
            0,
            "continued USER assistant closure depends on unavailable context");
    }
    if (result.assistant_closure.empty())
      return Fail(0, "continued USER needs qualified assistant frame facts");
    if (failed_)
      return false;
    *out = std::move(result);
    if (diagnostic_)
      *diagnostic_ = {};
    return true;
  }

private:
  bool FinalizeOutputDescriptions(TextFormatAnalysis *result) {
    // Merging may replace a tool body, and value classification needs the
    // final semantic bindings. Publish only completely analyzed descriptions.
    for (auto &format : result->alternatives) {
      if (format.tools.kind != TextToolKind::TaggedNamed &&
          format.tools.kind != TextToolKind::TaggedKeyValue)
        continue;
      std::string error;
      if (!DescribeTagged(format.tools, *result, &format.tools.tagged, &error))
        return Fail(format.evidence.empty() ? 0 : format.evidence[0].begin,
                    std::move(error));
    }
    auto prefix_conflict = [](std::string_view a, std::string_view b) {
      return !a.empty() && !b.empty() &&
             (a.substr(0, b.size()) == b || b.substr(0, a.size()) == a);
    };
    for (auto &format : result->alternatives) {
      const size_t offset =
          format.evidence.empty() ? 0 : format.evidence[0].begin;
      auto value_captures = [&](auto &&self,
                                const TextFormatPart &part) -> int {
        int captures = part.capture == TextCapture::FunctionName ? 1
                       : part.capture == TextCapture::Arguments  ? 2
                                                                 : 0;
        for (const auto &child : part.body)
          captures |= self(self, child);
        for (const auto &child : part.otherwise)
          captures |= self(self, child);
        return captures;
      };
      auto qualify = [&](auto &&self,
                         const std::vector<TextFormatPart> &parts) -> bool {
        for (const auto &part : parts) {
          const int captures = value_captures(value_captures, part);
          const bool value_choice = part.capture == TextCapture::Choice &&
                                    (captures == 1 || captures == 2);
          if (part.capture == TextCapture::FunctionName ||
              part.capture == TextCapture::Arguments || value_choice) {
            TextValueCodec codec;
            std::string error;
            if (!ClassifyTextValueCodec(part, *result, &codec, &error))
              return Fail(part.source.begin, std::move(error));
            if (captures == 2) {
              const auto wire = codec.wire == TextValueWire::Json
                                    ? TextJsonArgumentWire::Json
                                    : TextJsonArgumentWire::RawJson;
              auto &prior = format.tools.json_argument_wire;
              if (prior && *prior != wire)
                return Fail(
                    part.source.begin,
                    "JSON arguments require one source-qualified wire encoding");
              prior = wire;
            }
            // A choice is one exact value expression; classifying its leaves
            // separately would lose the predicate's wire interpretation.
            continue;
          }
          if (!self(self, part.body) || !self(self, part.otherwise))
            return false;
        }
        return true;
      };
      if (format.tools.kind == TextToolKind::JsonEnvelope &&
          !qualify(qualify, format.tools.call_body))
        return false;
      if (format.tools.kind == TextToolKind::JsonEnvelope &&
          !format.tools.json_argument_wire)
        return Fail(offset, "JSON arguments lack a qualified source capture");
      if (format.reasoning_open.empty() != format.reasoning_close.empty())
        return Fail(offset,
                    "reasoning output requires both source-qualified markers");
      if (format.tools.kind == TextToolKind::JsonEnvelope &&
          (format.tools.open.empty() || format.tools.close.empty() ||
           format.tools.name_field.empty() ||
           format.tools.arguments_field.empty()))
        return Fail(offset,
                    "JSON calls require nonempty envelope and member names");
      if (prefix_conflict(format.tools.open, format.reasoning_close) ||
          prefix_conflict(format.tools.open, format.reasoning_open) ||
          prefix_conflict(format.reasoning_open, format.reasoning_close))
        return Fail(offset,
                    "output marker prefixes require an unambiguous priority");
      if (format.reasoning_after_open.size() >= (1u << 24) ||
          format.reasoning_after_close.size() >= (1u << 24))
        return Fail(offset,
                    "reasoning framing exceeds bounded stream progress");
    }
    return true;
  }

  Expr DefaultGuard(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Name) {
      if (const auto known = default_conditions_.find(expression.text);
          known != default_conditions_.end() && known->second)
        return Boolean(*known->second);
    }
    Expr result = expression;
    for (auto &argument : result.args)
      argument = DefaultGuard(argument);
    return result;
  }
  unsigned Domain(Origin origin, const Path &path) const {
    if (const auto found = path.domains.find(origin);
        found != path.domains.end())
      return found->second;
    switch (origin) {
    // Native completed-assistant construction always includes content as
    // STRING or NULL (tool-only); reasoning is omitted or a nonempty STRING.
    // This is the generated record domain, not original request history.
    case Origin::Content:
      return 2 | 4;
    case Origin::Reasoning:
      return 1 | 4;
    case Origin::FunctionName:
    case Origin::ArgumentKey:
      return 4;
    case Origin::Calls:
      return 1 | 64;
    case Origin::Call:
    case Origin::Function:
      return 32;
    case Origin::Arguments:
      return 4 | 32;
    case Origin::ArgumentValue:
      return 2 | 4 | 8 | 16 | 32 | 64;
    case Origin::RepeatFirst:
    case Origin::RepeatLast:
      return 8;
    case Origin::RepeatIndex:
      return 16;
    default:
      return 127;
    }
  }
  unsigned TestMask(std::string_view test) const {
    if (test == "undefined")
      return 1;
    if (test == "defined")
      return 126;
    if (test == "none")
      return 2;
    if (test == "string")
      return 4;
    if (test == "boolean")
      return 8;
    if (test == "number" || test == "integer" || test == "float")
      return 16;
    if (test == "mapping")
      return 32;
    if (test == "sequence")
      return 4 | 64;
    if (test == "iterable")
      return 4 | 32 | 64;
    return 127;
  }
  void Narrow(const Expr &condition, bool truth, Path *path) {
    if (condition.kind == Expr::Kind::Unary && condition.text == "not" &&
        condition.args.size() == 1) {
      Narrow(condition.args[0], !truth, path);
      return;
    }
    if (condition.kind == Expr::Kind::Name) {
      for (const auto &[origin, name] : semantic_names_)
        if (name == condition.text) {
          path->future_truth[origin] = truth;
          if (origin == Origin::Calls) {
            path->domains[origin] = truth ? 64 : 1;
            if (!truth)
              path->domains[Origin::Content] = 4;
          }
        }
      return;
    }
    if (condition.kind == Expr::Kind::Binary && condition.args.size() == 2 &&
        ((condition.text == "and" && truth) ||
         (condition.text == "or" && !truth))) {
      Narrow(condition.args[0], truth, path);
      Narrow(condition.args[1], truth, path);
      return;
    }
    if (condition.kind != Expr::Kind::Test || condition.args.empty() ||
        condition.args[0].kind != Expr::Kind::Name)
      return;
    for (const auto &[origin, name] : semantic_names_)
      if (name == condition.args[0].text) {
        const unsigned mask = TestMask(condition.text);
        path->domains[origin] =
            Domain(origin, *path) & (truth ? mask : 127 ^ mask);
      }
  }
  Value Semantic(Origin origin, const Expr &source) {
    if (projection_mode_) {
      Expr message = Name(worker_message_);
      Expr call = Name(worker_call_);
      auto member = [&](Expr base, const char *key) {
        Expr result;
        result.kind = Expr::Kind::GetAttr;
        result.text = key;
        result.args = {std::move(base)};
        result.offset = source.offset;
        result.end = source.end;
        return result;
      };
      Expr expression;
      switch (origin) {
      case Origin::Content:
        expression = member(message, "content");
        break;
      case Origin::Reasoning:
        expression = member(message, "reasoning_content");
        break;
      case Origin::Calls:
        expression = member(message, "tool_calls");
        break;
      case Origin::Call:
        expression = call;
        break;
      case Origin::Function:
        expression = member(call, "function");
        break;
      case Origin::FunctionName:
        expression = member(member(call, "function"), "name");
        break;
      case Origin::Arguments:
        expression = member(member(call, "function"), "arguments");
        break;
      default:
        expression = Name("__text_raw_argument_value");
        break;
      }
      return {origin, std::move(expression)};
    }
    auto found = semantic_names_.find(origin);
    if (found == semantic_names_.end()) {
      std::string name = SemanticName(origin);
      while (source_names_.count(name) || generated_names_.count(name))
        name += '_';
      generated_names_.insert(name);
      found = semantic_names_.emplace(origin, std::move(name)).first;
      auto capture = Capture(origin);
      if (capture)
        predicate_bindings_[found->second] = *capture;
      else if (origin == Origin::Calls)
        predicate_bindings_[found->second] = TextCapture::CallsValue;
      else if (origin == Origin::Call)
        predicate_bindings_[found->second] = TextCapture::CallRecord;
      else if (origin == Origin::Function)
        predicate_bindings_[found->second] = TextCapture::FunctionRecord;
      else if (origin == Origin::RepeatFirst)
        predicate_bindings_[found->second] = TextCapture::RepeatFirst;
      else if (origin == Origin::RepeatIndex)
        predicate_bindings_[found->second] = TextCapture::RepeatIndex;
      else if (origin == Origin::RepeatLast)
        predicate_bindings_[found->second] = TextCapture::RepeatLast;
    }
    Value value;
    value.origin = origin;
    value.expression = Name(found->second, Range(source));
    return value;
  }
  bool DirectSemantic(const Value &value) const {
    const auto name = semantic_names_.find(value.origin);
    return value.expression.kind == Expr::Kind::Name &&
           name != semantic_names_.end() &&
           value.expression.text == name->second;
  }
  bool Future(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Name &&
        predicate_bindings_.count(expression.text))
      return true;
    for (const auto &child : expression.args)
      if (Future(child))
        return true;
    return false;
  }
  bool Future(const Template &statements) const {
    std::set<std::string> names;
    NodeReferences(statements.body, &names);
    for (const auto &name : names)
      if (predicate_bindings_.count(name))
        return true;
    return false;
  }
  std::optional<Expr> PatternExpression(const Parts &parts) {
    Expr result = String("");
    for (const auto &part : parts) {
      Expr value;
      if (part.capture == TextCapture::Literal)
        value = String(part.literal);
      else if (part.capture == TextCapture::Choice) {
        if (!part.body_error_statements.body.empty() ||
            !part.otherwise_error_statements.body.empty())
          return std::nullopt;
        auto yes = PatternExpression(part.body);
        auto no = PatternExpression(part.otherwise);
        if (part.HasBodyError())
          yes = part.body_error;
        if (part.HasOtherwiseError())
          no = part.otherwise_error;
        if (!yes || !no)
          return std::nullopt;
        value.kind = Expr::Kind::Conditional;
        value.args = {part.predicate, *yes, *no};
      } else if (part.capture == TextCapture::RepeatCalls ||
                 part.capture == TextCapture::RepeatArguments) {
        return std::nullopt;
      } else {
        value = part.value_expression;
        if (value.kind == Expr::Kind::None)
          return std::nullopt;
      }
      result = Binary("~", std::move(result), std::move(value));
    }
    return result;
  }
  Value TransformedPayload(Value value, Expr expression, const Expr &source) {
    value.expression = std::move(expression);
    value.string.reset();
    value.boolean.reset();
    value.defined.reset();
    value.truth.reset();
    value.string_result = false;
    value.append_invariant = false;
    if (!value.expansion.empty()) {
      if (!PatternExpression(value.expansion)) {
        Fail(source.offset, "payload transform over repetition is unsupported");
        return value;
      }
      TextFormatPart transformed;
      transformed.capture = TextCapture::Expression;
      transformed.source = Range(source);
      transformed.value_expression = value.expression;
      transformed.body = std::move(value.expansion);
      value.expansion = {std::move(transformed)};
    }
    return value;
  }
  Value Phi(const Expr &condition, const Value &yes, const Value &no) {
    if (SameExpression(yes.expression, no.expression) &&
        yes.origin == no.origin)
      return yes;
    Value value;
    if (yes.last_expression && no.last_expression &&
        SameExpression(*yes.last_expression, *no.last_expression))
      value.last_expression = yes.last_expression;
    value.origin = yes.origin == no.origin ? yes.origin
                   : IsFuture(yes.origin)  ? yes.origin
                   : IsFuture(no.origin)   ? no.origin
                                           : Origin::Request;
    value.expression.kind = Expr::Kind::Conditional;
    value.expression.args = {condition, yes.expression, no.expression};
    if (yes.defined && no.defined && *yes.defined == *no.defined)
      value.defined = yes.defined;
    if (yes.string && no.string && *yes.string == *no.string)
      value.string = yes.string;
    if (yes.boolean && no.boolean && *yes.boolean == *no.boolean)
      value.boolean = yes.boolean;
    value.string_result = yes.string_result && no.string_result;
    if (Truth(yes) && Truth(no) && *Truth(yes) == *Truth(no))
      value.truth = Truth(yes);
    auto parts = [&](const Value &arm) {
      if (!arm.expansion.empty())
        return arm.expansion;
      Parts result;
      if (arm.string)
        Append(&result, Literal(*arm.string, Range(arm.expression)));
      else if (auto capture = Capture(arm.origin)) {
        TextFormatPart part;
        part.capture = *capture;
        part.value_expression = arm.expression;
        part.source = Range(arm.expression);
        Append(&result, std::move(part));
      }
      return result;
    };
    if (yes.string || no.string || !yes.expansion.empty() ||
        !no.expansion.empty() || Capture(value.origin)) {
      TextFormatPart part;
      part.capture = TextCapture::Choice;
      part.predicate = condition;
      part.body = parts(yes);
      part.otherwise = parts(no);
      value.expansion.push_back(std::move(part));
    }
    return value;
  }
  void JoinBindings(const Expr &condition, const Bindings &yes,
                    const Bindings &no, Bindings *out, bool module = false,
                    size_t join_offset = 0, bool invariant_condition = false) {
    std::set<std::string> names;
    for (const auto &[name, value] : yes)
      names.insert(name);
    for (const auto &[name, value] : no)
      names.insert(name);
    for (const auto &name : names) {
      const auto a = yes.find(name), b = no.find(name);
      if (!Relevant(name, analysis_needed_)) {
        // Macro formals are deliberately absent from global dependencies. An
        // unchanged actual target binding must still survive a local branch;
        // a later field read cannot become an external request for the formal.
        if (a != yes.end() && b != no.end() &&
            a->second.origin == Origin::Message &&
            b->second.origin == Origin::Message &&
            SameExpression(a->second.expression, b->second.expression))
          (*out)[name] = a->second;
        else
          (*out)[name] = {Origin::Request, Name(name)};
        continue;
      }
      const auto *last_a = a != yes.end() && a->second.last_expression
                               ? &*a->second.last_expression
                               : nullptr;
      const auto *last_b = b != no.end() && b->second.last_expression
                               ? &*b->second.last_expression
                               : nullptr;
      if ((last_a || last_b) &&
          (!last_a || !last_b || !SameExpression(*last_a, *last_b)))
        completion_rewrite_diagnostics_[join_offset] =
            "a401 completion cannot qualify divergent last-sensitive local definitions";
      if (a != yes.end() && b != no.end()) {
        if (module && a->second.origin == Origin::Messages &&
            b->second.origin == Origin::Messages &&
            a->second.sequence != b->second.sequence) {
          if (!continuation_mode_ && CountDependent(condition)) {
            Fail(
                join_offset,
                "count-dependent message-sequence mutation has no supported target view");
            continue;
          }
          Value sequence{Origin::Messages, Name(name)};
          const auto key = std::make_pair(join_offset, name);
          if (!message_joins_.count(key))
            message_joins_[key] = ++next_sequence_;
          sequence.sequence = message_joins_.at(key);
          sequence.count_root =
              a->second.count_root == b->second.count_root &&
                      a->second.prefix_drop == b->second.prefix_drop
                  ? a->second.count_root
                  : sequence.sequence;
          sequence.prefix_drop = sequence.count_root == sequence.sequence
                                     ? 0
                                     : a->second.prefix_drop;
          sequence.append_preserving =
              a->second.append_preserving && b->second.append_preserving;
          auto surviving = [](const Value &value) {
            Expr result = Boolean(true);
            for (const auto &guard : value.target_survival)
              result = Binary("and", std::move(result), guard);
            return result;
          };
          Expr survival;
          survival.kind = Expr::Kind::Conditional;
          survival.args = {TargetCounts(condition), surviving(a->second),
                           surviving(b->second)};
          auto retained_survival = survival_counts_.find(key);
          if (retained_survival == survival_counts_.end()) {
            const auto id =
                RegisterCount(std::move(survival), join_offset, true, false);
            FinishTargetCount(id, Name(count_definitions_[id].ordinary));
            retained_survival = survival_counts_.emplace(key, id).first;
          }
          sequence.target_survival.push_back(
              Name(count_definitions_[retained_survival->second].target));
          sequence.defined = true;
          RegisterSequenceCount(sequence.sequence, sequence.count_root,
                                sequence.prefix_drop, sequence.expression,
                                join_offset, true);
          (*out)[name] = std::move(sequence);
          continue;
        }
        const bool invariant_join = invariant_condition &&
                                    a->second.append_invariant &&
                                    b->second.append_invariant;
        if (module && !invariant_join && !Relevant(name, literal_needed_) &&
            (CountDependent(condition) ||
             CountDependent(a->second.expression) ||
             CountDependent(b->second.expression)) &&
            !SameExpression(a->second.expression, b->second.expression)) {
          Fail(
              join_offset,
              "mutable count-derived scalar alias has no supported target definition");
          continue;
        }
        if (module && a->second.origin == Origin::Request &&
            b->second.origin == Origin::Request &&
            !Relevant(name, literal_needed_)) {
          // The original source prelude already owns this runtime phi. Do not
          // recursively substitute its computation into every later read.
          Value reference{Origin::Request, Name(name)};
          reference.append_invariant = invariant_join;
          if (a->second.boolean && b->second.boolean &&
              *a->second.boolean == *b->second.boolean)
            reference.boolean = a->second.boolean;
          if (a->second.defined && b->second.defined &&
              *a->second.defined == *b->second.defined)
            reference.defined = a->second.defined;
          if (Truth(a->second) && Truth(b->second) &&
              *Truth(a->second) == *Truth(b->second))
            reference.truth = Truth(a->second);
          reference.string_result =
              a->second.string_result && b->second.string_result;
          (*out)[name] = std::move(reference);
        } else {
          (*out)[name] = Phi(condition, a->second, b->second);
          (*out)[name].append_invariant = invariant_join;
          (*out)[name].root_index_identity =
              a->second.root_index_identity && b->second.root_index_identity;
          if (module && (*out)[name].origin == Origin::Request)
            (*out)[name].expression = Name(name);
        }
      } else if (a != yes.end())
        (*out)[name] = Phi(condition, a->second, {Origin::Request, Name(name)});
      else
        (*out)[name] = Phi(condition, {Origin::Request, Name(name)}, b->second);
    }
  }
  std::optional<std::string> Access(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Name)
      return expression.text;
    if (expression.kind == Expr::Kind::GetAttr && expression.args.size() == 1) {
      if (auto parent = Access(expression.args[0]))
        return *parent + "." + expression.text;
    }
    if (expression.kind == Expr::Kind::GetItem && expression.args.size() == 2 &&
        expression.args[1].kind == Expr::Kind::String) {
      if (auto parent = Access(expression.args[0]))
        return *parent + "." + expression.args[1].text;
    }
    return std::nullopt;
  }
  void References(const Expr &expression, std::set<std::string> *out) const {
    if (expression.kind == Expr::Kind::Call && !expression.args.empty() &&
        expression.args[0].kind == Expr::Kind::GetAttr) {
      // A method is not another mutable namespace field.
      for (const auto &base : expression.args[0].args)
        References(base, out);
      for (size_t i = 1; i < expression.args.size(); ++i)
        References(expression.args[i], out);
      return;
    }
    if (auto access = Access(expression)) {
      out->insert(*access);
      return;
    }
    for (const auto &child : expression.args)
      References(child, out);
  }
  void NodeReferences(const std::vector<Node> &nodes,
                      std::set<std::string> *out) const {
    for (const auto &node : nodes) {
      References(node.expression, out);
      if (node.predicate)
        References(*node.predicate, out);
      // A retained macro can capture a namespace solely to write it. Its
      // constructor/aliases must survive slicing even without a field read.
      if (node.kind == Node::Kind::Set &&
          node.text.find('.') != std::string::npos)
        out->insert(node.text);
      for (const auto &value : node.defaults)
        References(value, out);
      NodeReferences(node.body, out);
      NodeReferences(node.otherwise, out);
    }
  }
  bool MessageSequence(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Name)
      return message_sequences_.count(expression.text);
    if ((expression.kind == Expr::Kind::Slice ||
         expression.kind == Expr::Kind::Filter) &&
        !expression.args.empty())
      return MessageSequence(expression.args[0]);
    return false;
  }
  std::optional<std::string> TargetRole(const Expr &expression,
                                        std::string_view message) const {
    if (expression.kind == Expr::Kind::String)
      return expression.text;
    if (auto member = MemberName(expression);
        member && *member == "role" && !expression.args.empty() &&
        expression.args[0].kind == Expr::Kind::Name &&
        expression.args[0].text == message)
      return continuation_mode_ ? "user" : "assistant";
    return std::nullopt;
  }
  std::optional<bool> RoleGuard(const Expr &expression,
                                std::string_view message) const {
    if (expression.kind == Expr::Kind::Unary && expression.text == "not" &&
        expression.args.size() == 1) {
      if (auto known = RoleGuard(expression.args[0], message))
        return !*known;
    }
    if (expression.kind != Expr::Kind::Binary || expression.args.size() != 2)
      return std::nullopt;
    if (expression.text == "==" || expression.text == "!=") {
      auto a = TargetRole(expression.args[0], message);
      auto b = TargetRole(expression.args[1], message);
      if (a && b)
        return (*a == *b) == (expression.text == "==");
    }
    auto a = RoleGuard(expression.args[0], message);
    auto b = RoleGuard(expression.args[1], message);
    if (expression.text == "and") {
      if ((a && !*a) || (b && !*b))
        return false;
      if (a && b)
        return *a && *b;
    }
    if (expression.text == "or") {
      if ((a && *a) || (b && *b))
        return true;
      if (a && b)
        return *a || *b;
    }
    return std::nullopt;
  }
  void OutputRoots(const std::vector<Node> &nodes, std::set<std::string> *out,
                   std::string_view message = {},
                   std::set<std::string> *literals = nullptr) const {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Macro)
        continue;
      if (node.kind == Node::Kind::Output) {
        References(node.expression, out);
        if (literals)
          References(node.expression, literals);
      }
      if (node.kind == Node::Kind::If && !message.empty())
        if (auto role = RoleGuard(node.expression, message)) {
          OutputRoots(*role ? node.body : node.otherwise, out, message,
                      literals);
          continue;
        }
      if ((node.kind == Node::Kind::If || node.kind == Node::Kind::For) &&
          (HasOutput(node.body) || HasOutput(node.otherwise)))
        References(node.expression, out);
      if (node.predicate)
        References(*node.predicate, out);
      OutputRoots(node.body, out, message, literals);
      OutputRoots(node.otherwise, out, message, literals);
    }
  }
  void GatherDependencies(const std::vector<Node> &nodes,
                          std::set<std::string> controls,
                          bool in_message_loop) {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Macro) {
        auto &refs = definitions_[node.text];
        NodeReferences(node.body, &refs);
        for (const auto &value : node.defaults)
          References(value, &refs);
        for (const auto &parameter : node.parameters) {
          refs.erase(parameter);
          for (auto it = refs.begin(); it != refs.end();) {
            if (it->compare(0, parameter.size() + 1, parameter + ".") == 0)
              it = refs.erase(it);
            else
              ++it;
          }
        }
        auto &literal = literal_definitions_[node.text];
        std::set<std::string> control_dependencies;
        OutputRoots(node.body, &control_dependencies, {}, &literal);
        for (const auto &parameter : node.parameters) {
          literal.erase(parameter);
          for (auto it = literal.begin(); it != literal.end();) {
            if (it->compare(0, parameter.size() + 1, parameter + ".") == 0)
              it = literal.erase(it);
            else
              ++it;
          }
        }
        continue;
      }
      if (node.kind == Node::Kind::Capture) {
        auto &refs = definitions_[node.text];
        NodeReferences(node.body, &refs);
        refs.insert(controls.begin(), controls.end());
        OutputRoots(node.body, &literal_definitions_[node.text]);
      }
      if (node.kind == Node::Kind::Set) {
        auto &refs = definitions_[node.text];
        References(node.expression, &refs);
        References(node.expression, &literal_definitions_[node.text]);
        refs.insert(controls.begin(), controls.end());
        if (MessageSequence(node.expression))
          message_sequences_.insert(node.text);
        if (node.expression.kind == Expr::Kind::Call &&
            !node.expression.args.empty() &&
            node.expression.args[0].kind == Expr::Kind::Name &&
            node.expression.args[0].text == "namespace") {
          namespace_roots_.insert(node.text);
          for (size_t i = 1; i < node.expression.args.size(); ++i)
            if (i < node.expression.keywords.size() &&
                !node.expression.keywords[i].empty()) {
              auto &field =
                  definitions_[node.text + "." + node.expression.keywords[i]];
              References(node.expression.args[i], &field);
              References(node.expression.args[i],
                         &literal_definitions_[node.text + "." +
                                               node.expression.keywords[i]]);
              field.insert(controls.begin(), controls.end());
            }
        }
      }
      auto nested_controls = controls;
      if (node.kind == Node::Kind::If || node.kind == Node::Kind::For)
        References(node.expression, &nested_controls);
      if (node.predicate)
        References(*node.predicate, &nested_controls);
      const bool message_loop =
          node.kind == Node::Kind::For && MessageSequence(node.expression);
      if (node.kind == Node::Kind::For) {
        const auto comma = node.text.find(',');
        for (const auto &name :
             {node.text.substr(0, comma), comma == std::string::npos
                                              ? std::string()
                                              : node.text.substr(comma + 1)})
          if (!name.empty()) {
            auto &refs = definitions_[name];
            References(node.expression, &refs);
            References(node.expression, &literal_definitions_[name]);
            refs.insert(controls.begin(), controls.end());
          }
        auto &loop = definitions_["loop"];
        References(node.expression, &loop);
      }
      if (message_loop ||
          (node.kind == Node::Kind::If && MentionsGeneration(node.expression) &&
           (HasOutput(node.body) || HasOutput(node.otherwise)))) {
        OutputRoots(node.body, &analysis_roots_, message_loop ? node.text : "",
                    &literal_roots_);
        OutputRoots(node.otherwise, &analysis_roots_,
                    message_loop ? node.text : "", &literal_roots_);
        References(node.expression, &analysis_roots_);
      }
      GatherDependencies(node.body, nested_controls,
                         in_message_loop || message_loop);
      GatherDependencies(node.otherwise, nested_controls,
                         in_message_loop || message_loop);
    }
  }
  std::set<std::string> NeededClosure(std::set<std::string> pending,
                                      bool literals_only = false) const {
    std::set<std::string> result;
    while (!pending.empty()) {
      auto name = *pending.begin();
      pending.erase(pending.begin());
      if (!result.insert(name).second)
        continue;
      const auto &definitions =
          literals_only ? literal_definitions_ : definitions_;
      if (const auto found = definitions.find(name); found != definitions.end())
        pending.insert(found->second.begin(), found->second.end());
      if (const auto snapshot = selector_binding_expressions_.find(name);
          snapshot != selector_binding_expressions_.end())
        References(snapshot->second, &pending);
      const auto dot = name.find('.');
      if (dot != std::string::npos &&
          !namespace_roots_.count(name.substr(0, dot)))
        pending.insert(name.substr(0, dot));
    }
    return result;
  }
  void SelectorEffectRoots(const std::vector<Node> &nodes,
                           const std::set<std::string> &needed,
                           std::set<std::string> *roots,
                           std::set<std::string> controls = {}) {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Macro)
        continue;
      if (node.kind == Node::Kind::Capture || node.predicate) {
        // Buffered computations execute even when their text is unused. Keep
        // their transitive macro/scalar inputs and source error paths alive.
        NodeReferences(node.body, roots);
        NodeReferences(node.otherwise, roots);
        References(node.expression, roots);
        if (node.predicate)
          References(*node.predicate, roots);
        roots->insert(controls.begin(), controls.end());
      }
      if (node.kind == Node::Kind::Output) {
        const auto &e = node.expression;
        bool retain = false;
        for (const auto &field : needed) {
          const auto dot = field.find('.');
          if (retain || dot == std::string::npos ||
              !namespace_roots_.count(field.substr(0, dot)))
            continue;
          Path caller;
          retain = MacroExpressionMayWrite(e, field, caller, {});
        }
        if (retain) {
          selector_effect_outputs_.insert(node.offset);
          References(e, roots);
          roots->insert(controls.begin(), controls.end());
        }
      }
      auto nested = controls;
      if (node.kind == Node::Kind::If || node.kind == Node::Kind::For)
        References(node.expression, &nested);
      if (node.predicate)
        References(*node.predicate, &nested);
      SelectorEffectRoots(node.body, needed, roots, nested);
      SelectorEffectRoots(node.otherwise, needed, roots, nested);
    }
  }
  std::set<std::string> SelectorDependencies(std::set<std::string> roots) {
    // Reuse the existing source write query and ordinary dependency closure.
    // A retained call may read another field whose earlier writer is needed.
    for (;;) {
      auto needed = NeededClosure(roots);
      const auto prior_roots = roots.size();
      SelectorEffectRoots(ast_.body, needed, &roots);
      if (roots.size() == prior_roots)
        return needed;
    }
  }
  void CloseNamespaceAliasDependencies(const std::vector<Node> &nodes) {
    std::map<std::string, std::set<std::string>> aliases;
    auto gather = [&](auto &&self, const std::vector<Node> &body) -> void {
      for (const auto &node : body) {
        if (node.kind == Node::Kind::Set &&
            node.text.find('.') == std::string::npos &&
            node.expression.kind == Expr::Kind::Name) {
          aliases[node.text].insert(node.expression.text);
          aliases[node.expression.text].insert(node.text);
        }
        self(self, node.body);
        self(self, node.otherwise);
      }
    };
    gather(gather, nodes);
    std::set<std::string> pending = namespace_roots_;
    while (!pending.empty()) {
      const auto root = *pending.begin();
      pending.erase(pending.begin());
      for (const auto &alias : aliases[root])
        if (namespace_roots_.insert(alias).second)
          pending.insert(alias);
    }
    std::vector<std::string> fields;
    for (const auto &[name, unused] : definitions_) {
      const auto dot = name.find('.');
      if (dot != std::string::npos &&
          namespace_roots_.count(name.substr(0, dot)))
        fields.push_back(name);
    }
    for (const auto &field : fields) {
      const auto dot = field.find('.');
      const auto root = field.substr(0, dot);
      std::set<std::string> reached{root}, work{root};
      definitions_[field].insert(root);
      while (!work.empty()) {
        const auto current = *work.begin();
        work.erase(work.begin());
        for (const auto &alias : aliases[current])
          if (reached.insert(alias).second)
            work.insert(alias);
      }
      for (const auto &alias : reached) {
        if (alias == root)
          continue;
        const auto shared_field = alias + field.substr(dot);
        definitions_[shared_field].insert(field);
        definitions_[field].insert(shared_field);
        literal_definitions_[shared_field].insert(field);
        literal_definitions_[field].insert(shared_field);
      }
    }
  }
  bool Relevant(const std::string &name,
                const std::set<std::string> &needed) const {
    if (needed.count(name))
      return true;
    if (namespace_roots_.count(name)) {
      const auto field = needed.lower_bound(name + ".");
      return field != needed.end() &&
             field->compare(0, name.size() + 1, name + ".") == 0;
    }
    return false;
  }
  bool RawArguments(const Expr &expression) const {
    const auto path = Access(expression);
    return path && *path == worker_call_ + ".function.arguments";
  }
  bool ArgumentTypeGuard(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Test && !expression.args.empty() &&
        RawArguments(expression.args[0]))
      return true;
    for (const auto &child : expression.args)
      if (ArgumentTypeGuard(child))
        return true;
    return false;
  }
  bool DecodedArgumentMember(const Expr &expression) const {
    if ((expression.kind == Expr::Kind::GetAttr ||
         expression.kind == Expr::Kind::GetItem) &&
        !expression.args.empty() && RawArguments(expression.args[0]))
      return true;
    for (const auto &child : expression.args)
      if (DecodedArgumentMember(child))
        return true;
    return false;
  }
  bool NegativeMappingGuard(const Expr &expression,
                            bool negative = false) const {
    if (expression.kind == Expr::Kind::Unary && expression.text == "not" &&
        expression.args.size() == 1)
      return NegativeMappingGuard(expression.args[0], !negative);
    if (expression.kind == Expr::Kind::Test && expression.text == "mapping" &&
        !expression.args.empty() && RawArguments(expression.args[0]))
      return negative;
    // Only a conjunct actually constrains the rejected path. An OR arm alone
    // cannot establish a source mapping invariant.
    if (expression.kind == Expr::Kind::Binary && expression.args.size() == 2 &&
        expression.text == (negative ? "or" : "and"))
      return NegativeMappingGuard(expression.args[0], negative) ||
             NegativeMappingGuard(expression.args[1], negative);
    return false;
  }
  bool ContainsConjunct(const Expr &expression, const Expr &required) const {
    if (required.kind == Expr::Kind::Boolean &&
        (required.text == "true" || required.text == "True"))
      return true;
    if (SameExpression(expression, required))
      return true;
    if (expression.kind == Expr::Kind::Binary && expression.text == "and" &&
        expression.args.size() == 2)
      return ContainsConjunct(expression.args[0], required) ||
             ContainsConjunct(expression.args[1], required);
    return false;
  }
  bool ImpliesConjunction(const Expr &expression, const Expr &required) const {
    if (required.kind == Expr::Kind::Binary && required.text == "and" &&
        required.args.size() == 2)
      return ImpliesConjunction(expression, required.args[0]) &&
             ImpliesConjunction(expression, required.args[1]);
    return ContainsConjunct(expression, required);
  }
  Expr SimplifyBoolean(Expr expression) const {
    for (auto &child : expression.args)
      child = SimplifyBoolean(std::move(child));
    auto boolean = [](const Expr &value) -> std::optional<bool> {
      if (value.kind != Expr::Kind::Boolean)
        return std::nullopt;
      return value.text == "true" || value.text == "True";
    };
    if (expression.kind == Expr::Kind::Unary && expression.text == "not" &&
        expression.args.size() == 1) {
      if (auto value = boolean(expression.args[0]))
        return Boolean(!*value);
    }
    if (expression.kind == Expr::Kind::Binary && expression.args.size() == 2 &&
        (expression.text == "and" || expression.text == "or")) {
      const bool conjunction = expression.text == "and";
      if (SameExpression(expression.args[0], expression.args[1]))
        return std::move(expression.args[0]);
      for (size_t i = 0; i < 2; ++i)
        if (expression.args[i].kind == Expr::Kind::Unary &&
            expression.args[i].text == "not" &&
            expression.args[i].args.size() == 1 &&
            SameExpression(expression.args[i].args[0], expression.args[1 - i]))
          return Boolean(!conjunction);
      for (size_t i = 0; i < 2; ++i)
        if (auto value = boolean(expression.args[i])) {
          if (*value != conjunction)
            return Boolean(*value);
          return std::move(expression.args[1 - i]);
        }
    }
    return expression;
  }
  Expr RepresentationWorld(const Expr &expression, bool mapping) const {
    if (expression.kind == Expr::Kind::Test && !expression.args.empty() &&
        RawArguments(expression.args[0])) {
      const auto mask = TestMask(expression.text);
      if (mask != 127)
        return Boolean((mask & (mapping ? 32 : 4)) != 0);
    }
    if (mapping && expression.kind == Expr::Kind::Binary &&
        expression.args.size() == 2 &&
        (expression.text == "==" || expression.text == "!="))
      for (size_t i = 0; i < 2; ++i)
        if (RawArguments(expression.args[i]) &&
            expression.args[1 - i].kind == Expr::Kind::String)
          return Boolean(expression.text == "!=");
    Expr result = expression;
    for (auto &child : result.args)
      child = RepresentationWorld(child, mapping);
    return SimplifyBoolean(std::move(result));
  }
  void RecordMappingDemand(Expr predicate, Expr raw_mapping) {
    if (DecodedArgumentMember(predicate) ||
        DecodedArgumentMember(raw_mapping)) {
      for (const auto &mandatory : projection_rules_[0])
        if (ImpliesConjunction(predicate, mandatory) &&
            ImpliesConjunction(raw_mapping, mandatory))
          return;
      Fail(predicate.offset,
           "argument projection guard depends on a decoded argument member");
      return;
    }
    projection_raw_mapping_rules_.push_back(std::move(raw_mapping));
    auto &rules = projection_rules_[0];
    if (std::none_of(rules.begin(), rules.end(), [&](const Expr &prior) {
          return SameExpression(prior, predicate);
        }))
      rules.push_back(std::move(predicate));
  }
  void ObserveProjection(size_t feature, Path &path) {
    if (feature == 0)
      path.adaptation.arguments_need_mapping = true;
    if (feature == 1)
      path.adaptation.arguments_observe_string = true;
    if (feature == 2)
      path.adaptation.content_observes_representation = true;
    if (!projection_mode_)
      return;
    Expr predicate = Conjunction(path.guards);
    if (feature < 2)
      predicate = RepresentationWorld(predicate, feature == 0);
    predicate = SimplifyBoolean(std::move(predicate));
    if (feature == 0) {
      RecordMappingDemand(std::move(predicate),
                          RepresentationWorld(Conjunction(path.guards), false));
      return;
    }
    auto &rules = projection_rules_[feature];
    if (std::none_of(rules.begin(), rules.end(), [&](const Expr &prior) {
          return SameExpression(prior, predicate);
        }))
      rules.push_back(std::move(predicate));
  }
  Expr ProjectionPredicate(size_t feature) const {
    Expr result = Boolean(false);
    for (const auto &rule : projection_rules_[feature])
      result = Binary("or", std::move(result), rule);
    return SimplifyBoolean(std::move(result));
  }
  Expr GuardedBoolean(const Expr &expression,
                      const std::vector<Expr> &guards) const {
    for (const auto &guard : guards) {
      if (SameExpression(expression, guard))
        return Boolean(true);
      if (guard.kind == Expr::Kind::Unary && guard.text == "not" &&
          guard.args.size() == 1 && SameExpression(expression, guard.args[0]))
        return Boolean(false);
    }
    Expr result = expression;
    for (auto &child : result.args)
      child = GuardedBoolean(child, guards);
    return SimplifyBoolean(std::move(result));
  }
  Node ProjectionTree(size_t feature, const std::vector<Expr> &predicates,
                      TextAdaptation facts, std::vector<Expr> path) {
    if (feature == predicates.size()) {
      // Explicit STRING handling wins over the mapping-only representation.
      if (facts.arguments_observe_string)
        facts.arguments_need_mapping = false;
      const int64_t mask =
          (facts.arguments_need_mapping ? kTextAdaptationMappingArguments : 0) |
          (facts.arguments_observe_string ? kTextAdaptationArgumentStrings
                                          : 0) |
          (facts.content_observes_representation
               ? kTextAdaptationOriginalContent
               : 0);
      return Selection(static_cast<size_t>(mask), TextStartupMode::Content);
    }
    const Expr predicate = GuardedBoolean(predicates[feature], path);
    auto set = [&](TextAdaptation *value, bool enabled) {
      if (feature == 0)
        value->arguments_need_mapping = enabled;
      if (feature == 1)
        value->arguments_observe_string = enabled;
      if (feature == 2)
        value->content_observes_representation = enabled;
    };
    if (predicate.kind == Expr::Kind::Boolean) {
      set(&facts, predicate.text == "true" || predicate.text == "True");
      return ProjectionTree(feature + 1, predicates, facts, std::move(path));
    }
    Node branch;
    branch.kind = Node::Kind::If;
    branch.expression = predicate;
    TextAdaptation yes = facts, no = facts;
    set(&yes, true);
    set(&no, false);
    auto otherwise = path;
    path.push_back(predicate);
    otherwise.push_back(Unary("not", predicate));
    branch.body.push_back(
        ProjectionTree(feature + 1, predicates, yes, std::move(path)));
    branch.otherwise.push_back(
        ProjectionTree(feature + 1, predicates, no, std::move(otherwise)));
    return branch;
  }
  std::optional<int64_t> GuardInteger(const Expr &expression) const {
    if (expression.kind != Expr::Kind::Integer)
      return std::nullopt;
    int64_t value = 0;
    const auto result =
        std::from_chars(expression.text.data(),
                        expression.text.data() + expression.text.size(), value);
    if (result.ec != std::errc{} ||
        result.ptr != expression.text.data() + expression.text.size())
      return std::nullopt;
    return value;
  }
  void EmptySequenceFacts(const Expr &guard, std::vector<Expr> *facts) const {
    if (guard.kind == Expr::Kind::Binary && guard.text == "and" &&
        guard.args.size() == 2) {
      EmptySequenceFacts(guard.args[0], facts);
      EmptySequenceFacts(guard.args[1], facts);
    }
    if (guard.kind != Expr::Kind::Binary || guard.text != "==" ||
        guard.args.size() != 2)
      return;
    for (size_t i = 0; i < 2; ++i)
      if (const auto value = GuardInteger(guard.args[i]);
          value && *value == 0 &&
          ((guard.args[1 - i].kind == Expr::Kind::Filter &&
            guard.args[1 - i].text == "length") ||
           CountDependent(guard.args[1 - i])))
        facts->push_back(guard.args[1 - i]);
  }
  Expr WithEmptySequences(const Expr &expression,
                          const std::vector<Expr> &facts) const {
    for (const auto &fact : facts)
      if (SameExpression(expression, fact))
        return Integer(0);
    Expr result = expression;
    for (auto &child : result.args)
      child = WithEmptySequences(child, facts);
    if (result.kind == Expr::Kind::Binary && result.args.size() == 2) {
      const auto a = GuardInteger(result.args[0]),
                 b = GuardInteger(result.args[1]);
      // Only the nonnegative ordinal parity used by source role validation
      // is needed here; no general source expression evaluator is introduced.
      if (a && b && result.text == "%" && *a >= 0 && *b > 0)
        return Integer(*a % *b);
      if (a && b && (result.text == "==" || result.text == "!="))
        return Boolean((*a == *b) == (result.text == "=="));
      if (result.args[0].kind == Expr::Kind::Boolean &&
          result.args[1].kind == Expr::Kind::Boolean &&
          (result.text == "==" || result.text == "!="))
        return Boolean((result.args[0].text == result.args[1].text) ==
                       (result.text == "=="));
    }
    return SimplifyBoolean(std::move(result));
  }
  bool AddGuard(std::vector<Expr> *guards, const Expr &guard) const {
    if (guard.kind == Expr::Kind::Boolean)
      return guard.text == "true" || guard.text == "True";
    for (const auto &prior : *guards) {
      if (SameExpression(prior, guard))
        return true;
      if (prior.kind == Expr::Kind::Unary && prior.text == "not" &&
          prior.args.size() == 1 && SameExpression(prior.args[0], guard))
        return false;
      if (guard.kind == Expr::Kind::Unary && guard.text == "not" &&
          guard.args.size() == 1 && SameExpression(guard.args[0], prior))
        return false;
    }
    std::vector<Expr> empty_sequences;
    EmptySequenceFacts(guard, &empty_sequences);
    for (const auto &prior : *guards)
      EmptySequenceFacts(prior, &empty_sequences);
    if (!empty_sequences.empty()) {
      auto contradicted = [&](const Expr &candidate) {
        const auto value = WithEmptySequences(candidate, empty_sequences);
        return value.kind == Expr::Kind::Boolean && value.text == "false";
      };
      if (contradicted(guard) ||
          std::any_of(guards->begin(), guards->end(), contradicted))
        return false;
    }
    guards->push_back(guard);
    return true;
  }
  void RecordRetentionChoice(size_t offset, const Value &condition) {
    if (!condition.retention || projection_mode_ ||
        ambiguous_retention_.count(offset))
      return;
    const auto [entry, inserted] =
        retention_choices_.emplace(offset, *condition.retention);
    if (!inserted && entry->second != *condition.retention) {
      retention_choices_.erase(entry);
      ambiguous_retention_.insert(offset);
    }
  }
  std::vector<Expr> RuntimeGuards(const Path &path) const {
    std::vector<Expr> result;
    for (const auto &guard : path.guards)
      if (std::none_of(
              path.proof_guards.begin(), path.proof_guards.end(),
              [&](const Expr &proof) { return SameExpression(guard, proof); }))
        result.push_back(guard);
    return result;
  }
  void RecursiveEffects(const TextFormatPart &part,
                        std::vector<const Expr *> *effects) const {
    if (part.capture == TextCapture::Expression) {
      std::vector<const Expr *> calls;
      GatherCalls(part.value_expression, &calls);
      if (std::any_of(calls.begin(), calls.end(), [&](const auto *call) {
            return recursive_macros_.count(call->args[0].text);
          }))
        effects->push_back(&part.value_expression);
    }
    for (const auto &child : part.body)
      RecursiveEffects(child, effects);
    for (const auto &child : part.otherwise)
      RecursiveEffects(child, effects);
  }
  bool PriorRecursiveEffect(const std::vector<const Expr *> &effects,
                            const Expr &predicate) {
    std::set<std::string> fields;
    References(predicate, &fields);
    for (const auto *expression : effects) {
      for (const auto &field : fields) {
        const auto dot = field.find('.');
        if (dot == std::string::npos ||
            !namespace_roots_.count(field.substr(0, dot)))
          continue;
        Path caller;
        if (MacroExpressionMayWrite(*expression, field, caller, {}))
          return true;
      }
    }
    return false;
  }
  bool SplitRequestChoice(const Parts &input, Parts *yes, Parts *no,
                          Expr *condition, bool *yes_error, bool *no_error,
                          size_t *choice_offset, bool preserve_validation,
                          const std::vector<std::pair<size_t, Expr>> &retained,
                          bool *validation_only, bool unconditional,
                          std::vector<const Expr *> prior_effects = {}) {
    for (size_t index = 0; index < input.size(); ++index) {
      const auto &part = input[index];
      // A post-call source validation observes the prospective assistant's
      // effects. Keep it after that call in the recipe; LIVE history has not
      // executed this future statement and cannot supply its predicate value.
      const bool after_effect =
          preserve_validation && part.capture == TextCapture::Choice &&
          ((part.body_error.kind == Expr::Kind::None) !=
           (part.otherwise_error.kind == Expr::Kind::None)) &&
          PriorRecursiveEffect(prior_effects, part.predicate);
      if (part.capture == TextCapture::Choice && !Future(part.predicate) &&
          !after_effect) {
        std::string bytes;
        const bool validation =
            preserve_validation &&
            ((!part.HasBodyError()) != (!part.HasOtherwiseError())) &&
            SuccessfulLiteral(part, &bytes) && bytes.empty();
        // Future-controlled branches and zero-trip repeats do not establish
        // unconditional assistant admission. Retain their validation locally.
        if (validation && !unconditional)
          continue;
        if (validation &&
            std::any_of(retained.begin(), retained.end(),
                        [&](const auto &prior) {
                          return prior.first == part.source.begin &&
                                 SameExpression(prior.second, part.predicate);
                        }))
          continue;
        *condition = part.predicate;
        *choice_offset = part.source.begin;
        *validation_only = validation;
        *yes_error = part.HasBodyError();
        *no_error = part.HasOtherwiseError();
        if (validation) {
          *yes = input;
          *no = input;
          return true;
        }
        yes->assign(input.begin(), input.begin() + index);
        no->assign(input.begin(), input.begin() + index);
        yes->insert(yes->end(), part.body.begin(), part.body.end());
        no->insert(no->end(), part.otherwise.begin(), part.otherwise.end());
        yes->insert(yes->end(), input.begin() + index + 1, input.end());
        no->insert(no->end(), input.begin() + index + 1, input.end());
        return true;
      }
      Parts body_yes, body_no;
      const bool always =
          unconditional &&
          !(part.capture == TextCapture::Choice && Future(part.predicate)) &&
          part.capture != TextCapture::RepeatCalls &&
          part.capture != TextCapture::RepeatArguments;
      if (SplitRequestChoice(part.body, &body_yes, &body_no, condition,
                             yes_error, no_error, choice_offset,
                             preserve_validation, retained, validation_only,
                             always, prior_effects)) {
        *yes = input;
        *no = input;
        (*yes)[index].body = std::move(body_yes);
        (*no)[index].body = std::move(body_no);
        return true;
      }
      if (SplitRequestChoice(part.otherwise, &body_yes, &body_no, condition,
                             yes_error, no_error, choice_offset,
                             preserve_validation, retained, validation_only,
                             always, prior_effects)) {
        *yes = input;
        *no = input;
        (*yes)[index].otherwise = std::move(body_yes);
        (*no)[index].otherwise = std::move(body_no);
        return true;
      }
      RecursiveEffects(part, &prior_effects);
    }
    return false;
  }
  std::vector<Path> FormatChoices(const std::vector<Path> &paths,
                                  bool preserve_validation = false) {
    std::vector<Path> pending = paths, result;
    while (!pending.empty()) {
      Path current = std::move(pending.back());
      pending.pop_back();
      Parts yes, no;
      Expr condition;
      bool yes_error = false, no_error = false, validation_only = false;
      size_t choice_offset = 0;
      if (!SplitRequestChoice(current.parts, &yes, &no, &condition, &yes_error,
                              &no_error, &choice_offset, preserve_validation,
                              current.retained_validations, &validation_only,
                              true)) {
        result.push_back(std::move(current));
        continue;
      }
      Path other = current;
      const Expr negated = Unary("not", condition);
      const auto routing = RuntimeGuards(current);
      const bool yes_possible =
          !yes_error && AddGuard(&current.guards, condition);
      const bool no_possible = !no_error && AddGuard(&other.guards, negated);
      if (validation_only) {
        current.retained_validations.emplace_back(choice_offset, condition);
        other.retained_validations.emplace_back(choice_offset, condition);
        // A redundant inner validation cannot erase a real outer framing
        // decision that already uses the same predicate.
        const auto already_routes = [&](const Expr &guard) {
          return std::any_of(
              routing.begin(), routing.end(),
              [&](const Expr &prior) { return SameExpression(prior, guard); });
        };
        if (!already_routes(condition))
          current.proof_guards.push_back(condition);
        if (!already_routes(negated))
          other.proof_guards.push_back(negated);
      } else if (preserve_validation && yes_possible != no_possible &&
                 !current.proof_guards.empty()) {
        // If the other framing arm is excluded only by completed-target
        // validation, its surviving arm is an inference fact, not input
        // admission. Genuine request/style branches remain runtime guards.
        auto without_validation = routing;
        if (yes_possible && !no_error && AddGuard(&without_validation, negated))
          current.proof_guards.push_back(condition);
        if (no_possible && !yes_error &&
            AddGuard(&without_validation, condition))
          other.proof_guards.push_back(negated);
      }
      if (const auto retention = retention_choices_.find(choice_offset);
          retention != retention_choices_.end()) {
        current.retention[choice_offset] = {
            retention->second.first, true, retention->second.second, condition};
        other.retention[choice_offset] = {retention->second.first, false,
                                          !retention->second.second, condition};
      }
      if (yes_possible) {
        current.parts = std::move(yes);
        pending.push_back(std::move(current));
      }
      if (no_possible) {
        other.parts = std::move(no);
        pending.push_back(std::move(other));
      }
    }
    return result;
  }
  bool Fail(size_t offset, std::string message) {
    if (!failed_ && diagnostic_) {
      diagnostic_->offset = offset;
      diagnostic_->message = std::move(message);
      LocateDiagnostic(source_, source_name_, diagnostic_);
    }
    failed_ = true;
    return false;
  }
  struct HistorySearch {
    std::string root;
    std::string index;
    std::string flag;
    Expr sequence;
    bool initial_flag = true;
    const Node *index_write = nullptr;
    const Node *flag_write = nullptr;
    bool requires_match = false;
    size_t loop_end = 0;
  };
  bool IntegerIs(const Expr &expression, int value) const {
    if (expression.kind == Expr::Kind::Integer)
      return expression.text == std::to_string(value);
    return value < 0 && expression.kind == Expr::Kind::Unary &&
           expression.text == "-" && expression.args.size() == 1 &&
           IntegerIs(expression.args[0], -value);
  }
  bool BooleanIs(const Expr &expression, bool value) const {
    return expression.kind == Expr::Kind::Boolean &&
           expression.text == (value ? "true" : "false");
  }
  bool LastIndex(const Expr &expression, Expr *sequence) const {
    if (expression.kind != Expr::Kind::Binary || expression.text != "-" ||
        expression.args.size() != 2 || !IntegerIs(expression.args[1], 1))
      return false;
    const auto &count = expression.args[0];
    if (count.kind != Expr::Kind::Filter || count.text != "length" ||
        count.args.size() != 1)
      return false;
    *sequence = count.args[0];
    return true;
  }
  Expr SearchFlag(const HistorySearch &search) const {
    Expr flag;
    flag.kind = Expr::Kind::GetAttr;
    flag.text = search.flag;
    flag.args.push_back(Name(search.root));
    return search.initial_flag ? flag : Unary("not", std::move(flag));
  }
  bool HistoricalOrdinal(const Expr &expression, const Expr &sequence,
                         bool reversed,
                         const std::map<std::string, Expr> &aliases,
                         std::set<std::string> seen = {}) const {
    if (expression.kind == Expr::Kind::Name) {
      const auto alias = aliases.find(expression.text);
      return alias != aliases.end() && seen.insert(expression.text).second &&
             HistoricalOrdinal(alias->second, sequence, reversed, aliases,
                               std::move(seen));
    }
    const auto access = Access(expression);
    if (access && *access == "loop.index0")
      return true;
    if (expression.kind == Expr::Kind::Binary && expression.text == "-" &&
        expression.args.size() == 2) {
      Expr anchor;
      const auto ordinal = Access(expression.args[1]);
      return LastIndex(expression.args[0], &anchor) &&
             SameExpression(anchor, sequence) && ordinal &&
             *ordinal == "loop.index0";
    }
    return false;
  }
  Expr OrdinalExpression(const Expr &expression,
                         const std::map<std::string, Expr> &aliases,
                         std::set<std::string> seen = {}) const {
    if (expression.kind == Expr::Kind::Name) {
      const auto alias = aliases.find(expression.text);
      if (alias != aliases.end() && seen.insert(expression.text).second)
        return OrdinalExpression(alias->second, aliases, std::move(seen));
    }
    Expr result = expression;
    for (auto &argument : result.args)
      argument = OrdinalExpression(argument, aliases, seen);
    return result;
  }
  bool NamespaceEscapes(const Expr &expression, std::string_view root) const {
    if (expression.kind == Expr::Kind::Name)
      return expression.text == root;
    for (size_t i = 0; i < expression.args.size(); ++i) {
      const bool field_base =
          i == 0 && (expression.kind == Expr::Kind::GetAttr ||
                     expression.kind == Expr::Kind::GetItem);
      if (field_base && expression.args[i].kind == Expr::Kind::Name &&
          expression.args[i].text == root)
        continue;
      if (NamespaceEscapes(expression.args[i], root))
        return true;
    }
    return false;
  }
  bool NamespaceEscapes(const std::vector<Node> &nodes,
                        std::string_view root) const {
    for (const auto &node : nodes) {
      if ((node.kind == Node::Kind::For && node.text == root) ||
          std::find(node.parameters.begin(), node.parameters.end(), root) !=
              node.parameters.end() ||
          NamespaceEscapes(node.expression, root) ||
          (node.predicate && NamespaceEscapes(*node.predicate, root)))
        return true;
      for (const auto &value : node.defaults)
        if (NamespaceEscapes(value, root))
          return true;
      if (NamespaceEscapes(node.body, root) ||
          NamespaceEscapes(node.otherwise, root))
        return true;
    }
    return false;
  }
  bool MacroExpressionMayWrite(const Expr &expression, std::string_view field,
                               Path &state, std::set<std::string> active) {
    if (expression.kind == Expr::Kind::Call && !expression.args.empty() &&
        expression.args[0].kind == Expr::Kind::Name &&
        macros_.count(expression.args[0].text) &&
        MacroMayWriteField(expression, field, state, active))
      return true;
    if (expression.kind == Expr::Kind::Binary && expression.args.size() == 2 &&
        (expression.text == "and" || expression.text == "or")) {
      if (MacroExpressionMayWrite(expression.args[0], field, state, active))
        return true;
      if (const auto truth = Truth(Resolve(expression.args[0], state, {}));
          truth && (*truth == (expression.text == "or")))
        return false;
      return MacroExpressionMayWrite(expression.args[1], field, state, active);
    }
    if (expression.kind == Expr::Kind::Conditional &&
        expression.args.size() == 3) {
      if (MacroExpressionMayWrite(expression.args[0], field, state, active))
        return true;
      if (const auto truth = Truth(Resolve(expression.args[0], state, {})))
        return MacroExpressionMayWrite(expression.args[*truth ? 1 : 2], field,
                                       state, active);
    }
    for (const auto &argument : expression.args)
      if (MacroExpressionMayWrite(argument, field, state, active))
        return true;
    return false;
  }
  void ForgetMacroAssignments(const std::vector<Node> &nodes, Path *state) {
    for (const auto &node : nodes) {
      if ((node.kind == Node::Kind::Set || node.kind == Node::Kind::Capture) &&
          node.text.find('.') == std::string::npos)
        state->bindings[node.text] = {Origin::Request, Name(node.text)};
      ForgetMacroAssignments(node.body, state);
      ForgetMacroAssignments(node.otherwise, state);
    }
  }
  bool MacroBlockMayWrite(const std::vector<Node> &nodes,
                          std::string_view field, Path *state,
                          const std::set<std::string> &active) {
    for (const auto &node : nodes) {
      if (MacroExpressionMayWrite(node.expression, field, *state, active))
        return true;
      if (node.predicate &&
          MacroExpressionMayWrite(*node.predicate, field, *state, active))
        return true;
      if (node.kind == Node::Kind::Capture) {
        Path captured = *state;
        if (MacroBlockMayWrite(node.body, field, &captured, active))
          return true;
        state->bindings[node.text] = {Origin::Request, Name(node.text)};
        continue;
      }
      if (node.kind == Node::Kind::Set) {
        const auto dot = node.text.find('.');
        if (dot == std::string::npos) {
          state->bindings[node.text] = Resolve(node.expression, *state, {});
        } else if (node.text.substr(dot) == field.substr(field.find('.'))) {
          const auto root = node.text.substr(0, dot);
          const auto binding = state->bindings.find(root);
          if (binding != state->bindings.end()) {
            const auto &value = binding->second.expression;
            const bool private_namespace =
                value.kind == Expr::Kind::Call && !value.args.empty() &&
                value.args[0].kind == Expr::Kind::Name &&
                value.args[0].text == "namespace" &&
                !macros_.count("namespace");
            if (!private_namespace)
              return true;
          } else if (namespace_roots_.count(root)) {
            // Free namespace roots may alias. Unknown effects diagnose rather
            // than selecting a history branch from stale LIVE state.
            return true;
          }
        }
      } else if (node.kind == Node::Kind::If) {
        if (const auto truth = Truth(Resolve(node.expression, *state, {}))) {
          if (MacroBlockMayWrite(*truth ? node.body : node.otherwise, field,
                                 state, active))
            return true;
        } else {
          Path yes = *state, no = *state;
          if (MacroBlockMayWrite(node.body, field, &yes, active) ||
              MacroBlockMayWrite(node.otherwise, field, &no, active))
            return true;
          ForgetMacroAssignments(node.body, state);
          ForgetMacroAssignments(node.otherwise, state);
        }
      } else if (node.kind == Node::Kind::For) {
        Path repeated = *state;
        const auto comma = node.text.find(',');
        for (const auto &name :
             {node.text.substr(0, comma), comma == std::string::npos
                                              ? std::string()
                                              : node.text.substr(comma + 1)})
          if (!name.empty())
            repeated.bindings[name] = {Origin::Request, Name(name)};
        repeated.bindings["loop"] = {Origin::Request, Name("loop")};
        if (MacroBlockMayWrite(node.body, field, &repeated, active))
          return true;
        ForgetMacroAssignments(node.body, state);
      } else if (node.kind == Node::Kind::Generation) {
        if (MacroBlockMayWrite(node.body, field, state, active))
          return true;
      }
    }
    return false;
  }
  bool MacroMayWriteField(const Expr &call, std::string_view field,
                          Path &caller, std::set<std::string> active) {
    const auto macro = macros_.find(call.args[0].text);
    if (macro == macros_.end())
      return false;
    if (!active.insert(macro->first).second)
      return true;
    const auto &definition = *macro->second;
    Path parameters;
    std::vector<bool> supplied(definition.parameters.size(), false);
    size_t positional = 0;
    for (size_t argument = 1; argument < call.args.size(); ++argument) {
      size_t parameter = positional++;
      if (argument < call.keywords.size() && !call.keywords[argument].empty()) {
        const auto found =
            std::find(definition.parameters.begin(),
                      definition.parameters.end(), call.keywords[argument]);
        parameter = static_cast<size_t>(found - definition.parameters.begin());
      }
      if (parameter >= supplied.size() || supplied[parameter])
        return true;
      supplied[parameter] = true;
      if (MacroExpressionMayWrite(call.args[argument], field, caller, active))
        return true;
      parameters.bindings[definition.parameters[parameter]] =
          Resolve(call.args[argument], caller, {});
    }
    for (size_t parameter = 0; parameter < supplied.size(); ++parameter) {
      if (supplied[parameter])
        continue;
      const size_t first = supplied.size() - definition.defaults.size();
      if (parameter < first)
        return true;
      const auto &value = definition.defaults[parameter - first];
      if (MacroExpressionMayWrite(value, field, parameters, active))
        return true;
      parameters.bindings[definition.parameters[parameter]] =
          Resolve(value, parameters, {});
    }
    return MacroBlockMayWrite(definition.body, field, &parameters, active);
  }
  bool UnsupportedMacroHistoryEffect(std::string_view field,
                                     const HistorySearch &search) {
    for (const auto *call : macro_calls_) {
      if (call->offset >= search.loop_end)
        continue;
      Path caller;
      if (MacroMayWriteField(*call, field, caller, {})) {
        Fail(
            call->offset,
            "unsupported append-sensitive macro namespace effect in history control: " +
                std::string(field));
        return true;
      }
    }
    return false;
  }
  using SearchDependencies = std::map<std::string, std::vector<Expr>>;
  void SearchGuardDependencies(const std::vector<Node> &nodes,
                               SearchDependencies *out,
                               std::vector<Expr> conditions = {}) const {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Set &&
          node.text.find('.') == std::string::npos) {
        auto &dependencies = (*out)[node.text];
        dependencies.push_back(node.expression);
        dependencies.insert(dependencies.end(), conditions.begin(),
                            conditions.end());
      }
      auto nested = conditions;
      if (node.kind == Node::Kind::If || node.kind == Node::Kind::For)
        nested.push_back(node.expression);
      SearchGuardDependencies(node.body, out, nested);
      SearchGuardDependencies(node.otherwise, out, std::move(nested));
    }
  }
  bool StableSearchGuard(const Expr &expression, const HistorySearch &search,
                         const Expr &ordinal,
                         const std::map<std::string, Expr> &aliases,
                         const std::map<std::string, Expr> &messages,
                         const SearchDependencies &locals,
                         std::set<std::string> seen = {}) {
    if (HistoricalOrdinal(expression, search.sequence, false, aliases) &&
        SameExpression(OrdinalExpression(expression, aliases), ordinal))
      return true;
    if (SameExpression(expression, SearchFlag(search)))
      return !UnsupportedMacroHistoryEffect(search.root + "." + search.flag,
                                            search);
    if (auto access = Access(expression)) {
      const auto dot = access->find('.');
      const auto root = access->substr(0, dot);
      if (const auto message = messages.find(root); message != messages.end()) {
        if (SameExpression(message->second, ordinal))
          return true;
        return Fail(
            expression.offset,
            "unsupported append-sensitive history guard record dependency");
      }
      if (root == "loop" || root == search.sequence.text || root == search.root)
        return Fail(expression.offset,
                    "unsupported append-sensitive history guard dependency");
      if (dot != std::string::npos) {
        if (UnsupportedMacroHistoryEffect(*access, search))
          return false;
        if (!seen.insert(*access).second)
          return true;
        if (const auto field = definitions_.find(*access);
            field != definitions_.end()) {
          for (const auto &dependency : field->second)
            if (!StableSearchGuard(Name(dependency), search, ordinal, aliases,
                                   messages, locals, seen))
              return false;
          return true;
        }
      }
      if (expression.kind == Expr::Kind::Name) {
        if (!seen.insert(expression.text).second)
          return true;
        if (const auto local = locals.find(expression.text);
            local != locals.end()) {
          for (const auto &dependency : local->second)
            if (!StableSearchGuard(dependency, search, ordinal, aliases,
                                   messages, locals, seen))
              return false;
          return true;
        }
        if (const auto global = definitions_.find(expression.text);
            global != definitions_.end()) {
          for (const auto &dependency : global->second)
            if (!StableSearchGuard(Name(dependency), search, ordinal, aliases,
                                   messages, locals, seen))
              return false;
          return true;
        }
      }
    }
    for (const auto &argument : expression.args)
      if (!StableSearchGuard(argument, search, ordinal, aliases, messages,
                             locals, seen))
        return false;
    return true;
  }
  void FindSearchWrite(const std::vector<Node> &nodes, const Node &loop,
                       bool reversed, HistorySearch *search,
                       std::map<std::string, Expr> aliases = {},
                       std::map<std::string, Expr> message_ordinals = {},
                       std::vector<Expr> protected_ordinals = {},
                       bool guarded = false, bool initialize = true,
                       std::vector<Expr> conditions = {},
                       SearchDependencies guard_dependencies = {}) {
    if (initialize) {
      SearchGuardDependencies(nodes, &guard_dependencies);
      Expr ordinal;
      ordinal.kind = Expr::Kind::GetAttr;
      ordinal.text = "index0";
      ordinal.args.push_back(Name("loop"));
      message_ordinals[loop.text] =
          reversed
              ? Binary("-", Binary("-", Length(search->sequence), Integer(1)),
                       ordinal)
              : ordinal;
    }
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Set &&
          node.text.find('.') == std::string::npos) {
        const auto old_messages = message_ordinals;
        message_ordinals.erase(node.text);
        if (node.expression.kind == Expr::Kind::GetItem &&
            node.expression.args.size() == 2 &&
            SameExpression(node.expression.args[0], search->sequence) &&
            HistoricalOrdinal(node.expression.args[1], search->sequence,
                              reversed, aliases))
          message_ordinals[node.text] =
              OrdinalExpression(node.expression.args[1], aliases);
        else if (node.expression.kind == Expr::Kind::Name) {
          const auto prior = old_messages.find(node.expression.text);
          if (prior != old_messages.end())
            message_ordinals[node.text] = prior->second;
        }
        aliases[node.text] = node.expression;
      }
      if (node.kind != Node::Kind::If)
        continue;
      auto protected_body = protected_ordinals;
      for (const auto &[message, ordinal] : message_ordinals)
        if (const auto role = RoleGuard(node.expression, message);
            role && !*role)
          protected_body.push_back(ordinal);
      const bool flag_guard =
          search->flag.empty() || guarded ||
          ContainsConjunct(node.expression, SearchFlag(*search));
      auto body_conditions = conditions;
      body_conditions.push_back(node.expression);
      if (!protected_body.empty() && flag_guard) {
        const Node *index = nullptr, *flag = nullptr;
        auto write_aliases = aliases;
        for (const auto &write : node.body) {
          if (write.kind != Node::Kind::Set)
            continue;
          if (write.text.find('.') == std::string::npos) {
            write_aliases[write.text] = write.expression;
            continue;
          }
          if (write.text == search->root + "." + search->index &&
              HistoricalOrdinal(write.expression, search->sequence, reversed,
                                write_aliases)) {
            const auto ordinal =
                OrdinalExpression(write.expression, write_aliases);
            if (std::any_of(protected_body.begin(), protected_body.end(),
                            [&](const Expr &protected_ordinal) {
                              return SameExpression(ordinal, protected_ordinal);
                            }) &&
                (search->flag.empty() ||
                 std::all_of(body_conditions.begin(), body_conditions.end(),
                             [&](const Expr &condition) {
                               return StableSearchGuard(
                                   condition, *search, ordinal, aliases,
                                   message_ordinals, guard_dependencies);
                             })))
              index = &write;
          }
          if (write.text == search->root + "." + search->flag &&
              BooleanIs(write.expression, !search->initial_flag))
            flag = &write;
        }
        if (index && (flag || search->flag.empty())) {
          search->index_write = index;
          search->flag_write = flag;
        }
      }
      FindSearchWrite(node.body, loop, reversed, search, aliases,
                      message_ordinals, std::move(protected_body), flag_guard,
                      false, std::move(body_conditions), guard_dependencies);
      // A negated branch may have different role/flag implications; do not
      // import a positive-body proof into that source branch.
      auto else_conditions = conditions;
      else_conditions.push_back(Unary("not", node.expression));
      FindSearchWrite(node.otherwise, loop, reversed, search, aliases,
                      message_ordinals, protected_ordinals, guarded, false,
                      std::move(else_conditions), guard_dependencies);
      auto invalidate = [&](auto &&self,
                            const std::vector<Node> &body) -> void {
        for (const auto &write : body) {
          if (write.kind == Node::Kind::Set) {
            aliases.erase(write.text);
            message_ordinals.erase(write.text);
          }
          self(self, write.body);
          self(self, write.otherwise);
        }
      };
      invalidate(invalidate, node.body);
      invalidate(invalidate, node.otherwise);
    }
  }
  void GatherHistorySearches(const std::vector<Node> &nodes) {
    std::map<std::string, std::vector<const Node *>> writes;
    std::map<const Node *, std::vector<Expr>> write_guards;
    auto gather = [&](auto &&self, const std::vector<Node> &body,
                      const std::vector<Expr> &guards) -> void {
      for (const auto &node : body) {
        if (node.kind == Node::Kind::Set) {
          writes[node.text].push_back(&node);
          write_guards[&node] = guards;
        }
        auto yes = guards, no = guards;
        if (node.kind == Node::Kind::If) {
          yes.push_back(node.expression);
          no.push_back(Unary("not", node.expression));
        }
        self(self, node.body, yes);
        self(self, node.otherwise, no);
      }
    };
    gather(gather, nodes, {});
    for (const auto &declaration : nodes) {
      if (declaration.kind != Node::Kind::Set ||
          declaration.expression.kind != Expr::Kind::Call ||
          declaration.expression.args.empty() ||
          declaration.expression.args[0].kind != Expr::Kind::Name ||
          declaration.expression.args[0].text != "namespace" ||
          writes[declaration.text].size() != 1 ||
          NamespaceEscapes(nodes, declaration.text))
        continue;
      const auto &constructor = declaration.expression;
      for (size_t index = 1; index < constructor.args.size(); ++index) {
        Expr sequence;
        if (index < constructor.keywords.size() &&
            IntegerIs(constructor.args[index], -1)) {
          // A strict-earlier sentinel remains earlier even if no search write
          // occurs; every possible write must select a historical record.
          for (const auto &loop : nodes) {
            if (loop.kind != Node::Kind::For || loop.predicate ||
                loop.offset <= declaration.offset)
              continue;
            Expr anchor = loop.expression;
            bool reversed = false;
            if (anchor.kind == Expr::Kind::Slice && anchor.args.size() == 4 &&
                anchor.args[1].kind == Expr::Kind::None &&
                anchor.args[2].kind == Expr::Kind::None &&
                IntegerIs(anchor.args[3], -1)) {
              anchor = anchor.args[0];
              reversed = true;
            }
            if (anchor.kind != Expr::Kind::Name || !MessageSequence(anchor) ||
                loop.text == anchor.text)
              continue;
            bool same_sequence = true;
            const auto updates = writes.find(anchor.text);
            if (updates != writes.end()) {
              for (const auto *write : updates->second) {
                const auto &update = write->expression;
                const bool identity = update.kind == Expr::Kind::Name &&
                                      update.text == anchor.text;
                const bool nonempty = std::any_of(
                    write_guards[write].begin(), write_guards[write].end(),
                    [&](const Expr &guard) {
                      return ContainsConjunct(guard, anchor);
                    });
                const bool prefix_drop =
                    update.kind == Expr::Kind::Slice &&
                    update.args.size() == 4 &&
                    SameExpression(update.args[0], anchor) &&
                    IntegerIs(update.args[1], 1) &&
                    update.args[2].kind == Expr::Kind::None &&
                    update.args[3].kind == Expr::Kind::None && nonempty;
                if (write->offset >= loop.offset || (!identity && !prefix_drop))
                  same_sequence = false;
              }
            }
            if (!same_sequence)
              continue;
            HistorySearch strict{declaration.text, constructor.keywords[index],
                                 "", anchor};
            strict.loop_end = loop.end;
            FindSearchWrite(loop.body, loop, reversed, &strict);
            const auto &field_writes = writes[strict.root + "." + strict.index];
            if (strict.index_write && field_writes.size() == 1 &&
                field_writes[0] == strict.index_write)
              history_searches_.push_back(std::move(strict));
          }
          continue;
        }
        if (index >= constructor.keywords.size() ||
            !LastIndex(constructor.args[index], &sequence) ||
            sequence.kind != Expr::Kind::Name || writes.count(sequence.text))
          continue;
        for (size_t flag = 1; flag < constructor.args.size(); ++flag) {
          if (flag >= constructor.keywords.size() ||
              constructor.args[flag].kind != Expr::Kind::Boolean)
            continue;
          HistorySearch search{declaration.text, constructor.keywords[index],
                               constructor.keywords[flag], sequence,
                               BooleanIs(constructor.args[flag], true)};
          for (const auto &node : nodes) {
            if (node.offset <= declaration.offset ||
                node.kind != Node::Kind::For || node.predicate)
              continue;
            Expr iterated = node.expression;
            bool reversed = false;
            if (iterated.kind == Expr::Kind::Slice &&
                iterated.args.size() == 4 &&
                iterated.args[1].kind == Expr::Kind::None &&
                iterated.args[2].kind == Expr::Kind::None &&
                IntegerIs(iterated.args[3], -1)) {
              iterated = iterated.args[0];
              reversed = true;
            }
            if (!SameExpression(iterated, sequence) ||
                node.text == sequence.text)
              continue;
            HistorySearch candidate = search;
            candidate.index_write = nullptr;
            candidate.flag_write = nullptr;
            candidate.loop_end = node.end;
            FindSearchWrite(node.body, node, reversed, &candidate);
            if (candidate.index_write && candidate.flag_write)
              search = std::move(candidate);
          }
          const auto &index_writes = writes[search.root + "." + search.index];
          const auto &flag_writes = writes[search.root + "." + search.flag];
          if (!search.index_write || !search.flag_write ||
              index_writes.size() != 1 || flag_writes.size() != 1 ||
              index_writes[0] != search.index_write ||
              flag_writes[0] != search.flag_write)
            continue;
          if (UnsupportedMacroHistoryEffect(search.root + "." + search.index,
                                            search))
            continue;
          for (const auto &node : nodes) {
            if (node.offset <= search.index_write->offset ||
                node.kind != Node::Kind::If ||
                !SameExpression(node.expression, SearchFlag(search)))
              continue;
            bool before_body = true;
            for (const auto &earlier : nodes)
              if (earlier.offset > search.index_write->offset &&
                  earlier.offset < node.offset &&
                  earlier.kind == Node::Kind::For && HasOutput(earlier.body))
                before_body = false;
            if (before_body && node.body.size() == 1 && Raises(node.body) &&
                node.otherwise.empty())
              search.requires_match = true;
          }
          history_searches_.push_back(std::move(search));
        }
      }
    }
  }
  struct CountDefinition {
    std::string ordinary;
    std::string target;
    Expr expression;
    Expr target_expression;
    size_t anchor = 0;
    bool after = false;
    bool start = false;
  };
  std::string FreshCountName(std::string name) {
    while (source_names_.count(name) || generated_names_.count(name))
      name += '_';
    generated_names_.insert(name);
    return name;
  }
  Expr TargetCounts(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Name) {
      const auto found = count_targets_.find(expression.text);
      if (found != count_targets_.end())
        return Name(found->second);
    }
    Expr result = expression;
    for (auto &argument : result.args)
      argument = TargetCounts(argument);
    return result;
  }
  bool CountDependent(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Name &&
        (count_targets_.count(expression.text) ||
         count_target_names_.count(expression.text)))
      return true;
    for (const auto &argument : expression.args)
      if (CountDependent(argument))
        return true;
    return false;
  }
  bool CountDependent(const Parts &parts) const {
    for (const auto &part : parts)
      if (CountDependent(part.predicate) ||
          CountDependent(part.value_expression) || CountDependent(part.body) ||
          CountDependent(part.otherwise) || CountDependent(part.body_error) ||
          CountDependent(part.otherwise_error) ||
          CountDependent(part.body_error_statements) ||
          CountDependent(part.otherwise_error_statements))
        return true;
    return false;
  }
  bool CountDependent(const Template &statements) const {
    std::set<std::string> names;
    NodeReferences(statements.body, &names);
    for (const auto &name : names)
      if (count_targets_.count(name) || count_target_names_.count(name))
        return true;
    return false;
  }
  void TargetCounts(std::vector<Node> *nodes) const {
    for (auto &node : *nodes) {
      node.expression = TargetCounts(node.expression);
      if (node.predicate)
        node.predicate = TargetCounts(*node.predicate);
      for (auto &value : node.defaults)
        value = TargetCounts(value);
      TargetCounts(&node.body);
      TargetCounts(&node.otherwise);
    }
  }
  void TargetCounts(Parts *parts) const {
    for (auto &part : *parts) {
      part.predicate = TargetCounts(part.predicate);
      part.value_expression = TargetCounts(part.value_expression);
      part.body_error = TargetCounts(part.body_error);
      part.otherwise_error = TargetCounts(part.otherwise_error);
      TargetCounts(&part.body_error_statements.body);
      TargetCounts(&part.otherwise_error_statements.body);
      TargetCounts(&part.body);
      TargetCounts(&part.otherwise);
    }
  }
  size_t RegisterCount(Expr expression, size_t anchor, bool after, bool start) {
    const auto id = count_definitions_.size();
    CountDefinition definition;
    definition.ordinary = FreshCountName("__text_count_" + std::to_string(id));
    definition.target =
        FreshCountName("__text_target_count_" + std::to_string(id));
    definition.expression = std::move(expression);
    definition.anchor = anchor;
    definition.after = after;
    definition.start = start;
    count_targets_[definition.ordinary] = definition.target;
    count_target_names_.insert(definition.target);
    References(definition.expression, &definitions_[definition.ordinary]);
    selector_binding_expressions_[definition.ordinary] = definition.expression;
    count_definitions_.push_back(std::move(definition));
    return id;
  }
  void FinishTargetCount(size_t id, Expr expression) {
    auto &definition = count_definitions_[id];
    definition.target_expression = std::move(expression);
    References(definition.target_expression, &definitions_[definition.target]);
    selector_binding_expressions_[definition.target] =
        definition.target_expression;
  }
  size_t RegisterSequenceCount(size_t sequence, size_t root, size_t drop,
                               Expr value, size_t anchor, bool after = false,
                               bool start = false) {
    const auto old = sequence_counts_.find(sequence);
    if (old != sequence_counts_.end())
      return old->second;
    const auto id =
        RegisterCount(Length(std::move(value)), anchor, after, start);
    sequence_counts_[sequence] = id;
    FinishTargetCount(id, Binary("+", Name(count_definitions_[id].ordinary),
                                 Name(target_missing_)));
    if (drop && sequence_counts_.count(root))
      count_affine_definitions_[count_definitions_[id].target] = Binary(
          "-", Name(count_definitions_[sequence_counts_.at(root)].target),
          Integer(drop));
    return id;
  }
  void EmitCountDefinitions(size_t anchor, bool after, bool start,
                            std::vector<Node> *out) const {
    for (const auto &definition : count_definitions_) {
      if (definition.anchor != anchor || definition.after != after ||
          definition.start != start)
        continue;
      for (bool target : {false, true}) {
        const auto &name = target ? definition.target : definition.ordinary;
        if (!selector_needed_.count(name))
          continue;
        Node set;
        set.kind = Node::Kind::Set;
        set.text = name;
        set.expression =
            target ? definition.target_expression : definition.expression;
        out->push_back(std::move(set));
      }
    }
  }
  bool PureCountScalar(const Expr &expression) const {
    if (expression.kind == Expr::Kind::Name ||
        expression.kind == Expr::Kind::Integer ||
        expression.kind == Expr::Kind::Boolean)
      return true;
    if (expression.kind == Expr::Kind::Unary && expression.text == "not" &&
        expression.args.size() == 1)
      return PureCountScalar(expression.args[0]);
    if (expression.kind != Expr::Kind::Binary || expression.args.size() != 2)
      return false;
    if (expression.text != "+" && expression.text != "-" &&
        expression.text != ">" && expression.text != ">=" &&
        expression.text != "<" && expression.text != "<=" &&
        expression.text != "==" && expression.text != "!=" &&
        expression.text != "and" && expression.text != "or")
      return false;
    return PureCountScalar(expression.args[0]) &&
           PureCountScalar(expression.args[1]);
  }
  struct CountAffine {
    std::map<std::string, int64_t> terms;
    int64_t constant = 0;
  };
  std::optional<CountAffine>
  AffineCount(const Expr &expression,
              std::set<std::string> visiting = {}) const {
    if (const auto number = GuardInteger(expression))
      return CountAffine{{}, *number};
    if (expression.kind == Expr::Kind::Name) {
      const auto definition = count_affine_definitions_.find(expression.text);
      if (definition != count_affine_definitions_.end()) {
        if (!visiting.insert(expression.text).second)
          return std::nullopt;
        return AffineCount(definition->second, std::move(visiting));
      }
      if (count_target_names_.count(expression.text) ||
          count_targets_.count(expression.text))
        return CountAffine{{{expression.text, 1}}, 0};
      return std::nullopt;
    }
    if (expression.kind != Expr::Kind::Binary || expression.args.size() != 2 ||
        (expression.text != "+" && expression.text != "-"))
      return std::nullopt;
    auto a = AffineCount(expression.args[0], visiting);
    const auto b = AffineCount(expression.args[1], std::move(visiting));
    if (!a || !b)
      return std::nullopt;
    const bool add = expression.text == "+";
    if (add ? __builtin_add_overflow(a->constant, b->constant, &a->constant)
            : __builtin_sub_overflow(a->constant, b->constant, &a->constant))
      return std::nullopt;
    for (const auto &[name, coefficient] : b->terms) {
      auto &term = a->terms[name];
      if (add ? __builtin_add_overflow(term, coefficient, &term)
              : __builtin_sub_overflow(term, coefficient, &term))
        return std::nullopt;
      if (!term)
        a->terms.erase(name);
    }
    return a;
  }
  std::optional<bool> CountComparison(const std::string &op, const Expr &left,
                                      const Expr &right) const {
    const auto a = AffineCount(left), b = AffineCount(right);
    if (!a || !b || a->terms != b->terms)
      return std::nullopt;
    if (op == "==")
      return a->constant == b->constant;
    if (op == "!=")
      return a->constant != b->constant;
    if (op == ">")
      return a->constant > b->constant;
    if (op == ">=")
      return a->constant >= b->constant;
    if (op == "<")
      return a->constant < b->constant;
    if (op == "<=")
      return a->constant <= b->constant;
    return std::nullopt;
  }
  Value Resolve(const Expr &expr, Path &path, const Context &context) {
    ++resolve_depth_;
    const bool prior_last = resolving_last_;
    resolving_last_ = false;
    auto value = ResolveOrdinary(expr, path, context);
    const bool reads_last =
        resolving_last_ || value.last_expression.has_value();
    if (reads_last && !projection_mode_ && context.target &&
        !context.continued_user)
      value.last_expression = ExpandLastAliases(expr, path);
    --resolve_depth_;
    resolving_last_ = resolve_depth_ ? prior_last || reads_last : false;
    if (!projection_mode_ && !context.continued_user &&
        (context.target || resolving_target_)) {
      value.expression = TargetCounts(value.expression);
      if (CountDependent(value.expansion)) {
        TargetCounts(&value.expansion);
        // A transform wrapper retains its typed value; its body is provenance.
        // Rendering it again would stringify a split list before later
        // indexing.
        const bool typed_wrapper =
            value.expansion.size() == 1 &&
            value.expansion.front().capture == TextCapture::Expression &&
            SameExpression(value.expression,
                           value.expansion.front().value_expression);
        if (!typed_wrapper)
          if (const auto expression = PatternExpression(value.expansion))
            value.expression = *expression;
      }
    }
    return value;
  }
  Expr ExpandLastAliases(Expr expr, const Path &path) const {
    if (expr.kind == Expr::Kind::Name) {
      const auto it = path.bindings.find(expr.text);
      if (it != path.bindings.end() && it->second.last_expression)
        return *it->second.last_expression;
    }
    for (auto &arg : expr.args)
      arg = ExpandLastAliases(std::move(arg), path);
    return expr;
  }
  Value ResolveOrdinary(const Expr &expr, Path &path, const Context &context) {
    if (expr.kind == Expr::Kind::Name) {
      if (const auto found = path.bindings.find(expr.text);
          found != path.bindings.end()) {
        Value value = found->second;
        if (const auto truth = path.future_truth.find(value.origin);
            truth != path.future_truth.end() && DirectSemantic(value))
          value.truth = truth->second;
        return value;
      }
      Value request{Origin::Request, expr};
      if (continuation_mode_) {
        request.defined = false;
        request.truth = false;
      }
      return request;
    }
    if (expr.kind == Expr::Kind::String)
      return Constant(expr.text, expr);
    if (expr.kind == Expr::Kind::Boolean)
      return Constant(expr.text == "true", expr);
    if (expr.kind == Expr::Kind::Integer) {
      Value value{Origin::Request, expr};
      value.defined = true;
      value.truth = expr.text != "0";
      return value;
    }
    if (expr.kind == Expr::Kind::None) {
      Value value{Origin::Request, expr};
      value.defined = true;
      return value;
    }
    if (expr.kind == Expr::Kind::Undefined) {
      Value value{Origin::Request, expr};
      value.defined = false;
      value.truth = false;
      return value;
    }
    if (expr.kind == Expr::Kind::GetAttr || expr.kind == Expr::Kind::GetItem) {
      if (expr.args.empty())
        return {Origin::Unknown, expr};
      Value parent = Resolve(expr.args[0], path, context);
      auto member = MemberName(expr);
      if (member && parent.origin == Origin::Message) {
        if (*member == "role" && !projection_mode_)
          return Constant(context.continued_user ? "user" : "assistant", expr);
        if (*member == "role") {
          Expr role = expr;
          role.args[0] = Name(worker_message_);
          return {Origin::Request, std::move(role)};
        }
        if (*member == "content")
          return Semantic(Origin::Content, expr);
        if (context.continued_user) {
          Value absent{Origin::Request, expr};
          absent.defined = false;
          absent.truth = false;
          return absent;
        }
        if (*member == "reasoning" || *member == "thinking") {
          if (context.target && !projection_mode_ && !context.continued_user) {
            // The completed-record constructor supplies reasoning_content only.
            // Historical workers retain their original, independently valued
            // alias fields; an equal semantic enum is not an aliasing proof.
            Expr missing = expr;
            missing.kind = Expr::Kind::Undefined;
            missing.args.clear();
            return Resolve(missing, path, context);
          }
          Expr access = expr;
          access.args[0] = parent.expression;
          return {Origin::Request, std::move(access)};
        }
        if (*member == "reasoning_content")
          return Semantic(Origin::Reasoning, expr);
        if (*member == "tool_calls") {
          Value calls = Semantic(Origin::Calls, expr);
          if (const auto truth = path.future_truth.find(Origin::Calls);
              truth != path.future_truth.end())
            calls.truth = truth->second;
          return calls;
        }
      }
      if (member && (parent.origin == Origin::Call ||
                     parent.origin == Origin::Function)) {
        if (*member == "function")
          return Semantic(Origin::Function, expr);
        if (*member == "arguments")
          return Semantic(Origin::Arguments, expr);
        if (*member == "name") {
          return Semantic(Origin::FunctionName, expr);
        }
      }
      if (parent.origin == Origin::Calls)
        return Semantic(Origin::Call, expr);
      if (parent.origin == Origin::Arguments) {
        ObserveProjection(0, path);
        if (projection_mode_) {
          Expr access = expr;
          access.args[0] = parent.expression;
          for (size_t i = 1; i < access.args.size(); ++i)
            access.args[i] = Resolve(expr.args[i], path, context).expression;
          return {Origin::ArgumentValue, std::move(access)};
        }
        return Semantic(Origin::ArgumentValue, expr);
      }
      if ((parent.origin == Origin::Content ||
           parent.origin == Origin::Reasoning) &&
          expr.kind == Expr::Kind::GetItem) {
        Expr rebuilt = expr;
        rebuilt.args[0] = parent.expression;
        for (size_t i = 1; i < rebuilt.args.size(); ++i)
          rebuilt.args[i] = Resolve(expr.args[i], path, context).expression;
        return TransformedPayload(std::move(parent), std::move(rebuilt), expr);
      }
      if (parent.origin == Origin::Loop && member && !context.loops.empty()) {
        const auto &[sequence, expression] = context.loops.back();
        if (context.continued_user && sequence == Origin::Messages) {
          if (*member == "first")
            return Constant(false, expr);
          if (*member == "last")
            return Constant(true, expr);
          if (*member == "index0")
            return {Origin::Request, Name(user_ordinal_, Range(expr))};
          if (*member == "index")
            return {Origin::Request,
                    Binary("+", Name(user_ordinal_), Integer(1))};
          return {Origin::Request, Length(Name("messages"))};
        }
        if (projection_mode_) {
          const Expr index =
              Name(sequence == Origin::Messages ? worker_message_index_
                                                : worker_call_index_);
          if (*member == "index0")
            return {Origin::Request, index};
          if (*member == "index")
            return {Origin::Request, Binary("+", index, Integer(1))};
          if (*member == "first")
            return {Origin::Request, Binary("==", index, Integer(0))};
          if (*member == "last")
            return {Origin::Request,
                    Binary("==", index,
                           Binary("-", Length(expression), Integer(1)))};
          if (*member == "length")
            return {Origin::Request, Length(expression)};
        }
        if (*member == "last") {
          auto value = sequence == Origin::Messages
                           ? Constant(true, expr)
                           : Semantic(Origin::RepeatLast, expr);
          if (sequence == Origin::Messages && !projection_mode_ &&
              context.target && !context.continued_user)
            value.last_expression = expr;
          return value;
        }
        if (sequence == Origin::Messages) {
          const auto count = sequence_counts_.find(context.target_sequence);
          if (count == sequence_counts_.end()) {
            Fail(expr.offset,
                 "target message sequence has no source-point count");
            return {Origin::Unknown, expr};
          }
          const Expr length = Name(count_definitions_[count->second].ordinary);
          const Expr index = Binary("-", length, Integer(1));
          if (*member == "length")
            return {Origin::Request, length};
          if (*member == "first")
            return {Origin::Request, Binary("==", index, Integer(0))};
          if (*member == "index0") {
            Value ordinal{Origin::Request, index};
            ordinal.target_ordinal = true;
            return ordinal;
          }
          if (*member == "index")
            return {Origin::Request, length};
        }
        if (*member == "first") {
          if (sequence != Origin::Messages)
            return Semantic(Origin::RepeatFirst, expr);
          return {Origin::Request,
                  Binary("==", Length(expression), Integer(0))};
        }
        if (*member == "index0")
          return {Origin::Request,
                  sequence == Origin::Messages
                      ? Length(expression)
                      : Semantic(Origin::RepeatIndex, expr).expression};
        if (*member == "index")
          return {Origin::Request,
                  Binary("+",
                         sequence == Origin::Messages
                             ? Length(expression)
                             : Semantic(Origin::RepeatIndex, expr).expression,
                         Integer(1))};
      }
      Expr rebuilt = expr;
      rebuilt.args[0] = parent.expression;
      for (size_t i = 1; i < rebuilt.args.size(); ++i)
        rebuilt.args[i] = Resolve(expr.args[i], path, context).expression;
      // An indexed existing message is request data, not the conceptual final
      // assistant whose role/body this pass is recognizing.
      return {Origin::Request, std::move(rebuilt)};
    }
    if (expr.kind == Expr::Kind::Slice) {
      Value result = Resolve(expr.args[0], path, context);
      result.truth.reset();
      result.expression = expr;
      for (size_t i = 0; i < expr.args.size(); ++i)
        result.expression.args[i] =
            Resolve(expr.args[i], path, context).expression;
      if (result.origin == Origin::Messages && !projection_mode_) {
        const bool tail = expr.args.size() == 4 &&
                          expr.args[2].kind == Expr::Kind::None &&
                          (expr.args[3].kind == Expr::Kind::None ||
                           IntegerIs(expr.args[3], 1));
        const bool identity = tail && (expr.args[1].kind == Expr::Kind::None ||
                                       IntegerIs(expr.args[1], 0));
        const bool prefix = tail && IntegerIs(expr.args[1], 1);
        const bool reverse = expr.args.size() == 4 &&
                             expr.args[1].kind == Expr::Kind::None &&
                             expr.args[2].kind == Expr::Kind::None &&
                             IntegerIs(expr.args[3], -1);
        if (!identity && !reverse) {
          const auto key = std::make_pair(expr.offset, result.sequence);
          if (!sequence_versions_.count(key))
            sequence_versions_[key] = ++next_sequence_;
          result.sequence = sequence_versions_.at(key);
          if (prefix)
            ++result.prefix_drop;
          else {
            result.count_root = result.sequence;
            result.prefix_drop = 0;
          }
          RegisterSequenceCount(result.sequence, result.count_root,
                                result.prefix_drop, result.expression,
                                context.target ? context.target_loop
                                               : current_statement_);
          if (prefix && sequence_counts_.count(result.count_root))
            result.target_survival.push_back(Binary(
                ">",
                Name(count_definitions_[sequence_counts_.at(result.count_root)]
                         .target),
                Integer(result.prefix_drop)));
        }
        result.append_preserving &= identity || prefix;
      }
      if (!result.expansion.empty()) {
        auto value = PatternExpression(result.expansion);
        if (!value) {
          Fail(expr.offset, "payload slice over repetition is unsupported");
          return result;
        }
        TextFormatPart transformed;
        transformed.capture = TextCapture::Expression;
        transformed.source = Range(expr);
        transformed.value_expression = result.expression;
        transformed.value_expression.args[0] = *value;
        transformed.body = std::move(result.expansion);
        result.expansion = {std::move(transformed)};
      }
      return result;
    }
    if (expr.kind == Expr::Kind::Filter) {
      Value subject = Resolve(expr.args[0], path, context);
      if ((expr.text == "length" || expr.text == "count") &&
          subject.origin == Origin::Messages) {
        if (projection_mode_)
          return {Origin::Request, Length(subject.expression)};
        const auto id = RegisterSequenceCount(
            subject.sequence, subject.count_root, subject.prefix_drop,
            subject.expression,
            context.target ? context.target_loop : current_statement_);
        Value count{Origin::Request, Name(count_definitions_[id].ordinary)};
        count.defined = true;
        return count;
      }
      if (expr.text == "default" || expr.text == "d") {
        const Value fallback = expr.args.size() > 1
                                   ? Resolve(expr.args[1], path, context)
                                   : Constant(std::string(), expr);
        if (subject.defined && !*subject.defined)
          return fallback;
        if (fallback.defined && *fallback.defined)
          subject.defined = true;
      }
      if (expr.text == "items" && subject.origin == Origin::Arguments) {
        ObserveProjection(0, path);
        return Semantic(Origin::ArgumentItems, expr);
      }
      // Historical projection and continued USER render forward. This does
      // not define an inverse or a codec for transformed assistant output.
      const bool forward_sort = expr.text == "sort" && expr.args.size() == 2 &&
                                expr.keywords.size() == 2 &&
                                expr.keywords[1] == "attribute" &&
                                expr.args[1].kind == Expr::Kind::String &&
                                expr.args[1].text == "type";
      const bool forward_transform =
          (expr.text == "replace" || forward_sort) &&
          subject.origin == Origin::Content &&
          (projection_mode_ || context.continued_user);
      if (expr.text != "trim" && expr.text != "lstrip" &&
          expr.text != "rstrip" && expr.text != "tojson" &&
          expr.text != "string" && expr.text != "safe" &&
          expr.text != "default" && expr.text != "d" && expr.text != "length" &&
          expr.text != "first" && expr.text != "last" && expr.text != "list" &&
          expr.text != "join" && !forward_transform) {
        if (IsFuture(subject.origin))
          Fail(expr.offset,
               "unsupported payload filter in format analysis: " + expr.text);
      }
      Expr rebuilt = expr;
      if (forward_transform)
        rebuilt.args[0] = subject.expression;
      for (size_t i = forward_transform ? 1 : 0; i < rebuilt.args.size(); ++i)
        rebuilt.args[i] = Resolve(expr.args[i], path, context).expression;
      if (forward_transform) {
        // A projection-irrelevant assignment may lose the result's Content
        // origin before a later loop. The eager sort itself observes it.
        if (forward_sort && projection_mode_)
          ObserveProjection(2, path);
        return TransformedPayload(std::move(subject), std::move(rebuilt), expr);
      }
      subject.expression = std::move(rebuilt);
      if (!subject.expansion.empty() && expr.text != "safe") {
        auto value = PatternExpression(subject.expansion);
        if (!value) {
          Fail(
              expr.offset,
              "payload filter over repetition requires a supported value codec");
          return subject;
        }
        TextFormatPart transformed;
        transformed.capture = TextCapture::Expression;
        transformed.source = Range(expr);
        transformed.value_expression = subject.expression;
        transformed.value_expression.args[0] = std::move(*value);
        transformed.body = std::move(subject.expansion);
        subject.expansion = {std::move(transformed)};
      }
      return subject;
    }
    if (expr.kind == Expr::Kind::Call && !expr.args.empty()) {
      const Expr &callee = expr.args[0];
      if (callee.kind == Expr::Kind::Name && macros_.count(callee.text)) {
        bool semantic_actual = false;
        Origin semantic_origin = Origin::Request;
        Expr request_call = expr;
        for (size_t i = 1; i < expr.args.size(); ++i) {
          const Value actual = Resolve(expr.args[i], path, context);
          semantic_actual |=
              IsFuture(actual.origin) || !actual.expansion.empty();
          if (IsFuture(actual.origin))
            semantic_origin = actual.origin;
          request_call.args[i] = actual.expression;
        }
        if (!semantic_actual && (!recursive_macros_.count(callee.text) ||
                                 RequestRecursiveCall(expr, path, context)))
          return {Origin::Request, std::move(request_call)};
        if (projection_mode_) {
          Emit(expr, path, context);
          return {semantic_origin, std::move(request_call)};
        }
        Path expanded = path;
        expanded.parts.clear();
        auto emitted = Emit(expr, std::move(expanded), context);
        if (emitted.size() == 1 && !emitted[0].rejected) {
          Value value;
          value.expression = expr;
          value.expansion = std::move(emitted[0].parts);
          if (auto flattened = PatternExpression(value.expansion))
            value.expression = *flattened;
          path.adaptation.arguments_need_mapping |=
              emitted[0].adaptation.arguments_need_mapping;
          path.adaptation.arguments_observe_string |=
              emitted[0].adaptation.arguments_observe_string;
          path.adaptation.content_observes_representation |=
              emitted[0].adaptation.content_observes_representation;
          if (HasCapture(value.expansion, TextCapture::Content))
            value.origin = Origin::Content;
          else if (HasCapture(value.expansion, TextCapture::Reasoning))
            value.origin = Origin::Reasoning;
          return value;
        }
        Fail(
            expr.offset,
            "macro assignment has request-dependent string alternatives that need an explicit source slice");
        return {Origin::Unknown, expr};
      }
      if (callee.kind == Expr::Kind::GetAttr && !callee.args.empty()) {
        Value parent = Resolve(callee.args[0], path, context);
        if (callee.text == "get" && expr.args.size() >= 2 &&
            expr.args[1].kind == Expr::Kind::String) {
          if (parent.origin == Origin::Message &&
              (expr.args[1].text == "thinking" ||
               expr.args[1].text == "reasoning")) {
            if (expr.args.size() != 2) {
              Fail(
                  expr.offset,
                  "defaulted reasoning field lookup is unsupported in format analysis");
              return {Origin::Unknown, expr};
            }
            if (context.target && !projection_mode_ &&
                !context.continued_user) {
              // dict.get without a default returns defined None, unlike a
              // missing attribute's Undefined. Both remain false in an OR.
              Expr missing = expr;
              missing.kind = Expr::Kind::None;
              missing.args.clear();
              Value value{Origin::Request, std::move(missing)};
              value.defined = true;
              value.truth = false;
              return value;
            }
            Expr lookup = expr;
            lookup.args[0].args[0] = parent.expression;
            return {Origin::Request, std::move(lookup)};
          }
          Expr access;
          access.kind = Expr::Kind::GetAttr;
          access.text = expr.args[1].text;
          access.args = {callee.args[0]};
          access.offset = expr.offset;
          access.end = expr.end;
          return Resolve(access, path, context);
        }
        if (parent.origin == Origin::Arguments && callee.text == "items") {
          ObserveProjection(0, path);
          return Semantic(Origin::ArgumentItems, expr);
        }
        if (callee.text == "strip" || callee.text == "lstrip" ||
            callee.text == "rstrip") {
          Expr rebuilt = expr;
          rebuilt.args[0].args[0] = parent.expression;
          for (size_t i = 1; i < rebuilt.args.size(); ++i)
            rebuilt.args[i] = Resolve(expr.args[i], path, context).expression;
          parent.expression = std::move(rebuilt);
          if (!parent.expansion.empty()) {
            auto value = PatternExpression(parent.expansion);
            if (!value) {
              Fail(expr.offset, "payload strip over repetition is unsupported");
              return parent;
            }
            TextFormatPart transformed;
            transformed.capture = TextCapture::Expression;
            transformed.source = Range(expr);
            transformed.value_expression = parent.expression;
            transformed.value_expression.args[0].args[0] = *value;
            transformed.body = std::move(parent.expansion);
            parent.expansion = {std::move(transformed)};
          }
          return parent;
        }
        if (parent.origin == Origin::Content && callee.text == "split") {
          Expr rebuilt = expr;
          rebuilt.args[0].args[0] = parent.expression;
          for (size_t i = 1; i < rebuilt.args.size(); ++i)
            rebuilt.args[i] = Resolve(expr.args[i], path, context).expression;
          return TransformedPayload(std::move(parent), std::move(rebuilt),
                                    expr);
        }
      }
      Expr rebuilt = expr;
      for (size_t i = 0; i < rebuilt.args.size(); ++i)
        rebuilt.args[i] = Resolve(expr.args[i], path, context).expression;
      return {Origin::Request, std::move(rebuilt)};
    }
    if (expr.kind == Expr::Kind::Test && !expr.args.empty()) {
      Value subject = Resolve(expr.args[0], path, context);
      if (subject.origin == Origin::Content)
        ObserveProjection(2, path);
      std::optional<bool> known;
      if (subject.defined &&
          (expr.text == "defined" || expr.text == "undefined"))
        known = expr.text == "defined" ? *subject.defined : !*subject.defined;
      if (continuation_mode_ && subject.defined && !*subject.defined &&
          expr.text != "defined" && expr.text != "undefined")
        known = false;
      if (subject.boolean) {
        if (expr.text == "true")
          known = *subject.boolean;
        if (expr.text == "false")
          known = !*subject.boolean;
        if (expr.text == "boolean" || expr.text == "defined")
          known = true;
        if (expr.text == "none" || expr.text == "undefined" ||
            expr.text == "string" || expr.text == "mapping" ||
            expr.text == "sequence" || expr.text == "iterable")
          known = false;
      }
      if (subject.string) {
        if (expr.text == "string" || expr.text == "sequence" ||
            expr.text == "iterable" || expr.text == "defined")
          known = true;
        if (expr.text == "none" || expr.text == "undefined" ||
            expr.text == "mapping" || expr.text == "boolean" ||
            expr.text == "true" || expr.text == "false")
          known = false;
      }
      if (IsFuture(subject.origin) && !projection_mode_ &&
          DirectSemantic(subject)) {
        const unsigned domain = Domain(subject.origin, path);
        const unsigned selected = domain & TestMask(expr.text);
        if (!selected)
          known = false;
        else if (selected == domain)
          known = true;
      }
      if (known)
        return Constant(*known, expr);
      Expr rebuilt = expr;
      rebuilt.args[0] = subject.expression;
      return {Origin::Request, std::move(rebuilt)};
    }
    if (expr.kind == Expr::Kind::Unary && !expr.args.empty()) {
      Value value = Resolve(expr.args[0], path, context);
      if (expr.text == "not" && Truth(value))
        return Constant(!*Truth(value), expr);
      Value result{Origin::Request, Unary(expr.text, value.expression)};
      if (expr.text == "not" && value.retention) {
        result.retention = value.retention;
        result.retention->second = !result.retention->second;
      }
      return result;
    }
    if (expr.kind == Expr::Kind::Binary && expr.args.size() == 2) {
      Value a = Resolve(expr.args[0], path, context);
      if (const auto truth = Truth(a);
          truth &&
          ((expr.text == "and" && !*truth) || (expr.text == "or" && *truth)))
        return a;
      Path rhs = path;
      if (projection_mode_ && (expr.text == "and" || expr.text == "or"))
        rhs.guards.push_back(expr.text == "and" ? a.expression
                                                : Unary("not", a.expression));
      Value b = Resolve(expr.args[1], projection_mode_ ? rhs : path, context);
      if (context.continued_user && expr.text == "%" &&
          a.expression.kind == Expr::Kind::Name &&
          a.expression.text == user_ordinal_ && IntegerIs(b.expression, 2))
        return {Origin::Request, Integer(0)};
      if (!projection_mode_)
        if (const auto comparison =
                CountComparison(expr.text, a.expression, b.expression))
          return Constant(*comparison, expr);
      if (!projection_mode_ && context.target && !context.loops.empty() &&
          context.loops.back().first == Origin::Messages) {
        for (const auto &search : history_searches_) {
          if (!SameExpression(context.messages, search.sequence) ||
              !a.target_ordinal)
            continue;
          const auto field = Access(b.expression);
          if (!field || *field != search.root + "." + search.index)
            continue;
          // In the completed source, an unwritten len(S)-1 initializer is
          // exactly the target ordinal. A paired historical write is earlier.
          // Appending an assistant cannot take the proven user-only write.
          Value found =
              (search.flag.empty() || search.requires_match)
                  ? Constant(true, expr)
                  : Value{Origin::Request, Unary("not", SearchFlag(search))};
          if (!search.requires_match && !search.flag.empty())
            found.retention = std::make_pair(search.index_write->offset, true);
          if (expr.text == ">" || expr.text == "!=")
            return found;
          if (expr.text == "<=" || expr.text == "==") {
            if (found.boolean)
              return Constant(!*found.boolean, expr);
            found.expression = Unary("not", found.expression);
            if (found.retention)
              found.retention->second = false;
            return found;
          }
          if (expr.text == ">=")
            return Constant(true, expr);
          if (expr.text == "<")
            return Constant(false, expr);
        }
      }
      if ((expr.text == "+" || expr.text == "~") && a.string && b.string)
        return Constant(*a.string + *b.string, expr);
      if ((expr.text == "+" || expr.text == "~") && a.string_result &&
          b.string_result && !IsFuture(a.origin) && !IsFuture(b.origin)) {
        Value result{Origin::Request,
                     Binary(expr.text, a.expression, b.expression)};
        result.string_result = true;
        if ((Truth(a) && *Truth(a)) || (Truth(b) && *Truth(b)))
          result.truth = true;
        else if (Truth(a) && Truth(b))
          result.truth = false;
        return result;
      }
      if ((expr.text == "+" || expr.text == "~") &&
          ((IsFuture(a.origin) && b.string) ||
           (a.string && IsFuture(b.origin)))) {
        Value result;
        result.origin = a.string ? b.origin : a.origin;
        result.expression = Binary(expr.text, a.expression, b.expression);
        auto append_value = [&](const Value &value) {
          if (value.string)
            Append(&result.expansion, Literal(*value.string, Range(expr)));
          else if (!value.expansion.empty())
            for (auto part : value.expansion)
              Append(&result.expansion, std::move(part));
          else if (auto capture = Capture(value.origin)) {
            TextFormatPart part;
            part.capture = *capture;
            part.value_expression = value.expression;
            part.source = Range(expr);
            Append(&result.expansion, std::move(part));
          }
        };
        append_value(a);
        append_value(b);
        return result;
      }
      if ((expr.text == "==" || expr.text == "!=") && a.string && b.string)
        return Constant((*a.string == *b.string) == (expr.text == "=="), expr);
      if (expr.text == "and") {
        if (Truth(a))
          return *Truth(a) ? b : a;
        if (Truth(b))
          return *Truth(b) ? a : b;
      }
      if (expr.text == "or") {
        if (Truth(a))
          return *Truth(a) ? a : b;
        if (Truth(b))
          return *Truth(b) ? b : a;
        if (IsFuture(a.origin) && b.string && b.string->empty()) {
          a.expression = Binary("or", a.expression, b.expression);
          return a;
        }
      }
      if ((expr.text == "in" || expr.text == "not in") && a.string) {
        if (*a.string == "function" && b.origin == Origin::Call)
          return Constant(expr.text == "in", expr);
        if (*a.string == "tool_calls" && b.origin == Origin::Message) {
          if (context.continued_user)
            return Constant(expr.text == "not in", expr);
          Value calls = Semantic(Origin::Calls, expr);
          if (const auto truth = path.future_truth.find(Origin::Calls);
              truth != path.future_truth.end())
            calls.boolean = expr.text == "in" ? truth->second : !truth->second;
          if (expr.text == "not in")
            calls.expression = Unary("not", calls.expression);
          return calls;
        }
      }
      return {Origin::Request, Binary(expr.text, a.expression, b.expression)};
    }
    if (expr.kind == Expr::Kind::Conditional && expr.args.size() == 3) {
      Value condition = Resolve(expr.args[0], path, context);
      if (Truth(condition))
        return Resolve(expr.args[*Truth(condition) ? 1 : 2], path, context);
      Path yes = path, no = path;
      if (projection_mode_) {
        yes.guards.push_back(condition.expression);
        no.guards.push_back(Unary("not", condition.expression));
      }
      Value a = Resolve(expr.args[1], projection_mode_ ? yes : path, context);
      Value b = Resolve(expr.args[2], projection_mode_ ? no : path, context);
      if (a.expression.kind == Expr::Kind::Undefined ||
          b.expression.kind == Expr::Kind::Undefined)
        return Phi(condition.expression, a, b);
      if (IsFuture(a.origin) && b.string && b.string->empty()) {
        Expr subject = a.expression;
        a.expression = expr;
        a.expression.args = {condition.expression, std::move(subject),
                             b.expression};
        return a;
      }
      if (IsFuture(b.origin) && a.string && a.string->empty()) {
        Expr subject = b.expression;
        b.expression = expr;
        b.expression.args = {condition.expression, a.expression,
                             std::move(subject)};
        return b;
      }
      if (a.origin == b.origin && IsFuture(a.origin)) {
        Expr combined = expr;
        combined.args = {condition.expression, a.expression, b.expression};
        a.expression = std::move(combined);
        return a;
      }
      Expr rebuilt = expr;
      rebuilt.args = {condition.expression, a.expression, b.expression};
      return {a.origin == b.origin ? a.origin : Origin::Request,
              std::move(rebuilt)};
    }
    return {Origin::Request, expr};
  }
  std::vector<Path> Emit(const Expr &expression, Path path,
                         const Context &context) {
    if (expression.kind == Expr::Kind::Binary && expression.args.size() == 2 &&
        (expression.text == "+" || expression.text == "~")) {
      auto paths = Emit(expression.args[0], std::move(path), context);
      std::vector<Path> result;
      for (auto &p : paths) {
        auto next = Emit(expression.args[1], std::move(p), context);
        for (auto &item : next)
          result.push_back(std::move(item));
      }
      return result;
    }
    if (expression.kind == Expr::Kind::Conditional &&
        expression.args.size() == 3) {
      Value condition = Resolve(expression.args[0], path, context);
      if (condition.last_expression && !projection_mode_ && context.target &&
          !context.continued_user)
        completion_rewrite_diagnostics_[expression.offset] =
            "a401 completion cannot qualify a last-sensitive emitted expression";
      if (Truth(condition))
        return Emit(expression.args[*Truth(condition) ? 1 : 2], std::move(path),
                    context);
      {
        RecordRetentionChoice(expression.offset, condition);
        TextFormatPart choice;
        choice.capture = TextCapture::Choice;
        choice.predicate = condition.expression;
        choice.source = Range(expression);
        Path empty = path;
        empty.parts.clear();
        Path otherwise = empty;
        if (projection_mode_) {
          empty.guards.push_back(condition.expression);
          otherwise.guards.push_back(Unary("not", condition.expression));
        }
        auto yes = Emit(expression.args[1], std::move(empty), context);
        auto no = Emit(expression.args[2], std::move(otherwise), context);
        if (yes.size() != 1 || no.size() != 1)
          Fail(
              expression.offset,
              "nested payload choices require distinct structural discriminators");
        else {
          choice.body = std::move(yes[0].parts);
          choice.otherwise = std::move(no[0].parts);
          if (yes[0].rejection)
            choice.body_error = *yes[0].rejection;
          if (no[0].rejection)
            choice.otherwise_error = *no[0].rejection;
          if (yes[0].rejection_statements)
            choice.body_error_statements = *yes[0].rejection_statements;
          if (no[0].rejection_statements)
            choice.otherwise_error_statements = *no[0].rejection_statements;
          Append(&path.parts, std::move(choice));
        }
        return {std::move(path)};
      }
    }
    if (expression.kind == Expr::Kind::Call && !expression.args.empty() &&
        expression.args[0].kind == Expr::Kind::Name) {
      const auto &name = expression.args[0].text;
      if (name == "raise_exception") {
        if (projection_mode_) {
          Expr rejected = RepresentationWorld(Conjunction(path.guards), false);
          projection_rejection_rules_.push_back(rejected);
          if (std::any_of(path.guards.begin(), path.guards.end(),
                          [&](const Expr &guard) {
                            return NegativeMappingGuard(guard);
                          })) {
            // An explicit source mapping invariant supplies the missing
            // positive evidence; rejection of STRING alone does not.
            RecordMappingDemand(rejected, rejected);
          }
        }
        path.rejected = true;
        Expr failure = expression;
        for (size_t i = 1; i < failure.args.size(); ++i)
          failure.args[i] =
              Resolve(expression.args[i], path, context).expression;
        path.rejection = std::move(failure);
        return {std::move(path)};
      }
      if (const auto macro = macros_.find(name); macro != macros_.end()) {
        bool unresolved_namespace = false;
        if (RequestRecursiveCall(expression, path, context,
                                 &unresolved_namespace)) {
          if (!RecordTargetRegion(context, expression.offset, true))
            return {};
          // Preserve original execution rather than expanding a private schema
          // into the future output grammar. The existing recipe/source slicing
          // owns this expression and its lexical declaration dependencies.
          TextFormatPart part;
          part.capture = TextCapture::Expression;
          part.source = Range(expression);
          part.value_expression = expression;
          for (size_t i = 1; i < expression.args.size(); ++i)
            part.value_expression.args[i] =
                Resolve(expression.args[i], path, context).expression;
          if (!projection_mode_)
            Append(&path.parts, std::move(part));
          if (!projection_mode_ && !continuation_mode_ &&
              context.source_output_owner &&
              completion_message_loops_.count(context.target_loop)) {
            recursive_target_outputs_[*context.source_output_owner] =
                context.target_loop;
            if (!recursive_history_guards_.count(context.target_loop))
              recursive_history_guards_[context.target_loop] =
                  FreshCountName("__text_recursive_history");
          }
          Node execution;
          execution.kind = Node::Kind::Output;
          execution.expression = expression;
          Node retained;
          retained.body = {std::move(execution)};
          ForgetCaptureWrites(retained, &path);
          return {std::move(path)};
        }
        if (unresolved_namespace) {
          Fail(
              expression.offset,
              "recursive macro namespace capture has future source dependencies");
          return {};
        }
        if (!macro_stack_.insert(name).second) {
          Fail(expression.offset, "recursive macro format cannot be analyzed");
          return {};
        }
        const Node &definition = *macro->second;
        const auto prior_bindings = path.bindings;
        std::vector<std::optional<Value>> actuals(definition.parameters.size());
        size_t positional = 0;
        for (size_t argument = 1; argument < expression.args.size();
             ++argument) {
          size_t parameter = positional++;
          if (argument < expression.keywords.size() &&
              !expression.keywords[argument].empty()) {
            const auto found = std::find(definition.parameters.begin(),
                                         definition.parameters.end(),
                                         expression.keywords[argument]);
            parameter =
                static_cast<size_t>(found - definition.parameters.begin());
          }
          if (parameter >= actuals.size() || actuals[parameter]) {
            Fail(expression.offset,
                 "invalid or duplicate macro argument in format analysis");
            macro_stack_.erase(name);
            return {};
          }
          // Every actual is resolved in the unchanged caller frame. Parameter
          // shadowing in this invocation cannot affect another actual.
          actuals[parameter] =
              Resolve(expression.args[argument], path, context);
        }
        // Free macro captures use the module frame, never a same-spelled
        // message-loop local. Actuals above were evaluated in the caller.
        path.bindings = path.module_bindings;
        for (size_t i = 0; i < definition.parameters.size(); ++i) {
          if (actuals[i])
            path.bindings[definition.parameters[i]] = *actuals[i];
          else {
            const size_t first_default =
                definition.parameters.size() - definition.defaults.size();
            if (i >= first_default)
              path.bindings[definition.parameters[i]] = Resolve(
                  definition.defaults[i - first_default], path, context);
          }
        }
        Context macro_context = context;
        macro_context.failing_loop.reset();
        macro_context.allow_statement_errors = false;
        auto result = Walk(definition.body, {std::move(path)}, macro_context);
        for (auto &item : result)
          item.bindings = prior_bindings;
        macro_stack_.erase(name);
        return result;
      }
    }
    Value value = Resolve(expression, path, context);
    if (value.expression.kind == Expr::Kind::Undefined)
      return {std::move(path)};
    if (value.last_expression && !projection_mode_ && context.target &&
        !context.continued_user)
      completion_rewrite_diagnostics_[expression.offset] =
          "a401 completion cannot qualify a last-sensitive emitted expression";
    if (projection_mode_) {
      // A type test or rejecting branch is not a STRING implementation. Only
      // an actual output of the arguments under a representation guard proves
      // that the source accepts the wire STRING without projection.
      if (value.origin == Origin::Arguments &&
          std::any_of(
              path.guards.begin(), path.guards.end(),
              [&](const Expr &guard) { return ArgumentTypeGuard(guard); }))
        ObserveProjection(1, path);
      return {std::move(path)};
    }
    if (!value.expansion.empty()) {
      for (auto part : value.expansion)
        Append(&path.parts, std::move(part));
    } else if (value.string) {
      Append(&path.parts, Literal(*value.string, Range(expression)));
    } else if (value.origin == Origin::FunctionName) {
      TextFormatPart part;
      part.capture = TextCapture::FunctionName;
      part.value_expression = value.expression;
      part.source = Range(expression);
      Append(&path.parts, std::move(part));
    } else if (auto capture = Capture(value.origin)) {
      TextFormatPart part;
      part.capture = *capture;
      part.value_expression = value.expression;
      part.source = Range(expression);
      Append(&path.parts, std::move(part));
    } else if (expression.kind == Expr::Kind::Filter &&
               !expression.args.empty()) {
      const auto start = path.parts.size();
      auto result = Emit(expression.args[0], std::move(path), context);
      for (auto &item : result) {
        Parts subject(item.parts.begin() + start, item.parts.end());
        const auto value = PatternExpression(subject);
        if (!value) {
          Fail(expression.offset,
               "filtered macro repetition has no bounded capture encoding");
          continue;
        }
        TextFormatPart transformed;
        transformed.capture = TextCapture::Expression;
        transformed.body = std::move(subject);
        transformed.value_expression = expression;
        transformed.value_expression.args[0] = *value;
        for (size_t i = 1; i < expression.args.size(); ++i)
          transformed.value_expression.args[i] =
              Resolve(expression.args[i], item, context).expression;
        transformed.source = Range(expression);
        item.parts.resize(start);
        Append(&item.parts, std::move(transformed));
      }
      return result;
    } else if (context.continued_user && Future(value.expression)) {
      // USER continuation renders forward, including scalar results derived
      // from its text. Body retains dependency provenance, not extra output.
      TextFormatPart part;
      part.capture = TextCapture::Expression;
      part.value_expression = value.expression;
      part.source = Range(expression);
      for (const auto &[name, capture] : predicate_bindings_)
        if (capture == TextCapture::Content &&
            ContainsName(value.expression, name)) {
          TextFormatPart input;
          input.capture = TextCapture::Content;
          input.value_expression = Name(name);
          input.source = Range(expression);
          part.body.push_back(std::move(input));
        }
      Append(&path.parts, std::move(part));
    } else if (!Future(value.expression)) {
      // Source-observable request text may precede a payload (for example a
      // first-turn system prefix). Keep it as an actual native expression;
      // it is not authorized as a finite stream/parser delimiter.
      TextFormatPart part;
      part.capture = TextCapture::Expression;
      part.value_expression = value.expression;
      part.source = Range(expression);
      Append(&path.parts, std::move(part));
    } else {
      Fail(expression.offset,
           "assistant framing contains an unclassified non-finite value");
    }
    return {std::move(path)};
  }

  void CaptureInputs(const std::vector<Node> &nodes,
                     std::set<std::string> *locals,
                     std::set<std::string> *inputs) const {
    const auto read = [&](const Expr &expression,
                          const std::set<std::string> &bound) {
      std::set<std::string> names;
      References(expression, &names);
      for (const auto &name : names)
        if (!bound.count(name.substr(0, name.find('.'))))
          inputs->insert(name);
    };
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Macro)
        continue; // Called macros retain their ordinary dependency closure.
      // Reads happen before an assignment binds its local, including x = x.
      read(node.expression, *locals);
      if (node.kind == Node::Kind::Set) {
        if (node.text.find('.') == std::string::npos)
          locals->insert(node.text);
        else {
          Expr target = Name(node.text);
          read(target, *locals);
        }
      } else if (node.kind == Node::Kind::If) {
        auto yes = *locals, no = *locals;
        CaptureInputs(node.body, &yes, inputs);
        CaptureInputs(node.otherwise, &no, inputs);
        for (auto it = yes.begin(); it != yes.end();)
          if (!no.count(*it))
            it = yes.erase(it);
          else
            ++it;
        *locals = std::move(yes);
      } else if (node.kind == Node::Kind::For) {
        auto inner = *locals;
        const auto comma = node.text.find(',');
        inner.insert(node.text.substr(0, comma));
        if (comma != std::string::npos)
          inner.insert(node.text.substr(comma + 1));
        if (node.predicate)
          read(*node.predicate, inner);
        inner.insert("loop");
        CaptureInputs(node.body, &inner, inputs);
        auto otherwise = *locals;
        CaptureInputs(node.otherwise, &otherwise, inputs);
      } else if (node.kind == Node::Kind::Capture ||
                 node.kind == Node::Kind::Generation) {
        auto inner = *locals;
        CaptureInputs(node.body, &inner, inputs);
        if (node.kind == Node::Kind::Capture)
          locals->insert(node.text);
      }
    }
  }
  void GatherRecursiveDependencies() {
    // Lexical direct summaries keep private locals out of module dependency
    // closure. The finite graph is also used to recognize private recursion;
    // no call is evaluated or symbolically unrolled here.
    for (const auto &[name, definition] : macros_) {
      std::set<std::string> locals(definition->parameters.begin(),
                                   definition->parameters.end());
      auto &inputs = recursive_dependencies_[name];
      CaptureInputs(definition->body, &locals, &inputs);
      recursive_body_inputs_[name] = inputs;
      GatherCalls(definition->body, &recursive_body_calls_[name]);
      for (const auto &value : definition->defaults)
        References(value, &inputs);
      for (const auto &parameter : definition->parameters)
        for (auto it = inputs.begin(); it != inputs.end();)
          if (*it == parameter ||
              it->compare(0, parameter.size() + 1, parameter + ".") == 0)
            it = inputs.erase(it);
          else
            ++it;
    }
    for (const auto &[name, unused] : macros_) {
      std::set<std::string> pending = recursive_dependencies_[name], seen;
      while (!pending.empty()) {
        const auto input = *pending.begin();
        pending.erase(pending.begin());
        if (!seen.insert(input).second)
          continue;
        if (input == name)
          recursive_macros_.insert(name);
        if (const auto callee = recursive_dependencies_.find(input);
            callee != recursive_dependencies_.end())
          pending.insert(callee->second.begin(), callee->second.end());
      }
      recursive_dependencies_[name] = std::move(seen);
    }
    // A private acyclic wrapper can own loops and effects around its recursive
    // callee. Preserve that original expression as one qualified computation.
    bool changed;
    do {
      changed = false;
      for (const auto &[name, inputs] : recursive_dependencies_)
        if (!recursive_macros_.count(name))
          for (const auto &input : inputs)
            if (recursive_macros_.count(input)) {
              changed |= recursive_macros_.insert(name).second;
              break;
            }
    } while (changed);
  }
  void GatherCalls(const Expr &expression,
                   std::vector<const Expr *> *calls) const {
    if (expression.kind == Expr::Kind::Call && !expression.args.empty() &&
        expression.args[0].kind == Expr::Kind::Name &&
        macros_.count(expression.args[0].text))
      calls->push_back(&expression);
    for (const auto &argument : expression.args)
      GatherCalls(argument, calls);
  }
  void GatherCalls(const std::vector<Node> &nodes,
                   std::vector<const Expr *> *calls) const {
    for (const auto &node : nodes) {
      GatherCalls(node.expression, calls);
      if (node.predicate)
        GatherCalls(*node.predicate, calls);
      GatherCalls(node.body, calls);
      GatherCalls(node.otherwise, calls);
    }
  }
  bool MacroRequestInputs(const Expr &expression,
                          std::set<std::string> *inputs) const {
    std::vector<const Expr *> pending;
    GatherCalls(expression, &pending);
    std::set<std::pair<std::string, std::vector<bool>>> seen;
    while (!pending.empty()) {
      const auto &call = *pending.back();
      pending.pop_back();
      const auto &name = call.args[0].text;
      const auto &definition = *macros_.at(name);
      std::vector<bool> supplied(definition.parameters.size(), false);
      size_t positional = 0;
      for (size_t i = 1; i < call.args.size(); ++i) {
        size_t parameter = positional++;
        if (i < call.keywords.size() && !call.keywords[i].empty()) {
          const auto found =
              std::find(definition.parameters.begin(),
                        definition.parameters.end(), call.keywords[i]);
          parameter = found - definition.parameters.begin();
        }
        if (parameter >= supplied.size() || supplied[parameter])
          return false;
        supplied[parameter] = true;
      }
      // There are finitely many source callsites/provided masks. Reaching the
      // same pair cannot introduce another lazy default or lexical dependency.
      if (!seen.emplace(name, supplied).second)
        continue;
      auto direct = recursive_body_inputs_.at(name);
      const auto &calls = recursive_body_calls_.at(name);
      pending.insert(pending.end(), calls.begin(), calls.end());
      const auto first = supplied.size() - definition.defaults.size();
      for (size_t i = first; i < supplied.size(); ++i)
        if (!supplied[i]) {
          const auto &value = definition.defaults[i - first];
          References(value, &direct);
          GatherCalls(value, &pending);
        }
      for (const auto &input : direct) {
        const auto root = input.substr(0, input.find('.'));
        if (!macros_.count(input) &&
            std::find(definition.parameters.begin(),
                      definition.parameters.end(),
                      root) == definition.parameters.end())
          inputs->insert(input);
      }
    }
    return true;
  }
  bool RequestRecursiveCall(const Expr &call, Path &path,
                            const Context &context,
                            bool *unresolved_namespace = nullptr) {
    if (call.kind != Expr::Kind::Call || call.args.empty() ||
        call.args[0].kind != Expr::Kind::Name ||
        !recursive_macros_.count(call.args[0].text))
      return false;
    const auto request_value = [&](const Value &value) {
      const auto parts = [&](auto &&self, const Parts &items) -> bool {
        for (const auto &part : items)
          if ((part.capture != TextCapture::Literal &&
               part.capture != TextCapture::Expression &&
               part.capture != TextCapture::Choice) ||
              Future(part.predicate) || Future(part.value_expression) ||
              Future(part.body_error) || Future(part.otherwise_error) ||
              !self(self, part.body) || !self(self, part.otherwise))
            return false;
        return true;
      };
      return value.origin == Origin::Request && !Future(value.expression) &&
             parts(parts, value.expansion);
    };
    std::set<std::string> actual_inputs;
    std::set<std::string> lexical_inputs;
    if (!MacroRequestInputs(call, &lexical_inputs))
      return false;
    bool actual_request = true;
    for (size_t i = 1; i < call.args.size(); ++i) {
      actual_request &= request_value(Resolve(call.args[i], path, context));
      References(call.args[i], &actual_inputs);
    }
    for (const auto &guard : path.guards) {
      if (Future(guard))
        actual_request = false;
      References(guard, &actual_inputs);
    }
    const auto request_inputs = [&](std::set<std::string> pending,
                                    const Bindings &bindings) {
      std::set<std::string> seen;
      while (!pending.empty()) {
        const auto input = *pending.begin();
        pending.erase(pending.begin());
        if (!seen.insert(input).second)
          continue;
        if (macros_.count(input))
          continue;
        const auto root = input.substr(0, input.find('.'));
        if (root == "messages" || message_sequences_.count(root) ||
            predicate_bindings_.count(root) || count_target_names_.count(root))
          return false;
        for (const auto &key : {input, root})
          if (const auto binding = bindings.find(key);
              binding != bindings.end()) {
            if (!request_value(binding->second))
              return false;
            References(binding->second.expression, &pending);
            if (!MacroRequestInputs(binding->second.expression, &pending))
              return false;
          }
      }
      return true;
    };
    // Walk intentionally does not replace namespace fields with speculative
    // values. Source/alias dependency closure still records writes, including
    // a future payload assigned after the module constructor.
    std::set<std::string> fields;
    for (const auto &input : lexical_inputs) {
      const auto dot = input.find('.');
      if (dot != std::string::npos &&
          namespace_roots_.count(input.substr(0, dot)))
        fields.insert(input);
    }
    const bool namespace_request =
        request_inputs(NeededClosure(std::move(fields)), path.bindings);
    if (unresolved_namespace)
      *unresolved_namespace = !namespace_request;
    return actual_request && namespace_request &&
           request_inputs(std::move(actual_inputs), path.bindings) &&
           request_inputs(lexical_inputs, path.module_bindings);
  }
  bool RequestCapture(const Node &node, const Path &path,
                      const Context &context) {
    if (!context.request_capture || *context.request_capture != node.offset)
      return false;
    const auto has_loop = [&](auto &&self,
                              const std::vector<Node> &nodes) -> bool {
      for (const auto &child : nodes)
        if (child.kind == Node::Kind::For || self(self, child.body) ||
            self(self, child.otherwise))
          return true;
      return false;
    };
    bool buffered_loop = has_loop(has_loop, node.body);
    // Capture locals must not pick up same-spelled producers in later message
    // loops. Only their source-ordered external inputs enter global closure.
    std::set<std::string> locals, inputs;
    CaptureInputs(node.body, &locals, &inputs);
    std::vector<const Expr *> calls;
    GatherCalls(node.body, &calls);
    for (const auto *call : calls) {
      const auto &name = call->args[0].text;
      buffered_loop |= recursive_macros_.count(name) ||
                       has_loop(has_loop, macros_.at(name)->body);
      if (!MacroRequestInputs(*call, &inputs))
        return false;
    }
    for (const auto &[name, unused] : macros_)
      inputs.erase(name);
    for (const auto &guard : path.guards)
      References(guard, &inputs);
    const auto dependencies = NeededClosure(std::move(inputs));
    const auto request_parts = [&](auto &&self, const Parts &parts) -> bool {
      for (const auto &part : parts)
        if ((part.capture != TextCapture::Literal &&
             part.capture != TextCapture::Expression &&
             part.capture != TextCapture::Choice) ||
            Future(part.predicate) || Future(part.value_expression) ||
            Future(part.body_error) || Future(part.otherwise_error) ||
            Future(part.body_error_statements) ||
            Future(part.otherwise_error_statements) || !self(self, part.body) ||
            !self(self, part.otherwise))
          return false;
      return true;
    };
    for (const auto &name : dependencies) {
      const auto root = name.substr(0, name.find('.'));
      if (root == "messages" || message_sequences_.count(root) ||
          predicate_bindings_.count(root) || count_target_names_.count(root))
        return false;
      if (const auto macro = macros_.find(name); macro != macros_.end())
        buffered_loop |= has_loop(has_loop, macro->second->body) ||
                         recursive_macros_.count(name);
      for (const auto &key : {name, root}) {
        const auto binding = path.bindings.find(key);
        if (binding == path.bindings.end())
          continue;
        const auto &value = binding->second;
        if (value.origin != Origin::Request || Future(value.expression) ||
            !request_parts(request_parts, value.expansion))
          return false;
      }
    }
    // Keep the existing literal and semantic expansion paths where no private
    // repetition needs opacity. A Request label alone never grants this path.
    return buffered_loop;
  }
  // This first-emission proof is deliberately lexical. It recognizes a fresh
  // pending flag, one original message loop and its immediate fallback, rather
  // than treating arbitrary request-origin text as an immutable prompt prefix.
  bool PrefixExpressionStable(const Expr &expression,
                              const std::set<std::string> &forbidden,
                              std::set<std::string> *active) const {
    if (expression.kind == Expr::Kind::Name && forbidden.count(expression.text))
      return false;
    if (expression.kind == Expr::Kind::Call && !expression.args.empty() &&
        expression.args.front().kind == Expr::Kind::Name) {
      const auto &name = expression.args.front().text;
      if (name == "namespace")
        return false;
      if (const auto found = macros_.find(name); found != macros_.end()) {
        if (!active->insert(name).second)
          return false;
        const auto &macro = *found->second;
        bool stable = PrefixNodesStable(macro.body, forbidden, active);
        for (const auto &value : macro.defaults)
          stable &= PrefixExpressionStable(value, forbidden, active);
        active->erase(name);
        if (!stable)
          return false;
      }
    }
    for (const auto &arg : expression.args)
      if (!PrefixExpressionStable(arg, forbidden, active))
        return false;
    return true;
  }
  bool PrefixNodesStable(const std::vector<Node> &nodes,
                         const std::set<std::string> &forbidden,
                         std::set<std::string> *active) const {
    for (const auto &node : nodes) {
      // No namespace writes/escapes or shadowing of a proof identity. Ordinary
      // local call/argument bindings remain normal original source operations.
      if ((node.kind == Node::Kind::Set || node.kind == Node::Kind::Capture ||
           node.kind == Node::Kind::For) &&
          (node.text.find('.') != std::string::npos ||
           forbidden.count(node.text)))
        return false;
      if (node.kind == Node::Kind::Macro ||
          node.kind == Node::Kind::Generation ||
          !PrefixExpressionStable(node.expression, forbidden, active) ||
          (node.predicate &&
           !PrefixExpressionStable(*node.predicate, forbidden, active)) ||
          !PrefixNodesStable(node.body, forbidden, active) ||
          !PrefixNodesStable(node.otherwise, forbidden, active))
        return false;
    }
    return true;
  }
  void ProvePromptPrefix() {
    const auto &nodes = ast_.body;
    size_t start = 0;
    while (start < nodes.size() && nodes[start].kind == Node::Kind::Macro)
      ++start;
    // Direct root siblings establish dominance without moving computations
    // out of an original conditional, capture or generation scope.
    if (nodes.size() != start + 5)
      return;
    const auto &capture = nodes[start], &initial = nodes[start + 1],
               &loop = nodes[start + 2], &fallback = nodes[start + 3],
               &generation = nodes[start + 4];
    const std::set<std::string> controls{"messages", "add_generation_prompt",
                                         "loop"};
    if (capture.kind != Node::Kind::Capture || controls.count(capture.text) ||
        controls.count(initial.text) ||
        !opaque_request_captures_.count(capture.offset) ||
        initial.kind != Node::Kind::Set ||
        initial.expression.kind != Expr::Kind::Call ||
        initial.expression.args.size() != 2 ||
        initial.expression.keywords.size() != 2 ||
        initial.expression.args[0].kind != Expr::Kind::Name ||
        initial.expression.args[0].text != "namespace" ||
        initial.expression.args[1].kind != Expr::Kind::Boolean ||
        initial.expression.args[1].text != "true" ||
        initial.expression.keywords[1].empty() ||
        initial.text.find('.') != std::string::npos ||
        initial.text == capture.text || loop.kind != Node::Kind::For ||
        loop.expression.kind != Expr::Kind::Name ||
        loop.expression.text != "messages" || loop.predicate ||
        !loop.otherwise.empty() || loop.body.empty() ||
        fallback.kind != Node::Kind::If || !fallback.otherwise.empty() ||
        fallback.body.size() != 1 || generation.kind != Node::Kind::If ||
        generation.expression.kind != Expr::Kind::Name ||
        generation.expression.text != "add_generation_prompt" ||
        !generation.otherwise.empty())
      return;
    const auto pending = initial.text + "." + initial.expression.keywords[1];
    const auto &first = loop.body.front();
    if (first.kind != Node::Kind::If || !first.otherwise.empty() ||
        first.body.size() != 2 || first.expression.kind != Expr::Kind::Binary ||
        first.expression.text != "and" || first.expression.args.size() != 2 ||
        Access(first.expression.args[0]) != pending ||
        Access(fallback.expression) != pending)
      return;
    const auto &role = first.expression.args[1];
    if (role.kind != Expr::Kind::Binary || role.text != "!=" ||
        role.args.size() != 2 || Access(role.args[0]) != loop.text + ".role" ||
        role.args[1].kind != Expr::Kind::String ||
        role.args[1].text == "assistant")
      return;
    const auto is_emission = [&](const Node &node) {
      return node.kind == Node::Kind::Output &&
             node.expression.kind == Expr::Kind::Name &&
             node.expression.text == capture.text;
    };
    const auto &clear = first.body[1];
    if (!is_emission(first.body[0]) || !is_emission(fallback.body[0]) ||
        clear.kind != Node::Kind::Set || clear.text != pending ||
        clear.expression.kind != Expr::Kind::Boolean ||
        clear.expression.text != "false")
      return;
    std::set<std::string> forbidden{initial.text, capture.text, "messages",
                                    "add_generation_prompt", "loop"};
    std::set<std::string> active;
    if (forbidden.count(loop.text) ||
        loop.text.find(',') != std::string::npos ||
        !PrefixNodesStable(capture.body, forbidden, &active))
      return;
    const std::vector<Node> body(loop.body.begin() + 1, loop.body.end());
    if (!PrefixNodesStable(body, forbidden, &active))
      return;
    for (const auto &node : generation.body)
      if (node.kind != Node::Kind::Text &&
          !(node.kind == Node::Kind::Output &&
            node.expression.kind == Expr::Kind::String))
        return;
    completion_prompt_prefixes_.push_back({{first.offset, first.end},
                                           Range(first.body[0].expression),
                                           capture.text});
  }
  void ForgetCaptureWrites(const Node &capture, Path *path) {
    for (const auto &[field, unused] : definitions_) {
      const auto dot = field.find('.');
      if (dot == std::string::npos ||
          !namespace_roots_.count(field.substr(0, dot)))
        continue;
      Path probe = *path;
      if (!MacroBlockMayWrite(capture.body, field, &probe, {}))
        continue;
      Expr access;
      access.kind = Expr::Kind::GetAttr;
      access.text = field.substr(dot + 1);
      access.args = {Name(field.substr(0, dot))};
      Value runtime{Origin::Request, std::move(access)};
      path->bindings[field] = runtime;
      path->module_bindings[field] = std::move(runtime);
    }
  }
  bool RecordTargetRegion(const Context &context, size_t offset,
                          bool opaque_call) {
    if (projection_mode_ || continuation_mode_ || !context.target ||
        !completion_message_loops_.count(context.target_loop))
      return true;
    auto &own = opaque_call ? recursive_target_call_loops_
                            : recursive_target_namespace_writes_;
    const auto &other = opaque_call ? recursive_target_namespace_writes_
                                    : recursive_target_call_loops_;
    if (other.count(context.target_loop))
      return Fail(
          offset,
          "target namespace statements mixed with request recursion require single-region recipe retention");
    // Whole request-only wrappers execute natively, so their internal writes
    // never enter this symbolic statement walk. Separately expanded statements
    // cannot share their original region with an extracted opaque call yet.
    own.insert(context.target_loop);
    return true;
  }
  std::vector<Path> Walk(const std::vector<Node> &nodes,
                         std::vector<Path> paths, const Context &context) {
    for (const auto &node : nodes) {
      current_statement_ = node.offset;
      std::vector<Path> next;
      for (auto &path : paths) {
        if (path.rejected) {
          next.push_back(std::move(path));
          continue;
        }
        if (node.kind == Node::Kind::Text) {
          if (!projection_mode_)
            Append(&path.parts, Literal(node.text, {node.offset, node.end}));
          next.push_back(std::move(path));
        } else if (node.kind == Node::Kind::Output) {
          const size_t prior_parts = path.parts.size();
          Context emission = context;
          if (emission.target && !emission.source_output_owner)
            emission.source_output_owner = node.offset;
          auto emitted = Emit(node.expression, std::move(path), emission);
          for (auto &item : emitted) {
            if (!projection_mode_ && context.target &&
                !context.continued_user && !item.rejected &&
                item.parts.size() >= prior_parts) {
              const Parts added(item.parts.begin() + prior_parts,
                                item.parts.end());
              if (HasCapture(added, TextCapture::Reasoning))
                completion_reasoning_outputs_.insert(node.offset);
            }
            next.push_back(std::move(item));
          }
        } else if (node.kind == Node::Kind::Capture) {
          if (RequestCapture(node, path, context)) {
            opaque_request_captures_.insert(node.offset);
            ForgetCaptureWrites(node, &path);
            Value value{Origin::Request,
                        Name(node.text, {node.offset, node.end})};
            value.defined = true;
            value.string_result = true;
            path.bindings[node.text] = std::move(value);
            next.push_back(std::move(path));
            continue;
          }
          Path captured = path;
          captured.parts.clear();
          Context capture_context = context;
          capture_context.failing_loop.reset();
          capture_context.allow_statement_errors = false;
          auto bodies = Walk(node.body, {std::move(captured)}, capture_context);
          if (bodies.size() != 1) {
            Fail(node.offset, "capture did not form one buffered value");
            continue;
          }
          auto &body = bodies.front();
          std::vector<const Expr *> recursive_effects;
          for (const auto &part : body.parts)
            RecursiveEffects(part, &recursive_effects);
          if (context.target && !recursive_effects.empty()) {
            Fail(
                node.offset,
                "target-local recursive capture requires single-region recipe retention");
            continue;
          }
          if (body.rejected) {
            path.rejected = true;
            path.rejection = body.rejection;
            path.rejection_statements = body.rejection_statements;
          }
          // Ordinary capture locals do not escape. Existing namespace fields
          // remain effects, just as in renderer lowering and retained prelude.
          for (const auto &[name, value] : body.bindings)
            if (name.find('.') != std::string::npos)
              path.bindings[name] = value;
          Value value{Origin::Request,
                      Name(node.text, {node.offset, node.end})};
          value.expansion = std::move(body.parts);
          value.defined = true;
          value.string_result = true;
          std::string literal;
          if (LiteralText(value.expansion, &literal)) {
            value.string = literal;
            value.truth = !literal.empty();
          }
          path.bindings[node.text] = std::move(value);
          next.push_back(std::move(path));
        } else if (node.kind == Node::Kind::Set) {
          if (node.text.find('.') != std::string::npos &&
              !RecordTargetRegion(context, node.offset, false))
            continue;
          if (context.target && node.text.find('.') == std::string::npos) {
            std::vector<const Expr *> calls;
            GatherCalls(node.expression, &calls);
            if (std::any_of(calls.begin(), calls.end(), [&](const auto *call) {
                  return recursive_macros_.count(call->args[0].text);
                })) {
              const auto value = Resolve(node.expression, path, context);
              calls.clear();
              GatherCalls(value.expression, &calls);
              if (std::any_of(
                      calls.begin(), calls.end(), [&](const auto *call) {
                        return recursive_macros_.count(call->args[0].text);
                      })) {
                Fail(
                    node.offset,
                    "target-local recursive assignment requires single-region recipe retention");
                continue;
              }
            }
          }
          if (!Relevant(node.text, analysis_needed_)) {
            if (projection_mode_)
              Resolve(node.expression, path, context);
            path.bindings[node.text] = {Origin::Request, Name(node.text)};
            next.push_back(std::move(path));
            continue;
          }
          Value value = Resolve(node.expression, path, context);
          if (value.last_expression &&
              node.text.find('.') != std::string::npos && !projection_mode_ &&
              context.target && !context.continued_user)
            completion_rewrite_diagnostics_[node.offset] =
                "a401 completion cannot qualify a last-sensitive namespace write";
          if (node.expression.kind == Expr::Kind::Call &&
              !node.expression.args.empty() &&
              node.expression.args[0].kind == Expr::Kind::Name &&
              node.expression.args[0].text == "namespace")
            value.expression = Name(node.text, {node.offset, node.end});
          if (projection_mode_ && node.expression.kind == Expr::Kind::Call &&
              !node.expression.args.empty() &&
              node.expression.args[0].kind == Expr::Kind::Name &&
              macros_.count(node.expression.args[0].text)) {
            auto projections = Emit(node.expression, std::move(path), context);
            for (auto &projection : projections) {
              projection.bindings[node.text] = value;
              next.push_back(std::move(projection));
            }
            continue;
          }
          if (node.text.find('.') == std::string::npos)
            path.bindings[node.text] = std::move(value);
          next.push_back(std::move(path));
        } else if (node.kind == Node::Kind::If) {
          Value condition = Resolve(node.expression, path, context);
          if (condition.last_expression && !projection_mode_ &&
              context.target && !context.continued_user &&
              !context.loops.empty() &&
              context.loops.back().first == Origin::Messages) {
            const auto [entry, inserted] = completion_last_conditions_.emplace(
                node.offset, *condition.last_expression);
            if (!inserted &&
                !SameExpression(entry->second, *condition.last_expression))
              completion_rewrite_diagnostics_[node.offset] =
                  "a401 completion cannot snapshot divergent last-sensitive source predicates";
          }
          if (Truth(condition)) {
            auto branch = Walk(*Truth(condition) ? node.body : node.otherwise,
                               {std::move(path)}, context);
            for (auto &item : branch)
              next.push_back(std::move(item));
          } else {
            // Future value predicates are parser choices, not request routes.
            Path yes_path = path, no_path = path;
            yes_path.parts.clear();
            no_path.parts.clear();
            if (Future(condition.expression)) {
              Narrow(condition.expression, true, &yes_path);
              Narrow(condition.expression, false, &no_path);
            } else if (projection_mode_) {
              yes_path.guards.push_back(condition.expression);
              no_path.guards.push_back(Unary("not", condition.expression));
            }
            const auto arm_context = [&](const std::vector<Node> &arm) {
              Context result = context;
              result.failing_loop.reset();
              if (context.allow_statement_errors && !projection_mode_ &&
                  Future(condition.expression) && !arm.empty() &&
                  arm.front().kind == Node::Kind::For && !arm.front().predicate)
                result.failing_loop = arm.front().offset;
              return result;
            };
            auto yes =
                Walk(node.body, {std::move(yes_path)}, arm_context(node.body));
            auto no = Walk(node.otherwise, {std::move(no_path)},
                           arm_context(node.otherwise));
            if (yes.size() == 1 && no.size() == 1) {
              RecordRetentionChoice(node.offset, condition);
              TextFormatPart choice;
              choice.capture = TextCapture::Choice;
              choice.predicate = condition.expression;
              choice.source = {node.offset, node.end};
              choice.body = std::move(yes[0].parts);
              choice.otherwise = std::move(no[0].parts);
              if (yes[0].rejection)
                choice.body_error = *yes[0].rejection;
              if (no[0].rejection)
                choice.otherwise_error = *no[0].rejection;
              if (yes[0].rejection_statements)
                choice.body_error_statements = *yes[0].rejection_statements;
              if (no[0].rejection_statements)
                choice.otherwise_error_statements = *no[0].rejection_statements;
              // A guarded phi preserves definitions without enumerating the
              // product of unrelated request/representation decisions.
              JoinBindings(condition.expression, yes[0].bindings,
                           no[0].bindings, &path.bindings, false, node.offset);
              path.adaptation.arguments_need_mapping |=
                  yes[0].adaptation.arguments_need_mapping ||
                  no[0].adaptation.arguments_need_mapping;
              path.adaptation.arguments_observe_string |=
                  yes[0].adaptation.arguments_observe_string ||
                  no[0].adaptation.arguments_observe_string;
              path.adaptation.content_observes_representation |=
                  yes[0].adaptation.content_observes_representation ||
                  no[0].adaptation.content_observes_representation;
              if (!choice.body.empty() || !choice.otherwise.empty() ||
                  choice.HasBodyError() || choice.HasOtherwiseError())
                Append(&path.parts, std::move(choice));
              next.push_back(std::move(path));
            } else {
              Fail(
                  node.offset,
                  "control-flow branch did not form one guarded definition join");
            }
          }
        } else if (node.kind == Node::Kind::For) {
          if (node.predicate) {
            Fail(
                node.predicate->offset,
                "filtered repetition is unsupported in output format analysis");
            continue;
          }
          Value sequence = Resolve(node.expression, path, context);
          if (context.allow_statement_errors && !projection_mode_ &&
              context.failing_loop == node.offset && DirectSemantic(sequence) &&
              Domain(sequence.origin, path) == 2) {
            Node failure = node;
            failure.expression = sequence.expression;
            failure.body.clear();
            failure.otherwise.clear();
            path.rejected = true;
            path.rejection_statements = Template{{std::move(failure)}};
            next.push_back(std::move(path));
            continue;
          }
          if (projection_mode_ && (sequence.origin == Origin::Arguments ||
                                   sequence.origin == Origin::ArgumentItems ||
                                   sequence.origin == Origin::ArgumentKeys)) {
            ObserveProjection(0, path);
            next.push_back(std::move(path));
            continue;
          }
          if (projection_mode_ && (sequence.origin == Origin::Content ||
                                   sequence.origin == Origin::Reasoning)) {
            ObserveProjection(2, path);
            next.push_back(std::move(path));
            continue;
          }
          if (projection_mode_ && sequence.origin != Origin::Calls) {
            // This pass selects projection of the explicit current call. Tool
            // definitions/content-reference serialization is original source
            // history work, not another native call-arguments projection.
            next.push_back(std::move(path));
            continue;
          }
          Context nested = context;
          nested.failing_loop.reset();
          nested.allow_statement_errors = false;
          nested.loops.push_back({sequence.origin, sequence.expression});
          Path repeated = path;
          repeated.parts.clear();
          repeated.bindings["loop"] = {Origin::Loop, Name("loop")};
          TextCapture repeat = TextCapture::RepeatArguments;
          if (sequence.origin == Origin::Calls) {
            repeated.bindings[node.text] =
                Semantic(Origin::Call, node.expression);
            if (projection_mode_) {
              Expr present;
              present.kind = Expr::Kind::Test;
              present.text = "defined";
              present.args.push_back(Name(worker_call_));
              repeated.guards.push_back(std::move(present));
            }
            repeat = TextCapture::RepeatCalls;
          } else if (sequence.origin == Origin::Arguments ||
                     sequence.origin == Origin::ArgumentKeys) {
            ObserveProjection(0, path);
            repeated.bindings[node.text] =
                Semantic(Origin::ArgumentKey, node.expression);
          } else if (sequence.origin == Origin::ArgumentItems &&
                     node.text.find(',') != std::string::npos) {
            const auto comma = node.text.find(',');
            repeated.bindings[node.text.substr(0, comma)] =
                Semantic(Origin::ArgumentKey, node.expression);
            repeated.bindings[node.text.substr(comma + 1)] =
                Semantic(Origin::ArgumentValue, node.expression);
          } else {
            Fail(
                node.offset,
                "assistant body contains an unsupported repeated semantic sequence");
            continue;
          }
          auto bodies = Walk(node.body, {std::move(repeated)}, nested);
          for (auto &body : bodies) {
            if (body.rejected)
              continue;
            Path result = path;
            // Projection observations inside the simulated current-call body
            // already retain their source guards. Loop-local conditions must
            // not constrain statements following the original loop.
            if (!projection_mode_)
              result.guards = body.guards;
            result.adaptation = body.adaptation;
            TextFormatPart part;
            part.capture = repeat;
            part.body = std::move(body.parts);
            part.source = {node.offset, node.end};
            Append(&result.parts, std::move(part));
            next.push_back(std::move(result));
          }
        } else if (node.kind == Node::Kind::Generation) {
          auto children = Walk(node.body, {std::move(path)}, context);
          for (auto &item : children)
            next.push_back(std::move(item));
        } else {
          next.push_back(std::move(path));
        }
      }
      paths = std::move(next);
      if (failed_)
        return {};
    }
    return paths;
  }

  bool MentionsGeneration(const Expr &expression) const {
    return ContainsName(expression, "add_generation_prompt");
  }
  bool RootMessageSequence(const Expr &expression, const Path &path) const {
    if (expression.kind != Expr::Kind::Name)
      return false;
    const auto binding = path.bindings.find(expression.text);
    return binding != path.bindings.end() &&
           binding->second.origin == Origin::Messages &&
           binding->second.root_index_identity;
  }
  // Finite positive role sets, all excluding the actual appended target role.
  // Return their one unchanged root-message index; no arbitrary list analysis.
  std::optional<int64_t> PrefixRoleIndex(const Expr &expression,
                                         const Path &path) const {
    if (expression.kind != Expr::Kind::Binary || expression.args.size() != 2)
      return std::nullopt;
    if (expression.text == "or") {
      const auto a = PrefixRoleIndex(expression.args[0], path);
      const auto b = PrefixRoleIndex(expression.args[1], path);
      return a && b && *a == *b ? a : std::nullopt;
    }
    const auto &role = expression.args[0];
    const auto member = MemberName(role);
    if (!member || *member != "role" || role.args.empty())
      return std::nullopt;
    const auto &item = role.args[0];
    if (item.kind != Expr::Kind::GetItem || item.args.size() != 2 ||
        !RootMessageSequence(item.args[0], path))
      return std::nullopt;
    const auto index = GuardInteger(item.args[1]);
    if (!index || *index < 0)
      return std::nullopt;
    const auto target = continuation_mode_ ? "user" : "assistant";
    const auto excluded = [&](const Expr &value) {
      return value.kind == Expr::Kind::String && value.text != target;
    };
    const auto &roles = expression.args[1];
    if (expression.text == "==")
      return excluded(roles) ? index : std::nullopt;
    if (expression.text == "in" &&
        (roles.kind == Expr::Kind::List || roles.kind == Expr::Kind::Tuple) &&
        !roles.args.empty() &&
        std::all_of(roles.args.begin(), roles.args.end(), excluded))
      return index;
    return std::nullopt;
  }
  bool AppendInvariant(const Expr &expression, const Path &path) const {
    if (expression.kind == Expr::Kind::Integer ||
        expression.kind == Expr::Kind::Boolean ||
        expression.kind == Expr::Kind::String ||
        expression.kind == Expr::Kind::None)
      return true;
    if (expression.kind == Expr::Kind::Name) {
      const auto binding = path.bindings.find(expression.text);
      return binding != path.bindings.end() && binding->second.append_invariant;
    }
    if (expression.kind == Expr::Kind::Unary && expression.text == "not" &&
        expression.args.size() == 1)
      return AppendInvariant(expression.args[0], path);
    if (const auto index = PrefixRoleIndex(expression, path))
      return *index == 0 && path.source_nonempty;
    if (expression.kind != Expr::Kind::Binary || expression.args.size() != 2 ||
        (expression.text != "and" && expression.text != "or"))
      return false;
    if (expression.text == "and") {
      const auto &bound = expression.args[0];
      const auto index = PrefixRoleIndex(expression.args[1], path);
      if (index && bound.kind == Expr::Kind::Binary && bound.text == ">" &&
          bound.args.size() == 2) {
        const auto &length = bound.args[0];
        if (length.kind == Expr::Kind::Filter &&
            (length.text == "length" || length.text == "count") &&
            length.args.size() == 1 &&
            RootMessageSequence(length.args[0], path) &&
            GuardInteger(bound.args[1]) == index)
          return true;
      }
    }
    return AppendInvariant(expression.args[0], path) &&
           AppendInvariant(expression.args[1], path);
  }
  bool RejectsEmptyRoot(const Node &node, const Path &path) const {
    const auto &condition = node.expression;
    return condition.kind == Expr::Kind::Unary && condition.text == "not" &&
           condition.args.size() == 1 &&
           RootMessageSequence(condition.args[0], path) &&
           node.body.size() == 1 && node.otherwise.empty() &&
           node.body.front().kind == Node::Kind::Output && Raises(node.body);
  }
  std::string BranchSnapshot(const Node &node, const Expr &condition,
                             bool invariant) {
    auto found = branch_snapshots_.find(node.offset);
    if (found != branch_snapshots_.end()) {
      if (invariant_snapshots_.count(found->second) != invariant &&
          CountDependent(condition))
        Fail(node.offset,
             "branch target invariance differs across source paths");
      return found->second;
    }
    std::string name = "__text_branch_" + std::to_string(node.offset);
    while (source_names_.count(name) || snapshot_names_.count(name))
      name += '_';
    snapshot_names_.insert(name);
    selector_binding_expressions_[name] = condition;
    References(condition, &definitions_[name]);
    if (invariant)
      invariant_snapshots_.insert(name);
    if (CountDependent(condition) && !invariant) {
      const auto target =
          FreshCountName("__text_target_branch_" + std::to_string(node.offset));
      count_targets_[name] = target;
      count_target_names_.insert(target);
      const auto expression = TargetCounts(condition);
      selector_binding_expressions_[target] = expression;
      References(expression, &definitions_[target]);
      branch_target_snapshots_[node.offset] = {target, expression};
    }
    branch_snapshots_[node.offset] = name;
    return name;
  }
  std::vector<Path> ScanTop(const std::vector<Node> &nodes,
                            std::vector<Path> paths) {
    for (const auto &node : nodes) {
      current_statement_ = node.offset;
      if (std::all_of(paths.begin(), paths.end(),
                      [](const Path &path) { return path.rejected; }))
        return paths;
      if (node.kind == Node::Kind::Macro)
        continue;
      if (node.kind == Node::Kind::Output &&
          node.expression.kind == Expr::Kind::Call &&
          !node.expression.args.empty() &&
          node.expression.args[0].kind == Expr::Kind::Name &&
          node.expression.args[0].text == "raise_exception") {
        if (!continuation_mode_)
          for (auto &path : paths)
            path = Emit(node.expression, std::move(path), {}).front();
      } else if (node.kind == Node::Kind::Capture) {
        Context capture_context;
        capture_context.request_capture = node.offset;
        paths = Walk({node}, std::move(paths), capture_context);
        for (auto &path : paths)
          path.module_bindings[node.text] = path.bindings[node.text];
      } else if (node.kind == Node::Kind::Set) {
        for (auto &path : paths) {
          if (!Relevant(node.text, analysis_needed_)) {
            path.bindings[node.text] = {Origin::Request, Name(node.text)};
            path.module_bindings[node.text] = path.bindings[node.text];
            continue;
          }
          const bool invariant = AppendInvariant(node.expression, path);
          const bool root_identity = RootMessageSequence(node.expression, path);
          Value value = Resolve(node.expression, path, {});
          value.append_invariant = invariant;
          value.root_index_identity = root_identity;
          if (!projection_mode_ && value.origin != Origin::Messages &&
              CountDependent(value.expression) &&
              PureCountScalar(value.expression)) {
            auto found = scalar_counts_.find(node.offset);
            size_t id;
            if (found == scalar_counts_.end()) {
              const Expr ordinary = value.expression;
              id = RegisterCount(ordinary, node.offset, false, false);
              FinishTargetCount(id, TargetCounts(ordinary));
              count_affine_definitions_[count_definitions_[id].target] =
                  TargetCounts(ordinary);
              scalar_counts_[node.offset] = id;
            } else
              id = found->second;
            value.expression = Name(count_definitions_[id].ordinary);
          }
          if (node.expression.kind == Expr::Kind::Call &&
              !node.expression.args.empty() &&
              node.expression.args[0].kind == Expr::Kind::Name &&
              node.expression.args[0].text == "namespace")
            value.expression = Name(node.text, {node.offset, node.end});
          if (value.origin == Origin::Messages)
            value.expression = Name(node.text, {node.offset, node.end});
          path.bindings[node.text] = std::move(value);
          path.module_bindings[node.text] = path.bindings[node.text];
        }
      } else if (!projection_mode_ && node.kind == Node::Kind::If &&
                 MentionsGeneration(node.expression) &&
                 (HasOutput(node.body) || HasOutput(node.otherwise))) {
        for (auto path : paths) {
          path.parts.clear();
          auto emitted = Walk({node}, {std::move(path)}, {});
          for (auto &item : emitted)
            if (!item.rejected)
              generations_.push_back(std::move(item));
        }
      } else if (node.kind == Node::Kind::For) {
        if (!projection_mode_ && !HasOutput(node.body))
          continue;
        for (auto path : paths) {
          Value sequence = Resolve(node.expression, path, {});
          if (sequence.origin != Origin::Messages)
            continue;
          if (node.predicate) {
            Fail(
                node.predicate->offset,
                "filtered message iteration is unsupported in output format analysis");
            continue;
          }
          if (!projection_mode_ && !continuation_mode_)
            completion_message_loops_.insert(node.offset);
          if (!projection_mode_ && !continuation_mode_ &&
              !sequence.append_preserving) {
            Fail(
                node.offset,
                "target message iteration is not an append-preserving source sequence");
            continue;
          }
          Context context;
          context.target = true;
          context.continued_user = continuation_mode_;
          context.target_sequence = sequence.sequence;
          context.target_loop = node.offset;
          context.messages = sequence.expression;
          context.loops.push_back({Origin::Messages, sequence.expression});
          if (!projection_mode_ && !continuation_mode_)
            for (auto &guard : path.guards)
              guard = TargetCounts(guard);
          if (!projection_mode_ && !continuation_mode_)
            for (const auto &guard : sequence.target_survival)
              path.guards.push_back(guard);
          if (continuation_mode_) {
            path.domains[Origin::Content] = 4;
            path.domains[Origin::Reasoning] = 1;
            path.domains[Origin::Calls] = 1;
          }
          path.bindings[node.text] = {
              Origin::Message,
              Name(projection_mode_ ? worker_message_ : node.text)};
          path.bindings["loop"] = {Origin::Loop, Name("loop")};
          path.parts.clear();
          auto emitted = Walk(node.body, {std::move(path)}, context);
          for (auto &item : emitted)
            if (projection_mode_)
              projection_paths_.push_back(std::move(item));
            else if (!item.rejected &&
                     (HasCapture(item.parts, TextCapture::Content) ||
                      HasCapture(item.parts, TextCapture::Reasoning) ||
                      HasCapture(item.parts, TextCapture::RepeatCalls))) {
              item.emission = node.offset;
              bodies_.push_back(std::move(item));
            }
        }
      } else if (node.kind == Node::Kind::If) {
        std::vector<Path> joined;
        for (auto &path : paths) {
          const bool nonempty_source = RejectsEmptyRoot(node, path);
          const bool invariant = AppendInvariant(node.expression, path);
          if (nonempty_source && !continuation_mode_)
            selector_effect_outputs_.insert(node.body.front().offset);
          Value condition = Resolve(node.expression, path, {});
          if (continuation_mode_) {
            const auto name =
                BranchSnapshot(node, condition.expression, invariant);
            const auto known = Truth(condition);
            const auto [record, inserted] =
                default_conditions_.emplace(name, known);
            if (!inserted && record->second != known)
              record->second.reset();
          }
          if (Truth(condition)) {
            auto selected =
                ScanTop(*Truth(condition) ? node.body : node.otherwise,
                        {std::move(path)});
            for (auto &item : selected) {
              if (nonempty_source && !item.rejected)
                item.source_nonempty = true;
              joined.push_back(std::move(item));
            }
          } else {
            const auto incoming_guards = path.guards;
            Path otherwise = path;
            // Preserve the value at its original source point: re-evaluating
            // "x is undefined" after its default assignment, or messages[0]
            // after slicing messages, changes the guard's meaning.
            Expr guard =
                Name(BranchSnapshot(node, condition.expression, invariant),
                     {node.offset, node.end});
            path.guards.push_back(guard);
            otherwise.guards.push_back(Unary("not", guard));
            auto yes = ScanTop(node.body, {std::move(path)});
            auto no = ScanTop(node.otherwise, {std::move(otherwise)});
            if (yes.size() == 1 && no.size() == 1 &&
                yes[0].rejected != no[0].rejected) {
              joined.push_back(yes[0].rejected ? std::move(no[0])
                                               : std::move(yes[0]));
            } else if (yes.size() == 1 && no.size() == 1) {
              Path merged = yes[0];
              merged.guards = incoming_guards;
              merged.source_nonempty =
                  yes[0].source_nonempty && no[0].source_nonempty;
              JoinBindings(guard, yes[0].bindings, no[0].bindings,
                           &merged.bindings, true, node.offset, invariant);
              JoinBindings(guard, yes[0].module_bindings, no[0].module_bindings,
                           &merged.module_bindings, true, node.offset,
                           invariant);
              joined.push_back(std::move(merged));
            } else if (yes.empty()) {
              for (auto &item : no)
                joined.push_back(std::move(item));
            } else if (no.empty()) {
              for (auto &item : yes)
                joined.push_back(std::move(item));
            } else {
              Fail(
                  node.offset,
                  "module control flow did not form one guarded definition join");
            }
          }
        }
        paths = std::move(joined);
      }
      if (failed_)
        return {};
    }
    return paths;
  }
  void SlicePrelude(const std::vector<Node> &nodes, std::vector<Node> *out) {
    for (const auto &node : nodes) {
      EmitCountDefinitions(node.offset, false, false, out);
      if (node.kind == Node::Kind::Output) {
        if (selector_effect_outputs_.count(node.offset)) {
          auto evaluate = node;
          evaluate.kind = Node::Kind::Set;
          evaluate.text = FreshCountName("__text_discarded_output");
          if (const auto target = recursive_target_outputs_.find(node.offset);
              target != recursive_target_outputs_.end()) {
            Node historical;
            historical.kind = Node::Kind::If;
            historical.expression =
                Name(recursive_history_guards_.at(target->second));
            historical.body.push_back(std::move(evaluate));
            out->push_back(std::move(historical));
          } else {
            out->push_back(std::move(evaluate));
          }
        }
        continue;
      }
      if (node.kind == Node::Kind::Text)
        continue;
      if (node.kind == Node::Kind::For && node.predicate) {
        // Preserve lazy predicate/body effects and errors, while consuming the
        // ordinary source text privately rather than emitting selector pairs.
        Node capture;
        capture.kind = Node::Kind::Capture;
        capture.text = FreshCountName("__text_filtered_output");
        capture.offset = node.offset;
        capture.end = node.end;
        capture.body = {node};
        out->push_back(std::move(capture));
        continue;
      }
      if (node.kind == Node::Kind::Set &&
          !Relevant(node.text, selector_needed_))
        continue;
      if (node.kind == Node::Kind::Macro &&
          !Relevant(node.text, selector_needed_))
        continue;
      Node copy = node;
      if (const auto snapshot = branch_snapshots_.find(node.offset);
          node.kind == Node::Kind::If && snapshot != branch_snapshots_.end() &&
          Relevant(snapshot->second, selector_needed_)) {
        Node retain;
        retain.kind = Node::Kind::Set;
        retain.text = snapshot->second;
        retain.expression = selector_binding_expressions_.at(snapshot->second);
        retain.offset = node.offset;
        retain.end = node.end;
        out->push_back(std::move(retain));
        copy.expression = Name(snapshot->second, {node.offset, node.end});
      }
      if (const auto target = branch_target_snapshots_.find(node.offset);
          node.kind == Node::Kind::If &&
          target != branch_target_snapshots_.end() &&
          selector_needed_.count(target->second.first)) {
        Node set;
        set.kind = Node::Kind::Set;
        set.text = target->second.first;
        set.expression = target->second.second;
        out->push_back(std::move(set));
      }
      // Captures execute once and retain their full buffered source and
      // effects, including when their string result is not used by the
      // selector.
      if (node.kind != Node::Kind::Macro && node.kind != Node::Kind::Capture) {
        copy.body.clear();
        copy.otherwise.clear();
        if (const auto historical = recursive_history_guards_.find(node.offset);
            node.kind == Node::Kind::For &&
            historical != recursive_history_guards_.end()) {
          Node snapshot;
          snapshot.kind = Node::Kind::Set;
          snapshot.text = historical->second;
          Expr last;
          last.kind = Expr::Kind::GetAttr;
          last.text = "last";
          last.args = {Name("loop")};
          snapshot.expression = Binary("or", Name(target_missing_),
                                       Unary("not", std::move(last)));
          copy.body.push_back(std::move(snapshot));
        }
        SlicePrelude(node.body, &copy.body);
        SlicePrelude(node.otherwise, &copy.otherwise);
      }
      if ((node.kind == Node::Kind::If || node.kind == Node::Kind::For ||
           node.kind == Node::Kind::Generation) &&
          copy.body.empty() && copy.otherwise.empty())
        continue;
      out->push_back(std::move(copy));
      EmitCountDefinitions(node.offset, true, false, out);
    }
  }
  bool LiteralText(const Parts &parts, std::string *out) const {
    for (const auto &part : parts) {
      if (part.capture != TextCapture::Literal)
        return false;
      *out += part.literal;
    }
    return true;
  }
  std::string Before(const Parts &parts, TextCapture capture) const {
    std::string bytes;
    for (const auto &part : parts) {
      if (part.capture == capture)
        return bytes;
      if (part.capture == TextCapture::Literal)
        bytes += part.literal;
      else if (const auto *found = FindCapture(part.body, capture)) {
        (void)found;
        return bytes + Before(part.body, capture);
      } else if (const auto *found = FindCapture(part.otherwise, capture)) {
        (void)found;
        return bytes + Before(part.otherwise, capture);
      } else {
        std::string literal;
        if ((part.capture == TextCapture::Choice ||
             part.capture == TextCapture::RepeatCalls ||
             part.capture == TextCapture::RepeatArguments) &&
            SuccessfulLiteral(part, &literal))
          bytes += literal;
        else
          bytes.clear();
      }
    }
    return {};
  }
  std::string After(const Parts &parts, TextCapture capture) const {
    bool seen = false;
    std::string bytes;
    for (const auto &part : parts) {
      if (part.capture == capture) {
        seen = true;
        bytes.clear();
      } else if (seen) {
        if (part.capture == TextCapture::Literal)
          bytes += part.literal;
        else {
          std::string literal;
          if ((part.capture == TextCapture::Choice ||
               part.capture == TextCapture::RepeatCalls ||
               part.capture == TextCapture::RepeatArguments) &&
              SuccessfulLiteral(part, &literal))
            bytes += literal;
          else
            break;
        }
      } else if (FindCapture(part.body, capture)) {
        bytes = After(part.body, capture);
        seen = true;
      } else if (FindCapture(part.otherwise, capture)) {
        bytes = After(part.otherwise, capture);
        seen = true;
      }
    }
    return bytes;
  }
  std::string FirstClose(std::string_view after) const {
    size_t begin = after.find("</");
    const size_t square = after.find("[/");
    if (square != std::string_view::npos &&
        (begin == std::string_view::npos || square < begin))
      begin = square;
    if (begin == std::string_view::npos)
      return {};
    const char terminator = after[begin] == '[' ? ']' : '>';
    const size_t end = after.find(terminator, begin);
    return end == std::string_view::npos
               ? std::string()
               : std::string(after.substr(begin, end - begin + 1));
  }
  std::string ReasoningOpen(std::string_view before,
                            std::string_view close) const {
    if (close.size() < 4 || close[1] != '/')
      return {};
    std::string prefix(1, close[0]);
    prefix += close.substr(2, close.size() - 3);
    const auto begin = before.rfind(prefix);
    if (begin == std::string_view::npos)
      return {};
    const char terminator = close.back();
    const auto end = before.find(terminator, begin + prefix.size());
    if (end == std::string_view::npos)
      return {};
    return std::string(before.substr(begin, end - begin + 1));
  }
  std::string JsonField(std::string_view before) const {
    size_t colon = before.rfind(':');
    if (colon == std::string_view::npos)
      return {};
    size_t end = colon;
    while (end && (before[end - 1] == ' ' || before[end - 1] == '\t' ||
                   before[end - 1] == '\n' || before[end - 1] == '\r'))
      --end;
    if (!end || before[end - 1] != '"')
      return {};
    --end;
    size_t begin = end;
    bool found = false;
    while (begin) {
      --begin;
      if (before[begin] != '"')
        continue;
      size_t escapes = 0;
      for (size_t at = begin; at && before[at - 1] == '\\'; --at)
        ++escapes;
      if (!(escapes & 1)) {
        found = true;
        break;
      }
    }
    if (!found)
      return {};
    const auto escaped = before.substr(begin + 1, end - begin - 1);
    std::string decoded(escaped.size(), '\0');
    iree_host_size_t length = 0;
    const auto status = iree_json_unescape_string(
        iree_make_string_view(escaped.data(), escaped.size()), decoded.size(),
        decoded.data(), &length);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return {};
    }
    decoded.resize(length);
    return decoded;
  }
  std::string Leading(const Parts &parts) const {
    std::string result;
    for (const auto &part : parts) {
      if (part.capture == TextCapture::Literal) {
        result += part.literal;
      } else if (part.capture == TextCapture::Choice) {
        std::string literal;
        if (part.HasBodyError() && part.HasOtherwiseError())
          return result;
        if (SuccessfulLiteral(part, &literal)) {
          result += literal;
          continue;
        }
        if (part.HasBodyError())
          return result + Leading(part.otherwise);
        if (part.HasOtherwiseError())
          return result + Leading(part.body);
        const auto a = Leading(part.body), b = Leading(part.otherwise);
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n])
          ++n;
        return result + a.substr(0, n);
      } else if (part.capture == TextCapture::Expression) {
        return result + Leading(part.body);
      } else {
        return result;
      }
    }
    return result;
  }
  std::string Trailing(const Parts &parts) const {
    std::string result;
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
      const auto &part = *it;
      if (part.capture == TextCapture::Literal) {
        result = part.literal + result;
      } else if (part.capture == TextCapture::Choice) {
        std::string literal;
        if (part.HasBodyError() && part.HasOtherwiseError())
          return result;
        if (SuccessfulLiteral(part, &literal)) {
          result = literal + result;
          continue;
        }
        if (part.HasBodyError())
          return Trailing(part.otherwise) + result;
        if (part.HasOtherwiseError())
          return Trailing(part.body) + result;
        const auto a = Trailing(part.body), b = Trailing(part.otherwise);
        size_t n = 0;
        while (n < a.size() && n < b.size() &&
               a[a.size() - n - 1] == b[b.size() - n - 1])
          ++n;
        return a.substr(a.size() - n) + result;
      } else if (part.capture == TextCapture::Expression) {
        return Trailing(part.body) + result;
      } else {
        return result;
      }
    }
    return result;
  }
  bool SuccessfulLiteral(const TextFormatPart &part, std::string *out) const {
    if (part.HasBodyError() && part.HasOtherwiseError())
      return false;
    auto literal = [&](const Parts &items, std::string *bytes) -> bool {
      for (const auto &item : items) {
        if (item.capture == TextCapture::Literal)
          *bytes += item.literal;
        else if (item.capture == TextCapture::Choice ||
                 item.capture == TextCapture::RepeatCalls ||
                 item.capture == TextCapture::RepeatArguments) {
          std::string nested;
          if (!SuccessfulLiteral(item, &nested))
            return false;
          *bytes += nested;
        } else {
          return false;
        }
      }
      return true;
    };
    if (part.capture == TextCapture::RepeatCalls ||
        part.capture == TextCapture::RepeatArguments) {
      // A source loop may perform validation without emitting any bytes.
      // Retain it in the recipe; only its proven empty successful emission
      // is transparent to adjacent delimiter/member-name tracing.
      std::string repeated;
      return literal(part.body, &repeated) && repeated.empty();
    }
    if (part.HasBodyError())
      return literal(part.otherwise, out);
    if (part.HasOtherwiseError())
      return literal(part.body, out);
    std::string yes, no;
    if (!literal(part.body, &yes) || !literal(part.otherwise, &no) || yes != no)
      return false;
    *out += yes;
    return true;
  }
  bool Describe(const Parts &parts, TextFormatAlternative *out) {
    out->assistant_pattern = parts;
    auto evidence = [&](auto &&self, const Parts &items) -> void {
      for (const auto &item : items) {
        if (item.source.end > item.source.begin)
          out->evidence.push_back(item.source);
        self(self, item.body);
        self(self, item.otherwise);
      }
    };
    evidence(evidence, parts);
    std::vector<const TextFormatPart *> call_patterns;
    auto gather_calls = [&](auto &&self, const Parts &items) -> void {
      for (const auto &item : items) {
        if (item.capture == TextCapture::RepeatCalls)
          call_patterns.push_back(&item);
        self(self, item.body);
        self(self, item.otherwise);
      }
    };
    gather_calls(gather_calls, parts);
    if (call_patterns.size() > 1) {
      // A future-dependent branch is not a request route. Until its separate
      // call grammars have a structural discriminator in the output emitter,
      // reject it instead of retaining only the first branch's field names.
      return Fail(
          call_patterns[1]->source.begin,
          "future-dependent multiple call bodies require explicit structural alternatives");
    }
    std::optional<std::string> reasoning_header;
    const auto *reason = FindCapture(parts, TextCapture::Reasoning);
    if (reason) {
      const auto before = Before(parts, TextCapture::Reasoning);
      const auto after = After(parts, TextCapture::Reasoning);
      out->reasoning_close = FirstClose(after);
      out->reasoning_open = ReasoningOpen(before, out->reasoning_close);
      if (out->reasoning_open.empty() || out->reasoning_close.empty())
        return Fail(reason->source.begin,
                    "reasoning capture lacks a bounded literal frame");
      const auto open_at = before.rfind(out->reasoning_open);
      reasoning_header = before.substr(0, open_at);
      out->reasoning_after_open =
          before.substr(open_at + out->reasoning_open.size());
      const auto close_at = after.find(out->reasoning_close);
      out->reasoning_before_close = after.substr(0, close_at);
      out->reasoning_after_close =
          after.substr(close_at + out->reasoning_close.size());
    }
    // The header ends at the first semantic body transition. Keep frame bytes
    // separate from markers: the final turn token is not a tool delimiter.
    std::string leading = Leading(parts);
    if (reasoning_header) {
      // The source range before the complete reasoning-open delimiter is
      // frame data. A common byte prefix of open/close marker alternatives
      // (for example their first '<') is still body framing, not a header.
      size_t common = 0;
      while (common < leading.size() && common < reasoning_header->size() &&
             leading[common] == (*reasoning_header)[common])
        ++common;
      leading.resize(common);
    }
    if (!out->reasoning_open.empty()) {
      const auto at = leading.find(out->reasoning_open);
      if (at != std::string::npos)
        leading.resize(at);
    }
    out->assistant_header = std::move(leading);
    out->frame_suffix = Trailing(parts);
    const auto *calls = FindCapture(parts, TextCapture::RepeatCalls);
    if (!calls) {
      out->ambiguous_bare_json_tools =
          HasCapture(parts, TextCapture::FunctionName) &&
          HasCapture(parts, TextCapture::Arguments);
      out->historical_single_call = out->ambiguous_bare_json_tools;
      return true;
    }
    out->tools.call_body = calls->body;
    const auto *name = FindCapture(calls->body, TextCapture::FunctionName);
    if (!name)
      return Fail(calls->source.begin,
                  "tool call does not expose a bounded function-name capture");
    std::string name_before = Before(calls->body, TextCapture::FunctionName);
    const auto json_open = name_before.find('{');
    if (json_open != std::string::npos) {
      auto prefix = name_before.substr(0, json_open);
      const auto first = prefix.find_first_not_of(" \t\r\n");
      if (first == std::string::npos) {
        out->tools.kind = TextToolKind::None;
        out->ambiguous_bare_json_tools = true;
        return true;
      }
      const auto last = prefix.find_last_not_of(" \t\r\n");
      out->tools.open = prefix.substr(first, last - first + 1);
      std::string after = After(calls->body, TextCapture::Arguments);
      const auto object_end = after.find('}');
      if (object_end == std::string::npos)
        return Fail(name->source.begin,
                    "JSON call lacks its literal object close");
      const auto close_first =
          after.find_first_not_of(" \t\r\n", object_end + 1);
      const auto close_last = after.find_last_not_of(" \t\r\n");
      if (close_first == std::string::npos)
        return Fail(name->source.begin,
                    "JSON call lacks its literal envelope close");
      out->tools.close =
          after.substr(close_first, close_last - close_first + 1);
      out->tools.kind = TextToolKind::JsonEnvelope;
      out->tools.name_field = JsonField(name_before);
      out->tools.arguments_field =
          JsonField(Before(calls->body, TextCapture::Arguments));
      if (out->tools.name_field.empty() || out->tools.arguments_field.empty())
        return Fail(name->source.begin,
                    "JSON call member name is invalid or unclassified");
      out->evidence.push_back(calls->source);
      return true;
    }
    const auto envelope_at = name_before.find('<');
    if (envelope_at == std::string::npos) {
      // Plain history may contain bare JSON. Its presence does not prove an
      // unambiguous output entry, and does not enable a new bare-JSON mode.
      out->tools.kind = TextToolKind::None;
      out->ambiguous_bare_json_tools = true;
      return true;
    }
    const auto envelope_end = name_before.find('>', envelope_at);
    if (envelope_end == std::string::npos)
      return Fail(name->source.begin,
                  "incomplete tool envelope before function-name capture");
    out->tools.open =
        name_before.substr(envelope_at, envelope_end - envelope_at + 1);
    std::string trailing;
    if (!calls->body.empty() &&
        calls->body.back().capture == TextCapture::Literal)
      trailing = calls->body.back().literal;
    const auto last_close = trailing.rfind("</");
    if (last_close == std::string::npos)
      return Fail(calls->source.begin,
                  "tool envelope has no literal closing boundary");
    const auto close_end = trailing.find('>', last_close);
    if (close_end == std::string::npos)
      return Fail(calls->source.begin,
                  "tool envelope has an incomplete closing boundary");
    out->tools.close = trailing.substr(last_close, close_end - last_close + 1);
    if (name_before.find('{', envelope_end) != std::string::npos) {
      out->tools.kind = TextToolKind::JsonEnvelope;
      out->tools.name_field = JsonField(name_before);
      out->tools.arguments_field =
          JsonField(Before(calls->body, TextCapture::Arguments));
      if (out->tools.name_field.empty() || out->tools.arguments_field.empty())
        return Fail(
            calls->source.begin,
            "JSON tool envelope lacks literal name/argument field paths");
    } else {
      const auto inner = name_before.rfind('<');
      out->tools.kind =
          inner != std::string::npos &&
                  name_before.find('=', inner) != std::string::npos
              ? TextToolKind::TaggedNamed
              : TextToolKind::TaggedKeyValue;
      if (!FindCapture(calls->body, TextCapture::RepeatArguments))
        return Fail(calls->source.begin,
                    "tagged tool call lacks a supported argument repetition");
    }
    out->evidence.push_back(calls->source);
    return true;
  }
  bool QualifyRetentionStreams(std::vector<TextFormatAlternative> *formats,
                               const std::vector<const Path *> &sources) {
    auto ordinary_guards = [&](const Path &path) {
      std::vector<Expr> result;
      for (const auto &guard : path.guards) {
        bool retention = false;
        for (const auto &[offset, decision] : path.retention) {
          const auto retained = decision.selected
                                    ? decision.predicate
                                    : Unary("not", decision.predicate);
          retention |= SameExpression(guard, retained);
        }
        if (!retention)
          result.push_back(guard);
      }
      return result;
    };
    for (size_t receiver = 0; receiver < formats->size(); ++receiver) {
      auto &plain = (*formats)[receiver];
      const auto *source = sources[receiver];
      if (!source || source->retention.size() != 1 ||
          !plain.reasoning_open.empty() ||
          HasCapture(plain.assistant_pattern, TextCapture::Reasoning))
        continue;
      const auto &[choice, retention] = *source->retention.begin();
      if (retention.keeps)
        continue;
      const TextFormatAlternative *frame = nullptr;
      for (size_t donor = 0; donor < formats->size(); ++donor) {
        const auto *other_source = sources[donor];
        const auto &other = (*formats)[donor];
        if (!other_source || other_source->emission != source->emission ||
            other_source->retention.size() != 1 ||
            !other_source->retention.count(choice))
          continue;
        const auto &other_retention = other_source->retention.at(choice);
        if (!other_retention.keeps ||
            other_retention.search != retention.search ||
            other_retention.selected == retention.selected ||
            !SameExpression(other_retention.predicate, retention.predicate) ||
            !SameGuards(ordinary_guards(*source),
                        ordinary_guards(*other_source)) ||
            other.reasoning_open.empty() || other.reasoning_close.empty() ||
            !HasCapture(other.assistant_pattern, TextCapture::Reasoning) ||
            plain.assistant_header != other.assistant_header ||
            plain.frame_suffix != other.frame_suffix ||
            plain.tools.kind != other.tools.kind ||
            plain.tools.open != other.tools.open ||
            plain.tools.close != other.tools.close ||
            plain.tools.name_field != other.tools.name_field ||
            plain.tools.arguments_field != other.tools.arguments_field)
          continue;
        if (frame && (frame->reasoning_open != other.reasoning_open ||
                      frame->reasoning_close != other.reasoning_close))
          return Fail(
              choice,
              "historical retention siblings have ambiguous stream reasoning frames");
        frame = &other;
      }
      if (frame) {
        plain.reasoning_open = frame->reasoning_open;
        plain.reasoning_close = frame->reasoning_close;
        plain.reasoning_after_open = frame->reasoning_after_open;
        plain.reasoning_before_close = frame->reasoning_before_close;
        plain.reasoning_after_close = frame->reasoning_after_close;
        plain.evidence.insert(plain.evidence.end(), frame->evidence.begin(),
                              frame->evidence.end());
      }
    }
    return true;
  }
  bool Merge(const TextFormatAlternative &from, TextFormatAlternative *to) {
    if ((HasStatementError(from.assistant_pattern) ||
         HasStatementError(to->assistant_pattern)) &&
        !SameErrorPattern(from.assistant_pattern, to->assistant_pattern))
      return Fail(0, "cannot merge distinct guarded source-statement errors");
    to->completion_guards.erase(
        std::remove_if(
            to->completion_guards.begin(), to->completion_guards.end(),
            [&](const Expr &guard) {
              return std::none_of(from.completion_guards.begin(),
                                  from.completion_guards.end(),
                                  [&](const Expr &other) {
                                    return SameExpression(guard, other);
                                  });
            }),
        to->completion_guards.end());
    to->ambiguous_bare_json_tools |= from.ambiguous_bare_json_tools;
    to->historical_single_call |= from.historical_single_call;
    if (to->reasoning_open.empty()) {
      to->reasoning_open = from.reasoning_open;
      to->reasoning_close = from.reasoning_close;
      to->reasoning_after_open = from.reasoning_after_open;
      to->reasoning_before_close = from.reasoning_before_close;
      to->reasoning_after_close = from.reasoning_after_close;
      if (!from.reasoning_open.empty())
        to->assistant_pattern = from.assistant_pattern;
    } else if (!from.reasoning_open.empty() &&
               (to->reasoning_open != from.reasoning_open ||
                to->reasoning_close != from.reasoning_close)) {
      return Fail(
          from.evidence.empty() ? 0 : from.evidence[0].begin,
          "future-dependent reasoning frames are not structurally unique");
    }
    if (to->tools.kind == TextToolKind::None)
      to->tools = from.tools;
    else if (from.tools.kind != TextToolKind::None &&
             (to->tools.kind != from.tools.kind ||
              to->tools.open != from.tools.open ||
              to->tools.close != from.tools.close ||
              to->tools.name_field != from.tools.name_field ||
              to->tools.arguments_field != from.tools.arguments_field))
      return Fail(
          0,
          "future-dependent tool formats require disjoint structural alternatives");
    if (to->assistant_header.empty())
      to->assistant_header = from.assistant_header;
    if (to->frame_suffix.empty())
      to->frame_suffix = from.frame_suffix;
    to->evidence.insert(to->evidence.end(), from.evidence.begin(),
                        from.evidence.end());
    return true;
  }

  std::string_view source_;
  std::string_view source_name_;
  const Template &ast_;
  const TextFormatBindings &globals_;
  Diagnostic *diagnostic_ = nullptr;
  bool failed_ = false;
  const std::map<std::string, const Node *> &macros_;
  std::set<std::string> macro_stack_;
  std::map<std::string, std::set<std::string>> recursive_dependencies_;
  std::map<std::string, std::set<std::string>> recursive_body_inputs_;
  std::map<std::string, std::vector<const Expr *>> recursive_body_calls_;
  std::set<std::string> recursive_macros_;
  std::map<size_t, size_t> recursive_target_outputs_;
  std::map<size_t, std::string> recursive_history_guards_;
  std::set<size_t> recursive_target_call_loops_;
  std::set<size_t> recursive_target_namespace_writes_;
  std::vector<Path> bodies_;
  std::vector<Path> generations_;
  std::vector<Path> projection_paths_;
  std::vector<Expr> projection_rules_[3];
  std::vector<Expr> projection_raw_mapping_rules_;
  std::vector<Expr> projection_rejection_rules_;
  bool projection_mode_ = false;
  const std::set<std::string> &source_names_;
  std::set<std::string> generated_names_;
  std::map<Origin, std::string> semantic_names_;
  std::map<std::string, TextCapture> predicate_bindings_;
  std::map<size_t, std::string> branch_snapshots_;
  std::set<std::string> snapshot_names_;
  std::set<std::string> invariant_snapshots_;
  std::map<std::string, Expr> selector_binding_expressions_;
  std::string worker_message_;
  std::string worker_message_index_;
  std::string worker_call_;
  std::string worker_call_index_;
  std::map<std::string, std::set<std::string>> definitions_;
  std::map<std::string, std::set<std::string>> literal_definitions_;
  std::set<std::string> analysis_roots_;
  std::set<std::string> analysis_needed_;
  std::set<std::string> literal_roots_;
  std::set<std::string> literal_needed_;
  std::set<std::string> selector_needed_;
  std::set<size_t> selector_effect_outputs_;
  bool resolving_last_ = false;
  size_t resolve_depth_ = 0;
  std::set<size_t> completion_message_loops_;
  std::set<size_t> opaque_request_captures_;
  std::vector<TextPromptPrefix> completion_prompt_prefixes_;
  std::map<size_t, Expr> completion_last_conditions_;
  std::set<size_t> completion_reasoning_outputs_;
  std::map<size_t, std::string> completion_rewrite_diagnostics_;
  std::set<std::string> message_sequences_{"messages"};
  std::set<std::string> namespace_roots_;
  std::vector<HistorySearch> history_searches_;
  const std::vector<const Expr *> &macro_calls_;
  std::map<size_t, std::pair<size_t, bool>> retention_choices_;
  std::set<size_t> ambiguous_retention_;
  std::string target_missing_;
  bool resolving_target_ = false;
  bool continuation_mode_ = false;
  std::string user_ordinal_;
  std::map<std::string, std::optional<bool>> default_conditions_;
  size_t current_statement_ = 0;
  size_t next_sequence_ = 0;
  std::vector<CountDefinition> count_definitions_;
  std::map<size_t, size_t> sequence_counts_;
  std::map<std::string, std::string> count_targets_;
  std::set<std::string> count_target_names_;
  std::map<std::string, Expr> count_affine_definitions_;
  std::map<std::pair<size_t, size_t>, size_t> sequence_versions_;
  std::map<std::pair<size_t, std::string>, size_t> message_joins_;
  std::map<std::pair<size_t, std::string>, size_t> survival_counts_;
  std::map<size_t, size_t> scalar_counts_;
  std::map<size_t, std::pair<std::string, Expr>> branch_target_snapshots_;
};

// This is a finite encoding classifier, not an expression evaluator. Only
// source type tests over the one captured JSON value can select wire forms.
class ValueCodecClassifier {
public:
  explicit ValueCodecClassifier(const TextFormatAnalysis &analysis)
      : analysis_(analysis) {}
  bool Run(const TextFormatPart &part, TextValueCodec *out,
           std::string *error) {
    error_ = error;
    if (!out)
      return Fail("value codec needs an output slot");
    std::set<TextCapture> captures;
    Captures(part, &captures);
    if (captures.size() != 1)
      return Fail("value encoding must refer to exactly one semantic capture");
    capture_ = *captures.begin();
    std::vector<Encoding> encodings;
    for (int type = 0; type != 6; ++type) {
      Encoding encoding;
      if (!PartEncoding(part, type, &encoding))
        return false;
      encodings.push_back(encoding);
    }
    auto matches = [&](auto expected) {
      for (int type = 0; type != 6; ++type)
        if (encodings[type].json != expected(type))
          return false;
      return true;
    };
    TextValueCodec codec;
    if (matches([](int) { return false; }))
      codec.wire = TextValueWire::RawScalar;
    else if (matches([](int) { return true; }))
      codec.wire = TextValueWire::Json;
    else if (matches([](int type) { return type != 0; }))
      codec.wire = TextValueWire::StringElseJson;
    else if (matches([](int type) { return type >= 4; }))
      codec.wire = TextValueWire::JsonContainersElseScalar;
    else if (matches([](int type) { return type != 0 && type != 3; }))
      codec.wire = TextValueWire::JsonContainersAndLiteralsElseScalar;
    else
      return Fail(
          "type-dependent value encoding has no supported output wire form");
    for (const auto &encoding : encodings)
      codec.trim |= encoding.trim;
    *out = codec;
    if (error_)
      error_->clear();
    return true;
  }

private:
  struct Encoding {
    bool json = false;
    bool trim = false;
    bool coerced = false;
  };
  bool Fail(std::string message) {
    if (error_)
      *error_ = std::move(message);
    return false;
  }
  void Captures(const TextFormatPart &part, std::set<TextCapture> *out) {
    switch (part.capture) {
    case TextCapture::Content:
    case TextCapture::Reasoning:
    case TextCapture::FunctionName:
    case TextCapture::Arguments:
    case TextCapture::ArgumentKey:
    case TextCapture::ArgumentValue:
      out->insert(part.capture);
      break;
    default:
      break;
    }
    auto referenced = [&](auto &&self, const Expr &expression) -> void {
      if (expression.kind == Expr::Kind::Name) {
        const auto found = analysis_.predicate_bindings.find(expression.text);
        if (found != analysis_.predicate_bindings.end()) {
          switch (found->second) {
          case TextCapture::Content:
          case TextCapture::Reasoning:
          case TextCapture::FunctionName:
          case TextCapture::Arguments:
          case TextCapture::ArgumentKey:
          case TextCapture::ArgumentValue:
            out->insert(found->second);
            break;
          default:
            break;
          }
        }
      }
      for (const auto &argument : expression.args)
        self(self, argument);
    };
    referenced(referenced, part.predicate);
    referenced(referenced, part.value_expression);
    for (const auto &child : part.body)
      Captures(child, out);
    for (const auto &child : part.otherwise)
      Captures(child, out);
  }
  bool Subject(const Expr &expression) const {
    if (expression.kind != Expr::Kind::Name)
      return false;
    const auto found = analysis_.predicate_bindings.find(expression.text);
    return found != analysis_.predicate_bindings.end() &&
           found->second == capture_;
  }
  std::optional<bool> TypePredicate(const Expr &expression, int type) const {
    // Types: string, null, boolean, number, array, object.
    if (expression.kind == Expr::Kind::Boolean)
      return expression.text == "true";
    if (expression.kind == Expr::Kind::Test && expression.args.size() == 1 &&
        Subject(expression.args[0])) {
      if (expression.text == "string")
        return type == 0;
      if (expression.text == "none")
        return type == 1;
      if (expression.text == "boolean")
        return type == 2;
      if (expression.text == "number")
        return type == 3;
      if (expression.text == "mapping")
        return type == 5;
      if (expression.text == "sequence")
        return type == 0 || type == 4;
      if (expression.text == "iterable")
        return type == 0 || type >= 4;
      if (expression.text == "defined")
        return true;
      if (expression.text == "undefined")
        return false;
    }
    if (expression.kind == Expr::Kind::Unary && expression.text == "not" &&
        expression.args.size() == 1) {
      if (const auto value = TypePredicate(expression.args[0], type))
        return !*value;
    }
    if (expression.kind == Expr::Kind::Binary && expression.args.size() == 2 &&
        (expression.text == "and" || expression.text == "or")) {
      const auto a = TypePredicate(expression.args[0], type);
      const auto b = TypePredicate(expression.args[1], type);
      if (expression.text == "and") {
        if ((a && !*a) || (b && !*b))
          return false;
        if (a && b)
          return *a && *b;
      } else {
        if ((a && *a) || (b && *b))
          return true;
        if (a && b)
          return *a || *b;
      }
    }
    return std::nullopt;
  }
  bool JsonArguments(const Expr &expression) const {
    int64_t indent;
    return internal::ToJsonIndent(expression, &indent);
  }
  bool LiteralExpression(const Expr &expression, std::string *out) const {
    if (expression.kind == Expr::Kind::String) {
      *out += expression.text;
      return true;
    }
    return expression.kind == Expr::Kind::Binary &&
           expression.args.size() == 2 &&
           (expression.text == "+" || expression.text == "~") &&
           LiteralExpression(expression.args[0], out) &&
           LiteralExpression(expression.args[1], out);
  }
  bool LiteralSequence(const std::vector<TextFormatPart> &parts,
                       std::string *out) const {
    for (const auto &part : parts) {
      if (part.capture == TextCapture::Literal)
        *out += part.literal;
      else if (part.capture != TextCapture::Expression ||
               !LiteralExpression(part.value_expression, out))
        return false;
    }
    return true;
  }
  bool BooleanLiterals(const Expr &predicate, std::string_view yes,
                       std::string_view no) const {
    if (Subject(predicate))
      return yes == "true" && no == "false";
    return predicate.kind == Expr::Kind::Unary && predicate.text == "not" &&
           predicate.args.size() == 1 && Subject(predicate.args[0]) &&
           yes == "false" && no == "true";
  }
  bool PartSequenceEncoding(const std::vector<TextFormatPart> &parts, int type,
                            Encoding *out) {
    std::string literal;
    if (type == 1 && LiteralSequence(parts, &literal) && literal == "null") {
      *out = {true};
      return true;
    }
    const TextFormatPart *value = nullptr;
    for (const auto &part : parts) {
      if (part.capture == TextCapture::Literal && part.literal.empty())
        continue;
      if (value || part.capture == TextCapture::Literal)
        return Fail(
            "value codec contains additional literal or capture framing");
      value = &part;
    }
    return value ? PartEncoding(*value, type, out)
                 : Fail("source value type has no captured encoding");
  }
  bool PartEncoding(const TextFormatPart &part, int type, Encoding *out) {
    if (part.capture != TextCapture::Choice)
      return EncodingFor(part.value_expression, type, out);
    if (type == 2 && !part.HasBodyError() && !part.HasOtherwiseError()) {
      std::string yes, no;
      if (LiteralSequence(part.body, &yes) &&
          LiteralSequence(part.otherwise, &no) &&
          BooleanLiterals(part.predicate, yes, no)) {
        *out = {true};
        return true;
      }
    }
    if (const auto decision = TypePredicate(part.predicate, type)) {
      if (*decision ? part.HasBodyError() : part.HasOtherwiseError())
        return Fail("source rejects this captured value type");
      return PartSequenceEncoding(*decision ? part.body : part.otherwise, type,
                                  out);
    }
    Encoding yes, no;
    if (part.HasBodyError() || part.HasOtherwiseError() ||
        !PartSequenceEncoding(part.body, type, &yes) ||
        !PartSequenceEncoding(part.otherwise, type, &no))
      return Fail("value codec has an unsupported conditional source domain");
    if (yes.json != no.json || yes.trim != no.trim)
      return Fail(
          "value codec depends on an unsupported non-type source predicate");
    *out = yes;
    out->coerced |= no.coerced;
    return true;
  }
  bool EncodingFor(const Expr &expression, int type, Encoding *out) {
    if (type == 1 && expression.kind == Expr::Kind::String &&
        expression.text == "null") {
      *out = {true};
      return true;
    }
    if (Subject(expression)) {
      *out = {};
      return true;
    }
    if (expression.kind == Expr::Kind::Binary && expression.args.size() == 2 &&
        (expression.text == "+" || expression.text == "~")) {
      for (size_t empty = 0; empty != 2; ++empty)
        if (expression.args[empty].kind == Expr::Kind::String &&
            expression.args[empty].text.empty()) {
          if (!EncodingFor(expression.args[1 - empty], type, out))
            return false;
          out->coerced = true;
          return true;
        }
    }
    if (expression.kind == Expr::Kind::Conditional &&
        expression.args.size() == 3) {
      if (type == 2) {
        std::string yes, no;
        if (LiteralExpression(expression.args[1], &yes) &&
            LiteralExpression(expression.args[2], &no) &&
            BooleanLiterals(expression.args[0], yes, no)) {
          *out = {true};
          return true;
        }
      }
      if (const auto decision = TypePredicate(expression.args[0], type))
        return EncodingFor(expression.args[*decision ? 1 : 2], type, out);
      Encoding yes, no;
      if (!EncodingFor(expression.args[1], type, &yes) ||
          !EncodingFor(expression.args[2], type, &no))
        return false;
      if (yes.json != no.json || yes.trim != no.trim)
        return Fail(
            "value codec depends on an unsupported non-type source predicate");
      *out = yes;
      out->coerced |= no.coerced;
      return true;
    }
    if (expression.kind == Expr::Kind::Filter && !expression.args.empty()) {
      if (!EncodingFor(expression.args[0], type, out))
        return false;
      if (expression.text == "safe" && expression.args.size() == 1)
        return true;
      if (expression.text == "string" && expression.args.size() == 1) {
        out->coerced = true;
        return true;
      }
      if (expression.text == "trim" && expression.args.size() == 1 &&
          !out->json) {
        out->trim = true;
        out->coerced = true;
        return true;
      }
      if (expression.text == "tojson" && JsonArguments(expression) &&
          !out->json && !out->trim && !out->coerced) {
        out->json = true;
        return true;
      }
      return Fail("unsupported value codec filter or transform chain: " +
                  expression.text);
    }
    if (expression.kind == Expr::Kind::Call && expression.args.size() == 1 &&
        expression.args[0].kind == Expr::Kind::GetAttr &&
        expression.args[0].text == "strip" &&
        expression.args[0].args.size() == 1) {
      if (!EncodingFor(expression.args[0].args[0], type, out))
        return false;
      if (out->json)
        return Fail("JSON value trimming is not a raw scalar codec");
      out->trim = true;
      out->coerced = true;
      return true;
    }
    return Fail("source value expression has no supported capture encoding");
  }
  const TextFormatAnalysis &analysis_;
  TextCapture capture_ = TextCapture::Literal;
  std::string *error_ = nullptr;
};
} // namespace

bool ClassifyTextValueCodec(const TextFormatPart &part,
                            const TextFormatAnalysis &analysis,
                            TextValueCodec *out, std::string *error) {
  return ValueCodecClassifier(analysis).Run(part, out, error);
}

bool HasHistoryReasoning(const TextFormatAlternative &format) {
  return HasCapture(format.assistant_pattern, TextCapture::Reasoning);
}

bool DecodeTextAdaptation(int64_t mask, TextAdaptation *out) {
  constexpr int64_t known = kTextAdaptationMappingArguments |
                            kTextAdaptationArgumentStrings |
                            kTextAdaptationOriginalContent;
  if (!out || mask < 0 || (mask & ~known) != 0)
    return false;
  TextAdaptation value;
  value.arguments_observe_string = (mask & kTextAdaptationArgumentStrings) != 0;
  value.arguments_need_mapping =
      (mask & kTextAdaptationMappingArguments) != 0 &&
      !value.arguments_observe_string;
  value.content_observes_representation =
      (mask & kTextAdaptationOriginalContent) != 0;
  *out = std::move(value);
  return true;
}

bool internal::AnalyzeTextFormat(const SourceTemplate &unit,
                                 const TextFormatBindings &bindings,
                                 TextFormatAnalysis *out,
                                 Diagnostic *diagnostic) {
  const auto source = unit.source(), source_name = unit.source_name();
  Diagnostic error;
  if (!out) {
    error.message = "format analysis needs an output slot";
  } else if (Analysis(unit, bindings, &error).Run(out)) {
    if (diagnostic)
      *diagnostic = {};
    return true;
  }
  LocateDiagnostic(source, source_name, &error);
  if (diagnostic)
    *diagnostic = std::move(error);
  return false;
}

bool internal::AnalyzeUserContinuation(const SourceTemplate &unit,
                                       const TextFormatBindings &bindings,
                                       const TextFormatAnalysis &formats,
                                       TextUserContinuation *out,
                                       Diagnostic *diagnostic) {
  const auto source = unit.source(), source_name = unit.source_name();
  Diagnostic error;
  if (!out) {
    error.message = "continued USER analysis needs an output slot";
  } else if (Analysis(unit, bindings, &error).RunContinuation(formats, out)) {
    if (diagnostic)
      *diagnostic = {};
    return true;
  }
  LocateDiagnostic(source, source_name, &error);
  if (diagnostic)
    *diagnostic = std::move(error);
  return false;
}
bool AnalyzeTextFormat(std::string_view source, std::string_view source_name,
                       const TextFormatBindings &bindings,
                       TextFormatAnalysis *out, Diagnostic *diagnostic) {
  Diagnostic error;
  std::unique_ptr<internal::SourceTemplate> unit;
  if (!out) {
    error.message = "format analysis needs an output slot";
  } else if (internal::SourceTemplate::Parse(source, source_name, &unit,
                                             &error) &&
             internal::AnalyzeTextFormat(*unit, bindings, out, &error)) {
    if (diagnostic)
      *diagnostic = {};
    return true;
  }
  LocateDiagnostic(source, source_name, &error);
  if (diagnostic)
    *diagnostic = std::move(error);
  return false;
}

bool AnalyzeUserContinuation(std::string_view source,
                             std::string_view source_name,
                             const TextFormatBindings &bindings,
                             const TextFormatAnalysis &formats,
                             TextUserContinuation *out,
                             Diagnostic *diagnostic) {
  Diagnostic error;
  std::unique_ptr<internal::SourceTemplate> unit;
  if (!out) {
    error.message = "continued USER analysis needs an output slot";
  } else if (internal::SourceTemplate::Parse(source, source_name, &unit,
                                             &error) &&
             internal::AnalyzeUserContinuation(*unit, bindings, formats, out,
                                               &error)) {
    if (diagnostic)
      *diagnostic = {};
    return true;
  }
  LocateDiagnostic(source, source_name, &error);
  if (diagnostic)
    *diagnostic = std::move(error);
  return false;
}
} // namespace hrx::loomify_gguf::jinja
