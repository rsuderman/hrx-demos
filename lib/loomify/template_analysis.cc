// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/template_analysis_internal.h"

#include "lib/loomify/template_probe_runner.h"
#include "lib/loomify/text/jinja/emission.h"
#include "lib/loomify/text/jinja/source_template.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::jinja::internal {
using Assignment = std::map<std::vector<std::string>, std::string>;
std::string Quote(std::string_view text) { return QuoteJsonString(text); }

std::string Json(const Object &object) {
  std::string result = "{";
  bool first = true;
  for (const auto &[key, value] : object) {
    if (value == kMissing)
      continue;
    if (!first)
      result += ',';
    first = false;
    result += Quote(key) + ':' + value;
  }
  return result + '}';
}

std::string Array(const std::vector<std::string> &values) {
  std::string result = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      result += ',';
    result += values[i];
  }
  return result + ']';
}

struct BindingTree {
  std::string value;
  std::map<std::string, BindingTree> children;
  std::string Serialize() const {
    if (children.empty())
      return value;
    Object fields;
    for (const auto &[key, child] : children)
      fields[key] = child.Serialize();
    return Json(fields);
  }
};

void Apply(const Assignment &assignment, Object *environment) {
  std::map<std::string, BindingTree> roots;
  for (const auto &[path, value] : assignment) {
    if (path.empty())
      continue;
    BindingTree *node = &roots[path.front()];
    for (size_t i = 1; i < path.size(); ++i)
      node = &node->children[path[i]];
    node->value = value;
  }
  for (const auto &[key, root] : roots) {
    if (root.value == kMissing && root.children.empty())
      environment->erase(key);
    else
      (*environment)[key] = root.Serialize();
  }
}

bool Path(const Expr &expression, std::vector<std::string> *out) {
  if (expression.kind == Expr::Kind::Name) {
    out->push_back(expression.text);
    return true;
  }
  if (expression.kind == Expr::Kind::Call && expression.args.size() >= 2 &&
      expression.args[0].kind == Expr::Kind::GetAttr &&
      expression.args[0].text == "get" && !expression.args[0].args.empty() &&
      expression.args[1].kind == Expr::Kind::String &&
      Path(expression.args[0].args[0], out)) {
    out->push_back(expression.args[1].text);
    return true;
  }
  if ((expression.kind == Expr::Kind::GetAttr ||
       expression.kind == Expr::Kind::GetItem) &&
      !expression.args.empty()) {
    if (!Path(expression.args[0], out))
      return false;
    if (expression.kind == Expr::Kind::GetAttr) {
      out->push_back(expression.text);
      return true;
    }
    if (expression.args.size() == 2 &&
        expression.args[1].kind == Expr::Kind::String) {
      out->push_back(expression.args[1].text);
      return true;
    }
  }
  out->clear();
  return false;
}

bool Literal(const Expr &expression, std::string *out) {
  switch (expression.kind) {
  case Expr::Kind::String:
    *out = Quote(expression.text);
    return true;
  case Expr::Kind::Integer:
    *out = expression.text;
    return true;
  case Expr::Kind::Boolean:
    *out = expression.text;
    return true;
  case Expr::Kind::None:
    *out = "null";
    return true;
  default:
    return false;
  }
}

bool Merge(const Assignment &a, const Assignment &b, Assignment *out) {
  *out = a;
  for (const auto &[path, value] : b) {
    const auto found = out->find(path);
    if (found != out->end() && found->second != value)
      return false;
    (*out)[path] = value;
  }
  return true;
}

void Deduplicate(std::vector<Assignment> *assignments) {
  std::set<Assignment> seen;
  const auto end = std::remove_if(
      assignments->begin(), assignments->end(),
      [&](const Assignment &value) { return !seen.insert(value).second; });
  assignments->erase(end, assignments->end());
}

using ResolveProbePath = std::function<bool(std::vector<std::string> *path)>;
using ProbeSuggestion = std::function<bool(const Assignment &)>;

bool Suggest(std::vector<std::string> path, std::string value,
             const Assignment &prefix, const ResolveProbePath &resolve,
             const ProbeSuggestion &consume) {
  // Local/snapshot conditions cannot be set through request kwargs. Their
  // actual value is established by native execution of the selector slice.
  if (!resolve(&path))
    return consume(prefix);
  Assignment result;
  return Merge(prefix, Assignment{{std::move(path), std::move(value)}}, &result)
             ? consume(result)
             : false;
}

// Construct finite input suggestions from supported guard syntax. This is not a
// condition evaluator: namespace/local expressions may yield no suggestion,
// and only actual native execution can establish a route. Missing coverage is
// reported instead of treating an unconstructed condition as false.
bool VisitGuardCandidates(const Expr &expression, bool truth,
                          const Assignment &prefix,
                          const ResolveProbePath &resolve,
                          const ProbeSuggestion &consume) {
  if (expression.kind == Expr::Kind::Unary && expression.text == "not" &&
      expression.args.size() == 1)
    return VisitGuardCandidates(expression.args[0], !truth, prefix, resolve,
                                consume);
  if (expression.kind == Expr::Kind::Binary && expression.args.size() == 2) {
    const bool and_operator = expression.text == "and";
    const bool or_operator = expression.text == "or";
    if (and_operator || or_operator) {
      if ((and_operator && truth) || (or_operator && !truth))
        return VisitGuardCandidates(expression.args[0], truth, prefix, resolve,
                                    [&](const Assignment &left) {
                                      return VisitGuardCandidates(
                                          expression.args[1], truth, left,
                                          resolve, consume);
                                    });
      return VisitGuardCandidates(expression.args[0], truth, prefix, resolve,
                                  consume) ||
             VisitGuardCandidates(expression.args[1], truth, prefix, resolve,
                                  consume);
    }
    if (expression.text == "==" || expression.text == "!=") {
      std::vector<std::string> path;
      std::string value;
      bool known = Path(expression.args[0], &path) &&
                   Literal(expression.args[1], &value);
      if (!known) {
        path.clear();
        known = Path(expression.args[1], &path) &&
                Literal(expression.args[0], &value);
      }
      if (known) {
        if (truth == (expression.text == "!="))
          value = value == "false" ? "true" : "false";
        return Suggest(std::move(path), std::move(value), prefix, resolve,
                       consume);
      }
    }
  }
  if (expression.kind == Expr::Kind::Test && !expression.args.empty()) {
    std::vector<std::string> path;
    if (Path(expression.args.front(), &path)) {
      std::string value;
      if (expression.text == "defined")
        value = truth ? "true" : kMissing;
      else if (expression.text == "undefined")
        value = truth ? kMissing : "true";
      else if (expression.text == "none")
        value = truth ? "null" : "true";
      else if (expression.text == "string")
        value = truth ? Quote("PROBE_BINDING") : "{}";
      else if (expression.text == "mapping")
        value = truth ? "{}" : Quote("PROBE_BINDING");
      if (!value.empty())
        return Suggest(std::move(path), std::move(value), prefix, resolve,
                       consume);
    }
  }
  if (expression.kind == Expr::Kind::Filter &&
      (expression.text == "default" || expression.text == "d") &&
      !expression.args.empty())
    return VisitGuardCandidates(expression.args.front(), truth, prefix, resolve,
                                consume);
  std::vector<std::string> path;
  if (Path(expression, &path))
    return Suggest(std::move(path), truth ? "true" : "false", prefix, resolve,
                   consume);
  return consume(prefix);
}

void CollectCandidates(const std::vector<std::vector<Expr>> &routes,
                       const ResolveProbePath &resolve,
                       std::vector<Assignment> *out) {
  for (const auto &guards : routes) {
    // Walk compatible suggestions lazily. A later conflict backtracks to the
    // earlier disjunction; no Cartesian product of paths is materialized.
    std::function<bool(size_t, const Assignment &)> route;
    route = [&](size_t index, const Assignment &prefix) {
      if (index == guards.size()) {
        out->push_back(prefix);
        return true;
      }
      return VisitGuardCandidates(
          guards[index], true, prefix, resolve,
          [&](const Assignment &next) { return route(index + 1, next); });
    };
    route(0, {});
    for (const auto &guard : guards) {
      for (bool truth : {false, true})
        VisitGuardCandidates(guard, truth, {}, resolve,
                             [&](const Assignment &candidate) {
                               out->push_back(candidate);
                               return true;
                             });
    }
  }
}

struct SourceControls {
  std::set<std::string> locals;
  std::set<std::string> externals;
  std::set<std::string> compiler_externals;
  std::set<std::string> forbidden;
  std::set<std::string> external_defaults;
  std::map<std::string, std::vector<std::string>> aliases;
  std::vector<Expr> expressions;

  void Reads(const Expr &expression, const std::set<std::string> &defined) {
    if (expression.kind == Expr::Kind::Name && !defined.count(expression.text))
      externals.insert(expression.text);
    for (const auto &child : expression.args)
      Reads(child, defined);
  }
  std::set<std::string> Discover(const std::vector<Node> &nodes,
                                 std::set<std::string> defined = {}) {
    for (const auto &node : nodes) {
      Reads(node.expression, defined);
      for (const auto &value : node.defaults)
        Reads(value, defined);
      if (node.kind == Node::Kind::Set) {
        defined.insert(node.text);
      } else if (node.kind == Node::Kind::Capture) {
        Discover(node.body, defined);
        defined.insert(node.text);
      } else if (node.kind == Node::Kind::If) {
        const auto yes = Discover(node.body, defined);
        const auto no = Discover(node.otherwise, defined);
        std::set<std::string> joined;
        std::set_intersection(yes.begin(), yes.end(), no.begin(), no.end(),
                              std::inserter(joined, joined.end()));
        defined = std::move(joined);
      } else if (node.kind == Node::Kind::For) {
        auto inner = defined;
        const auto comma = node.text.find(',');
        inner.insert(node.text.substr(0, comma));
        if (comma != std::string::npos)
          inner.insert(node.text.substr(comma + 1));
        if (node.predicate)
          Reads(*node.predicate, inner);
        inner.insert("loop");
        Discover(node.body, std::move(inner));
        Discover(node.otherwise, defined);
      } else if (node.kind == Node::Kind::Macro) {
        auto inner = defined;
        inner.insert(node.parameters.begin(), node.parameters.end());
        Discover(node.body, std::move(inner));
        defined.insert(node.text);
      } else {
        Discover(node.body, defined);
        Discover(node.otherwise, defined);
      }
    }
    return defined;
  }

  void Expression(const Expr &expression) {
    if (expression.kind == Expr::Kind::Conditional && !expression.args.empty())
      expressions.push_back(expression.args.front());
    if (expression.kind == Expr::Kind::Binary ||
        expression.kind == Expr::Kind::Test ||
        (expression.kind == Expr::Kind::Filter &&
         (expression.text == "default" || expression.text == "d")))
      expressions.push_back(expression);
    for (const auto &child : expression.args)
      Expression(child);
  }
  void Nodes(const std::vector<Node> &nodes) {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::If)
        expressions.push_back(node.expression);
      if (node.kind == Node::Kind::Capture) {
        locals.insert(node.text);
      } else if (node.kind == Node::Kind::Set) {
        locals.insert(node.text);
        const Expr *value = &node.expression;
        while (value->kind == Expr::Kind::Filter && !value->args.empty() &&
               (value->text == "default" || value->text == "d" ||
                value->text == "string"))
          value = &value->args.front();
        std::vector<std::string> path;
        if (Path(*value, &path)) {
          if (path.front() == node.text)
            external_defaults.insert(node.text);
          else
            aliases[node.text] = std::move(path);
        }
      } else if (node.kind == Node::Kind::For) {
        const auto comma = node.text.find(',');
        locals.insert(node.text.substr(0, comma));
        if (comma != std::string::npos)
          locals.insert(node.text.substr(comma + 1));
      } else if (node.kind == Node::Kind::Macro) {
        locals.insert(node.text);
        locals.insert(node.parameters.begin(), node.parameters.end());
      }
      Expression(node.expression);
      if (node.predicate)
        Expression(*node.predicate);
      for (const auto &value : node.defaults)
        Expression(value);
      Nodes(node.body);
      Nodes(node.otherwise);
    }
  }
  bool Resolve(std::vector<std::string> *path) const {
    std::set<std::string> seen;
    while (!path->empty()) {
      const auto found = aliases.find(path->front());
      if (found == aliases.end())
        break;
      if (!seen.insert(path->front()).second)
        return false;
      auto next = found->second;
      next.insert(next.end(), path->begin() + 1, path->end());
      *path = std::move(next);
    }
    return !path->empty() && compiler_externals.count(path->front()) &&
           (!locals.count(path->front()) || externals.count(path->front()) ||
            external_defaults.count(path->front())) &&
           path->front() != "loop" && !forbidden.count(path->front());
  }
  void Add(std::vector<Assignment> *candidates) const {
    const ResolveProbePath resolve = [&](std::vector<std::string> *path) {
      return Resolve(path);
    };
    for (const auto &expression : expressions)
      for (bool truth : {false, true})
        VisitGuardCandidates(expression, truth, {}, resolve,
                             [&](const Assignment &candidate) {
                               candidates->push_back(candidate);
                               return true;
                             });
    // Independent single-control variations remain useful even when the
    // analyzer mistakenly loses an entire condition or reaching definition.
    for (const auto &name : externals)
      for (bool value : {false, true})
        Suggest({name}, value ? "true" : "false", {}, resolve,
                [&](const Assignment &candidate) {
                  candidates->push_back(candidate);
                  return true;
                });
    // A truthiness witness is not necessarily in an external control's finite
    // source domain (for example a string compared with enum literals). Try
    // the actual comparison/default literals alongside existing route seeds.
    // Native execution still decides which combinations are valid.
    std::map<std::vector<std::string>, std::set<std::string>> domains;
    const auto add_values = [&](const Expr &target, const Expr &values) {
      std::vector<std::string> path;
      if (!Path(target, &path) || !Resolve(&path))
        return;
      std::string literal;
      if (Literal(values, &literal)) {
        domains[path].insert(literal);
      } else if (values.kind == Expr::Kind::List ||
                 values.kind == Expr::Kind::Tuple) {
        for (const auto &value : values.args)
          if (Literal(value, &literal))
            domains[path].insert(literal);
      }
    };
    for (const auto &expression : expressions) {
      if (expression.kind == Expr::Kind::Binary &&
          expression.args.size() == 2) {
        add_values(expression.args[0], expression.args[1]);
        add_values(expression.args[1], expression.args[0]);
      } else if (expression.kind == Expr::Kind::Filter &&
                 (expression.text == "default" || expression.text == "d") &&
                 expression.args.size() >= 2) {
        add_values(expression.args[0], expression.args[1]);
      }
    }
    // Deduplicate before finite-domain refinement: repeated AST tests often
    // produce the same empty/boolean suggestion, which is not a new input.
    Deduplicate(candidates);
    const size_t seed_count = candidates->size();
    for (size_t i = 0; i < seed_count; ++i)
      for (const auto &[path, values] : domains)
        for (const auto &value : values) {
          auto candidate = (*candidates)[i];
          candidate[path] = value;
          candidates->push_back(std::move(candidate));
        }
    Deduplicate(candidates);
  }
};

std::string ProbeCall::Serialize(bool wire) const {
  const auto arguments_json = Json(arguments);
  return Json(
      {{"id", Quote(id)},
       {"type", Quote("function")},
       {"function", Json({{"name", Quote(name)},
                          {"arguments", wire || !mapped ? Quote(arguments_json)
                                                        : arguments_json}})}});
}

std::string ProbeMessage::Serialize(bool wire) const {
  Object fields{{"role", Quote(role)}, {"content", content}};
  if (!reasoning.empty())
    fields["reasoning_content"] = reasoning;
  if (!call_id.empty())
    fields["tool_call_id"] = Quote(call_id);
  if (!calls.empty()) {
    std::vector<std::string> values;
    for (const auto &call : calls)
      values.push_back(call.Serialize(wire));
    fields["tool_calls"] = Array(values);
  }
  return Json(fields);
}

std::string Messages(const std::vector<ProbeMessage> &messages, bool wire) {
  std::vector<std::string> values;
  for (const auto &message : messages)
    values.push_back(message.Serialize(wire));
  return Array(values);
}

ProbeMessage Message(std::string role, std::string_view content) {
  return {std::move(role), Quote(content), {}, {}, {}};
}

std::vector<ProbeMessage> History(int kind) {
  if (kind == 0)
    return {Message("user", "PROBE_USER")};
  if (kind == 1)
    return {Message("user", "PROBE_USER_OLD"),
            Message("assistant", "PROBE_ASSISTANT_OLD"),
            Message("user", "PROBE_USER")};
  if (kind == 3)
    return {Message("system", "PROBE_SYSTEM"), Message("user", "PROBE_USER")};
  if (kind == 4)
    return {Message("system", "PROBE_SYSTEM")};
  if (kind == 5)
    return {Message("assistant", "PROBE_ASSISTANT_OLD")};
  ProbeMessage previous = Message("assistant", "PROBE_HISTORY_CONTENT");
  previous.calls.push_back(
      {"probe_history_function",
       "probe_history_call",
       {{"probe_history_key", Quote("PROBE_HISTORY_VALUE")}},
       false});
  auto tool = Message("tool", "PROBE_TOOL_RESULT");
  tool.call_id = "probe_history_call";
  return {Message("user", "PROBE_USER_OLD"), std::move(previous),
          std::move(tool), Message("user", "PROBE_USER")};
}

std::string Tools(int count) {
  std::vector<std::string> tools;
  for (int version = 0; version < 2; ++version)
    for (int i = 0; i < count; ++i) {
      const char suffix = version ? 'B' : 'A';
      tools.push_back(
          Json({{"type", Quote("function")},
                {"function",
                 Json({{"name", Quote(std::string("probe_function_") + suffix +
                                      std::to_string(i))},
                       {"description", Quote("Deterministic analysis probe")},
                       {"parameters",
                        "{\"type\":\"object\",\"properties\":{}}"}})}}));
    }
  return Array(tools);
}

bool Fail(std::string_view source, std::string_view source_name, size_t offset,
          const std::string &message, Diagnostic *diagnostic) {
  if (diagnostic) {
    diagnostic->message = message;
    diagnostic->offset = offset;
    LocateDiagnostic(source, source_name, diagnostic);
  }
  return false;
}

size_t Offset(const TextFormatAlternative &format) {
  return format.evidence.empty() ? 0 : format.evidence.front().begin;
}

Expr Name(std::string value) {
  Expr result;
  result.kind = Expr::Kind::Name;
  result.text = std::move(value);
  return result;
}

Expr Member(Expr subject, const char *key) {
  Expr result;
  result.kind = Expr::Kind::GetAttr;
  result.text = key;
  result.args.push_back(std::move(subject));
  return result;
}

RecipeBuilder::RecipeBuilder(const TextFormatAnalysis &analysis,
                             std::string_view source)
    : analysis_(analysis), source_(source) {
  assistant_ = Fresh("assistant");
  formatted_tools_ = Fresh("formatted_tools");
  ordinal_ = Fresh("ordinal");
}
bool RecipeBuilder::BuildContinuation(const TextUserContinuation &continuation,
                                      Template *out, std::string *error) {
  continuation_ = &continuation;
  out->body = continuation.prelude.body;
  Bind({}, &out->body);
  return Parts(continuation.pattern, {}, &out->body, error);
}
bool RecipeBuilder::BuildTurn(
    const std::vector<TextFormatPart> &parts,
    const std::vector<TextFormatPart> &first_call_prefix, Template *out,
    std::string *error) {
  turn_ = true;
  first_call_prefix_ = &first_call_prefix;
  Bind({}, &out->body);
  return Parts(parts, {}, &out->body, error);
}
bool RecipeBuilder::BuildCall(const std::vector<TextFormatPart> &parts,
                              Template *out, std::string *error) {
  native_call_ = true;
  Context context{assistant_, {}, {}, true};
  Bind(context, &out->body);
  return Parts(parts, context, &out->body, error);
}
std::string RecipeBuilder::Fresh(const char *suffix) {
  std::string result = "__loomify_recipe_" + std::string(suffix);
  while (source_.find(result) != std::string_view::npos ||
         Bindings().count(result) || names_.count(result))
    result += '_';
  names_.insert(result);
  return result;
}
bool RecipeBuilder::Semantic(TextCapture capture, const Context &context,
                             Expr *out) {
  if (continuation_) {
    if (capture != TextCapture::Content)
      return false;
    *out = Name(continuation_->text_binding);
    return true;
  }
  switch (capture) {
  case TextCapture::Content:
    *out = Member(Name(assistant_), "content");
    return true;
  case TextCapture::Reasoning:
    *out = Member(Name(assistant_), "reasoning_content");
    return true;
  case TextCapture::CallsValue:
    *out = Member(Name(assistant_), "tool_calls");
    return true;
  case TextCapture::CallRecord:
    if (context.call.empty())
      return false;
    *out = Name(context.call);
    return true;
  case TextCapture::FunctionRecord:
    if (context.call.empty())
      return false;
    *out = Member(Name(context.call), "function");
    return true;
  case TextCapture::FunctionName:
  case TextCapture::Arguments:
    if (context.call.empty())
      return false;
    *out = Member(Member(Name(context.call), "function"),
                  capture == TextCapture::FunctionName ? "name" : "arguments");
    return true;
  case TextCapture::ArgumentKey:
  case TextCapture::ArgumentValue:
    if (context.key.empty())
      return false;
    *out =
        Name(capture == TextCapture::ArgumentKey ? context.key : context.value);
    return true;
  case TextCapture::RepeatFirst:
  case TextCapture::RepeatIndex:
  case TextCapture::RepeatLast:
    if (!context.loop)
      return false;
    if (native_call_ && context.key.empty()) {
      if (capture == TextCapture::RepeatLast)
        return false;
      *out = Name(ordinal_);
      if (capture == TextCapture::RepeatFirst) {
        Expr comparison;
        comparison.kind = Expr::Kind::Binary;
        comparison.text = "==";
        Expr zero;
        zero.kind = Expr::Kind::Integer;
        zero.text = "0";
        comparison.args = {*out, zero};
        *out = std::move(comparison);
      }
      return true;
    }
    *out =
        Member(Name("loop"), capture == TextCapture::RepeatFirst  ? "first"
                             : capture == TextCapture::RepeatLast ? "last"
                                                                  : "index0");
    return true;
  default:
    return false;
  }
}
void RecipeBuilder::Bind(const Context &context, std::vector<Node> *out) {
  for (const auto &[name, capture] : Bindings()) {
    Expr value;
    if (!Semantic(capture, context, &value))
      continue;
    Node set;
    set.kind = Node::Kind::Set;
    set.text = name;
    set.expression = std::move(value);
    out->push_back(std::move(set));
  }
}
bool RecipeBuilder::Parts(const std::vector<TextFormatPart> &parts,
                          const Context &context, std::vector<Node> *out,
                          std::string *error) {
  for (const auto &part : parts) {
    Node node;
    node.offset = part.source.begin;
    node.end = part.source.end;
    switch (part.capture) {
    case TextCapture::Literal:
      node.kind = Node::Kind::Text;
      node.text = part.literal;
      break;
    case TextCapture::Choice:
      node.kind = Node::Kind::If;
      node.expression = part.predicate;
      if (part.body_error.kind != Expr::Kind::None) {
        Node failure;
        failure.kind = Node::Kind::Output;
        failure.expression = part.body_error;
        node.body.push_back(std::move(failure));
      } else if (!Parts(part.body, context, &node.body, error)) {
        return false;
      }
      if (part.otherwise_error.kind != Expr::Kind::None) {
        Node failure;
        failure.kind = Node::Kind::Output;
        failure.expression = part.otherwise_error;
        node.otherwise.push_back(std::move(failure));
      } else if (!Parts(part.otherwise, context, &node.otherwise, error)) {
        return false;
      }
      break;
    case TextCapture::RepeatCalls: {
      if (turn_) {
        node.kind = Node::Kind::If;
        node.expression = Name(formatted_tools_);
        if (!Parts(*first_call_prefix_, {}, &node.body, error))
          return false;
        Node fragment;
        fragment.kind = Node::Kind::Output;
        fragment.expression = Name(formatted_tools_);
        node.body.push_back(std::move(fragment));
        break;
      }
      if (continuation_) {
        *error = "continued USER recipe needs an unavailable call record";
        return false;
      }
      node.kind = Node::Kind::For;
      node.text = Fresh("call");
      node.expression = Member(Name(assistant_), "tool_calls");
      Context inner{node.text, {}, {}, true};
      Bind(inner, &node.body);
      if (!Parts(part.body, inner, &node.body, error))
        return false;
      break;
    }
    case TextCapture::RepeatArguments: {
      if (context.call.empty()) {
        *error = "argument repetition outside a call recipe";
        return false;
      }
      node.kind = Node::Kind::For;
      Context inner{context.call, Fresh("key"), Fresh("value"), true};
      node.text = inner.key + ',' + inner.value;
      node.expression.kind = Expr::Kind::Filter;
      node.expression.text = "items";
      node.expression.args = {
          Member(Member(Name(context.call), "function"), "arguments")};
      Bind(inner, &node.body);
      if (!Parts(part.body, inner, &node.body, error))
        return false;
      break;
    }
    default:
      if (part.value_expression.kind == Expr::Kind::None) {
        *error = "capture recipe has no resolved source transform";
        return false;
      }
      node.kind = Node::Kind::Output;
      node.expression = part.value_expression;
      break;
    }
    out->push_back(std::move(node));
  }
  return true;
}
const std::map<std::string, TextCapture> &RecipeBuilder::Bindings() const {
  return continuation_ ? continuation_->predicate_bindings
                       : analysis_.predicate_bindings;
}

bool Normalize(const Object &base, const std::vector<ProbeMessage> &original,
               const TextFormatAnalysis &analysis,
               PreparedTemplate &adaptation_selector,
               std::vector<ProbeMessage> *normalized, std::string *error) {
  *normalized = original;
  for (size_t i = 0; i < original.size(); ++i) {
    Object worker = base;
    worker[analysis.adaptation_message_binding] = original[i].Serialize(true);
    worker[analysis.adaptation_message_index_binding] = std::to_string(i);
    const size_t count = std::max<size_t>(1, original[i].calls.size());
    bool content_observed = false;
    for (size_t j = 0; j < count; ++j) {
      if (!original[i].calls.empty()) {
        worker[analysis.adaptation_call_binding] =
            original[i].calls[j].Serialize(true);
        worker[analysis.adaptation_call_index_binding] = std::to_string(j);
      } else {
        worker.erase(analysis.adaptation_call_binding);
        worker.erase(analysis.adaptation_call_index_binding);
      }
      int64_t mask = -1, zero = -1;
      if (!adaptation_selector.SelectFormat(Json(worker), &mask, &zero, error))
        return false;
      TextAdaptation projection;
      if (zero != 0 || !DecodeTextAdaptation(mask, &projection)) {
        *error = "adaptation selector returned an invalid feature mask";
        return false;
      }
      content_observed |= projection.content_observes_representation;
      if (!original[i].calls.empty())
        (*normalized)[i].calls[j].mapped = projection.arguments_need_mapping &&
                                           !projection.arguments_observe_string;
    }
    // Every worker observes the original message/environment. One call route
    // preserving representation keeps the message's original content for all
    // calls; an earlier nonobserving route must never erase that information.
    if (!content_observed && original[i].content == "null")
      (*normalized)[i].content = Quote("");
  }
  return true;
}

bool VisitTemplateInputs(
    const Template &original, const GeneratedInputRenderer &input,
    const TextFormatAnalysis &analysis, const TextFormatBindings &bindings,
    const std::function<InputVisit(const TemplateInput &)> &visit) {
  SourceControls source_controls;
  source_controls.Discover(original.body);
  source_controls.Nodes(original.body);
  source_controls.compiler_externals.insert(input.external_bindings.begin(),
                                            input.external_bindings.end());
  source_controls.forbidden.insert(input.protected_bindings.begin(),
                                   input.protected_bindings.end());
  for (const auto &[name, unused] : bindings.strings)
    source_controls.forbidden.insert(name);
  // These original-source inputs have explicit probe domains below. They are
  // not free kwargs, and Apply() cannot replace their owned histories/tools.
  source_controls.forbidden.insert("messages");
  source_controls.forbidden.insert("tools");
  for (const auto &[name, unused] : analysis.predicate_bindings)
    source_controls.forbidden.insert(name);
  for (const auto &[name, unused] : analysis.selector_binding_expressions)
    source_controls.forbidden.insert(name);
  source_controls.forbidden.insert(analysis.adaptation_message_binding);
  source_controls.forbidden.insert(analysis.adaptation_message_index_binding);
  source_controls.forbidden.insert(analysis.adaptation_call_binding);
  source_controls.forbidden.insert(analysis.adaptation_call_index_binding);
  const ResolveProbePath resolve = [&](std::vector<std::string> *path) {
    return source_controls.Resolve(path);
  };
  // Original-source controls must run even when an incomplete analysis happens
  // to cover every format it declared on the default input. Route suggestions
  // are only search work after those independent counterexamples are checked.
  std::vector<Assignment> candidates{Assignment{}};
  source_controls.Add(&candidates);
  const size_t mandatory_candidates = candidates.size();
  std::vector<Assignment> route_candidates;
  CollectCandidates(analysis.format_route_guards, resolve, &route_candidates);
  source_controls.Add(&route_candidates);
  candidates.insert(candidates.end(), route_candidates.begin(),
                    route_candidates.end());
  std::set<std::string> visited;
  for (size_t candidate_index = 0; candidate_index < candidates.size();
       ++candidate_index) {
    const auto &candidate = candidates[candidate_index];
    for (int history_kind = 0; history_kind < 6; ++history_kind) {
      const auto history = History(history_kind);
      for (int tools_kind = 0; tools_kind < 4; ++tools_kind) {
        Object environment;
        for (const auto &[key, value] : bindings.strings)
          environment[key] = Quote(value);
        environment["add_generation_prompt"] = "true";
        Apply(candidate, &environment);
        // The shared environment always supplies this binding. Missing raw
        // request control is represented by the projector's default true,
        // rather than an impossible undefined internal builtin.
        if (!environment.count("add_generation_prompt"))
          environment["add_generation_prompt"] = "true";
        // Native-owned bindings cannot be replaced by probe suggestions for a
        // local source alias. They always describe actual generated probe data.
        environment["messages"] = Messages(history, true);
        if (tools_kind == 0)
          environment.erase("tools");
        else if (tools_kind == 1)
          environment["tools"] = "null";
        else if (tools_kind == 2)
          environment["tools"] = "[]";
        else
          environment["tools"] = Tools(2);
        for (const auto &[key, value] : bindings.strings)
          environment[key] = Quote(value);
        const auto raw_live_json = Json(environment);
        if (!visited.insert(raw_live_json).second)
          continue;
        const auto action = visit({environment, history, candidate_index,
                                   candidate_index >= mandatory_candidates,
                                   history_kind, tools_kind});
        if (action == InputVisit::Fail)
          return false;
        if (action == InputVisit::Stop)
          return true;
      }
    }
  }
  return true;
}

Node EffortSelection(int value) {
  Node result;
  result.kind = Node::Kind::Output;
  result.expression.kind = Expr::Kind::List;
  Expr number;
  number.kind = Expr::Kind::Integer;
  number.text = std::to_string(value);
  result.expression.args.push_back(number);
  number.text = "0";
  result.expression.args.push_back(std::move(number));
  return result;
}

// Retain actual native setup semantics, not an interpreter of their values.
// Ordinary prompt output is irrelevant to the domain selector. Macro output
// remains intact because a setup expression may call the macro.
std::vector<Node> EffortSetup(const std::vector<Node> &nodes) {
  std::vector<Node> result;
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Text || node.kind == Node::Kind::Output)
      continue;
    auto copy = node;
    if (node.kind != Node::Kind::Macro) {
      copy.body = EffortSetup(node.body);
      copy.otherwise = EffortSetup(node.otherwise);
    }
    result.push_back(std::move(copy));
  }
  return result;
}

bool EffortAlias(const Expr &expression, const std::set<std::string> &aliases) {
  if (expression.kind == Expr::Kind::Name)
    return aliases.count(expression.text);
  return expression.kind == Expr::Kind::Filter &&
         (expression.text == "default" || expression.text == "d") &&
         !expression.args.empty() && EffortAlias(expression.args[0], aliases);
}

bool Raises(const std::vector<Node> &body) {
  bool raises = false;
  for (const auto &node : body) {
    if (node.kind == Node::Kind::Text &&
        node.text.find_first_not_of(" \t\r\n") == std::string::npos)
      continue;
    if (node.kind != Node::Kind::Output ||
        node.expression.kind != Expr::Kind::Call ||
        node.expression.args.empty() ||
        node.expression.args[0].kind != Expr::Kind::Name ||
        node.expression.args[0].text != "raise_exception")
      return false;
    raises = true;
  }
  return raises;
}

struct EffortGuard {
  std::set<std::string> domain;
  std::vector<Node> selector;
  size_t offset = 0;
};

void ForgetAssigned(const std::vector<Node> &nodes,
                    std::set<std::string> *aliases) {
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Set)
      aliases->erase(node.text);
    if (node.kind != Node::Kind::Macro) {
      ForgetAssigned(node.body, aliases);
      ForgetAssigned(node.otherwise, aliases);
    }
  }
}

// Recognize only the finite literal error-guard shape. Actual source
// conditions, defaults, scopes and membership execute through the existing
// compiler below.
std::vector<EffortGuard> FindEffortGuards(const std::vector<Node> &nodes,
                                          std::set<std::string> aliases = {
                                              "reasoning_effort"}) {
  std::vector<EffortGuard> result;
  std::vector<Node> setup;
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Set) {
      const bool alias = EffortAlias(node.expression, aliases);
      aliases.erase(node.text);
      if (alias)
        aliases.insert(node.text);
    } else if (node.kind == Node::Kind::If) {
      const auto &condition = node.expression;
      if (condition.kind == Expr::Kind::Binary && condition.text == "not in" &&
          condition.args.size() == 2 &&
          EffortAlias(condition.args[0], aliases) &&
          (condition.args[1].kind == Expr::Kind::Tuple ||
           condition.args[1].kind == Expr::Kind::List) &&
          Raises(node.body)) {
        EffortGuard guard;
        guard.offset = node.offset;
        for (const auto &literal : condition.args[1].args) {
          if (literal.kind != Expr::Kind::String) {
            guard.domain.clear();
            break;
          }
          guard.domain.insert(literal.text);
        }
        if (!guard.domain.empty()) {
          guard.selector = setup;
          Node branch;
          branch.kind = Node::Kind::If;
          branch.expression = condition;
          branch.body.push_back(EffortSelection(1));      // rejected
          branch.otherwise.push_back(EffortSelection(0)); // admitted
          guard.selector.push_back(std::move(branch));
          result.push_back(std::move(guard));
        }
      } else {
        for (bool yes : {true, false}) {
          auto children =
              FindEffortGuards(yes ? node.body : node.otherwise, aliases);
          for (auto &guard : children) {
            Node branch;
            branch.kind = Node::Kind::If;
            branch.expression = condition;
            branch.body = yes ? std::move(guard.selector)
                              : std::vector<Node>{EffortSelection(2)};
            branch.otherwise = yes ? std::vector<Node>{EffortSelection(2)}
                                   : std::move(guard.selector);
            guard.selector = setup;
            guard.selector.push_back(std::move(branch));
            result.push_back(std::move(guard));
          }
        }
      }
      ForgetAssigned(node.body, &aliases);
      ForgetAssigned(node.otherwise, &aliases);
    } else if (node.kind == Node::Kind::For) {
      // A finite discovery guard inside a history loop needs a scope-specific
      // proof; do not turn one iteration's domain into a global capability.
      ForgetAssigned(node.body, &aliases);
      ForgetAssigned(node.otherwise, &aliases);
    }
    auto retained = EffortSetup({node});
    setup.insert(setup.end(), retained.begin(), retained.end());
  }
  return result;
}

bool QualifyColdReasoning(const SourceTemplate &unit,
                          const GeneratedInputRenderer &input,
                          const TextFormatAnalysis &format,
                          const GeneratedInputRenderer &format_selector,
                          const GeneratedInputRenderer &adaptation,
                          const TextFormatBindings &bindings,
                          const TemplateAnalysisOptions &options,
                          TextReasoningFacts *out, Diagnostic *diagnostic) {
  const auto source = unit.source(), source_name = unit.source_name();
  TextReasoningFacts result;
  for (const auto &alternative : format.alternatives)
    result.enabled |= !alternative.reasoning_open.empty();
  if (!result.enabled ||
      std::find(input.external_bindings.begin(), input.external_bindings.end(),
                "reasoning_effort") == input.external_bindings.end()) {
    *out = result;
    return true;
  }
  const auto &original = unit.ast();
  const auto guards = FindEffortGuards(original.body);
  if (guards.size() != 1)
    return Fail(source, source_name, 0,
                "cold reasoning requires one source-qualified finite literal "
                "effort error guard before history iteration",
                diagnostic);
  const auto &guard = guards.front();
  std::string fallback;
  if (!ReadTemplateStringDefault(unit, "reasoning_effort", &fallback,
                                 diagnostic))
    return false;
  const std::vector<std::string> vocabulary{"none", "minimal", "low", "medium",
                                            "high", "xhigh",   "max"};
  for (const auto &value : guard.domain) {
    const auto found = std::find(vocabulary.begin(), vocabulary.end(), value);
    if (found == vocabulary.end())
      return Fail(source, source_name, guard.offset,
                  "source reasoning effort is outside native discovery "
                  "vocabulary: " +
                      value,
                  diagnostic);
    result.effort_mask |= int64_t{1} << (found - vocabulary.begin());
  }
  const auto default_value =
      std::find(vocabulary.begin(), vocabulary.end(), fallback);
  if (!guard.domain.count(fallback) || default_value == vocabulary.end() ||
      (result.effort_mask & ~int64_t{1}) == 0)
    return Fail(source, source_name, guard.offset,
                "source reasoning effort default/domain cannot be advertised "
                "by native discovery",
                diagnostic);
  result.default_effort = default_value - vocabulary.begin();
  Template selector_ast;
  selector_ast.body = guard.selector;
  GeneratedInputRenderer generated;
  if (!CompileTemplateSelector(source, source_name, selector_ast, &generated,
                               diagnostic))
    return false;
  std::unique_ptr<PreparedTemplate> renderer, selector, adaptation_selector;
  std::string error;
  for (const auto &[generated, prepared] :
       std::vector<std::pair<const GeneratedInputRenderer *,
                             std::unique_ptr<PreparedTemplate> *>>{
           {&input, &renderer},
           {&format_selector, &selector},
           {&adaptation, &adaptation_selector}})
    if (!PreparedTemplate::Prepare(*generated, options.helper_root,
                                   options.scratch_parent, options.bytecode,
                                   prepared, &error))
      return Fail(source, source_name, 0,
                  "cannot prepare compiled template analysis: " + error,
                  diagnostic);

  std::unique_ptr<PreparedTemplate> effort_selector;
  if (!PreparedTemplate::Prepare(generated, options.helper_root,
                                 options.scratch_parent, options.bytecode,
                                 &effort_selector, &error))
    return Fail(source, source_name, guard.offset,
                "cannot prepare source effort guard: " + error, diagnostic);
  bool found = false;
  const auto ColdFailure = [&](std::string_view text, std::string_view name,
                               size_t offset, const std::string &message,
                               Diagnostic *out_diagnostic) {
    Fail(text, name, offset, message, out_diagnostic);
    return InputVisit::Fail;
  };
  if (!VisitTemplateInputs(
          original, input, format, bindings,
          [&](const TemplateInput &candidate) {
            std::vector<ProbeMessage> normalized;
            if (!Normalize(candidate.environment, candidate.history, format,
                           *adaptation_selector, &normalized, &error))
              return InputVisit::Continue;
            Object context = candidate.environment;
            context["messages"] = Messages(normalized);
            std::string prompt;
            if (!renderer->Render(Json(context), &prompt, &error))
              return InputVisit::Continue;
            int64_t format_id = -1, startup = -1;
            if (!selector->SelectFormat(Json(context), &format_id, &startup,
                                        &error))
              return InputVisit::Continue;
            if (format_id < 0 ||
                static_cast<size_t>(format_id) >= format.alternatives.size() ||
                startup < 0 || startup > 2)
              return ColdFailure(
                  source, source_name, 0,
                  "compiled selector returned an invalid format/startup",
                  diagnostic);
            if (format.alternatives[format_id].reasoning_open.empty())
              return InputVisit::Continue;
            context.erase("reasoning_effort");
            int64_t admission = -1, zero = -1;
            if (!effort_selector->SelectFormat(Json(context), &admission, &zero,
                                               &error))
              return ColdFailure(source, source_name, guard.offset,
                                 "cannot execute source effort guard: " + error,
                                 diagnostic);
            if (admission == 2 && zero == 0)
              return InputVisit::Continue; // Inactive guards establish no
                                           // domain.
            if (admission != 0 || zero != 0)
              return ColdFailure(
                  source, source_name, guard.offset,
                  "source effort default is rejected in active context",
                  diagnostic);
            std::vector<std::pair<std::string, bool>> values{{kMissing, true}};
            for (const auto &value : vocabulary)
              values.emplace_back(Quote(value), guard.domain.count(value));
            values.emplace_back("null", false);
            values.emplace_back("false", false);
            values.emplace_back(Quote("PROBE_UNKNOWN_EFFORT"), false);
            for (const auto &[value, expected] : values) {
              context["reasoning_effort"] = value;
              const auto json = Json(context);
              std::string rendered;
              const bool accepted = renderer->Render(json, &rendered, &error);
              if (!effort_selector->SelectFormat(json, &admission, &zero,
                                                 &error) ||
                  zero != 0 || admission != (expected ? 0 : 1) ||
                  accepted != expected)
                return ColdFailure(
                    source, source_name, guard.offset,
                    "compiled source reasoning effort admission disagrees "
                    "with its finite discovery domain for " +
                        value + ": " + error,
                    diagnostic);
            }
            found = true;
            return InputVisit::Stop;
          }))
    return false;
  if (found) {
    *out = result;
    return true;
  }
  return Fail(source, source_name, guard.offset,
              "no source-valid reasoning context activates the effort guard",
              diagnostic);
}

} // namespace hrx::loomify_gguf::jinja::internal

namespace hrx::loomify_gguf::jinja {
using namespace internal;
bool internal::AnalyzeTemplate(const SourceTemplate &unit,
                               const GeneratedInputRenderer &input,
                               const TextFormatBindings &bindings,
                               const TemplateAnalysisOptions &options,
                               AnalyzedTemplate *out, Diagnostic *diagnostic) {
  const auto source = unit.source(), source_name = unit.source_name();
  if (!out)
    return Fail(source, source_name, 0, "analysis output must not be null",
                diagnostic);
  AnalyzedTemplate result;
  result.renderer = input;
  if (!AnalyzeTextFormat(unit, bindings, &result.format, diagnostic) ||
      !CompileTemplateSelector(source, source_name,
                               result.format.selector_template,
                               &result.selector, diagnostic) ||
      !CompileTemplateSelector(source, source_name,
                               result.format.adaptation_selector_template,
                               &result.adaptation_selector, diagnostic,
                               TemplateSelectorKind::Adaptation))
    return false;
  if (result.format.alternatives.empty())
    return Fail(source, source_name, 0,
                "format analysis returned no executable alternatives",
                diagnostic);
  if (!QualifyColdReasoning(unit, input, result.format, result.selector,
                            result.adaptation_selector, bindings, options,
                            &result.reasoning, diagnostic))
    return false;
  TextUserContinuation continuation;
  if (AnalyzeUserContinuation(unit, bindings, result.format, &continuation,
                              &result.continuation_diagnostic)) {
    RecipeBuilder builder(result.format, source);
    Template recipe;
    std::string continuation_error;
    UserContinuationProduct product;
    if (!builder.BuildContinuation(continuation, &recipe,
                                   &continuation_error)) {
      Fail(source, source_name, 0,
           "cannot materialize continued USER recipe: " + continuation_error,
           &result.continuation_diagnostic);
    } else if (CompileTemplateRenderer(source, source_name, recipe,
                                       &product.renderer,
                                       &result.continuation_diagnostic)) {
      bool available = true;
      for (const auto &name : product.renderer.external_bindings)
        if (name != continuation.text_binding &&
            !bindings.strings.count(name)) {
          Fail(source, source_name, 0,
               "continued USER recipe retains unavailable binding: " + name,
               &result.continuation_diagnostic);
          available = false;
          break;
        }
      if (available) {
        product.text_binding = continuation.text_binding;
        product.assistant_closure = continuation.assistant_closure;
        result.continuation = std::move(product);
      }
    }
  }
  *out = std::move(result);
  if (diagnostic)
    *diagnostic = {};
  return true;
}

bool AnalyzeTemplate(std::string_view source, std::string_view source_name,
                     const GeneratedInputRenderer &input,
                     const TextFormatBindings &bindings,
                     const TemplateAnalysisOptions &options,
                     AnalyzedTemplate *out, Diagnostic *diagnostic) {
  if (!out)
    return Fail(source, source_name, 0, "analysis output must not be null",
                diagnostic);
  std::unique_ptr<SourceTemplate> unit;
  return SourceTemplate::Parse(source, source_name, &unit, diagnostic) &&
         internal::AnalyzeTemplate(*unit, input, bindings, options, out,
                                   diagnostic);
}

} // namespace hrx::loomify_gguf::jinja

namespace hrx::loomify_gguf::jinja {
namespace {
using namespace internal;
using TurnParts = std::vector<TextFormatPart>;
Expr TurnBool(bool value) {
  Expr e;
  e.kind = Expr::Kind::Boolean;
  e.text = value ? "true" : "false";
  return e;
}
Expr TurnInt(int value) {
  Expr e;
  e.kind = Expr::Kind::Integer;
  e.text = std::to_string(value);
  return e;
}
Expr TurnBinary(std::string op, Expr a, Expr b) {
  Expr e;
  e.kind = Expr::Kind::Binary;
  e.text = std::move(op);
  e.args = {std::move(a), std::move(b)};
  return e;
}
bool TurnSame(const Expr &a, const Expr &b) {
  if (a.kind != b.kind || a.text != b.text || a.keywords != b.keywords ||
      a.args.size() != b.args.size())
    return false;
  for (size_t i = 0; i < a.args.size(); ++i)
    if (!TurnSame(a.args[i], b.args[i]))
      return false;
  return true;
}
std::optional<bool> TurnTruth(const Expr &e) {
  if (e.kind == Expr::Kind::Boolean)
    return e.text == "true";
  return std::nullopt;
}
Expr TurnSimplify(Expr e, const TextFormatAnalysis &analysis,
                  const std::vector<Expr> &guards, std::optional<bool> first,
                  std::optional<int> calls = {}) {
  const auto is_calls = [&](const Expr &value) {
    const auto it = analysis.predicate_bindings.find(value.text);
    return value.kind == Expr::Kind::Name &&
           it != analysis.predicate_bindings.end() &&
           it->second == TextCapture::CallsValue;
  };
  // Native completion introduces either no call member or an actual list.
  // These tests are invariant over that semantic domain; they do not observe
  // the temporary fragment STRING used solely for presence below.
  if (e.kind == Expr::Kind::Test && e.args.size() == 1 && is_calls(e.args[0])) {
    if (e.text == "iterable" || e.text == "sequence")
      return TurnBool(true);
    if (e.text == "mapping" || e.text == "none" || e.text == "string" ||
        e.text == "number" || e.text == "boolean" || e.text == "true" ||
        e.text == "false")
      return TurnBool(false);
  }
  if (calls) {
    if (e.kind == Expr::Kind::Filter && e.text == "length" &&
        e.args.size() == 1 && is_calls(e.args[0]))
      return TurnInt(*calls);
    if (e.kind == Expr::Kind::Test && !e.args.empty() && is_calls(e.args[0])) {
      if (e.text == "defined")
        return TurnBool(*calls != 0);
      if (e.text == "undefined")
        return TurnBool(*calls == 0);
      if (e.text == "iterable" || e.text == "sequence")
        return TurnBool(true);
      if (e.text == "mapping" || e.text == "none" || e.text == "string" ||
          e.text == "number" || e.text == "boolean" || e.text == "true" ||
          e.text == "false")
        return TurnBool(false);
    }
    if (is_calls(e))
      return TurnBool(*calls != 0);
  }
  for (const auto &guard : guards) {
    if (TurnSame(e, guard))
      return TurnBool(true);
    if (guard.kind == Expr::Kind::Unary && guard.text == "not" &&
        guard.args.size() == 1 && TurnSame(e, guard.args[0]))
      return TurnBool(false);
  }
  if (first && e.kind == Expr::Kind::Name) {
    const auto it = analysis.predicate_bindings.find(e.text);
    if (it != analysis.predicate_bindings.end()) {
      if (it->second == TextCapture::RepeatFirst)
        return TurnBool(*first);
      if (*first && it->second == TextCapture::RepeatIndex)
        return TurnInt(0);
    }
  }
  for (auto &arg : e.args)
    arg = TurnSimplify(std::move(arg), analysis, guards, first, calls);
  if (e.kind == Expr::Kind::Unary && e.text == "not" && e.args.size() == 1)
    if (auto value = TurnTruth(e.args[0]))
      return TurnBool(!*value);
  if (e.kind == Expr::Kind::Binary && e.args.size() == 2) {
    if (e.args[0].kind == Expr::Kind::Integer &&
        e.args[1].kind == Expr::Kind::Integer) {
      if (e.text == "==")
        return TurnBool(e.args[0].text == e.args[1].text);
      if (e.text == "!=")
        return TurnBool(e.args[0].text != e.args[1].text);
    }
    const auto a = TurnTruth(e.args[0]), b = TurnTruth(e.args[1]);
    if (e.text == "and") {
      if (a)
        return *a ? e.args[1] : TurnBool(false);
      if (b && *b)
        return e.args[0];
    }
    if (e.text == "or") {
      if (a)
        return *a ? TurnBool(true) : e.args[1];
      if (b && !*b)
        return e.args[0];
    }
  }
  if (e.kind == Expr::Kind::Conditional && e.args.size() == 3)
    if (auto condition = TurnTruth(e.args[0]))
      return e.args[*condition ? 1 : 2];
  return e;
}
TurnParts TurnSimplifyParts(TurnParts parts, const TextFormatAnalysis &analysis,
                            const std::vector<Expr> &guards,
                            std::optional<bool> first = {},
                            std::optional<int> calls = {}) {
  TurnParts result;
  for (auto &part : parts) {
    part.predicate =
        TurnSimplify(std::move(part.predicate), analysis, guards, first, calls);
    part.value_expression = TurnSimplify(std::move(part.value_expression),
                                         analysis, guards, first, calls);
    part.body_error = TurnSimplify(std::move(part.body_error), analysis, guards,
                                   first, calls);
    part.otherwise_error = TurnSimplify(std::move(part.otherwise_error),
                                        analysis, guards, first, calls);
    part.body =
        TurnSimplifyParts(std::move(part.body), analysis, guards, first, calls);
    part.otherwise = TurnSimplifyParts(std::move(part.otherwise), analysis,
                                       guards, first, calls);
    if (part.capture == TextCapture::Choice) {
      if (auto selected = TurnTruth(part.predicate)) {
        const auto &failure =
            *selected ? part.body_error : part.otherwise_error;
        if (failure.kind == Expr::Kind::None) {
          auto &body = *selected ? part.body : part.otherwise;
          result.insert(result.end(), body.begin(), body.end());
          continue;
        }
      }
    }
    result.push_back(std::move(part));
  }
  return result;
}
bool TurnPartsSame(const TurnParts &a, const TurnParts &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].capture != b[i].capture || a[i].literal != b[i].literal ||
        !TurnSame(a[i].predicate, b[i].predicate) ||
        !TurnSame(a[i].value_expression, b[i].value_expression) ||
        !TurnSame(a[i].body_error, b[i].body_error) ||
        !TurnSame(a[i].otherwise_error, b[i].otherwise_error) ||
        !TurnPartsSame(a[i].body, b[i].body) ||
        !TurnPartsSame(a[i].otherwise, b[i].otherwise))
      return false;
  }
  return true;
}
bool TurnCallCapture(TextCapture c) {
  return c == TextCapture::FunctionName || c == TextCapture::Arguments ||
         c == TextCapture::CallRecord || c == TextCapture::FunctionRecord ||
         c == TextCapture::ArgumentKey || c == TextCapture::ArgumentValue;
}
bool TurnExprCapture(const Expr &e, const TextFormatAnalysis &analysis,
                     bool call) {
  if (e.kind == Expr::Kind::Name) {
    auto it = analysis.predicate_bindings.find(e.text);
    if (it != analysis.predicate_bindings.end()) {
      if (call && TurnCallCapture(it->second))
        return true;
      if (!call && (it->second == TextCapture::Content ||
                    it->second == TextCapture::Reasoning ||
                    it->second == TextCapture::CallsValue ||
                    it->second == TextCapture::RepeatLast))
        return true;
    }
  }
  for (const auto &arg : e.args)
    if (TurnExprCapture(arg, analysis, call))
      return true;
  return false;
}
bool TurnCallsName(const Expr &e, const TextFormatAnalysis &analysis) {
  const auto it = analysis.predicate_bindings.find(e.text);
  return e.kind == Expr::Kind::Name &&
         it != analysis.predicate_bindings.end() &&
         it->second == TextCapture::CallsValue;
}
bool TurnCallsPresence(const Expr &e, const TextFormatAnalysis &analysis,
                       bool truth_context) {
  if (TurnCallsName(e, analysis))
    return truth_context;
  if (e.kind == Expr::Kind::Unary && e.text == "not" && e.args.size() == 1)
    return TurnCallsPresence(e.args[0], analysis, true);
  if (e.kind == Expr::Kind::Binary && (e.text == "and" || e.text == "or") &&
      e.args.size() == 2)
    return TurnCallsPresence(e.args[0], analysis, truth_context) &&
           TurnCallsPresence(e.args[1], analysis, truth_context);
  if (e.kind == Expr::Kind::Conditional && e.args.size() == 3)
    return TurnCallsPresence(e.args[0], analysis, true) &&
           TurnCallsPresence(e.args[1], analysis, truth_context) &&
           TurnCallsPresence(e.args[2], analysis, truth_context);
  if (e.kind == Expr::Kind::Test && !e.args.empty() &&
      TurnCallsName(e.args[0], analysis))
    return e.args.size() == 1 &&
           (e.text == "defined" || e.text == "undefined" || e.text == "none" ||
            e.text == "iterable" || e.text == "sequence" ||
            e.text == "mapping");
  // A fallback changes only the absent value. Present call collections and
  // nonempty native fragments have identical truth in this context.
  if (truth_context && e.kind == Expr::Kind::Filter &&
      (e.text == "default" || e.text == "d") && !e.args.empty()) {
    for (const auto &arg : e.args)
      if (!TurnCallsPresence(arg, analysis, true))
        return false;
    return true;
  }
  for (const auto &arg : e.args)
    if (!TurnCallsPresence(arg, analysis, false))
      return false;
  return true;
}
bool TurnCallsRepresentable(const TurnParts &parts,
                            const TextFormatAnalysis &analysis,
                            size_t *offset) {
  for (const auto &part : parts) {
    if (!TurnCallsPresence(part.predicate, analysis, true) ||
        !TurnCallsPresence(part.value_expression, analysis, false) ||
        !TurnCallsPresence(part.body_error, analysis, false) ||
        !TurnCallsPresence(part.otherwise_error, analysis, false)) {
      *offset = part.source.begin;
      return false;
    }
    // The source loop itself is replaced by actual native formatted records.
    // Its per-call body is checked separately against render_tool's inputs.
    if (part.capture != TextCapture::RepeatCalls &&
        (!TurnCallsRepresentable(part.body, analysis, offset) ||
         !TurnCallsRepresentable(part.otherwise, analysis, offset)))
      return false;
  }
  return true;
}
const Expr *TurnValidationError(const TurnParts &parts, const Expr &success) {
  for (const auto &part : parts) {
    if (part.capture == TextCapture::Choice) {
      if (TurnSame(success, part.predicate) &&
          part.otherwise_error.kind != Expr::Kind::None)
        return &part.otherwise_error;
      if (success.kind == Expr::Kind::Unary && success.text == "not" &&
          success.args.size() == 1 &&
          TurnSame(success.args[0], part.predicate) &&
          part.body_error.kind != Expr::Kind::None)
        return &part.body_error;
    }
    if (const auto *error = TurnValidationError(part.body, success))
      return error;
    if (const auto *error = TurnValidationError(part.otherwise, success))
      return error;
  }
  return nullptr;
}
bool TurnCallDependent(const TextFormatPart &part,
                       const TextFormatAnalysis &analysis) {
  if (TurnCallCapture(part.capture) ||
      part.capture == TextCapture::RepeatArguments ||
      TurnExprCapture(part.predicate, analysis, true) ||
      TurnExprCapture(part.value_expression, analysis, true) ||
      TurnExprCapture(part.body_error, analysis, true) ||
      TurnExprCapture(part.otherwise_error, analysis, true))
    return true;
  for (const auto &p : part.body)
    if (TurnCallDependent(p, analysis))
      return true;
  for (const auto &p : part.otherwise)
    if (TurnCallDependent(p, analysis))
      return true;
  return false;
}
// Factor only a source prefix independent of the current call record. Choice
// arms can contribute different prefixes when their remaining call is
// identical.
bool TurnSplitCall(const TurnParts &parts, const TextFormatAnalysis &analysis,
                   TurnParts *prefix, TurnParts *body) {
  for (size_t i = 0; i < parts.size(); ++i) {
    const auto &part = parts[i];
    if (!TurnCallDependent(part, analysis)) {
      prefix->push_back(part);
      continue;
    }
    if (part.capture == TextCapture::Choice &&
        !TurnExprCapture(part.predicate, analysis, true) &&
        part.body_error.kind == Expr::Kind::None &&
        part.otherwise_error.kind == Expr::Kind::None) {
      TurnParts yes_prefix, no_prefix, yes_body, no_body;
      if (!TurnSplitCall(part.body, analysis, &yes_prefix, &yes_body) ||
          !TurnSplitCall(part.otherwise, analysis, &no_prefix, &no_body) ||
          !TurnPartsSame(yes_body, no_body))
        return false;
      auto choice = part;
      choice.body = std::move(yes_prefix);
      choice.otherwise = std::move(no_prefix);
      prefix->push_back(std::move(choice));
      *body = std::move(yes_body);
    } else {
      body->push_back(part);
    }
    body->insert(body->end(), parts.begin() + i + 1, parts.end());
    return true;
  }
  return true;
}
std::string TurnCommonSuffix(std::string a, std::string_view b) {
  size_t count = 0;
  while (count < a.size() && count < b.size() &&
         a[a.size() - count - 1] == b[b.size() - count - 1])
    ++count;
  return a.substr(a.size() - count);
}
std::string TurnLiteralSuffix(const TurnParts &parts) {
  std::string result;
  for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
    if (it->capture == TextCapture::Literal) {
      result = it->literal + result;
    } else if (it->capture == TextCapture::Choice) {
      return TurnCommonSuffix(TurnLiteralSuffix(it->body),
                              TurnLiteralSuffix(it->otherwise)) +
             result;
    } else {
      break;
    }
  }
  return result;
}
bool TurnRemoveSuffix(TurnParts *parts, std::string_view suffix) {
  size_t remaining = suffix.size();
  for (size_t i = parts->size(); i-- > 0 && remaining;) {
    auto &part = (*parts)[i];
    if (part.capture == TextCapture::Literal) {
      const auto count = std::min(remaining, part.literal.size());
      if (std::string_view(part.literal).substr(part.literal.size() - count) !=
          suffix.substr(remaining - count, count))
        return false;
      part.literal.resize(part.literal.size() - count);
      remaining -= count;
    } else if (part.capture == TextCapture::Choice) {
      if (!TurnRemoveSuffix(&part.body, suffix.substr(0, remaining)) ||
          !TurnRemoveSuffix(&part.otherwise, suffix.substr(0, remaining)))
        return false;
      remaining = 0;
    } else {
      return false;
    }
  }
  return remaining == 0;
}
bool TurnForbiddenFormatter(const TurnParts &parts,
                            const TextFormatAnalysis &analysis) {
  for (const auto &part : parts) {
    if (TurnExprCapture(part.predicate, analysis, false) ||
        TurnExprCapture(part.value_expression, analysis, false) ||
        TurnExprCapture(part.body_error, analysis, false) ||
        TurnExprCapture(part.otherwise_error, analysis, false) ||
        TurnForbiddenFormatter(part.body, analysis) ||
        TurnForbiddenFormatter(part.otherwise, analysis))
      return true;
  }
  return false;
}
bool TurnNeedsMapping(const TurnParts &parts) {
  for (const auto &part : parts)
    if (part.capture == TextCapture::RepeatArguments ||
        TurnNeedsMapping(part.body) || TurnNeedsMapping(part.otherwise))
      return true;
  return false;
}
bool TurnHasArgumentOutput(const TurnParts &parts) {
  for (const auto &part : parts)
    if (part.capture == TextCapture::Arguments ||
        TurnHasArgumentOutput(part.body) ||
        TurnHasArgumentOutput(part.otherwise))
      return true;
  return false;
}
bool TurnObservesArgumentString(const TurnParts &parts,
                                const TextFormatAnalysis &analysis) {
  for (const auto &part : parts) {
    const Expr *test = &part.predicate;
    bool negated = false;
    if (test->kind == Expr::Kind::Unary && test->text == "not" &&
        test->args.size() == 1) {
      negated = true;
      test = &test->args[0];
    }
    if (test->kind == Expr::Kind::Test && test->text == "string" &&
        test->args.size() == 1) {
      const auto &subject = test->args[0];
      auto it = analysis.predicate_bindings.find(subject.text);
      if (subject.kind == Expr::Kind::Name &&
          it != analysis.predicate_bindings.end() &&
          it->second == TextCapture::Arguments &&
          (negated ? part.otherwise_error : part.body_error).kind ==
              Expr::Kind::None &&
          TurnHasArgumentOutput(negated ? part.otherwise : part.body))
        return true;
    }
    if (TurnObservesArgumentString(part.body, analysis) ||
        TurnObservesArgumentString(part.otherwise, analysis))
      return true;
  }
  return false;
}
// Compute only source control facts. Value-producing macros retain their full
// bodies; removing their output would change assignments and branch truth.
bool TurnNodesSame(const std::vector<Node> &a, const std::vector<Node> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].kind != b[i].kind || a[i].text != b[i].text ||
        a[i].parameters != b[i].parameters ||
        a[i].defaults.size() != b[i].defaults.size() ||
        !TurnSame(a[i].expression, b[i].expression) ||
        !TurnNodesSame(a[i].body, b[i].body) ||
        !TurnNodesSame(a[i].otherwise, b[i].otherwise))
      return false;
  return true;
}
bool TurnLiveWrite(const std::vector<Node> &nodes,
                   const TextFormatAnalysis &analysis) {
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Set)
      for (const auto &name : analysis.completion_output_dependencies)
        if (name == node.text || name.rfind(node.text + ".", 0) == 0 ||
            node.text.rfind(name + ".", 0) == 0)
          return true;
    if (TurnLiveWrite(node.body, analysis) ||
        TurnLiveWrite(node.otherwise, analysis))
      return true;
  }
  return false;
}
bool TurnHasOutput(const std::vector<Node> &nodes) {
  for (const auto &node : nodes)
    if ((node.kind == Node::Kind::Text && !node.text.empty()) ||
        node.kind == Node::Kind::Output || TurnHasOutput(node.body) ||
        TurnHasOutput(node.otherwise))
      return true;
  return false;
}
bool TurnHasReasoningOutput(const std::vector<Node> &nodes,
                            const TextFormatAnalysis &analysis) {
  for (const auto &node : nodes)
    if (node.kind == Node::Kind::Output &&
        analysis.completion_reasoning_outputs.count(node.offset))
      return true;
  return false;
}
bool TurnUnconditionalLiteral(const Expr &e, std::string_view text) {
  if (e.kind == Expr::Kind::String)
    return e.text.find(text) != std::string::npos;
  // Concatenation always emits each operand. A marker merely present inside a
  // conditional/filter/call does not prove this arm emits that framing.
  if (e.kind == Expr::Kind::Binary && (e.text == "+" || e.text == "~"))
    for (const auto &arg : e.args)
      if (TurnUnconditionalLiteral(arg, text))
        return true;
  return false;
}
bool TurnUnconditionalLiteral(const std::vector<Node> &nodes,
                              std::string_view text) {
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Text &&
        node.text.find(text) != std::string::npos)
      return true;
    if (node.kind == Node::Kind::Output &&
        TurnUnconditionalLiteral(node.expression, text))
      return true;
  }
  return false;
}
bool TurnLiteralContains(const Expr &e, std::string_view text) {
  if (e.kind == Expr::Kind::String && e.text.find(text) != std::string::npos)
    return true;
  for (const auto &arg : e.args)
    if (TurnLiteralContains(arg, text))
      return true;
  return false;
}
bool TurnLiteralContains(const std::vector<Node> &nodes,
                         std::string_view text) {
  for (const auto &node : nodes)
    if ((node.kind == Node::Kind::Text &&
         node.text.find(text) != std::string::npos) ||
        TurnLiteralContains(node.expression, text) ||
        TurnLiteralContains(node.body, text) ||
        TurnLiteralContains(node.otherwise, text))
      return true;
  return false;
}
bool TurnReasoningRewrite(const Node &node,
                          const TextFormatAnalysis &analysis) {
  if (TurnLiveWrite(node.body, analysis) ||
      TurnLiveWrite(node.otherwise, analysis))
    return false;
  for (const auto &format : analysis.alternatives) {
    if (format.reasoning_open.empty() || format.reasoning_close.empty())
      continue;
    for (bool yes : {true, false}) {
      const auto &framed = yes ? node.body : node.otherwise;
      const auto &plain = yes ? node.otherwise : node.body;
      if (TurnHasReasoningOutput(framed, analysis) &&
          !TurnHasReasoningOutput(plain, analysis) &&
          TurnUnconditionalLiteral(framed, format.reasoning_open) &&
          TurnUnconditionalLiteral(framed, format.reasoning_close) &&
          !TurnLiteralContains(plain, format.reasoning_open) &&
          !TurnLiteralContains(plain, format.reasoning_close))
        return true;
    }
  }
  return false;
}
Expr TurnReplaceLast(Expr e, bool last) {
  if (e.kind == Expr::Kind::GetAttr && e.text == "last" && e.args.size() == 1 &&
      e.args[0].kind == Expr::Kind::Name && e.args[0].text == "loop")
    return TurnBool(last);
  for (auto &a : e.args)
    a = TurnReplaceLast(std::move(a), last);
  return e;
}
Expr TurnAsBoolean(Expr value) {
  Expr not_value;
  not_value.kind = Expr::Kind::Unary;
  not_value.text = "not";
  not_value.args.push_back(std::move(value));
  Expr result;
  result.kind = Expr::Kind::Unary;
  result.text = "not";
  result.args.push_back(std::move(not_value));
  return result;
}
std::vector<Node> TurnReflowSlice(const std::vector<Node> &nodes,
                                  const std::string &flag,
                                  const std::string &discard,
                                  const TextFormatAnalysis &analysis,
                                  std::set<size_t> *handled,
                                  std::map<size_t, std::string> *errors,
                                  const std::string &message = {}) {
  std::vector<Node> result;
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Macro) {
      result.push_back(node);
      continue;
    }
    if (node.kind == Node::Kind::Text)
      continue;
    if (node.kind == Node::Kind::Output) {
      // Output expressions can invoke namespace-writing macros. Evaluate each
      // once in its original scope, discarding bytes without discarding
      // effects.
      auto evaluate = node;
      evaluate.kind = Node::Kind::Set;
      evaluate.text = discard;
      result.push_back(std::move(evaluate));
      continue;
    }
    const auto condition =
        analysis.completion_last_conditions.find(node.offset);
    if (node.kind == Node::Kind::If &&
        condition != analysis.completion_last_conditions.end()) {
      handled->insert(node.offset);
      const bool dead = !TurnHasOutput(node.body) &&
                        !TurnHasOutput(node.otherwise) &&
                        !TurnLiveWrite(node.body, analysis) &&
                        !TurnLiveWrite(node.otherwise, analysis);
      if (!dead && !TurnNodesSame(node.body, node.otherwise) &&
          (message.empty() || !TurnReasoningRewrite(node, analysis)))
        (*errors)[node.offset] =
            "a401 completion cannot qualify this source history rewrite";
      if (!dead && !TurnNodesSame(node.body, node.otherwise) &&
          !errors->count(node.offset)) {
        Node changed;
        changed.kind = Node::Kind::If;
        changed.offset = node.offset;
        Expr role;
        role.kind = Expr::Kind::String;
        role.text = "assistant";
        auto assistant = TurnBinary("==", Member(Name(message), "role"), role);
        auto last = Member(Name("loop"), "last");
        auto differs = TurnBinary(
            "!=", TurnAsBoolean(TurnReplaceLast(condition->second, true)),
            TurnAsBoolean(TurnReplaceLast(condition->second, false)));
        changed.expression = TurnBinary(
            "and", std::move(last),
            TurnBinary("and", std::move(assistant), std::move(differs)));
        Node remember;
        remember.kind = Node::Kind::Set;
        remember.text = flag + ".changed";
        remember.expression = TurnBool(true);
        changed.body.push_back(std::move(remember));
        result.push_back(std::move(changed));
      }
    }
    auto copy = node;
    std::string nested_message = message;
    if (node.kind == Node::Kind::For) {
      nested_message.clear();
      if (analysis.completion_message_loops.count(node.offset))
        nested_message = node.text;
    }
    copy.body = TurnReflowSlice(node.body, flag, discard, analysis, handled,
                                errors, nested_message);
    copy.otherwise = TurnReflowSlice(node.otherwise, flag, discard, analysis,
                                     handled, errors, nested_message);
    result.push_back(std::move(copy));
  }
  return result;
}
bool TurnReflow(const SourceTemplate &source,
                const TextFormatAnalysis &analysis, Template *out,
                Diagnostic *diagnostic) {
  std::string flag = "__loomify_prior_reflow";
  while (source.source_names().count(flag))
    flag += '_';
  std::string discard = "__loomify_discarded_render";
  while (source.source_names().count(discard) || discard == flag)
    discard += '_';
  Template result;
  Node start;
  start.kind = Node::Kind::Set;
  start.text = flag;
  start.expression.kind = Expr::Kind::Call;
  start.expression.args = {Name("namespace"), TurnBool(false)};
  start.expression.keywords = {"", "changed"};
  result.body.push_back(std::move(start));
  std::set<size_t> handled;
  auto errors = analysis.completion_rewrite_diagnostics;
  auto slice = TurnReflowSlice(source.ast().body, flag, discard, analysis,
                               &handled, &errors);
  for (const auto &[offset, condition] : analysis.completion_last_conditions)
    if (!handled.count(offset))
      errors[offset] =
          "a401 completion cannot snapshot this source macro history rewrite";
  if (!errors.empty())
    return Fail(source.source(), source.source_name(), errors.begin()->first,
                errors.begin()->second, diagnostic);
  result.body.insert(result.body.end(), slice.begin(), slice.end());
  Node test;
  test.kind = Node::Kind::If;
  test.expression = Member(Name(flag), "changed");
  Node yes, no;
  yes.kind = no.kind = Node::Kind::Text;
  yes.text = "1";
  no.text = "0";
  test.body.push_back(std::move(yes));
  test.otherwise.push_back(std::move(no));
  result.body.push_back(std::move(test));
  *out = std::move(result);
  return true;
}

bool TurnAvailable(const GeneratedInputRenderer &product,
                   const TextFormatBindings &bindings,
                   const std::set<std::string> &runtime,
                   std::string_view source, std::string_view name,
                   size_t offset, Diagnostic *diagnostic) {
  for (const auto &binding : product.external_bindings)
    if (!runtime.count(binding) && !bindings.strings.count(binding))
      return Fail(
          source, name, offset,
          "a401 turn hook requires unavailable original-request binding: " +
              binding,
          diagnostic);
  return true;
}
} // namespace

bool internal::EmitTextTurnControls(const SourceTemplate &unit,
                                    const TextFormatAnalysis &analysis,
                                    const TextFormatBindings &bindings,
                                    GeneratedInputRenderer *out,
                                    Diagnostic *diagnostic) {
  const auto source = unit.source(), name = unit.source_name();
  if (!out || analysis.alternatives.empty())
    return Fail(source, name, 0,
                "turn controls require analyzed source and output", diagnostic);
  if (analysis.selector_prelude_size > analysis.selector_template.body.size())
    return Fail(source, name, 0, "invalid append-validation dependency prefix",
                diagnostic);
  std::vector<GeneratedInputRenderer> products;
  std::vector<std::string> turns;
  std::vector<std::string> append_checks, append_errors;
  GeneratedInputRenderer formatter;
  bool have_formatter = false;
  std::string assistant_binding;
  std::string formatted_tools_binding, ordinal_binding;
  bool mapped_arguments = false;
  bool singleton_formatter = false;
  std::string formatter_header, formatter_suffix;
  GeneratedInputRenderer wrapper;
  wrapper.entrypoint = "complete_text";
  wrapper.helper_requirements = {"lib/text/common/strings.loom",
                                 "lib/text/common/json.loom",
                                 "lib/text/common/jinja.loom"};
  auto &code = wrapper.source;
  for (size_t id = 0; id < analysis.alternatives.size(); ++id) {
    const auto &format = analysis.alternatives[id];
    // Evaluate original-request-dependent append validation before its context
    // disappears, retaining only validity. Rejection remains a completion hook
    // effect, so a valid original prompt is never rejected by this preflight.
    Template append_check;
    if (!format.completion_guards.empty())
      append_check.body.assign(analysis.selector_template.body.begin(),
                               analysis.selector_template.body.begin() +
                                   analysis.selector_prelude_size);
    Node admitted;
    admitted.kind = Node::Kind::If;
    admitted.expression = TurnBool(true);
    const Expr *append_error = nullptr;
    for (const auto &guard : format.completion_guards) {
      const auto *error = TurnValidationError(format.assistant_pattern, guard);
      // proof_guards also includes facts inferred from successful validation
      // (for example nonfirst ordinal). Those are not separate runtime errors.
      // Only a retained source error arm discharged by this exact guard needs
      // the deferred validation fact; its implications simplify the turn below.
      if (!error)
        continue;
      if (append_error && !TurnSame(*append_error, *error))
        return Fail(
            source, name, Offset(format),
            "completion backend cannot yet retain distinct source append-validation errors",
            diagnostic);
      admitted.expression =
          TurnBinary("and", std::move(admitted.expression), guard);
      append_error = error;
    }
    Node accepted, rejected;
    accepted.kind = rejected.kind = Node::Kind::Text;
    accepted.text = "1";
    rejected.text = "0";
    admitted.body.push_back(std::move(accepted));
    admitted.otherwise.push_back(std::move(rejected));
    append_check.body.push_back(std::move(admitted));
    GeneratedInputRenderer check;
    if (append_error) {
      if (!CompileTemplateRenderer(source, name, append_check, &check,
                                   diagnostic,
                                   "_append_check_" + std::to_string(id)))
        return false;
      products.push_back(check);
    }
    append_checks.push_back(check.entrypoint);
    std::string error_entry;
    if (append_error) {
      RecipeBuilder error_builder(analysis, source);
      TextFormatPart error_part;
      error_part.capture = TextCapture::Expression;
      error_part.value_expression = *append_error;
      Template error_ast;
      std::string error;
      if (!error_builder.BuildTurn({error_part}, {}, &error_ast, &error))
        return Fail(source, name, Offset(format), error, diagnostic);
      GeneratedInputRenderer generated_error;
      if (!CompileTemplateRenderer(source, name, error_ast, &generated_error,
                                   diagnostic,
                                   "_append_error_" + std::to_string(id)) ||
          !TurnAvailable(generated_error, bindings,
                         {error_builder.assistant_binding(),
                          error_builder.formatted_tools_binding()},
                         source, name, Offset(format), diagnostic))
        return false;
      error_entry = generated_error.entrypoint;
      products.push_back(std::move(generated_error));
    }
    append_errors.push_back(std::move(error_entry));
    auto guards = id < analysis.format_route_guards.size()
                      ? analysis.format_route_guards[id]
                      : std::vector<Expr>{};
    guards.insert(guards.end(), format.completion_guards.begin(),
                  format.completion_guards.end());
    auto pattern = TurnSimplifyParts(
        format.assistant_pattern, analysis, guards, {},
        format.tools.kind == TextToolKind::None ? std::optional<int>(0)
                                                : std::nullopt);
    size_t unavailable_offset = 0;
    if (!TurnCallsRepresentable(pattern, analysis, &unavailable_offset))
      return Fail(
          source, name, unavailable_offset,
          "a401 complete_text receives formatted calls, not call-list count, records or representation",
          diagnostic);
    TurnParts prefix, body;
    if (!format.tools.call_body.empty() &&
        !TurnSplitCall(
            TurnSimplifyParts(format.tools.call_body, analysis, guards),
            analysis, &prefix, &body))
      return Fail(
          source, name, Offset(format),
          "a401 render_tool cannot factor source call arms with different records",
          diagnostic);
    // Keep the invariant opening bytes in each genuine call fragment. Only
    // content/ordinal-dependent spacing belongs to completion's first prefix.
    const auto opening = TurnLiteralSuffix(prefix);
    if (!opening.empty()) {
      if (!TurnRemoveSuffix(&prefix, opening))
        return Fail(source, name, Offset(format),
                    "cannot factor source call opening", diagnostic);
      TextFormatPart literal;
      literal.capture = TextCapture::Literal;
      literal.literal = opening;
      body.insert(body.begin(), std::move(literal));
    }
    auto first_prefix = TurnSimplifyParts(prefix, analysis, guards, true);
    RecipeBuilder turn_builder(analysis, source);
    Template turn_ast;
    std::string error;
    if (!turn_builder.BuildTurn(pattern, first_prefix, &turn_ast, &error))
      return Fail(source, name, Offset(format), error, diagnostic);
    GeneratedInputRenderer turn;
    if (!CompileTemplateRenderer(source, name, turn_ast, &turn, diagnostic,
                                 "_turn_" + std::to_string(id)))
      return false;
    assistant_binding = turn_builder.assistant_binding();
    formatted_tools_binding = turn_builder.formatted_tools_binding();
    if (!TurnAvailable(turn, bindings,
                       {assistant_binding, formatted_tools_binding}, source,
                       name, Offset(format), diagnostic))
      return false;
    turns.push_back(turn.entrypoint);
    products.push_back(std::move(turn));
    AppendBufferLiteral("turn.generation." + std::to_string(id),
                        format.generation_prefix, &code);
    if (format.tools.kind != TextToolKind::None ||
        format.historical_single_call || !format.tools.call_body.empty()) {
      if (format.historical_single_call) {
        body = TurnSimplifyParts(format.assistant_pattern, analysis, guards, {},
                                 1);
      }
      auto following = TurnSimplifyParts(prefix, analysis, guards, false);
      if (TurnForbiddenFormatter(following, analysis) ||
          TurnForbiddenFormatter(body, analysis))
        return Fail(
            source, name, Offset(format),
            "a401 render_tool requires unavailable content/reasoning/call-count context",
            diagnostic);
      RecipeBuilder call_builder(analysis, source);
      ordinal_binding = call_builder.ordinal_binding();
      TextFormatPart separator;
      separator.capture = TextCapture::Choice;
      separator.predicate = TurnBinary("!=", Name(ordinal_binding), TurnInt(0));
      separator.body = std::move(following);
      TurnParts call_parts{std::move(separator)};
      call_parts.insert(call_parts.end(), body.begin(), body.end());
      Template call_ast;
      GeneratedInputRenderer call;
      if (!call_builder.BuildCall(call_parts, &call_ast, &error))
        return Fail(source, name, Offset(format), error, diagnostic);
      if (!CompileTemplateRenderer(source, name, call_ast, &call, diagnostic,
                                   "_native_call"))
        return false;
      if (!TurnAvailable(call, bindings,
                         {call_builder.assistant_binding(), ordinal_binding},
                         source, name, Offset(format), diagnostic))
        return false;
      const bool mapping =
          TurnNeedsMapping(body) && !TurnObservesArgumentString(body, analysis);
      if (have_formatter &&
          (call.source != formatter.source || mapping != mapped_arguments))
        return Fail(
            source, name, Offset(format),
            "a401 render_tool has source routes requiring unavailable request context",
            diagnostic);
      formatter = std::move(call);
      mapped_arguments = mapping;
      if (have_formatter &&
          singleton_formatter != format.historical_single_call)
        return Fail(
            source, name, Offset(format),
            "a401 render_tool source routes disagree about singleton call framing",
            diagnostic);
      singleton_formatter = format.historical_single_call;
      if (singleton_formatter) {
        formatter_header = format.assistant_header;
        formatter_suffix = format.frame_suffix;
      }
      have_formatter = true;
    }
  }
  if (have_formatter)
    products.push_back(formatter);
  GeneratedInputRenderer reflow;
  Template reflow_ast;
  if (!TurnReflow(unit, analysis, &reflow_ast, diagnostic))
    return false;
  if (!CompileTemplateRenderer(source, name, reflow_ast, &reflow, diagnostic,
                               "_prior_reflow"))
    return false;
  const auto reflow_entry = reflow.entrypoint;
  products.push_back(std::move(reflow));
  AppendBufferLiteral("turn.assistant_key", assistant_binding, &code);
  AppendBufferLiteral("turn.formatted_tools_key", formatted_tools_binding,
                      &code);
  AppendBufferLiteral("turn.ordinal_key", ordinal_binding, &code);
  AppendBufferLiteral("turn.call_header", formatter_header, &code);
  AppendBufferLiteral("turn.call_suffix", formatter_suffix, &code);
  code += R"(
func.decl @text.json_quote(%value: buffer) -> (buffer)
func.decl @text.input_set_member(%object: buffer, %key: buffer, %value: buffer) -> (buffer)
func.decl @text.equal(%a: buffer, %b: buffer) -> (i1)
func.decl @text.join8(%a: buffer, %b: buffer, %c: buffer, %d: buffer, %e: buffer, %f: buffer, %g: buffer, %h: buffer) -> (buffer)
func.decl @text.slice(%text: buffer, %begin: offset, %end: offset) -> (buffer)
func.decl @jinja.integer(%a0: i64) -> (buffer)
func.decl import("input") @require(%condition: i32, %diagnostic: buffer)

func.def @text.turn_check(%valid: i1) {
  %flag = scalar.extui %valid : i1 to i32
  %error = buffer.constant utf8("invalid source-derived turn completion") : buffer
  func.call @require(%flag, %error) : (i32, buffer)
  func.return
}

func.def @text.turn_set(%object: buffer, %key: buffer, %value: buffer) -> (buffer) {
  %quoted = func.call @text.json_quote(%value) : (buffer) -> (buffer)
  %result = func.call @text.input_set_member(%object, %key, %quoted) : (buffer, buffer, buffer) -> (buffer)
  func.return %result : buffer
}
func.def @text.turn_environment(%content: buffer, %reasoning: buffer, %tools: buffer) -> (buffer) {
  %empty_object = buffer.constant utf8("{}") : buffer
  %content_key = buffer.constant utf8("content") : buffer
  %reason_key = buffer.constant utf8("reasoning_content") : buffer
  %calls_key = buffer.constant utf8("tool_calls") : buffer
  %content_object = func.call @text.turn_set(%empty_object, %content_key, %content) : (buffer, buffer, buffer) -> (buffer)
  %zero = index.constant 0 : offset
  %reason_bytes = buffer.length %reasoning
  %has_reason = index.cmp ne, %reason_bytes, %zero : offset
  %with_reason = scf.if %has_reason -> (buffer) {
    %updated = func.call @text.turn_set(%content_object, %reason_key, %reasoning) : (buffer, buffer, buffer) -> (buffer)
    scf.yield %updated : buffer
  } else { scf.yield %content_object : buffer }
  %tool_bytes = buffer.length %tools
  %has_tools = index.cmp ne, %tool_bytes, %zero : offset
  %assistant = scf.if %has_tools -> (buffer) {
    %updated = func.call @text.turn_set(%with_reason, %calls_key, %tools) : (buffer, buffer, buffer) -> (buffer)
    scf.yield %updated : buffer
  } else { scf.yield %with_reason : buffer }
  %globals = func.call @text.template_globals() : () -> (buffer)
  %assistant_key = global.load @turn.assistant_key : buffer
  %with_assistant = func.call @text.input_set_member(%globals, %assistant_key, %assistant) : (buffer, buffer, buffer) -> (buffer)
  %tools_key = global.load @turn.formatted_tools_key : buffer
  %environment = func.call @text.turn_set(%with_assistant, %tools_key, %tools) : (buffer, buffer, buffer) -> (buffer)
  func.return %environment : buffer
}
func.def @text.turn_prefix(%prompt: buffer, %tail: buffer) -> (buffer) {
  %length = buffer.length %prompt
  %tail_length = buffer.length %tail
  %fits = index.cmp ule, %tail_length, %length : offset
  func.call @text.turn_check(%fits) : (i1)
  %end = index.sub %length, %tail_length : offset
  %actual = func.call @text.slice(%prompt, %end, %length) : (buffer, offset, offset) -> (buffer)
  %exact = func.call @text.equal(%actual, %tail) : (buffer, buffer) -> (i1)
  func.call @text.turn_check(%exact) : (i1)
  %zero = index.constant 0 : offset
  %prefix = func.call @text.slice(%prompt, %zero, %end) : (buffer, offset, offset) -> (buffer)
  func.return %prefix : buffer
}
func.def @text.append_valid(%environment: buffer, %state: i64) -> (i1) {
  %shift = scalar.constant 32 : i64
  %format = scalar.shrui %state, %shift : i64
)";
  std::string valid_indent = "  ";
  for (size_t id = 0; id < append_checks.size(); ++id) {
    const auto n = std::to_string(id);
    code += valid_indent + "%id" + n + " = scalar.constant " + n + " : i64\n";
    code += valid_indent + "%selected" + n + " = scalar.cmpi eq, %format, %id" +
            n + " : i64\n";
    code += valid_indent + "%valid" + n + " = scf.if %selected" + n +
            " -> (i1) {\n";
    if (append_checks[id].empty()) {
      code += valid_indent + "  %ok = scalar.constant true : i1\n";
    } else {
      code += valid_indent + "  %checked = func.call @" + append_checks[id] +
              "(%environment) : (buffer) -> (buffer)\n";
      code += valid_indent + "  %one = buffer.constant utf8(\"1\") : buffer\n";
      code +=
          valid_indent +
          "  %ok = func.call @text.equal(%checked, %one) : (buffer, buffer) -> (i1)\n";
    }
    code +=
        valid_indent + "  scf.yield %ok : i1\n" + valid_indent + "} else {\n";
    valid_indent += "  ";
  }
  code += valid_indent + "%invalid = scalar.constant false : i1\n";
  code += valid_indent + "func.call @text.turn_check(%invalid) : (i1)\n";
  code += valid_indent + "scf.yield %invalid : i1\n";
  for (size_t id = append_checks.size(); id-- > 0;) {
    valid_indent.resize(valid_indent.size() - 2);
    code += valid_indent + "}\n";
    if (id)
      code +=
          valid_indent + "scf.yield %valid" + std::to_string(id) + " : i1\n";
  }
  code += R"(
  func.return %valid0 : i1
}
func.def public @text.completion_initial(%environment: buffer, %state: i64) -> (i64) {
)";
  code += "  %reflow = func.call @" + reflow_entry +
          "(%environment) : (buffer) -> (buffer)\n";
  code += R"(
  %one = buffer.constant utf8("1") : buffer
  %changed = func.call @text.equal(%reflow, %one) : (buffer, buffer) -> (i1)
  %flag = scalar.constant 8 : i64
  %zero = scalar.constant 0 : i64
  %bits = scf.select %changed, %flag, %zero : i64
  %append_valid = func.call @text.append_valid(%environment, %state) : (buffer, i64) -> (i1)
  %invalid_bit = scalar.constant 16 : i64
  %append_bits = scf.select %append_valid, %zero, %invalid_bit : i64
  %metadata = scalar.ori %bits, %append_bits : i64
  %result = scalar.ori %state, %metadata : i64
  func.return %result : i64
}
func.def public @complete_text(%prompt: buffer, %response: buffer, %content: buffer, %reasoning: buffer, %tools: buffer, %initial_state: i64, %final_state: i64) -> (buffer, i32) {
  %shift = scalar.constant 32 : i64
  %format = scalar.shrui %initial_state, %shift : i64
  %environment = func.call @text.turn_environment(%content, %reasoning, %tools) : (buffer, buffer, buffer) -> (buffer)
  %append_mask = scalar.constant 16 : i64
  %append_bits = scalar.andi %initial_state, %append_mask : i64
  %zero64 = scalar.constant 0 : i64
  %append_invalid = scalar.cmpi ne, %append_bits, %zero64 : i64
)";
  std::string indent = "  ";
  for (size_t id = 0; id < turns.size(); ++id) {
    const auto n = std::to_string(id);
    code += indent + "%id" + n + " = scalar.constant " + n + " : i64\n";
    code += indent + "%selected" + n + " = scalar.cmpi eq, %format, %id" + n +
            " : i64\n";
    code += indent + "%canonical" + n + " = scf.if %selected" + n +
            " -> (buffer) {\n";
    code +=
        indent + "  %tail = global.load @turn.generation." + n + " : buffer\n";
    code +=
        indent +
        "  %prefix = func.call @text.turn_prefix(%prompt, %tail) : (buffer, buffer) -> (buffer)\n";
    code += indent + "  %turn = scf.if %append_invalid -> (buffer) {\n";
    if (!append_errors[id].empty()) {
      code += indent + "    %error = func.call @" + append_errors[id] +
              "(%environment) : (buffer) -> (buffer)\n";
      code += indent + "    scf.yield %error : buffer\n";
    } else {
      code += indent + "    %valid = scalar.constant false : i1\n";
      code += indent + "    func.call @text.turn_check(%valid) : (i1)\n";
      code += indent + "    %empty = buffer.constant utf8(\"\") : buffer\n";
      code += indent + "    scf.yield %empty : buffer\n";
    }
    code += indent + "  } else {\n";
    code += indent + "    %rendered = func.call @" + turns[id] +
            "(%environment) : (buffer) -> (buffer)\n";
    code += indent + "    scf.yield %rendered : buffer\n";
    code += indent + "  }\n";
    code += indent + "  %empty = buffer.constant utf8(\"\") : buffer\n";
    code +=
        indent +
        "  %result = func.call @text.join8(%prefix, %turn, %empty, %empty, %empty, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)\n";
    code += indent + "  scf.yield %result : buffer\n" + indent + "} else {\n";
    indent += "  ";
  }
  code += indent + "%invalid = scalar.constant false : i1\n";
  code += indent + "func.call @text.turn_check(%invalid) : (i1)\n";
  code += indent + "%empty = buffer.constant utf8(\"\") : buffer\n";
  code += indent + "scf.yield %empty : buffer\n";
  for (size_t i = turns.size(); i-- > 0;) {
    indent.resize(indent.size() - 2);
    code += indent + "}\n";
    if (i)
      code +=
          indent + "scf.yield %canonical" + std::to_string(i) + " : buffer\n";
  }
  code += R"(
  %terminal = func.call @loomify.tokens.terminal_marker() : () -> (buffer)
  %empty = buffer.constant utf8("") : buffer
  %expected = func.call @text.join8(%prompt, %response, %terminal, %empty, %empty, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)
  %exact = func.call @text.equal(%canonical0, %expected) : (buffer, buffer) -> (i1)
  %visible = func.call @text.equal(%response, %content) : (buffer, buffer) -> (i1)
  %no_reason = func.call @text.equal(%reasoning, %empty) : (buffer, buffer) -> (i1)
  %no_tools = func.call @text.equal(%tools, %empty) : (buffer, buffer) -> (i1)
  %empty_terminal = func.call @text.equal(%terminal, %empty) : (buffer, buffer) -> (i1)
  %true = scalar.constant true : i1
  %has_terminal = scalar.xori %empty_terminal, %true : i1
  %mask = scalar.constant 8 : i64
  %reflow = scalar.andi %initial_state, %mask : i64
  %zero = scalar.constant 0 : i64
  %stable = scalar.cmpi eq, %reflow, %zero : i64
  %same = scalar.andi %exact, %visible : i1
  %plain = scalar.andi %no_reason, %no_tools : i1
  %terminal_ok = scalar.andi %has_terminal, %stable : i1
  %eligible = scalar.andi %same, %plain : i1
  %candidate = scalar.andi %eligible, %terminal_ok : i1
  %candidate32 = scalar.extui %candidate : i1 to i32
  func.return %canonical0, %candidate32 : buffer, i32
}
)";
  if (!have_formatter) {
    code += R"(
func.def public @render_tool(%name: buffer, %arguments: buffer, %ordinal: i64) -> (buffer, i64) {
  %empty = buffer.constant utf8("") : buffer
  %zero = scalar.constant 0 : i64
  func.return %empty, %zero : buffer, i64
}
)";
  } else {
    code += R"(
func.def public @render_tool(%name: buffer, %arguments: buffer, %ordinal: i64) -> (buffer, i64) {
  %object = buffer.constant utf8("{}") : buffer
  %name_key = buffer.constant utf8("name") : buffer
  %arguments_key = buffer.constant utf8("arguments") : buffer
  %function_key = buffer.constant utf8("function") : buffer
  %with_name = func.call @text.turn_set(%object, %name_key, %name) : (buffer, buffer, buffer) -> (buffer)
)";
    code +=
        "  %function = func.call @text." +
        std::string(mapped_arguments ? "input_set_member" : "turn_set") +
        "(%with_name, %arguments_key, %arguments) : (buffer, buffer, buffer) -> (buffer)\n";
    code += R"(
  %call = func.call @text.input_set_member(%object, %function_key, %function) : (buffer, buffer, buffer) -> (buffer)
  %globals = func.call @text.template_globals() : () -> (buffer)
  %assistant_key = global.load @turn.assistant_key : buffer
  %with_call = func.call @text.input_set_member(%globals, %assistant_key, %call) : (buffer, buffer, buffer) -> (buffer)
  %ordinal_key = global.load @turn.ordinal_key : buffer
  %ordinal_text = func.call @jinja.integer(%ordinal) : (i64) -> (buffer)
  %environment = func.call @text.input_set_member(%with_call, %ordinal_key, %ordinal_text) : (buffer, buffer, buffer) -> (buffer)
)";
    code += "  %rendered_call = func.call @" + formatter.entrypoint +
            "(%environment) : (buffer) -> (buffer)\n";
    if (singleton_formatter) {
      code += R"(
  %zero64 = scalar.constant 0 : i64
  %first = scalar.cmpi eq, %ordinal, %zero64 : i64
  func.call @text.turn_check(%first) : (i1)
  %header = global.load @turn.call_header : buffer
  %suffix = global.load @turn.call_suffix : buffer
  %without_suffix = func.call @text.turn_prefix(%rendered_call, %suffix) : (buffer, buffer) -> (buffer)
  %zero = index.constant 0 : offset
  %header_length = buffer.length %header
  %full_length = buffer.length %without_suffix
  %fits = index.cmp ule, %header_length, %full_length : offset
  func.call @text.turn_check(%fits) : (i1)
  %actual_header = func.call @text.slice(%without_suffix, %zero, %header_length) : (buffer, offset, offset) -> (buffer)
  %same_header = func.call @text.equal(%actual_header, %header) : (buffer, buffer) -> (i1)
  func.call @text.turn_check(%same_header) : (i1)
  %result = func.call @text.slice(%without_suffix, %header_length, %full_length) : (buffer, offset, offset) -> (buffer)
)";
    }
    const std::string result_name =
        singleton_formatter ? "%result" : "%rendered_call";
    code += "  %length = buffer.length " + result_name + "\n";
    code += "  %zero_length = index.constant 0 : offset\n";
    code += "  %nonempty = index.cmp ne, %length, %zero_length : offset\n";
    code += "  func.call @text.turn_check(%nonempty) : (i1)\n";
    code += "  %length64 = index.cast %length : offset to i64\n";
    code += "  func.return " + result_name + ", %length64 : buffer, i64\n}\n";
  }
  products.push_back(std::move(wrapper));
  GeneratedInputRenderer result;
  if (!CombineGeneratedTemplates(products, "complete_text", &result,
                                 diagnostic))
    return false;
  *out = std::move(result);
  if (diagnostic)
    *diagnostic = {};
  return true;
}

bool EmitTextTurnControls(std::string_view source, std::string_view source_name,
                          const TextFormatAnalysis &analysis,
                          const TextFormatBindings &bindings,
                          GeneratedInputRenderer *out, Diagnostic *diagnostic) {
  std::unique_ptr<SourceTemplate> unit;
  return SourceTemplate::Parse(source, source_name, &unit, diagnostic) &&
         internal::EmitTextTurnControls(*unit, analysis, bindings, out,
                                        diagnostic);
}
} // namespace hrx::loomify_gguf::jinja
