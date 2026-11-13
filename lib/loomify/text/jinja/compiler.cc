// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/compiler.h"
#include "lib/loomify/text/jinja/parser.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace hrx::loomify_gguf::jinja {
namespace {

// Values use native JSON kinds with decoded string data and an explicit
// Undefined kind. The helper functions operate on values, never syntax nodes.
enum class KnownKind {
  Unknown,
  String,
  Integer,
  Boolean,
  None,
  Undefined,
  List,
  Tuple,
  Mapping,
  Callable,
};
struct Value {
  std::string data;
  std::string kind;
  // A property established by construction or a typed operation, never inferred
  // from a source variable name. JSON number kind1 alone may contain a float.
  KnownKind known = KnownKind::Unknown;
  // Tuple literals share JSON storage for existing finite operations, but list
  // addition must not silently acquire tuple semantics through a value merge.
  bool tuple_data = false;
};
KnownKind MergeKind(const Value &a, const Value &b) {
  return a.known == b.known ? a.known : KnownKind::Unknown;
}
struct State {
  std::map<std::string, Value> locals;
  std::map<std::string, Value> module_locals;
  std::set<std::string> defined_macros;
  bool module_frame = true;
  // Static identities never become JSON values. Only their presence and fields
  // flow through structured control flow; ordinary loop locals remain scoped.
  struct Namespace {
    std::string present;
    int known_present = 0;
    std::map<std::string, Value> fields;
  };
  std::map<std::string, size_t> bindings;
  std::map<size_t, Namespace> namespaces;
  std::string output;
};
struct Loop {
  Value sequence;
  std::string position;
  std::string count;
};

std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() * 2);
  for (unsigned char byte : value) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

class Lowering {
public:
  Lowering(std::string_view source, Diagnostic *diagnostic)
      : source_(source), diagnostic_(diagnostic) {}

  bool Run(const Template &ast, GeneratedInputRenderer *out) {
    State state;
    GatherMacros(ast.body, true);
    for (const auto &[name, node] : macros_)
      for (const auto &parameter : node->parameters)
        if (macros_.count(parameter))
          Fail(node->offset,
               "macro parameters cannot shadow statically resolved macros");
    GatherModule(ast.body);
    for (const auto &[name, node] : macros_)
      CheckCalls(name, *node);
    for (const auto &[name, node] : macros_)
      AnalyzeCaptures(name);
    if (failed_)
      return false;
    Line("func.def public @jinja.evaluate(%environment: buffer) -> (buffer) {");
    ++indent_;
    environment_ = {"%environment", I64(2), KnownKind::Mapping};
    // The explicit environment ABI is a JSON object, not an arbitrary JSON
    // scalar/container whose missing fields would accidentally look Undefined.
    declarations_["members"] =
        "func.decl import(\"json\") @members(%value: buffer) -> (buffer, i32, i64)";
    const auto records = Fresh(), type = Fresh(), count = Fresh();
    Line(
        records + ", " + type + ", " + count +
        " = func.call @members(%environment) : (buffer) -> (buffer, i32, i64)");
    const auto object = Fresh(), is_object = Fresh();
    Line(object + " = scalar.constant 2 : i32");
    Line(is_object + " = scalar.cmpi eq, " + type + ", " + object + " : i32");
    Require(is_object, Literal("template environment must be a JSON object"));
    state.output = Literal("");
    GatherNamespaces(ast.body);
    for (const auto &[id, target] : namespace_targets_) {
      auto &object = state.namespaces[id];
      object.present = Bool(false);
      for (const auto &field : namespace_fields_)
        object.fields[field] = Undefined();
    }
    if (!macros_.empty())
      for (const auto &name : module_names_)
        state.module_locals[name] = Get(environment_, Literal(name));
    EmitNodes(ast.body, state);
    if (failed_)
      return false;
    Line("func.return " + state.output + " : buffer");
    --indent_;
    Line("}");
    for (const auto &[name, node] : macros_)
      EmitMacro(*node);
    if (failed_)
      return false;
    GeneratedInputRenderer result;
    for (const auto &[bytes, name] : literals_)
      result.source +=
          "global.rodata.def @" + name + " = bytes(\"" + Hex(bytes) + "\")\n";
    for (const auto &[name, declaration] : declarations_)
      result.source += declaration + "\n";
    result.source += body_.str();
    result.entrypoint = "jinja.evaluate";
    result.helper_requirements = {"lib/text/common/strings.loom",
                                  "lib/text/common/json.loom",
                                  "lib/text/common/jinja.loom"};
    *out = std::move(result);
    return true;
  }

private:
  std::string Fresh() { return "%j" + std::to_string(next_++); }
  void Line(const std::string &line) {
    body_ << std::string(indent_ * 2, ' ') << line << '\n';
  }
  void Fail(size_t offset, const std::string &message) {
    if (!failed_ && diagnostic_) {
      diagnostic_->offset = std::min(offset, source_.size());
      diagnostic_->message = message;
    }
    failed_ = true;
  }
  std::string I64(int64_t value) {
    const auto name = Fresh();
    Line(name + " = scalar.constant " + std::to_string(value) + " : i64");
    return name;
  }
  std::string Bool(bool value) {
    const auto name = Fresh();
    Line(name + " = scalar.constant " + (value ? "true" : "false") + " : i1");
    return name;
  }
  std::string Literal(std::string_view bytes) {
    auto found = literals_.find(std::string(bytes));
    if (found == literals_.end()) {
      const auto name = "jinja.literal." + std::to_string(literals_.size());
      found = literals_.emplace(std::string(bytes), name).first;
    }
    const auto name = Fresh();
    Line(name + " = global.load @" + found->second + " : buffer");
    return name;
  }
  Value Undefined() { return {Literal(""), I64(7), KnownKind::Undefined}; }
  Value String(std::string_view text) {
    return {Literal(text), I64(0), KnownKind::String};
  }
  Value Boolean(std::string condition) {
    const auto yes = I64(4), no = I64(5), kind = Fresh();
    Line(kind + " = scf.select " + condition + ", " + yes + ", " + no +
         " : i64");
    return {Literal(""), kind, KnownKind::Boolean};
  }
  Value Integer(const std::string &scalar) {
    return {Call("jinja.integer", {scalar}, {"i64"}, "buffer"), I64(1),
            KnownKind::Integer};
  }
  Value IntegerLiteral(std::string_view text) {
    int64_t scalar = 0;
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), scalar);
    // Preserve arbitrary JSON integer text, but only use the scalar fast path
    // when the literal is representable. Huge sequence keys are simply missing,
    // like identical keys loaded dynamically from the explicit environment.
    const auto known =
        parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size()
            ? KnownKind::Integer
            : KnownKind::Unknown;
    return {Literal(text), I64(1), known};
  }
  std::string Call(const std::string &name,
                   const std::vector<std::string> &arguments,
                   const std::vector<std::string> &types,
                   const std::string &result_type) {
    Declare(name, types, {result_type});
    const auto value = Fresh();
    Line(value + " = func.call @" + name + "(" + Join(arguments) + ") : (" +
         Join(types) + ") -> (" + result_type + ")");
    return value;
  }
  Value Pair(const std::string &name, const std::vector<std::string> &arguments,
             const std::vector<std::string> &types) {
    Declare(name, types, {"buffer", "i64"});
    Value result{Fresh(), Fresh()};
    Line(result.data + ", " + result.kind + " = func.call @" + name + "(" +
         Join(arguments) + ") : (" + Join(types) + ") -> (buffer, i64)");
    return result;
  }
  static std::string Join(const std::vector<std::string> &values) {
    std::string result;
    for (const auto &value : values) {
      if (!result.empty())
        result += ", ";
      result += value;
    }
    return result;
  }
  void Declare(const std::string &name, const std::vector<std::string> &types,
               const std::vector<std::string> &results) {
    std::vector<std::string> arguments;
    for (size_t i = 0; i < types.size(); ++i)
      arguments.push_back("%a" + std::to_string(i) + ": " + types[i]);
    declarations_[name] = "func.decl @" + name + "(" + Join(arguments) +
                          ") -> (" + Join(results) + ")";
  }
  std::string Truth(Value value) {
    return Call("jinja.truth", {value.data, value.kind}, {"buffer", "i64"},
                "i1");
  }
  std::string ToString(Value value) {
    return Call("jinja.string", {value.data, value.kind}, {"buffer", "i64"},
                "buffer");
  }
  std::string ToInteger(Value value) {
    return Call("jinja.tointeger", {value.data, value.kind}, {"buffer", "i64"},
                "i64");
  }
  std::string Concat(const std::string &left, const std::string &right) {
    return Call("jinja.join", {left, right}, {"buffer", "buffer"}, "buffer");
  }
  std::string Compare(const std::string &predicate, const std::string &a,
                      const std::string &b) {
    const auto result = Fresh();
    Line(result + " = scalar.cmpi " + predicate + ", " + a + ", " + b +
         " : i64");
    return result;
  }
  std::string Not(const std::string &condition) {
    const auto no = Bool(false), result = Fresh();
    Line(result + " = scalar.cmpi eq, " + condition + ", " + no + " : i1");
    return result;
  }
  Value Get(Value subject, const std::string &key) {
    return TupleResult(Pair("jinja.get", {subject.data, subject.kind, key},
                            {"buffer", "i64", "buffer"}),
                       subject);
  }
  static Value TupleResult(Value result, const Value &source) {
    result.tuple_data = source.tuple_data;
    return result;
  }
  void Require(const std::string &condition, const std::string &diagnostic) {
    declarations_["require"] =
        "func.decl import(\"input\") @require(%condition: i32, %diagnostic: buffer)";
    const auto yes = Fresh(), no = Fresh(), condition32 = Fresh();
    Line(yes + " = scalar.constant 1 : i32");
    Line(no + " = scalar.constant 0 : i32");
    Line(condition32 + " = scf.select " + condition + ", " + yes + ", " + no +
         " : i32");
    Line("func.call @require(" + condition32 + ", " + diagnostic +
         ") : (i32, buffer)");
  }
  void GatherMacros(const std::vector<Node> &nodes, bool module) {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Macro) {
        if (!module || macros_.count(node.text) || Reserved(node.text))
          Fail(node.offset, "macros require unique module-level names");
        else
          macros_[node.text] = &node;
        for (const auto &parameter : node.parameters)
          if (Reserved(parameter))
            Fail(node.offset, "reserved macro parameter: " + parameter);
        GatherMacros(node.body, false);
      } else {
        GatherMacros(node.body, false);
        GatherMacros(node.otherwise, false);
      }
    }
  }
  static bool Reserved(const std::string &name) {
    return name == "loop" || name == "namespace" || name == "raise_exception" ||
           name == "strftime_now";
  }
  void GatherModule(const std::vector<Node> &nodes) {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Set &&
          node.text.find('.') == std::string::npos) {
        module_names_.insert(node.text);
        if (Constructor(node)) {
          const auto [it, inserted] =
              module_bindings_.emplace(node.text, node.offset);
          if (!inserted && !macros_.empty())
            Fail(
                node.offset,
                "multiple module namespace identities cannot be captured by macros");
        } else if (node.expression.kind == Expr::Kind::Name &&
                   module_bindings_.count(node.expression.text)) {
          module_bindings_[node.text] =
              module_bindings_.at(node.expression.text);
        }
      }
      if (node.kind == Node::Kind::If) {
        GatherModule(node.body);
        GatherModule(node.otherwise);
      }
    }
  }
  void Calls(const Expr &expr, std::set<std::string> *calls) {
    if (expr.kind == Expr::Kind::Call && !expr.args.empty() &&
        expr.args[0].kind == Expr::Kind::Name &&
        macros_.count(expr.args[0].text))
      calls->insert(expr.args[0].text);
    for (const auto &arg : expr.args)
      Calls(arg, calls);
  }
  void Calls(const std::vector<Node> &nodes, std::set<std::string> *calls) {
    for (const auto &node : nodes) {
      Calls(node.expression, calls);
      Calls(node.body, calls);
      Calls(node.otherwise, calls);
    }
  }
  void CheckCalls(const std::string &name, const Node &node) {
    if (checked_macros_.count(name))
      return;
    if (!checking_macros_.insert(name).second) {
      Fail(node.offset, "recursive macros are not supported");
      return;
    }
    auto &calls = macro_calls_[name];
    Calls(node.body, &calls);
    for (const auto &value : node.defaults)
      Calls(value, &calls);
    for (const auto &callee : calls)
      CheckCalls(callee, *macros_.at(callee));
    checking_macros_.erase(name);
    checked_macros_.insert(name);
  }
  void ReferencedNames(const Expr &expr, std::set<std::string> *names) {
    if (expr.kind == Expr::Kind::Name)
      names->insert(expr.text);
    for (const auto &arg : expr.args)
      ReferencedNames(arg, names);
  }
  void ReferencedNames(const std::vector<Node> &nodes,
                       std::set<std::string> *names,
                       std::set<std::string> *writes,
                       std::set<size_t> *local_namespaces) {
    for (const auto &node : nodes) {
      ReferencedNames(node.expression, names);
      if (node.kind == Node::Kind::Set) {
        const auto dot = node.text.find('.');
        if (dot != std::string::npos) {
          names->insert(node.text.substr(0, dot));
          writes->insert(node.text.substr(0, dot));
        }
      }
      if (Constructor(node))
        local_namespaces->insert(node.offset);
      ReferencedNames(node.body, names, writes, local_namespaces);
      ReferencedNames(node.otherwise, names, writes, local_namespaces);
    }
  }
  void AnalyzeCaptures(const std::string &name) {
    if (macro_captures_.count(name) || failed_)
      return;
    const auto &macro = *macros_.at(name);
    auto &capture = macro_captures_[name];
    std::set<std::string> names, writes;
    ReferencedNames(macro.body, &names, &writes, &capture.local_namespaces);
    for (const auto &value : macro.defaults)
      ReferencedNames(value, &names);
    for (const auto &parameter : macro.parameters)
      names.erase(parameter);
    for (const auto &used : names) {
      if (module_names_.count(used))
        capture.values.insert(used);
      if (module_bindings_.count(used))
        capture.namespaces.insert(module_bindings_.at(used));
    }
    for (const auto &callee : macro_calls_[name]) {
      AnalyzeCaptures(callee);
      const auto &child = macro_captures_.at(callee);
      capture.values.insert(child.values.begin(), child.values.end());
      capture.namespaces.insert(child.namespaces.begin(),
                                child.namespaces.end());
      capture.effects.insert(child.effects.begin(), child.effects.end());
    }
    // A direct field write may use an admitted static alias, including one
    // selected by equal branch bindings. Return all captured identities rather
    // than create a second alias/branch analysis beside ordinary lowering.
    if (!writes.empty())
      capture.effects.insert(capture.namespaces.begin(),
                             capture.namespaces.end());
  }
  static State NamespaceSubset(const State &state,
                               const std::set<size_t> &ids) {
    State selected;
    for (const auto id : ids)
      selected.namespaces[id] = state.namespaces.at(id);
    return selected;
  }
  static bool Constructor(const Node &node) {
    return node.kind == Node::Kind::Set &&
           node.expression.kind == Expr::Kind::Call &&
           !node.expression.args.empty() &&
           node.expression.args[0].kind == Expr::Kind::Name &&
           node.expression.args[0].text == "namespace";
  }
  void GatherNamespaces(const std::vector<Node> &nodes) {
    for (const auto &node : nodes) {
      if (Constructor(node)) {
        namespace_targets_[node.offset] = node.text;
        for (size_t i = 1; i < node.expression.keywords.size(); ++i)
          namespace_fields_.insert(node.expression.keywords[i]);
      }
      if (node.kind == Node::Kind::Set) {
        const auto dot = node.text.find('.');
        if (dot != std::string::npos)
          namespace_fields_.insert(node.text.substr(dot + 1));
      }
      GatherNamespaces(node.body);
      GatherNamespaces(node.otherwise);
    }
  }
  void Prebind(const std::vector<Node> &nodes, State &state) {
    for (const auto &node : nodes) {
      if (Constructor(node)) {
        const auto existing = state.bindings.find(node.text);
        if (existing != state.bindings.end() &&
            existing->second != node.offset) {
          Fail(node.offset,
               "conditional namespace rebinding has ambiguous identity");
          continue;
        }
        if (!state.locals.count(node.text))
          state.locals[node.text] = Name(node.text, state);
        state.bindings[node.text] = node.offset;
      }
      if (node.kind == Node::Kind::If) {
        Prebind(node.body, state);
        Prebind(node.otherwise, state);
      }
    }
  }
  void ResetLoopNamespaces(const std::vector<Node> &nodes, State &state) {
    for (const auto &node : nodes) {
      if (Constructor(node)) {
        auto &object = state.namespaces.at(node.offset);
        object.present = Bool(false);
        object.known_present = 0;
        for (auto &[field, value] : object.fields)
          value = Undefined();
      }
      ResetLoopNamespaces(node.body, state);
      ResetLoopNamespaces(node.otherwise, state);
    }
  }
  // The same finite flatten/merge seam is used by statement and lazy expression
  // branches, and is available for later macro calls that mutate namespaces.
  void NamespaceValues(const State &state, std::vector<std::string> *values,
                       std::vector<std::string> *types) {
    for (const auto &[id, object] : state.namespaces) {
      values->push_back(object.present);
      types->push_back("i1");
      for (const auto &[field, value] : object.fields) {
        values->push_back(value.data);
        values->push_back(value.kind);
        types->push_back("buffer");
        types->push_back("i64");
      }
    }
  }
  State NamespaceResults(const State &state, std::vector<std::string> *values,
                         std::vector<std::string> *types) {
    State result = state;
    for (auto &[id, object] : result.namespaces) {
      object.present = Fresh();
      object.known_present = -1;
      for (auto &[field, value] : object.fields)
        value = {Fresh(), Fresh(), KnownKind::Unknown, value.tuple_data};
    }
    NamespaceValues(result, values, types);
    return result;
  }
  void MergeNamespaces(State &state, State result, const State &yes,
                       const State &no) {
    for (auto &[id, object] : result.namespaces) {
      const auto &a = yes.namespaces.at(id), &b = no.namespaces.at(id);
      object.known_present =
          a.known_present == b.known_present ? a.known_present : -1;
      for (auto &[field, value] : object.fields) {
        value.known = MergeKind(a.fields.at(field), b.fields.at(field));
        value.tuple_data =
            a.fields.at(field).tuple_data || b.fields.at(field).tuple_data;
      }
    }
    state.namespaces = std::move(result.namespaces);
  }
  template <typename Yes, typename No>
  Value SelectValue(const std::string &condition, State &state, Yes yes_fn,
                    No no_fn) {
    Value result{Fresh(), Fresh()};
    std::vector<std::string> outputs{result.data, result.kind},
        types{"buffer", "i64"};
    auto merged = NamespaceResults(state, &outputs, &types);
    State yes = state, no = state;
    Line(Join(outputs) + " = scf.if " + condition + " -> (" + Join(types) +
         ") {");
    ++indent_;
    const auto a = yes_fn(yes);
    std::vector<std::string> av{a.data, a.kind}, at{"buffer", "i64"};
    NamespaceValues(yes, &av, &at);
    Line("scf.yield " + Join(av) + " : " + Join(at));
    --indent_;
    Line("} else {");
    ++indent_;
    const auto b = no_fn(no);
    std::vector<std::string> bv{b.data, b.kind}, bt{"buffer", "i64"};
    NamespaceValues(no, &bv, &bt);
    Line("scf.yield " + Join(bv) + " : " + Join(bt));
    --indent_;
    Line("}");
    MergeNamespaces(state, std::move(merged), yes, no);
    result.known = MergeKind(a, b);
    result.tuple_data = a.tuple_data || b.tuple_data;
    return result;
  }
  Value NamespaceField(const std::string &name, const std::string &field,
                       State &state) {
    const auto id = state.bindings.at(name);
    auto &object = state.namespaces.at(id);
    const auto found = object.fields.find(field);
    const auto value =
        found == object.fields.end() ? Undefined() : found->second;
    if (object.known_present == 1)
      return value;
    const auto fallback = state.locals.at(name);
    if (object.known_present == 0)
      return Get(fallback, Literal(field));
    return SelectValue(
        object.present, state, [&](State &) { return value; },
        [&](State &) { return Get(fallback, Literal(field)); });
  }
  Value Name(const std::string &name, State &state, size_t offset = 0) {
    if (macros_.count(name)) {
      Fail(offset, "macro objects cannot escape as values");
      return Undefined();
    }
    if (state.bindings.count(name)) {
      Fail(
          offset,
          "namespace objects support field access and defined tests, not JSON values");
      return Undefined();
    }
    const auto found = state.locals.find(name);
    if (found != state.locals.end())
      return found->second;
    if (name == "raise_exception")
      return {Literal(""), I64(8), KnownKind::Callable};
    if (name == "strftime_now") {
      const auto capability =
          Get(environment_, Literal("__loomify_strftime_now"));
      const auto present = Compare("eq", capability.kind, I64(2));
      const auto callable = I64(8), absent = I64(7), kind = Fresh();
      Line(kind + " = scf.select " + present + ", " + callable + ", " + absent +
           " : i64");
      return {capability.data, kind};
    }
    return Get(environment_, Literal(name));
  }

  Value Choice(const std::string &condition, const Expr &yes, const Expr &no,
               State &state) {
    return SelectValue(
        condition, state,
        [&](State &branch) { return Expression(yes, branch); },
        [&](State &branch) { return Expression(no, branch); });
  }

  Value Expression(const Expr &expr, State &state) {
    using Kind = Expr::Kind;
    if (failed_)
      return Undefined();
    switch (expr.kind) {
    case Kind::Name:
      return Name(expr.text, state, expr.offset);
    case Kind::String:
      return String(expr.text);
    case Kind::Integer:
      return IntegerLiteral(expr.text);
    case Kind::Boolean:
      return {Literal(""),
              I64(expr.text == "true" || expr.text == "True" ? 4 : 5),
              KnownKind::Boolean};
    case Kind::None:
      return {Literal(""), I64(6), KnownKind::None};
    case Kind::Unary: {
      if (expr.args[0].kind == Kind::Integer &&
          (expr.text == "+" || expr.text == "-")) {
        const auto &integer = expr.args[0].text;
        return IntegerLiteral(expr.text == "-" && integer != "0" ? "-" + integer
                                                                 : integer);
      }
      const auto value = Expression(expr.args[0], state);
      if (expr.text == "not")
        return Boolean(Not(Truth(value)));
      if (expr.text == "+")
        return Integer(ToInteger(value));
      if (expr.text == "-") {
        const auto scalar = ToInteger(value), zero = I64(0), negated = Fresh();
        Require(Compare("ne", scalar, I64(std::numeric_limits<int64_t>::min())),
                Literal("template integer negation exceeds signed range"));
        Line(negated + " = scalar.subi " + zero + ", " + scalar + " : i64");
        return Integer(negated);
      }
      break;
    }
    case Kind::Binary: {
      const auto left = Expression(expr.args[0], state);
      if (expr.text == "and" || expr.text == "or") {
        const auto truth = Truth(left);
        return SelectValue(
            truth, state,
            [&](State &branch) {
              return expr.text == "and" ? Expression(expr.args[1], branch)
                                        : left;
            },
            [&](State &branch) {
              return expr.text == "or" ? Expression(expr.args[1], branch)
                                       : left;
            });
      }
      const auto right = Expression(expr.args[1], state);
      if (expr.text == "+") {
        if (left.tuple_data || right.tuple_data) {
          Fail(expr.offset,
               "tuple-derived addition is outside the supported list domain");
          return Undefined();
        }
        if (left.known == KnownKind::Integer &&
            right.known == KnownKind::Integer)
          return Integer(Call("jinja.add_integer",
                              {ToInteger(left), ToInteger(right)},
                              {"i64", "i64"}, "i64"));
        if (left.known == KnownKind::String && right.known == KnownKind::String)
          return {Concat(left.data, right.data), I64(0), KnownKind::String};
        auto result =
            Pair("jinja.add", {left.data, left.kind, right.data, right.kind},
                 {"buffer", "i64", "buffer", "i64"});
        const auto integral = [](KnownKind kind) {
          return kind == KnownKind::Integer || kind == KnownKind::Boolean;
        };
        if (integral(left.known) && integral(right.known))
          result.known = KnownKind::Integer;
        // The checked generic operation can succeed with a list operand only
        // as list+list. Keep its validation, but publish that successful result
        // tag directly rather than rediscover it across helper/namespace phis.
        if (left.known == KnownKind::List || right.known == KnownKind::List) {
          result.known = KnownKind::List;
          result.kind = I64(3);
        }
        return result;
      }
      if (expr.text == "~")
        return {Concat(ToString(left), ToString(right)), I64(0),
                KnownKind::String};
      if (expr.text == "%")
        return Integer(Call("jinja.remainder",
                            {ToInteger(left), ToInteger(right)}, {"i64", "i64"},
                            "i64"));
      if (expr.text == "==" || expr.text == "!=") {
        auto equal =
            Call("jinja.equal", {left.data, left.kind, right.data, right.kind},
                 {"buffer", "i64", "buffer", "i64"}, "i1");
        return Boolean(expr.text == "!=" ? Not(equal) : equal);
      }
      if (expr.text == "in" || expr.text == "not in") {
        auto contains = Call("jinja.contains",
                             {right.data, right.kind, left.data, left.kind},
                             {"buffer", "i64", "buffer", "i64"}, "i1");
        return Boolean(expr.text == "not in" ? Not(contains) : contains);
      }
      const std::map<std::string, std::string> predicates = {
          {"<", "slt"}, {"<=", "sle"}, {">", "sgt"}, {">=", "sge"}};
      const auto predicate = predicates.find(expr.text);
      if (predicate != predicates.end())
        return Boolean(
            Compare(predicate->second, ToInteger(left), ToInteger(right)));
      if (expr.text == "-") {
        const auto a = ToInteger(left), b = ToInteger(right), result = Fresh();
        Line(result + " = scalar.subi " + a + ", " + b + " : i64");
        const auto a_negative = Compare("slt", a, I64(0));
        const auto b_negative = Compare("slt", b, I64(0));
        const auto r_negative = Compare("slt", result, I64(0));
        const auto different = Fresh(), changed = Fresh(), overflow = Fresh();
        Line(different + " = scalar.cmpi ne, " + a_negative + ", " +
             b_negative + " : i1");
        Line(changed + " = scalar.cmpi ne, " + a_negative + ", " + r_negative +
             " : i1");
        Line(overflow + " = scalar.andi " + different + ", " + changed +
             " : i1");
        Require(Not(overflow),
                Literal("template integer subtraction exceeds signed range"));
        return Integer(result);
      }
      break;
    }
    case Kind::Conditional:
      return Choice(Truth(Expression(expr.args[0], state)), expr.args[1],
                    expr.args[2], state);
    case Kind::GetAttr:
      if (expr.args[0].kind == Kind::Name &&
          state.bindings.count(expr.args[0].text))
        return NamespaceField(expr.args[0].text, expr.text, state);
      if (expr.args[0].kind == Kind::Name && expr.args[0].text == "loop" &&
          !loops_.empty())
        return LoopAttribute(expr, state);
      return Get(Expression(expr.args[0], state), Literal(expr.text));
    case Kind::GetItem: {
      const auto value = Expression(expr.args[0], state);
      const auto key = Expression(expr.args[1], state);
      if (key.known == KnownKind::String)
        return Get(value, key.data);
      if (key.known == KnownKind::Integer)
        return TupleResult(Pair("jinja.index",
                                {value.data, value.kind, ToInteger(key)},
                                {"buffer", "i64", "i64"}),
                           value);
      if (key.known == KnownKind::Boolean) {
        const auto yes = I64(1), no = I64(0), ordinal = Fresh();
        Line(ordinal + " = scf.select " + Compare("eq", key.kind, I64(4)) +
             ", " + yes + ", " + no + " : i64");
        return TupleResult(Pair("jinja.index",
                                {value.data, value.kind, ordinal},
                                {"buffer", "i64", "i64"}),
                           value);
      }
      if (key.known == KnownKind::None || key.known == KnownKind::Undefined ||
          key.known == KnownKind::List || key.known == KnownKind::Tuple ||
          key.known == KnownKind::Mapping) {
        // JSON objects have string keys, and sequences have integral indices.
        // A bad key is missing, but a missing subject must still raise.
        Require(Compare("ne", value.kind, I64(7)),
                Literal("cannot access a member of Undefined"));
        return Undefined();
      }
      // Unknown keys retain checked generic item semantics. In particular an
      // invalid ordinary key produces Undefined, rather than eagerly parsing
      // every nonstring key as an integer or assuming a branch's impossible
      // tag.
      return TupleResult(Pair("jinja.item",
                              {value.data, value.kind, key.data, key.kind},
                              {"buffer", "i64", "buffer", "i64"}),
                         value);
    }
    case Kind::Slice: {
      const auto value = Expression(expr.args[0], state);
      const bool omitted_begin = expr.args[1].kind == Kind::None;
      const bool omitted_end = expr.args[2].kind == Kind::None;
      const auto begin = expr.args[1].kind == Kind::None
                             ? I64(0)
                             : ToInteger(Expression(expr.args[1], state));
      const auto end = expr.args[2].kind == Kind::None
                           ? I64(std::numeric_limits<int64_t>::max())
                           : ToInteger(Expression(expr.args[2], state));
      const bool reverse = expr.args[3].kind == Kind::Unary &&
                           expr.args[3].text == "-" &&
                           expr.args[3].args[0].kind == Kind::Integer &&
                           expr.args[3].args[0].text == "1";
      const bool forward = expr.args[3].kind == Kind::Unary &&
                           expr.args[3].text == "+" &&
                           expr.args[3].args[0].kind == Kind::Integer &&
                           expr.args[3].args[0].text == "1";
      if (expr.args[3].kind != Kind::None && !reverse && !forward &&
          !(expr.args[3].kind == Kind::Integer && expr.args[3].text == "1")) {
        Fail(
            expr.offset,
            "slice steps other than 1 or -1 are not supported by this compiler checkpoint");
        return Undefined();
      }
      auto result =
          reverse ? Pair("jinja.reverse_slice",
                         {value.data, value.kind, begin, end, I64(-1),
                          Bool(omitted_begin), Bool(omitted_end)},
                         {"buffer", "i64", "i64", "i64", "i64", "i1", "i1"})
                  : Pair("jinja.slice", {value.data, value.kind, begin, end},
                         {"buffer", "i64", "i64", "i64"});
      // A slice can only return the same supported sequence family or error.
      if (value.known == KnownKind::String || value.known == KnownKind::List ||
          value.known == KnownKind::Tuple)
        result.known = value.known;
      result.tuple_data = value.tuple_data;
      return result;
    }
    case Kind::Test:
      return Test(expr, state);
    case Kind::Filter:
      return Filter(expr, state);
    case Kind::Call:
      return Function(expr, state);
    case Kind::List:
    case Kind::Tuple:
    case Kind::Dict:
      return Collection(expr, state);
    }
    Fail(expr.offset, "unsupported expression operation: " + expr.text);
    return Undefined();
  }

  Value LoopAttribute(const Expr &expr, State &) {
    const auto &loop = loops_.back();
    if (expr.text == "index0")
      return Integer(loop.position);
    if (expr.text == "index") {
      const auto one = I64(1), next = Fresh();
      Line(next + " = scalar.addi " + loop.position + ", " + one + " : i64");
      return Integer(next);
    }
    if (expr.text == "first")
      return Boolean(Compare("eq", loop.position, I64(0)));
    if (expr.text == "length")
      return Integer(loop.count);
    const auto one = I64(1), next = Fresh();
    Line(next + " = scalar.addi " + loop.position + ", " + one + " : i64");
    if (expr.text == "last")
      return Boolean(Compare("eq", next, loop.count));
    if (expr.text == "previtem" || expr.text == "nextitem") {
      const auto position = Fresh();
      Line(position + " = scalar." +
           (expr.text == "previtem" ? "subi " : "addi ") + loop.position +
           ", " + one + " : i64");
      const auto exists = expr.text == "previtem"
                              ? Compare("sgt", loop.position, I64(0))
                              : Compare("slt", position, loop.count);
      Value result{Fresh(), Fresh()};
      Line(result.data + ", " + result.kind + " = scf.if " + exists +
           " -> (buffer, i64) {");
      ++indent_;
      const auto value = Pair(
          "jinja.index", {loop.sequence.data, loop.sequence.kind, position},
          {"buffer", "i64", "i64"});
      Line("scf.yield " + value.data + ", " + value.kind + " : buffer, i64");
      --indent_;
      Line("} else {");
      ++indent_;
      const auto absent = Undefined();
      Line("scf.yield " + absent.data + ", " + absent.kind + " : buffer, i64");
      --indent_;
      Line("}");
      result.tuple_data = loop.sequence.tuple_data;
      return result;
    }
    Fail(expr.offset, "unsupported loop attribute: " + expr.text);
    return Undefined();
  }

  Value Test(const Expr &expr, State &state) {
    if (expr.args.size() != 1) {
      Fail(expr.offset, "this test does not accept arguments: " + expr.text);
      return Undefined();
    }
    if (expr.args[0].kind == Expr::Kind::Name &&
        state.bindings.count(expr.args[0].text) &&
        (expr.text == "defined" || expr.text == "undefined")) {
      const auto &name = expr.args[0].text;
      const auto &object = state.namespaces.at(state.bindings.at(name));
      const auto fallback = Compare("ne", state.locals.at(name).kind, I64(7)),
                 defined = Fresh();
      Line(defined + " = scalar.ori " + object.present + ", " + fallback +
           " : i1");
      return Boolean(expr.text == "defined" ? defined : Not(defined));
    }
    const auto value = Expression(expr.args[0], state);
    std::vector<int64_t> kinds;
    if (expr.text == "defined")
      return Boolean(Compare("ne", value.kind, I64(7)));
    if (expr.text == "undefined")
      kinds = {7};
    else if (expr.text == "none")
      kinds = {6};
    else if (expr.text == "string")
      kinds = {0};
    else if (expr.text == "mapping")
      kinds = {2};
    // Default Jinja Undefined supplies an empty iterator. The original Llama
    // tool-content branch consequently reaches tojson, which rejects Undefined.
    else if (expr.text == "iterable" || expr.text == "sequence")
      kinds = {0, 2, 3, 7};
    else if (expr.text == "number")
      kinds = {1, 4, 5};
    else if (expr.text == "boolean")
      kinds = {4, 5};
    else if (expr.text == "true")
      kinds = {4};
    else if (expr.text == "false")
      kinds = {5};
    else {
      Fail(expr.offset, "unsupported value test: " + expr.text);
      return Undefined();
    }
    auto condition = Bool(false);
    for (int64_t kind : kinds) {
      const auto matches = Compare("eq", value.kind, I64(kind)), next = Fresh();
      Line(next + " = scalar.ori " + condition + ", " + matches + " : i1");
      condition = next;
    }
    return Boolean(condition);
  }

  Value Filter(const Expr &expr, State &state) {
    if (expr.args.empty()) {
      Fail(expr.offset, "filter needs a subject");
      return Undefined();
    }
    const auto value = Expression(expr.args[0], state);
    if (expr.text == "list" && expr.args.size() == 1) {
      auto result = TupleResult(
          Pair("jinja.list", {value.data, value.kind}, {"buffer", "i64"}),
          value);
      result.known = KnownKind::List;
      return result;
    }
    if (expr.text == "join" && expr.args.size() <= 2) {
      if (expr.keywords.size() > 1 && !expr.keywords[1].empty() &&
          expr.keywords[1] != "d") {
        Fail(expr.offset, "join supports only the delimiter argument");
        return Undefined();
      }
      const auto separator =
          expr.args.size() == 2 ? Expression(expr.args[1], state) : String("");
      auto result =
          Pair("jinja.join_values",
               {value.data, value.kind, separator.data, separator.kind},
               {"buffer", "i64", "buffer", "i64"});
      result.known = KnownKind::String;
      return result;
    }
    if (expr.text == "items" && expr.args.size() == 1) {
      if (items_expression_ != &expr) {
        Fail(expr.offset,
             "items views are supported only as immediate loop iterables");
        return Undefined();
      }
      return TupleResult(Pair("jinja.items",
                              {value.data, value.kind, Bool(true)},
                              {"buffer", "i64", "i1"}),
                         value);
    }
    if ((expr.text == "default" || expr.text == "d") && expr.args.size() <= 2) {
      if (expr.keywords.size() > 1 && !expr.keywords[1].empty() &&
          expr.keywords[1] != "default_value") {
        Fail(expr.offset, "default supports only the default_value argument");
        return Undefined();
      }
      const auto fallback =
          expr.args.size() == 2 ? Expression(expr.args[1], state) : String("");
      auto result = Pair("jinja.default",
                         {value.data, value.kind, fallback.data, fallback.kind},
                         {"buffer", "i64", "buffer", "i64"});
      result.tuple_data = value.tuple_data || fallback.tuple_data;
      return result;
    }
    if ((expr.text == "first" || expr.text == "last") && expr.args.size() == 1)
      return TupleResult(Pair("jinja." + expr.text, {value.data, value.kind},
                              {"buffer", "i64"}),
                         value);
    if (expr.text == "trim" && expr.args.size() == 1) {
      auto result =
          Pair("jinja.trim", {value.data, value.kind}, {"buffer", "i64"});
      result.known = KnownKind::String;
      return result;
    }
    if ((expr.text == "length" || expr.text == "count") &&
        expr.args.size() == 1) {
      const auto length = Call("jinja.length", {value.data, value.kind},
                               {"buffer", "i64"}, "i64");
      return Integer(length);
    }
    if (expr.text == "safe" && expr.args.size() == 1)
      return {ToString(value), I64(0), KnownKind::String};
    if (expr.text == "string" && expr.args.size() == 1)
      return {ToString(value), I64(0), KnownKind::String};
    if (expr.text == "tojson" && expr.args.size() <= 2) {
      int64_t indent = -1;
      if (expr.args.size() == 2) {
        const auto &arg = expr.args[1];
        if (expr.keywords.size() > 1 && expr.keywords[1] == "separators" &&
            arg.kind == Expr::Kind::Tuple && arg.args.size() == 2 &&
            arg.args[0].kind == Expr::Kind::String &&
            arg.args[1].kind == Expr::Kind::String &&
            ((arg.args[0].text == "," && arg.args[1].text == ":") ||
             (arg.args[0].text == ", " && arg.args[1].text == ": "))) {
          indent = arg.args[0].text == "," ? -2 : -1;
        } else if (arg.kind != Expr::Kind::Integer || arg.text != "4" ||
                   (expr.keywords.size() > 1 && !expr.keywords[1].empty() &&
                    expr.keywords[1] != "indent")) {
          Fail(expr.offset,
               "tojson supports only default formatting or indent=4");
          return Undefined();
        } else
          indent = 4;
      }
      return {Call("jinja.tojson", {value.data, value.kind, I64(indent)},
                   {"buffer", "i64", "i64"}, "buffer"),
              I64(0), KnownKind::String};
    }
    Fail(expr.offset, "unsupported filter or arguments: " + expr.text);
    return Undefined();
  }

  std::string MacroSymbol(const std::string &name) {
    return "jinja.macro." + name;
  }
  Value MacroCall(const Expr &expr, State &state) {
    const auto &name = expr.args[0].text;
    const auto &macro = *macros_.at(name);
    const auto &captures = macro_captures_.at(name);
    if (!in_macro_ && !state.defined_macros.count(name)) {
      Fail(expr.offset, "macro call precedes its declaration: " + name);
      return Undefined();
    }
    if (!in_macro_) {
      std::set<std::string> pending{name}, seen;
      while (!pending.empty()) {
        const auto current = *pending.begin();
        pending.erase(pending.begin());
        if (!seen.insert(current).second)
          continue;
        if (!state.defined_macros.count(current)) {
          Fail(expr.offset,
               "macro dependency is not declared at this call: " + current);
          return Undefined();
        }
        pending.insert(macro_calls_[current].begin(),
                       macro_calls_[current].end());
      }
    }
    std::vector<Value> parameters(macro.parameters.size());
    std::vector<bool> supplied(parameters.size(), false);
    size_t positional = 0;
    for (size_t i = 1; i < expr.args.size(); ++i) {
      size_t parameter = positional++;
      if (!expr.keywords[i].empty()) {
        const auto found = std::find(macro.parameters.begin(),
                                     macro.parameters.end(), expr.keywords[i]);
        parameter = found - macro.parameters.begin();
      }
      if (parameter >= parameters.size() || supplied[parameter]) {
        Fail(expr.offset, "invalid or duplicate macro argument: " + name);
        return Undefined();
      }
      parameters[parameter] = Expression(expr.args[i], state);
      if (parameters[parameter].tuple_data) {
        Fail(
            expr.offset,
            "tuple-derived macro arguments are outside the supported list domain");
        return Undefined();
      }
      supplied[parameter] = true;
    }
    std::vector<std::string> arguments{environment_.data}, types{"buffer"};
    for (size_t i = 0; i < parameters.size(); ++i) {
      if (!supplied[i])
        parameters[i] = Undefined();
      arguments.insert(arguments.end(), {parameters[i].data, parameters[i].kind,
                                         Bool(supplied[i])});
      types.insert(types.end(), {"buffer", "i64", "i1"});
    }
    for (const auto &capture : captures.values) {
      const auto found = state.module_locals.find(capture);
      const auto value = found == state.module_locals.end()
                             ? Get(environment_, Literal(capture))
                             : found->second;
      if (value.tuple_data) {
        Fail(
            expr.offset,
            "tuple-derived macro captures are outside the supported list domain");
        return Undefined();
      }
      arguments.insert(arguments.end(), {value.data, value.kind});
      types.insert(types.end(), {"buffer", "i64"});
    }
    NamespaceValues(NamespaceSubset(state, captures.namespaces), &arguments,
                    &types);
    const auto output = Fresh();
    std::vector<std::string> results{output}, result_types{"buffer"};
    auto updated = NamespaceResults(NamespaceSubset(state, captures.effects),
                                    &results, &result_types);
    Line(Join(results) + " = func.call @" + MacroSymbol(name) + "(" +
         Join(arguments) + ") : (" + Join(types) + ") -> (" +
         Join(result_types) + ")");
    for (auto &[id, object] : updated.namespaces) {
      // A macro can mutate captured fields, but cannot construct or delete the
      // caller's namespace identity. Its own constructors use distinct slots.
      object.present = state.namespaces.at(id).present;
      object.known_present = state.namespaces.at(id).known_present;
      state.namespaces[id] = std::move(object);
    }
    return {output, I64(0), KnownKind::String};
  }
  void EmitMacro(const Node &macro) {
    const auto &captures = macro_captures_.at(macro.text);
    State state;
    state.module_frame = false;
    state.bindings = module_bindings_;
    auto saved_environment = environment_;
    auto saved_loops = std::move(loops_);
    loops_.clear();
    in_macro_ = true;
    std::vector<std::string> arguments{"%environment: buffer"};
    std::vector<std::string> provided;
    for (const auto &parameter : macro.parameters) {
      Value value{Fresh(), Fresh()};
      const auto present = Fresh();
      arguments.insert(
          arguments.end(),
          {value.data + ": buffer", value.kind + ": i64", present + ": i1"});
      state.locals[parameter] = value;
      provided.push_back(present);
      state.bindings.erase(parameter);
    }
    for (const auto &capture : captures.values) {
      const Value value{Fresh(), Fresh()};
      arguments.insert(arguments.end(),
                       {value.data + ": buffer", value.kind + ": i64"});
      state.module_locals[capture] = value;
      if (!state.locals.count(capture))
        state.locals[capture] = value;
    }
    std::vector<std::string> result_types{"buffer"};
    for (const auto id : captures.namespaces) {
      auto &object = state.namespaces[id];
      object.present = Fresh();
      object.known_present = -1;
      arguments.push_back(object.present + ": i1");
      if (captures.effects.count(id))
        result_types.push_back("i1");
      for (const auto &field : namespace_fields_) {
        const Value value{Fresh(), Fresh()};
        object.fields[field] = value;
        arguments.insert(arguments.end(),
                         {value.data + ": buffer", value.kind + ": i64"});
        if (captures.effects.count(id))
          result_types.insert(result_types.end(), {"buffer", "i64"});
      }
    }
    Line("func.def @" + MacroSymbol(macro.text) + "(" + Join(arguments) +
         ") -> (" + Join(result_types) + ") {");
    ++indent_;
    environment_ = {"%environment", I64(2), KnownKind::Mapping};
    state.output = Literal("");
    for (const auto id : captures.local_namespaces)
      if (!state.namespaces.count(id)) {
        auto &object = state.namespaces[id];
        object.present = Bool(false);
        for (const auto &field : namespace_fields_)
          object.fields[field] = Undefined();
      }
    ResetLoopNamespaces(macro.body, state);
    const size_t first_default =
        macro.parameters.size() - macro.defaults.size();
    for (size_t i = first_default; i < macro.parameters.size(); ++i) {
      const auto original = state.locals.at(macro.parameters[i]);
      const auto value = SelectValue(
          provided[i], state, [&](State &) { return original; },
          [&](State &branch) {
            return Expression(macro.defaults[i - first_default], branch);
          });
      state.locals[macro.parameters[i]] = value;
    }
    EmitNodes(macro.body, state);
    std::vector<std::string> values{state.output}, types{"buffer"};
    NamespaceValues(NamespaceSubset(state, captures.effects), &values, &types);
    Line("func.return " + Join(values) + " : " + Join(types));
    --indent_;
    Line("}");
    environment_ = saved_environment;
    loops_ = std::move(saved_loops);
    in_macro_ = false;
  }
  Value Function(const Expr &expr, State &state) {
    if (!expr.args.empty() && expr.args[0].kind == Expr::Kind::Name &&
        macros_.count(expr.args[0].text))
      return MacroCall(expr, state);
    if (!expr.args.empty() && expr.args[0].kind == Expr::Kind::GetAttr) {
      const auto &method = expr.args[0];
      for (size_t i = 1; i < expr.keywords.size(); ++i)
        if (!expr.keywords[i].empty()) {
          Fail(expr.offset,
               "method keyword arguments are not supported: " + method.text);
          return Undefined();
        }
      const auto subject = Expression(method.args[0], state);
      std::vector<Value> args;
      for (size_t i = 1; i < expr.args.size(); ++i)
        args.push_back(Expression(expr.args[i], state));
      if (method.text == "items" && args.empty()) {
        if (items_expression_ != &expr) {
          Fail(expr.offset,
               "items views are supported only as immediate loop iterables");
          return Undefined();
        }
        return TupleResult(Pair("jinja.items",
                                {subject.data, subject.kind, Bool(false)},
                                {"buffer", "i64", "i1"}),
                           subject);
      }
      if ((method.text == "startswith" || method.text == "endswith") &&
          args.size() == 1)
        return Boolean(
            Call("jinja." + method.text,
                 {subject.data, subject.kind, args[0].data, args[0].kind},
                 {"buffer", "i64", "buffer", "i64"}, "i1"));
      if (method.text == "split" && args.size() == 1) {
        auto result =
            Pair("jinja.split",
                 {subject.data, subject.kind, args[0].data, args[0].kind},
                 {"buffer", "i64", "buffer", "i64"});
        result.known = KnownKind::List;
        return result;
      }
      if ((method.text == "strip" || method.text == "lstrip" ||
           method.text == "rstrip") &&
          args.size() <= 1) {
        const auto chars = args.empty()
                               ? Value{Literal(""), I64(6), KnownKind::None}
                               : args[0];
        auto result =
            Pair("jinja.strip",
                 {subject.data, subject.kind, chars.data, chars.kind,
                  Bool(method.text != "rstrip"), Bool(method.text != "lstrip")},
                 {"buffer", "i64", "buffer", "i64", "i1", "i1"});
        result.known = KnownKind::String;
        return result;
      }
      if (method.text == "get" && (args.size() == 1 || args.size() == 2)) {
        const auto fallback = args.size() == 2
                                  ? args[1]
                                  : Value{Literal(""), I64(6), KnownKind::None};
        auto result = Pair("jinja.mapping_get",
                           {subject.data, subject.kind, args[0].data,
                            args[0].kind, fallback.data, fallback.kind},
                           {"buffer", "i64", "buffer", "i64", "buffer", "i64"});
        result.tuple_data = subject.tuple_data || fallback.tuple_data;
        return result;
      }
      Fail(expr.offset, "unsupported method or arguments: " + method.text);
      return Undefined();
    }
    if (expr.args.empty() || expr.args[0].kind != Expr::Kind::Name) {
      Fail(expr.offset, "only declared template builtins can be called");
      return Undefined();
    }
    const auto &name = expr.args[0].text;
    for (size_t i = 1; i < expr.keywords.size(); ++i) {
      const auto &keyword = expr.keywords[i];
      if (!keyword.empty() &&
          !((name == "raise_exception" && keyword == "message") ||
            (name == "strftime_now" && keyword == "format"))) {
        Fail(expr.offset,
             "unsupported keyword argument to " + name + ": " + keyword);
        return Undefined();
      }
    }
    if (name == "raise_exception" && expr.args.size() == 2) {
      const auto diagnostic = ToString(Expression(expr.args[1], state));
      Require(Bool(false), diagnostic);
      return Undefined();
    }
    if (name == "strftime_now" && expr.args.size() == 2 &&
        expr.args[1].kind == Expr::Kind::String &&
        expr.args[1].text == "%d %b %Y") {
      const auto capability = Name(name, state);
      Require(Compare("eq", capability.kind, I64(8)),
              Literal("strftime_now is undefined"));
      const Value object{capability.data, I64(2), KnownKind::Mapping};
      const auto format = Get(object, Literal("format"));
      const auto expected = String("%d %b %Y");
      const auto valid =
          Call("jinja.equal",
               {format.data, format.kind, expected.data, expected.kind},
               {"buffer", "i64", "buffer", "i64"}, "i1");
      Require(valid, Literal("unsupported supplied date format"));
      auto result = Get(object, Literal("result"));
      Require(Compare("eq", result.kind, I64(0)),
              Literal("supplied date result must be a string"));
      result.known = KnownKind::String;
      return result;
    }
    Fail(expr.offset, "unsupported call or arguments: " + name);
    return Undefined();
  }

  Value Collection(const Expr &expr, State &state) {
    const bool mapping = expr.kind == Expr::Kind::Dict;
    bool tuple_data = expr.kind == Expr::Kind::Tuple;
    auto json = Literal(mapping ? "{" : "[");
    for (size_t i = 0; i < expr.args.size(); ++i) {
      if (mapping && i % 2 == 0 && expr.args[i].kind != Expr::Kind::String) {
        Fail(expr.args[i].offset, "mapping literals require string keys");
        return Undefined();
      }
      if (i > 0)
        json = Concat(json, Literal(mapping && i % 2 ? ": " : ", "));
      const auto value = Expression(expr.args[i], state);
      tuple_data = tuple_data || value.tuple_data;
      const auto serialized =
          Call("jinja.tojson", {value.data, value.kind, I64(-1)},
               {"buffer", "i64", "i64"}, "buffer");
      json = Concat(json, serialized);
    }
    return {Concat(json, Literal(mapping ? "}" : "]")), I64(mapping ? 2 : 3),
            mapping                          ? KnownKind::Mapping
            : expr.kind == Expr::Kind::Tuple ? KnownKind::Tuple
                                             : KnownKind::List,
            tuple_data};
  }

  static void Assigned(const std::vector<Node> &nodes,
                       std::set<std::string> *names) {
    for (const auto &node : nodes) {
      if (node.kind == Node::Kind::Set &&
          node.text.find('.') == std::string::npos && !Constructor(node))
        names->insert(node.text);
      if (node.kind == Node::Kind::If) {
        Assigned(node.body, names);
        Assigned(node.otherwise, names);
      }
    }
  }

  void EmitIf(const Node &node, State &state) {
    const auto condition = Truth(Expression(node.expression, state));
    Prebind(node.body, state);
    Prebind(node.otherwise, state);
    std::set<std::string> names;
    Assigned(node.body, &names);
    Assigned(node.otherwise, &names);
    for (auto it = names.begin(); it != names.end();)
      if (state.bindings.count(*it))
        it = names.erase(it);
      else
        ++it;
    for (const auto &name : names)
      if (!state.locals.count(name))
        state.locals[name] = Name(name, state);
    State yes = state, no = state;
    std::vector<std::string> outputs{Fresh()}, types{"buffer"};
    std::map<std::string, Value> results;
    for (const auto &name : names) {
      Value result{Fresh(), Fresh()};
      outputs.push_back(result.data);
      outputs.push_back(result.kind);
      types.push_back("buffer");
      types.push_back("i64");
      results[name] = result;
    }
    auto merged = NamespaceResults(state, &outputs, &types);
    Line(Join(outputs) + " = scf.if " + condition + " -> (" + Join(types) +
         ") {");
    ++indent_;
    EmitNodes(node.body, yes);
    Yield(yes, names, types);
    --indent_;
    Line("} else {");
    ++indent_;
    EmitNodes(node.otherwise, no);
    Yield(no, names, types);
    --indent_;
    Line("}");
    state.output = outputs[0];
    if (yes.bindings != no.bindings)
      Fail(node.offset,
           "conditional namespace aliases have ambiguous identity");
    state.bindings = yes.bindings;
    MergeNamespaces(state, std::move(merged), yes, no);
    for (auto &[name, result] : results) {
      result.known = MergeKind(yes.locals.at(name), no.locals.at(name));
      result.tuple_data =
          yes.locals.at(name).tuple_data || no.locals.at(name).tuple_data;
      state.locals[name] = result;
      if (state.module_frame)
        state.module_locals[name] = result;
    }
  }

  void Yield(const State &state, const std::set<std::string> &names,
             const std::vector<std::string> &types) {
    std::vector<std::string> values{state.output};
    for (const auto &name : names) {
      const auto &value = state.locals.at(name);
      values.push_back(value.data);
      values.push_back(value.kind);
    }
    std::vector<std::string> namespace_types;
    NamespaceValues(state, &values, &namespace_types);
    Line("scf.yield " + Join(values) + " : " + Join(types));
  }

  void EmitLoop(const Node &node, State &state) {
    const auto comma = node.text.find(',');
    const std::vector<std::string> targets =
        comma == std::string::npos
            ? std::vector<std::string>{node.text}
            : std::vector<std::string>{node.text.substr(0, comma),
                                       node.text.substr(comma + 1)};
    for (const auto &target : targets)
      if (Reserved(target) || macros_.count(target)) {
        Fail(node.offset,
             "loop assignment to reserved template builtin: " + node.text);
        return;
      }
    const auto *previous_items = items_expression_;
    const bool items_rows =
        (node.expression.kind == Expr::Kind::Filter &&
         node.expression.text == "items") ||
        (node.expression.kind == Expr::Kind::Call &&
         !node.expression.args.empty() &&
         node.expression.args[0].kind == Expr::Kind::GetAttr &&
         node.expression.args[0].text == "items");
    items_expression_ = &node.expression;
    const auto input = Expression(node.expression, state);
    items_expression_ = previous_items;
    const auto sequence = TupleResult(
        Pair("jinja.iteration", {input.data, input.kind}, {"buffer", "i64"}),
        input);
    const auto count = Call("jinja.length", {sequence.data, sequence.kind},
                            {"buffer", "i64"}, "i64");
    const auto nonnegative = Compare("sge", count, I64(0));
    const auto fits = Compare("sle", count, I64(2147483647)), valid = Fresh();
    Line(valid + " = scalar.andi " + nonnegative + ", " + fits + " : i1");
    Require(valid, Literal("template sequence exceeds native index extent"));
    const auto bounded = Fresh(), extent = Fresh();
    Line(bounded + " = scalar.assume " + count + " [range(" + count +
         ", 0, 2147483647)] : i64");
    Line(extent + " = index.cast " + bounded + " : i64 to index");
    const auto zero = Fresh(), one = Fresh();
    Line(zero + " = index.constant 0 : index");
    Line(one + " = index.constant 1 : index");
    const auto position = Fresh(), prior = Fresh(), output = Fresh();
    std::vector<std::string> outputs{output}, types{"buffer"};
    auto merged = NamespaceResults(state, &outputs, &types);
    std::vector<std::string> header{prior + " = " + state.output + " : buffer"};
    State inner = state;
    inner.module_frame = false;
    for (auto &[id, object] : inner.namespaces) {
      const auto &initial = state.namespaces.at(id);
      object.present = Fresh();
      // Constructors may run in any iteration; fields are conservatively typed
      // at the loop header rather than assuming the initial value's category.
      object.known_present = -1;
      header.push_back(object.present + " = " + initial.present + " : i1");
      for (auto &[field, value] : object.fields) {
        const auto &start = initial.fields.at(field);
        value = {Fresh(), Fresh(), KnownKind::Unknown, start.tuple_data};
        header.push_back(value.data + " = " + start.data + " : buffer");
        header.push_back(value.kind + " = " + start.kind + " : i64");
      }
    }
    Line(Join(outputs) + " = scf.for " + position + " = [" + zero + " to " +
         extent + " step " + one + "](" + Join(header) + ") -> (" +
         Join(types) + ") {");
    ++indent_;
    const auto ordinal = Fresh();
    Line(ordinal + " = index.cast " + position + " : index to i64");
    // A loop has a fresh local frame each iteration. A conditionally untaken
    // constructor must not retain the previous iteration's namespace presence.
    ResetLoopNamespaces(node.body, inner);
    inner.output = prior;
    auto row =
        TupleResult(Pair("jinja.index", {sequence.data, sequence.kind, ordinal},
                         {"buffer", "i64", "i64"}),
                    sequence);
    if (targets.size() == 1) {
      row.tuple_data = row.tuple_data || items_rows;
      inner.bindings.erase(targets[0]);
      inner.locals[targets[0]] = row;
    } else {
      const auto view =
          Pair("jinja.iteration", {row.data, row.kind}, {"buffer", "i64"});
      const auto length = Call("jinja.length", {view.data, view.kind},
                               {"buffer", "i64"}, "i64");
      Require(Compare("eq", length, I64(2)),
              Literal("loop pair unpack requires exactly two values"));
      for (size_t i = 0; i < 2; ++i) {
        inner.bindings.erase(targets[i]);
        inner.locals[targets[i]] =
            TupleResult(Pair("jinja.index", {view.data, view.kind, I64(i)},
                             {"buffer", "i64", "i64"}),
                        row);
      }
    }
    auto neighbor_sequence = sequence;
    neighbor_sequence.tuple_data = neighbor_sequence.tuple_data || items_rows;
    loops_.push_back({neighbor_sequence, ordinal, count});
    EmitNodes(node.body, inner);
    loops_.pop_back();
    std::vector<std::string> yields{inner.output}, yield_types{"buffer"};
    NamespaceValues(inner, &yields, &yield_types);
    Line("scf.yield " + Join(yields) + " : " + Join(yield_types));
    --indent_;
    Line("}");
    state.output = output;
    MergeNamespaces(state, std::move(merged), state, inner);
    if (!node.otherwise.empty()) {
      const auto empty = Compare("eq", count, I64(0));
      state.output =
          SelectValue(
              empty, state,
              [&](State &branch) {
                branch.module_frame = false;
                EmitNodes(node.otherwise, branch);
                return Value{branch.output, I64(0), KnownKind::String};
              },
              [&](State &branch) {
                return Value{branch.output, I64(0), KnownKind::String};
              })
              .data;
    }
  }

  void EmitNodes(const std::vector<Node> &nodes, State &state) {
    for (const auto &node : nodes) {
      if (failed_)
        return;
      switch (node.kind) {
      case Node::Kind::Text:
        if (!node.text.empty())
          state.output = Concat(state.output, Literal(node.text));
        break;
      case Node::Kind::Output:
        state.output =
            Concat(state.output, ToString(Expression(node.expression, state)));
        break;
      case Node::Kind::Set:
        if (node.text.find('.') != std::string::npos) {
          const auto dot = node.text.find('.');
          const auto name = node.text.substr(0, dot),
                     field = node.text.substr(dot + 1);
          if (!state.bindings.count(name)) {
            Fail(node.offset, "field assignment requires a namespace binding");
            break;
          }
          const auto value = Expression(node.expression, state);
          if (value.tuple_data) {
            Fail(
                node.offset,
                "tuple-derived namespace fields are outside the supported list domain");
            break;
          }
          auto &object = state.namespaces.at(state.bindings.at(name));
          Require(object.present,
                  Literal("field assignment requires a defined namespace"));
          object.fields[field] = value;
          break;
        }
        if (macros_.count(node.text) || node.text == "loop" ||
            node.text == "raise_exception" || node.text == "strftime_now" ||
            node.text == "namespace") {
          Fail(node.offset,
               "assignment to reserved template builtin: " + node.text);
          break;
        }
        if (Constructor(node)) {
          std::map<std::string, Value> fields;
          for (size_t i = 1; i < node.expression.args.size(); ++i) {
            const auto &keyword = node.expression.keywords[i];
            if (keyword.empty()) {
              Fail(node.offset, "namespace constructors require named fields");
              break;
            }
            fields[keyword] = Expression(node.expression.args[i], state);
            if (fields[keyword].tuple_data)
              Fail(
                  node.offset,
                  "tuple-derived namespace fields are outside the supported list domain");
          }
          if (!state.locals.count(node.text)) {
            const auto previous = state.bindings.find(node.text);
            if (previous == state.bindings.end())
              state.locals[node.text] = Name(node.text, state);
          }
          state.bindings[node.text] = node.offset;
          auto &object = state.namespaces.at(node.offset);
          object.present = Bool(true);
          object.known_present = 1;
          for (auto &[field, value] : object.fields)
            value = Undefined();
          for (const auto &[field, value] : fields)
            object.fields[field] = value;
        } else if (node.expression.kind == Expr::Kind::Name &&
                   state.bindings.count(node.expression.text)) {
          state.bindings[node.text] = state.bindings.at(node.expression.text);
          state.locals[node.text] = state.locals.at(node.expression.text);
        } else {
          const auto value = Expression(node.expression, state);
          if (state.bindings.count(node.text)) {
            Fail(node.offset,
                 "namespace and JSON rebinding has ambiguous identity");
            break;
          }
          state.locals[node.text] = value;
        }
        if (state.module_frame && state.locals.count(node.text))
          state.module_locals[node.text] = state.locals.at(node.text);
        break;
      case Node::Kind::Macro:
        state.defined_macros.insert(node.text);
        break;
      case Node::Kind::Generation: {
        State inner = state;
        inner.module_frame = false;
        EmitNodes(node.body, inner);
        state.output = inner.output;
        state.namespaces = std::move(inner.namespaces);
        break;
      }
      case Node::Kind::If:
        EmitIf(node, state);
        break;
      case Node::Kind::For:
        EmitLoop(node, state);
        break;
      }
    }
  }

  std::string_view source_;
  Diagnostic *diagnostic_ = nullptr;
  bool failed_ = false;
  size_t next_ = 0;
  unsigned indent_ = 0;
  Value environment_;
  std::vector<Loop> loops_;
  std::map<std::string, std::string> literals_;
  std::map<std::string, std::string> declarations_;
  std::map<size_t, std::string> namespace_targets_;
  std::set<std::string> namespace_fields_;
  std::map<std::string, const Node *> macros_;
  std::map<std::string, std::set<std::string>> macro_calls_;
  struct MacroCaptures {
    std::set<std::string> values;
    std::set<size_t> namespaces, effects, local_namespaces;
  };
  std::map<std::string, MacroCaptures> macro_captures_;
  std::set<std::string> checking_macros_, checked_macros_, module_names_;
  std::map<std::string, size_t> module_bindings_;
  bool in_macro_ = false;
  const Expr *items_expression_ = nullptr;
  std::ostringstream body_;
};

void Locate(std::string_view source, std::string_view name,
            Diagnostic *diagnostic) {
  if (!diagnostic)
    return;
  diagnostic->source_name = std::string(name);
  diagnostic->offset = std::min(diagnostic->offset, source.size());
  diagnostic->line = 1;
  diagnostic->column = 1;
  for (size_t i = 0; i < diagnostic->offset; ++i) {
    if (source[i] == '\n') {
      ++diagnostic->line;
      diagnostic->column = 1;
    } else {
      ++diagnostic->column;
    }
  }
}

} // namespace

bool CompileChatTemplate(std::string_view source, std::string_view source_name,
                         GeneratedInputRenderer *out, Diagnostic *diagnostic) {
  Diagnostic error;
  if (!out) {
    error.message = "chat template compilation needs an output slot";
    Locate(source, source_name, &error);
    if (diagnostic)
      *diagnostic = std::move(error);
    return false;
  }
  Template ast;
  if (!ParseTemplate(source, &ast, &error)) {
    Locate(source, source_name, &error);
    if (diagnostic)
      *diagnostic = std::move(error);
    return false;
  }
  GeneratedInputRenderer generated;
  Lowering lowering(source, &error);
  if (!lowering.Run(ast, &generated)) {
    Locate(source, source_name, &error);
    if (diagnostic)
      *diagnostic = std::move(error);
    return false;
  }
  *out = std::move(generated);
  if (diagnostic)
    *diagnostic = Diagnostic{};
  return true;
}

} // namespace hrx::loomify_gguf::jinja
