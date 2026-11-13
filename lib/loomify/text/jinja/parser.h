// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_PARSER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_PARSER_H_

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "lib/loomify/text/jinja/compiler.h"

namespace hrx::loomify_gguf::jinja {

// A transient source AST. Strings are decoded values; other text fields contain
// names/operators or integer lexemes. Every expression owns its children.
struct Expr {
  enum class Kind {
    Name,
    String,
    Integer,
    Boolean,
    None,
    Unary,
    Binary,
    Conditional,
    GetItem,
    GetAttr,
    Slice,
    Call,
    Filter,
    Test,
    List,
    Tuple,
    Dict,
  };
  Kind kind = Kind::None;
  std::string text;
  size_t offset = 0;
  size_t end = 0;
  std::vector<Expr> args;
  // Empty for positional children, otherwise the corresponding keyword name.
  // Call: args[0] is callee. Filter/Test: text is name, args[0] subject.
  // Slice: args = [subject, start, end, step], None denotes an omitted bound.
  // Conditional: args = [condition, yes, no]. Dict alternates key/value
  // children.
  std::vector<std::string> keywords;
};

struct Node {
  enum class Kind { Text, Output, Set, If, For, Macro, Generation };
  Kind kind = Kind::Text;
  // Literal text, assignment target or loop variable, depending on kind.
  std::string text;
  size_t offset = 0;
  size_t end = 0;
  Expr expression;
  std::vector<std::string> parameters;
  // Trailing parameter defaults, evaluated afresh at each invocation.
  std::vector<Expr> defaults;
  std::vector<Node> body;
  // elif is represented by a nested If in otherwise.
  std::vector<Node> otherwise;
};

struct Template {
  std::vector<Node> body;
};

// Parses a complete template with HF chat whitespace settings. Failure leaves
// output unchanged and records a source byte offset/message in diagnostic.
bool ParseTemplate(std::string_view source, Template *out,
                   Diagnostic *diagnostic);

} // namespace hrx::loomify_gguf::jinja
#endif // HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_PARSER_H_
