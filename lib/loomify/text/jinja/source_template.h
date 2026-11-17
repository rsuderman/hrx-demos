// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_SOURCE_TEMPLATE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_SOURCE_TEMPLATE_H_

#include <map>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "lib/loomify/text/jinja/parser.h"

namespace hrx::loomify_gguf::jinja::internal {

// Recognizes the literal JSON formatting subset shared by lowering and codec
// analysis. On failure leaves indent unchanged. Modes are default(-1),
// compact(-2), or pretty-print widths 2 and 4.
bool ToJsonIndent(const Expr &expression, int64_t *indent);

// Owns immutable original syntax and source-only analysis indexes. These
// indexes do not replace validation of each renderer/selector/recipe AST by
// Lowering. Node/expression pointers borrow this object's AST; the object never
// moves.
class SourceTemplate final {
public:
  static bool Parse(std::string_view source, std::string_view source_name,
                    std::unique_ptr<SourceTemplate> *out,
                    Diagnostic *diagnostic);
  SourceTemplate(const SourceTemplate &) = delete;
  SourceTemplate &operator=(const SourceTemplate &) = delete;
  SourceTemplate(SourceTemplate &&) = delete;
  SourceTemplate &operator=(SourceTemplate &&) = delete;
  ~SourceTemplate() = default;

  std::string_view source() const { return source_; }
  std::string_view source_name() const { return source_name_; }
  const Template &ast() const { return ast_; }
  const std::map<std::string, const Node *> &analysis_macros() const {
    return macros_;
  }
  const std::set<std::string> &source_names() const { return source_names_; }
  const std::vector<const Expr *> &analysis_macro_calls() const {
    return macro_calls_;
  }

private:
  SourceTemplate() = default;
  void GatherMacros(const std::vector<Node> &nodes);
  void GatherExpressionNames(const Expr &expression);
  void GatherSourceNames(const std::vector<Node> &nodes);
  void GatherMacroCallSites(const std::vector<Node> &nodes);

  std::string source_;
  std::string source_name_;
  Template ast_;
  std::map<std::string, const Node *> macros_;
  std::set<std::string> source_names_;
  std::vector<const Expr *> macro_calls_;
};

} // namespace hrx::loomify_gguf::jinja::internal
#endif // HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_SOURCE_TEMPLATE_H_
