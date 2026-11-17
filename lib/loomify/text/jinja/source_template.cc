// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/jinja/source_template.h"

#include <utility>

namespace hrx::loomify_gguf::jinja::internal {

bool SourceTemplate::Parse(std::string_view source,
                           std::string_view source_name,
                           std::unique_ptr<SourceTemplate> *out,
                           Diagnostic *diagnostic) {
  Diagnostic error;
  if (!out) {
    error.message = "source preparation needs an output slot";
  } else {
    std::unique_ptr<SourceTemplate> parsed(new SourceTemplate);
    parsed->source_ = std::string(source);
    parsed->source_name_ = std::string(source_name);
    if (ParseTemplate(parsed->source_, &parsed->ast_, &error)) {
      parsed->GatherMacros(parsed->ast_.body);
      parsed->GatherMacroCallSites(parsed->ast_.body);
      parsed->GatherSourceNames(parsed->ast_.body);
      *out = std::move(parsed);
      if (diagnostic)
        *diagnostic = {};
      return true;
    }
  }
  LocateDiagnostic(source, source_name, &error);
  if (diagnostic)
    *diagnostic = std::move(error);
  return false;
}

void SourceTemplate::GatherMacros(const std::vector<Node> &nodes) {
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Macro)
      macros_[node.text] = &node;
    GatherMacros(node.body);
    GatherMacros(node.otherwise);
  }
}

void SourceTemplate::GatherExpressionNames(const Expr &expression) {
  if (expression.kind == Expr::Kind::Name)
    source_names_.insert(expression.text);
  for (const auto &child : expression.args)
    GatherExpressionNames(child);
}

void SourceTemplate::GatherSourceNames(const std::vector<Node> &nodes) {
  for (const auto &node : nodes) {
    source_names_.insert(node.text);
    for (const auto &name : node.parameters)
      source_names_.insert(name);
    GatherExpressionNames(node.expression);
    for (const auto &value : node.defaults)
      GatherExpressionNames(value);
    GatherSourceNames(node.body);
    GatherSourceNames(node.otherwise);
  }
}

void SourceTemplate::GatherMacroCallSites(const std::vector<Node> &nodes) {
  auto expression = [&](auto &&self, const Expr &value) -> void {
    if (value.kind == Expr::Kind::Call && !value.args.empty() &&
        value.args[0].kind == Expr::Kind::Name &&
        macros_.count(value.args[0].text))
      macro_calls_.push_back(&value);
    for (const auto &argument : value.args)
      self(self, argument);
  };
  for (const auto &node : nodes) {
    if (node.kind == Node::Kind::Macro)
      continue;
    expression(expression, node.expression);
    GatherMacroCallSites(node.body);
    GatherMacroCallSites(node.otherwise);
  }
}

} // namespace hrx::loomify_gguf::jinja::internal
