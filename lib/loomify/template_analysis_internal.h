// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_ANALYSIS_INTERNAL_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEMPLATE_ANALYSIS_INTERNAL_H_

#include "lib/loomify/template_analysis.h"
#include "lib/loomify/template_probe_runner.h"
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace hrx::loomify_gguf::jinja::internal {
using Object = std::map<std::string, std::string>;
inline constexpr char kMissing[] = "\001undefined";
std::string Quote(std::string_view text);
std::string Json(const Object &object);
std::string Array(const std::vector<std::string> &values);
bool Fail(std::string_view source, std::string_view source_name, size_t offset,
          const std::string &message, Diagnostic *diagnostic);
size_t Offset(const TextFormatAlternative &format);

struct ProbeCall {
  std::string name;
  std::string id;
  Object arguments;
  bool mapped = false;
  std::string Serialize(bool wire = false) const;
};
struct ProbeMessage {
  std::string role;
  std::string content;
  std::string reasoning;
  std::string call_id;
  std::vector<ProbeCall> calls;
  std::string Serialize(bool wire = false) const;
};
std::string Messages(const std::vector<ProbeMessage> &messages,
                     bool wire = false);
ProbeMessage Message(std::string role, std::string_view content);
bool Normalize(const Object &base, const std::vector<ProbeMessage> &original,
               const TextFormatAnalysis &analysis,
               PreparedTemplate &adaptation_selector,
               std::vector<ProbeMessage> *normalized, std::string *error);

enum class InputVisit { Continue, Stop, Fail };
// Borrowed only for this synchronous callback; no environment/history escapes.
struct TemplateInput {
  const Object &environment;
  const std::vector<ProbeMessage> &history;
  size_t candidate_index;
  bool route_candidate;
  int history_kind;
  int tools_kind;
};
bool VisitTemplateInputs(
    const Template &original, const GeneratedInputRenderer &input,
    const TextFormatAnalysis &analysis, const TextFormatBindings &bindings,
    const std::function<InputVisit(const TemplateInput &)> &visit);

// Shared existing pattern lowering. Only the test-only helper defines/calls
// Build; production constructs actual USER fragments through BuildContinuation.
class RecipeBuilder {
public:
  RecipeBuilder(const TextFormatAnalysis &analysis, std::string_view source);
  bool Build(const TextFormatAlternative &format, Template *out,
             std::string *error);
  bool BuildContinuation(const TextUserContinuation &continuation,
                         Template *out, std::string *error);
  bool BuildTurn(const std::vector<TextFormatPart> &parts,
                 const std::vector<TextFormatPart> &first_call_prefix,
                 Template *out, std::string *error);
  bool BuildCall(const std::vector<TextFormatPart> &parts, Template *out,
                 std::string *error);
  const std::string &assistant_binding() const { return assistant_; }
  const std::string &formatted_tools_binding() const {
    return formatted_tools_;
  }
  const std::string &ordinal_binding() const { return ordinal_; }

private:
  struct Context {
    std::string call;
    std::string key;
    std::string value;
    bool loop = false;
  };
  std::string Fresh(const char *suffix);
  bool Semantic(TextCapture capture, const Context &context, Expr *out);
  void Bind(const Context &context, std::vector<Node> *out);
  bool Parts(const std::vector<TextFormatPart> &parts, const Context &context,
             std::vector<Node> *out, std::string *error);
  const std::map<std::string, TextCapture> &Bindings() const;
  const TextFormatAnalysis &analysis_;
  const TextUserContinuation *continuation_ = nullptr;
  bool turn_ = false;
  bool native_call_ = false;
  const std::vector<TextFormatPart> *first_call_prefix_ = nullptr;
  std::string_view source_;
  std::string assistant_;
  std::string formatted_tools_;
  std::string ordinal_;
  std::set<std::string> names_;
};
} // namespace hrx::loomify_gguf::jinja::internal
#endif
