// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/text/matching.h"

namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
const std::vector<const char *> kRoots = {"step",
                                          "epoch",
                                          "encode_epoch",
                                          "publish_epoch",
                                          "encode_state",
                                          "resize_state",
                                          "render_tool",
                                          "prepare_input",
                                          "output_next",
                                          "complete_text",
                                          "model_name",
                                          "chat_reasoning",
                                          "chat_image_limits",
                                          "chat_begin",
                                          "chat_message",
                                          "chat_end",
                                          "parse_tools",
                                          "request_defaults",
                                          "request_options",
                                          "bind_request"};
} // namespace
bool SelectChatPolicy(const model::TextMetadata &metadata,
                      const std::optional<std::string> &template_name,
                      ChatPolicySelection *out, std::string *error) {
  if (!out)
    return Fail(error, "chat policy needs an output slot");
  if (template_name && template_name->empty())
    return Fail(error, "chat template name must not be empty");
  if (!metadata.template_issues.empty()) {
    const auto &issue = metadata.template_issues.front();
    return Fail(error, "invalid chat template metadata at " + issue.source_key +
                           ": " + issue.message);
  }
  const model::ChatTemplateSource *selected = nullptr;
  if (template_name) {
    for (const auto &candidate : metadata.templates) {
      if (!candidate.name || *candidate.name != *template_name)
        continue;
      if (!candidate.listed)
        return Fail(error, "chat template is not listed: " + *template_name);
      if (selected)
        return Fail(error,
                    "duplicate selected chat template: " + *template_name);
      selected = &candidate;
    }
    if (!selected)
      return Fail(error, "chat template name is absent: " + *template_name);
  } else {
    for (const auto &candidate : metadata.templates)
      if (!candidate.name) {
        if (selected)
          return Fail(error, "multiple scalar chat template defaults");
        selected = &candidate;
      }
    if (!selected) {
      for (const auto &candidate : metadata.templates)
        if (candidate.listed) {
          if (selected)
            return Fail(error,
                        "named chat templates require an explicit selection");
          selected = &candidate;
        }
    }
    if (!selected)
      return Fail(error, "GGUF has no selectable chat template");
  }
  for (auto select :
       {text_policy::SelectLlamaHeaders, text_policy::SelectChatmlThinking,
        text_policy::SelectGemmaTurns}) {
    ChatPolicySelection result;
    const auto match = select(selected->text, &result);
    if (match == text_policy::Match::InvalidBuiltin)
      return Fail(error, "built-in chat template identity data is missing");
    if (match == text_policy::Match::NoMatch)
      continue;
    result.source_key = selected->source_key;
    result.template_name = selected->name;
    result.source_text = selected->text;
    result.roots = kRoots;
    *out = std::move(result);
    if (error)
      error->clear();
    return true;
  }
  return Fail(error, "unsupported exact chat template revision at " +
                         selected->source_key);
}
} // namespace hrx::loomify_gguf
