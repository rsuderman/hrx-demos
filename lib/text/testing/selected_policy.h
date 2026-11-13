// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_TEXT_TESTING_SELECTED_POLICY_H_
#define HRX_SOLUTIONS_LIB_TEXT_TESTING_SELECTED_POLICY_H_
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/linking.h"
#include "lib/serving/library_resources.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace hrx::loomify_gguf::testing {
inline std::filesystem::path TextRunfiles() {
  return std::filesystem::path(
             std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") /
         (std::getenv("TEST_WORKSPACE") ? std::getenv("TEST_WORKSPACE")
                                        : "_main");
}
// Component tests use the production selector and configuration/link path.
// Expected parser/transcript values remain authored independently by the tests.
inline bool LinkSelectedText(loomc_context_t *context,
                             loomc_workspace_t *workspace,
                             const char *template_path,
                             std::vector<const char *> roots,
                             loomc_module_t **out, std::string *error) {
  const auto runfiles = TextRunfiles();
  const auto path = runfiles / template_path;
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot read original template: " + path.string();
    return false;
  }
  model::TextMetadata metadata;
  metadata.templates.push_back(
      {"tokenizer.chat_template",
       {},
       false,
       std::string(std::istreambuf_iterator<char>(file), {})});
  ChatPolicySelection policy;
  if (!SelectChatPolicy(metadata, {}, &policy, error))
    return false;
  roots.insert(roots.end(), policy.data_roots.begin(), policy.data_roots.end());
  return LinkConfiguredProducts(context, workspace, policy.configuration_source,
                                {(runfiles / policy.text_product).string(),
                                 (runfiles / resources::kTextCommon).string()},
                                roots, false, out, error);
}
} // namespace hrx::loomify_gguf::testing
#endif
