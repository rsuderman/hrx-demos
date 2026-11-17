// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_TEXT_TESTING_SELECTED_POLICY_H_
#define HRX_SOLUTIONS_LIB_TEXT_TESTING_SELECTED_POLICY_H_
#include "lib/loomify/linking.h"
#include "lib/loomify/template_analysis.h"
#include "lib/loomify/testing/template_qualification.h"
#include "lib/loomify/text/jinja/output_emitter.h"
#include "lib/loomify/text/jinja/request_emitter.h"
#include "lib/loomify/token_policy.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::testing {
inline std::filesystem::path TextRunfiles() {
  return std::filesystem::path(
             std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".") /
         (std::getenv("TEST_WORKSPACE") ? std::getenv("TEST_WORKSPACE")
                                        : "_main");
}
// Component tests use source compilation and generated control directly.
// Expected parser/transcript values remain authored independently by the tests.
inline bool BuildTextComponents(std::string_view source,
                                std::string_view source_name,
                                const jinja::TextFormatBindings &globals,
                                std::string_view terminal,
                                jinja::GeneratedInputRenderer *out,
                                std::string *error) {
  const auto runfiles = TextRunfiles();
  if (!out) {
    *error = "source component needs an output";
    return false;
  }
  const char *scratch = std::getenv("TEST_TMPDIR");
  if (!scratch) {
    *error = "source component requires TEST_TMPDIR";
    return false;
  }
  jinja::Diagnostic diagnostic;
  jinja::GeneratedInputRenderer renderer, adapters, output, turn, combined;
  jinja::AnalyzedTemplate analyzed;
  jinja::TemplateQualification qualification;
  const auto failure = [&]() {
    *error = diagnostic.source_name + ":" + std::to_string(diagnostic.line) +
             ":" + diagnostic.message;
    return false;
  };
  if (!jinja::CompileChatTemplate(source, source_name, &renderer, &diagnostic))
    return failure();
  auto protected_names = renderer.protected_bindings;
  for (const auto &[name, unused] : globals.strings)
    protected_names.push_back(name);
  if (!jinja::AnalyzeTemplate(source, source_name, renderer, globals,
                              {runfiles.string(), scratch, false}, &analyzed,
                              &diagnostic) ||
      !jinja::QualifyTemplate(source, source_name, globals, analyzed,
                              {runfiles.string(), scratch, false},
                              &qualification, &diagnostic) ||
      !jinja::EmitRequestAdapters(source_name, analyzed.format,
                                  analyzed.adaptation_selector, globals,
                                  protected_names, &adapters, &diagnostic) ||
      !jinja::EmitOutputParser(source_name, analyzed.format, &output,
                               &diagnostic) ||
      !jinja::EmitTextTurnControls(source, source_name, analyzed.format,
                                   globals, &turn, &diagnostic) ||
      !jinja::CombineGeneratedTemplates(
          {renderer, analyzed.selector, adapters, output, turn},
          renderer.entrypoint, &combined, &diagnostic)) {
    return failure();
  }
  combined.source += EmitTerminalMarkerSource(terminal);
  combined.helper_requirements.push_back("lib/text/generated_input/host.loom");
  *out = std::move(combined);
  return true;
}

inline bool BuildSelectedText(const char *template_path,
                              const jinja::TextFormatBindings &globals,
                              std::string_view terminal,
                              jinja::GeneratedInputRenderer *out,
                              std::string *error) {
  const auto path = TextRunfiles() / template_path;
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    *error = "cannot read original template: " + path.string();
    return false;
  }
  const std::string source(std::istreambuf_iterator<char>(file), {});
  return BuildTextComponents(source, template_path, globals, terminal, out,
                             error);
}

inline bool LinkSelectedText(loomc_context_t *context,
                             loomc_workspace_t *workspace,
                             const char *template_path,
                             const jinja::TextFormatBindings &globals,
                             std::string_view terminal,
                             std::vector<const char *> roots,
                             loomc_module_t **out, std::string *error) {
  jinja::GeneratedInputRenderer generated;
  if (!BuildSelectedText(template_path, globals, terminal, &generated, error))
    return false;
  const auto runfiles = TextRunfiles();
  std::vector<std::string> products;
  for (const auto &product : generated.helper_requirements)
    products.push_back((runfiles / product).string());
  return LinkConfiguredProducts(context, workspace, generated.source, products,
                                roots, false, out, error);
}
} // namespace hrx::loomify_gguf::testing
#endif
