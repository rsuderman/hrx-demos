// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/chat_policy.h"
#include "lib/text/testing/selected_policy.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
namespace hrx::loomify_gguf {
namespace {
model::ChatTemplateSource Candidate(std::string source,
                                    std::optional<std::string> name = {}) {
  return {name ? "tokenizer.chat_template." + *name : "tokenizer.chat_template",
          name, name.has_value(), std::move(source)};
}
TEST(ChatTemplateSource, SelectsOwnedActualTextWithoutRendererRecognition) {
  model::TextMetadata metadata;
  metadata.templates = {Candidate("arbitrary selected source"),
                        Candidate("named source", "chat")};
  ChatTemplateSelection selected;
  std::string error = "stale";
  ASSERT_TRUE(SelectChatTemplateSource(metadata, {}, &selected, &error))
      << error;
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(selected.source_key, "tokenizer.chat_template");
  EXPECT_EQ(selected.source_text, "arbitrary selected source");
  EXPECT_FALSE(selected.template_name);
  ASSERT_TRUE(SelectChatTemplateSource(metadata, "chat", &selected, &error))
      << error;
  EXPECT_EQ(selected.template_name, "chat");
  EXPECT_EQ(selected.source_text, "named source");
  metadata.templates.clear();
  EXPECT_EQ(selected.source_text, "named source");
}
TEST(ChatTemplateSource, PreservesSelectionOnMetadataAndNameFailures) {
  ChatTemplateSelection selected{"sentinel-key", "sentinel-name",
                                 "sentinel-source"};
  const auto check = [&](const model::TextMetadata &metadata,
                         std::optional<std::string> name) {
    std::string error;
    EXPECT_FALSE(SelectChatTemplateSource(metadata, name, &selected, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(selected.source_key, "sentinel-key");
    EXPECT_EQ(selected.template_name, "sentinel-name");
    EXPECT_EQ(selected.source_text, "sentinel-source");
  };
  model::TextMetadata metadata;
  check(metadata, {});
  metadata.templates = {Candidate("a", "a"), Candidate("b", "b")};
  check(metadata, {});
  check(metadata, "missing");
  check(metadata, "");
  metadata.templates = {Candidate("a"), Candidate("b")};
  check(metadata, {});
  metadata.templates = {Candidate("a", "a"), Candidate("b", "a")};
  check(metadata, "a");
  metadata.templates = {Candidate("a", "a")};
  metadata.templates.front().listed = false;
  check(metadata, "a");
  metadata.template_issues.push_back(
      {"tokenizer.chat_template", "malformed source metadata"});
  check(metadata, {});
}
struct SourceFixture {
  const char *name;
  const char *path;
  const char *terminal;
};
class GeneratedTextSource : public ::testing::TestWithParam<SourceFixture> {};
TEST_P(GeneratedTextSource, BuildsOwnedGeneratedControlsFromActualSource) {
  const auto &fixture = GetParam();
  jinja::TextFormatBindings globals;
  globals.strings = {{"bos_token", "SOURCE_BOS_FIXTURE"},
                     {"eos_token", "SOURCE_EOS_FIXTURE"}};
  jinja::GeneratedInputRenderer generated;
  std::string error;
  ASSERT_TRUE(testing::BuildSelectedText(fixture.path, globals,
                                         fixture.terminal, &generated, &error))
      << error;
  EXPECT_FALSE(generated.source.empty());
  EXPECT_EQ(generated.entrypoint, "jinja.evaluate");
  for (const auto *symbol :
       {"@jinja.evaluate", "@output_next", "@parse_tools", "@complete_text",
        "@render_tool", "@loomify.tokens.terminal_marker"})
    EXPECT_NE(generated.source.find(symbol), std::string::npos) << symbol;
  EXPECT_EQ(std::set<std::string>(generated.helper_requirements.begin(),
                                  generated.helper_requirements.end())
                .size(),
            generated.helper_requirements.size());
  EXPECT_NE(std::find(generated.helper_requirements.begin(),
                      generated.helper_requirements.end(),
                      "lib/text/generated_input/host.loom"),
            generated.helper_requirements.end());
  for (const auto &helper : generated.helper_requirements) {
    EXPECT_EQ(helper.find("/protocols/"), std::string::npos) << helper;
    EXPECT_EQ(helper.find("generated.loom"), std::string::npos) << helper;
  }
}
INSTANTIATE_TEST_SUITE_P(
    ActualSources, GeneratedTextSource,
    ::testing::Values(
        SourceFixture{"Llama", "lib/text/llama/templates/5816fce1.jinja",
                      "<|eot_id|>"},
        SourceFixture{"Qwen08", "lib/text/qwen35/templates/7f0e5290.jinja",
                      "<|im_end|>"},
        SourceFixture{"Qwen27", "lib/text/qwen35/templates/c3cf9e34.jinja",
                      "<|im_end|>"},
        SourceFixture{"Plain", "lib/text/plain_chatml/templates/58c1a1f0.jinja",
                      "<|im_end|>"},
        SourceFixture{"Qwen25", "lib/text/qwen25/templates/cd8e9439.jinja",
                      "<|im_end|>"},
        SourceFixture{"Qwen3", "lib/text/qwen3/templates/5da44855.jinja",
                      "<|im_end|>"},
        SourceFixture{"Gemma", "lib/text/gemma/templates/7de1c58e.jinja",
                      "<end_of_turn>"},
        SourceFixture{"Compact",
                      "lib/text/compact_chatml/templates/f45005c5.jinja",
                      "<|im_end|>"},
        SourceFixture{"Spark", "lib/text/spark/templates/93d6aa90.jinja",
                      "<｜end▁of▁sentence｜>"}),
    [](const ::testing::TestParamInfo<SourceFixture> &info) {
      return info.param.name;
    });
TEST(GeneratedTextComponents, UncataloguedSourceOwnsDeterministicProducts) {
  std::string source =
      "{{ bos_token }}{% for m in messages %}"
      "{{ '<uncatalogued>' + m.role + ':' + m.content + '<END>' }}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<uncatalogued>assistant:' }}{% endif %}";
  jinja::TextFormatBindings globals;
  globals.strings["bos_token"] = "EXPLICIT_GLOBAL";
  jinja::GeneratedInputRenderer first, second;
  std::string error;
  ASSERT_TRUE(testing::BuildTextComponents(source, "uncatalogued.jinja",
                                           globals, "<END>", &first, &error))
      << error;
  ASSERT_TRUE(testing::BuildTextComponents(source, "uncatalogued.jinja",
                                           globals, "<END>", &second, &error))
      << error;
  EXPECT_EQ(first.source, second.source);
  EXPECT_EQ(first.helper_requirements, second.helper_requirements);
  source.clear();
  globals.strings.clear();
  EXPECT_EQ(first.source, second.source);
  EXPECT_FALSE(first.source.empty());
}
TEST(GeneratedTextComponents, UnsupportedSourcePreservesOwnedOutput) {
  jinja::GeneratedInputRenderer generated;
  generated.source = "previous source";
  generated.helper_requirements = {"previous helper"};
  std::string error;
  EXPECT_FALSE(testing::BuildTextComponents("{% include 'external' %}",
                                            "unsupported.jinja", {}, "<END>",
                                            &generated, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(generated.source, "previous source");
  EXPECT_EQ(generated.helper_requirements,
            std::vector<std::string>({"previous helper"}));
  EXPECT_FALSE(testing::BuildTextComponents("unused", "unsupported.jinja", {},
                                            "<END>", nullptr, &error));
  EXPECT_EQ(generated.source, "previous source");
}
} // namespace
} // namespace hrx::loomify_gguf
