// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/chat_policy.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
namespace hrx::loomify_gguf {
namespace {
std::string Source(const std::string& file) {
  const auto path = std::filesystem::path(std::getenv("TEST_SRCDIR")) /
      std::getenv("TEST_WORKSPACE") / file;
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input) << path;
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
model::ChatTemplateSource Candidate(std::string source, std::optional<std::string> name = {}) {
  return {name ? "tokenizer.chat_template." + *name : "tokenizer.chat_template", name, name.has_value(), std::move(source)};
}
TEST(ChatPolicy, ExactThreeIdentitiesAndDefaults) {
  const char* paths[] = {"lib/text/llama/templates/5816fce1.jinja",
    "lib/text/qwen35/templates/7f0e5290.jinja", "lib/text/qwen35/templates/c3cf9e34.jinja"};
  for (int i=0;i<3;++i) {
    model::TextMetadata metadata;
    metadata.templates.push_back(Candidate(Source(paths[i])));
    // Selection is independent of token/vocabulary issues and numerical shape.
    metadata.token_issues.push_back({"some.token", "unused by template selection"});
    ChatPolicySelection policy; std::string error="stale";
    ASSERT_TRUE(SelectChatPolicy(metadata, {}, &policy, &error)) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(policy.tokens.pre_tokenizer, i==0 ? "llama-bpe" : "qwen35");
    EXPECT_EQ(policy.source_key, "tokenizer.chat_template");
    EXPECT_FALSE(policy.template_name);
    const char* identities[] = {
      "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4",
      "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67",
      "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"};
    EXPECT_EQ(policy.skeleton_sha256,identities[i]);
    EXPECT_EQ(policy.preserved_baseline,i==0);
    EXPECT_EQ(policy.source_text, metadata.templates.front().text);
    EXPECT_FALSE(policy.configuration_source.empty());
    EXPECT_EQ(policy.data_roots.size(), i==0 ? 2u : 6u);
    ASSERT_EQ(policy.roots.size(),20u);
    std::set<std::string> roots(policy.roots.begin(),policy.roots.end());
    EXPECT_EQ(roots.size(),20u); EXPECT_EQ(roots.count("render_tool"),1u);
    EXPECT_EQ(roots.count("chat_reasoning"),1u);
    EXPECT_EQ(roots.count("resize_state"),1u);
    EXPECT_EQ(roots.count("prepare_input"),1u); EXPECT_EQ(roots.count("output_next"),1u);
    EXPECT_FALSE(policy.markers.empty());
    for (const auto& marker : policy.markers) EXPECT_FALSE(marker.text.empty());
    EXPECT_NE(policy.text_product.find(i==0 ? "llama_headers/implementation" : i==1 ? "chatml_thinking/inline_history" : "chatml_thinking/explicit_reasoning"), std::string::npos);
  }
}
std::string Replace(std::string source, const std::string& before, const std::string& after) {
  const auto offset = source.find(before);
  EXPECT_NE(offset, std::string::npos);
  if (offset != std::string::npos) source.replace(offset, before.size(), after);
  return source;
}
std::string Hex(std::string_view value) {
  const char* digits="0123456789abcdef"; std::string out;
  for (unsigned char c : value) { out += digits[c>>4]; out += digits[c&15]; }
  return out;
}
TEST(ChatPolicy, CapturesSupportedSourceDataAndKeepsActualProvenance) {
  const char* paths[] = {"lib/text/llama/templates/5816fce1.jinja",
    "lib/text/qwen35/templates/7f0e5290.jinja", "lib/text/qwen35/templates/c3cf9e34.jinja"};
  for (int i=0;i<3;++i) {
    auto source=Source(paths[i]);
    if (i==0) source=Replace(source,"26 Jul 2024","10 Nov 2026");
    else source=Replace(source,"You have access to the following functions:",u8"Available functions café:");
    if (i==2) source=Replace(source,"reasoning_effort|default('xhigh')","reasoning_effort|default('low')");
    model::TextMetadata metadata; metadata.templates={Candidate(source)};
    ChatPolicySelection policy; std::string error;
    ASSERT_TRUE(SelectChatPolicy(metadata,{},&policy,&error)) << error;
    EXPECT_EQ(policy.source_text,source);
    EXPECT_NE(policy.configuration_source.find(Hex(i==0 ? "10 Nov 2026" : u8"Available functions café:")),std::string::npos);
    if (i==2) {
      EXPECT_NE(policy.configuration_source.find("@loomify.text.default_effort = bytes(\"6c6f77\")"),std::string::npos);
      EXPECT_NE(policy.configuration_source.find("@loomify.text.default_effort_instruction = bytes(\""+Hex("Reasoning effort is set to low.")),std::string::npos);
    }
  }
}
TEST(ChatPolicy, Qwen3HermesUsesSourceProseAndDistinctTokenRequirements) {
  const auto source = Source("lib/text/qwen3/templates/5da44855.jinja");
  model::TextMetadata metadata;
  metadata.templates = {Candidate(source)};
  ChatPolicySelection policy;
  std::string error;
  ASSERT_TRUE(SelectChatPolicy(metadata, {}, &policy, &error)) << error;
  EXPECT_EQ(policy.revision, ChatRevision::ChatmlHermes);
  EXPECT_EQ(policy.tokens.tokenizer_model, "gpt2");
  EXPECT_EQ(policy.tokens.pre_tokenizer, "qwen2");
  EXPECT_FALSE(policy.tokens.require_bos);
  EXPECT_EQ(policy.tokens.terminal_piece, "<|im_end|>");
  EXPECT_EQ(policy.skeleton_sha256, "5da44855ab7e0641774d5ed0ff0b4f89671d01df7d316e54d06557a962211cb4");
  EXPECT_NE(policy.text_product.find("chatml_thinking/hermes"), std::string::npos);
  for (const auto& marker : policy.markers) EXPECT_NE(marker.text, "<function=");
  const auto original = policy.configuration_source;
  metadata.templates = {Candidate(Replace(source, "You may call one or more functions", "Call the supplied functions"))};
  ASSERT_TRUE(SelectChatPolicy(metadata, {}, &policy, &error)) << error;
  EXPECT_NE(policy.configuration_source, original);
  EXPECT_NE(policy.configuration_source.find(Hex("Call the supplied functions")), std::string::npos);
  // The template's actual semantics, not marker spellings, identify its policy.
  metadata.templates = {Candidate(Replace(source, "enable_thinking is false", "enable_thinking is true"))};
  EXPECT_FALSE(SelectChatPolicy(metadata, {}, &policy, &error));
  metadata.templates = {Candidate(Replace(source, "<|im_end|>\\n\" }}", "<|im_end|>tail\\n\" }}"))};
  EXPECT_FALSE(SelectChatPolicy(metadata, {}, &policy, &error));
}

TEST(ChatPolicy, RejectsNonSlotStructureMarkersEscapesAndOversizeData) {
  const auto source=Source("lib/text/qwen35/templates/c3cf9e34.jinja");
  const std::vector<std::pair<std::string,std::string>> changes={
    {"if tools and tools is iterable","if tools or tools is iterable"},
    {"tool | tojson","tool | string"},
    {"<|im_start|>system","<|im_start|>other"},
    {"<tool_call>","<tool_new>"},
    {"# Tools", "# Tools\\u000a"},
    {"reasoning_effort|default('xhigh')","reasoning_effort|default('ultra')"},
    {"# Tools",std::string(65537,'x')},
    {"# Tools",std::string("bad ")+char(0xc0)+char(0xaf)},
    {"# Tools", "# Tools>"},
  };
  for (const auto& change:changes) {
    SCOPED_TRACE(change.first);
    model::TextMetadata metadata; metadata.templates={Candidate(Replace(source,change.first,change.second))};
    ChatPolicySelection policy; policy.source_text="prior"; std::string error;
    EXPECT_FALSE(SelectChatPolicy(metadata,{},&policy,&error));
    EXPECT_EQ(policy.source_text,"prior"); EXPECT_FALSE(error.empty());
  }
}
TEST(ChatPolicy, GemmaTurnsHasIndependentSourceTokenRequirements) {
  model::TextMetadata metadata;
  metadata.templates={Candidate(Source("lib/text/gemma/templates/7de1c58e.jinja"))};
  ChatPolicySelection selected; std::string error;
  ASSERT_TRUE(SelectChatPolicy(metadata,{},&selected,&error)) << error;
  EXPECT_EQ(selected.revision,ChatRevision::GemmaTurns);
  EXPECT_EQ(selected.tokens.tokenizer_model,"llama");
  EXPECT_EQ(selected.tokens.pre_tokenizer,"default");
  EXPECT_TRUE(selected.tokens.require_bos); EXPECT_TRUE(selected.tokens.bos_piece.empty());
  EXPECT_FALSE(selected.tokens.terminal_is_eos);
  EXPECT_EQ(selected.tokens.terminal_piece,"<end_of_turn>");
  EXPECT_TRUE(selected.configuration_source.empty());
  EXPECT_NE(selected.text_product.find("gemma_turns/implementation"),std::string::npos);
  metadata.templates[0].text+="\n";
  EXPECT_FALSE(SelectChatPolicy(metadata,{},&selected,&error));
}
TEST(ChatPolicy, NamesDoNotImplyRevisionOrPriority) {
  const auto a=Source("lib/text/qwen35/templates/7f0e5290.jinja");
  const auto b=Source("lib/text/qwen35/templates/c3cf9e34.jinja");
  model::TextMetadata metadata;
  metadata.templates={Candidate(a,"other"),Candidate(b,"default")};
  metadata.named_template_order={"default","other"};
  ChatPolicySelection selected; std::string error;
  EXPECT_FALSE(SelectChatPolicy(metadata,{},&selected,&error));
  ASSERT_TRUE(SelectChatPolicy(metadata,std::string("default"),&selected,&error));
  EXPECT_EQ(selected.revision,ChatRevision::ChatmlExplicitReasoning);
  EXPECT_EQ(selected.template_name,std::optional<std::string>("default"));
  metadata.templates.push_back(Candidate(a));
  ASSERT_TRUE(SelectChatPolicy(metadata,{},&selected,&error));
  EXPECT_EQ(selected.revision,ChatRevision::ChatmlInlineHistory);
  EXPECT_FALSE(selected.template_name);
  ASSERT_TRUE(SelectChatPolicy(metadata,std::string("default"),&selected,&error));
  EXPECT_EQ(selected.revision,ChatRevision::ChatmlExplicitReasoning);
  metadata.templates={Candidate(a,"only")};
  ASSERT_TRUE(SelectChatPolicy(metadata,{},&selected,&error));
  EXPECT_EQ(selected.template_name,std::optional<std::string>("only"));
}
TEST(ChatPolicy, RejectsUnknownAndMalformedWithoutReplacingPriorResult) {
  const auto source=Source("lib/text/qwen35/templates/7f0e5290.jinja");
  model::TextMetadata metadata; metadata.templates={Candidate(source)};
  ChatPolicySelection prior; std::string error;
  ASSERT_TRUE(SelectChatPolicy(metadata,{},&prior,&error));
  const auto identity=prior.skeleton_sha256;
  auto reject=[&](model::TextMetadata bad,std::optional<std::string> name={}) {
    EXPECT_FALSE(SelectChatPolicy(bad,name,&prior,&error));
    EXPECT_FALSE(error.empty()); EXPECT_EQ(prior.skeleton_sha256,identity);
  };
  auto changed=metadata; changed.templates[0].text[0]='!';reject(changed);
  changed=metadata;changed.templates[0].text+='\n';reject(changed);
  changed=metadata;changed.templates[0].text.clear();reject(changed);
  reject({});reject(metadata,std::string("default"));reject(metadata,std::string(""));
  changed=metadata;changed.templates.push_back(Candidate(source));reject(changed);
  changed=metadata;changed.template_issues.push_back({"tokenizer.chat_templates","duplicate name"});reject(changed);
  changed.templates.clear();changed.template_issues.clear();
  changed.templates={Candidate(source,"x"),Candidate(source,"x")};reject(changed,std::string("x"));
  changed.templates.resize(1);changed.templates[0].listed=false;reject(changed,std::string("x"));
  EXPECT_FALSE(SelectChatPolicy(metadata,{},nullptr,&error));
}
TEST(ChatPolicy, EveryKnownProtocolKeepsOrderedDataRootsAndAtomicMarkers) {
  struct Case {
    const char* file;
    std::vector<std::string> roots;
    std::vector<std::string> atomic_markers;
  };
  const std::vector<std::string> chatml_roots = {
      "loomify.text.tools_prefix", "loomify.text.tools_suffix",
      "loomify.text.effort_xhigh", "loomify.text.effort_low",
      "loomify.text.default_effort", "loomify.text.default_effort_instruction"};
  const Case cases[] = {
      {"lib/text/llama/templates/5816fce1.jinja",
       {"loomify.text.fallback_date", "loomify.text.system_preamble"},
       {"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}},
      {"lib/text/qwen35/templates/7f0e5290.jinja", chatml_roots,
       {"<|im_start|>", "<|im_end|>"}},
      {"lib/text/qwen35/templates/c3cf9e34.jinja", chatml_roots,
       {"<|im_start|>", "<|im_end|>"}},
      {"lib/text/qwen3/templates/5da44855.jinja", chatml_roots,
       {"<|im_start|>", "<|im_end|>"}},
      {"lib/text/gemma/templates/7de1c58e.jinja", {},
       {"<start_of_turn>", "<end_of_turn>"}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.file);
    model::TextMetadata metadata;
    metadata.templates = {Candidate(Source(test.file))};
    ChatPolicySelection policy;
    std::string error;
    ASSERT_TRUE(SelectChatPolicy(metadata, {}, &policy, &error)) << error;
    const std::vector<std::string> roots(policy.data_roots.begin(), policy.data_roots.end());
    EXPECT_EQ(roots, test.roots);
    std::vector<std::string> atomic;
    for (const auto& marker : policy.markers)
      if (marker.require_single_token) atomic.push_back(marker.text);
    EXPECT_EQ(atomic, test.atomic_markers);
    // Every retained data root must name a public definition, in the same order.
    size_t cursor = 0;
    for (const auto& root : roots) {
      const auto declaration = "global.rodata.def public @" + root + " = bytes(\"";
      const auto found = policy.configuration_source.find(declaration, cursor);
      ASSERT_NE(found, std::string::npos);
      cursor = found + declaration.size();
    }
  }
}
TEST(ChatPolicy, QuotedDataKeepsDecodedBytesAndExactSizeBoundary) {
  const auto original = Source("lib/text/llama/templates/5816fce1.jinja");
  const std::pair<std::string, std::string> cases[] = {
      {R"(a\nb\rc\td\\e\'f\"g café)", "a\nb\rc\td\\e'f\"g café"},
      {std::string(65536, 'x'), std::string(65536, 'x')},
  };
  for (const auto& test : cases) {
    model::TextMetadata metadata;
    metadata.templates = {Candidate(Replace(original, "26 Jul 2024", test.first))};
    ChatPolicySelection policy;
    std::string error;
    ASSERT_TRUE(SelectChatPolicy(metadata, {}, &policy, &error)) << error;
    EXPECT_EQ(policy.configuration_source,
              "global.rodata.def public @loomify.text.fallback_date = bytes(\"" +
                  Hex(test.second) + "\")\n" +
                  "global.rodata.def public @loomify.text.system_preamble = bytes(\"" +
                  Hex("Cutting Knowledge Date: December 2023\nToday Date: ") + "\")\n");
    metadata.templates = {Candidate(Replace(original, "26 Jul 2024", std::string(65537, 'x')))};
    const auto prior = policy.configuration_source;
    EXPECT_FALSE(SelectChatPolicy(metadata, {}, &policy, &error));
    EXPECT_EQ(policy.configuration_source, prior);
  }
}
TEST(ChatPolicy, LlamaSystemPreambleComesFromBothSourceLiterals) {
  const auto original = Source("lib/text/llama/templates/5816fce1.jinja");
  struct Case {
    std::string cutoff_source, label_source, preamble;
  };
  const Case cases[] = {
      {R"(Cutting Knowledge Date: December 2023\n)", "Today Date: ",
       "Cutting Knowledge Date: December 2023\nToday Date: "},
      {R"(Knowledge snapshot: May 2025\n)", "Today Date: ",
       "Knowledge snapshot: May 2025\nToday Date: "},
      {R"(Cutting Knowledge Date: December 2023\n)", u8"Current date café: ",
       u8"Cutting Knowledge Date: December 2023\nCurrent date café: "},
      {R"(Knowledge snapshot:\tMay 2025\n)", u8"Current date café: ",
       u8"Knowledge snapshot:\tMay 2025\nCurrent date café: "},
      {"", "", ""},
  };
  for (const auto& test : cases) {
    auto source = Replace(original, R"(Cutting Knowledge Date: December 2023\n)",
                          test.cutoff_source);
    source = Replace(source, "Today Date: ", test.label_source);
    source = Replace(source, "26 Jul 2024", "10 Nov 2026");
    model::TextMetadata metadata;
    metadata.templates = {Candidate(source)};
    ChatPolicySelection policy;
    std::string error;
    ASSERT_TRUE(SelectChatPolicy(metadata, {}, &policy, &error)) << error;
    EXPECT_EQ(policy.source_text, source);
    EXPECT_EQ(policy.configuration_source,
              "global.rodata.def public @loomify.text.fallback_date = bytes(\"" +
                  Hex("10 Nov 2026") + "\")\n" +
                  "global.rodata.def public @loomify.text.system_preamble = bytes(\"" +
                  Hex(test.preamble) + "\")\n");
    EXPECT_EQ(policy.skeleton_sha256,
              "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4");
  }
}
TEST(ChatPolicy, LlamaPreambleStillRejectsTagsEscapesAndStructuralChanges) {
  const auto original = Source("lib/text/llama/templates/5816fce1.jinja");
  const std::pair<std::string, std::string> changes[] = {
      {"December 2023", "<|eot_id|>"},
      {"Today Date: ", "<new_tag>"},
      {"Today Date: ", R"(bad\u000a)"},
      {"Today Date: ", std::string(65537, 'x')},
      {"December 2023", std::string("bad ") + char(0xc0) + char(0xaf)},
      {R"("Today Date: ")", "'Today Date: '"},
      {" + date_string + ", " + unsupported_variable + "},
  };
  model::TextMetadata metadata;
  metadata.templates = {Candidate(original)};
  ChatPolicySelection policy;
  std::string error;
  ASSERT_TRUE(SelectChatPolicy(metadata, {}, &policy, &error)) << error;
  const auto prior = policy.configuration_source;
  for (const auto& change : changes) {
    SCOPED_TRACE(change.first);
    metadata.templates = {Candidate(Replace(original, change.first, change.second))};
    EXPECT_FALSE(SelectChatPolicy(metadata, {}, &policy, &error));
    EXPECT_EQ(policy.configuration_source, prior);
    EXPECT_EQ(policy.source_text, original);
    EXPECT_FALSE(error.empty());
  }
}
} // namespace
} // namespace hrx::loomify_gguf
