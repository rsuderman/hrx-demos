// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/token_policy.h"
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/publication_internal.h"
#include "lib/loomify/template_analysis.h"
#include "lib/loomify/text/jinja/format_analysis.h"
#include "lib/loomify/template_probe_runner.h"
#include "tools/model/text_metadata.h"
#include "gtest/gtest.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include <algorithm>
#include <memory>
#include <cstdlib>
#include <filesystem>
namespace hrx::loomify_gguf {
namespace {
void Number(std::string &s, uint64_t n, int width) {
  for (int i = 0; i < width; ++i)
    s += static_cast<char>(n >> (8 * i));
}
void String(std::string &s, std::string_view v) {
  Number(s, v.size(), 8);
  s += v;
}
bool BindAnalyzedFrame(FILE *file, const std::vector<std::string> &globals,
                       const std::vector<std::string> &frames,
                       const model::TextMetadata &metadata,
                       const std::optional<PublicationTokenFacts> &publication,
                       const std::vector<std::string> &markers,
                       std::string_view json, BoundTokenPolicy *out,
                       std::string *error) {
  PreparedTokenBindings prepared;
  if (!PrepareTokenBindings(globals, metadata, json, &prepared, error))
    return false;
  return FinalizeTokenPolicy(file, prepared, metadata, publication, frames,
                             markers, out, error);
}
std::string PreparationSource(bool default_feature = false) {
  return "{% set enabled = feature | default(" +
         std::string(default_feature ? "true" : "false") +
         ") %}"
         "{{ bos_token }}{% for m in messages %}"
         "{% if m.role == 'assistant' %}"
         "{% if enabled %}{{ '[bright]' }}{% else %}{{ '[assistant]' }}{% endif %}"
         "{{ m.content + '<|eot_id|>' }}"
         "{% else %}{{ '[' + m.role + ']' + m.content + '<|eot_id|>' }}{% endif %}"
         "{% endfor %}{% if add_generation_prompt %}"
         "{% if enabled %}{{ '[bright]' }}{% else %}{{ '[assistant]' }}{% endif %}"
         "{% endif %}";
}
jinja::TemplateAnalysisOptions PreparationOptions(bool bytecode) {
  const char *runfiles = std::getenv("TEST_SRCDIR");
  const char *workspace = std::getenv("TEST_WORKSPACE");
  const char *scratch = std::getenv("TEST_TMPDIR");
  EXPECT_NE(runfiles, nullptr);
  EXPECT_NE(workspace, nullptr);
  EXPECT_NE(scratch, nullptr);
  return {(std::filesystem::path(runfiles ? runfiles : "") /
           (workspace ? workspace : ""))
              .string(),
          scratch ? scratch : "", bytecode};
}
jinja::GeneratedInputRenderer
PreparedChatProduct(const PreparedTextControl &prepared) {
  jinja::GeneratedInputRenderer product;
  product.entrypoint = "test.prepared_chat";
  product.helper_requirements = prepared.helper_requirements;
  product.source = prepared.source + R"(
func.decl @chat_begin(%request: buffer, %tools: buffer) -> (buffer, i64)
func.def public @test.prepared_chat(%request: buffer) -> (buffer) {
  %empty = buffer.constant utf8("") : buffer
  %enabled, %efforts, %default = func.call @chat_reasoning() : () -> (i32, i32, i32)
  %zero = scalar.constant 0 : i32
  %no_reasoning = scalar.cmpi eq, %enabled, %zero : i32
  %no_efforts = scalar.cmpi eq, %efforts, %zero : i32
  %no_default = scalar.cmpi eq, %default, %zero : i32
  %first = scalar.andi %no_reasoning, %no_efforts : i1
  %valid = scalar.andi %first, %no_default : i1
  func.call @text.output_check(%valid) : (i1)
  %rendered, %state = func.call @chat_begin(%request, %empty) : (buffer, buffer) -> (buffer, i64)
  func.return %rendered : buffer
}
)";
  return product;
}
std::string UserPreparationSource(bool needs_history = false) {
  return "{{ bos_token }}{% for m in messages %}"
         "{% if m.role == 'assistant' %}{{ '[assistant]' + m.content }}"
         "{% else %}{{ '[user]' }}" +
         std::string(
             needs_history
                 ? "{% if loop.length > 2 %}{{ 'long:' }}{% else %}{{ 'short:' }}{% endif %}"
                 : "") +
         "{{ m.content }}{% endif %}{{ '<|eot_id|>\\n' }}{% endfor %}"
         "{% if add_generation_prompt %}{{ '[assistant]' }}{% endif %}";
}
jinja::GeneratedInputRenderer
PreparedUserProduct(const PreparedTextControl &prepared, bool terminated,
                    bool include_user = true) {
  jinja::GeneratedInputRenderer product;
  product.entrypoint = "test.prepared_user";
  product.helper_requirements = prepared.helper_requirements;
  product.source = prepared.source + R"(
func.def public @test.prepared_user(%text: buffer) -> (buffer) {
)";
  product.source += "  %separator = func.call @text.input_" +
                    std::string(terminated ? "terminated" : "open") +
                    "_separator() : () -> (buffer)\n";
  if (include_user) {
    product.source += R"(
  %fragment = func.call @text.input_user_continuation(%text) : (buffer) -> (buffer)
  %empty = buffer.constant utf8("") : buffer
  %value = func.call @text.join8(%separator, %fragment, %empty, %empty, %empty, %empty, %empty, %empty) : (buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer) -> (buffer)
  func.return %value : buffer
}
)";
  } else {
    product.source += "  func.return %separator : buffer\n}\n";
  }
  return product;
}
std::string EncodedInputSource() {
  // Use actual special pieces from the GGUF/BPE fixture so every framing byte
  // has an independently known token ID; ordinary text is the fixture's 'a'.
  return "{{ bos_token }}{% for m in messages %}"
         "{% if m.role == 'assistant' %}"
         "{{ '<|header|>' + m.content + '<|eot_id|>' }}"
         "{% else %}{{ '<|header|>' + m.content + '<|eot_id|>' }}{% endif %}"
         "{% endfor %}{% if add_generation_prompt %}{{ '<|header|>' }}{% endif %}";
}
jinja::GeneratedInputRenderer
PreparedEncodedInputProduct(const PreparedTextControl &prepared) {
  jinja::GeneratedInputRenderer product;
  product.entrypoint = "test.prepared_input";
  product.helper_requirements = prepared.helper_requirements;
  // This wrapper marshals the hook arguments and returns only its live ID
  // bytes. The production encode capability and all input preparation logic
  // come from their actual modules and source-derived package products.
  product.source = prepared.source + R"(
func.decl @prepare_input(%text: buffer, %media: buffer, %media_count: i64, %format: i32, %boundary: i32, %capacity: i64) -> (buffer, i64, buffer)
func.decl @text.lookup(%json: buffer, %key: buffer) -> (buffer, i64, i1)
func.decl @text.string(%json: buffer, %key: buffer) -> (buffer, i1)
func.decl @text.range(%text: buffer, %begin: i64, %length: i64) -> (buffer)
func.def @test.input_number(%request: buffer, %key: buffer) -> (i64) {
  %raw, %kind, %found = func.call @text.lookup(%request, %key) : (buffer, buffer) -> (buffer, i64, i1)
  func.call @text.output_check(%found) : (i1)
  %value = func.call @jinja.tointeger(%raw, %kind) : (buffer, i64) -> (i64)
  func.return %value : i64
}
func.def public @test.prepared_input(%request: buffer) -> (buffer) {
  %text_key = buffer.constant utf8("text") : buffer
  %text, %found = func.call @text.string(%request, %text_key) : (buffer, buffer) -> (buffer, i1)
  func.call @text.output_check(%found) : (i1)
  %format_key = buffer.constant utf8("format") : buffer
  %boundary_key = buffer.constant utf8("boundary") : buffer
  %capacity_key = buffer.constant utf8("capacity") : buffer
  %media_key = buffer.constant utf8("media_count") : buffer
  %format64 = func.call @test.input_number(%request, %format_key) : (buffer, buffer) -> (i64)
  %boundary64 = func.call @test.input_number(%request, %boundary_key) : (buffer, buffer) -> (i64)
  %capacity = func.call @test.input_number(%request, %capacity_key) : (buffer, buffer) -> (i64)
  %media_count = func.call @test.input_number(%request, %media_key) : (buffer, buffer) -> (i64)
  %format = scalar.trunci %format64 : i64 to i32
  %boundary = scalar.trunci %boundary64 : i64 to i32
  %empty = buffer.constant utf8("") : buffer
  %tokens, %count, %media = func.call @prepare_input(%text, %empty, %media_count, %format, %boundary, %capacity) : (buffer, buffer, i64, i32, i32, i64) -> (buffer, i64, buffer)
  %zero = index.constant 0 : offset
  %media_bytes = buffer.length %media
  %no_media = index.cmp eq, %media_bytes, %zero : offset
  func.call @text.output_check(%no_media) : (i1)
  %four = scalar.constant 4 : i64
  %bytes = scalar.muli %count, %four : i64
  %zero64 = scalar.constant 0 : i64
  %live = func.call @text.range(%tokens, %zero64, %bytes) : (buffer, i64, i64) -> (buffer)
  func.return %live : buffer
}
)";
  return product;
}
std::string InputRequest(int format, int boundary, size_t capacity,
                         std::string_view text = "a", int media_count = 0) {
  return "{\"text\":\"" + std::string(text) +
         "\",\"format\":" + std::to_string(format) +
         ",\"boundary\":" + std::to_string(boundary) +
         ",\"capacity\":" + std::to_string(capacity) +
         ",\"media_count\":" + std::to_string(media_count) + "}";
}
std::string TokenBytes(const std::vector<int32_t> &ids) {
  std::string result;
  for (int32_t id : ids)
    Number(result, static_cast<uint32_t>(id), 4);
  return result;
}
class TokenPolicyTest : public ::testing::Test {
protected:
  void Open(bool missing_stop = false, bool bad_template = true,
            bool chatml = false, std::string terminal = {}, bool gemma = false,
            bool qwen3 = false, bool qwen2 = false, bool plain = false,
            bool duplicate_stop = false, std::string extra_piece = {},
            bool duplicate_terminal = false, bool missing_end_text = false) {
    pieces = {"a",
              "<|begin_of_text|>",
              missing_end_text ? "other_end_text" : "<|end_of_text|>",
              missing_stop ? "other" : "<|eom_id|>",
              terminal.empty() ? (chatml ? "<|im_end|>" : "<|eot_id|>")
                               : terminal,
              "<|header|>"};
    if (gemma)
      pieces = {"a",
                "<pad>",
                "<eos>",
                "<bos>",
                "<unk>",
                "<start_of_turn>",
                missing_stop ? "other" : "<end_of_turn>"};
    if (qwen3 || qwen2)
      pieces = {
          "a",       "<|im_start|>", missing_stop ? "other" : "<|endoftext|>",
          "<think>", "<|im_end|>",   "</think>"};
    if (plain)
      pieces = {"a",
                "<|im_start|>",
                "<|endoftext|>",
                missing_stop ? "other" : "<|im_end|>",
                "<|endofturn|>",
                "<|other|>"};
    if (duplicate_stop)
      pieces.push_back(pieces[2]);
    if (duplicate_terminal)
      pieces.push_back(pieces[4]);
    if (!extra_piece.empty())
      pieces.push_back(std::move(extra_piece));
    std::string bytes = "GGUF";
    Number(bytes, 3, 4);
    Number(bytes, 0, 8);
    Number(bytes, 7, 8);
    auto field = [&](std::string_view k, std::string_view v) {
      String(bytes, k);
      Number(bytes, 8, 4);
      String(bytes, v);
    };
    field("general.architecture", qwen2   ? "qwen2"
                                  : qwen3 ? "qwen3"
                                  : gemma ? "gemma3"
                                          : "llama");
    field("tokenizer.ggml.model", gemma ? "llama" : "gpt2");
    field("tokenizer.ggml.pre", plain              ? "dbrx"
                                : (qwen3 || qwen2) ? "qwen2"
                                : gemma            ? "default"
                                : chatml           ? "qwen35"
                                                   : "llama-bpe");
    String(bytes, "tokenizer.ggml.tokens");
    Number(bytes, 9, 4);
    Number(bytes, 8, 4);
    Number(bytes, pieces.size(), 8);
    for (const auto &p : pieces)
      String(bytes, p);
    String(bytes, "tokenizer.ggml.eos_token_id");
    Number(bytes, 4, 4);
    Number(bytes, gemma ? 2 : 4, 4);
    String(bytes, "tokenizer.ggml.bos_token_id");
    Number(bytes, 4, 4);
    Number(bytes, (qwen2 || plain) ? 2 : gemma ? 3 : 1, 4);
    String(bytes, "tokenizer.chat_template");
    if (bad_template) {
      Number(bytes, 4, 4);
      Number(bytes, 42, 4);
    } else {
      Number(bytes, 8, 4);
      String(bytes, "unknown template");
    }
    while (bytes.size() % 32)
      bytes += '\0';
    file.reset(std::tmpfile());
    ASSERT_TRUE(file);
    ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
    ASSERT_EQ(std::fwrite(bytes.data(), 1, bytes.size(), file.get()),
              bytes.size());
    model::GgufCatalog catalog;
    ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
    ASSERT_TRUE(model::ReadTextMetadata(file.get(), catalog, &text, &error))
        << error;
  }
  std::string Json(bool wrong_eos = false, bool marker_special = true) {
    std::string json = R"({"version":"1.0","model":{"type":"BPE","vocab":{)";
    for (size_t i = 0; i < pieces.size(); ++i) {
      if (i)
        json += ',';
      json += "\"" + (wrong_eos && i == 4 ? "wrong" : pieces[i]) +
              "\":" + std::to_string(i);
    }
    json += R"(},"merges":[]},"added_tokens":[)";
    for (size_t i = 1; i < pieces.size(); ++i) {
      if (i != 1)
        json += ',';
      json +=
          "{\"id\":" + std::to_string(i) + ",\"content\":\"" +
          (wrong_eos && i == 4 ? "wrong" : pieces[i]) +
          "\",\"special\":" + (i == 5 && !marker_special ? "false" : "true") +
          ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false}";
    }
    return json + "]}";
  }
  std::unique_ptr<FILE, decltype(&std::fclose)> file{nullptr, std::fclose};
  model::TextMetadata text;
  std::vector<std::string> pieces;
  std::string error;
};
TEST_F(TokenPolicyTest,
       GeneratedTextPreparationOwnsNativeSourceAfterInputTeardown) {
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    const auto options = PreparationOptions(bytecode);
    PreparedTextControl prepared;
    {
      Open();
      std::optional<PublicationTokenFacts> publication;
      ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "llama", text,
                                              &publication, &error))
          << error;
      ChatTemplateSelection selected{
          "uncatalogued.template", {}, PreparationSource()};
      const auto tokenizer = Json();
      const auto metadata = text;
      ASSERT_TRUE(PrepareGeneratedText(selected, file.get(), metadata,
                                       publication, tokenizer, options,
                                       &prepared, &error))
          << error;
    }
    // Every input owner and all temporary preparation VMs/tokenizers are gone
    // before the returned source is linked and invoked through the real host.
    file.reset();
    text = {};
    pieces.clear();
    std::unique_ptr<jinja::PreparedTemplate> native;
    ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
        PreparedChatProduct(prepared), options.helper_root,
        options.scratch_parent, bytecode, &native, &error))
        << error;
    std::string rendered;
    ASSERT_TRUE(native->Render(
        R"({"messages":[{"role":"user","content":"a"}]})", &rendered, &error))
        << error;
    EXPECT_EQ(rendered, "<|begin_of_text|>[user]a<|eot_id|>[assistant]");
  }
}
TEST_F(TokenPolicyTest,
       GeneratedTextPreparationMissingHelperPreservesOwnedResult) {
  Open();
  auto options = PreparationOptions(false);
  // The existing test scratch root has no declared Loom helper closure.
  options.helper_root = options.scratch_parent;
  // Missing helpers fail during eligible cold discovery. Non-effort source
  // preparation only constructs owned products and does not load those helpers.
  std::string source =
      "{% set effort = reasoning_effort | default('high') %}"
      "{% if effort not in ('high', 'low') %}"
      "{{ raise_exception('unsupported effort') }}{% endif %}" +
      PreparationSource();
  const auto content = source.find("{{ m.content + '<|eot_id|>' }}");
  ASSERT_NE(content, std::string::npos);
  source.insert(
      content, "{% if m.reasoning_content %}"
               "{{ '<think>' + m.reasoning_content + '</think>' }}{% endif %}");
  const ChatTemplateSelection selected{"uncatalogued.template", {}, source};
  PreparedTextControl prepared{
      "prior source", "prior token", {"prior helper"}, {"prior root"}};
  EXPECT_FALSE(PrepareGeneratedText(selected, file.get(), text, {}, Json(),
                                    options, &prepared, &error));
  EXPECT_NE(error.find("missing template helper resource"), std::string::npos)
      << error;
  EXPECT_EQ(prepared.source, "prior source");
  EXPECT_EQ(prepared.token_source, "prior token");
  EXPECT_EQ(prepared.helper_requirements,
            (std::vector<std::string>{"prior helper"}));
  EXPECT_EQ(prepared.roots, (std::vector<std::string>{"prior root"}));
}
TEST_F(TokenPolicyTest,
       NonEffortPreparationPublishesProductsWithoutLoadingHelpers) {
  Open();
  auto options = PreparationOptions(false);
  options.helper_root = options.scratch_parent;
  const ChatTemplateSelection selected{
      "uncatalogued.template", {}, PreparationSource()};
  PreparedTextControl prepared{
      "prior source", "prior token", {"prior helper"}, {"prior root"}};
  ASSERT_TRUE(PrepareGeneratedText(selected, file.get(), text, {}, Json(),
                                   options, &prepared, &error))
      << error;
  EXPECT_TRUE(error.empty());
  EXPECT_NE(prepared.source.find("func.def public @jinja.evaluate("),
            std::string::npos);
  EXPECT_NE(prepared.source.find("func.def public @chat_reasoning("),
            std::string::npos);
  EXPECT_FALSE(prepared.token_source.empty());
  EXPECT_NE(prepared.token_source, "prior token");
  EXPECT_NE(std::find(prepared.helper_requirements.begin(),
                      prepared.helper_requirements.end(),
                      "lib/text/common/jinja.loom"),
            prepared.helper_requirements.end());
  EXPECT_NE(
      std::find(prepared.roots.begin(), prepared.roots.end(), "chat_begin"),
      prepared.roots.end());
}
TEST_F(TokenPolicyTest,
       GeneratedTextPreparationTracksUncataloguedDefaultAndKwarg) {
  Open();
  for (bool bytecode : {false, true}) {
    const auto options = PreparationOptions(bytecode);
    for (bool default_feature : {false, true}) {
      SCOPED_TRACE(bytecode ? "bytecode" : "source");
      SCOPED_TRACE(default_feature);
      const ChatTemplateSelection selected{"uncatalogued.changed-default",
                                           "selected-by-gguf-name",
                                           PreparationSource(default_feature)};
      PreparedTextControl prepared;
      ASSERT_TRUE(PrepareGeneratedText(selected, file.get(), text, {}, Json(),
                                       options, &prepared, &error))
          << error;
      std::unique_ptr<jinja::PreparedTemplate> native;
      ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
          PreparedChatProduct(prepared), options.helper_root,
          options.scratch_parent, bytecode, &native, &error))
          << error;
      std::string rendered;
      ASSERT_TRUE(native->Render(
          R"({"messages":[{"role":"user","content":"a"}]})", &rendered, &error))
          << error;
      const std::string prefix = "<|begin_of_text|>[user]a<|eot_id|>";
      EXPECT_EQ(rendered,
                prefix + (default_feature ? "[bright]" : "[assistant]"));
      const std::string override_request =
          R"({"messages":[{"role":"user","content":"a"}],"model_options":{"feature":)" +
          std::string(default_feature ? "false" : "true") + "}}";
      ASSERT_TRUE(native->Render(override_request, &rendered, &error)) << error;
      EXPECT_EQ(rendered,
                prefix + (default_feature ? "[assistant]" : "[bright]"));
    }
  }
}
TEST_F(TokenPolicyTest,
       GeneratedUserPreparationOwnsContinuationAndBoundaryBytes) {
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    const auto options = PreparationOptions(bytecode);
    PreparedTextControl prepared;
    {
      Open();
      const ChatTemplateSelection selected{
          "uncatalogued.user", {}, UserPreparationSource()};
      const auto metadata = text;
      const auto tokenizer = Json();
      ASSERT_TRUE(PrepareGeneratedText(selected, file.get(), metadata, {},
                                       tokenizer, options, &prepared, &error))
          << error;
    }
    file.reset();
    text = {};
    pieces.clear();
    const std::string user = "  a\"\\é\n  ";
    for (bool terminated : {false, true}) {
      SCOPED_TRACE(terminated ? "terminated" : "open");
      std::unique_ptr<jinja::PreparedTemplate> native;
      ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
          PreparedUserProduct(prepared, terminated), options.helper_root,
          options.scratch_parent, bytecode, &native, &error))
          << error;
      std::string actual;
      ASSERT_TRUE(native->Render(user, &actual, &error)) << error;
      // OPEN supplies S; TERMINATED supplies only W after the actual validated
      // token T in S=T+W. The source recipe owns user text and generation
      // bytes.
      const std::string separator = terminated ? "\n" : "<|eot_id|>\n";
      EXPECT_EQ(actual,
                separator + "[user]" + user + "<|eot_id|>\n[assistant]");
      ASSERT_TRUE(native->Render("", &actual, &error)) << error;
      EXPECT_EQ(actual, separator + "[user]<|eot_id|>\n[assistant]");
    }
  }
}
TEST_F(TokenPolicyTest,
       UnavailableUserShortcutPreservesChatAndRenderedBoundaries) {
  Open();
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    const auto options = PreparationOptions(bytecode);
    const ChatTemplateSelection selected{
        "context-dependent.user", {}, UserPreparationSource(true)};
    PreparedTextControl prepared;
    ASSERT_TRUE(PrepareGeneratedText(selected, file.get(), text, {}, Json(),
                                     options, &prepared, &error))
        << error;
    std::unique_ptr<jinja::PreparedTemplate> chat;
    ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
        PreparedChatProduct(prepared), options.helper_root,
        options.scratch_parent, bytecode, &chat, &error))
        << error;
    std::string actual;
    ASSERT_TRUE(chat->Render(R"({"messages":[{"role":"user","content":"a"}]})",
                             &actual, &error))
        << error;
    EXPECT_EQ(actual, "<|begin_of_text|>[user]short:a<|eot_id|>\n[assistant]");
    for (bool terminated : {false, true}) {
      SCOPED_TRACE(terminated ? "terminated" : "open");
      std::unique_ptr<jinja::PreparedTemplate> boundary, user;
      ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
          PreparedUserProduct(prepared, terminated, false), options.helper_root,
          options.scratch_parent, bytecode, &boundary, &error))
          << error;
      ASSERT_TRUE(boundary->Render("ignored", &actual, &error)) << error;
      EXPECT_EQ(actual, terminated ? "\n" : "<|eot_id|>\n");
      ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
          PreparedUserProduct(prepared, terminated), options.helper_root,
          options.scratch_parent, bytecode, &user, &error))
          << error;
      actual = "preserved";
      EXPECT_FALSE(user->Render("a", &actual, &error));
      EXPECT_EQ(actual, "preserved");
      EXPECT_NE(error.find("context-dependent.user"), std::string::npos)
          << error;
      EXPECT_NE(error.find("continued USER"), std::string::npos) << error;
    }
  }
}
TEST_F(TokenPolicyTest, GeneratedInputEncodesAllSixModesWithActualTokenizer) {
  Open();
  const auto json = Json();
  iree_tokenizer_t *raw = nullptr;
  auto status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(),
      &raw);
  ASSERT_TRUE(iree_status_is_ok(status));
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(
      raw, iree_tokenizer_free);
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    const auto options = PreparationOptions(bytecode);
    const ChatTemplateSelection selected{
        "uncatalogued.encoded-input", {}, EncodedInputSource()};
    PreparedTextControl prepared;
    ASSERT_TRUE(PrepareGeneratedText(selected, file.get(), text, {}, json,
                                     options, &prepared, &error))
        << error;
    std::unique_ptr<jinja::PreparedTemplate> native;
    ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
        PreparedEncodedInputProduct(prepared), options.helper_root,
        options.scratch_parent, bytecode, &native, &error, tokenizer.get()))
        << error;
    // IDs follow the independently authored GGUF/BPE fixture: a=0, BOS=1,
    // terminal=4, header=5. No encoder invocation constructs expectations.
    const std::vector<std::vector<int32_t>> rendered = {{0}, {4, 0}, {0}};
    const std::vector<std::vector<int32_t>> user = {
        {1, 5, 0, 4, 5}, {4, 5, 0, 4, 5}, {5, 0, 4, 5}};
    for (int format : {0, 1}) {
      for (int boundary : {0, 1, 2}) {
        SCOPED_TRACE("format=" + std::to_string(format) +
                     " boundary=" + std::to_string(boundary));
        const auto &ids = format ? user[boundary] : rendered[boundary];
        std::string actual;
        ASSERT_TRUE(native->Render(InputRequest(format, boundary, ids.size()),
                                   &actual, &error))
            << error;
        EXPECT_EQ(actual, TokenBytes(ids));
        actual = "preserved";
        EXPECT_FALSE(native->Render(
            InputRequest(format, boundary, ids.size() - 1), &actual, &error));
        EXPECT_EQ(actual, "preserved");
        EXPECT_NE(error.find("token capacity"), std::string::npos) << error;
        ASSERT_TRUE(native->Render(InputRequest(format, boundary, ids.size()),
                                   &actual, &error))
            << error;
        EXPECT_EQ(actual, TokenBytes(ids));
      }
    }
    for (const auto &request : {InputRequest(2, 0, 8), InputRequest(0, 3, 8),
                                InputRequest(0, 0, 8, "a", 1)}) {
      std::string actual = "preserved";
      EXPECT_FALSE(native->Render(request, &actual, &error));
      EXPECT_EQ(actual, "preserved");
      EXPECT_NE(error.find("format, boundary, media or capacity"),
                std::string::npos)
          << error;
    }
  }
}
TEST_F(TokenPolicyTest,
       GeneratedInputRejectsAddedIdsOutsideNumericalVocabulary) {
  Open();
  auto json = Json();
  // A genuine native tokenizer addition is legal externally while its ID=6
  // lies beyond the GGUF numerical vocabulary's six rows.
  json.resize(json.size() - 2);
  json +=
      R"(,{"id":6,"content":"<|outside|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}]})";
  iree_tokenizer_t *raw = nullptr;
  auto status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(),
      &raw);
  ASSERT_TRUE(iree_status_is_ok(status));
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(
      raw, iree_tokenizer_free);
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode ? "bytecode" : "source");
    const auto options = PreparationOptions(bytecode);
    const ChatTemplateSelection selected{
        "uncatalogued.numerical-domain", {}, EncodedInputSource()};
    PreparedTextControl prepared;
    ASSERT_TRUE(PrepareGeneratedText(selected, file.get(), text, {}, json,
                                     options, &prepared, &error))
        << error;
    std::unique_ptr<jinja::PreparedTemplate> native;
    ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
        PreparedEncodedInputProduct(prepared), options.helper_root,
        options.scratch_parent, bytecode, &native, &error, tokenizer.get()))
        << error;
    for (int format : {0, 1}) {
      for (int boundary : {0, 1, 2}) {
        SCOPED_TRACE("format=" + std::to_string(format) +
                     " boundary=" + std::to_string(boundary));
        std::string actual = "preserved";
        EXPECT_FALSE(
            native->Render(InputRequest(format, boundary, 16, "<|outside|>"),
                           &actual, &error));
        EXPECT_EQ(actual, "preserved");
        EXPECT_NE(error.find("numerical token domain"), std::string::npos)
            << error;
      }
    }
    std::string actual;
    ASSERT_TRUE(native->Render(InputRequest(1, 0, 5), &actual, &error))
        << error;
    EXPECT_EQ(actual, TokenBytes({1, 5, 0, 4, 5}));
  }
}
TEST_F(TokenPolicyTest, SourceBindingsKeepTokenizerNamesAsFacts) {
  Open();
  text.tokenizer_model = "uncatalogued-native-compatible-model";
  text.pre_tokenizer = "uncatalogued-pre";
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({"bos_token", "messages"}, text, Json(),
                                   &prepared, &error))
      << error;
  EXPECT_EQ(prepared.facts().tokenizer_model, text.tokenizer_model);
  EXPECT_EQ(prepared.facts().pre_tokenizer, text.pre_tokenizer);
  EXPECT_EQ(prepared.globals().size(), 1u);
  EXPECT_EQ(prepared.globals().at("bos_token"), "<|begin_of_text|>");
  EXPECT_EQ(prepared.globals_json(), "{\"bos_token\":\"<|begin_of_text|>\"}");
  BoundTokenPolicy bound;
  ASSERT_TRUE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                  {"<|eot_id|>\n", "<|eot_id|>  "}, {}, &bound,
                                  &error))
      << error;
  EXPECT_EQ(bound.terminal.id, 4);
  EXPECT_EQ(bound.encode_flags, 0u);
}
TEST_F(TokenPolicyTest, MissingGuardedGlobalsRemainUndefined) {
  Open();
  text.tokens.clear();
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({"bos_token", "eos_token", "pad_token"},
                                   text, Json(), &prepared, &error))
      << error;
  EXPECT_TRUE(prepared.globals().empty());
  EXPECT_EQ(prepared.globals_json(), "{}");
  EXPECT_EQ(prepared.global_names(),
            (std::vector<std::string>{"bos_token", "eos_token", "pad_token"}));
  BoundTokenPolicy bound;
  ASSERT_TRUE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                  {"<|eot_id|>"}, {}, &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.piece, "<|eot_id|>");
  EXPECT_TRUE(bound.roles.empty());
}
TEST_F(TokenPolicyTest,
       UnconsumedMalformedAndMismatchedRolesAreNotObligations) {
  Open();
  text.token_issues.push_back(
      {"tokenizer.ggml.padding_token_id", "bad unused PAD"});
  auto json = Json(true);
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({}, text, json, &prepared, &error)) << error;
  EXPECT_TRUE(prepared.globals().empty());
  // The mismatch becomes an obligation only when that actual source global is
  // bound.
  EXPECT_FALSE(
      PrepareTokenBindings({"eos_token"}, text, json, &prepared, &error));
  EXPECT_NE(error.find("identity mismatch"), std::string::npos);
  EXPECT_TRUE(prepared.globals().empty());
  EXPECT_FALSE(
      PrepareTokenBindings({"pad_token"}, text, Json(), &prepared, &error));
  EXPECT_NE(error.find("bad unused PAD"), std::string::npos);
}
TEST_F(TokenPolicyTest, SourceBindingFailurePreservesOwnedGlobals) {
  Open();
  PreparedTokenBindings prepared;
  ASSERT_TRUE(
      PrepareTokenBindings({"bos_token"}, text, Json(), &prepared, &error));
  EXPECT_FALSE(PrepareTokenBindings({}, text, "{invalid", &prepared, &error));
  EXPECT_NE(error.find("native tokenizer JSON"), std::string::npos);
  EXPECT_EQ(prepared.globals().at("bos_token"), "<|begin_of_text|>");
  EXPECT_FALSE(PrepareTokenBindings({}, text, Json(), nullptr, &error));
}
TEST_F(TokenPolicyTest, SourceGlobalsRequireNativeSpecialIdentity) {
  Open();
  text.tokens.push_back({"tokenizer.ggml.padding_token_id", 4, 5});
  text.token_pieces[5] = "<|header|>";
  PreparedTokenBindings prepared;
  EXPECT_FALSE(PrepareTokenBindings({"pad_token"}, text, Json(false, false),
                                    &prepared, &error));
  EXPECT_NE(error.find("special piece"), std::string::npos);
}
TEST_F(TokenPolicyTest, AnalyzedGemmaFrameIsIndependentOfDeclaredEos) {
  Open(false, true, false, {}, true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "gemma3", text, &facts, &error));
  PreparedTokenBindings prepared;
  ASSERT_TRUE(
      PrepareTokenBindings({"bos_token"}, text, Json(), &prepared, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(FinalizeTokenPolicy(file.get(), prepared, text, facts,
                                  {"<end_of_turn>\n"}, {}, &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.piece, "<end_of_turn>");
  EXPECT_EQ(bound.terminal.id, 6);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.eos_token_id").id, 2);
  EXPECT_EQ(prepared.globals().at("bos_token"), "<bos>");
}
TEST_F(TokenPolicyTest, AnalyzedFrameMustResolveOneSharedNativeTerminal) {
  Open();
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({}, text, Json(), &prepared, &error));
  BoundTokenPolicy bound;
  bound.text_source = "prior";
  for (const std::vector<std::string> suffixes :
       {std::vector<std::string>{},
        {""},
        {"a"},
        {"<|eot_id|>extra"},
        {"<|eot_id|>", "<|eom_id|>"}}) {
    EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, {}, suffixes,
                                     {}, &bound, &error));
    EXPECT_EQ(bound.text_source, "prior");
  }
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                   {"<|eot_id|>"}, {}, nullptr, &error));
  PreparedTokenBindings empty;
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), empty, text, {}, {"<|eot_id|>"},
                                   {}, &bound, &error));
}
TEST_F(TokenPolicyTest, AnalyzedFrameChoosesLongestSpecialPrefix) {
  Open(false, true, false, {}, false, false, false, false, false,
       "<|eot_id|> ");
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({}, text, Json(), &prepared, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                  {"<|eot_id|> \n"}, {}, &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.id, 6);
  EXPECT_EQ(bound.terminal.piece, "<|eot_id|> ");
}
TEST_F(TokenPolicyTest, AnalyzedTerminalRequiresMatchingNativeIdentity) {
  Open();
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({}, text, Json(true), &prepared, &error));
  BoundTokenPolicy bound;
  bound.terminal.id = 77;
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                   {"<|eot_id|>"}, {}, &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
}
TEST_F(TokenPolicyTest,
       SourcePublicationChecksAllNativeStopsAndSelectedTerminal) {
  Open();
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({}, text, Json(), &prepared, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(FinalizeTokenPolicy(file.get(), prepared, text, facts,
                                  {"<|eot_id|>"}, {}, &bound, &error))
      << error;
  for (size_t i = 0; i < facts->raw_stops.size(); ++i) {
    auto wrong = *facts;
    wrong.raw_stops[i].id = 1;
    EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, wrong,
                                     {"<|eot_id|>"}, {}, &bound, &error));
    EXPECT_NE(error.find("identity mismatch"), std::string::npos);
  }
  auto wrong = *facts;
  wrong.terminal_index = 1;
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, wrong,
                                   {"<|eot_id|>"}, {}, &bound, &error));
  EXPECT_NE(error.find("differs from numerical publication"),
            std::string::npos);
  wrong.terminal_index = 99;
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, wrong,
                                   {"<|eot_id|>"}, {}, &bound, &error));
  EXPECT_NE(error.find("terminal index"), std::string::npos);
}
TEST_F(TokenPolicyTest, OnlyExplicitNativeMarkersAddAtomicObligations) {
  Open();
  PreparedTokenBindings prepared;
  ASSERT_TRUE(
      PrepareTokenBindings({}, text, Json(false, false), &prepared, &error));
  BoundTokenPolicy bound;
  // Tool/reasoning delimiters may consist of ordinary text tokens. They are
  // absent from this list unless an actual native consumer needs atomicity.
  ASSERT_TRUE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                  {"<|eot_id|>"}, {}, &bound, &error))
      << error;
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                   {"<|eot_id|>"}, {"<|header|>"}, &bound,
                                   &error));
  EXPECT_NE(error.find("special piece"), std::string::npos);
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, {},
                                   {"<|eot_id|>"}, {"<missing>"}, &bound,
                                   &error));
  EXPECT_NE(error.find("missing required atomic marker"), std::string::npos);
}
TEST_F(TokenPolicyTest, PublicationUsesSourceIdsDespiteMalformedTemplate) {
  Open();
  ASSERT_FALSE(text.template_issues.empty());
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->raw_stops[0].id, 2);
  EXPECT_EQ(facts->raw_stops[1].id, 3);
  EXPECT_EQ(facts->raw_stops[2].id, 4);
  EXPECT_NE(facts->configuration_source.find("end_message_id = 3 : i32"),
            std::string::npos);
  EXPECT_EQ(facts->terminal_index, 2u);
  EXPECT_EQ(facts->configuration_source,
            "config.def @llama.publication.end_text_id = 2 : i32\n"
            "config.def @llama.publication.end_message_id = 3 : i32\n"
            "config.def @llama.publication.end_turn_id = 4 : i32\n");
}
TEST_F(TokenPolicyTest, SparkSingleStopUsesRelocatedDeclaredEosOnly) {
  Open(false, true, false, u8"<｜end▁of▁sentence｜>");
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(internal::DeriveSourcePublication(
      file.get(), PublicationLayout::kEos, "spark2_5", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 1u);
  EXPECT_EQ(facts->raw_stops.front().id, 4);
  EXPECT_EQ(facts->raw_stops.front().piece, u8"<｜end▁of▁sentence｜>");
  EXPECT_EQ(facts->terminal_index, 0u);
  EXPECT_EQ(facts->configuration_source,
            "config.def @spark2_5.publication.eos_id = 4 : i32\n");
  facts->configuration_source = "prior";
  Open(false, true, false, "unknown");
  EXPECT_FALSE(internal::DeriveSourcePublication(
      file.get(), PublicationLayout::kEos, "spark2_5", text, &facts, &error));
  EXPECT_EQ(facts->configuration_source, "prior");
}
TEST_F(TokenPolicyTest,
       UnspecifiedAndInvalidLayoutsNeverUseHostManagedFallback) {
  for (const auto layout :
       {PublicationLayout::kUnspecified, static_cast<PublicationLayout>(99)}) {
    std::optional<PublicationTokenFacts> facts =
        PublicationTokenFacts{{{77, "prior"}}, 0, "prior"};
    EXPECT_FALSE(internal::DeriveSourcePublication(nullptr, layout, "unused",
                                                   {}, &facts, &error));
    ASSERT_TRUE(facts);
    EXPECT_EQ(facts->raw_stops[0].id, 77);
    EXPECT_EQ(facts->configuration_source, "prior");
    EXPECT_NE(error.find("publication layout"), std::string::npos);
  }
}
TEST_F(TokenPolicyTest, SourceProfileUsesSlotLayoutRatherThanFamilyName) {
  Open();
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(internal::DeriveSourcePublication(
      file.get(), PublicationLayout::kThreeStops, "relocated", text, &facts,
      &error))
      << error;
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->configuration_source,
            "config.def @relocated.publication.end_text_id = 2 : i32\n"
            "config.def @relocated.publication.end_message_id = 3 : i32\n"
            "config.def @relocated.publication.end_turn_id = 4 : i32\n");
  EXPECT_EQ(facts->terminal_index, 2u);
  facts->configuration_source = "prior";
  EXPECT_FALSE(internal::DeriveSourcePublication(
      file.get(), PublicationLayout::kTextAndTurn, "relocated", text, &facts,
      &error));
  EXPECT_EQ(facts->configuration_source, "prior");
}
TEST_F(TokenPolicyTest, UnknownEosAndDuplicateSourcePiecePreservePriorFacts) {
  Open(false, true, false, "<|unknown_stop|>");
  std::optional<PublicationTokenFacts> facts =
      PublicationTokenFacts{{{77, "prior"}}, 0, "prior"};
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_EQ(facts->configuration_source, "prior");
  Open(false, true, false, {}, false, false, false, false, true);
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->raw_stops[0].id, 77);
  EXPECT_EQ(facts->configuration_source, "prior");
  EXPECT_NE(error.find("invalid publication piece <|end_of_text|>"),
            std::string::npos);
}
TEST_F(TokenPolicyTest, PublicationRequiresOutputEvenForHostManagedLayout) {
  EXPECT_FALSE(
      DerivePublicationTokenFacts(nullptr, "qwen35", {}, nullptr, &error));
  EXPECT_NE(error.find("output is required"), std::string::npos);
}
TEST_F(TokenPolicyTest,
       PlainChatmlLlamaSeparatesDeclaredEosAndPublishedTerminal) {
  Open(false, true, false, {}, false, false, false, true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 3u);
  EXPECT_EQ(facts->raw_stops[0].piece, "<|endoftext|>");
  EXPECT_EQ(facts->raw_stops[1].piece, "<|im_end|>");
  EXPECT_EQ(facts->raw_stops[2].piece, "<|endofturn|>");
  EXPECT_EQ(facts->terminal_index, 1u);
  EXPECT_EQ(facts->configuration_source,
            "config.def @llama.publication.end_text_id = 2 : i32\n"
            "config.def @llama.publication.end_message_id = 3 : i32\n"
            "config.def @llama.publication.end_turn_id = 4 : i32\n");
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(
      file.get(), {"bos_token", "eos_token"}, {"<|im_end|>\n"}, text, facts,
      {"<|im_start|>", "<|im_end|>"}, Json(), &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.id, 3);
  EXPECT_EQ(bound.terminal.piece, "<|im_end|>");
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.eos_token_id").id, 4);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.bos_token_id").id, 2);
  EXPECT_EQ(bound.encode_flags, 0u);
  auto wrong_pre = text;
  wrong_pre.pre_tokenizer = "qwen2";
  PreparedTokenBindings source_names;
  ASSERT_TRUE(PrepareTokenBindings({"bos_token", "eos_token"}, wrong_pre,
                                   Json(), &source_names, &error))
      << error;
  EXPECT_EQ(source_names.facts().pre_tokenizer, "qwen2");
  bound.terminal.id = 77;
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<|im_end|>\n"}, text, facts, {}, Json(true),
                                 &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
  auto not_special = Json();
  const auto marker = not_special.find("\"content\":\"<|im_start|>\"");
  ASSERT_NE(marker, std::string::npos);
  const auto special = not_special.find("\"special\":true", marker);
  ASSERT_NE(special, std::string::npos);
  not_special.replace(special, std::string("\"special\":true").size(),
                      "\"special\":false");
  EXPECT_FALSE(BindAnalyzedFrame(
      file.get(), {"bos_token", "eos_token"}, {"<|im_end|>\n"}, text, facts,
      {"<|im_start|>"}, not_special, &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
  auto mismatched = *facts;
  mismatched.terminal_index = 2;
  bound.terminal.id = 77;
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<|im_end|>\n"}, text, mismatched, {}, Json(),
                                 &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
}
TEST_F(TokenPolicyTest,
       PlainChatmlMissingStopOrWrongDeclaredEosPreservesFacts) {
  Open(true, true, false, {}, false, false, false, true);
  std::optional<PublicationTokenFacts> facts =
      PublicationTokenFacts{{{77, "prior"}}, 0, "prior"};
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_EQ(facts->configuration_source, "prior");
  Open(false, true, false, {}, false, false, false, true);
  for (auto &role : text.tokens)
    if (role.source_key == "tokenizer.ggml.eos_token_id")
      role.id = 2;
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_EQ(facts->configuration_source, "prior");
}
TEST_F(TokenPolicyTest, SingleSentenceEosAliasesExistingNumericalSlots) {
  // EOS is deliberately relocated to4; missing unrelated3-stop piece is legal.
  Open(true, true, false, "</s>");
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 3u);
  EXPECT_EQ(facts->terminal_index, 0u);
  for (const auto &stop : facts->raw_stops) {
    EXPECT_EQ(stop.id, 4);
    EXPECT_EQ(stop.piece, "</s>");
  }
  EXPECT_EQ(facts->configuration_source,
            "config.def @llama.publication.end_text_id = 4 : i32\n"
            "config.def @llama.publication.end_message_id = 4 : i32\n"
            "config.def @llama.publication.end_turn_id = 4 : i32\n");
}
TEST_F(TokenPolicyTest, SingleSentenceEosRejectsInvalidOrDuplicateIdentities) {
  std::optional<PublicationTokenFacts> facts =
      PublicationTokenFacts{{{77, "prior"}}, 0, "prior"};
  Open(false, true, false, "</s>");
  text.token_pieces.erase(4);
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_EQ(facts->configuration_source, "prior");
  Open(false, true, false, "</s>");
  for (auto &role : text.tokens)
    if (role.source_key == "tokenizer.ggml.eos_token_id")
      role.id = 2;
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_EQ(facts->configuration_source, "prior");
  Open(false, true, false, "</s>", false, false, false, false, false, {}, true);
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_NE(error.find("invalid publication piece"), std::string::npos)
      << error;
  EXPECT_EQ(facts->configuration_source, "prior");
}
TEST_F(TokenPolicyTest, MissingRequiredPublicationPiecePreservesOutput) {
  Open(true, true, false, {}, false, false, false, false, false, {}, false,
       true);
  std::optional<PublicationTokenFacts> facts = PublicationTokenFacts{};
  facts->raw_stops.resize(3);
  facts->raw_stops[2].id = 99;
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->raw_stops[2].id, 99);
  EXPECT_NE(error.find("<|end_of_text|>"), std::string::npos);
}
TEST_F(TokenPolicyTest, MissingOptionalMessageStopAliasesActualTurnStop) {
  Open(true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 3u);
  EXPECT_EQ(facts->terminal_index, 2u);
  EXPECT_EQ(facts->raw_stops[0].id, 2);
  for (size_t i : {1u, 2u}) {
    EXPECT_EQ(facts->raw_stops[i].id, 4);
    EXPECT_EQ(facts->raw_stops[i].piece, "<|eot_id|>");
  }
  EXPECT_EQ(facts->configuration_source,
            "config.def @llama.publication.end_text_id = 2 : i32\n"
            "config.def @llama.publication.end_message_id = 4 : i32\n"
            "config.def @llama.publication.end_turn_id = 4 : i32\n");
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({}, text, Json(), &prepared, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(FinalizeTokenPolicy(file.get(), prepared, text, facts,
                                  {"<|eot_id|>"}, {}, &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.id, 4);
  auto wrong = *facts;
  wrong.raw_stops[1].id = 3;
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, wrong,
                                   {"<|eot_id|>"}, {}, &bound, &error));
  EXPECT_NE(error.find("identity mismatch"), std::string::npos);
}
TEST_F(TokenPolicyTest, OptionalMessageStopDoesNotMaskSourceObligations) {
  Open(true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  PreparedTokenBindings prepared;
  ASSERT_TRUE(PrepareTokenBindings({}, text, Json(), &prepared, &error));
  BoundTokenPolicy bound;
  bound.text_source = "prior";
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, facts,
                                   {"<|eot_id|>"}, {"<|eom_id|>"}, &bound,
                                   &error));
  EXPECT_NE(error.find("GGUF missing required atomic marker: <|eom_id|>"),
            std::string::npos);
  EXPECT_EQ(bound.text_source, "prior");
  EXPECT_FALSE(FinalizeTokenPolicy(file.get(), prepared, text, facts,
                                   {"<|eom_id|>"}, {}, &bound, &error));
  EXPECT_NE(error.find("no native special terminal"), std::string::npos);
  EXPECT_EQ(bound.text_source, "prior");
}
TEST_F(TokenPolicyTest, OptionalMessageStopDoesNotMaskInvalidSourceIdentities) {
  std::optional<PublicationTokenFacts> facts =
      PublicationTokenFacts{{{77, "prior"}}, 0, "prior"};
  Open(false, true, false, {}, false, false, false, false, false, "<|eom_id|>");
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_NE(error.find("invalid publication piece <|eom_id|>"),
            std::string::npos);
  EXPECT_EQ(facts->configuration_source, "prior");
  Open(true, true, false, {}, false, false, false, false, false, {}, true);
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_NE(error.find("invalid publication piece <|eot_id|>"),
            std::string::npos);
  EXPECT_EQ(facts->configuration_source, "prior");
  // Isolate the missing mandatory EOT; when EOM is also absent the ordered
  // unresolved-slot diagnostic reports that earlier slot first.
  Open(false, true, false, "other_terminal");
  text.token_pieces[4] = "<|eot_id|>";
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_NE(error.find("publication requires <|eot_id|>"), std::string::npos);
  EXPECT_EQ(facts->configuration_source, "prior");
  Open(true);
  for (auto &token : text.tokens)
    if (token.source_key == "tokenizer.ggml.eos_token_id")
      token.id = 3;
  text.token_pieces[3] = "<|eot_id|>";
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_NE(error.find("disagrees with declared EOS"), std::string::npos);
  EXPECT_EQ(facts->configuration_source, "prior");
  Open(true);
  text.tokens.erase(std::remove_if(text.tokens.begin(), text.tokens.end(),
                                   [](const auto &t) {
                                     return t.source_key ==
                                            "tokenizer.ggml.eos_token_id";
                                   }),
                    text.tokens.end());
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  EXPECT_NE(error.find("declared EOS"), std::string::npos);
  EXPECT_EQ(facts->configuration_source, "prior");
}
TEST_F(TokenPolicyTest, UnknownPublicationProviderPreservesOutput) {
  std::optional<PublicationTokenFacts> facts =
      PublicationTokenFacts{{{77, "prior"}}, 0, "prior"};
  EXPECT_FALSE(
      DerivePublicationTokenFacts(nullptr, "unknown", {}, &facts, &error));
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->raw_stops[0].id, 77);
  EXPECT_EQ(facts->configuration_source, "prior");
  EXPECT_NE(error.find("unsupported publication token provider"),
            std::string::npos);
}
TEST_F(TokenPolicyTest, ChatmlPublicationUsesSelectedPrefixAndDeclaredEos) {
  Open(false, true, false, {}, false, true);
  for (const std::string family : {"qwen2", "qwen3"}) {
    std::optional<PublicationTokenFacts> facts;
    ASSERT_TRUE(
        DerivePublicationTokenFacts(file.get(), family, text, &facts, &error))
        << error;
    ASSERT_TRUE(facts);
    EXPECT_EQ(facts->configuration_source,
              "config.def @" + family + ".publication.end_text_id = 2 : i32\n" +
                  "config.def @" + family +
                  ".publication.end_turn_id = 4 : i32\n");
    auto inconsistent = text;
    for (auto &role : inconsistent.tokens)
      if (role.source_key == "tokenizer.ggml.eos_token_id")
        role.id = 2;
    facts->configuration_source = "prior";
    EXPECT_FALSE(DerivePublicationTokenFacts(file.get(), family, inconsistent,
                                             &facts, &error));
    EXPECT_EQ(facts->configuration_source, "prior");
    EXPECT_NE(error.find("declared EOS"), std::string::npos);
  }
}
TEST_F(TokenPolicyTest, QwenNumericalNeedsNoTextData) {
  std::optional<PublicationTokenFacts> facts = PublicationTokenFacts{};
  ASSERT_TRUE(
      DerivePublicationTokenFacts(nullptr, "qwen35", {}, &facts, &error));
  EXPECT_FALSE(facts);
}
TEST_F(TokenPolicyTest, TextStrategyDoesNotRequireNumericalPublication) {
  Open();
  // A numerical consumer that uses no raw-stop mapper contributes no facts.
  // Token/text requirements still follow the selected Llama-header renderer.
  std::optional<PublicationTokenFacts> publication;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "qwen35", text,
                                          &publication, &error));
  ASSERT_FALSE(publication);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                {"<|eot_id|>"}, text, publication, {}, Json(),
                                &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.piece, "<|eot_id|>");
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"eos_token"}, {"<|im_end|>"},
                                 text, publication, {}, Json(), &bound,
                                 &error));
  EXPECT_NE(error.find("native special terminal"), std::string::npos);
}
TEST_F(TokenPolicyTest, EmptyAssistantRoutesDoNotWeakenTerminalBinding) {
  Open();
  const std::string prefix =
      "{% for m in messages %}{% if m.role == 'assistant' %}"
      "{% if emit_target %}{{ '<A>' + m.content + '<|eot_id|>' }}";
  const std::string suffix =
      "{% endif %}{% endif %}{% endfor %}"
      "{% if add_generation_prompt %}{{ '<A>' }}{% endif %}";
  BoundTokenPolicy bound;
  for (bool emitting_alternative : {false, true}) {
    SCOPED_TRACE(emitting_alternative);
    const auto source =
        prefix +
        (emitting_alternative ? "{% else %}{{ '<partial>bad-terminal' }}"
                              : "") +
        suffix;
    jinja::TextFormatAnalysis analysis;
    jinja::Diagnostic diagnostic;
    ASSERT_TRUE(jinja::AnalyzeTextFormat(source, "terminal-choice.jinja", {},
                                         &analysis, &diagnostic))
        << diagnostic.message;
    ASSERT_FALSE(analysis.alternatives.empty());
    std::vector<std::string> frames;
    for (const auto &format : analysis.alternatives)
      frames.push_back(format.frame_suffix);
    if (!emitting_alternative) {
      for (const auto &frame : frames)
        EXPECT_EQ(frame, "<|eot_id|>");
      ASSERT_TRUE(BindAnalyzedFrame(file.get(), {}, frames, text, {}, {},
                                    Json(), &bound, &error))
          << error;
      EXPECT_EQ(bound.terminal.piece, "<|eot_id|>");
    } else {
      ASSERT_TRUE(
          std::any_of(frames.begin(), frames.end(), [](const auto &frame) {
            return frame.find("bad-terminal") != std::string::npos;
          }));
      const auto prior = bound;
      EXPECT_FALSE(BindAnalyzedFrame(file.get(), {}, frames, text, {}, {},
                                     Json(), &bound, &error));
      EXPECT_NE(error.find("native special terminal"), std::string::npos);
      EXPECT_EQ(bound.terminal.id, prior.terminal.id);
      EXPECT_EQ(bound.terminal.piece, prior.terminal.piece);
      EXPECT_EQ(bound.text_source, prior.text_source);
    }
  }
}
TEST_F(TokenPolicyTest, TextTerminalGuardAppliesWithoutNumericalPublication) {
  // GGUF and JSON agree on the ID/piece, but the selected header protocol
  // requires eot rather than this otherwise valid ChatML terminal.
  Open(false, true, false, "<|im_end|>");
  BoundTokenPolicy bound;
  bound.terminal.id = 77;
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<|eot_id|>"}, text, {}, {}, Json(), &bound,
                                 &error));
  EXPECT_NE(error.find("native special terminal"), std::string::npos);
  EXPECT_EQ(bound.terminal.id, 77);
}
TEST_F(TokenPolicyTest, OptionalPublicationIsValidatedForChatmlStrategy) {
  Open(false, true, true);
  PublicationTokenFacts facts{
      {{2, pieces[2]}, {3, pieces[3]}, {4, pieces[4]}}, 2, ""};
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(file.get(), {"eos_token"}, {"<|im_end|>"}, text,
                                facts, {}, Json(), &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.piece, "<|im_end|>");
  for (int ordinal = 0; ordinal < 3; ++ordinal) {
    auto bad = facts;
    TokenIdentity *identities[] = {&bad.raw_stops[0], &bad.raw_stops[1],
                                   &bad.raw_stops[2]};
    identities[ordinal]->id = 1;
    bound.terminal.id = 77;
    EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"eos_token"}, {"<|im_end|>"},
                                   text, bad, {}, Json(), &bound, &error));
    EXPECT_NE(error.find("identity mismatch"), std::string::npos);
    EXPECT_EQ(bound.terminal.id, 77);
  }
  auto mismatched = facts;
  mismatched.raw_stops[2] = {3, pieces[3]};
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"eos_token"}, {"<|im_end|>"},
                                 text, mismatched, {}, Json(), &bound, &error));
  EXPECT_NE(error.find("differs from numerical publication"),
            std::string::npos);
}
TEST_F(TokenPolicyTest, NativeBindingRecoversUndeclaredAtomicMarker) {
  Open();
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(
      file.get(), {"bos_token", "eos_token"}, {"<|eot_id|>"}, text, facts,
      {"<|header|>", "<|header|>"}, Json(), &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.id, 4);
  EXPECT_EQ(bound.terminal.piece, "<|eot_id|>");
  EXPECT_EQ(bound.encode_flags, 0u);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.bos_token_id").id, 1);
  EXPECT_NE(bound.text_source.find("3c7c656f745f69647c3e"), std::string::npos);
  PreparedTokenBindings globals;
  ASSERT_TRUE(
      PrepareTokenBindings({"bos_token"}, text, Json(), &globals, &error))
      << error;
  EXPECT_EQ(globals.globals().at("bos_token"), "<|begin_of_text|>");
  EXPECT_EQ(bound.text_source.find("@loomify.tokens.bos_marker"),
            std::string::npos);
  EXPECT_NE(bound.text_source.find("@loomify.tokens.vocabulary_size"),
            std::string::npos);
}
TEST_F(TokenPolicyTest, NativeBindingRejectsWrongIdAndNonspecialMarker) {
  Open();
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound;
  bound.terminal.id = 77;
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<|eot_id|>"}, text, facts, {}, Json(true),
                                 &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<|eot_id|>"}, text, facts, {"<|header|>"},
                                 Json(false, false), &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
}
TEST_F(TokenPolicyTest, UnusedDeclaredRoleIsPreservedWithoutNativeAdmission) {
  Open();
  text.tokens.push_back({"tokenizer.ggml.padding_token_id", 4, 5});
  text.token_pieces.emplace(5, "<|header|>");
  text.token_issues.push_back(
      {"tokenizer.ggml.mask_token_id", "malformed unused role"});
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  auto json = Json();
  for (size_t pos = 0;
       (pos = json.find("<|header|>", pos)) != std::string::npos;) {
    json.replace(pos, 10, "<|other|>");
    pos += 9;
  }
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                {"<|eot_id|>"}, text, facts, {}, json, &bound,
                                &error))
      << error;
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.padding_token_id").piece,
            "<|header|>");
  EXPECT_EQ(bound.terminal.id, 4);
}
TEST_F(TokenPolicyTest, RenderedBosIsEncodedOnceDespiteJsonPostprocessor) {
  Open();
  auto json = Json();
  json.pop_back();
  json +=
      R"(,"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<|begin_of_text|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":0}}],"special_tokens":{"<|begin_of_text|>":{"id":"<|begin_of_text|>","ids":[1],"tokens":["<|begin_of_text|>"]}}}})";
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                {"<|eot_id|>"}, text, facts, {}, json, &bound,
                                &error))
      << error;
  iree_tokenizer_t *raw = nullptr;
  auto status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(),
      &raw);
  ASSERT_TRUE(iree_status_is_ok(status));
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(
      raw, iree_tokenizer_free);
  int32_t ids[8] = {};
  iree_host_size_t count = 0;
  status = iree_tokenizer_encode(
      tokenizer.get(), iree_make_cstring_view("<|begin_of_text|>a"),
      bound.encode_flags,
      iree_tokenizer_make_token_output(ids, nullptr, nullptr, 8),
      iree_allocator_system(), &count);
  ASSERT_TRUE(iree_status_is_ok(status));
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(ids[0], 1);
  EXPECT_EQ(ids[1], 0);
  // The same already-rendered bytes with automatic insertion would duplicate
  // BOS.
  status = iree_tokenizer_encode(
      tokenizer.get(), iree_make_cstring_view("<|begin_of_text|>a"),
      IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      iree_tokenizer_make_token_output(ids, nullptr, nullptr, 8),
      iree_allocator_system(), &count);
  ASSERT_TRUE(iree_status_is_ok(status));
  ASSERT_EQ(count, 3u);
  EXPECT_EQ(ids[0], 1);
  EXPECT_EQ(ids[1], 1);
  EXPECT_EQ(ids[2], 0);
}
TEST_F(TokenPolicyTest, GemmaSourceStopsAreIndependentOfChatAndCheckpointIds) {
  Open(false, true, false, {}, true);
  ASSERT_FALSE(text.template_issues.empty());
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "gemma3", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 2u);
  EXPECT_EQ(facts->raw_stops[0].id, 2);
  EXPECT_EQ(facts->raw_stops[0].piece, "<eos>");
  EXPECT_EQ(facts->terminal_index, 1u);
  EXPECT_EQ(facts->raw_stops[1].id, 6);
  EXPECT_NE(
      facts->configuration_source.find("@gemma3.publication.eos_id = 2 : i32"),
      std::string::npos);
  EXPECT_NE(facts->configuration_source.find(
                "@gemma3.publication.end_turn_id = 6 : i32"),
            std::string::npos);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(
      file.get(), {"bos_token", "eos_token"}, {"<end_of_turn>\n"}, text, facts,
      {"<start_of_turn>", "<end_of_turn>"}, Json(), &bound, &error))
      << error;
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.eos_token_id").id, 2);
  EXPECT_EQ(bound.terminal.id, 6);
  EXPECT_EQ(bound.encode_flags, 0u);
  PreparedTokenBindings globals;
  ASSERT_TRUE(
      PrepareTokenBindings({"bos_token"}, text, Json(), &globals, &error))
      << error;
  EXPECT_EQ(globals.globals().at("bos_token"), "<bos>");
  EXPECT_EQ(bound.text_source.find("@loomify.tokens.bos_marker"),
            std::string::npos);
  EXPECT_NE(bound.text_source.find("@loomify.tokens.vocabulary_size"),
            std::string::npos);
  auto invalid = *facts;
  invalid.terminal_index = 99;
  bound.terminal.id = 77;
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<end_of_turn>\n"}, text, invalid, {}, Json(),
                                 &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
}
TEST_F(TokenPolicyTest, GemmaMissingTurnStopRejectsWithoutReplacingFacts) {
  Open(true, true, false, {}, true);
  std::optional<PublicationTokenFacts> facts =
      PublicationTokenFacts{{{77, "prior"}}, 0, "prior"};
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "gemma3", text, &facts, &error));
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->raw_stops[0].id, 77);
  EXPECT_NE(error.find("<end_of_turn>"), std::string::npos);
}
TEST_F(TokenPolicyTest, RoleBoundEosCanShareTheTurnSlotWithoutDeduplication) {
  Open(false, true, false, {}, true);
  for (auto &role : text.tokens)
    if (role.source_key == "tokenizer.ggml.eos_token_id")
      role.id = 6;
  text.token_pieces[6] = "<end_of_turn>";
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "gemma3", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 2u);
  EXPECT_EQ(facts->raw_stops[0].id, 6);
  EXPECT_EQ(facts->raw_stops[1].id, 6);
  EXPECT_EQ(facts->terminal_index, 1u);
  EXPECT_EQ(facts->configuration_source,
            "config.def @gemma3.publication.eos_id = 6 : i32\n"
            "config.def @gemma3.publication.end_turn_id = 6 : i32\n");
}
TEST_F(TokenPolicyTest, Qwen3SourcePolicyDerivesBothEogIdsWithoutChat) {
  Open(false, true, false, {}, false, true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "qwen3", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 2u);
  EXPECT_EQ(facts->raw_stops[0].id, 2);
  EXPECT_EQ(facts->raw_stops[0].piece, "<|endoftext|>");
  EXPECT_EQ(facts->raw_stops[1].id, 4);
  EXPECT_EQ(facts->terminal_index, 1u);
  EXPECT_NE(
      facts->configuration_source.find("@qwen3.publication.end_text_id = 2"),
      std::string::npos);
  EXPECT_NE(
      facts->configuration_source.find("@qwen3.publication.end_turn_id = 4"),
      std::string::npos);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(
      file.get(), {"bos_token", "eos_token"}, {"<|im_end|>\n"}, text, facts,
      {"<|im_start|>", "<|im_end|>"}, Json(), &bound, &error))
      << error;
  EXPECT_EQ(bound.terminal.id, 4);
  EXPECT_EQ(bound.encode_flags, 0u);
}
TEST_F(TokenPolicyTest, Qwen2StopsAndDeclaredBosDoNotInsertBos) {
  Open(false, true, false, {}, false, false, true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "qwen2", text, &facts, &error))
      << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 2u);
  EXPECT_EQ(facts->raw_stops[0].piece, "<|endoftext|>");
  EXPECT_EQ(facts->raw_stops[1].piece, "<|im_end|>");
  EXPECT_EQ(facts->terminal_index, 1u);
  EXPECT_NE(
      facts->configuration_source.find("@qwen2.publication.end_text_id = 2"),
      std::string::npos);
  EXPECT_NE(
      facts->configuration_source.find("@qwen2.publication.end_turn_id = 4"),
      std::string::npos);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindAnalyzedFrame(
      file.get(), {"bos_token", "eos_token"}, {"<|im_end|>\n"}, text, facts,
      {"<|im_start|>", "<|im_end|>"}, Json(), &bound, &error))
      << error;
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.bos_token_id").id, 2);
  EXPECT_EQ(bound.terminal.id, 4);
  EXPECT_EQ(bound.encode_flags, 0u);
  EXPECT_EQ(bound.text_source.find("@loomify.tokens.bos_marker"),
            std::string::npos);
  EXPECT_NE(bound.text_source.find("@loomify.tokens.vocabulary_size"),
            std::string::npos);
}
TEST_F(TokenPolicyTest, Qwen3MissingAdditionalStopDoesNotReplaceFacts) {
  Open(true, true, false, {}, false, true);
  std::optional<PublicationTokenFacts> facts = PublicationTokenFacts{};
  facts->configuration_source = "unchanged";
  EXPECT_FALSE(
      DerivePublicationTokenFacts(file.get(), "qwen3", text, &facts, &error));
  EXPECT_NE(error.find("<|endoftext|>"), std::string::npos);
  EXPECT_EQ(facts->configuration_source, "unchanged");
}

TEST_F(TokenPolicyTest, MissingMarkerAndMalformedJsonAreDiagnosed) {
  Open();
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(
      DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound;
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<|eot_id|>"}, text, facts, {"<|missing|>"},
                                 Json(), &bound, &error));
  EXPECT_NE(error.find("<|missing|>"), std::string::npos);
  EXPECT_FALSE(BindAnalyzedFrame(file.get(), {"bos_token", "eos_token"},
                                 {"<|eot_id|>"}, text, facts, {}, "{broken",
                                 &bound, &error));
  EXPECT_NE(error.find("native tokenizer JSON"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf
