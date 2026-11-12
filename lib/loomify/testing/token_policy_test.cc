// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/token_policy.h"
#include "tools/model/text_metadata.h"
#include "gtest/gtest.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include <memory>
namespace hrx::loomify_gguf {
namespace {
void Number(std::string& s, uint64_t n, int width) {
  for (int i = 0; i < width; ++i) s += static_cast<char>(n >> (8 * i));
}
void String(std::string& s, std::string_view v) {
  Number(s, v.size(), 8); s += v;
}
TextTokenRequirements LlamaTokens() { return {"gpt2","llama-bpe",true,"<|begin_of_text|>","<|eot_id|>"}; }
TextTokenRequirements ChatmlTokens() { return {"gpt2","qwen35",false,"","<|im_end|>"}; }
class TokenPolicyTest : public ::testing::Test {
protected:
  void Open(bool missing_stop = false, bool bad_template = true,
            bool chatml = false, std::string terminal = {}, bool gemma = false, bool qwen3 = false, bool qwen2 = false) {
    pieces = {"a", "<|begin_of_text|>", "<|end_of_text|>",
              missing_stop ? "other" : "<|eom_id|>",
              terminal.empty() ? (chatml ? "<|im_end|>" : "<|eot_id|>") : terminal,
              "<|header|>"};
    if (gemma) pieces={"a","<pad>","<eos>","<bos>","<unk>","<start_of_turn>",missing_stop ? "other" : "<end_of_turn>"};
    if (qwen3 || qwen2) pieces={"a","<|im_start|>",missing_stop ? "other" : "<|endoftext|>","<think>","<|im_end|>","</think>"};
    std::string bytes = "GGUF";
    Number(bytes, 3, 4); Number(bytes, 0, 8); Number(bytes, 7, 8);
    auto field = [&](std::string_view k, std::string_view v) {
      String(bytes, k); Number(bytes, 8, 4); String(bytes, v);
    };
    field("general.architecture", qwen2 ? "qwen2" : qwen3 ? "qwen3" : gemma ? "gemma3" : "llama");
    field("tokenizer.ggml.model", gemma ? "llama" : "gpt2");
    field("tokenizer.ggml.pre", (qwen3 || qwen2) ? "qwen2" : gemma ? "default" : chatml ? "qwen35" : "llama-bpe");
    String(bytes, "tokenizer.ggml.tokens"); Number(bytes, 9, 4);
    Number(bytes, 8, 4); Number(bytes, pieces.size(), 8);
    for (const auto& p : pieces) String(bytes, p);
    String(bytes, "tokenizer.ggml.eos_token_id"); Number(bytes, 4, 4); Number(bytes, gemma ? 2 : 4, 4);
    String(bytes, "tokenizer.ggml.bos_token_id"); Number(bytes, 4, 4); Number(bytes, qwen2 ? 2 : gemma ? 3 : 1, 4);
    String(bytes, "tokenizer.chat_template");
    if (bad_template) { Number(bytes, 4, 4); Number(bytes, 42, 4); }
    else { Number(bytes, 8, 4); String(bytes, "unknown template"); }
    while (bytes.size() % 32) bytes += '\0';
    file.reset(std::tmpfile()); ASSERT_TRUE(file);
    ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
    ASSERT_EQ(std::fwrite(bytes.data(), 1, bytes.size(), file.get()), bytes.size());
    model::GgufCatalog catalog;
    ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
    ASSERT_TRUE(model::ReadTextMetadata(file.get(), catalog, &text, &error)) << error;
  }
  std::string Json(bool wrong_eos = false, bool marker_special = true) {
    std::string json = R"({"version":"1.0","model":{"type":"BPE","vocab":{)";
    for (size_t i = 0; i < pieces.size(); ++i) {
      if (i) json += ',';
      json += "\"" + (wrong_eos && i == 4 ? "wrong" : pieces[i]) + "\":" + std::to_string(i);
    }
    json += R"(},"merges":[]},"added_tokens":[)";
    for (size_t i = 1; i < pieces.size(); ++i) {
      if (i != 1) json += ',';
      json += "{\"id\":" + std::to_string(i) + ",\"content\":\"" +
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
TEST_F(TokenPolicyTest, PublicationUsesSourceIdsDespiteMalformedTemplate) {
  Open(); ASSERT_FALSE(text.template_issues.empty());
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error)) << error;
  ASSERT_TRUE(facts);
  EXPECT_EQ(facts->raw_stops[0].id, 2); EXPECT_EQ(facts->raw_stops[1].id, 3); EXPECT_EQ(facts->raw_stops[2].id, 4);
  EXPECT_NE(facts->configuration_source.find("end_message_id = 3 : i32"), std::string::npos);
}
TEST_F(TokenPolicyTest, MissingRequiredPublicationPiecePreservesOutput) {
  Open(true); std::optional<PublicationTokenFacts> facts = PublicationTokenFacts{};
  facts->raw_stops.resize(3); facts->raw_stops[2].id = 99;
  EXPECT_FALSE(DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  ASSERT_TRUE(facts); EXPECT_EQ(facts->raw_stops[2].id, 99);
  EXPECT_NE(error.find("<|eom_id|>"), std::string::npos);
}
TEST_F(TokenPolicyTest, QwenNumericalNeedsNoTextData) {
  std::optional<PublicationTokenFacts> facts = PublicationTokenFacts{};
  ASSERT_TRUE(DerivePublicationTokenFacts(nullptr, "qwen35", {}, &facts, &error));
  EXPECT_FALSE(facts);
}
TEST_F(TokenPolicyTest, TextStrategyDoesNotRequireNumericalPublication) {
  Open();
  // A numerical consumer that uses no raw-stop mapper contributes no facts.
  // Token/text requirements still follow the selected Llama-header renderer.
  std::optional<PublicationTokenFacts> publication;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "qwen35", text, &publication, &error));
  ASSERT_FALSE(publication);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), LlamaTokens(), text, publication, {}, Json(), &bound, &error)) << error;
  EXPECT_EQ(bound.terminal.piece, "<|eot_id|>");
  EXPECT_FALSE(BindTokenPolicy(file.get(), ChatmlTokens(), text, publication, {}, Json(), &bound, &error));
  EXPECT_NE(error.find("model/pre"),std::string::npos);
}
TEST_F(TokenPolicyTest, TextTerminalGuardAppliesWithoutNumericalPublication) {
  // GGUF and JSON agree on the ID/piece, but the selected header protocol
  // requires eot rather than this otherwise valid ChatML terminal.
  Open(false,true,false,"<|im_end|>");
  BoundTokenPolicy bound; bound.terminal.id=77;
  EXPECT_FALSE(BindTokenPolicy(file.get(), LlamaTokens(), text, {}, {}, Json(), &bound, &error));
  EXPECT_NE(error.find("requires EOS <|eot_id|>"),std::string::npos);
  EXPECT_EQ(bound.terminal.id,77);
}
TEST_F(TokenPolicyTest, OptionalPublicationIsValidatedForChatmlStrategy) {
  Open(false,true,true);
  PublicationTokenFacts facts{{{2,pieces[2]},{3,pieces[3]},{4,pieces[4]}},2,""};
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), ChatmlTokens(), text, facts, {}, Json(), &bound, &error)) << error;
  EXPECT_EQ(bound.terminal.piece,"<|im_end|>");
  for (int ordinal=0;ordinal<3;++ordinal) {
    auto bad=facts;
    TokenIdentity* identities[]={&bad.raw_stops[0],&bad.raw_stops[1],&bad.raw_stops[2]};
    identities[ordinal]->id=1;
    bound.terminal.id=77;
    EXPECT_FALSE(BindTokenPolicy(file.get(), ChatmlTokens(), text, bad, {}, Json(), &bound, &error));
    EXPECT_NE(error.find("identity mismatch"),std::string::npos);
    EXPECT_EQ(bound.terminal.id,77);
  }
  auto mismatched=facts; mismatched.raw_stops[2]={3,pieces[3]};
  EXPECT_FALSE(BindTokenPolicy(file.get(), ChatmlTokens(), text, mismatched, {}, Json(), &bound, &error));
  EXPECT_NE(error.find("differs from numerical publication"),std::string::npos);
}
TEST_F(TokenPolicyTest, NativeBindingRecoversUndeclaredAtomicMarker) {
  Open(); std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), LlamaTokens(), text, facts, {"<|header|>", "<|header|>"}, Json(), &bound, &error)) << error;
  EXPECT_EQ(bound.terminal.id, 4); EXPECT_EQ(bound.terminal.piece, "<|eot_id|>");
  EXPECT_EQ(bound.encode_flags, 0u);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.bos_token_id").id, 1);
  EXPECT_NE(bound.text_source.find("3c7c656f745f69647c3e"), std::string::npos);
}
TEST_F(TokenPolicyTest, NativeBindingRejectsWrongIdAndNonspecialMarker) {
  Open(); std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound; bound.terminal.id = 77;
  EXPECT_FALSE(BindTokenPolicy(file.get(), LlamaTokens(), text, facts, {}, Json(true), &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
  EXPECT_FALSE(BindTokenPolicy(file.get(), LlamaTokens(), text, facts, {"<|header|>"}, Json(false, false), &bound, &error));
  EXPECT_EQ(bound.terminal.id, 77);
}
TEST_F(TokenPolicyTest, UnusedDeclaredRoleIsPreservedWithoutNativeAdmission) {
  Open();
  text.tokens.push_back({"tokenizer.ggml.padding_token_id", 4, 5});
  text.token_pieces.emplace(5, "<|header|>");
  text.token_issues.push_back({"tokenizer.ggml.mask_token_id", "malformed unused role"});
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  auto json = Json();
  for (size_t pos = 0; (pos = json.find("<|header|>", pos)) != std::string::npos; ) {
    json.replace(pos, 10, "<|other|>"); pos += 9;
  }
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), LlamaTokens(), text, facts, {}, json, &bound, &error)) << error;
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.padding_token_id").piece, "<|header|>");
  EXPECT_EQ(bound.terminal.id, 4);
}
TEST_F(TokenPolicyTest, RenderedBosIsEncodedOnceDespiteJsonPostprocessor) {
  Open();
  auto json = Json(); json.pop_back();
  json += R"(,"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<|begin_of_text|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":0}}],"special_tokens":{"<|begin_of_text|>":{"id":"<|begin_of_text|>","ids":[1],"tokens":["<|begin_of_text|>"]}}}})";
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), LlamaTokens(), text, facts, {}, json, &bound, &error)) << error;
  iree_tokenizer_t* raw = nullptr;
  auto status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(), &raw);
  ASSERT_TRUE(iree_status_is_ok(status));
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(raw, iree_tokenizer_free);
  int32_t ids[8] = {}; iree_host_size_t count = 0;
  status = iree_tokenizer_encode(tokenizer.get(), iree_make_cstring_view("<|begin_of_text|>a"),
      bound.encode_flags, iree_tokenizer_make_token_output(ids, nullptr, nullptr, 8),
      iree_allocator_system(), &count);
  ASSERT_TRUE(iree_status_is_ok(status));
  ASSERT_EQ(count, 2u); EXPECT_EQ(ids[0], 1); EXPECT_EQ(ids[1], 0);
  // The same already-rendered bytes with automatic insertion would duplicate BOS.
  status = iree_tokenizer_encode(tokenizer.get(), iree_make_cstring_view("<|begin_of_text|>a"),
      IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      iree_tokenizer_make_token_output(ids, nullptr, nullptr, 8), iree_allocator_system(), &count);
  ASSERT_TRUE(iree_status_is_ok(status));
  ASSERT_EQ(count, 3u); EXPECT_EQ(ids[0], 1); EXPECT_EQ(ids[1], 1); EXPECT_EQ(ids[2], 0);
}
TEST_F(TokenPolicyTest, GemmaSourceStopsAreIndependentOfChatAndCheckpointIds) {
  Open(false,true,false,{},true);
  ASSERT_FALSE(text.template_issues.empty());
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(),"gemma3",text,&facts,&error)) << error;
  ASSERT_TRUE(facts); ASSERT_EQ(facts->raw_stops.size(),2u);
  EXPECT_EQ(facts->raw_stops[0].id,2); EXPECT_EQ(facts->raw_stops[0].piece,"<eos>");
  EXPECT_EQ(facts->terminal_index,1u); EXPECT_EQ(facts->raw_stops[1].id,6);
  EXPECT_NE(facts->configuration_source.find("@gemma3.publication.eos_id = 2 : i32"),std::string::npos);
  EXPECT_NE(facts->configuration_source.find("@gemma3.publication.end_turn_id = 6 : i32"),std::string::npos);
  TextTokenRequirements requirements{"llama","default",true,"","<end_of_turn>",false};
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(),requirements,text,facts,{"<start_of_turn>","<end_of_turn>"},Json(),&bound,&error)) << error;
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.eos_token_id").id,2);
  EXPECT_EQ(bound.terminal.id,6); EXPECT_EQ(bound.encode_flags,0u);
  EXPECT_NE(bound.text_source.find("@loomify.tokens.bos_marker"),std::string::npos);
  EXPECT_NE(bound.text_source.find("3c626f733e"),std::string::npos);
  auto invalid=*facts; invalid.terminal_index=99;
  bound.terminal.id=77;
  EXPECT_FALSE(BindTokenPolicy(file.get(),requirements,text,invalid,{},Json(),&bound,&error));
  EXPECT_EQ(bound.terminal.id,77);
}
TEST_F(TokenPolicyTest, GemmaMissingTurnStopRejectsWithoutReplacingFacts) {
  Open(true,true,false,{},true);
  std::optional<PublicationTokenFacts> facts=PublicationTokenFacts{{{77,"prior"}},0,"prior"};
  EXPECT_FALSE(DerivePublicationTokenFacts(file.get(),"gemma3",text,&facts,&error));
  ASSERT_TRUE(facts); EXPECT_EQ(facts->raw_stops[0].id,77);
  EXPECT_NE(error.find("<end_of_turn>"),std::string::npos);
}
TEST_F(TokenPolicyTest, Qwen3SourcePolicyDerivesBothEogIdsWithoutChat) {
  Open(false, true, false, {}, false, true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "qwen3", text, &facts, &error)) << error;
  ASSERT_TRUE(facts);
  ASSERT_EQ(facts->raw_stops.size(), 2u);
  EXPECT_EQ(facts->raw_stops[0].id, 2);
  EXPECT_EQ(facts->raw_stops[0].piece, "<|endoftext|>");
  EXPECT_EQ(facts->raw_stops[1].id, 4);
  EXPECT_EQ(facts->terminal_index, 1u);
  EXPECT_NE(facts->configuration_source.find("@qwen3.publication.end_text_id = 2"), std::string::npos);
  EXPECT_NE(facts->configuration_source.find("@qwen3.publication.end_turn_id = 4"), std::string::npos);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), {"gpt2","qwen2",false,"","<|im_end|>"},
      text, facts, {"<|im_start|>","<|im_end|>"}, Json(), &bound, &error)) << error;
  EXPECT_EQ(bound.terminal.id, 4);
  EXPECT_EQ(bound.encode_flags, 0u);
}
TEST_F(TokenPolicyTest, Qwen2StopsAndDeclaredBosDoNotInsertBos) {
  Open(false, true, false, {}, false, false, true);
  std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "qwen2", text, &facts, &error)) << error;
  ASSERT_TRUE(facts); ASSERT_EQ(facts->raw_stops.size(), 2u);
  EXPECT_EQ(facts->raw_stops[0].piece, "<|endoftext|>");
  EXPECT_EQ(facts->raw_stops[1].piece, "<|im_end|>");
  EXPECT_EQ(facts->terminal_index, 1u);
  EXPECT_NE(facts->configuration_source.find("@qwen2.publication.end_text_id = 2"), std::string::npos);
  EXPECT_NE(facts->configuration_source.find("@qwen2.publication.end_turn_id = 4"), std::string::npos);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), {"gpt2","qwen2",false,"","<|im_end|>"},
      text, facts, {"<|im_start|>","<|im_end|>"}, Json(), &bound, &error)) << error;
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.bos_token_id").id, 2);
  EXPECT_EQ(bound.terminal.id, 4); EXPECT_EQ(bound.encode_flags, 0u);
  EXPECT_EQ(bound.text_source.find("@loomify.tokens.bos_marker"), std::string::npos);
}
TEST_F(TokenPolicyTest, Qwen3MissingAdditionalStopDoesNotReplaceFacts) {
  Open(true, true, false, {}, false, true);
  std::optional<PublicationTokenFacts> facts = PublicationTokenFacts{};
  facts->configuration_source = "unchanged";
  EXPECT_FALSE(DerivePublicationTokenFacts(file.get(), "qwen3", text, &facts, &error));
  EXPECT_NE(error.find("<|endoftext|>"), std::string::npos);
  EXPECT_EQ(facts->configuration_source, "unchanged");
}

TEST_F(TokenPolicyTest, MissingMarkerAndMalformedJsonAreDiagnosed) {
  Open(); std::optional<PublicationTokenFacts> facts;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "llama", text, &facts, &error));
  BoundTokenPolicy bound;
  EXPECT_FALSE(BindTokenPolicy(file.get(), LlamaTokens(), text, facts, {"<|missing|>"}, Json(), &bound, &error));
  EXPECT_NE(error.find("<|missing|>"), std::string::npos);
  EXPECT_FALSE(BindTokenPolicy(file.get(), LlamaTokens(), text, facts, {}, "{broken", &bound, &error));
  EXPECT_NE(error.find("native tokenizer JSON"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf
