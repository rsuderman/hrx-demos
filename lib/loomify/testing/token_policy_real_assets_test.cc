// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/token_policy.h"
#include "tools/model/text_metadata.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cstddef>
#include <vector>
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include <fstream>
#include <iterator>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
TEST(TokenPolicyRealAssets, BindsActualLlamaAndQwenTokenizers) {
  for (bool llama : {true, false}) {
    const char* asset = std::getenv(llama ? "LOOMIFY_TEST_GGUF" : "LOOMIFY_TEST_QWEN_GGUF");
    const char* json_path = std::getenv(llama ? "LOOMIFY_TEST_LLAMA_TOKENIZER" : "LOOMIFY_TEST_QWEN_TOKENIZER");
    ASSERT_TRUE(asset && json_path) << "explicit immutable GGUF/tokenizer paths required";
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(asset, "rb"), std::fclose);
    ASSERT_TRUE(file); ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
    model::GgufCatalog catalog; model::TextMetadata text; std::string error;
    ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
    ASSERT_TRUE(model::ReadTextMetadata(file.get(), catalog, &text, &error)) << error;
    std::ifstream stream(json_path, std::ios::binary); ASSERT_TRUE(stream);
    std::string json{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(stream.bad()); ASSERT_FALSE(json.empty());
    const std::string family = llama ? "llama" : "qwen35";
    std::optional<PublicationTokenFacts> publication;
    ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), family, text, &publication, &error)) << error;
    EXPECT_EQ(publication.has_value(), llama);
    BoundTokenPolicy bound;
    const std::vector<std::string> markers = llama
        ? std::vector<std::string>{"<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}
        : std::vector<std::string>{"<|im_start|>", "<|im_end|>"};
    ASSERT_TRUE(BindTokenPolicy(file.get(), !llama ? TextTokenRequirements{"gpt2","qwen35",false,"","<|im_end|>"} : TextTokenRequirements{"gpt2","llama-bpe",true,"<|begin_of_text|>","<|eot_id|>"}, text, publication, markers, json, &bound, &error)) << error;
    EXPECT_EQ(bound.terminal.piece, llama ? "<|eot_id|>" : "<|im_end|>");
    EXPECT_EQ(bound.encode_flags, 0u);
    if (llama) {
      EXPECT_EQ(bound.terminal.id, 128009);
      EXPECT_EQ(publication->raw_stops[0].id, 128001);
      EXPECT_EQ(publication->raw_stops[1].id, 128008);
    } else {
      EXPECT_EQ(bound.terminal.id, 248046);
      EXPECT_EQ(bound.roles.at("tokenizer.ggml.padding_token_id").id, 248055);
    }
  }
}
TEST(TokenPolicyRealAssets, BindsActualGemmaSourceStopsAndNativeTokenizer) {
  const char* asset = std::getenv("LOOMIFY_TEST_GEMMA_GGUF");
  const char* json_path = std::getenv("LOOMIFY_TEST_GEMMA_TOKENIZER");
  ASSERT_TRUE(asset && json_path) << "explicit immutable Gemma GGUF/tokenizer required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(asset, "rb"), std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::GgufCatalog catalog; model::TextMetadata text; std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  ASSERT_TRUE(model::ReadTextMetadata(file.get(), catalog, &text, &error)) << error;
  std::ifstream stream(json_path, std::ios::binary); ASSERT_TRUE(stream);
  std::string json{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  ASSERT_FALSE(stream.bad()); ASSERT_FALSE(json.empty());
  std::optional<PublicationTokenFacts> publication;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "gemma3", text, &publication, &error)) << error;
  ASSERT_TRUE(publication); ASSERT_EQ(publication->raw_stops.size(), 2u);
  EXPECT_EQ(publication->raw_stops[0].id, 1);
  EXPECT_EQ(publication->raw_stops[0].piece, "<eos>");
  EXPECT_EQ(publication->raw_stops[1].id, 106);
  EXPECT_EQ(publication->raw_stops[1].piece, "<end_of_turn>");
  EXPECT_EQ(publication->terminal_index, 1u);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(),
      TextTokenRequirements{"llama", "default", true, "", "<end_of_turn>", false},
      text, publication, {"<start_of_turn>", "<end_of_turn>"}, json, &bound, &error)) << error;
  EXPECT_EQ(bound.terminal.id, 106); EXPECT_EQ(bound.encode_flags, 0u);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.bos_token_id").id, 2);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.padding_token_id").id, 0);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.unknown_token_id").id, 3);
  EXPECT_NE(bound.text_source.find("@loomify.tokens.bos_marker"), std::string::npos);
  // JSON contains added media ID262144 outside model vocabulary: it is not a
  // consumed ordinary-text marker or raw-stop identity and must not block binding.
}
TEST(TokenPolicyRealAssets, BindsActualQwen3SourceStopsWithoutBos) {
  const char* asset = std::getenv("LOOMIFY_TEST_QWEN3_GGUF");
  const char* json_path = std::getenv("LOOMIFY_TEST_QWEN3_TOKENIZER");
  ASSERT_TRUE(asset && json_path) << "explicit immutable Qwen3 GGUF/tokenizer required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(asset, "rb"), std::fclose);
  ASSERT_TRUE(file); ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::GgufCatalog catalog; model::TextMetadata text; std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  ASSERT_TRUE(model::ReadTextMetadata(file.get(), catalog, &text, &error)) << error;
  std::ifstream stream(json_path, std::ios::binary); ASSERT_TRUE(stream);
  std::string json{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  std::optional<PublicationTokenFacts> publication;
  ASSERT_TRUE(DerivePublicationTokenFacts(file.get(), "qwen3", text, &publication, &error)) << error;
  ASSERT_TRUE(publication); ASSERT_EQ(publication->raw_stops.size(), 2u);
  EXPECT_EQ(publication->raw_stops[0].id, 151643);
  EXPECT_EQ(publication->raw_stops[1].id, 151645);
  EXPECT_EQ(publication->terminal_index, 1u);
  BoundTokenPolicy bound;
  ASSERT_TRUE(BindTokenPolicy(file.get(), {"gpt2","qwen2",false,"","<|im_end|>"},
      text, publication, {"<|im_start|>","<|im_end|>"}, json, &bound, &error)) << error;
  EXPECT_EQ(bound.terminal.id, 151645); EXPECT_EQ(bound.encode_flags, 0u);
  EXPECT_EQ(bound.roles.count("tokenizer.ggml.bos_token_id"), 0u);
  EXPECT_EQ(bound.roles.at("tokenizer.ggml.padding_token_id").id, 151654);
}
TEST(TokenPolicyRealAssets, Qwen3OneShotAndIncrementalDecodeKeepReasoningMarkers) {
  const char* json_path = std::getenv("LOOMIFY_TEST_QWEN3_TOKENIZER");
  ASSERT_TRUE(json_path);
  std::ifstream stream(json_path, std::ios::binary); ASSERT_TRUE(stream);
  std::string json{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  iree_tokenizer_t* raw = nullptr;
  auto status = iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(), &raw);
  ASSERT_TRUE(iree_status_is_ok(status));
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(raw, iree_tokenizer_free);
  const std::string input = u8"<|im_start|><think>\n café 🦙 e\u0301 </think><|im_end|>";
  int32_t ids[128] = {}; iree_host_size_t count = 0;
  status = iree_tokenizer_encode(tokenizer.get(), iree_make_string_view(input.data(), input.size()),
      IREE_TOKENIZER_ENCODE_FLAG_NONE, iree_tokenizer_make_token_output(ids, nullptr, nullptr, 128),
      iree_allocator_system(), &count);
  ASSERT_TRUE(iree_status_is_ok(status)); ASSERT_GT(count, 0u);
  for (bool skip : {false, true}) {
    const auto flags = skip ? IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS : IREE_TOKENIZER_DECODE_FLAG_NONE;
    char buffer[256]; iree_host_size_t length = 0;
    status = iree_tokenizer_decode(tokenizer.get(), iree_tokenizer_make_token_id_list(ids, count), flags,
        iree_make_mutable_string_view(buffer, sizeof(buffer)), iree_allocator_system(), &length);
    ASSERT_TRUE(iree_status_is_ok(status));
    const std::string expected = skip ? u8"<think>\n café 🦙 é </think>" : u8"<|im_start|><think>\n café 🦙 é </think><|im_end|>";
    EXPECT_EQ(std::string(buffer, length), expected);
    iree_host_size_t state_size = 0;
    status = iree_tokenizer_decode_state_calculate_size(tokenizer.get(), &state_size);
    ASSERT_TRUE(iree_status_is_ok(status));
    std::vector<std::max_align_t> storage((state_size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
    iree_tokenizer_decode_state_t* state = nullptr;
    status = iree_tokenizer_decode_state_initialize(tokenizer.get(), flags,
        iree_make_byte_span(storage.data(), storage.size() * sizeof(std::max_align_t)), &state);
    ASSERT_TRUE(iree_status_is_ok(status));
    struct Cleanup { iree_tokenizer_decode_state_t* state; ~Cleanup() { iree_tokenizer_decode_state_deinitialize(state); } } cleanup{state};
    std::string incremental;
    for (iree_host_size_t i = 0; i < count;) {
      iree_host_size_t consumed = 0, written = 0;
      status = iree_tokenizer_decode_state_feed(state, iree_tokenizer_make_token_id_list(ids + i, 1),
          iree_make_mutable_string_view(buffer, sizeof(buffer)), &consumed, &written);
      ASSERT_TRUE(iree_status_is_ok(status)); ASSERT_GT(consumed + written, 0u);
      incremental.append(buffer, written); i += consumed;
    }
    length = 0;
    status = iree_tokenizer_decode_state_finalize(state, iree_make_mutable_string_view(buffer, sizeof(buffer)), &length);
    ASSERT_TRUE(iree_status_is_ok(status)); incremental.append(buffer, length);
    EXPECT_EQ(incremental, expected);
  }
}
} // namespace
} // namespace hrx::loomify_gguf
