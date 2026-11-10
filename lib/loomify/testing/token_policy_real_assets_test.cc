// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/token_policy.h"
#include "tools/model/text_metadata.h"
#include "gtest/gtest.h"
#include <cstdlib>
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
    ASSERT_TRUE(BindTokenPolicy(file.get(), family, text, publication, markers, json, &bound, &error)) << error;
    EXPECT_EQ(bound.terminal.piece, llama ? "<|eot_id|>" : "<|im_end|>");
    EXPECT_EQ(bound.encode_flags, 0u);
    if (llama) {
      EXPECT_EQ(bound.terminal.id, 128009);
      EXPECT_EQ(publication->end_text.id, 128001);
      EXPECT_EQ(publication->end_message.id, 128008);
    } else {
      EXPECT_EQ(bound.terminal.id, 248046);
      EXPECT_EQ(bound.roles.at("tokenizer.ggml.padding_token_id").id, 248055);
    }
  }
}
} // namespace
} // namespace hrx::loomify_gguf
