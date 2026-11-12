// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/plain_chatml/testing/revision_cases.h"
#include <memory>
namespace hrx::loomify_gguf::testing {
namespace {
constexpr const char *kHeader = "<|im_start|>assistant\n";
void CheckPlainInput(ChatVM &vm, const RenderReferences &refs) {
  const std::string body = u8" \u2003Hello\u00a0 ";
  const std::string framed = "<|im_start|>user\n" + body + "<|im_end|>\n";
  CheckInputPolicy(vm, {std::string(kHeader) + body, body, kHeader, framed,
                        "<|im_end|>\n", "\n", framed + kHeader, 100273});
  std::string parsed;
  ASSERT_TRUE(Ok(vm.ParseTools("[]", "<tool_call>literal</tool_call>", &parsed)));
  EXPECT_EQ(parsed, "[]");
  auto status = vm.ParseTools(
      R"([{"type":"function","function":{"name":"f","parameters":{}}}])",
      "", &parsed);
  EXPECT_FALSE(iree_status_is_ok(status));
  Diagnostic(status);
}
const TextRevisionCase kCase{
    "58c1a1f0", "58C1A1F0", "loomify-llama", "one-user", 28,
    3, 7, false, CheckPlainInput};
class PlainChatmlTest : public ::testing::TestWithParam<bool> {};
void InitPlain(ChatVM &vm, bool bc) {
  ASSERT_TRUE(vm.Init(Env(bc ? "LOOMIFY_RENDER_58C1A1F0_BC"
                              : "LOOMIFY_RENDER_58C1A1F0_TEXT"),
                      Env("LOOMIFY_RENDER_TOKENIZER_58C1A1F0"),
                      "loomify-llama"));
}
TEST_P(PlainChatmlTest, ActualTransportMatchesOriginalGgufBytesAndFullTokens) {
  CheckRenderRevision(kCase, GetParam(),
                      Env("LOOMIFY_RENDER_REFERENCE_58C1A1F0"));
}
void Complete(ChatVM &vm, bool generation, std::string_view visible,
              int32_t last_token, loom_serve_text_chat_completion_t *completion) {
  loom_serve_text_chat_t chat{};
  const std::string request =
      std::string(R"({"messages":[{"role":"user","content":"Hello"}],"model_options":{"add_generation_prompt":)") +
      (generation ? "true}}" : "false}}");
  ASSERT_TRUE(Ok(vm.Render(request, &chat)));
  const std::string prompt = Copy(iree_string_builder_view(&chat.prompt));
  EXPECT_EQ(prompt, std::string("<|im_start|>user\nHello<|im_end|>\n") +
                        (generation ? kHeader : ""));
  for (size_t end = 0; end <= visible.size(); ++end) {
    const auto prefix = View(visible.substr(0, end));
    if (!iree_unicode_utf8_validate(prefix)) continue;
    loom_serve_text_chat_output_t span{};
    ASSERT_TRUE(Ok(loom_serve_text_chat_output_next(
        &chat, prefix, LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
        iree_allocator_system(), &span)));
    if (span.advanced) EXPECT_EQ(span.kind, LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT);
  }
  EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), visible);
  loom_serve_text_tool_calls_t calls{};
  loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls);
  loom_serve_text_chat_scratch_t scratch{};
  loom_serve_text_chat_scratch_initialize(iree_allocator_system(), &scratch);
  ASSERT_TRUE(Ok(loom_serve_text_chat_complete(
      &chat, View(visible), false, last_token, &calls, &scratch,
      iree_allocator_system(), completion)));
  EXPECT_EQ(calls.count, 0u);
  EXPECT_FALSE(completion->reusable);
  EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), visible);
  EXPECT_EQ(Copy(completion->transcript),
            prompt + (generation ? "" : kHeader) + std::string(visible) +
                "<|im_end|>\n");
  loom_serve_text_chat_scratch_deinitialize(&scratch);
  loom_serve_text_tool_calls_deinitialize(&calls);
  loom_serve_text_chat_deinitialize(&chat);
}
TEST_P(PlainChatmlTest, OutputWhitespaceMarkupUtf8AndCanonicalFollowup) {
  ChatVM vm;
  InitPlain(vm, GetParam());
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  for (bool generation : {false, true}) {
    for (const std::string visible :
         {std::string(), std::string(" \t\r\n "),
          std::string(u8"  café 🙂 <think>x</think> <tool_call>literal</tool_call>  ")}) {
      SCOPED_TRACE(generation ? "with generation header" : "without generation header");
      SCOPED_TRACE(visible);
      for (int32_t last_token : {7, 100273}) {
        loom_serve_text_chat_completion_t completion{};
        Complete(vm, generation, visible, last_token, &completion);
        loom_serve_text_chat_completion_deinitialize(&completion);
      }
    }
    for (int32_t last_token : {7, 100273}) {
      loom_serve_text_chat_completion_t completion{};
      Complete(vm, generation, "Hi", last_token, &completion);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      loom_serve_text_chat_t followup{};
      ASSERT_TRUE(Ok(vm.Render(
          R"({"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"},{"role":"user","content":"Again"}]})",
          &followup)));
      const std::string expected =
          "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n"
          "<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n";
      EXPECT_EQ(Copy(iree_string_builder_view(&followup.prompt)), expected);
      std::vector<int32_t> ids;
      ASSERT_TRUE(vm.Encode(expected, &ids));
      // This exact-capacity input fixture reserves one decode token, matching
      // the existing neutral/Gemma input witnesses rather than HTTP's128 default.
      followup.max_tokens = 1;
      const bool stopped = last_token == 100273;
      const loom_serve_text_input_prefix_t prefix{
          &completion, 3, last_token,
          stopped ? LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED
                  : LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN};
      loom_serve_text_input_t input{};
      ASSERT_TRUE(Ok(loom_serve_text_input_prepare(
          &followup, prefix, false, ids.size(), &input,
          iree_allocator_system())));
      EXPECT_EQ(input.history, nullptr);
      EXPECT_EQ(input.retained_count, 0u);
      EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count), ids);
      loom_serve_text_input_deinitialize(&input, iree_allocator_system());
      loom_serve_text_chat_deinitialize(&followup);
      loom_serve_text_chat_completion_deinitialize(&completion);
    }
  }
  for (const char *request : {
       R"({"messages":[{"role":"user","content":"Hello"}],"model_options":{"unknown":true}})",
       R"({"messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"low"})",
       R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file://image"}}]}]})"}) {
    loom_serve_text_chat_t chat{};
    auto status = vm.Render(request, &chat);
    const bool accepted = iree_status_is_ok(status);
    Diagnostic(status);
    EXPECT_FALSE(accepted);
    if (accepted) loom_serve_text_chat_deinitialize(&chat);
  }
}
TEST(PlainChatmlTokenizer, NfcAndIncrementalByteDecodePreserveOrdinaryAndSpecialPolicy) {
  std::string json;
  ASSERT_TRUE(Read(Env("LOOMIFY_RENDER_TOKENIZER_58C1A1F0"), &json));
  iree_tokenizer_t *raw = nullptr;
  ASSERT_TRUE(Ok(iree_tokenizer_from_huggingface_json(
      View(json), iree_allocator_system(), &raw)));
  std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)>
      tokenizer(raw, iree_tokenizer_free);
  const std::string input = u8"<|im_start|>cafe\u0301 🙂 \u1112\u1161\u11ab<|im_end|>";
  int32_t ids[128] = {};
  iree_host_size_t count = 0;
  ASSERT_TRUE(Ok(iree_tokenizer_encode(
      tokenizer.get(), View(input), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(ids, nullptr, nullptr, 128),
      iree_allocator_system(), &count)));
  ASSERT_GT(count, 0u);
  for (bool skip : {false, true}) {
    const auto flags = skip ? IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS
                            : IREE_TOKENIZER_DECODE_FLAG_NONE;
    const std::string expected = skip ? u8"café 🙂 한"
                                      : u8"<|im_start|>café 🙂 한<|im_end|>";
    char buffer[256];
    iree_host_size_t length = 0;
    ASSERT_TRUE(Ok(iree_tokenizer_decode(
        tokenizer.get(), iree_tokenizer_make_token_id_list(ids, count), flags,
        iree_make_mutable_string_view(buffer, sizeof(buffer)),
        iree_allocator_system(), &length)));
    EXPECT_EQ(std::string(buffer, length), expected);
    iree_host_size_t state_size = 0;
    ASSERT_TRUE(Ok(iree_tokenizer_decode_state_calculate_size(tokenizer.get(), &state_size)));
    std::vector<std::max_align_t> storage(
        (state_size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
    iree_tokenizer_decode_state_t *state = nullptr;
    ASSERT_TRUE(Ok(iree_tokenizer_decode_state_initialize(
        tokenizer.get(), flags,
        iree_make_byte_span(storage.data(), storage.size() * sizeof(std::max_align_t)),
        &state)));
    std::string incremental;
    for (iree_host_size_t i = 0; i < count;) {
      iree_host_size_t consumed = 0, written = 0;
      ASSERT_TRUE(Ok(iree_tokenizer_decode_state_feed(
          state, iree_tokenizer_make_token_id_list(ids + i, 1),
          iree_make_mutable_string_view(buffer, sizeof(buffer)), &consumed, &written)));
      ASSERT_GT(consumed + written, 0u);
      incremental.append(buffer, written);
      i += consumed;
    }
    length = 0;
    ASSERT_TRUE(Ok(iree_tokenizer_decode_state_finalize(
        state, iree_make_mutable_string_view(buffer, sizeof(buffer)), &length)));
    incremental.append(buffer, length);
    iree_tokenizer_decode_state_deinitialize(state);
    EXPECT_EQ(incremental, expected);
  }
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, PlainChatmlTest,
                         ::testing::Values(false, true));
} // namespace
} // namespace hrx::loomify_gguf::testing
