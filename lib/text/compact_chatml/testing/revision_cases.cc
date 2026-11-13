// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/compact_chatml/testing/revision_cases.h"
namespace hrx::loomify_gguf::testing {
namespace {
void CheckCompactInput(ChatVM &vm, const RenderReferences &) {
  const std::string text = u8" café 🙂 \n";
  const std::string header = "<|im_start|>assistant\n<think>";
  const std::string user = "<|im_start|>user\n" + text + "<|im_end|>\n";
  CheckInputPolicy(vm, {header + text, text, header, user, "<|im_end|>\n", "\n",
                        "<|startoftext|>" + user + header, 124900});
}
const TextRevisionCase kCase{"f45005c5", "F45005C5", "loomify-lfm3",
                             "one-user", 41,         1,
                             7,          true,       CheckCompactInput};
class CompactChatmlTest : public ::testing::TestWithParam<bool> {};
void Init(ChatVM &vm, bool bytecode) {
  ASSERT_TRUE(vm.Init(Env(bytecode ? "LOOMIFY_RENDER_F45005C5_BC"
                                   : "LOOMIFY_RENDER_F45005C5_TEXT"),
                      Env("LOOMIFY_RENDER_TOKENIZER_F45005C5"),
                      "loomify-lfm3"));
}
TEST_P(CompactChatmlTest,
       ActualTransportMatchesOriginalGgufBytesAndFullTokens) {
  CheckRenderRevision(kCase, GetParam(),
                      Env("LOOMIFY_RENDER_REFERENCE_F45005C5"));
}
TEST_P(CompactChatmlTest,
       CompactReasoningPreservesContentAndCanonicalFollowup) {
  ChatVM vm;
  Init(vm, GetParam());
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  for (bool thinking : {false, true}) {
    const std::string request =
        std::string(
            R"({"messages":[{"role":"user","content":"Hello"}],"enable_thinking":)") +
        (thinking ? "true}" : "false}");
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(Ok(vm.Render(request, &chat)));
    const std::string prompt = Copy(iree_string_builder_view(&chat.prompt));
    const std::string prefix =
        "<|startoftext|><|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(prompt, prefix + (thinking ? "<think>" : "<think></think>"));
    const std::string content = u8"\n\n Answer café 🙂  \n";
    const std::string reason = " thought \n";
    const std::string visible =
        thinking ? reason + "</think>" + content : content;
    for (size_t end = 0; end <= visible.size(); ++end) {
      const auto part = View(visible.substr(0, end));
      if (!iree_unicode_utf8_validate(part))
        continue;
      // The public API consumes one span per call. Drain hidden delimiters and
      // channel transitions before moving to the next complete UTF8 prefix.
      bool settled = false;
      for (size_t guard = 0; guard <= part.size; ++guard) {
        loom_serve_text_chat_output_t span{};
        ASSERT_TRUE(Ok(loom_serve_text_chat_output_next(
            &chat, part, LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
            iree_allocator_system(), &span)));
        if (!span.advanced) {
          settled = true;
          break;
        }
      }
      ASSERT_TRUE(settled);
      const auto published =
          Copy(iree_string_builder_view(&chat.output.content));
      EXPECT_EQ(published, content.substr(0, published.size()));
    }
    loom_serve_text_tool_calls_t calls{};
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls);
    loom_serve_text_chat_scratch_t scratch{};
    loom_serve_text_chat_scratch_initialize(iree_allocator_system(), &scratch);
    loom_serve_text_chat_completion_t completion{};
    ASSERT_TRUE(Ok(loom_serve_text_chat_complete(
        &chat, View(visible), false, 124900, &calls, &scratch,
        iree_allocator_system(), &completion)));
    // COMPLETE drains retained UTF8/marker tails; STREAMING may withhold them.
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), content);
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.reasoning)),
              thinking ? reason : "");
    EXPECT_EQ(Copy(completion.transcript), prompt + visible + "<|im_end|>\n");
    EXPECT_FALSE(completion.reusable);
    // A later user drops old reasoning, emits the compact empty block and
    // replays from BOS rather than retaining incompatible history bytes.
    loom_serve_text_chat_t followup{};
    ASSERT_TRUE(Ok(vm.Render(
        R"({"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Answer","reasoning_content":"old"},{"role":"user","content":"Again"}]})",
        &followup)));
    EXPECT_EQ(Copy(iree_string_builder_view(&followup.prompt)),
              "<|startoftext|><|im_start|>user\nHello<|im_end|>\n"
              "<|im_start|>assistant\n<think></think>Answer<|im_end|>\n"
              "<|im_start|>user\nAgain<|im_end|>\n"
              "<|im_start|>assistant\n<think>");
    loom_serve_text_chat_deinitialize(&followup);
    loom_serve_text_chat_completion_deinitialize(&completion);
    loom_serve_text_chat_scratch_deinitialize(&scratch);
    loom_serve_text_tool_calls_deinitialize(&calls);
    loom_serve_text_chat_deinitialize(&chat);
  }
}
TEST_P(CompactChatmlTest, XmlCallsAndSplitOutputUseExistingProtocol) {
  ChatVM vm;
  Init(vm, GetParam());
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const std::string schema =
      R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}}}])";
  const std::string response =
      "<tool_call>\n<function=f>\n<parameter=n>\n7\n</parameter>\n"
      "</function>\n</tool_call>";
  std::string parsed;
  ASSERT_TRUE(Ok(vm.ParseTools(schema, response, &parsed)));
  EXPECT_NE(parsed.find("f"), std::string::npos);
  EXPECT_NE(parsed.find("7"), std::string::npos);
  loom_serve_text_chat_t chat{};
  ASSERT_TRUE(Ok(vm.Render(
      R"({"messages":[{"role":"user","content":"Hello"}],"enable_thinking":false})",
      &chat)));
  for (size_t end = 0; end <= response.size(); ++end) {
    loom_serve_text_chat_output_t span{};
    ASSERT_TRUE(Ok(
        loom_serve_text_chat_output_next(&chat, View(response.substr(0, end)),
                                         LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
                                         iree_allocator_system(), &span)));
  }
  EXPECT_TRUE(iree_string_builder_view(&chat.output.content).size == 0);
  loom_serve_text_chat_deinitialize(&chat);
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, CompactChatmlTest,
                         ::testing::Values(false, true));
} // namespace
} // namespace hrx::loomify_gguf::testing
