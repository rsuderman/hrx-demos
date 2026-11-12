// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/gemma/testing/revision_cases.h"
namespace hrx::loomify_gguf::testing {
namespace {
void CheckGemmaInput(ChatVM &vm, const std::string &fresh_user) {
  const std::string raw = u8" \u2003Hello\u00a0 ";
  for (auto format :
       {LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, LOOM_SERVE_TEXT_CHAT_INPUT_USER}) {
    for (auto boundary : {LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
                          LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,
                          LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED}) {
      SCOPED_TRACE("Gemma input=" + std::to_string(format) +
                   " boundary=" + std::to_string(boundary));
      const bool user = format == LOOM_SERVE_TEXT_CHAT_INPUT_USER;
      const std::string separator =
          boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN ? "<end_of_turn>\n"
          : boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED ? "\n"
                                                                 : "";
      const std::string expected_text =
          user ? (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH
                      ? fresh_user
                      : separator + "<start_of_turn>user\nHello<end_of_turn>\n<"
                                    "start_of_turn>model\n")
               : separator + raw;
      std::vector<int32_t> expected;
      ASSERT_TRUE(vm.Encode(expected_text, &expected));
      loom_serve_text_chat_t chat{};
      ASSERT_TRUE(RawChat(vm, raw, &chat));
      std::vector<int32_t> actual(expected.size(), -991);
      iree_host_size_t count = 0;
      ASSERT_TRUE(Ok(PrepareMode(chat, format, boundary, expected.size(),
                                 actual.data(), &count)));
      EXPECT_EQ(count, expected.size());
      EXPECT_EQ(actual, expected);
      if (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH) {
        std::fill(actual.begin(), actual.end(), -991);
        count = 99;
        auto status = PrepareMode(chat, format, boundary, expected.size() - 1,
                                  actual.data(), &count);
        EXPECT_FALSE(iree_status_is_ok(status));
        Diagnostic(status);
        EXPECT_EQ(count, 0u);
        EXPECT_EQ(actual, std::vector<int32_t>(expected.size(), -991));
      }
      loom_serve_text_chat_deinitialize(&chat);
    }
  }
  // Direct raw-input retained-boundary proof. Production structured completion
  // below refuses reuse; this fixture does not claim a structured cache hit.
  std::vector<int32_t> eot;
  ASSERT_TRUE(vm.Encode("<end_of_turn>", &eot));
  ASSERT_EQ(eot.size(), 1u);
  for (auto boundary : {LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,
                        LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED}) {
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(RawChat(vm, "HISTORY|NEXT", &chat));
    loom_serve_text_chat_completion_t completion{};
    ASSERT_TRUE(
        Ok(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                                iree_make_const_byte_span("HISTORY|", 8), 1,
                                iree_allocator_system(), &completion.storage)));
    completion.transcript = iree_make_string_view(
        static_cast<const char *>(
            iree_vm_buffer_const_data(completion.storage)),
        8);
    completion.reusable = true;
    const int32_t anchor =
        boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN ? 7 : eot[0];
    const std::string sep = boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN
                                ? "<end_of_turn>\n"
                                : "\n";
    std::vector<int32_t> expected;
    ASSERT_TRUE(vm.Encode(sep + "NEXT", &expected));
    expected.insert(expected.begin(), anchor);
    const loom_serve_text_input_prefix_t prefix{&completion, 3, anchor,
                                                boundary};
    std::vector<int32_t> scratch(expected.size() + 3, -991);
    loom_serve_text_input_t input{};
    ASSERT_TRUE(Ok(loom_serve_text_input_prepare(
        &chat, prefix, true, scratch.size(), &input, iree_allocator_system())));
    EXPECT_EQ(input.retained_count, 3u);
    EXPECT_EQ(input.count, expected.size());
    EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count),
              expected);
    loom_serve_text_input_deinitialize(&input, iree_allocator_system());
    loom_serve_text_chat_completion_deinitialize(&completion);
    loom_serve_text_chat_deinitialize(&chat);
  }
}
void CheckGemmaOutputReplay(ChatVM &vm, const std::string &replay_request,
                            const std::string &replay_text,
                            const std::vector<int32_t> &replay_ids) {
  const std::string visible = u8"  spaced café 🙂  ";
  std::vector<int32_t> terminal;
  ASSERT_TRUE(vm.Encode("<end_of_turn>", &terminal));
  ASSERT_EQ(terminal, std::vector<int32_t>({106}));
  // The canonical106 case is the host boundary shared by both independently
  // forced raw-stop publisher witnesses. This test does not implement or prove
  // the raw1/raw106 mapping; it verifies actual SDK completion and fresh
  // followup at their common published boundary, alongside the ordinary open
  // boundary.
  for (int32_t last_token : {9, terminal[0]}) {
    SCOPED_TRACE("Gemma canonical followup last_token=" +
                 std::to_string(last_token));
    loom_serve_text_chat_completion_t completion{};
    {
      loom_serve_text_chat_t chat{};
      ASSERT_TRUE(Ok(vm.Render(
          R"({"messages":[{"role":"user","content":"Hello"}]})", &chat)));
      for (size_t end = 0; end <= visible.size(); ++end) {
        const auto prefix = View(std::string_view(visible).substr(0, end));
        if (!iree_unicode_utf8_validate(prefix))
          continue;
        loom_serve_text_chat_output_t span{};
        ASSERT_TRUE(Ok(loom_serve_text_chat_output_next(
            &chat, prefix, LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
            iree_allocator_system(), &span)));
        if (span.advanced)
          EXPECT_EQ(span.kind, LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT);
      }
      EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), visible);
      loom_serve_text_tool_calls_t calls{};
      loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls);
      loom_serve_text_chat_scratch_t scratch{};
      loom_serve_text_chat_scratch_initialize(iree_allocator_system(),
                                              &scratch);
      ASSERT_TRUE(Ok(loom_serve_text_chat_complete(
          &chat, View(visible), false, last_token, &calls, &scratch,
          iree_allocator_system(), &completion)));
      EXPECT_EQ(calls.count, 0u);
      EXPECT_FALSE(completion.reusable);
      EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), visible);
      EXPECT_EQ(Copy(completion.transcript),
                std::string("<bos><start_of_turn>user\nHello<end_of_turn>\n<"
                            "start_of_turn>model\n") +
                    u8"spaced café 🙂<end_of_turn>\n");
      loom_serve_text_chat_scratch_deinitialize(&scratch);
      loom_serve_text_tool_calls_deinitialize(&calls);
      loom_serve_text_chat_deinitialize(&chat);
    }
    loom_serve_text_chat_t fresh{};
    ASSERT_TRUE(Ok(vm.Render(replay_request, &fresh)));
    EXPECT_EQ(Copy(iree_string_builder_view(&fresh.prompt)), replay_text);
    fresh.max_tokens = 1;
    const bool stopped = last_token == terminal[0];
    const loom_serve_text_input_prefix_t prefix{
        &completion, 3, stopped ? terminal[0] : 7,
        stopped ? LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED
                : LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN};
    std::vector<int32_t> scratch(replay_ids.size(), -991);
    loom_serve_text_input_t input{};
    ASSERT_TRUE(
        Ok(loom_serve_text_input_prepare(&fresh, prefix, false, scratch.size(),
                                         &input, iree_allocator_system())));
    EXPECT_EQ(input.history, nullptr);
    EXPECT_EQ(input.retained_count, 0u);
    EXPECT_EQ(input.count, replay_ids.size());
    EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count),
              replay_ids);
    loom_serve_text_input_deinitialize(&input, iree_allocator_system());
    loom_serve_text_chat_deinitialize(&fresh);
    loom_serve_text_chat_completion_deinitialize(&completion);
  }
}
// An added tokenizer media spelling in a plain string is not a media request.
// Binding accepts the JSON, but the actual encoded OOV ID must not reach
// gather.
void CheckGemmaTokenDomain(ChatVM &vm) {
  std::vector<int32_t> encoded;
  ASSERT_TRUE(vm.Encode("<image_soft_token>", &encoded));
  ASSERT_EQ(encoded, std::vector<int32_t>({262144}));
  for (auto format :
       {LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, LOOM_SERVE_TEXT_CHAT_INPUT_USER}) {
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(RawChat(vm, "<image_soft_token>", &chat));
    std::vector<int32_t> scratch(128, -991);
    iree_host_size_t count = 99;
    auto status = PrepareMode(chat, format, LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
                              scratch.size(), scratch.data(), &count);
    EXPECT_FALSE(iree_status_is_ok(status));
    Diagnostic(status);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(scratch, std::vector<int32_t>(128, -991));
    loom_serve_text_chat_deinitialize(&chat);
  }
  loom_serve_text_chat_t chat{};
  ASSERT_TRUE(Ok(vm.Render(
      R"({"messages":[{"role":"user","content":"<image_soft_token>"}]})",
      &chat)));
  std::vector<int32_t> scratch(128, -991);
  iree_host_size_t count = 99;
  auto status = PrepareMode(chat, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
                            LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, scratch.size(),
                            scratch.data(), &count);
  EXPECT_FALSE(iree_status_is_ok(status));
  Diagnostic(status);
  EXPECT_EQ(count, 0u);
  EXPECT_EQ(scratch, std::vector<int32_t>(128, -991));
  loom_serve_text_chat_deinitialize(&chat);
}
class GemmaRenderTest : public ::testing::TestWithParam<bool> {};
TEST_P(GemmaRenderTest, ActualTransportInputOutputAndReplayMatchesHF) {
  const bool bc = GetParam();
  ChatVM vm;
  ASSERT_TRUE(
      vm.Init(Env(bc ? "LOOMIFY_RENDER_GEMMA_BC" : "LOOMIFY_RENDER_GEMMA_TEXT"),
              Env("LOOMIFY_RENDER_TOKENIZER_GEMMA"), "loomify-gemma3"));
  CheckGemmaTokenDomain(vm);
  std::string data;
  ASSERT_TRUE(Read(Env("LOOMIFY_RENDER_GEMMA_REFERENCE"), &data));
  std::vector<iree_string_view_t> rows;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(data), Row, &rows)));
  size_t exact = 0, negative = 0;
  std::string one_user, replay_request, replay_text;
  std::vector<int32_t> replay_ids;
  for (auto row : rows) {
    iree_string_view_t name{}, error{}, kwargs{}, generation{};
    ASSERT_TRUE(Field(row, "case", &name));
    ASSERT_TRUE(Field(row, "error", &error));
    ASSERT_TRUE(Field(row, "kwargs", &kwargs));
    ASSERT_TRUE(Field(kwargs, "add_generation_prompt", &generation));
    const bool gen = Copy(generation) == "true";
    SCOPED_TRACE(Copy(name) + (gen ? "/generation" : "/no-generation"));
    std::string request;
    ASSERT_TRUE(TransportRequest(row, vm.name(), &request));
    loom_serve_text_chat_t chat{};
    auto status = vm.Render(request, &chat);
    if (error.data) {
      ++negative;
      const bool accepted = iree_status_is_ok(status);
      Diagnostic(status);
      EXPECT_FALSE(accepted);
      if (accepted)
        loom_serve_text_chat_deinitialize(&chat);
      continue;
    }
    ASSERT_TRUE(Ok(status));
    ++exact;
    iree_string_view_t text{}, ids{};
    ASSERT_TRUE(Field(row, "text", &text));
    ASSERT_TRUE(Field(row, "token_ids", &ids));
    std::string expected;
    ASSERT_TRUE(Decode(text, &expected));
    std::vector<int32_t> expected_ids, actual_ids, prepared;
    ASSERT_TRUE(Ok(iree_json_enumerate_array(ids, Token, &expected_ids)));
    const auto actual = Copy(iree_string_builder_view(&chat.prompt));
    EXPECT_EQ(actual, expected);
    ASSERT_TRUE(vm.Encode(actual, &actual_ids));
    EXPECT_EQ(actual_ids, expected_ids);
    ASSERT_TRUE(vm.Prepare(chat, expected_ids.size(), &prepared));
    EXPECT_EQ(prepared, expected_ids);
    if (Copy(name) == "one-user" && gen)
      one_user = expected;
    if (Copy(name) == "replay" && gen) {
      replay_request = request;
      replay_text = expected;
      replay_ids = expected_ids;
    }
    loom_serve_text_chat_deinitialize(&chat);
  }
  EXPECT_EQ(exact, 24u);
  EXPECT_EQ(negative, 8u);
  ASSERT_FALSE(one_user.empty());
  ASSERT_FALSE(replay_request.empty());
  CheckGemmaInput(vm, one_user);
  CheckGemmaOutputReplay(vm, replay_request, replay_text, replay_ids);
  for (
      const char *request :
      {R"({"messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"fetch","parameters":{"type":"object","properties":{}}}}]})",
       R"({"messages":[{"role":"user","content":"U"}],"enable_thinking":true})",
       R"({"messages":[{"role":"user","content":"U"}],"reasoning_effort":"low"})",
       R"({"messages":[{"role":"user","content":"U"}],"reasoning_effort":null})",
       R"({"messages":[{"role":"user","content":"U"},{"role":"tool","content":"T"}]})",
       R"({"messages":[{"role":"user","content":"U"}],"model_options":{"unknown":true}})"}) {
    loom_serve_text_chat_t chat{};
    auto status = vm.Render(request, &chat);
    const bool accepted = iree_status_is_ok(status);
    Diagnostic(status);
    EXPECT_FALSE(accepted);
    if (accepted)
      loom_serve_text_chat_deinitialize(&chat);
  }
  loom_serve_text_chat_t escaped{};
  ASSERT_TRUE(Ok(vm.Render(
      R"({"messages":[{"role":"u\u0073er","content":"Hello"}]})", &escaped)));
  EXPECT_EQ(Copy(iree_string_builder_view(&escaped.prompt)), one_user);
  loom_serve_text_chat_deinitialize(&escaped);
  std::cout
      << "GEMMA_COVERAGE " << (bc ? "bc" : "source")
      << " exact24 source_errors8 native_rejections6 input_modes6 overflow2 "
         "raw_retained2 outputUTF8 canonicalReplay2 oovRejected3\n";
}
INSTANTIATE_TEST_SUITE_P(GemmaSourceAndBytecode, GemmaRenderTest,
                         ::testing::Values(false, true));
} // namespace
} // namespace hrx::loomify_gguf::testing
