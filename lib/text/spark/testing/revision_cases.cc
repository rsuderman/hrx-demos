// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/spark/testing/revision_cases.h"
namespace hrx::loomify_gguf::testing {
namespace {
const std::string kStart = u8"<｜start▁of▁sentence｜>";
const std::string kEnd = u8"<｜end▁of▁sentence｜>";
const std::string kLead = "you are a helpful assistant.";
const std::string kHigh =
    "\n`thinking_effort` guides on how much to think in your thinking channel "
    "(not including the response channel), supported values include `low` and "
    "`high`.\nNow the system is invoked with `thinking_effort=high`.";
void CheckSparkInput(ChatVM &vm, const RenderReferences &) {
  const std::string text = u8" café 🙂 \n";
  const std::string header = kStart + "<|Bot|><think>";
  const std::string user = kStart + "<|User|>" + text + kEnd;
  CheckInputPolicy(
      vm, {header + text, text, header, user, kEnd, "",
           kStart + "<|System|>\n" + kLead + kHigh + kEnd + user + header, 1});
}
const TextRevisionCase kCase{"93d6aa90", "93D6AA90", "loomify-spark2_5",
                             "one-user", 26,         4,
                             3,          true,       CheckSparkInput};
class SparkTextTest : public ::testing::TestWithParam<bool> {};
void Init(ChatVM &vm, bool bc) {
  ASSERT_TRUE(vm.Init(
      Env(bc ? "LOOMIFY_RENDER_93D6AA90_BC" : "LOOMIFY_RENDER_93D6AA90_TEXT"),
      Env("LOOMIFY_RENDER_TOKENIZER_93D6AA90"), "loomify-spark2_5"));
}
TEST_P(SparkTextTest, OriginalTemplateBytesAndCompleteIds) {
  CheckRenderRevision(kCase, GetParam(),
                      Env("LOOMIFY_RENDER_REFERENCE_93D6AA90"));
}
TEST_P(SparkTextTest, StreamingPreservesWhitespaceAndCanonicalSecondTurn) {
  ChatVM vm;
  Init(vm, GetParam());
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  for (bool thinking : {false, true}) {
    const auto request =
        std::string(
            R"({"messages":[{"role":"user","content":"Hello"}],"enable_thinking":)") +
        (thinking ? "true}" : "false}");
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(Ok(vm.Render(request, &chat)));
    const auto prompt = Copy(iree_string_builder_view(&chat.prompt));
    const std::string content = u8"\n\n Answer café 🙂  \n";
    const std::string reason = " thought \n";
    const auto response = thinking ? reason + "</think>" + content : content;
    for (size_t end = 0; end <= response.size(); ++end) {
      const auto prefix_text = response.substr(0, end);
      const auto prefix = View(prefix_text);
      if (!iree_unicode_utf8_validate(prefix))
        continue;
      bool settled = false;
      for (size_t i = 0; i <= prefix.size; ++i) {
        loom_serve_text_chat_output_t span{};
        ASSERT_TRUE(Ok(loom_serve_text_chat_output_next(
            &chat, prefix, LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
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
        &chat, View(response), false, 1, &calls, &scratch,
        iree_allocator_system(), &completion)));
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), content);
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.reasoning)),
              thinking ? reason : "");
    const auto initial = kStart + "<|System|>\n" + kLead +
                         (thinking ? kHigh : "") + kEnd + kStart +
                         "<|User|>Hello" + kEnd;
    const auto canonical_assistant =
        kStart + "<|Bot|>" +
        (thinking ? "<think>" + reason + "</think>" : "</think>") + content +
        kEnd;
    EXPECT_EQ(Copy(completion.transcript), initial + canonical_assistant);
    EXPECT_FALSE(completion.reusable);
    loom_serve_text_chat_completion_deinitialize(&completion);
    loom_serve_text_chat_scratch_deinitialize(&scratch);
    loom_serve_text_tool_calls_deinitialize(&calls);
    loom_serve_text_chat_deinitialize(&chat);
    const std::string next =
        std::string(
            R"({"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"\n\n Answer café 🙂  \n","reasoning_content":)") +
        (thinking ? R"(" thought \n")" : R"("")") +
        R"(},{"role":"user","content":"Next"}],"enable_thinking":)" +
        (thinking ? "true}" : "false}");
    loom_serve_text_chat_t next_chat{};
    ASSERT_TRUE(Ok(vm.Render(next, &next_chat)));
    EXPECT_EQ(Copy(iree_string_builder_view(&next_chat.prompt)),
              initial + canonical_assistant + kStart + "<|User|>Next" + kEnd +
                  kStart + "<|Bot|>" + (thinking ? "<think>" : "</think>"));
    loom_serve_text_chat_deinitialize(&next_chat);
  }
}
TEST_P(SparkTextTest, GlmSchemaTypesAndUnambiguousStringValues) {
  ChatVM vm;
  Init(vm, GetParam());
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const std::string schemas =
      R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"s":{"type":"string"},"b":{"type":"boolean"},"n":{"type":"number"},"a":{"type":"array"},"o":{"type":"object"}}}}}])";
  std::string parsed;
  ASSERT_TRUE(Ok(vm.ParseTools(
      schemas,
      "<tool_call>f<arg_key>s</arg_key><arg_value> café <arg_key>literal</arg_value>"
      "<arg_key>b</arg_key><arg_value>true</arg_value><arg_key>n</arg_key><arg_value>1e0</arg_value>"
      "<arg_key>a</arg_key><arg_value>[1, 2]</arg_value><arg_key>o</arg_key><arg_value>{\"k\": 3}</arg_value></tool_call>",
      &parsed)));
  std::vector<iree_string_view_t> rows;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(parsed), Row, &rows)));
  ASSERT_EQ(rows.size(), 1u);
  iree_string_view_t function{}, name{}, raw_args{};
  ASSERT_TRUE(Field(rows[0], "function", &function));
  ASSERT_TRUE(Field(function, "name", &name));
  EXPECT_EQ(Copy(name), "f");
  ASSERT_TRUE(Field(function, "arguments", &raw_args));
  std::string args;
  ASSERT_TRUE(Decode(raw_args, &args));
  iree_string_view_t s{}, b{}, n{}, a{}, o{};
  ASSERT_TRUE(Field(View(args), "s", &s));
  std::string value;
  ASSERT_TRUE(Decode(s, &value));
  EXPECT_EQ(value, u8" café <arg_key>literal");
  ASSERT_TRUE(Field(View(args), "b", &b));
  EXPECT_EQ(Copy(b), "true");
  ASSERT_TRUE(Field(View(args), "n", &n));
  double number = 0;
  ASSERT_TRUE(Ok(iree_json_parse_double(n, &number)));
  EXPECT_DOUBLE_EQ(number,
                   1.0); // Float spelling remains deferred, type/value do not.
  ASSERT_TRUE(Field(View(args), "a", &a));
  EXPECT_EQ(Copy(a), "[1, 2]");
  ASSERT_TRUE(Field(View(args), "o", &o));
  EXPECT_EQ(Copy(o), "{\"k\": 3}");
  for (
      const std::string malformed :
      {"<tool_call>f<arg_key>s</arg_key><arg_value>x",
       "<tool_call>f<arg_key>s</arg_key><arg_value>x</arg_value>bad</tool_call>",
       "<tool_call>f<arg_key>s</arg_key><arg_value>x</arg_value></tool_call>tail",
       "<tool_call>unknown</tool_call>",
       "<tool_call>f<arg_key>s</arg_key><arg_value>x</arg_value><arg_key>s</arg_key><arg_value>y</arg_value></tool_call>",
       "<tool_call>f<arg_key>s</arg_key><arg_value>literal </parameter> delimiter</arg_value></tool_call>",
       "<tool_call>f<arg_key>s</arg_key><arg_value>literal </tool_call> delimiter</arg_value></tool_call>"}) {
    auto status = vm.ParseTools(schemas, malformed, &parsed);
    EXPECT_FALSE(iree_status_is_ok(status)) << malformed;
    Diagnostic(status);
  }
  auto unknown_schema =
      vm.ParseTools("[]", "<tool_call>f</tool_call>", &parsed);
  EXPECT_FALSE(iree_status_is_ok(unknown_schema));
  Diagnostic(unknown_schema);
}
TEST_P(SparkTextTest, ToolMarkerSplitsMultipleCallsAndNoGenerationPrefix) {
  ChatVM vm;
  Init(vm, GetParam());
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const std::string call1 =
      "<tool_call>f<arg_key>n</arg_key><arg_value>1</arg_value></tool_call>";
  const std::string call2 =
      "<tool_call>f<arg_key>n</arg_key><arg_value>2</arg_value></tool_call>";
  for (bool generation : {false, true}) {
    const auto request =
        std::string(
            R"({"messages":[{"role":"user","content":"Hello"}],"enable_thinking":false,"model_options":{"add_generation_prompt":)") +
        (generation ? "true" : "false") +
        R"(},"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"n":{"type":"number"}}}}}]})";
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(Ok(vm.Render(request, &chat)));
    const auto prompt = Copy(iree_string_builder_view(&chat.prompt));
    const auto response = "A" + call1 + call2;
    for (size_t end = 0; end <= response.size(); ++end) {
      const auto prefix_text = response.substr(0, end);
      const auto prefix = View(prefix_text);
      bool settled = false;
      for (size_t guard = 0; guard <= prefix.size; ++guard) {
        loom_serve_text_chat_output_t span{};
        ASSERT_TRUE(Ok(loom_serve_text_chat_output_next(
            &chat, prefix, LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,
            iree_allocator_system(), &span)));
        if (!span.advanced) {
          settled = true;
          break;
        }
      }
      ASSERT_TRUE(settled);
      const auto content = Copy(iree_string_builder_view(&chat.output.content));
      EXPECT_TRUE(content.empty() || content == "A");
    }
    loom_serve_text_tool_calls_t calls{};
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls);
    loom_serve_text_chat_scratch_t scratch{};
    loom_serve_text_chat_scratch_initialize(iree_allocator_system(), &scratch);
    loom_serve_text_chat_completion_t completion{};
    ASSERT_TRUE(Ok(loom_serve_text_chat_complete(
        &chat, View(response), false, 1, &calls, &scratch,
        iree_allocator_system(), &completion)));
    ASSERT_EQ(calls.count, 2u);
    for (size_t i = 0; i < calls.count; ++i) {
      const auto call = loom_serve_text_tool_calls_at(&calls, i);
      EXPECT_FALSE(Copy(call.id).empty());
      EXPECT_EQ(Copy(call.name), "f");
      EXPECT_EQ(Copy(call.arguments), i == 0 ? "{\"n\":1}" : "{\"n\":2}");
    }
    EXPECT_NE(Copy(loom_serve_text_tool_calls_at(&calls, 0).id),
              Copy(loom_serve_text_tool_calls_at(&calls, 1).id));
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), "A");
    const auto header = kStart + "<|Bot|></think>";
    const auto history =
        generation ? prompt.substr(0, prompt.size() - header.size()) : prompt;
    EXPECT_EQ(Copy(completion.transcript), history + header + response + kEnd);
    EXPECT_FALSE(completion.reusable);
    loom_serve_text_chat_completion_deinitialize(&completion);
    loom_serve_text_chat_scratch_deinitialize(&scratch);
    loom_serve_text_tool_calls_deinitialize(&calls);
    loom_serve_text_chat_deinitialize(&chat);
  }
}
TEST_P(SparkTextTest, EmptyThoughtOnlyAndTruncatedCallFinalization) {
  ChatVM vm;
  Init(vm, GetParam());
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  for (const auto &response : {std::string(), std::string(" unfinished ")}) {
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(Ok(vm.Render(
        R"({"messages":[{"role":"user","content":"Hello"}]})", &chat)));
    const auto prompt = Copy(iree_string_builder_view(&chat.prompt));
    loom_serve_text_tool_calls_t calls{};
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls);
    loom_serve_text_chat_scratch_t scratch{};
    loom_serve_text_chat_scratch_initialize(iree_allocator_system(), &scratch);
    loom_serve_text_chat_completion_t completion{};
    ASSERT_TRUE(Ok(loom_serve_text_chat_complete(
        &chat, View(response), true, 1, &calls, &scratch,
        iree_allocator_system(), &completion)));
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.reasoning)), response);
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)), "");
    const auto header = kStart + "<|Bot|>";
    const auto history = prompt.substr(0, prompt.size() - header.size() -
                                              std::string("<think>").size());
    EXPECT_EQ(Copy(completion.transcript),
              history + header +
                  (response.empty() ? "</think>"
                                    : "<think>" + response + "</think>") +
                  kEnd);
    EXPECT_FALSE(completion.reusable);
    loom_serve_text_chat_completion_deinitialize(&completion);
    loom_serve_text_chat_scratch_deinitialize(&scratch);
    loom_serve_text_tool_calls_deinitialize(&calls);
    loom_serve_text_chat_deinitialize(&chat);
  }
  const std::string schemas =
      R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}])";
  std::string parsed;
  auto status = vm.ParseTools(schemas, "<tool_call>f", &parsed, true);
  EXPECT_FALSE(
      iree_status_is_ok(status)); // Incomplete GLM envelope rejects explicitly.
  Diagnostic(status);
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, SparkTextTest, ::testing::Bool());
} // namespace
} // namespace hrx::loomify_gguf::testing
