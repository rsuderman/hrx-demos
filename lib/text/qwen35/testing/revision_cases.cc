// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/qwen35/testing/revision_cases.h"
namespace hrx::loomify_gguf::testing {
namespace {
std::string ToolRequest(std::string_view escaped_arguments,
                        std::string_view schemas = {}) {
  constexpr std::string_view default_schemas =
      R"([{"type":"function","function":{"name":"fetch","parameters":{"type":"object","properties":{"n":{"type":"number"},"nested":{"type":"object","properties":{"x":{"type":"number"}}},"text":{"type":"string"}}}}}])";
  if (schemas.empty())
    schemas = default_schemas;
  return std::string(
             R"({"messages":[{"role":"user","content":"fetch"},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"fetch","arguments":")") +
         std::string(escaped_arguments) + R"("}}]}],"tools":)" +
         std::string(schemas) + "}";
}
void CheckAdmission(ChatVM &vm, std::string_view revision) {
  auto rejects = [&](std::string_view label, const std::string &request) {
    SCOPED_TRACE(std::string(label));
    // These are transport-valid JSON/JSONC inputs. Admission failures must
    // exercise the selected source contract, not a malformed outer fixture.
    auto cursor = View(request);
    iree_string_view_t parsed{};
    ASSERT_TRUE(Ok(iree_json_consume_value(&cursor, &parsed)));
    ASSERT_EQ(cursor.size, 0u);
    loom_serve_text_chat_t chat{};
    auto status = vm.Render(request, &chat);
    const bool accepted = iree_status_is_ok(status);
    const auto diagnostic = Diagnostic(status);
    EXPECT_FALSE(accepted) << "request unexpectedly admitted";
    if (accepted)
      loom_serve_text_chat_deinitialize(&chat);
  };
  rejects("duplicate literal argument keys",
          ToolRequest(R"({\"n\":1,\"n\":2})"));
  rejects("duplicate escaped-equivalent argument keys",
          ToolRequest(R"({\"n\":1,\"\\u006e\":2})"));
  rejects("nested duplicate argument keys",
          ToolRequest(R"({\"nested\":{\"x\":1,\"x\":2}})"));
  rejects("JSONC argument comment", ToolRequest(R"({\"n\":1/*comment*/})"));
  rejects("argument trailing comma", ToolRequest(R"({\"n\":1,})"));
  rejects(
      "JSONC schema comment",
      ToolRequest(
          R"({})",
          R"([{"type":"function","function":{"name":"fetch","parameters":{"type":/*comment*/"object","properties":{}}}}])"));
  rejects(
      "schema trailing comma",
      ToolRequest(
          R"({})",
          R"([{"type":"function","function":{"name":"fetch","parameters":{"type":"object","properties":{}}},}])"));
  rejects(
      "unknown model option",
      R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"typo":true}})");
  rejects(
      "wrong thinking boolean type",
      R"({"messages":[{"role":"user","content":"hello"}],"enable_thinking":"true"})");
  if (revision == "7f0e5290") {
    rejects(
        "unsupported 0.8B effort",
        R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"reasoning_effort":"low"}})");
  } else {
    rejects(
        "wrong preserve boolean type",
        R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"preserve_thinking":"false"}})");
    rejects(
        "wrong effort type",
        R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"reasoning_effort":true}})");
  }
  std::string plain_prompt;
  for (bool escaped : {false, true}) {
    const std::string body =
        escaped ? R"({"messages":[{"role":"u\u0073er","content":"Hello"}]})"
                : R"({"messages":[{"role":"user","content":"Hello"}]})";
    loom_serve_text_chat_t role_chat{};
    ASSERT_TRUE(Ok(vm.Render(body, &role_chat)));
    const auto prompt = Copy(iree_string_builder_view(&role_chat.prompt));
    if (escaped)
      EXPECT_EQ(prompt, plain_prompt);
    else
      plain_prompt = prompt;
    loom_serve_text_chat_deinitialize(&role_chat);
  }
  // Strings containing JSONC-looking punctuation remain ordinary string data.
  const auto request = ToolRequest(R"({\"text\":\"literal //, \\\",} café\"})");
  loom_serve_text_chat_t chat{};
  ASSERT_TRUE(Ok(vm.Render(request, &chat)));
  const auto prompt = Copy(iree_string_builder_view(&chat.prompt));
  EXPECT_NE(prompt.find(R"(literal //, ",} café)"), std::string::npos);
  loom_serve_text_chat_deinitialize(&chat);
}

void CheckPublicToolParser(ChatVM &vm, std::string_view revision) {
  const std::string schemas =
      R"([{"type":"function","function":{"name":"inspect","parameters":{"type":"object","properties":{"flag":{"type":"boolean"},"word":{"type":"string"},"nested":{"type":"object","properties":{"label":{"type":"string"},"z":{"type":"array","items":{"type":"boolean"}}},"required":["label","z"]}},"required":["flag","word","nested"]}}}])";
  auto xml = [](std::string_view flag) {
    return std::string("<tool_call>\n<function=inspect>\n<parameter=flag>\n") +
           std::string(flag) +
           "\n</parameter>\n<parameter=word>\nTrue\n</parameter>\n"
           "<parameter=nested>\n{\"label\":\"True\",\"z\":[true,false]}\n</"
           "parameter>\n"
           "</function>\n</tool_call>";
  };
  for (bool value : {true, false}) {
    const char *spelling = revision == "7f0e5290" ? (value ? "True" : "False")
                                                  : (value ? "true" : "false");
    std::string parsed;
    ASSERT_TRUE(Ok(vm.ParseTools(schemas, xml(spelling), &parsed)));
    std::vector<iree_string_view_t> calls;
    ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(parsed), Row, &calls)));
    ASSERT_EQ(calls.size(), 1u);
    iree_string_view_t function{}, name{}, raw_arguments{};
    ASSERT_TRUE(Field(calls[0], "function", &function));
    ASSERT_TRUE(Field(function, "name", &name));
    EXPECT_EQ(Copy(name), "inspect");
    ASSERT_TRUE(Field(function, "arguments", &raw_arguments));
    std::string arguments;
    ASSERT_TRUE(Decode(raw_arguments, &arguments));
    EXPECT_EQ(arguments, std::string("{\"flag\":") +
                             (value ? "true" : "false") +
                             ",\"word\":\"True\",\"nested\":{\"label\":"
                             "\"True\",\"z\":[true,false]}}");
  }
  if (revision == "c3cf9e34") {
    // The native XML codec uses schema types as hints, not validation. A
    // non-JSON Boolean spelling remains a string; callers own schema and
    // permission validation before executing a generated tool call.
    std::string parsed;
    ASSERT_TRUE(Ok(vm.ParseTools(schemas, xml("True"), &parsed)));
    std::vector<iree_string_view_t> calls;
    ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(parsed), Row, &calls)));
    ASSERT_EQ(calls.size(), 1u);
    iree_string_view_t function{}, raw_arguments{};
    ASSERT_TRUE(Field(calls[0], "function", &function));
    ASSERT_TRUE(Field(function, "arguments", &raw_arguments));
    std::string arguments;
    ASSERT_TRUE(Decode(raw_arguments, &arguments));
    EXPECT_EQ(arguments, "{\"flag\":\"True\",\"word\":\"True\",\"nested\":{"
                         "\"label\":\"True\",\"z\":[true,false]}}");
  }
  std::cout
      << "PUBLIC_TOOL_PARSER_COVERAGE " << revision
      << " typed_booleans=2 string_True_preserved nested_JSON_preserved\n";
}
void CheckCanonicalReplay(ChatVM &vm, std::string_view revision,
                          const std::string &schema_request,
                          const std::string &history_request,
                          const std::string &history_reference,
                          const std::vector<int32_t> &history_ids) {
  const std::string next_tool = "<|im_start|>user\n<tool_response>";
  const auto boundary = history_reference.find(next_tool);
  ASSERT_NE(boundary, std::string::npos);
  const auto expected = history_reference.substr(0, boundary);
  const std::string assistant = "<|im_start|>assistant\n<think>\n";
  const auto start = expected.rfind(assistant);
  const auto end = expected.rfind("<|im_end|>\n");
  ASSERT_NE(start, std::string::npos);
  ASSERT_NE(end, std::string::npos);
  ASSERT_GT(end, start + assistant.size());
  // Independent HF assistant-history bytes supply plan/Working/two tool calls.
  // Enable thinking for the initial 0.8 request so its generated suffix can
  // carry that reasoning. This modified initial prompt is not a new HF oracle
  // row; the final assistant-history rendering is independent of
  // enable_thinking.
  const auto response =
      expected.substr(start + assistant.size(), end - start - assistant.size());
  auto initial_request = schema_request;
  if (revision == "7f0e5290")
    initial_request.insert(initial_request.rfind('}'),
                           ",\"enable_thinking\":true");
  loom_serve_text_chat_completion_t completion{};
  struct CompletionCleanup {
    loom_serve_text_chat_completion_t *value;
    ~CompletionCleanup() {
      loom_serve_text_chat_completion_deinitialize(value);
    }
  } completion_cleanup{&completion};
  {
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(Ok(vm.Render(initial_request, &chat)));
    struct Cleanup {
      loom_serve_text_chat_t *chat;
      ~Cleanup() { loom_serve_text_chat_deinitialize(chat); }
    } cleanup{&chat};
    loom_serve_text_tool_calls_t calls{};
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls);
    struct CallsCleanup {
      loom_serve_text_tool_calls_t *value;
      ~CallsCleanup() { loom_serve_text_tool_calls_deinitialize(value); }
    } calls_cleanup{&calls};
    loom_serve_text_chat_scratch_t scratch{};
    loom_serve_text_chat_scratch_initialize(iree_allocator_system(), &scratch);
    struct ScratchCleanup {
      loom_serve_text_chat_scratch_t *value;
      ~ScratchCleanup() { loom_serve_text_chat_scratch_deinitialize(value); }
    } scratch_cleanup{&scratch};
    ASSERT_TRUE(Ok(loom_serve_text_chat_complete(
        &chat, View(response), false, 123, &calls, &scratch,
        iree_allocator_system(), &completion)));
    ASSERT_EQ(calls.count, 2u);
    for (size_t i = 0; i < calls.count; ++i) {
      const auto call = loom_serve_text_tool_calls_at(&calls, i);
      EXPECT_FALSE(Copy(call.id).empty());
      EXPECT_EQ(Copy(call.name), "fetch");
      EXPECT_EQ(Copy(call.arguments),
                i == 0 ? "{\"path\":\"a\"}" : "{\"path\":\"b\"}");
    }
    EXPECT_NE(Copy(loom_serve_text_tool_calls_at(&calls, 0).id),
              Copy(loom_serve_text_tool_calls_at(&calls, 1).id));
    EXPECT_EQ(Copy(completion.transcript), expected);
    EXPECT_FALSE(completion.reusable);
  }
  // Completion owns its transcript after the original request and all source
  // call scratch retire. The actual host must replay even for a matching byte
  // prefix when the source refuses cached-state reuse.
  loom_serve_text_chat_t fresh{};
  ASSERT_TRUE(Ok(vm.Render(history_request, &fresh)));
  struct Cleanup {
    loom_serve_text_chat_t *chat;
    ~Cleanup() { loom_serve_text_chat_deinitialize(chat); }
  } cleanup{&fresh};
  EXPECT_EQ(Copy(iree_string_builder_view(&fresh.prompt)), history_reference);
  EXPECT_TRUE(loom_serve_text_chat_completion_is_prefix(&completion, &fresh));
  fresh.max_tokens = 1;
  const loom_serve_text_input_prefix_t prefix{
      &completion, 3, 7, LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN};
  std::vector<int32_t> scratch(history_ids.size(), -991);
  loom_serve_text_input_t input{};
  struct InputCleanup {
    loom_serve_text_input_t *value;
    ~InputCleanup() {
      loom_serve_text_input_deinitialize(value, iree_allocator_system());
    }
  } input_cleanup{&input};
  ASSERT_TRUE(Ok(loom_serve_text_input_prepare(
      &fresh, prefix, false, scratch.size(), &input, iree_allocator_system())));
  EXPECT_EQ(input.history, nullptr);
  EXPECT_EQ(input.retained_count, 0u);
  ASSERT_EQ(input.count, history_ids.size());
  EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count),
            history_ids);
  std::cout
      << "CANONICAL_REPLAY_COVERAGE " << revision
      << " reasoning_content_two_tools_HF_prefix actual_host_full_replay\n";
}

// The only deferred spelling is the explicitly identified n floating argument.
// Keep the entire surrounding Unicode/tool/nested payload exact in these rows.
bool MaskFloat(std::string *prompt, std::string *spelling, double *value) {
  const std::string open = "<parameter=n>\n", close = "\n</parameter>";
  auto begin = prompt->find(open);
  if (begin == std::string::npos ||
      prompt->find(open, begin + open.size()) != std::string::npos)
    return false;
  begin += open.size();
  const auto end = prompt->find(close, begin);
  if (end == std::string::npos)
    return false;
  *spelling = prompt->substr(begin, end - begin);
  if (!Ok(iree_json_parse_double(View(*spelling), value)) ||
      !std::isfinite(*value))
    return false;
  prompt->replace(begin, end - begin, "[deferred floating spelling]");
  return true;
}
bool IsFloatCase(std::string_view name) {
  return name == "tool-arguments-types" ||
         name == "tool-arguments-float-decimal" ||
         name == "tool-arguments-float-exponent" ||
         name == "tool-arguments-negative-zero" ||
         name == "tool-arguments-small-exponent";
}

void CheckThinkingPolicy(ChatVM &vm, const RenderReferences &refs,
                         const char *revision) {
  auto fresh = refs.one_user;
  CheckInputPolicy(vm, {u8"<|im_start|>assistant\n \u2003Hello\u00a0 ",
                        u8" \u2003Hello\u00a0 ", "<|im_start|>assistant\n",
                        "<|im_start|>user\nHello<|im_end|>\n", "<|im_end|>\n",
                        "\n", fresh, 248046});
  CheckAdmission(vm, revision);
  CheckPublicToolParser(vm, revision);
  ASSERT_FALSE(refs.schema_request.empty());
  ASSERT_FALSE(refs.history_request.empty());
  ASSERT_FALSE(refs.history_ids.empty());
  CheckCanonicalReplay(vm, revision, refs.schema_request, refs.history_request,
                       refs.history_text, refs.history_ids);
}
void CheckInlineHistory(ChatVM &vm, const RenderReferences &refs) {
  CheckThinkingPolicy(vm, refs, "7f0e5290");
}
void CheckExplicitReasoning(ChatVM &vm, const RenderReferences &refs) {
  CheckThinkingPolicy(vm, refs, "c3cf9e34");
}
void CheckDateOptions(ChatVM &vm, bool accepts_numeric_date) {
  // Both are well-formed requests. Only the policy-specific date value differs;
  // no chat_begin/rendering call can supply the expected Llama failure here.
  const std::string valid =
      R"({"messages":[{"role":"user","content":"Hello"}],"date_string":"26 Jul 2024"})";
  const std::string invalid =
      R"({"messages":[{"role":"user","content":"Hello"}],"date_string":26})";
  ASSERT_TRUE(Ok(vm.RequestOptions(valid)));
  auto status = vm.RequestOptions(invalid);
  const bool accepted = iree_status_is_ok(status);
  const auto diagnostic = Diagnostic(status);
  EXPECT_EQ(accepted, accepts_numeric_date) << diagnostic;
  // A rejected call must not prevent subsequent valid use of the same binding.
  EXPECT_TRUE(Ok(vm.RequestOptions(valid)));
}
void CheckQwen35Revision(const TextRevisionCase &revision, bool bytecode,
                         const std::string &reference) {
  const auto product = Env(std::string("LOOMIFY_RENDER_") + revision.key +
                           (bytecode ? "_BC" : "_TEXT"));
  const auto tokenizer =
      Env(std::string("LOOMIFY_RENDER_TOKENIZER_") + revision.key);
  ASSERT_FALSE(product.empty());
  ASSERT_FALSE(tokenizer.empty());
  ASSERT_FALSE(reference.empty());
  ChatVM vm;
  ASSERT_TRUE(vm.Init(product, tokenizer, revision.package_name));
  std::string data;
  ASSERT_TRUE(Read(reference, &data));
  std::vector<iree_string_view_t> rows;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(data), Row, &rows)));
  size_t count = 0, exact = 0, negative = 0, deferred = 0;
  std::string one_user_reference, schema_request, history_request,
      history_reference;
  std::vector<int32_t> history_ids;
  for (const auto row : rows) {
    iree_string_view_t raw_revision{}, raw_name{};
    ASSERT_TRUE(Field(row, "revision", &raw_revision));
    ASSERT_EQ(Copy(raw_revision), revision.id);
    ASSERT_TRUE(Field(row, "case", &raw_name));
    const auto name = Copy(raw_name);
    SCOPED_TRACE(std::string(revision.id) + "/" + name +
                 (bytecode ? "/bc" : "/text"));
    ++count;
    std::string request;
    ASSERT_TRUE(TransportRequest(row, vm.name(), &request));
    iree_string_view_t error{};
    ASSERT_TRUE(Field(row, "error", &error));
    loom_serve_text_chat_t chat{};
    auto status = vm.Render(request, &chat);
    if (error.data) {
      ++negative;
      const bool accepted = iree_status_is_ok(status);
      const auto diagnostic = Diagnostic(status);
      EXPECT_FALSE(accepted)
          << "reference rejected request but native transport accepted it";
      if (accepted)
        loom_serve_text_chat_deinitialize(&chat);
      continue;
    }
    ASSERT_TRUE(Ok(status));
    struct Cleanup {
      loom_serve_text_chat_t *chat;
      ~Cleanup() { loom_serve_text_chat_deinitialize(chat); }
    } cleanup{&chat};
    const auto actual = Copy(iree_string_builder_view(&chat.prompt));
    iree_string_view_t text{}, ids{};
    ASSERT_TRUE(Field(row, "text", &text));
    ASSERT_TRUE(Field(row, "token_ids", &ids));
    std::string expected;
    ASSERT_TRUE(Decode(text, &expected));
    if (name == "one-user")
      one_user_reference = expected;
    ASSERT_TRUE(ids.data) << "oracle must include complete token vectors";
    std::vector<int32_t> expected_ids, actual_ids, prepared_ids;
    ASSERT_TRUE(Ok(iree_json_enumerate_array(ids, Token, &expected_ids)));
    if (name == "tool-schema")
      schema_request = request;
    if (name == "tool-multiple-results") {
      history_request = request;
      history_reference = expected;
      history_ids = expected_ids;
    }
    ASSERT_TRUE(vm.Encode(actual, &actual_ids));
    ASSERT_TRUE(vm.Prepare(chat, actual.size() + 16, &prepared_ids));
    EXPECT_EQ(prepared_ids, actual_ids)
        << "actual source input adapter differs from native flag0 encoding";
    if (IsFloatCase(name)) {
      ++deferred;
      auto masked_actual = actual, masked_expected = expected;
      std::string actual_spelling, reference_spelling;
      double a = 0, e = 0;
      ASSERT_TRUE(MaskFloat(&masked_actual, &actual_spelling, &a));
      ASSERT_TRUE(MaskFloat(&masked_expected, &reference_spelling, &e));
      EXPECT_EQ(masked_actual, masked_expected)
          << "only floating parameter spelling is deferred";
      EXPECT_EQ(a, e);
      if (e == 0)
        EXPECT_EQ(std::signbit(a), std::signbit(e));
      std::cout << "DEFERRED_FLOAT_SPELLING " << revision.id << '/' << name
                << " reference=" << reference_spelling
                << " native=" << actual_spelling << '\n';
      if (actual == expected)
        EXPECT_EQ(actual_ids, expected_ids);
    } else {
      ++exact;
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(actual_ids, expected_ids);
    }
  }
  ASSERT_FALSE(one_user_reference.empty());
  revision.check_policy(vm, {one_user_reference, schema_request,
                             history_request, history_reference, history_ids});
  EXPECT_EQ(count, revision.expected_rows);
  EXPECT_EQ(negative, revision.expected_template_errors);
  EXPECT_EQ(exact + deferred + negative, count);
  EXPECT_GT(exact, 0u);
  std::cout << "RENDER_COVERAGE " << revision.id << (bytecode ? " bc" : " text")
            << " rows=" << count << " exact=" << exact
            << " negative=" << negative << " deferred_float=" << deferred
            << '\n';
}

struct RevisionRun {
  const TextRevisionCase *test_case;
  bool bytecode;
};
const TextRevisionCase kInline{"7f0e5290", "7F0E5290", "loomify-qwen35",
                               "one-user", 34,         6,
                               0,          true,       CheckInlineHistory};
const TextRevisionCase kExplicit{
    "c3cf9e34", "C3CF9E34", "loomify-qwen35",      "one-user", 41, 7,
    0,          true,       CheckExplicitReasoning};
class Qwen35RevisionTest : public ::testing::TestWithParam<RevisionRun> {};
TEST_P(Qwen35RevisionTest, ActualTransportMatchesPinnedReference) {
  const auto run = GetParam();
  CheckQwen35Revision(
      *run.test_case, run.bytecode,
      Env(std::string("LOOMIFY_RENDER_REFERENCE_") + run.test_case->key));
}
TEST_P(Qwen35RevisionTest,
       DirectRequestOptionsPreservesDateValidationBeforeChatBegin) {
  const auto run = GetParam();
  ChatVM vm;
  ASSERT_TRUE(vm.Init(
      Env(std::string("LOOMIFY_RENDER_") + run.test_case->key +
          (run.bytecode ? "_BC" : "_TEXT")),
      Env(std::string("LOOMIFY_RENDER_TOKENIZER_") + run.test_case->key),
      run.test_case->package_name));
  CheckDateOptions(vm, true);
}
INSTANTIATE_TEST_SUITE_P(KnownRevisions, Qwen35RevisionTest,
                         ::testing::Values(RevisionRun{&kInline, false},
                                           RevisionRun{&kInline, true},
                                           RevisionRun{&kExplicit, false},
                                           RevisionRun{&kExplicit, true}));
} // namespace
} // namespace hrx::loomify_gguf::testing
