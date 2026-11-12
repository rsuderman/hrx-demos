// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/protocols/chatml_thinking/testing/hermes_policy.h"
namespace hrx::loomify_gguf::testing {
void CheckHermesPolicy(ChatVM &vm, const RenderReferences &refs) {
  auto fresh = refs.one_user;
  const auto hello = fresh.find("Hello");
  ASSERT_NE(hello, std::string::npos);
  fresh.replace(hello, 5, u8" \u2003Hello\u00a0 ");
  CheckInputPolicy(vm, {u8"<|im_start|>assistant\n \u2003Hello\u00a0 ",
                        u8" \u2003Hello\u00a0 ", "<|im_start|>assistant\n",
                        u8"<|im_start|>user\n \u2003Hello\u00a0 <|im_end|>\n",
                        "<|im_end|>\n", "\n", fresh, 248046});
  const std::string schemas =
      R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}])";
  const std::string call =
      R"(<tool_call>{"name":"f","arguments":{"text":"é </tool_call>","nested":{"ok":true},"n":1}}</tool_call>)";
  std::string parsed;
  ASSERT_TRUE(Ok(vm.ParseTools(schemas, call + "\n" + call, &parsed)));
  std::vector<iree_string_view_t> calls;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(parsed), Row, &calls)));
  ASSERT_EQ(calls.size(), 2u);
  for (auto item : calls) {
    iree_string_view_t function{}, name{}, raw_args{};
    ASSERT_TRUE(Field(item, "function", &function));
    ASSERT_TRUE(Field(function, "name", &name));
    EXPECT_EQ(Copy(name), "f");
    ASSERT_TRUE(Field(function, "arguments", &raw_args));
    std::string args;
    ASSERT_TRUE(Decode(raw_args, &args));
    EXPECT_EQ(args, R"({"text":"é </tool_call>","nested":{"ok":true},"n":1})");
  }
  ASSERT_TRUE(
      Ok(vm.ParseTools(schemas, "<tool_call>{\"name\":\"f\"", &parsed)));
  EXPECT_EQ(parsed, "[]"); // Incomplete generation is not a complete call.
  for (const auto &bad :
       {R"(<tool_call>{"name":"other","arguments":{}}</tool_call>)",
        R"(<tool_call>{"name":"f","name":"f","arguments":{}}</tool_call>)",
        R"(<tool_call>{"name":"f","arguments":[],}</tool_call>)",
        R"(<tool_call>{"name":"f","arguments":[]}</tool_call>)"}) {
    parsed = "unchanged";
    auto status = vm.ParseTools(schemas, bad, &parsed);
    const bool accepted = iree_status_is_ok(status);
    const auto diagnostic = Diagnostic(status);
    EXPECT_FALSE(accepted) << bad;
    EXPECT_EQ(parsed, "unchanged");
  }
}
} // namespace hrx::loomify_gguf::testing
