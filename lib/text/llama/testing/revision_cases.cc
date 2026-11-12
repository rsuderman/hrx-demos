// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/llama/testing/revision_cases.h"
namespace hrx::loomify_gguf::testing {
namespace {
void CheckLlamaPolicy(ChatVM &vm, const RenderReferences &refs) {
  CheckInputPolicy(
      vm, {u8"<|begin_of_text|> \u2003Hello\u00a0 ", u8" \u2003Hello\u00a0 ",
           "<|start_header_id|>assistant<|end_header_id|>\n\n",
           "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>",
           "<|eot_id|>", "", refs.one_user, 128009});
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

const TextRevisionCase kCase{"5816fce1", "5816FCE1", "loomify-llama",
                             "one-user", 7,          0,
                             0,          false,      CheckLlamaPolicy};
class LlamaRevisionTest : public ::testing::TestWithParam<bool> {};
TEST_P(LlamaRevisionTest, ActualTransportMatchesPinnedReference) {
  CheckRenderRevision(kCase, GetParam(),
                      Env("LOOMIFY_RENDER_REFERENCE_5816FCE1"));
}
TEST_P(LlamaRevisionTest,
       DirectRequestOptionsPreservesDateValidationBeforeChatBegin) {
  ChatVM vm;
  ASSERT_TRUE(vm.Init(Env(GetParam() ? "LOOMIFY_RENDER_5816FCE1_BC"
                                     : "LOOMIFY_RENDER_5816FCE1_TEXT"),
                      Env("LOOMIFY_RENDER_TOKENIZER_5816FCE1"),
                      "loomify-llama"));
  CheckDateOptions(vm, false);
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, LlamaRevisionTest,
                         ::testing::Values(false, true));
} // namespace
} // namespace hrx::loomify_gguf::testing
