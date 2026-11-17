// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/qwen3/testing/revision_cases.h"
namespace hrx::loomify_gguf::testing {
namespace {
void CheckInput(ChatVM &vm, const RenderReferences &references) {
  CheckPreservedChatmlInput(vm, references, 248046);
}
const TextRevisionCase kCase{
    "5da44855", "5DA44855", "loomify-qwen3", "qwen3-one-user", 40, 1,
    5,          true,       CheckInput};
class Qwen3RevisionTest : public ::testing::TestWithParam<bool> {};
TEST_P(Qwen3RevisionTest, ActualTransportPreservesGgufTemplateBytesAndTokens) {
  CheckRenderRevision(kCase, GetParam(),
                      Env("LOOMIFY_RENDER_REFERENCE_5DA44855"));
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, Qwen3RevisionTest,
                         ::testing::Values(false, true));
} // namespace
} // namespace hrx::loomify_gguf::testing
