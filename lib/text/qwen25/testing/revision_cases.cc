// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/qwen25/testing/revision_cases.h"
namespace hrx::loomify_gguf::testing {
namespace {
void CheckInput(ChatVM &vm, const RenderReferences &references) {
  CheckPreservedChatmlInput(vm, references, 248046);
}
const TextRevisionCase kCase{
    "cd8e9439", "CD8E9439", "loomify-qwen2", "qwen25-one-user", 28,
    2,          4,          false,           CheckInput};
class Qwen2RevisionTest : public ::testing::TestWithParam<bool> {};
TEST_P(Qwen2RevisionTest, ActualTransportPreservesGgufTemplateBytesAndTokens) {
  CheckRenderRevision(kCase, GetParam(),
                      Env("LOOMIFY_RENDER_REFERENCE_CD8E9439"));
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, Qwen2RevisionTest,
                         ::testing::Values(false, true));
} // namespace
} // namespace hrx::loomify_gguf::testing
