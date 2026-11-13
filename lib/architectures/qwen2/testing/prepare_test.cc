// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/testing/qwen2_fixture.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf::testing {
namespace {
const OrdinaryPreparationCase kCase{
    "qwen2",
    +[]() { return qwen2::ModelFacts(); },
    "lib/architectures/qwen2/generated_prepare.loom",
    "lib/architectures/qwen2/prepare_library.loombc",
    1536,
    28,
    256,
    "generated.qwen2_epoch",
    "<|im_end|>"};

TEST(Qwen2PreparationTest, CompleteRecordsRegistrationsAndNativeImports) {
  CheckPreparationRecords(
      kCase, {{1, 63}, {8, 256}, {128, 129}, {129, 129}, {512, 513}});
}

TEST(Qwen2PreparationTest, UnsupportedRequestsReachRealRequireImport) {
  CheckRejectedPreparationRequests(kCase);
}

TEST(Qwen2PreparationTest, MissingLiveConfigCannotLink) {
  CheckMissingPreparationConfig(kCase);
}
} // namespace
} // namespace hrx::loomify_gguf::testing
