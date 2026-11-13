// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/testing/qwen3_fixture.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf::testing {
namespace {
const OrdinaryPreparationCase kCase{
    "qwen3",
    +[]() { return Qwen3ModelFacts(); },
    "lib/architectures/qwen3/generated_prepare.loom",
    "lib/architectures/qwen3/prepare_library.loombc",
    1024,
    2,
    1024,
    "generated.qwen3_epoch",
    "<|im_end|>"};

TEST(Qwen3PreparationTest, CompleteRecordsRegistrationsAndNativeImports) {
  CheckPreparationRecords(
      kCase, {{1, 63}, {8, 256}, {128, 129}, {129, 129}, {512, 513}});
}

TEST(Qwen3PreparationTest, UnsupportedRequestsReachRealRequireImport) {
  CheckRejectedPreparationRequests(kCase);
}

TEST(Qwen3PreparationTest, MissingLiveConfigCannotLink) {
  CheckMissingPreparationConfig(kCase);
}
} // namespace
} // namespace hrx::loomify_gguf::testing
