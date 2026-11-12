// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/testing/configuration_fixture.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf::testing {
namespace {
const OrdinaryPreparationCase kCase{
    "llama", +[]() { return LlamaModelFacts(); },
    "lib/architectures/llama/prepare.loom",
    "lib/architectures/llama/prepare_library.loombc",
    3072, 28, 1024, "generated.llama_epoch", "<|eot_id|>"};

TEST(LlamaPreparationTest, CompleteRecordsRegistrationsAndNativeImports) {
  CheckPreparationRecords(kCase, {
      {1, 1}, {1, 63}, {128, 128}, {128, 129}, {129, 129},
      {512, 512}, {512, 513}, {512, 131072}});
}

TEST(LlamaPreparationTest, EffectivePrefillUsesRequestedShapes) {
  CheckEffectivePrefill(kCase);
}

TEST(LlamaPreparationTest, NativeSingleRowShapeInventoryPreservesStageOrdinals) {
  CheckSingleRowStageOrdinals(kCase);
}

TEST(LlamaPreparationTest, UnsupportedRequestsReachRealRequireImport) {
  CheckRejectedPreparationRequests(kCase);
}

TEST(LlamaPreparationTest, MissingLiveConfigCannotLink) {
  CheckMissingPreparationConfig(kCase);
}
}  // namespace
}  // namespace hrx::loomify_gguf::testing
