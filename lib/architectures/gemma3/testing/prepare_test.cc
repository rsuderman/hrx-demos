// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/testing/gemma3_fixture.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf::testing {
namespace {
const OrdinaryPreparationCase kCase{
    "gemma3",
    +[]() { return gemma3::ModelFacts(); },
    "lib/architectures/gemma3/generated_prepare.loom",
    "lib/architectures/gemma3/prepare_library.loombc",
    1152,
    6,
    256,
    "generated.gemma3_epoch",
    "<end_of_turn>"};

TEST(Gemma3PreparationTest, CompleteRecordsRegistrationsAndNativeImports) {
  CheckPreparationRecords(
      kCase, {{1, 63}, {8, 256}, {128, 129}, {129, 129}, {512, 513}});
}

TEST(Gemma3PreparationTest, LargeGeometryUsesCompletePooledKVPlanes) {
  auto large = kCase;
  large.model = +[]() { return gemma3::LargeModelFacts(); };
  large.hidden = 3840;
  large.layers = 48;
  large.kv = 2048;
  CheckPreparationRecords(large, {{1, 63}, {8, 1025}, {128, 2048}});
}

TEST(Gemma3PreparationTest, UnsupportedRequestsReachRealRequireImport) {
  CheckRejectedPreparationRequests(kCase);
}

TEST(Gemma3PreparationTest, MissingLiveConfigCannotLink) {
  CheckMissingPreparationConfig(kCase);
}
} // namespace
} // namespace hrx::loomify_gguf::testing
