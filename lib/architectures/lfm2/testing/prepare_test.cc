// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/testing/lfm2_fixture.h"
#include <gtest/gtest.h>
namespace hrx::loomify_gguf::testing {
namespace {
const OrdinaryPreparationCase kCase{
    "lfm2",
    +[]() {
      model::ModelDescriptor m;
      std::string error;
      if (!model::LoadModelDescriptor(Lfm2Catalog(), &m, &error))
        std::abort();
      return m;
    },
    "lib/architectures/lfm2/generated_prepare.loom",
    "lib/architectures/lfm2/prepare_library.loombc",
    2048,
    6,
    512,
    "generated.lfm2_epoch",
    "<|im_end|>",
    10 * 2 * 2048 * 4};
TEST(Lfm2Preparation, CompactKvPlanesAndAppendedHistory) {
  CheckPreparationRecords(kCase, {{1, 64}, {8, 256}, {17, 256}, {128, 512}});
}
TEST(Lfm2Preparation, RejectsCheckpointsAndUnsupportedRequests) {
  CheckRejectedPreparationRequests(kCase);
}
TEST(Lfm2Preparation, MissingLiveFactsReject) {
  CheckMissingPreparationConfig(kCase);
}
} // namespace
} // namespace hrx::loomify_gguf::testing
