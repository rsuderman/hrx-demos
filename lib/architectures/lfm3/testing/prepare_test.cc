// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/testing/lfm3_fixture.h"
#include <gtest/gtest.h>
namespace hrx::loomify_gguf::testing {
namespace {
const OrdinaryPreparationCase kCase{
    "lfm3",
    +[]() {
      model::ModelDescriptor m;
      std::string error;
      if (!model::LoadModelDescriptor(Lfm3Catalog(), &m, &error))
        std::abort();
      return m;
    },
    "lib/architectures/lfm3/prepare.loom",
    "lib/architectures/lfm3/prepare_library.loombc",
    2048,
    2,
    512,
    "generated.lfm3_epoch",
    "<|im_end|>",
    6 * 2048 * 4};
TEST(Lfm3Preparation, CompactKvPlanesAndAppendedHistory) {
  CheckPreparationRecords(kCase, {{1, 64}, {8, 256}, {17, 256}, {128, 512}});
}
TEST(Lfm3Preparation, RejectsCheckpointsAndUnsupportedRequests) {
  CheckRejectedPreparationRequests(kCase);
}
TEST(Lfm3Preparation, MissingLiveFactsReject) {
  CheckMissingPreparationConfig(kCase);
}
} // namespace
} // namespace hrx::loomify_gguf::testing
