// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/testing/spark2_5_fixture.h"
#include <gtest/gtest.h>
namespace hrx::loomify_gguf::testing {
namespace {
const OrdinaryPreparationCase kCase{
    "spark2_5",
    +[]() {
      model::ModelDescriptor m;
      std::string error;
      if (!model::LoadModelDescriptor(Spark25Catalog(), &m, &error))
        std::abort();
      return m;
    },
    "lib/architectures/spark2_5/prepare.loom",
    "lib/architectures/spark2_5/prepare_library.loombc",
    2560,
    2,
    1024,
    "generated.spark2_5_epoch",
    "[eos]",
    0};
TEST(Spark25Preparation, FullKvPlanesWithoutTrailingState) {
  CheckPreparationRecords(kCase, {{1, 64}, {8, 256}, {17, 256}, {128, 1024}});
}
TEST(Spark25Preparation, RejectsCheckpointsAndUnsupportedRequests) {
  CheckRejectedPreparationRequests(kCase);
}
TEST(Spark25Preparation, MissingLiveFactsReject) {
  CheckMissingPreparationConfig(kCase);
}
} // namespace
} // namespace hrx::loomify_gguf::testing
