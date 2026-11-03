// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/llama_region.h"
#include "lib/dispatch/testing/ffn_fixture.h"
#include "gtest/gtest.h"
namespace hrx::dispatch {
namespace {
using namespace operations;
TEST(FfnRegion, RetainsSelectedOwnerAndDispatchesWithoutGraphMutation) {
  for (auto phase : {Phase::Prefill, Phase::Decode}) {
    std::string error;
    auto owner = testing::FfnFixture(testing::FfnFacts(), phase, 1, &error);
    ASSERT_TRUE(owner) << error;
    auto *identity = owner.get();
    const auto values = owner->graph().values().size();
    const auto ops = owner->graph().invocations().size();
    auto registry = Registry::Llama3B();
    std::unique_ptr<FfnRegion> region;
    ASSERT_TRUE(FfnRegion::Select(owner, {llama::LlamaRegionKind::Ffn, 0, 1},
                                  registry, &region, &error)) << error;
    owner.reset();
    EXPECT_EQ(&region->owner(), identity);
    EXPECT_EQ(region->owner().facts().layers[0].index, 7u);
    EXPECT_EQ(region->owner().graph().values().size(), values);
    EXPECT_EQ(region->owner().graph().invocations().size(), ops);
    EXPECT_EQ(region->view().operation_begin, 0u);
    EXPECT_EQ(region->view().operation_end, 3u);
    const bool decode = phase == Phase::Decode;
    EXPECT_EQ(region->dispatches(),
              (std::vector<Id>{Id::RmsNorm,
                               decode ? Id::GateUpDecode : Id::GateUpPrefill,
                               decode ? Id::DownDecode : Id::DownPrefill}));
    EXPECT_EQ(region->owner().parameters().size(), 4u);
  }
}
TEST(FfnRegion, RejectsGeometryRowsEpsilonAndMissingRegistrations) {
  auto registry = Registry::Llama3B();
  std::string error;
  std::unique_ptr<FfnRegion> result;
  auto baseline = testing::FfnFixture(testing::FfnFacts(), Phase::Prefill, 1,
                                     &error);
  ASSERT_TRUE(baseline) << error;
  ASSERT_TRUE(FfnRegion::Select(baseline, {llama::LlamaRegionKind::Ffn, 0, 1},
                                registry, &result, &error)) << error;
  auto *unchanged = result.get();
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto facts = testing::FfnFacts(mutation == 0 ? 256 : 3072);
    if (mutation == 2)
      facts.rms_epsilon = 1e-4;
    auto owner = testing::FfnFixture(std::move(facts), Phase::Prefill,
                                    mutation == 1 ? 2 : 1, &error);
    ASSERT_TRUE(owner) << error;
    Registry empty;
    EXPECT_FALSE(FfnRegion::Select(owner, {llama::LlamaRegionKind::Ffn, 0, 1},
                                   mutation == 3 ? empty : registry, &result,
                                   &error));
    EXPECT_EQ(result.get(), unchanged);
    EXPECT_FALSE(error.empty());
  }
  EXPECT_FALSE(FfnRegion::Select(baseline, {llama::LlamaRegionKind::Layer, 0, 1},
                                 registry, &result, &error));
  EXPECT_FALSE(FfnRegion::Select(baseline, {llama::LlamaRegionKind::Ffn, 1, 1},
                                 registry, &result, &error));
  EXPECT_EQ(result.get(), unchanged);
}
} // namespace
} // namespace hrx::dispatch
