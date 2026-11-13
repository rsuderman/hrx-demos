// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/lfm3_fixture.h"
#include "tools/model/tensor_names.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>
namespace hrx::loomify_gguf {
namespace {
model::MetadataRecord *Field(model::GgufCatalog &c, const char *key) {
  for (auto &f : c.metadata)
    if (f.key == key)
      return &f;
  return nullptr;
}
TEST(Lfm3Descriptor, ExtractsKindsCompactOrdinalsAndSourceRoles) {
  auto c = testing::Lfm3Catalog();
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  ASSERT_TRUE(m.short_convolution);
  EXPECT_FALSE(m.ssm);
  EXPECT_EQ(m.short_convolution->kernel, 7u);
  EXPECT_EQ(m.short_convolution->history_rows, 6u);
  ASSERT_EQ(m.layers.size(), 3u);
  EXPECT_EQ(m.layers[0].kind, model::LayerKind::kShortConvolution);
  EXPECT_EQ(m.layers[0].convolution_ordinal, 0u);
  EXPECT_FALSE(m.layers[0].attention_ordinal);
  EXPECT_EQ(m.layers[1].attention_ordinal, 0u);
  EXPECT_EQ(m.layers[2].attention_ordinal, 1u);
  EXPECT_EQ(m.layers[1].attention_window, 0u);
  EXPECT_EQ(m.layers[2].attention_window, 256u);
  EXPECT_EQ(m.layers[1].rope_base, 1e6);
  EXPECT_EQ(m.layers[2].rope_base, 1e4);
  EXPECT_TRUE(m.output_tied_to_embedding);
  std::string name;
  ASSERT_TRUE(model::ExpectedTensorName(m, "convolution_input", 0, &name));
  EXPECT_EQ(name, "blk.0.shortconv.in_proj.weight");
  ASSERT_TRUE(model::ExpectedTensorName(m, "rope_frequencies", {}, &name));
  EXPECT_EQ(name, "rope_freqs.weight");
}
TEST(Lfm3Descriptor, RejectsMalformedRoutingWithoutChangingOutput) {
  auto good = testing::Lfm3Catalog();
  std::string error;
  model::ModelDescriptor out;
  out.name = "preserve";
  auto c = good;
  Field(c, "lfm3.attention.sliding_window_pattern")->elements[0] = true;
  EXPECT_FALSE(model::LoadModelDescriptor(c, &out, &error));
  EXPECT_EQ(out.name, "preserve");
  c = good;
  Field(c, "lfm3.attention.head_count_kv")->elements[1] = uint64_t(3);
  EXPECT_FALSE(model::LoadModelDescriptor(c, &out, &error));
  EXPECT_EQ(out.name, "preserve");
  c = good;
  Field(c, "lfm3.attention.sliding_window_pattern")->elements.pop_back();
  EXPECT_FALSE(model::LoadModelDescriptor(c, &out, &error));
  c = good;
  c.tensors.erase(std::remove_if(c.tensors.begin(), c.tensors.end(),
                                 [](const auto &t) {
                                   return t.name ==
                                          "blk.0.shortconv.conv.weight";
                                 }),
                  c.tensors.end());
  EXPECT_FALSE(model::LoadModelDescriptor(c, &out, &error));
  EXPECT_EQ(out.name, "preserve");
}
TEST(Lfm3Configuration, DerivesVaryingShapesMixedFormatsAndHybridStorage) {
  for (uint64_t hidden : {1024, 2048})
    for (bool tied : {false, true}) {
      auto c = testing::Lfm3Catalog(hidden, tied);
      model::ModelDescriptor m;
      std::string error;
      ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
      ConfigurationFacts facts;
      ASSERT_TRUE(
          DeriveConfiguration(m, "fixture.gguf", {8, 256}, &facts, &error))
          << error;
      EXPECT_NE(facts.source.find("@lfm3.model.layer_count = 3 : index"),
                std::string::npos);
      EXPECT_NE(
          facts.source.find("@lfm3.model.attention_layer_count = 2 : index"),
          std::string::npos);
      EXPECT_NE(facts.source.find("@lfm3.model.hidden_size = " +
                                  std::to_string(hidden)),
                std::string::npos);
      EXPECT_NE(facts.source.find("@lfm3.model.embedding_tag = 13 : index"),
                std::string::npos);
      EXPECT_NE(facts.source.find("@lfm3.model.tied_output = " +
                                  std::to_string(tied)),
                std::string::npos);
      EXPECT_NE(facts.source.find("@loomify.serving.layer_count = 2 : index"),
                std::string::npos);
      EXPECT_NE(facts.source.find("@loomify.serving.trailing_state_bytes = " +
                                  std::to_string(6 * hidden * 4)),
                std::string::npos);
      EXPECT_NE(
          facts.source.find("@lfm3.shortconv.history_base = 1049860 : offset"),
          std::string::npos);
      EXPECT_NE(facts.source.find("@loomify.config.layer_info"),
                std::string::npos);
      EXPECT_NE(facts.source.find("@loomify.config.layer_conv"),
                std::string::npos);
      EXPECT_FALSE(facts.parameters.empty());
    }
}
TEST(Lfm3Configuration, RejectsUnsupportedFactsTransactionally) {
  auto c = testing::Lfm3Catalog();
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error));
  ConfigurationFacts facts;
  facts.source = "preserve";
  auto bad = m;
  bad.short_convolution->kernel = 4;
  EXPECT_FALSE(
      DeriveConfiguration(bad, "fixture.gguf", {8, 256}, &facts, &error));
  EXPECT_EQ(facts.source, "preserve");
  bad = m;
  bad.layers[2].attention_ordinal = 0;
  EXPECT_FALSE(
      DeriveConfiguration(bad, "fixture.gguf", {8, 256}, &facts, &error));
  EXPECT_EQ(facts.source, "preserve");
  bad = m;
  bad.layers[1].tensors[0].tensor.byte_size++;
  EXPECT_FALSE(
      DeriveConfiguration(bad, "fixture.gguf", {8, 256}, &facts, &error));
  EXPECT_EQ(facts.source, "preserve");
  bad = m;
  bad.layers[0].tensors[4].tensor.storage_type = 2;
  EXPECT_FALSE(
      DeriveConfiguration(bad, "fixture.gguf", {8, 256}, &facts, &error));
  EXPECT_EQ(facts.source, "preserve");
}
#ifdef LOOMIFY_LFM3_REAL_ASSETS
TEST(Lfm3Configuration, ActualPinnedHeaderGeneratesAllConsumedBindings) {
  const char *path = std::getenv("LOOMIFY_TEST_LFM3_GGUF");
  ASSERT_NE(path, nullptr);
  FILE *f = std::fopen(path, "rb");
  ASSERT_NE(f, nullptr);
  std::setvbuf(f, nullptr, _IONBF, 0);
  model::GgufCatalog c;
  std::string error;
  const bool read = model::ReadGgufCatalog(f, &c, &error);
  std::fclose(f);
  ASSERT_TRUE(read) << error;
  model::ModelDescriptor m;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(m, "model.gguf", {8, 256}, &facts, &error))
      << error;
  EXPECT_EQ(m.layers.size(), 24u);
  EXPECT_EQ(m.embedding_width, 2048u);
  EXPECT_NE(facts.source.find("@lfm3.model.attention_layer_count = 11 : index"),
            std::string::npos);
  EXPECT_NE(facts.source.find(
                "@loomify.serving.trailing_state_bytes = 638976 : index"),
            std::string::npos);
  EXPECT_NE(facts.source.find("@lfm3.model.embedding_tag = 13 : index"),
            std::string::npos);
  EXPECT_GT(facts.parameters.size(), 200u);
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
