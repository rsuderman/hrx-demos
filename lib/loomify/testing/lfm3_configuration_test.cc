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
TEST(Lfm3Configuration, UsesCheckedOperandAndEmbeddingTags) {
  for (uint32_t tag : {12u, 13u, 14u}) {
    auto c = testing::Lfm3Catalog(1024, true);
    model::ModelDescriptor m;
    std::string error;
    ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
    for (auto &binding : m.globals) {
      if (binding.role != "token_embedding" && binding.role != "output")
        continue;
      binding.tensor.storage_type = tag;
      ASSERT_TRUE(model::EncodedTensorByteSize(tag, m.embedding_width,
                                               m.vocabulary_size,
                                               &binding.tensor.byte_size));
    }
    auto &layer = m.layers[0];
    layer.feed_forward_width = 32768;
    for (auto &binding : layer.tensors) {
      const bool down = binding.role == "ffn_down";
      if (!down && binding.role != "ffn_gate" && binding.role != "ffn_up")
        continue;
      auto &tensor = binding.tensor;
      tensor.normalized_extents = {int64_t(down ? 32768 : m.embedding_width),
                                   int64_t(down ? m.embedding_width : 32768), 1,
                                   1};
      tensor.storage_type = 6;
      ASSERT_TRUE(model::EncodedTensorByteSize(6, tensor.normalized_extents[0],
                                               tensor.normalized_extents[1],
                                               &tensor.byte_size));
    }
    ConfigurationFacts facts;
    ASSERT_TRUE(
        DeriveConfiguration(m, "fixture.gguf", {8, 256}, &facts, &error))
        << error;
    EXPECT_NE(facts.source.find("@lfm3.model.embedding_tag = " +
                                std::to_string(tag) + " : index"),
              std::string::npos);
    for (auto &binding : layer.tensors) {
      if (binding.role != "ffn_down")
        continue;
      binding.tensor.storage_type = 14;
      ASSERT_TRUE(model::EncodedTensorByteSize(14, 32768, m.embedding_width,
                                               &binding.tensor.byte_size));
    }
    facts.source = "preserve";
    facts.parameters.clear();
    EXPECT_FALSE(
        DeriveConfiguration(m, "fixture.gguf", {8, 256}, &facts, &error));
    EXPECT_EQ(facts.source, "preserve");
    EXPECT_TRUE(facts.parameters.empty());
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
TEST(Lfm3Configuration, AttentionGateRetainsPerLayerF32AndQ4Facts) {
  auto c = testing::Lfm3Catalog(2048);
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  auto gate = [](model::ModelDescriptor &m) -> model::TensorRecord & {
    for (auto &binding : m.layers.at(1).tensors)
      if (binding.role == "attention_gate")
        return binding.tensor;
    std::abort();
  };
  auto &q4 = gate(m);
  q4.storage_type = 12;
  ASSERT_TRUE(
      model::EncodedTensorByteSize(12, m.embedding_width, 16, &q4.byte_size));
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture.gguf", {3, 256}, &facts, &error))
      << error;
  const auto start = facts.source.find(
      "func.def public pure inline @loomify.config.layer_attention_gate");
  ASSERT_NE(start, std::string::npos);
  const auto lookup =
      facts.source.substr(start, facts.source.find("\n}\n", start) - start);
  EXPECT_NE(lookup.find("index.constant 12 : index"), std::string::npos);
  EXPECT_NE(lookup.find("index.constant " + std::to_string(q4.byte_size) +
                        " : index"),
            std::string::npos);
  EXPECT_NE(lookup.find("index.constant " +
                        std::to_string(4 * m.embedding_width * 16) +
                        " : index"),
            std::string::npos);
  auto bound = std::find_if(
      facts.parameters.begin(), facts.parameters.end(),
      [](const auto &p) { return p.role == "attention_gate" && p.layer == 1; });
  ASSERT_NE(bound, facts.parameters.end());
  EXPECT_EQ(bound->tensor.storage_type, 12u);
  EXPECT_EQ(bound->tensor.byte_size, q4.byte_size);
  for (int mutation = 0; mutation < 5; ++mutation) {
    auto bad = m;
    auto &t = gate(bad);
    if (mutation == 0) {
      t.storage_type =
          1; // Valid F16 extent, unsupported canonical gate format.
      ASSERT_TRUE(
          model::EncodedTensorByteSize(1, m.embedding_width, 16, &t.byte_size));
    }
    if (mutation == 1)
      t.rank = 1;
    if (mutation == 2)
      t.normalized_extents[1]++;
    if (mutation == 3)
      t.byte_size++;
    if (mutation == 4)
      t.absolute_offset = uint64_t(INT64_MAX);
    ConfigurationFacts rejected;
    rejected.source = "preserve";
    EXPECT_FALSE(
        DeriveConfiguration(bad, "fixture.gguf", {3, 256}, &rejected, &error));
    EXPECT_EQ(rejected.source, "preserve");
    EXPECT_TRUE(rejected.parameters.empty());
  }
}

TEST(Lfm3Configuration,
     AttentionGateUsesExistingProjectionFormatsAndCapacities) {
  for (uint32_t tag : {0u, 6u, 8u, 12u, 13u, 14u}) {
    SCOPED_TRACE(tag);
    auto c = testing::Lfm3Catalog(2048);
    model::ModelDescriptor m;
    std::string error;
    ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
    // Make each added format exclusive to the selected gate, so its exact
    // capacity proves gate sizing instead of inheriting a larger sibling role.
    if (tag != 0 && tag != 12) {
      auto use_q4 = [&](model::TensorRecord &tensor) {
        if (tensor.storage_type != tag)
          return;
        tensor.storage_type = 12;
        ASSERT_TRUE(model::EncodedTensorByteSize(
            12, tensor.normalized_extents[0], tensor.normalized_extents[1],
            &tensor.byte_size));
      };
      for (auto &other : m.layers)
        for (auto &binding : other.tensors)
          if (binding.role != "attention_gate")
            use_q4(binding.tensor);
      // Preserve the tied embedding/output alias while isolating projection
      // capacity.
      for (auto &binding : m.globals)
        if (binding.role == "output" || binding.role == "token_embedding")
          use_q4(binding.tensor);
    }
    auto &layer = m.layers.at(1);
    for (auto &binding : layer.tensors) {
      if (binding.role != "attention_gate")
        continue;
      binding.tensor.storage_type = tag;
      ASSERT_TRUE(model::EncodedTensorByteSize(tag, m.embedding_width, 16,
                                               &binding.tensor.byte_size));
    }
    ConfigurationFacts facts;
    ASSERT_TRUE(
        DeriveConfiguration(m, "fixture.gguf", {3, 256}, &facts, &error))
        << error;
    auto bound = std::find_if(
        facts.parameters.begin(), facts.parameters.end(), [](const auto &p) {
          return p.role == "attention_gate" && p.layer == 1;
        });
    ASSERT_NE(bound, facts.parameters.end());
    EXPECT_EQ(bound->tensor.storage_type, tag);
    const uint64_t block_values =
        tag == 0 ? 1 : (tag == 6 || tag == 8 ? 32 : 256);
    const uint64_t block_bytes = tag == 0    ? 4
                                 : tag == 6  ? 22
                                 : tag == 8  ? 34
                                 : tag == 12 ? 144
                                 : tag == 13 ? 176
                                             : 210;
    EXPECT_EQ(bound->tensor.byte_size,
              m.embedding_width / block_values * 16 * block_bytes);
    const auto start = facts.source.find(
        "func.def public pure inline @loomify.config.layer_attention_gate");
    ASSERT_NE(start, std::string::npos);
    const auto lookup =
        facts.source.substr(start, facts.source.find("\n}\n", start) - start);
    EXPECT_NE(lookup.find("index.constant " + std::to_string(tag) + " : index"),
              std::string::npos);
    EXPECT_NE(lookup.find("index.constant " +
                          std::to_string(bound->tensor.byte_size) + " : index"),
              std::string::npos);
    // Gate-only formats require sixteen outputs instead of unused default1.
    if (tag != 0 && tag != 12) {
      const std::string encoding = tag == 6    ? "q5_0"
                                   : tag == 8  ? "q8_0"
                                   : tag == 13 ? "q5k"
                                               : "q6k";
      EXPECT_NE(facts.source.find("config.def @ggml.linear_" + encoding +
                                  "_q8_1_x4.output_capacity = 16 : index"),
                std::string::npos);
    }
    // Other attention layers retain their raw F32 format and exact byte range.
    EXPECT_NE(lookup.find("index.constant " +
                          std::to_string(4 * m.embedding_width * 16) +
                          " : index"),
              std::string::npos);
  }
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
