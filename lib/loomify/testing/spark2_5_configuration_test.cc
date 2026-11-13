// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/spark2_5_fixture.h"
#include "tools/model/tensor_names.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
namespace hrx::loomify_gguf {
namespace {
model::MetadataRecord &Field(model::GgufCatalog &c, const char *key) {
  for (auto &f : c.metadata)
    if (f.key == std::string("spark2_5.") + key)
      return f;
  std::abort();
}
model::TensorRecord &Tensor(model::ModelDescriptor &m, size_t layer,
                            const char *role) {
  for (auto &t : m.layers.at(layer).tensors)
    if (t.role == role)
      return t.tensor;
  std::abort();
}
TEST(SparkDescriptor, ExtractsSourceProfilesFusedRolesAndTiedIdentity) {
  auto c = testing::Spark25Catalog();
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  EXPECT_EQ(m.training_context, 1048576u);
  EXPECT_EQ(m.embedding_width, 2560u);
  EXPECT_EQ(m.key_head_width, 256u);
  EXPECT_EQ(m.rope.width, 64u);
  EXPECT_EQ(m.layers[0].attention_window, 512u);
  EXPECT_EQ(m.layers[0].rope_width, 256u);
  EXPECT_EQ(m.layers[0].rope_base, 10000);
  EXPECT_EQ(m.layers[1].attention_window, 0u);
  EXPECT_EQ(m.layers[1].rope_width, 64u);
  EXPECT_EQ(m.layers[1].rope_base, 5000000);
  EXPECT_EQ(m.layers[1].attention_ordinal, 1u);
  EXPECT_EQ(m.layers[0].tensors.size(), 8u);
  EXPECT_TRUE(m.output_tied_to_embedding);
  std::string name;
  ASSERT_TRUE(model::ExpectedTensorName(m, "query_key_value", 0, &name));
  EXPECT_EQ(name, "blk.0.attn_qkv.weight");
  ASSERT_TRUE(model::ExpectedTensorName(m, "output", {}, &name));
  EXPECT_EQ(name, "token_embd.weight");
  EXPECT_EQ(Tensor(m, 0, "query_key_value").normalized_extents,
            (std::array<int64_t, 4>{2560, 6144, 1, 1}));
}
TEST(SparkDescriptor, RejectsMalformedFactsTransactionally) {
  using Mutation = std::function<void(model::GgufCatalog &)>;
  for (const auto &mutate : std::vector<Mutation>{
           [](auto &c) {
             Field(c, "attention.sliding_window_pattern").elements.pop_back();
           },
           [](auto &c) {
             Field(c, "attention.sliding_window_pattern").elements[0] =
                 uint64_t(1);
           },
           [](auto &c) {
             Field(c, "attention.sliding_window_pattern").elements_omitted =
                 true;
           },
           [](auto &c) {
             Field(c, "rope.dimension_count_swa").scalar = uint64_t(257);
           },
           [](auto &c) {
             Field(c, "attention.head_count_kv").scalar = uint64_t(3);
           },
           [](auto &c) {
             Field(c, "attention.sliding_window").scalar = uint64_t(0);
           },
           [](auto &c) {
             c.metadata.erase(
                 std::remove_if(c.metadata.begin(), c.metadata.end(),
                                [](const auto &f) {
                                  return f.key == "spark2_5.rope.freq_base_swa";
                                }),
                 c.metadata.end());
           },
           [](auto &c) { c.tensors.front().normalized_extents[0]++; },
           [](auto &c) {
             auto t = c.tensors[2];
             t.name = "blk.0.attn_q_norm.weight";
             c.tensors.push_back(t);
           },
           [](auto &c) {
             auto t = c.tensors[2];
             t.name = "blk.0.post_attention_norm.weight";
             c.tensors.push_back(t);
           },
           [](auto &c) {
             auto t = c.tensors[2];
             t.name = "blk.0.attn_qkv.bias";
             c.tensors.push_back(t);
           }}) {
    auto c = testing::Spark25Catalog();
    mutate(c);
    model::ModelDescriptor m;
    m.name = "preserve";
    std::string error;
    EXPECT_FALSE(model::LoadModelDescriptor(c, &m, &error));
    EXPECT_EQ(m.name, "preserve");
    EXPECT_FALSE(error.empty());
  }
}
TEST(SparkConfiguration, DerivesGeometryPerOperandTagsAndOrdinaryState) {
  struct Shape {
    uint64_t e, l, qh, kh, f;
    bool tied;
  };
  for (const auto &s : std::vector<Shape>{{1024, 1, 4, 1, 256, true},
                                          {1536, 2, 8, 2, 3072, false},
                                          {2560, 36, 16, 4, 10240, true},
                                          {4096, 64, 32, 8, 32768, false},
                                          {32768, 2, 128, 32, 256, false}}) {
    auto c = testing::Spark25Catalog(s.e, s.l, s.qh, s.kh, s.f, s.tied);
    model::ModelDescriptor m;
    std::string error;
    ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
    ConfigurationFacts facts;
    ASSERT_TRUE(
        DeriveConfiguration(m, "fixture.gguf", {17, 1025}, &facts, &error))
        << error;
    auto fact = [&](const std::string &key, uint64_t value) {
      EXPECT_NE(
          facts.source.find("@" + key + " = " + std::to_string(value) + " :"),
          std::string::npos)
          << key;
    };
    fact("spark2_5.model.hidden_size", s.e);
    fact("spark2_5.model.query_size", s.qh * 256);
    fact("spark2_5.model.kv_size", s.kh * 256);
    fact("spark2_5.model.output_tag", s.tied ? 14 : 12);
    fact("spark2_5.model.tied_output", s.tied);
    fact("loomify.serving.layer_count", s.l);
    fact("loomify.serving.kv_size", s.kh * 256);
    fact("loomify.serving.cache_capacity", 1025);
    fact("loomify.serving.pool_capacity", 1088);
    fact("loomify.serving.cache_base", 1284);
    fact("loomify.serving.trailing_state_bytes", 0);
    EXPECT_EQ(facts.parameters.size(), 3 + 8 * s.l);
    EXPECT_NE(facts.source.find("@loomify.config.layer_attention"),
              std::string::npos);
    EXPECT_NE(facts.source.find("@loomify.config.layer_window"),
              std::string::npos);
    EXPECT_NE(facts.source.find("@loomify.config.layer_ffn"),
              std::string::npos);
    EXPECT_EQ(m.training_context, 1048576u);
  }
}
TEST(SparkConfiguration, InheritsAndVariesEffectiveRotaryWidths) {
  for (uint64_t width : {32, 64, 128, 256}) {
    auto c = testing::Spark25Catalog();
    Field(c, "rope.dimension_count").scalar = width;
    Field(c, "rope.dimension_count_swa").scalar = width;
    model::ModelDescriptor m;
    std::string error;
    ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
    for (auto &l : m.layers)
      l.rope_width = 0;
    ConfigurationFacts facts;
    EXPECT_TRUE(DeriveConfiguration(m, "fixture.gguf", {1, 64}, &facts, &error))
        << error;
  }
}
TEST(SparkConfiguration, PreservesMatrixRankForOneHead) {
  auto c = testing::Spark25Catalog(1024, 1, 1, 1, 256, true, 256);
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture.gguf", {1, 64}, &facts, &error))
      << error;
  EXPECT_EQ(Tensor(m, 0, "attention_gate").rank, 2u);
  auto bad = m;
  Tensor(bad, 0, "attention_gate").rank = 1;
  EXPECT_FALSE(
      DeriveConfiguration(bad, "fixture.gguf", {1, 64}, &facts, &error));
}
TEST(SparkConfiguration,
     RejectsInvalidDescriptorsWithoutPublishingPartialFacts) {
  auto c = testing::Spark25Catalog();
  model::ModelDescriptor good;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &good, &error));
  using Mutation = std::function<void(model::ModelDescriptor &)>;
  for (const auto &mutate : std::vector<Mutation>{
           [](auto &m) { m.embedding_width = 768; },
           [](auto &m) { m.vocabulary_size = 255; },
           [](auto &m) { m.layers[0].query_heads = 15; },
           [](auto &m) { m.layers[0].attention_ordinal = 1; },
           [](auto &m) { m.layers[1].rope_width = 63; },
           [](auto &m) { m.layers[0].rope_width = 258; },
           [](auto &m) { m.rope.scaling_factor = 2; },
           [](auto &m) { Tensor(m, 0, "query_key_value").byte_size++; },
           [](auto &m) {
             Tensor(m, 0, "ffn_up").absolute_offset = uint64_t(INT64_MAX);
           },
           [](auto &m) { Tensor(m, 0, "ffn_gate").storage_type = 2; },
           [](auto &m) { m.layers[0].tensors.pop_back(); },
           [](auto &m) {
             m.layers[0].tensors.push_back(m.layers[0].tensors[0]);
           },
           [](auto &m) { m.layers.resize(65); }}) {
    auto m = good;
    mutate(m);
    ConfigurationFacts facts;
    facts.source = "preserve";
    EXPECT_FALSE(
        DeriveConfiguration(m, "fixture.gguf", {8, 1024}, &facts, &error));
    EXPECT_EQ(facts.source, "preserve");
    EXPECT_TRUE(facts.parameters.empty());
    EXPECT_FALSE(error.empty());
  }
  auto wide = testing::Spark25Catalog(1024, 2, 128, 4, 32768);
  ASSERT_TRUE(model::LoadModelDescriptor(wide, &good, &error));
  for (const char *role : {"attention_output", "ffn_down"}) {
    auto m = good;
    auto &t = Tensor(m, 0, role);
    t.storage_type = 14;
    ASSERT_TRUE(model::EncodedTensorByteSize(
        14, t.normalized_extents[0], t.normalized_extents[1], &t.byte_size));
    ConfigurationFacts facts;
    EXPECT_FALSE(
        DeriveConfiguration(m, "fixture.gguf", {8, 1024}, &facts, &error));
  }
}
#ifdef LOOMIFY_SPARK_REAL_ASSETS
TEST(SparkConfiguration, ActualPinnedHeaderPreservesFactsAndAllBindings) {
  const auto *path = std::getenv("LOOMIFY_TEST_SPARK_GGUF");
  ASSERT_NE(path, nullptr);
  FILE *f = std::fopen(path, "rb");
  ASSERT_NE(f, nullptr);
  model::GgufCatalog c;
  std::string error;
  const bool read = model::ReadGgufCatalog(f, &c, &error);
  std::fclose(f);
  ASSERT_TRUE(read) << error;
  model::ModelDescriptor m;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  EXPECT_EQ(c.tensors.size(), 290u);
  EXPECT_EQ(m.layers.size(), 36u);
  EXPECT_EQ(m.training_context, 1048576u);
  for (size_t i = 0; i < m.layers.size(); ++i) {
    EXPECT_EQ(m.layers[i].attention_window, i % 4 == 3 ? 0u : 512u);
    EXPECT_EQ(m.layers[i].rope_width, i % 4 == 3 ? 64u : 256u);
  }
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(m, "model.gguf", {128, 1024}, &facts, &error))
      << error;
  EXPECT_EQ(facts.parameters.size(), 291u);
  EXPECT_EQ(Tensor(m, 0, "query_key_value").byte_size, 12902400u);
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
