// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/spark2_5_fixture.h"
#include "tools/model/tensor_names.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
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
    EXPECT_NE(facts.source.find("@loomify.config.layer_rotary"),
              std::string::npos);
    EXPECT_NE(facts.source.find("@loomify.config.layer_ffn"),
              std::string::npos);
    EXPECT_EQ(m.training_context, 1048576u);
  }
}
TEST(SparkConfiguration, ConsumesActualEmbeddingAndCanonicalOperandFacts) {
  for (uint32_t embedding_tag : {12u, 13u, 14u}) {
    for (uint32_t projection_tag : {6u, 8u, 12u, 13u, 14u}) {
      auto c = testing::Spark25Catalog(1024, 2, 4, 1, 256, false);
      model::ModelDescriptor m;
      std::string error;
      ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
      auto encode = [&](model::TensorRecord &tensor, uint32_t tag) {
        tensor.storage_type = tag;
        ASSERT_TRUE(model::EncodedTensorByteSize(
            tag, tensor.normalized_extents[0], tensor.normalized_extents[1],
            &tensor.byte_size));
      };
      for (auto &binding : m.globals) {
        if (binding.role == "token_embedding")
          encode(binding.tensor, embedding_tag);
        if (binding.role == "output")
          encode(binding.tensor, projection_tag);
      }
      // Gate/up/down may have different encodings; the sequence uses each fact.
      encode(Tensor(m, 0, "query_key_value"), projection_tag);
      encode(Tensor(m, 0, "ffn_up"), projection_tag);
      ConfigurationFacts facts;
      ASSERT_TRUE(
          DeriveConfiguration(m, "fixture.gguf", {8, 1024}, &facts, &error))
          << error;
      EXPECT_NE(facts.source.find("@spark2_5.model.embedding_tag = " +
                                  std::to_string(embedding_tag) + " : index"),
                std::string::npos);
      EXPECT_NE(facts.source.find("@spark2_5.model.output_tag = " +
                                  std::to_string(projection_tag) + " : index"),
                std::string::npos);
      ASSERT_EQ(facts.parameters.size(), 19u);
      EXPECT_EQ(facts.parameters[0].tensor.storage_type, embedding_tag);
      EXPECT_EQ(facts.parameters[1].tensor.storage_type, projection_tag);
      const auto &qkv = Tensor(m, 0, "query_key_value");
      const auto found =
          std::find_if(facts.parameters.begin(), facts.parameters.end(),
                       [&](const auto &parameter) {
                         return parameter.tensor.name == qkv.name;
                       });
      ASSERT_NE(found, facts.parameters.end());
      EXPECT_EQ(found->tensor.byte_size, qkv.byte_size);
      EXPECT_EQ(found->tensor.storage_type, projection_tag);
    }
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
TEST(SparkConfiguration, DerivesIndependentLayerRotaryFacts) {
  auto c = testing::Spark25Catalog(1024, 3, 4, 1, 256, true);
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  m.layers[0].rope_base = 10000;
  m.layers[0].rope_width = 256;
  m.layers[1].rope_base = 100000;
  m.layers[1].rope_width = 128;
  m.layers[2].attention_window = 0;
  m.layers[2].rope_base = 2000000;
  m.layers[2].rope_width =
      0; // Inherit the model width, independently of window.
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture.gguf", {8, 1024}, &facts, &error))
      << error;
  EXPECT_NE(
      facts.source.find(
          "@loomify.config.layer_rotary(%layer: index) -> (index, index, f32)"),
      std::string::npos);
  for (size_t i = 0; i < m.layers.size(); ++i) {
    const auto &layer = m.layers[i];
    const auto width = layer.rope_width ? layer.rope_width : m.rope.width;
    const auto prefix =
        i + 1 == m.layers.size() ? "v0_" : "c" + std::to_string(i) + "_";
    EXPECT_NE(facts.source.find("%" + prefix + "w = index.constant " +
                                std::to_string(layer.attention_window) +
                                " : index"),
              std::string::npos);
    EXPECT_NE(facts.source.find("%" + prefix + "r = index.constant " +
                                std::to_string(width) + " : index"),
              std::string::npos);
    std::ostringstream step;
    step << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10)
         << float(-2 * std::log(layer.rope_base) / width);
    EXPECT_NE(facts.source.find("%" + prefix + "s = scalar.constant " +
                                step.str() + " : f32"),
              std::string::npos);
  }
  EXPECT_EQ(facts.source.find("rope_log_step_local"), std::string::npos);
  EXPECT_EQ(facts.source.find("rope_log_step_global"), std::string::npos);
  EXPECT_EQ(facts.source.find("layer_window"), std::string::npos);
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
  size_t mutation_index = 0;
  for (const auto &mutate : std::vector<Mutation>{
           [](auto &m) { m.embedding_width = 768; },
           [](auto &m) { m.vocabulary_size = 255; },
           [](auto &m) { m.layers[0].query_heads = 15; },
           [](auto &m) { m.layers[0].attention_ordinal = 1; },
           [](auto &m) { m.layers[1].rope_width = 63; },
           [](auto &m) { m.layers[0].rope_width = 258; },
           [](auto &m) { m.rope.scaling_factor = 2; },
           [](auto &m) {
             m.layers[0].rope_base = std::numeric_limits<double>::quiet_NaN();
           },
           [](auto &m) {
             m.layers[0].rope_base = std::numeric_limits<double>::infinity();
           },
           [](auto &m) {
             for (auto &binding : m.globals)
               if (binding.role == "output")
                 binding.tensor.absolute_offset++;
           },
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
    SCOPED_TRACE(mutation_index++);
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
TEST(SparkConfiguration, AttentionGateRetainsPerLayerF32AndQ4Facts) {
  auto c = testing::Spark25Catalog(2560, 3, 16, 4, 10240, true);
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  auto gate = [](model::ModelDescriptor &m) -> model::TensorRecord & {
    for (auto &binding : m.layers.at(0).tensors)
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
      [](const auto &p) { return p.role == "attention_gate" && p.layer == 0; });
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

TEST(SparkConfiguration,
     AttentionGateUsesExistingProjectionFormatsAndCapacities) {
  for (uint32_t tag : {0u, 6u, 8u, 12u, 13u, 14u}) {
    SCOPED_TRACE(tag);
    auto c = testing::Spark25Catalog(2560, 3, 16, 4, 10240, true);
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
    auto &layer = m.layers.at(0);
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
          return p.role == "attention_gate" && p.layer == 0;
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
