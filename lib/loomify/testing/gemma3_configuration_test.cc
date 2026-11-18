// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/gemma3_fixture.h"
#include "tools/model/gguf_catalog.h"
#include "tools/model/model_descriptor.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <limits>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
using testing::gemma3::Catalog;
using testing::gemma3::Integer;
using testing::gemma3::Real;
void SetInteger(model::GgufCatalog &c, const char *key, uint64_t value) {
  for (auto &f : c.metadata)
    if (f.key == std::string("gemma3.") + key) {
      f.scalar = value;
      return;
    }
  Integer(c, key, value);
}
TEST(Gemma3Configuration,
     DerivesDistinctQueryWidthMixedFormatsAndTiedIdentity) {
  auto c = Catalog();
  model::ModelDescriptor m;
  std::string e;
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e)) << e;
  ASSERT_EQ(m.layers.size(), 6);
  EXPECT_TRUE(m.output_tied_to_embedding);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(m.layers[i].attention_window, i == 5 ? 0 : 512);
    EXPECT_EQ(m.layers[i].rope_base, i == 5 ? 1e6 : 1e4);
  }
  ConfigurationFacts f;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {128, 1024}, &f, &e)) << e;
  EXPECT_EQ(f.parameters.size(),
            81); // 2 globals, tied alias, 6*13 layer roles.
  EXPECT_NE(f.source.find("@gemma3.model.query_size = 1024"),
            std::string::npos);
  EXPECT_NE(f.source.find("@gemma3.model.hidden_size = 1152"),
            std::string::npos);
  EXPECT_NE(f.source.find("@ggml.linear_q5_0_q8_1_x4.output_capacity = 6912"),
            std::string::npos);
  EXPECT_EQ(f.source.find("command.program"), std::string::npos);
  std::map<std::string, uint32_t> formats;
  for (const auto &p : f.parameters)
    formats[p.tensor.name] = p.tensor.storage_type;
  for (int i = 0; i < 6; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    EXPECT_EQ(formats[p + "attn_v.weight"], i % 2 ? 8 : 6);
    EXPECT_EQ(formats[p + "ffn_down.weight"], i % 2 ? 12 : 14);
  }
  const auto input =
      std::find_if(f.parameters.begin(), f.parameters.end(),
                   [](const auto &p) { return p.role == "token_embedding"; });
  const auto output =
      std::find_if(f.parameters.begin(), f.parameters.end(),
                   [](const auto &p) { return p.role == "output"; });
  ASSERT_NE(input, f.parameters.end());
  ASSERT_NE(output, f.parameters.end());
  EXPECT_EQ(input->tensor.name, output->tensor.name);
  EXPECT_EQ(input->tensor.absolute_offset, output->tensor.absolute_offset);
  c = {};
  m = {};
  EXPECT_FALSE(f.source.empty());
  EXPECT_EQ(output->tensor.storage_type, 8);
}
TEST(Gemma3Configuration, WindowDefaultsOverridesAndUnsupportedMetadata) {
  auto c = Catalog();
  model::ModelDescriptor m;
  std::string e;
  c.metadata.erase(std::remove_if(c.metadata.begin(), c.metadata.end(),
                                  [](const auto &f) {
                                    return f.key ==
                                           "gemma3.attention.sliding_window";
                                  }),
                   c.metadata.end());
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e)) << e;
  for (auto &l : m.layers) {
    EXPECT_EQ(l.attention_window, 0);
    EXPECT_EQ(l.rope_base, 1e6);
  }
  Integer(c, "attention.sliding_window", 512);
  Integer(c, "attention.sliding_window_pattern", 2);
  Real(c, "rope.freq_base_swa", 20000);
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e)) << e;
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(m.layers[i].attention_window, i % 2 ? 0 : 512);
    EXPECT_EQ(m.layers[i].rope_base, i % 2 ? 1e6 : 20000);
  }
  SetInteger(c, "attention.sliding_window_pattern", 0);
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e)) << e;
  for (auto &l : m.layers)
    EXPECT_EQ(l.attention_window, 512);
  SetInteger(c, "attention.sliding_window", 0);
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e)) << e;
  for (auto &l : m.layers)
    EXPECT_EQ(l.attention_window, 0);
  auto missing_base = c;
  missing_base.metadata.erase(
      std::remove_if(
          missing_base.metadata.begin(), missing_base.metadata.end(),
          [](const auto &f) { return f.key == "gemma3.rope.freq_base"; }),
      missing_base.metadata.end());
  EXPECT_FALSE(model::LoadModelDescriptor(missing_base, &m, &e));
  EXPECT_NE(e.find("rope.freq_base"), std::string::npos);
  Real(c, "final_logit_softcapping", 30);
  m.name = "sentinel";
  EXPECT_FALSE(model::LoadModelDescriptor(c, &m, &e));
  EXPECT_EQ(m.name, "sentinel");
  EXPECT_NE(e.find("softcap"), std::string::npos);
  c = Catalog();
  model::MetadataRecord pattern;
  pattern.key = "gemma3.attention.sliding_window_pattern";
  pattern.source_type = 9;
  c.metadata.push_back(pattern);
  EXPECT_FALSE(model::LoadModelDescriptor(c, &m, &e));
  EXPECT_NE(e.find("scalar integer"), std::string::npos);
}
TEST(Gemma3Configuration, RejectsMissingNormWrongShapeEncodingAndInvalidRange) {
  auto c = Catalog();
  std::string e;
  model::ModelDescriptor m;
  c.tensors.erase(std::remove_if(c.tensors.begin(), c.tensors.end(),
                                 [](const auto &t) {
                                   return t.name ==
                                          "blk.0.post_ffw_norm.weight";
                                 }),
                  c.tensors.end());
  EXPECT_FALSE(model::LoadModelDescriptor(c, &m, &e));
  EXPECT_NE(e.find("post_ffw_norm"), std::string::npos);
  c = Catalog();
  for (auto &t : c.tensors)
    if (t.name == "blk.0.attn_output.weight")
      t.normalized_extents[0] = 1152;
  EXPECT_FALSE(model::LoadModelDescriptor(c, &m, &e));
  c = Catalog();
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e)) << e;
  ConfigurationFacts f;
  f.source = "sentinel";
  m.layers[0].tensors[0].tensor.byte_size++;
  EXPECT_FALSE(DeriveConfiguration(m, "fixture", {}, &f, &e));
  EXPECT_EQ(f.source, "sentinel");
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e));
  for (auto &b : m.layers[0].tensors)
    if (b.role == "query")
      b.tensor.storage_type = 13;
  EXPECT_FALSE(DeriveConfiguration(m, "fixture", {}, &f, &e));
  EXPECT_EQ(f.source, "sentinel");
}

TEST(Gemma3Configuration,
     DerivedLargeGeometryPreservesActualFormatsAndCacheWidth) {
  for (uint32_t tag : {12u, 14u, 8u}) {
    SCOPED_TRACE(tag);
    model::ModelDescriptor m;
    ConfigurationFacts f;
    std::string e;
    ASSERT_TRUE(
        model::LoadModelDescriptor(testing::gemma3::LargeCatalog(tag), &m, &e))
        << e;
    ASSERT_TRUE(DeriveConfiguration(m, "large-fixture", {128, 2048}, &f, &e))
        << e;
    EXPECT_EQ(f.parameters.size(), 627);
    for (const char *fact :
         {"@gemma3.model.hidden_size = 3840 : index",
          "@gemma3.model.vocabulary_size = 262208 : index",
          "@gemma3.model.layer_count = 48 : index",
          "@gemma3.model.query_heads = 16 : index",
          "@gemma3.model.kv_heads = 8 : index",
          "@gemma3.model.query_size = 4096 : index",
          "@gemma3.model.kv_size = 2048 : index",
          "@loomify.serving.kv_size = 2048 : index",
          "@gemma3.model.rope_factor_global = 8.000000000e+00 : f32",
          "@gemma3.attention.cache_base = 1284 : offset"})
      EXPECT_NE(f.source.find(fact), std::string::npos) << fact;
    const auto embedding_tag = tag == 8 ? 8 : 14;
    EXPECT_NE(f.source.find("@gemma3.model.embedding_weight_tag = " +
                            std::to_string(embedding_tag)),
              std::string::npos);
    EXPECT_NE(f.source.find("@gemma3.model.output_weight_tag = " +
                            std::to_string(embedding_tag)),
              std::string::npos);
    for (size_t i = 0; i < m.layers.size(); ++i) {
      EXPECT_EQ(m.layers[i].attention_window, i % 6 == 5 ? 0 : 1024);
      EXPECT_EQ(m.layers[i].rope_base, i % 6 == 5 ? 1e6 : 1e4);
    }
    for (const auto &binding : f.parameters) {
      if (binding.role == "output" || binding.role == "token_embedding") {
        EXPECT_EQ(binding.tensor.storage_type, embedding_tag);
        EXPECT_EQ(binding.tensor.byte_size,
                  tag == 8 ? 1069808640u : 825955200u);
      }
      if (binding.layer && binding.role == "query")
        EXPECT_EQ(binding.tensor.storage_type, tag);
    }
  }
}

TEST(Gemma3Configuration, RejectsInvalidLargeGroupingRotaryAndTiedExtents) {
  const auto valid = testing::gemma3::LargeModelFacts();
  std::string e;
  ConfigurationFacts f;
  f.source = "sentinel";
  auto reject = [&](const model::ModelDescriptor &m) {
    EXPECT_FALSE(DeriveConfiguration(m, "large-fixture", {128, 2048}, &f, &e));
    EXPECT_EQ(f.source, "sentinel");
  };
  for (double factor : {0., -1., std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
    auto m = valid;
    m.rope.scaling_factor = factor;
    reject(m);
  }
  for (const char *kind : {"none", "yarn"}) {
    auto m = valid;
    m.rope.scaling_kind = kind;
    reject(m);
  }
  for (uint32_t heads : {0u, 3u, 33u}) {
    auto m = valid;
    m.layers[0].query_heads = heads;
    reject(m);
  }
  auto m = valid;
  m.layers[47].kv_heads = 4;
  reject(m);
  m = valid;
  for (auto &b : m.globals)
    if (b.role == "output")
      b.tensor.absolute_offset++;
  reject(m);
  m = valid;
  for (auto &b : m.globals)
    if (b.role == "output")
      b.tensor.byte_size--;
  reject(m);
  m = valid;
  for (auto &b : m.layers[0].tensors)
    if (b.role == "query")
      b.tensor.normalized_extents[1] = 3840;
  reject(m);
}

TEST(Gemma3Configuration, ActualCheckpointLogicalReaderAndOwnedConfiguration) {
  const char *path = std::getenv("LOOMIFY_TEST_GEMMA_GGUF");
  if (!path || !*path)
    GTEST_SKIP() << "LOOMIFY_TEST_GEMMA_GGUF not supplied";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::GgufCatalog c;
  model::ModelDescriptor m;
  ConfigurationFacts f;
  std::string e;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &c, &e)) << e;
  EXPECT_EQ(c.tensors.size(), 340);
  EXPECT_LE(c.bytes_read, c.header_end);
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &e)) << e;
  ASSERT_EQ(m.layers.size(), 26);
  EXPECT_EQ(m.embedding_width, 1152);
  EXPECT_EQ(m.vocabulary_size, 262144);
  std::map<uint32_t, int> formats;
  for (auto &t : c.tensors)
    formats[t.storage_type]++;
  EXPECT_EQ(formats[0], 157);
  EXPECT_EQ(formats[6], 117);
  EXPECT_EQ(formats[8], 14);
  EXPECT_EQ(formats[12], 39);
  EXPECT_EQ(formats[14], 13);
  for (size_t i = 0; i < m.layers.size(); ++i)
    EXPECT_EQ(m.layers[i].attention_window, (i + 1) % 6 ? 512 : 0);
  ASSERT_TRUE(DeriveConfiguration(m, "actual-model", {128, 1024}, &f, &e)) << e;
  EXPECT_EQ(f.parameters.size(), 341);
  file.reset();
  c = {};
  m = {};
  EXPECT_NE(f.source.find("@gemma3.model.layer_count = 26"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf
