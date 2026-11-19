// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/qwen3moe_fixture.h"
#include "tools/model/gguf_catalog.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <string>

namespace hrx::loomify_gguf {
namespace {
model::TensorBinding &Binding(model::ModelDescriptor &m,
                              const std::string &role) {
  auto &bindings = m.layers.front().tensors;
  auto found = std::find_if(bindings.begin(), bindings.end(),
                            [&](const auto &b) { return b.role == role; });
  if (found == bindings.end())
    std::abort();
  return *found;
}
void ExpectPreserved(const model::GgufCatalog &catalog,
                     const ConfigurationFacts &facts) {
  ASSERT_EQ(facts.parameters.size(), catalog.tensors.size());
  std::set<std::string> names;
  for (const auto &p : facts.parameters) {
    EXPECT_TRUE(names.insert(p.tensor.name).second) << p.tensor.name;
    const auto *t = model::FindTensor(catalog, p.tensor.name);
    ASSERT_NE(t, nullptr) << p.tensor.name;
    EXPECT_EQ(p.tensor.rank, t->rank);
    EXPECT_EQ(p.tensor.normalized_extents, t->normalized_extents);
    EXPECT_EQ(p.tensor.absolute_offset, t->absolute_offset);
    EXPECT_EQ(p.tensor.byte_size, t->byte_size);
    EXPECT_EQ(p.tensor.storage_type, t->storage_type);
  }
}
void ExpectConfiguration(const model::ModelDescriptor &m,
                         const ConfigurationFacts &facts, uint32_t head_tag) {
  EXPECT_NE(facts.source.find("@qwen3.model.output_tag = " +
                              std::to_string(head_tag) + " : index"),
            std::string::npos);
  EXPECT_NE(facts.source.find("@qwen3.model.vocabulary_size = 151936"),
            std::string::npos);
  EXPECT_NE(facts.source.find("@loomify.config.layer_experts(%layer: index) -> "
                              "(index, index, index, index, index, index)"),
            std::string::npos);
  EXPECT_EQ(facts.source.find("layer_ffn"), std::string::npos);
  EXPECT_EQ(facts.source.find("command."), std::string::npos);
  ASSERT_TRUE(m.routed_experts);
  EXPECT_EQ(m.routed_experts->expert_count, 128u);
  EXPECT_EQ(m.routed_experts->experts_per_token, 8u);
  EXPECT_EQ(m.routed_experts->feed_forward_width, 768u);
}
#ifndef LOOMIFY_QWEN3MOE_REAL_ASSETS
TEST(Qwen3MoeReader, PreservesOrdinaryAttentionAndTrueExpertRanks) {
  for (bool q8 : {false, true}) {
    auto catalog = testing::qwen3moe::Catalog(48, q8);
    // Explicit zero is present in an actual Coder publication.
    catalog.metadata.push_back(
        {"qwen3moe.expert_shared_count", 4, uint64_t{0}});
    model::ModelDescriptor m;
    ConfigurationFacts facts;
    std::string error;
    ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
    ASSERT_EQ(m.family, "qwen3moe");
    ASSERT_EQ(m.layers.size(), 48u);
    ASSERT_EQ(catalog.tensors.size(), 579u);
    for (uint32_t i = 0; i < m.layers.size(); ++i) {
      EXPECT_EQ(m.layers[i].kind, model::LayerKind::kAttention);
      EXPECT_EQ(m.layers[i].index, i);
      EXPECT_EQ(m.layers[i].attention_ordinal, i);
      EXPECT_EQ(m.layers[i].feed_forward_width, 5472u);
    }
    ASSERT_TRUE(
        DeriveConfiguration(m, "fixture.gguf", {17, 513}, &facts, &error))
        << error;
    ExpectPreserved(catalog, facts);
    ExpectConfiguration(m, facts, q8 ? 8 : 14);
    EXPECT_EQ(Binding(m, "expert_router").tensor.byte_size, 1048576u);
    EXPECT_EQ(Binding(m, "expert_gate").tensor.rank, 3u);
    EXPECT_EQ(Binding(m, "expert_gate").tensor.byte_size, 113246208u);
    EXPECT_EQ(Binding(m, "expert_down").tensor.byte_size, 113246208u);
    if (!q8) {
      const auto &last = m.layers.back().tensors;
      const auto down =
          std::find_if(last.begin(), last.end(),
                       [](const auto &b) { return b.role == "expert_down"; });
      ASSERT_NE(down, last.end());
      EXPECT_EQ(down->tensor.byte_size, 165150720u);
    }
  }
}
TEST(Qwen3MoeReader, RejectsMissingWrongRankShapeAndUnboundFeatures) {
  for (int trial = 0; trial < 15; ++trial) {
    SCOPED_TRACE(trial);
    auto c = testing::qwen3moe::Catalog();
    auto &gate = *std::find_if(c.tensors.begin(), c.tensors.end(), [](auto &t) {
      return t.name == "blk.0.ffn_gate_exps.weight";
    });
    switch (trial) {
    case 0:
      gate.name = "missing.weight";
      break;
    case 1:
      gate.rank = 2;
      break;
    case 2:
      gate.normalized_extents[2] = 127;
      break;
    case 3:
      gate.normalized_extents[3] = 2;
      break;
    case 4:
      c.tensors.push_back(gate);
      break;
    case 5:
      c.metadata.erase(
          std::remove_if(c.metadata.begin(), c.metadata.end(),
                         [](const auto &r) {
                           return r.key ==
                                  "qwen3moe.expert_feed_forward_length";
                         }),
          c.metadata.end());
      break;
    case 6:
    case 7:
    case 8:
    case 9: {
      const char *key[] = {"expert_shared_count", "expert_group_count",
                           "expert_group_used_count", "nextn_predict_layers"};
      c.metadata.push_back(
          {std::string("qwen3moe.") + key[trial - 6], 4, uint64_t{1}});
      break;
    }
    default: {
      const char *name[] = {"blk.0.ffn_gate_up_exps.weight",
                            "blk.0.ffn_gate_shexp.weight",
                            "blk.0.ffn_exp_probs_b.bias", "mtp.0.weight",
                            "blk.0.ffn_down_exps.bias"};
      auto extra = gate;
      extra.name = name[trial - 10];
      c.tensors.push_back(extra);
      break;
    }
    }
    model::ModelDescriptor out;
    out.name = "unchanged";
    std::string error;
    EXPECT_FALSE(model::LoadModelDescriptor(c, &out, &error));
    EXPECT_EQ(out.name, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen3MoeReader, RejectsUnsupportedRoutingMetadata) {
  const std::vector<model::MetadataRecord> unsupported{
      {"qwen3moe.expert_gating_func", 4, uint64_t{2}},
      {"qwen3moe.expert_weights_scale", 6, 2.0},
      {"qwen3moe.expert_weights_norm", 7, false},
      {"qwen3moe.expert_shared_feed_forward_length", 4, uint64_t{768}},
      {"qwen3moe.leading_dense_block_count", 4, uint64_t{1}}};
  for (const auto &field : unsupported) {
    SCOPED_TRACE(field.key);
    auto c = testing::qwen3moe::Catalog();
    c.metadata.push_back(field);
    model::ModelDescriptor out;
    out.name = "unchanged";
    std::string error;
    EXPECT_FALSE(model::LoadModelDescriptor(c, &out, &error));
    EXPECT_EQ(out.name, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen3MoeConfiguration, IndependentExpertTagsAndLookupColumnOrder) {
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(
      model::LoadModelDescriptor(testing::qwen3moe::Catalog(), &m, &error))
      << error;
  // Deliberately use independently selected codecs, including down-only Q5_0.
  for (auto [role, tag] : std::vector<std::pair<const char *, uint32_t>>{
           {"expert_gate", 13}, {"expert_up", 8}, {"expert_down", 6}}) {
    auto &t = Binding(m, role).tensor;
    t.storage_type = tag;
    ASSERT_TRUE(model::EncodedTensorByteSize(
        tag, t.normalized_extents[0],
        t.normalized_extents[1] * t.normalized_extents[2], &t.byte_size));
  }
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {}, &facts, &error)) << error;
  const std::array<uint64_t, 6> values{
      13,
      8,
      6,
      Binding(m, "expert_gate").tensor.byte_size,
      Binding(m, "expert_up").tensor.byte_size,
      Binding(m, "expert_down").tensor.byte_size};
  for (size_t column = 0; column < values.size(); ++column)
    EXPECT_NE(facts.source.find("%c0_" + std::to_string(column) +
                                " = index.constant " +
                                std::to_string(values[column]) + " : index"),
              std::string::npos)
        << column;
}
TEST(Qwen3MoeConfiguration,
     RejectsMalformedBindingsWithoutPublishingPartialFacts) {
  model::ModelDescriptor good;
  std::string error;
  ASSERT_TRUE(
      model::LoadModelDescriptor(testing::qwen3moe::Catalog(), &good, &error))
      << error;
  for (int trial = 0; trial < 22; ++trial) {
    SCOPED_TRACE(trial);
    auto m = good;
    auto &b = Binding(m, "expert_gate");
    auto &t = b.tensor;
    switch (trial) {
    case 0:
      b.role = "missing";
      break;
    case 1:
      m.layers.front().tensors.push_back(b);
      break;
    case 2:
      t.name = "blk.0.ffn_up_exps.weight";
      break;
    case 3:
      t.rank = 2;
      break;
    case 4:
      t.rank = 4;
      break;
    case 5:
      t.normalized_extents[0] = 0;
      break;
    case 6:
      t.normalized_extents[0] = 2047;
      break;
    case 7:
      t.normalized_extents[1] = 767;
      break;
    case 8:
      t.normalized_extents[2] = 127;
      break;
    case 9:
      t.normalized_extents[3] = 2;
      break;
    case 10:
      --t.byte_size;
      break;
    case 11:
      t.absolute_offset = std::numeric_limits<int64_t>::max();
      break;
    case 12:
      t.normalized_extents[1] = std::numeric_limits<int64_t>::max();
      break;
    case 13:
      t.storage_type = 0;
      break;
    case 14:
      Binding(m, "expert_router").tensor.storage_type = 8;
      break;
    case 15:
      m.routed_experts.reset();
      break;
    case 16:
      m.routed_experts->expert_count = 64;
      break;
    case 17:
      m.routed_experts->experts_per_token = 4;
      break;
    case 18:
      m.routed_experts->feed_forward_width = 1024;
      break;
    case 19:
      m.layers[1].index = 0;
      break;
    case 20:
      m.layers[1].index = 2;
      break;
    case 21:
      Binding(m, "expert_up").tensor.storage_type = 6;
      break;
    }
    ConfigurationFacts out;
    out.source = "unchanged";
    out.specialized_decode = true;
    out.parameters.push_back({"sentinel", {}, "sentinel", {}});
    EXPECT_FALSE(DeriveConfiguration(m, "bad", {}, &out, &error));
    EXPECT_EQ(out.source, "unchanged");
    EXPECT_TRUE(out.specialized_decode);
    ASSERT_EQ(out.parameters.size(), 1u);
    EXPECT_EQ(out.parameters[0].role, "sentinel");
    EXPECT_FALSE(error.empty());
  }
}
#endif
#ifdef LOOMIFY_QWEN3MOE_REAL_ASSETS
void CheckActualHeader(const char *env, bool q8, uint64_t dense_width) {
  SCOPED_TRACE(env);
  const char *path = std::getenv(env);
  ASSERT_NE(path, nullptr) << "explicit immutable GGUF fixture required: "
                           << env;
  ASSERT_NE(*path, '\0') << env;
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file) << path;
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::GgufCatalog c;
  model::ModelDescriptor m;
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &c, &error)) << error;
  ASSERT_EQ(c.architecture, "qwen3moe");
  ASSERT_EQ(c.tensors.size(), 579u);
  ASSERT_TRUE(model::LoadModelDescriptor(c, &m, &error)) << error;
  ASSERT_EQ(m.layers.size(), 48u);
  EXPECT_EQ(m.embedding_width, 2048u);
  EXPECT_EQ(m.vocabulary_size, 151936u);
  EXPECT_EQ(m.training_context, 262144u);
  EXPECT_EQ(m.key_head_width, 128u);
  EXPECT_EQ(m.value_head_width, 128u);
  EXPECT_EQ(m.rope.base, 1e7);
  EXPECT_NEAR(m.rms_epsilon, 1e-6, 1e-12);
  EXPECT_FALSE(m.output_tied_to_embedding);
  ASSERT_TRUE(DeriveConfiguration(m, path, {128, 512}, &facts, &error))
      << error;
  ExpectConfiguration(m, facts, q8 ? 8 : 14);
  ExpectPreserved(c, facts);
  EXPECT_EQ(Binding(m, "expert_router").tensor.byte_size, 1048576u);
  unsigned q6_values = 0, q6_down = 0;
  for (const auto &layer : m.layers) {
    SCOPED_TRACE(layer.index);
    EXPECT_EQ(layer.kind, model::LayerKind::kAttention);
    EXPECT_EQ(layer.query_heads, 32u);
    EXPECT_EQ(layer.kv_heads, 4u);
    EXPECT_EQ(layer.feed_forward_width, dense_width);
    ASSERT_EQ(layer.tensors.size(), 12u);
    for (const auto &binding : layer.tensors) {
      const auto &t = binding.tensor;
      const auto &role = binding.role;
      if (role == "expert_gate" || role == "expert_up") {
        EXPECT_EQ(t.rank, 3u);
        EXPECT_EQ(t.normalized_extents,
                  (std::array<int64_t, 4>{2048, 768, 128, 1}));
        EXPECT_EQ(t.storage_type, 12u);
        EXPECT_EQ(t.byte_size, 113246208u);
      } else if (role == "expert_down") {
        EXPECT_EQ(t.rank, 3u);
        EXPECT_EQ(t.normalized_extents,
                  (std::array<int64_t, 4>{768, 2048, 128, 1}));
        EXPECT_TRUE(t.storage_type == 12 || (!q8 && t.storage_type == 14));
        q6_down += t.storage_type == 14;
        EXPECT_EQ(t.byte_size, t.storage_type == 14 ? 165150720u : 113246208u);
      } else if (role == "value") {
        EXPECT_TRUE(q8 ? t.storage_type == 8
                       : t.storage_type == 12 || t.storage_type == 14);
        q6_values += t.storage_type == 14;
      } else if (role == "query" || role == "key" ||
                 role == "attention_output") {
        EXPECT_EQ(t.storage_type, q8 ? 8u : 12u);
      } else {
        EXPECT_EQ(t.storage_type, 0u);
      }
    }
  }
  EXPECT_EQ(q6_values, q8 ? 0u : 24u);
  EXPECT_EQ(q6_down, q8 ? 0u : 24u);
  for (const auto &g : m.globals) {
    if (g.role == "token_embedding")
      EXPECT_EQ(g.tensor.storage_type, q8 ? 8u : 12u);
    if (g.role == "output")
      EXPECT_EQ(g.tensor.storage_type, q8 ? 8u : 14u);
  }
}
TEST(Qwen3MoeConfigurationRealAssets, Instruct2507HeaderAndAll579Bindings) {
  CheckActualHeader("HRX_QWEN3MOE_GGUF", false, 6144);
}
TEST(Qwen3MoeConfigurationRealAssets, GgmlCoderHeaderAndAll579Bindings) {
  CheckActualHeader("HRX_QWEN3CODER_GGML_GGUF", true, 6144);
}
TEST(Qwen3MoeConfigurationRealAssets, UnslothCoderRetains5472DenseMetadata) {
  CheckActualHeader("HRX_QWEN3CODER_UNSLOTH_GGUF", false, 5472);
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
