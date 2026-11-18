// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/qwen3_fixture.h"
#include "tools/model/gguf_catalog.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <map>
namespace hrx::loomify_gguf {
namespace {
TEST(Qwen3Configuration,
     ClassicQ4UsesActualFormatsAndIndependentE2560QueryWidth) {
  auto model = testing::Qwen3ModelFacts(2560);
  uint64_t next = 4096;
  std::map<std::string, model::TensorRecord> aliases;
  auto relocate = [&](model::TensorRecord &tensor) {
    const auto existing = aliases.find(tensor.name);
    if (existing != aliases.end()) {
      tensor = existing->second;
      return;
    }
    ASSERT_TRUE(model::EncodedTensorByteSize(
        tensor.storage_type, tensor.normalized_extents[0],
        tensor.normalized_extents[1], &tensor.byte_size));
    tensor.absolute_offset = next;
    next += (tensor.byte_size + 255) / 256 * 256;
    aliases.emplace(tensor.name, tensor);
  };
  for (auto &binding : model.globals)
    relocate(binding.tensor);
  for (auto &layer : model.layers) {
    layer.query_heads = 32;
    layer.feed_forward_width = 9728;
    for (auto &binding : layer.tensors) {
      auto &tensor = binding.tensor;
      if (tensor.storage_type != 0) {
        tensor.storage_type = binding.role == "ffn_down" ? 3 : 2;
        if (binding.role == "query")
          tensor.normalized_extents[1] = 4096;
        if (binding.role == "attention_output")
          tensor.normalized_extents[0] = 4096;
        if (binding.role == "ffn_gate" || binding.role == "ffn_up")
          tensor.normalized_extents[1] = 9728;
        if (binding.role == "ffn_down")
          tensor.normalized_extents[0] = 9728;
      }
      relocate(tensor);
    }
  }
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(
      DeriveConfiguration(model, "classic-q4", {17, 512}, &facts, &error))
      << error;
  EXPECT_NE(facts.source.find("@qwen3.model.hidden_size = 2560"),
            std::string::npos);
  EXPECT_NE(facts.source.find("@qwen3.model.query_size = 4096"),
            std::string::npos);
  EXPECT_NE(
      facts.source.find("@ggml.linear_q4_0_q8_1_x4.output_capacity = 9728"),
      std::string::npos);
  EXPECT_NE(
      facts.source.find("@ggml.linear_q4_1_q8_1_x4.output_capacity = 2560"),
      std::string::npos);
  for (const auto &binding : facts.parameters) {
    if (!binding.layer)
      continue;
    if (binding.role == "ffn_down")
      EXPECT_EQ(binding.tensor.storage_type, 3u);
    else if (binding.tensor.storage_type != 0)
      EXPECT_EQ(binding.tensor.storage_type, 2u);
  }
}

TEST(Qwen3Configuration, IndependentQueryWidthAndDataOnlyBindings) {
  for (const auto hidden : {1024u, 1536u, 2048u}) {
    const auto m = testing::Qwen3ModelFacts(hidden);
    ConfigurationFacts f;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(m, "asset", {17, 513}, &f, &error))
        << error;
    EXPECT_NE(
        f.source.find("@qwen3.model.hidden_size = " + std::to_string(hidden)),
        std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.model.query_size = 2048"),
              std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.model.vocabulary_size = 151936"),
              std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.attention.pool_capacity = 576"),
              std::string::npos);
    EXPECT_EQ(f.source.find("command."), std::string::npos);
    EXPECT_EQ(f.source.find("kernel."), std::string::npos);
    EXPECT_EQ(f.parameters.size(), 3 + 11 * m.layers.size());
    const std::vector<std::string> globals{"token_embedding", "output",
                                           "output_norm"};
    const std::vector<std::string> roles{
        "input_norm", "post_norm", "query_norm", "key_norm",
        "query",      "key",       "value",      "attention_output",
        "ffn_gate",   "ffn_up",    "ffn_down"};
    for (size_t i = 0; i < f.parameters.size(); ++i) {
      const auto &binding = f.parameters[i];
      if (i < globals.size()) {
        EXPECT_EQ(binding.role, globals[i]);
        EXPECT_FALSE(binding.layer);
      } else {
        EXPECT_EQ(binding.role, roles[(i - globals.size()) % roles.size()]);
        EXPECT_EQ(binding.layer, (i - globals.size()) / roles.size());
      }
    }
    for (const auto &p : f.parameters) {
      const auto &source = p.layer ? m.layers[*p.layer].tensors : m.globals;
      const auto t =
          std::find_if(source.begin(), source.end(),
                       [&](const auto &b) { return b.role == p.role; });
      ASSERT_NE(t, source.end());
      EXPECT_EQ(p.tensor.name, t->tensor.name);
      EXPECT_EQ(p.source_asset, "asset");
      EXPECT_EQ(p.tensor.absolute_offset, t->tensor.absolute_offset);
      EXPECT_EQ(p.tensor.byte_size, t->tensor.byte_size);
      EXPECT_EQ(p.tensor.storage_type, t->tensor.storage_type);
      EXPECT_EQ(p.tensor.normalized_extents, t->tensor.normalized_extents);
    }
  }
}
TEST(Qwen3Configuration,
     RejectsUnsupportedOrInconsistentInputsWithoutReplacingOutput) {
  for (int mutation = 0; mutation < 12; ++mutation) {
    auto m = testing::Qwen3ModelFacts();
    if (mutation == 0)
      m.layers[0].tensors.erase(m.layers[0].tensors.begin() + 2);
    if (mutation == 1)
      m.layers[0].tensors[4].tensor.normalized_extents[1] = 1024;
    if (mutation == 2)
      m.layers[0].query_heads = 8;
    if (mutation == 3)
      m.layers[0].tensors[8].tensor.storage_type = 14;
    if (mutation == 4)
      m.rope.width = 64;
    if (mutation == 5)
      m.output_tied_to_embedding = false;
    if (mutation == 6)
      m.layers[1].attention_window = 512;
    if (mutation == 7)
      m.embedding_width = 2432;
    if (mutation == 8)
      m.layers[0].tensors[2].tensor.storage_type = 14;
    if (mutation == 9)
      m.layers[0].tensors[3].tensor.normalized_extents[0] = 256;
    if (mutation == 10)
      m.layers[0].tensors[2].tensor.byte_size -= 4;
    if (mutation == 11)
      m.layers[0].tensors.erase(m.layers[0].tensors.begin() + 3);
    ConfigurationFacts out;
    out.source = "unchanged";
    std::string error;
    EXPECT_FALSE(DeriveConfiguration(m, "asset", {}, &out, &error)) << mutation;
    EXPECT_EQ(out.source, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen3Configuration, MixedLayerTablesAndNeutralPreparationFacts) {
  auto m = testing::Qwen3ModelFacts(1024, 6);
  // Match the real Q4/Q6 transitions without making the fixture's filename a
  // selector.
  for (auto &layer : m.layers)
    for (auto &binding : layer.tensors)
      if (binding.role == "value" || binding.role == "ffn_down") {
        binding.tensor.storage_type =
            layer.index == 2 || layer.index == 3 || layer.index == 4 ? 12 : 14;
        ASSERT_TRUE(model::EncodedTensorByteSize(
            binding.tensor.storage_type, binding.tensor.normalized_extents[0],
            binding.tensor.normalized_extents[1], &binding.tensor.byte_size));
      }
  ConfigurationFacts out;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {17, 513}, &out, &error))
      << error;
  for (const auto &[name, value] :
       std::vector<std::pair<const char *, uint64_t>>{{"hidden_size", 1024},
                                                      {"layer_count", 6},
                                                      {"kv_size", 1024},
                                                      {"history_capacity", 129},
                                                      {"token_capacity", 17},
                                                      {"cache_capacity", 513},
                                                      {"pool_capacity", 576},
                                                      {"cache_base", 1284},
                                                      {"map_origin", 2048}}) {
    const std::string type =
        std::string(name) == "cache_base" || std::string(name) == "map_origin"
            ? "offset"
            : "index";
    EXPECT_NE(out.source.find("config.def @loomify.serving." +
                              std::string(name) + " = " +
                              std::to_string(value) + " : " + type + "\n"),
              std::string::npos)
        << name;
  }
  EXPECT_NE(out.source.find(
                "@qwen38.greedy_argmax.output_capacity = 151936 : index\n"),
            std::string::npos);
  EXPECT_EQ(out.source.find("@qwen3.greedy_argmax.output_capacity"),
            std::string::npos);
  // Inspect every tuple element, including encoded byte extents, in both
  // lookups.
  const auto check_row = [&](const std::string &lookup, size_t layer,
                             const std::vector<uint64_t> &values) {
    const auto start = out.source.find("@loomify.config." + lookup);
    ASSERT_NE(start, std::string::npos);
    const auto text =
        out.source.substr(start, out.source.find("\n}\n", start) - start);
    const auto variable = layer + 1 == m.layers.size()
                              ? std::string("%v0_")
                              : "%c" + std::to_string(layer) + "_";
    for (size_t j = 0; j < values.size(); ++j)
      EXPECT_NE(text.find(variable + std::to_string(j) + " = index.constant " +
                          std::to_string(values[j]) + " : index\n"),
                std::string::npos)
          << lookup << " layer=" << layer << " field=" << j;
  };
  for (const auto &layer : m.layers) {
    const auto tensor =
        [&](const std::string &role) -> const model::TensorRecord & {
      return std::find_if(layer.tensors.begin(), layer.tensors.end(),
                          [&](const auto &b) { return b.role == role; })
          ->tensor;
    };
    const auto &q = tensor("query"), &k = tensor("key"), &v = tensor("value"),
               &o = tensor("attention_output"), &g = tensor("ffn_gate"),
               &u = tensor("ffn_up"), &d = tensor("ffn_down");
    const uint32_t varied =
        layer.index == 2 || layer.index == 3 || layer.index == 4 ? 12 : 14;
    EXPECT_EQ(v.storage_type, varied);
    EXPECT_EQ(d.storage_type, varied);
    check_row("layer_attention", layer.index,
              {q.storage_type, k.storage_type, v.storage_type, o.storage_type,
               q.byte_size, k.byte_size, v.byte_size, o.byte_size});
    check_row("layer_ffn", layer.index,
              {layer.feed_forward_width, g.storage_type, u.storage_type,
               d.storage_type, g.byte_size, u.byte_size, d.byte_size});
  }
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {256, 513}, &out, &error))
      << error;
  EXPECT_NE(
      out.source.find("@loomify.serving.history_capacity = 256 : index\n"),
      std::string::npos);
  EXPECT_NE(out.source.find("@loomify.serving.cache_base = 1792 : offset\n"),
            std::string::npos);
}
#ifdef LOOMIFY_QWEN3_REAL_ASSETS
TEST(Qwen3ConfigurationRealAssets, ReadsActualCheckpointAndAllNamedRanges) {
  const char *path = std::getenv("HRX_QWEN3_GGUF");
  ASSERT_NE(path, nullptr) << "explicit immutable Qwen3 GGUF required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file);
  std::setvbuf(file.get(), nullptr, _IONBF, 0);
  model::GgufCatalog catalog;
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
  ASSERT_EQ(m.family, "qwen3");
  ASSERT_EQ(m.layers.size(), 28u);
  EXPECT_EQ(m.embedding_width, 1024u);
  EXPECT_EQ(m.layers.front().query_heads * m.key_head_width, 2048u);
  ConfigurationFacts out;
  ASSERT_TRUE(DeriveConfiguration(m, path, {128, 512}, &out, &error)) << error;
  EXPECT_EQ(out.parameters.size(), 311u);
  for (const auto &binding : out.parameters) {
    const auto *tensor = model::FindTensor(catalog, binding.tensor.name);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(binding.tensor.absolute_offset, tensor->absolute_offset);
    EXPECT_EQ(binding.tensor.byte_size, tensor->byte_size);
    EXPECT_EQ(binding.tensor.storage_type, tensor->storage_type);
  }
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
