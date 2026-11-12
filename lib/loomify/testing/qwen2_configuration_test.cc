// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/qwen2_fixture.h"
#include "tools/model/tensor_names.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
const model::TensorRecord *
Role(const std::vector<model::TensorBinding> &bindings,
     const std::string &role) {
  auto it = std::find_if(bindings.begin(), bindings.end(),
                         [&](const auto &b) { return b.role == role; });
  return it == bindings.end() ? nullptr : &it->tensor;
}
TEST(Qwen2Descriptor, DefaultsBiasRolesMixedFormatsAndTiedHead) {
  auto catalog = testing::qwen2::Catalog();
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
  EXPECT_EQ(catalog.tensors.size(), 338u);
  EXPECT_EQ(m.family, "qwen2");
  EXPECT_EQ(m.embedding_width, 1536u);
  EXPECT_EQ(m.vocabulary_size, 151936u);
  EXPECT_EQ(m.training_context, 32768u);
  EXPECT_EQ(m.key_head_width, 128u);
  EXPECT_EQ(m.value_head_width, 128u);
  EXPECT_EQ(m.rope.width, 128u);
  EXPECT_EQ(m.rope.base, 1e6);
  EXPECT_EQ(m.rope.scaling_factor, 1);
  EXPECT_TRUE(m.output_tied_to_embedding);
  EXPECT_FALSE(m.ssm);
  ASSERT_EQ(m.layers.size(), 28u);
  for (const auto &layer : m.layers) {
    ASSERT_EQ(layer.tensors.size(), 12u);
    EXPECT_EQ(layer.query_heads, 12u);
    EXPECT_EQ(layer.kv_heads, 2u);
    EXPECT_EQ(layer.feed_forward_width, 8960u);
    EXPECT_EQ(layer.kind, model::LayerKind::kAttention);
    EXPECT_EQ(layer.attention_ordinal, layer.index);
    EXPECT_FALSE(layer.recurrent_ordinal);
    const auto *q = Role(layer.tensors, "query_bias");
    const auto *k = Role(layer.tensors, "key_bias");
    const auto *v = Role(layer.tensors, "value_bias");
    ASSERT_TRUE(q && k && v);
    EXPECT_EQ(q->normalized_extents[0], 1536);
    EXPECT_EQ(k->normalized_extents[0], 256);
    EXPECT_EQ(v->normalized_extents[0], 256);
    EXPECT_EQ(q->storage_type, 0u);
    EXPECT_EQ(Role(layer.tensors, "query_norm"), nullptr);
    EXPECT_EQ(Role(layer.tensors, "key_norm"), nullptr);
  }
  // Explicit transition witnesses; the filename is not an encoding selector.
  for (const auto &[layer, tag] : std::vector<std::pair<size_t, uint32_t>>{
           {1, 14}, {2, 12}, {4, 12}, {5, 14}}) {
    EXPECT_EQ(Role(m.layers[layer].tensors, "value")->storage_type, tag);
    EXPECT_EQ(Role(m.layers[layer].tensors, "ffn_down")->storage_type, tag);
  }
}
TEST(Qwen2Descriptor, RejectsMissingOrWrongBiasAndUnknownGraphFeatures) {
  for (int mutation = 0; mutation < 8; ++mutation) {
    auto catalog = testing::qwen2::Catalog();
    auto bias = std::find_if(
        catalog.tensors.begin(), catalog.tensors.end(),
        [](const auto &t) { return t.name == "blk.0.attn_k.bias"; });
    ASSERT_NE(bias, catalog.tensors.end());
    if (mutation == 0)
      catalog.tensors.erase(bias);
    if (mutation == 1)
      bias->normalized_extents[0] = 128;
    if (mutation == 2) {
      auto extra = *bias;
      extra.name = "blk.0.attn_output.bias";
      catalog.tensors.push_back(extra);
    }
    if (mutation == 3) {
      auto extra = *bias;
      extra.name = "blk.0.attn_q_norm.weight";
      catalog.tensors.push_back(extra);
    }
    if (mutation == 4) {
      auto extra = *bias;
      extra.name = "blk.0.attn_qkv.weight";
      catalog.tensors.push_back(extra);
    }
    if (mutation == 5)
      catalog.metadata.push_back(
          {"qwen2.attention.sliding_window", 4, uint64_t{128}});
    if (mutation == 6)
      catalog.metadata.push_back(
          {"qwen2.rope.scaling.type", 8, std::string("yarn")});
    if (mutation == 7)
      catalog.metadata.push_back(
          {"qwen2.rope.dimension_count", 4, uint64_t{64}});
    model::ModelDescriptor out;
    out.name = "unchanged";
    std::string error;
    EXPECT_FALSE(model::LoadModelDescriptor(catalog, &out, &error)) << mutation;
    EXPECT_EQ(out.name, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen2Configuration, DataOnlyFactsAndAll339BindingsFollowInputs) {
  for (const auto hidden : {1536u, 2048u}) {
    const auto m = testing::qwen2::ModelFacts(hidden);
    ConfigurationFacts out;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(m, "fixture.gguf", {17, 513}, &out, &error))
        << error;
    EXPECT_EQ(out.parameters.size(), 339u);
    const std::vector<std::string> globals{"token_embedding", "output",
                                           "output_norm"};
    const std::vector<std::string> roles{
        "input_norm",       "post_norm", "query_bias", "key_bias",
        "value_bias",       "query",     "key",        "value",
        "attention_output", "ffn_gate",  "ffn_up",     "ffn_down"};
    for (size_t i = 0; i < out.parameters.size(); ++i) {
      const auto &binding = out.parameters[i];
      if (i < globals.size()) {
        EXPECT_EQ(binding.role, globals[i]);
        EXPECT_FALSE(binding.layer);
      } else {
        EXPECT_EQ(binding.role, roles[(i - globals.size()) % roles.size()]);
        EXPECT_EQ(binding.layer, (i - globals.size()) / roles.size());
      }
    }
    for (const auto &binding : out.parameters) {
      const auto *tensor =
          Role(binding.layer ? m.layers[*binding.layer].tensors : m.globals,
               binding.role);
      ASSERT_NE(tensor, nullptr);
      EXPECT_EQ(binding.source_asset, "fixture.gguf");
      EXPECT_EQ(binding.tensor.name, tensor->name);
      EXPECT_EQ(binding.tensor.absolute_offset, tensor->absolute_offset);
      EXPECT_EQ(binding.tensor.byte_size, tensor->byte_size);
      EXPECT_EQ(binding.tensor.storage_type, tensor->storage_type);
      EXPECT_EQ(binding.tensor.normalized_extents, tensor->normalized_extents);
    }
    EXPECT_NE(
        out.source.find("@qwen2.model.hidden_size = " + std::to_string(hidden)),
        std::string::npos);
    EXPECT_NE(
        out.source.find("@qwen2.model.query_size = " + std::to_string(hidden)),
        std::string::npos);
    EXPECT_NE(out.source.find("@qwen2.model.kv_size = 256 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@qwen2.model.layer_count = 28 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@qwen2.attention.pool_capacity = 576 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.serving.hidden_size = " +
                              std::to_string(hidden)),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.serving.kv_size = 256 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.config.layer_attention"),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.config.layer_ffn"), std::string::npos);
    EXPECT_NE(out.source.find(
                  "@qwen38.greedy_argmax.output_capacity = 151936 : index"),
              std::string::npos);
    EXPECT_EQ(out.source.find("qwen28."), std::string::npos);
    EXPECT_EQ(out.source.find("command."), std::string::npos);
    EXPECT_EQ(out.source.find("kernel."), std::string::npos);
  }
  ConfigurationFacts shortened;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::qwen2::ModelFacts(1536, 2, 9216),
                                  "fixture", {}, &shortened, &error))
      << error;
  EXPECT_EQ(shortened.parameters.size(), 27u);
  EXPECT_NE(shortened.source.find("@qwen2.model.layer_count = 2 : index"),
            std::string::npos);
}
TEST(Qwen2Configuration,
     RejectsMalformedConstructedBindingsWithoutReplacingOutput) {
  for (int mutation = 0; mutation < 11; ++mutation) {
    auto m = testing::qwen2::ModelFacts();
    auto bias =
        std::find_if(m.layers[0].tensors.begin(), m.layers[0].tensors.end(),
                     [](const auto &b) { return b.role == "query_bias"; });
    ASSERT_NE(bias, m.layers[0].tensors.end());
    if (mutation == 0)
      bias->tensor.storage_type = 14;
    if (mutation == 1)
      bias->tensor.absolute_offset = uint64_t(INT64_MAX);
    if (mutation == 2)
      bias->tensor.name = "blk.1.attn_q.bias";
    if (mutation == 3)
      m.layers[0].tensors.push_back(*bias);
    if (mutation == 4)
      m.output_tied_to_embedding = false;
    if (mutation == 5)
      m.layers[0].attention_window = 256;
    if (mutation == 6)
      m.rope.scaling_factor = 2;
    if (mutation == 7) {
      auto norm = *bias;
      norm.role = "query_norm";
      m.layers[0].tensors.push_back(norm);
    }
    if (mutation == 8)
      bias->tensor.byte_size -= 4;
    if (mutation == 9)
      m.layers[0].tensors.erase(bias);
    if (mutation == 10)
      bias->tensor.normalized_extents[0] = 128;
    ConfigurationFacts out;
    out.source = "unchanged";
    std::string error;
    EXPECT_FALSE(DeriveConfiguration(m, "fixture", {}, &out, &error))
        << mutation;
    EXPECT_EQ(out.source, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen2Configuration, MixedLayerTablesAndNeutralPreparationFacts) {
  auto m = testing::qwen2::ModelFacts(1536, 6);
  ConfigurationFacts out;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {17, 513}, &out, &error))
      << error;
  for (const auto &[name, value] :
       std::vector<std::pair<const char *, uint64_t>>{{"hidden_size", 1536},
                                                      {"layer_count", 6},
                                                      {"kv_size", 256},
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
  EXPECT_EQ(out.source.find("@qwen2.greedy_argmax.output_capacity"),
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
#ifdef LOOMIFY_QWEN2_REAL_ASSETS
TEST(Qwen2ConfigurationRealAssets,
     Actual338TensorsAllRangesAndFormatTransitions) {
  const char *path = std::getenv("LOOMIFY_TEST_QWEN2_GGUF");
  ASSERT_NE(path, nullptr) << "explicit immutable Qwen2 GGUF required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file);
  std::setvbuf(file.get(), nullptr, _IONBF, 0);
  model::GgufCatalog catalog;
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  EXPECT_LE(catalog.bytes_read, catalog.header_end);
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
  ASSERT_EQ(catalog.tensors.size(), 338u);
  ASSERT_EQ(m.layers.size(), 28u);
  EXPECT_EQ(m.embedding_width, 1536u);
  EXPECT_EQ(m.vocabulary_size, 151936u);
  EXPECT_TRUE(m.output_tied_to_embedding);
  ConfigurationFacts out;
  ASSERT_TRUE(DeriveConfiguration(m, path, {128, 512}, &out, &error)) << error;
  ASSERT_EQ(out.parameters.size(), 339u);
  std::map<uint32_t, size_t> physical_formats;
  for (const auto &tensor : catalog.tensors)
    ++physical_formats[tensor.storage_type];
  EXPECT_EQ(physical_formats[0], 141u);
  EXPECT_EQ(physical_formats[12], 168u);
  EXPECT_EQ(physical_formats[14], 29u);
  for (const auto &binding : out.parameters) {
    const auto *tensor = model::FindTensor(catalog, binding.tensor.name);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(binding.tensor.absolute_offset, tensor->absolute_offset);
    EXPECT_EQ(binding.tensor.byte_size, tensor->byte_size);
    EXPECT_EQ(binding.tensor.storage_type, tensor->storage_type);
    EXPECT_EQ(binding.tensor.rank, tensor->rank);
    EXPECT_EQ(binding.tensor.normalized_extents, tensor->normalized_extents);
  }
  for (const auto &[layer, tag] : std::vector<std::pair<size_t, uint32_t>>{
           {1, 14}, {2, 12}, {4, 12}, {5, 14}}) {
    EXPECT_EQ(Role(m.layers[layer].tensors, "value")->storage_type, tag);
    EXPECT_EQ(Role(m.layers[layer].tensors, "ffn_down")->storage_type, tag);
  }
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
