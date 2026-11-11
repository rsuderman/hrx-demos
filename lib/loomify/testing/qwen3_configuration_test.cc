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
namespace hrx::loomify_gguf {
namespace {
TEST(Qwen3Configuration, IndependentQueryWidthAndDataOnlyBindings) {
  for (const auto hidden : {1024u, 2048u}) {
    const auto m = testing::Qwen3ModelFacts(hidden);
    ConfigurationFacts f;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(m, "asset", {17, 513}, &f, &error)) << error;
    EXPECT_NE(f.source.find("@qwen3.model.hidden_size = " + std::to_string(hidden)), std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.model.query_size = 2048"), std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.model.vocabulary_size = 151936"), std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.attention.pool_capacity = 576"), std::string::npos);
    EXPECT_EQ(f.source.find("command."), std::string::npos);
    EXPECT_EQ(f.source.find("kernel."), std::string::npos);
    EXPECT_EQ(f.parameters.size(), 3 + 11 * m.layers.size());
    for (const auto &p : f.parameters) {
      const auto &source = p.layer ? m.layers[*p.layer].tensors : m.globals;
      const auto t = std::find_if(source.begin(), source.end(), [&](const auto &b) { return b.role == p.role; });
      ASSERT_NE(t, source.end());
      EXPECT_EQ(p.tensor.absolute_offset, t->tensor.absolute_offset);
      EXPECT_EQ(p.tensor.byte_size, t->tensor.byte_size);
      EXPECT_EQ(p.tensor.storage_type, t->tensor.storage_type);
      EXPECT_EQ(p.tensor.normalized_extents, t->tensor.normalized_extents);
    }
  }
}
TEST(Qwen3Configuration, RejectsUnsupportedOrInconsistentInputsWithoutReplacingOutput) {
  for (int mutation = 0; mutation < 7; ++mutation) {
    auto m = testing::Qwen3ModelFacts();
    if (mutation == 0) m.layers[0].tensors.erase(m.layers[0].tensors.begin() + 2);
    if (mutation == 1) m.layers[0].tensors[4].tensor.normalized_extents[1] = 1024;
    if (mutation == 2) m.layers[0].query_heads = 8;
    if (mutation == 3) m.layers[0].tensors[8].tensor.storage_type = 14;
    if (mutation == 4) m.rope.width = 64;
    if (mutation == 5) m.output_tied_to_embedding = false;
    if (mutation == 6) m.layers[1].attention_window = 512;
    ConfigurationFacts out;
    out.source = "unchanged";
    std::string error;
    EXPECT_FALSE(DeriveConfiguration(m, "asset", {}, &out, &error)) << mutation;
    EXPECT_EQ(out.source, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
#ifdef LOOMIFY_QWEN3_REAL_ASSETS
TEST(Qwen3ConfigurationRealAssets, ReadsActualCheckpointAndAllNamedRanges) {
  const char *path = std::getenv("HRX_QWEN3_GGUF");
  ASSERT_NE(path, nullptr) << "explicit immutable Qwen3 GGUF required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"), std::fclose);
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
