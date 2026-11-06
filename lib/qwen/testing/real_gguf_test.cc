// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/qwen/structure.h"
#include "tools/model/qwen_configuration.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
namespace hrx::qwen {
TEST(RealQwen, ProductionReaderOwnedDescriptorAndFullGraph) {
  const char *path = std::getenv("LOOMIFY_QWEN_TEST_GGUF");
  ASSERT_NE(path, nullptr)
      << "explicit immutable Qwen Q5_K_XL fixture required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     &std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::ModelDescriptor facts;
  std::string error;
  ASSERT_TRUE(model::ExtractQwenConfiguration(file.get(), &facts, &error))
      << error;
  EXPECT_EQ(std::fseek(file.get(), 0, SEEK_SET), 0);
  file.reset();
  ASSERT_EQ(facts.layers.size(), 24u);
  EXPECT_EQ(facts.embedding_width, 1024u);
  EXPECT_EQ(facts.vocabulary_size, 248320u);
  std::shared_ptr<const QwenStructure> owner;
  ASSERT_TRUE(BuildQwenStructure(std::move(facts), {128, 1, 1, 96, "real-qwen"},
                                 &owner, &error))
      << error;
  ASSERT_EQ(owner->graph().invocations().size(), 26u);
  EXPECT_EQ(owner->parameters().size(),
            321u); // tied output has its own coordinate.
  size_t gdn = 0, attention = 0, q5 = 0;
  for (const auto &layer : owner->inputs().layers) {
    if (layer.kind == model::LayerKind::kGatedDeltaNet) {
      EXPECT_EQ(layer.ordinal, gdn++);
    } else {
      EXPECT_EQ(layer.ordinal, attention++);
    }
  }
  for (const auto &a : owner->parameters()) {
    const auto *t = owner->parameter(a.coordinate);
    ASSERT_NE(t, nullptr);
    if (t->tensor.storage_type == 13) {
      ++q5;
      EXPECT_EQ(owner->graph().value(a.value)->tensorType()->encoding(),
                operations::Encoding::Q5_K);
    }
  }
  EXPECT_EQ(gdn, 18u);
  EXPECT_EQ(attention, 6u);
  EXPECT_GT(q5, 0u);
  model::ModelDescriptor sentinel;
  sentinel.name = "unchanged";
  std::unique_ptr<FILE, decltype(&std::fclose)> invalid(std::tmpfile(),
                                                        &std::fclose);
  ASSERT_TRUE(invalid);
  ASSERT_EQ(std::setvbuf(invalid.get(), nullptr, _IONBF, 0), 0);
  EXPECT_FALSE(
      model::ExtractQwenConfiguration(invalid.get(), &sentinel, &error));
  EXPECT_EQ(sentinel.name, "unchanged");
}
} // namespace hrx::qwen
