// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/testing/embedding_fixture.h"
#include "tools/loomify-gguf/embedding.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
namespace hrx::loomify_gguf {
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
class EmbeddingBuilderTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  }
  void TearDown() override {
    if (workspace)
      loomc_workspace_release(workspace);
    if (context)
      loomc_context_release(context);
  }
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
};
TEST_F(EmbeddingBuilderTest, RealGgufRolesDriveCompleteProgramAndDimensions) {
  for (const uint32_t width : {4, 8}) {
    model::GgufCatalog catalog;
    model::ModelDescriptor descriptor;
    std::string error;
    const auto bytes = testing::EmbeddingGguf(width, 3);
    ASSERT_TRUE(testing::ReadFixture(bytes, &catalog, &descriptor, &error))
        << error;
    const auto *tensor = model::FindTensor(catalog, "token_embd.weight");
    ASSERT_NE(tensor, nullptr);
    EXPECT_GT(tensor->absolute_offset, catalog.data_offset);
    std::vector<float> payload;
    ASSERT_TRUE(
        testing::ReadNamedF32(bytes, "token_embd.weight", &payload, &error))
        << error;
    ASSERT_EQ(payload.size(), width * 3);
    EXPECT_EQ(payload[width], 110);
    loomc_module_t *module = nullptr;
    ASSERT_TRUE(
        BuildEmbeddingProgram(descriptor, context, workspace, &module, &error))
        << error;
    std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> owner(
        module, loomc_module_release);
    for (const auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options = {};
      options.format = format;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          module, &options, loomc_allocator_system(), &source)));
      std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)>
          source_owner(source, loomc_source_release);
      auto contents = loomc_source_contents(source);
      if (format == LOOMC_SOURCE_FORMAT_TEXT) {
        std::string text(reinterpret_cast<const char *>(contents.data),
                         contents.data_length);
        EXPECT_NE(text.find("model.embedding"), std::string::npos);
        EXPECT_NE(text.find("token_embd.weight"), std::string::npos);
        EXPECT_EQ(text.find("output_norm.weight"), std::string::npos);
        EXPECT_NE(
            text.find("config.def @loomify.embedding.width = " + std::to_string(width) + " : index"),
            std::string::npos);
        EXPECT_NE(text.find("kernel.launch @loomify.embedding.f32"),
                  std::string::npos);
      }
      loomc_module_t *restored = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context, workspace, source, nullptr, loomc_allocator_system(),
          &restored, &result)));
      EXPECT_TRUE(result && loomc_result_succeeded(result));
      if (result)
        loomc_result_release(result);
      if (restored)
        loomc_module_release(restored);
    }
  }
}
TEST_F(EmbeddingBuilderTest, UnsupportedStorageAndRoleFailuresReturnNoModule) {
  model::GgufCatalog catalog;
  model::ModelDescriptor baseline;
  std::string error;
  ASSERT_TRUE(testing::ReadFixture(testing::EmbeddingGguf(), &catalog,
                                   &baseline, &error))
      << error;
  for (int variation = 0; variation < 7; ++variation) {
    auto descriptor = baseline;
    auto *table = &descriptor.globals[0];
    for (auto &b : descriptor.globals)
      if (b.role == "token_embedding")
        table = &b;
    switch (variation) {
    case 0:
      table->tensor.storage_type = 14;
      table->tensor.storage_name = "q6_k";
      break;
    case 1:
      table->role = "unknown";
      break;
    case 2:
      table->tensor.byte_size -= 4;
      break;
    case 3:
      ++descriptor.embedding_width;
      break;
    case 4:
      descriptor.family = "qwen35";
      break;
    case 5:
      descriptor.globals.push_back(*table);
      break;
    case 6:
      descriptor.embedding_width = UINT64_MAX;
      break;
    }
    loomc_module_t *module = reinterpret_cast<loomc_module_t *>(uintptr_t(1));
    EXPECT_FALSE(
        BuildEmbeddingProgram(descriptor, context, workspace, &module, &error));
    EXPECT_EQ(module, nullptr);
    EXPECT_FALSE(error.empty());
  }
}
} // namespace
} // namespace hrx::loomify_gguf
