// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/model/gguf_catalog.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace hrx::model {
namespace {
void Number(std::string &bytes, uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    bytes.push_back(static_cast<char>(value >> (8 * i)));
}
void String(std::string &bytes, const std::string &value) {
  Number(bytes, value.size(), 8);
  bytes += value;
}
std::string Metadata(bool architecture = true, bool bad_alignment = false) {
  std::string bytes;
  String(bytes, architecture ? "general.architecture" : "wrong.architecture");
  Number(bytes, 8, 4);
  String(bytes, "llama");
  String(bytes, "general.name");
  Number(bytes, 8, 4);
  String(bytes, "small test");
  String(bytes, "general.alignment");
  Number(bytes, bad_alignment ? 8 : 4, 4);
  if (bad_alignment)
    String(bytes, "wrong");
  else
    Number(bytes, 32, 4);
  String(bytes, "tokenizer.ggml.tokens");
  Number(bytes, 9, 4); // array
  Number(bytes, 8, 4); // strings
  Number(bytes, 200, 8);
  for (int i = 0; i < 200; ++i)
    String(bytes, "private-token-content");
  return bytes;
}
std::string Fixture(unsigned tensors = 1, uint64_t dimension = 4,
                    unsigned rank = 2, bool architecture = true,
                    bool bad_alignment = false, uint64_t offset = 0) {
  std::string bytes = "GGUF";
  Number(bytes, 3, 4);
  Number(bytes, tensors, 8);
  Number(bytes, 4, 8);
  bytes += Metadata(architecture, bad_alignment);
  for (unsigned i = 0; i < tensors; ++i) {
    String(bytes, "test.weight." + std::to_string(i));
    Number(bytes, rank, 4);
    for (unsigned d = 0; d < rank; ++d)
      Number(bytes, d == 0 ? dimension : 2, 8);
    Number(bytes, 0, 4); // f32
    Number(bytes, offset + i * 32, 8);
  }
  while (bytes.size() % 32)
    bytes.push_back(0);
  bytes.append(tensors * 32, '\0');
  return bytes;
}
bool Inspect(const std::string &bytes, GgufCatalog *catalog,
             std::string *error) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(),
                                                     std::fclose);
  if (!file || std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0 ||
      std::fwrite(bytes.data(), 1, bytes.size(), file.get()) != bytes.size()) {
    *error = "unable to prepare temporary fixture";
    return false;
  }
  return ReadGgufCatalog(file.get(), catalog, error);
}

TEST(GgufCatalog, RetainsCompleteCatalogAfterFileCloses) {
  GgufCatalog catalog;
  std::string error;
  ASSERT_TRUE(Inspect(Fixture(40), &catalog, &error)) << error;
  EXPECT_EQ(catalog.version, 3);
  EXPECT_EQ(catalog.architecture, "llama");
  ASSERT_EQ(catalog.tensors.size(), 40);
  EXPECT_EQ(catalog.tensors[39].name, "test.weight.39");
  EXPECT_EQ(catalog.tensors[0].normalized_extents,
            (std::array<int64_t, 4>{4, 2, 1, 1}));
  EXPECT_EQ(catalog.tensors[0].absolute_offset, catalog.data_offset);
  EXPECT_EQ(catalog.tensors[0].byte_size, 32);
  EXPECT_LE(catalog.bytes_read, catalog.data_offset + 24);
  const auto *tokens = FindMetadata(catalog, "tokenizer.ggml.tokens");
  ASSERT_NE(tokens, nullptr);
  EXPECT_EQ(tokens->element_count, 200);
  EXPECT_TRUE(tokens->elements_omitted);
  EXPECT_TRUE(tokens->elements.empty());
}

TEST(GgufCatalog, RejectsMalformedInputsWithoutPublishingPartialData) {
  const auto valid = Fixture(40);
  auto bad_magic = valid;
  bad_magic[0] = 'X';
  auto bad_version = valid;
  bad_version[4] = 99;
  auto bad_counts = valid;
  for (unsigned i = 8; i < 16; ++i)
    bad_counts[i] = char(255);
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"bad magic", bad_magic},
      {"bad version", bad_version},
      {"truncated header", valid.substr(0, 12)},
      {"truncated metadata", valid.substr(0, 100)},
      {"truncated tensor span", valid.substr(0, valid.size() - 1)},
      {"required architecture", Fixture(1, 4, 2, false)},
      {"mistyped alignment", Fixture(1, 4, 2, true, true)},
      {"zero dimension", Fixture(1, 0)},
      {"negative dimension", Fixture(1, UINT64_MAX)},
      {"zero rank", Fixture(1, 4, 0)},
      {"oversized rank", Fixture(1, 4, 5)},
      {"overflow range", Fixture(1, 4, 2, true, false, UINT64_MAX)},
      {"bounded tensor count", bad_counts},
  };
  for (const auto &[label, bytes] : cases) {
    SCOPED_TRACE(label);
    GgufCatalog catalog;
    catalog.name = "unchanged";
    std::string error;
    EXPECT_FALSE(Inspect(bytes, &catalog, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(catalog.name, "unchanged");
    EXPECT_TRUE(catalog.tensors.empty());
  }
}
} // namespace
} // namespace hrx::model
