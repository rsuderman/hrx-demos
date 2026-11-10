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

// Header-only fixture with caller-selected wire fields. Payload is sparse so
// range validation can cover large offsets without allocating model bytes.
std::string TensorHeader(uint32_t type, const std::vector<uint64_t> &shape,
                         uint64_t relative = 0, uint32_t alignment = 32) {
  std::string s = "GGUF";
  Number(s, 3, 4);
  Number(s, 1, 8);
  Number(s, 2, 8);
  String(s, "general.architecture");
  Number(s, 8, 4);
  String(s, "llama");
  String(s, "general.alignment");
  Number(s, 4, 4);
  Number(s, alignment, 4);
  String(s, "blk.0.ffn_down.weight");
  Number(s, shape.size(), 4);
  for (auto d : shape)
    Number(s, d, 8);
  Number(s, type, 4);
  Number(s, relative, 8);
  return s;
}
bool InspectSparse(const std::string &header, uint64_t size,
                   GgufCatalog *catalog, std::string *error) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(),
                                                     std::fclose);
  if (!file || std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0 ||
      std::fwrite(header.data(), 1, header.size(), file.get()) !=
          header.size() ||
      fseeko(file.get(), static_cast<int64_t>(size - 1), SEEK_SET) != 0 ||
      fputc(0, file.get()) == EOF) {
    *error = "unable to prepare sparse fixture";
    return false;
  }
  return ReadGgufCatalog(file.get(), catalog, error);
}

TEST(GgufCatalog, FirstModelEncodingsRetainRankIdentityAndSparseStorageSpans) {
  struct Case {
    uint32_t type;
    const char *name;
    uint64_t bytes;
  };
  const Case cases[] = {{0, "f32", 2048}, {1, "f16", 1024},  {6, "q5_0", 352},
                        {8, "q8_0", 544}, {12, "q4_K", 288}, {13, "q5_K", 352},
                        {14, "q6_K", 420}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    const uint64_t relative = uint64_t{1} << 32;
    const auto header = TensorHeader(test.type, {256, 2, 1}, relative);
    const auto data = (header.size() + 31) / 32 * 32;
    GgufCatalog catalog;
    std::string error;
    ASSERT_TRUE(
        InspectSparse(header, data + relative + test.bytes, &catalog, &error))
        << error;
    ASSERT_EQ(catalog.tensors.size(), 1);
    const auto &tensor = catalog.tensors[0];
    EXPECT_EQ(tensor.name, "blk.0.ffn_down.weight");
    EXPECT_EQ(tensor.rank, 3);
    EXPECT_EQ(tensor.normalized_extents,
              (std::array<int64_t, 4>{256, 2, 1, 1}));
    EXPECT_EQ(tensor.storage_type, test.type);
    EXPECT_EQ(tensor.storage_name, test.name);
    EXPECT_EQ(tensor.absolute_offset, data + relative);
    EXPECT_EQ(tensor.byte_size, test.bytes);
    EXPECT_EQ(catalog.bytes_read, header.size());
  }
}

TEST(GgufCatalog, RejectsBadStorageGeometryAndTruncationTransactionally) {
  std::vector<std::string> cases = {
      TensorHeader(99, {256, 2}),
      TensorHeader(12, {255, 2}),
      TensorHeader(14, {128, 2}),
      TensorHeader(13, {255, 2}),
      TensorHeader(13, {257, 2}),
      TensorHeader(13, {256, UINT64_MAX}),
      TensorHeader(0, {INT64_MAX, 4}),
      TensorHeader(0, {INT64_MAX}), // encoded byte multiplication overflow
      TensorHeader(0, {256, 2}, 1),
      TensorHeader(0, {256}, 0, 0),
      TensorHeader(0, {256}, 0, 3),
      TensorHeader(0, {256, 0}),
      TensorHeader(0, {256, UINT64_MAX})};
  const auto good = TensorHeader(12, {256, 2});
  // Every prefix includes truncated strings, dimensions, type and offset.
  for (size_t i = 0; i < good.size(); ++i)
    cases.push_back(good.substr(0, i));
  for (const auto &bytes : cases) {
    GgufCatalog catalog;
    catalog.name = "unchanged";
    TensorRecord sentinel;
    sentinel.name = "existing";
    catalog.tensors.push_back(sentinel);
    std::string error;
    EXPECT_FALSE(Inspect(bytes, &catalog, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(catalog.name, "unchanged");
    ASSERT_EQ(catalog.tensors.size(), 1);
    EXPECT_EQ(catalog.tensors[0].name, "existing");
  }
}

TEST(GgufCatalog, RejectsOverlappingInBoundsSpansWithoutPublishing) {
  // Each F32 [8,2] tensor occupies 64 bytes but starts only 32 bytes apart.
  const auto bytes = Fixture(2, 8);
  GgufCatalog catalog;
  catalog.name = "unchanged";
  std::string error;
  ASSERT_FALSE(InspectSparse(bytes, bytes.size() + 64, &catalog, &error));
  EXPECT_NE(error.find("overlapping"), std::string::npos) << error;
  EXPECT_EQ(catalog.name, "unchanged");
  EXPECT_TRUE(catalog.tensors.empty());
}

TEST(GgufCatalog, ReportsArithmeticOverflowBeforeFileRangeFailure) {
  for (const auto &shape :
       std::vector<std::vector<uint64_t>>{{INT64_MAX, 4}, {INT64_MAX}}) {
    GgufCatalog catalog;
    std::string error;
    EXPECT_FALSE(Inspect(TensorHeader(0, shape), &catalog, &error));
    EXPECT_NE(error.find("tensor size overflow"), std::string::npos) << error;
  }
}

TEST(GgufCatalog, Q5BlockRowsAndExactEncodedRange) {
  for (uint64_t width : {256, 512}) {
    const auto header = TensorHeader(13, {width, 3});
    const auto data = (header.size() + 31) / 32 * 32;
    const auto bytes = width / 256 * 3 * 176;
    GgufCatalog catalog;
    std::string error;
    ASSERT_TRUE(InspectSparse(header, data + bytes, &catalog, &error)) << error;
    EXPECT_EQ(catalog.tensors[0].byte_size, bytes);
    EXPECT_FALSE(InspectSparse(header, data + bytes - 1, &catalog, &error));
    EXPECT_EQ(catalog.tensors[0].byte_size, bytes);
  }
}

TEST(GgufCatalog, EnforcesHeaderLimitBeforeAllocatingOversizedStrings) {
  std::string bytes = "GGUF";
  Number(bytes, 3, 4);
  Number(bytes, 0, 8);
  Number(bytes, 1, 8);
  Number(bytes, 64 * 1024 * 1024, 8);
  GgufCatalog catalog;
  catalog.name = "unchanged";
  std::string error;
  EXPECT_FALSE(InspectSparse(bytes, 128 * 1024 * 1024, &catalog, &error));
  EXPECT_NE(error.find("64 MiB"), std::string::npos);
  EXPECT_EQ(catalog.name, "unchanged");
}

TEST(GgufCatalog, RejectsInvalidMetadataLengthsCountsAndTypes) {
  std::string base = "GGUF";
  Number(base, 3, 4);
  Number(base, 0, 8);
  Number(base, 1, 8);
  std::vector<std::string> cases;
  auto s = base;
  Number(s, UINT64_MAX, 8); // key length
  cases.push_back(s);
  for (auto type : {9u, 99u}) {
    s = base;
    String(s, "array");
    Number(s, 9, 4);
    Number(s, type, 4); // nested array or unknown scalar
    Number(s, 1, 8);
    cases.push_back(s);
  }
  s = base;
  String(s, "array");
  Number(s, 9, 4);
  Number(s, 4, 4);
  Number(s, UINT64_MAX, 8);
  cases.push_back(s);
  s = base;
  s[16] = 1;
  s[17] = 16; // 4097 metadata entries
  cases.push_back(s);
  for (const auto &bytes : cases) {
    GgufCatalog catalog;
    catalog.architecture = "unchanged";
    std::string error;
    EXPECT_FALSE(Inspect(bytes, &catalog, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(catalog.architecture, "unchanged");
  }
}

TEST(GgufCatalog, RetainsCompleteCatalogAfterFileCloses) {
  GgufCatalog catalog;
  std::string error;
  ASSERT_TRUE(Inspect(Fixture(40), &catalog, &error)) << error;
  EXPECT_EQ(catalog.version, 3);
  EXPECT_EQ(catalog.architecture, "llama");
  ASSERT_EQ(catalog.tensors.size(), 40);
  EXPECT_EQ(catalog.tensors[39].name, "test.weight.39");
  EXPECT_EQ(catalog.tensors[0].rank, 2);
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
TEST(GgufCatalog, Q5ZeroBlockExtentsAreDistinctFromQ5K) {
  uint64_t bytes = 99;
  EXPECT_TRUE(EncodedTensorByteSize(6, 1152, 1, &bytes));
  EXPECT_EQ(bytes, 792);
  EXPECT_TRUE(EncodedTensorByteSize(6, 6912, 1, &bytes));
  EXPECT_EQ(bytes, 4752);
  bytes = 99;
  EXPECT_FALSE(EncodedTensorByteSize(6, 33, 1, &bytes));
  EXPECT_EQ(bytes, 99);
  EXPECT_FALSE(EncodedTensorByteSize(6, 32, UINT64_MAX, &bytes));
  EXPECT_EQ(bytes, 99);
}
} // namespace
} // namespace hrx::model
