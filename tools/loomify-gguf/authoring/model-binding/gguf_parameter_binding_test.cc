// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/parameter_index.h"
#include "tools/model/gguf_catalog.h"

namespace hrx::loomify_gguf {
namespace {

void Number(std::vector<uint8_t>& bytes, uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i) {
    bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
  }
}

void String(std::vector<uint8_t>& bytes, const std::string& value) {
  Number(bytes, value.size(), 8);
  bytes.insert(bytes.end(), value.begin(), value.end());
}

struct Fixture {
  std::vector<uint8_t> bytes;
  size_t data_offset = 0;
};

// Both selected tensor offsets are nonzero, and the first tensor/padding contain
// neither tensor's expected content. Header order varies while key identity stays
// fixed. The catalog's GGML parser requires canonical packed tensor ordering.
Fixture MakeFixture(bool reverse_header = false) {
  Fixture fixture;
  auto& bytes = fixture.bytes;
  bytes = {'G', 'G', 'U', 'F'};
  Number(bytes, 3, 4);
  Number(bytes, 3, 8);
  Number(bytes, 2, 8);
  String(bytes, "general.architecture");
  Number(bytes, 8, 4);
  String(bytes, "llama");
  String(bytes, "general.alignment");
  Number(bytes, 4, 4);
  Number(bytes, 32, 4);
  const auto tensor = [&](const char* name, uint64_t offset) {
    String(bytes, name);
    Number(bytes, 1, 4);
    Number(bytes, 4, 8);
    Number(bytes, 0, 4);  // GGML F32.
    Number(bytes, offset, 8);
  };
  tensor("distractor.weight", 0);
  if (reverse_header) {
    tensor("blk.0.attn_norm.weight", 32);
    tensor("output_norm.weight", 64);
  } else {
    tensor("output_norm.weight", 32);
    tensor("blk.0.attn_norm.weight", 64);
  }
  while (bytes.size() % 32) bytes.push_back(0);
  fixture.data_offset = bytes.size();
  bytes.resize(fixture.data_offset + 96, 0xA5);
  // Independent F32 payload bit patterns: [1, 2, 3, 4] and [-1, -2, -3, -4].
  const std::array<uint32_t, 4> positive = {
      0x3F800000, 0x40000000, 0x40400000, 0x40800000};
  const std::array<uint32_t, 4> negative = {
      0xBF800000, 0xC0000000, 0xC0400000, 0xC0800000};
  for (size_t i = 0; i < 4; ++i) {
    for (unsigned b = 0; b < 4; ++b) {
      bytes[fixture.data_offset + (reverse_header ? 64 : 32) + i * 4 + b] =
          positive[i] >> (8 * b);
      bytes[fixture.data_offset + (reverse_header ? 32 : 64) + i * 4 + b] =
          negative[i] >> (8 * b);
    }
  }
  return fixture;
}

::testing::AssertionResult Ok(iree_status_t status) {
  if (iree_status_is_ok(status)) return ::testing::AssertionSuccess();
  const auto code = iree_status_code(status);
  iree_status_fprint(stderr, status);
  iree_status_ignore(status);
  return ::testing::AssertionFailure() << "IREE status " << code;
}

bool ReadCatalog(const Fixture& fixture, model::GgufCatalog* catalog,
                 std::string* error) {
  struct CloseFile {
    void operator()(FILE* file) const { std::fclose(file); }
  };
  std::unique_ptr<FILE, CloseFile> file(std::tmpfile());
  if (!file || std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0 ||
      std::fwrite(fixture.bytes.data(), 1, fixture.bytes.size(), file.get()) !=
          fixture.bytes.size()) {
    *error = "unable to write temporary GGUF";
    return false;
  }
  return model::ReadGgufCatalog(file.get(), catalog, error);
}

using Index = std::unique_ptr<iree_io_parameter_index_t,
                              decltype(&iree_io_parameter_index_release)>;
using File = std::unique_ptr<iree_io_file_handle_t,
                             decltype(&iree_io_file_handle_release)>;

TEST(GgufParameterBinding, CatalogAndRuntimeIndexSelectNamedBytes) {
  for (bool reverse_header : {false, true}) {
    SCOPED_TRACE(reverse_header);
    auto fixture = MakeFixture(reverse_header);
    model::GgufCatalog catalog;
    std::string error;
    ASSERT_TRUE(ReadCatalog(fixture, &catalog, &error)) << error;
    iree_io_parameter_index_t* raw_index = nullptr;
    ASSERT_TRUE(Ok(iree_io_parameter_index_create(iree_allocator_system(),
                                                &raw_index)));
    Index index(raw_index, iree_io_parameter_index_release);
    iree_io_file_handle_t* raw_file = nullptr;
    ASSERT_TRUE(Ok(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ,
        iree_make_byte_span(fixture.bytes.data(), fixture.bytes.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &raw_file)));
    File file(raw_file, iree_io_file_handle_release);
    ASSERT_TRUE(Ok(iree_io_parse_gguf_index(file.get(), index.get(),
                                         iree_allocator_system())));
    file.reset();  // The index retains its source handle, not its payload owner.

    const struct {
      const char* key;
      uint64_t relative_offset;
      uint32_t second_element;
    } cases[] = {{"output_norm.weight", reverse_header ? 64u : 32u, 0x40000000},
                 {"blk.0.attn_norm.weight", reverse_header ? 32u : 64u,
                  0xC0000000}};
    for (const auto& test : cases) {
      const auto* tensor = model::FindTensor(catalog, test.key);
      ASSERT_NE(tensor, nullptr);
      const iree_io_parameter_index_entry_t* entry = nullptr;
      ASSERT_TRUE(Ok(iree_io_parameter_index_lookup(
          index.get(), iree_make_cstring_view(test.key), &entry)));
      ASSERT_EQ(entry->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
      EXPECT_EQ(entry->length, 16u);
      EXPECT_EQ(entry->length, tensor->byte_size);
      EXPECT_EQ(entry->storage.file.offset, tensor->absolute_offset);
      EXPECT_EQ(entry->storage.file.offset,
                fixture.data_offset + test.relative_offset);
      EXPECT_EQ(entry->storage.file.minimum_alignment, 32u);
      const auto primitive =
          iree_io_file_handle_primitive(entry->storage.file.handle);
      ASSERT_EQ(primitive.type, IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION);
      // Provider parameter offsets are tensor-relative. Requesting byte four
      // must read this key's second element, not file byte four or root byte four.
      const size_t source = entry->storage.file.offset + 4;
      ASSERT_LE(source + 4, primitive.value.host_allocation.data_length);
      const auto* payload = primitive.value.host_allocation.data + source;
      const uint32_t actual = static_cast<uint32_t>(payload[0]) |
                              static_cast<uint32_t>(payload[1]) << 8 |
                              static_cast<uint32_t>(payload[2]) << 16 |
                              static_cast<uint32_t>(payload[3]) << 24;
      EXPECT_EQ(actual, test.second_element);
    }
    const iree_io_parameter_index_entry_t* missing = nullptr;
    iree_status_t status = iree_io_parameter_index_lookup(
        index.get(), IREE_SV("not-a-tensor"), &missing);
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
    iree_status_ignore(status);
  }
}

TEST(GgufParameterBinding, CatalogAndRuntimeRejectTruncatedTensorPayload) {
  auto fixture = MakeFixture();
  fixture.bytes.resize(fixture.bytes.size() - 17);
  model::GgufCatalog catalog;
  std::string error;
  EXPECT_FALSE(ReadCatalog(fixture, &catalog, &error));
  iree_io_parameter_index_t* raw_index = nullptr;
  ASSERT_TRUE(Ok(iree_io_parameter_index_create(iree_allocator_system(),
                                              &raw_index)));
  Index index(raw_index, iree_io_parameter_index_release);
  iree_io_file_handle_t* raw_file = nullptr;
  ASSERT_TRUE(Ok(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(fixture.bytes.data(), fixture.bytes.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &raw_file)));
  File file(raw_file, iree_io_file_handle_release);
  iree_status_t status = iree_io_parse_gguf_index(
      file.get(), index.get(), iree_allocator_system());
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

}  // namespace
}  // namespace hrx::loomify_gguf
