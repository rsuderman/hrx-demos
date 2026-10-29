// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_MODEL_GGUF_CATALOG_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_GGUF_CATALOG_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

namespace hrx::model {

// Numeric tags are the GGUF/ggml on-disk enum values, independent of headers.
using MetadataScalar =
    std::variant<uint64_t, int64_t, double, bool, std::string>;

struct MetadataRecord {
  std::string key;
  uint32_t source_type = 0;
  MetadataScalar scalar = uint64_t{0};
  uint32_t element_type = 0;
  uint64_t element_count = 0;
  bool elements_omitted = false;
  std::vector<MetadataScalar> elements;
};

struct TensorRecord {
  std::string name;
  // GGML ne[] order; absent dimensions are padded with one. Not original rank.
  std::array<int64_t, 4> normalized_extents{1, 1, 1, 1};
  uint32_t storage_type = 0;
  std::string storage_name;
  uint64_t absolute_offset = 0;
  uint64_t byte_size = 0;
};

struct GgufCatalog {
  uint32_t version = 0;
  uint64_t file_size = 0;
  uint64_t alignment = 0;
  uint64_t data_offset = 0;
  uint64_t bytes_read = 0;

  std::string name;
  std::string architecture;
  std::vector<MetadataRecord> metadata;
  std::vector<TensorRecord> tensors;
};

// Borrows an unbuffered, seekable file; no payload reads. Owns all returned
// data. On failure leaves output unchanged. Admission is limited to a 64 MiB
// header, 4,096 metadata entries and 16,384 tensors.
bool ReadGgufCatalog(FILE *file, GgufCatalog *catalog, std::string *error);

const MetadataRecord *FindMetadata(const GgufCatalog &catalog,
                                   const std::string &key);
const TensorRecord *FindTensor(const GgufCatalog &catalog,
                               const std::string &name);

} // namespace hrx::model

#endif // HRX_SOLUTIONS_TOOLS_MODEL_GGUF_CATALOG_H_
