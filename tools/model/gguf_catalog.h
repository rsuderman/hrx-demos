// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_MODEL_GGUF_CATALOG_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_GGUF_CATALOG_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace hrx::model {

struct GgufCatalog;
// Only the catalog parser can mint a range. Copies remain usable after the
// catalog is destroyed, against the same immutable checkpoint.
class GgufStringArraySpan {
public:
  uint64_t begin() const { return begin_; }
  uint64_t end() const { return end_; }
  uint64_t count() const { return count_; }
  uint64_t header_end() const { return header_end_; }

private:
  friend bool ReadGgufCatalog(FILE *, GgufCatalog *, std::string *);
  GgufStringArraySpan(uint64_t begin, uint64_t end, uint64_t count,
                      uint64_t header_end)
      : begin_(begin), end_(end), count_(count), header_end_(header_end) {}
  uint64_t begin_, end_, count_, header_end_;
};

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
  std::optional<GgufStringArraySpan> string_array;
};

struct TensorRecord {
  std::string name;
  // GGUF dimension order; absent dimensions are padded with one.
  std::array<int64_t, 4> normalized_extents{1, 1, 1, 1};
  uint32_t storage_type = 0;
  std::string storage_name;
  uint64_t absolute_offset = 0;
  uint64_t byte_size = 0;
  uint32_t rank = 0;
};

struct GgufCatalog {
  uint32_t version = 0;
  uint64_t file_size = 0;
  uint64_t alignment = 0;
  uint64_t data_offset = 0;
  uint64_t header_end = 0;
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

// Computes packed matrix bytes from the same on-disk layouts as the reader.
// The first GGML extent must contain complete encoding blocks. Failure leaves
// *bytes unchanged, including overflow/unknown format/zero dimensions.
bool EncodedTensorByteSize(uint32_t storage_type, uint64_t row_elements,
                           uint64_t rows, uint64_t *bytes);

// These borrow the same immutable unbuffered seekable file as the issuing
// catalog and restore its position even on failure. Each callback view expires
// on return. No read/seek leaves the admitted encoded-array/header bounds.
using StringArrayVisitor =
    std::function<bool(uint64_t, std::string_view, std::string *)>;
bool VisitGgufStringArray(FILE *, const GgufStringArraySpan &,
                          const StringArrayVisitor &, std::string *error);
// Output follows requested ID order, including duplicates. Bounded retained
// storage; failure leaves *out unchanged. Unrequested piece bytes are skipped.
bool ReadGgufTokenPieces(FILE *, const GgufStringArraySpan &,
                         const std::vector<uint64_t> &ids,
                         std::vector<std::string> *out, std::string *error);

const MetadataRecord *FindMetadata(const GgufCatalog &catalog,
                                   const std::string &key);
const TensorRecord *FindTensor(const GgufCatalog &catalog,
                               const std::string &name);

} // namespace hrx::model

#endif // HRX_SOLUTIONS_TOOLS_MODEL_GGUF_CATALOG_H_
