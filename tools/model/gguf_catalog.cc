// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/model/gguf_catalog.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace hrx::model {
namespace {
constexpr uint64_t kHeaderLimit = 64 * 1024 * 1024;
constexpr uint64_t kTensorLimit = 16384;
constexpr uint64_t kMetadataLimit = 4096;

#ifdef _WIN32
int Seek(FILE *file, int64_t offset, int origin) {
  return _fseeki64(file, offset, origin);
}
int64_t Tell(FILE *file) { return _ftelli64(file); }
#else
int Seek(FILE *file, int64_t offset, int origin) {
  return fseeko(file, offset, origin);
}
int64_t Tell(FILE *file) { return ftello(file); }
#endif

void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
uint64_t Multiply(uint64_t a, uint64_t b) {
  Require(b == 0 || a <= UINT64_MAX / b, "GGUF tensor size overflow");
  return a * b;
}

// Sequential bounded reads never prefetch tensor payload. Skipped tokenizer
// strings are range checked but neither allocated nor read.
struct Reader {
  FILE *file;
  uint64_t limit;
  uint64_t position = 0;
  uint64_t bytes = 0;
  uint64_t retained_elements = 0;

  void Check(uint64_t size) const {
    Require(size <= limit - position,
            "truncated GGUF catalog or header exceeds 64 MiB limit");
  }
  void Read(void *output, size_t size) {
    Check(size);
    Require(std::fread(output, 1, size, file) == size,
            "unreadable GGUF catalog");
    position += size;
    bytes += size;
  }
  uint64_t Number(unsigned width) {
    unsigned char data[8];
    Read(data, width);
    uint64_t value = 0;
    for (unsigned i = 0; i < width; ++i)
      value |= uint64_t(data[i]) << (8 * i);
    return value;
  }
  void Skip(uint64_t size) {
    Check(size);
    Require(Seek(file, static_cast<int64_t>(position + size), SEEK_SET) == 0,
            "unseekable GGUF catalog");
    position += size;
  }
  std::string String(bool retain = true) {
    const auto size = Number(8);
    Check(size);
    if (!retain) {
      Skip(size);
      return {};
    }
    std::string value(static_cast<size_t>(size), '\0');
    Read(value.data(), value.size());
    Require(value.find('\0') == std::string::npos,
            "embedded NUL in GGUF string");
    return value;
  }
};

unsigned ScalarWidth(uint32_t type) {
  switch (type) {
  case 0:
  case 1:
  case 7:
    return 1;
  case 2:
  case 3:
    return 2;
  case 4:
  case 5:
  case 6:
    return 4;
  case 10:
  case 11:
  case 12:
    return 8;
  case 8:
    return 0; // variable-length string
  default:
    throw std::runtime_error("unsupported GGUF metadata type");
  }
}
MetadataScalar Scalar(Reader &reader, uint32_t type) {
  if (type == 8)
    return reader.String();
  const auto width = ScalarWidth(type);
  const uint64_t bits = reader.Number(width);
  switch (type) {
  case 1:
  case 3:
  case 5:
  case 11: {
    // Sign extension without implementation-defined unsigned-to-signed casts.
    const uint64_t sign = uint64_t{1} << (width * 8 - 1);
    return bits & sign ? -int64_t((~bits) & (sign - 1)) - 1 : int64_t(bits);
  }
  case 6: {
    const uint32_t raw = static_cast<uint32_t>(bits);
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return double(value);
  }
  case 12: {
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  case 7:
    Require(bits <= 1, "invalid GGUF boolean");
    return bool(bits);
  default:
    return bits;
  }
}

struct Encoding {
  uint64_t block;
  uint64_t bytes;
  const char *name;
};
Encoding Storage(uint32_t type) {
  // Only the first checkpoint's encodings and existing useful tested formats.
  switch (type) {
  case 0:
    return {1, 4, "f32"};
  case 1:
    return {1, 2, "f16"};
  case 8:
    return {32, 34, "q8_0"};
  case 12:
    return {256, 144, "q4_K"};
  case 14:
    return {256, 210, "q6_K"};
  default:
    throw std::runtime_error("unsupported GGUF tensor encoding");
  }
}
void StringField(const GgufCatalog &catalog, const char *key, bool required,
                 std::string *value) {
  const auto *field = FindMetadata(catalog, key);
  if (!field) {
    Require(!required, "missing required general.architecture string");
    return;
  }
  Require(field->source_type == 8 &&
              !std::get<std::string>(field->scalar).empty(),
          "expected nonempty GGUF string metadata");
  *value = std::get<std::string>(field->scalar);
}
} // namespace

bool ReadGgufCatalog(FILE *file, GgufCatalog *info, std::string *error) {
  if (!file || !info || !error)
    return false;
  try {
    Require(Seek(file, 0, SEEK_END) == 0, "input must be a seekable GGUF file");
    const int64_t length = Tell(file);
    Require(length >= 24 && Seek(file, 0, SEEK_SET) == 0,
            "truncated or unreadable GGUF header");
    Reader reader{file, std::min<uint64_t>(length, kHeaderLimit)};
    Require(reader.Number(4) == 0x46554747, "invalid GGUF magic");
    GgufCatalog result;
    result.version = reader.Number(4);
    Require(result.version == 2 || result.version == 3,
            "unsupported GGUF version/endian encoding (expected little-endian "
            "v2 or v3)");
    const auto tensor_count = reader.Number(8);
    const auto metadata_count = reader.Number(8);
    Require(tensor_count <= kTensorLimit && metadata_count <= kMetadataLimit,
            "GGUF catalog count exceeds inspection limits");
    result.file_size = static_cast<uint64_t>(length);
    result.alignment = 32;
    std::unordered_set<std::string> keys;
    for (uint64_t i = 0; i < metadata_count; ++i) {
      MetadataRecord field;
      field.key = reader.String();
      Require(!field.key.empty() && keys.insert(field.key).second,
              "empty or duplicate GGUF metadata key");
      field.source_type = reader.Number(4);
      if (field.source_type == 9) {
        field.element_type = reader.Number(4);
        const auto width = ScalarWidth(field.element_type);
        field.element_count = reader.Number(8);
        // A string needs at least its eight-byte length record.
        Require(field.element_count <=
                    (reader.limit - reader.position) / (width ? width : 8),
                "invalid GGUF array count");
        field.elements_omitted =
            field.element_type == 8 || field.key.rfind("tokenizer.", 0) == 0;
        if (field.elements_omitted && width) {
          reader.Skip(field.element_count * width);
        } else {
          if (!field.elements_omitted) {
            Require(field.element_count <=
                        kHeaderLimit / sizeof(MetadataScalar) -
                            reader.retained_elements,
                    "GGUF retained metadata exceeds inspection limit");
            reader.retained_elements += field.element_count;
            field.elements.reserve(field.element_count);
          }
          for (uint64_t j = 0; j < field.element_count; ++j) {
            if (field.elements_omitted)
              reader.String(false);
            else
              field.elements.push_back(Scalar(reader, field.element_type));
          }
        }
      } else {
        field.scalar = Scalar(reader, field.source_type);
      }
      if (field.key == "general.alignment") {
        Require(field.source_type == 4, "GGUF alignment must be uint32");
        result.alignment = std::get<uint64_t>(field.scalar);
        Require(result.alignment &&
                    (result.alignment & (result.alignment - 1)) == 0,
                "GGUF alignment must be a nonzero power of two");
      }
      result.metadata.push_back(std::move(field));
    }
    StringField(result, "general.architecture", true, &result.architecture);
    StringField(result, "general.name", false, &result.name);
    std::unordered_set<std::string> names;
    for (uint64_t i = 0; i < tensor_count; ++i) {
      TensorRecord tensor;
      tensor.name = reader.String();
      Require(!tensor.name.empty() && names.insert(tensor.name).second,
              "empty or duplicate GGUF tensor name");
      tensor.rank = reader.Number(4);
      Require(tensor.rank >= 1 && tensor.rank <= 4, "invalid GGUF tensor rank");
      uint64_t elements = 1;
      for (uint32_t d = 0; d < tensor.rank; ++d) {
        const auto extent = reader.Number(8);
        Require(extent > 0 && extent <= INT64_MAX,
                "invalid GGUF tensor extent");
        tensor.normalized_extents[d] = static_cast<int64_t>(extent);
        elements = Multiply(elements, extent);
      }
      tensor.storage_type = reader.Number(4);
      const auto encoding = Storage(tensor.storage_type);
      Require(uint64_t(tensor.normalized_extents[0]) % encoding.block == 0,
              "GGUF tensor row is not divisible by encoding block size");
      tensor.storage_name = encoding.name;
      tensor.byte_size = Multiply(elements / encoding.block, encoding.bytes);
      tensor.absolute_offset =
          reader.Number(8); // relative until header is complete
      Require(tensor.absolute_offset % result.alignment == 0,
              "misaligned GGUF tensor offset");
      result.tensors.push_back(std::move(tensor));
    }
    result.data_offset =
        (reader.position + result.alignment - 1) & ~(result.alignment - 1);
    Require(result.data_offset <= result.file_size,
            "GGUF data offset exceeds file size");
    for (auto &tensor : result.tensors) {
      Require(tensor.absolute_offset <= result.file_size - result.data_offset &&
                  tensor.byte_size <= result.file_size - result.data_offset -
                                          tensor.absolute_offset,
              "GGUF tensor range exceeds file size");
      tensor.absolute_offset += result.data_offset;
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(result.tensors.size());
    for (const auto &tensor : result.tensors)
      ranges.emplace_back(tensor.absolute_offset,
                          tensor.absolute_offset + tensor.byte_size);
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
      Require(ranges[i - 1].second <= ranges[i].first,
              "overlapping GGUF tensor storage ranges");
    result.bytes_read = reader.bytes;
    *info = std::move(result);
    return true;
  } catch (const std::exception &ex) {
    *error = std::string("unable to read bounded GGUF metadata: ") + ex.what();
    return false;
  }
}

const MetadataRecord *FindMetadata(const GgufCatalog &catalog,
                                   const std::string &key) {
  for (const auto &field : catalog.metadata)
    if (field.key == key)
      return &field;
  return nullptr;
}
const TensorRecord *FindTensor(const GgufCatalog &catalog,
                               const std::string &name) {
  for (const auto &tensor : catalog.tensors)
    if (tensor.name == name)
      return &tensor;
  return nullptr;
}
} // namespace hrx::model
