// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/gguf_info.h"
#include "tools/loomify-gguf/gguf_catalog.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <sstream>
#include <type_traits>

#include "gguf.h"

namespace hrx::loomify_gguf {
namespace {
constexpr uint64_t kHeaderLimit = 64 * 1024 * 1024;
constexpr uint64_t kTensorLimit = 16384;
constexpr uint64_t kMetadataLimit = 4096;
constexpr size_t kReportMetadataLimit = 64;
constexpr size_t kReportTensorLimit = 32;
constexpr size_t kTextLimit = 160;

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

uint64_t LittleEndian(const unsigned char *bytes, size_t count) {
  uint64_t value = 0;
  for (size_t i = 0; i < count; ++i)
    value |= uint64_t(bytes[i]) << (i * 8);
  return value;
}

// Escape controls and bound source-controlled text before it reaches a
// terminal.
std::string Text(const char *value) {
  std::string out;
  size_t i = 0;
  for (; i < kTextLimit && value[i]; ++i) {
    const auto c = static_cast<unsigned char>(value[i]);
    if (c < 32 || c == 127) {
      out += '?';
    } else {
      out += static_cast<char>(c);
    }
  }
  if (i == kTextLimit && value[i])
    out += "...";
  return out;
}

struct Reader {
  FILE *file;
  uint64_t extent;
  uint64_t bytes = 0;
};
size_t Read(void *opaque, void *output, uint64_t offset, size_t size) {
  auto &reader = *static_cast<Reader *>(opaque);
  if (offset > reader.extent || size > reader.extent - offset ||
      Seek(reader.file, static_cast<int64_t>(offset), SEEK_SET) != 0) {
    return 0;
  }
  const size_t count = std::fread(output, 1, size, reader.file);
  reader.bytes += count;
  return count;
}

bool StringField(const gguf_context *ctx, const char *key, bool required,
                 std::string *value, std::string *error) {
  const int64_t id = gguf_find_key(ctx, key);
  if (id < 0) {
    if (!required)
      return true;
    *error = std::string("missing required string metadata: ") + key;
    return false;
  }
  if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING ||
      gguf_get_val_str(ctx, id)[0] == '\0') {
    *error = std::string("expected nonempty string metadata: ") + key;
    return false;
  }
  *value = gguf_get_val_str(ctx, id);
  return true;
}

template <typename T> MetadataScalar ArrayElement(const void *data, size_t i) {
  T value;
  std::memcpy(&value, static_cast<const unsigned char *>(data) + i * sizeof(T),
              sizeof(T));
  if constexpr (std::is_floating_point_v<T>)
    return double(value);
  else if constexpr (std::is_signed_v<T>)
    return int64_t(value);
  else
    return uint64_t(value);
}
MetadataScalar ScalarValue(const gguf_context *ctx, int64_t id) {
  switch (gguf_get_kv_type(ctx, id)) {
  case GGUF_TYPE_UINT8:
    return uint64_t(gguf_get_val_u8(ctx, id));
  case GGUF_TYPE_UINT16:
    return uint64_t(gguf_get_val_u16(ctx, id));
  case GGUF_TYPE_UINT32:
    return uint64_t(gguf_get_val_u32(ctx, id));
  case GGUF_TYPE_UINT64:
    return gguf_get_val_u64(ctx, id);
  case GGUF_TYPE_INT8:
    return int64_t(gguf_get_val_i8(ctx, id));
  case GGUF_TYPE_INT16:
    return int64_t(gguf_get_val_i16(ctx, id));
  case GGUF_TYPE_INT32:
    return int64_t(gguf_get_val_i32(ctx, id));
  case GGUF_TYPE_INT64:
    return gguf_get_val_i64(ctx, id);
  case GGUF_TYPE_FLOAT32:
    return double(gguf_get_val_f32(ctx, id));
  case GGUF_TYPE_FLOAT64:
    return gguf_get_val_f64(ctx, id);
  case GGUF_TYPE_BOOL:
    return gguf_get_val_bool(ctx, id);
  case GGUF_TYPE_STRING:
    return std::string(gguf_get_val_str(ctx, id));
  default:
    return uint64_t{0};
  }
}
MetadataScalar ElementValue(uint32_t type, const void *data, size_t i) {
  switch (type) {
  case GGUF_TYPE_UINT8:
    return ArrayElement<uint8_t>(data, i);
  case GGUF_TYPE_UINT16:
    return ArrayElement<uint16_t>(data, i);
  case GGUF_TYPE_UINT32:
    return ArrayElement<uint32_t>(data, i);
  case GGUF_TYPE_UINT64:
    return ArrayElement<uint64_t>(data, i);
  case GGUF_TYPE_INT8:
    return ArrayElement<int8_t>(data, i);
  case GGUF_TYPE_INT16:
    return ArrayElement<int16_t>(data, i);
  case GGUF_TYPE_INT32:
    return ArrayElement<int32_t>(data, i);
  case GGUF_TYPE_INT64:
    return ArrayElement<int64_t>(data, i);
  case GGUF_TYPE_FLOAT32:
    return ArrayElement<float>(data, i);
  case GGUF_TYPE_FLOAT64:
    return ArrayElement<double>(data, i);
  case GGUF_TYPE_BOOL:
    return bool(std::get<uint64_t>(ArrayElement<uint8_t>(data, i)));
  default:
    return uint64_t{0};
  }
}
std::string DisplayValue(const MetadataRecord &field) {
  std::ostringstream out;
  if (field.source_type == GGUF_TYPE_ARRAY) {
    out << "array<"
        << gguf_type_name(static_cast<gguf_type>(field.element_type)) << ">["
        << field.element_count << ']';
  } else {
    std::visit(
        [&](const auto &value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, std::string>)
            out << '\"' << Text(value.c_str()) << '\"';
          else if constexpr (std::is_same_v<T, bool>)
            out << (value ? "true" : "false");
          else
            out << value;
        },
        field.scalar);
  }
  return out.str();
}
} // namespace

bool ReadGgufCatalog(FILE *file, GgufCatalog *info, std::string *error) {
  if (!file || !info || !error)
    return false;
  if (Seek(file, 0, SEEK_END) != 0) {
    *error = "input must be a seekable GGUF file";
    return false;
  }
  const int64_t length = Tell(file);
  if (length < 24 || Seek(file, 0, SEEK_SET) != 0) {
    *error = "truncated or unreadable GGUF header (expected at least 24 bytes)";
    return false;
  }
  unsigned char header[24];
  if (std::fread(header, 1, sizeof(header), file) != sizeof(header) ||
      std::memcmp(header, "GGUF", 4) != 0) {
    *error = "invalid GGUF magic (expected GGUF)";
    return false;
  }
  const uint64_t version = LittleEndian(header + 4, 4);
  if (version != 2 && version != 3) {
    *error = "unsupported GGUF version/endian encoding (expected little-endian "
             "v2 or v3)";
    return false;
  }
  const uint64_t tensor_count = LittleEndian(header + 8, 8);
  const uint64_t metadata_count = LittleEndian(header + 16, 8);
  if (tensor_count > kTensorLimit || metadata_count > kMetadataLimit) {
    *error = "GGUF catalog count exceeds inspection limits (16384 tensors, "
             "4096 metadata entries)";
    return false;
  }
  Reader reader{file, std::min<uint64_t>(length, kHeaderLimit)};
  try {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
        gguf_init_from_callback(Read, &reader, 64 * 1024, reader.extent,
                                gguf_init_params{true, nullptr}),
        gguf_free);
    if (!ctx) {
      *error = "invalid/truncated GGUF catalog, or header exceeds 64 MiB "
               "inspection limit (see reader diagnostic)";
      return false;
    }
    GgufCatalog result;
    result.version = gguf_get_version(ctx.get());
    result.file_size = static_cast<uint64_t>(length);
    result.alignment = gguf_get_alignment(ctx.get());
    result.data_offset = gguf_get_data_offset(ctx.get());
    result.bytes_read = reader.bytes + sizeof(header);
    if (!StringField(ctx.get(), "general.architecture", true,
                     &result.architecture, error) ||
        !StringField(ctx.get(), "general.name", false, &result.name, error))
      return false;
    for (int64_t i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
      MetadataRecord field;
      field.key = gguf_get_key(ctx.get(), i);
      field.source_type = gguf_get_kv_type(ctx.get(), i);
      if (field.source_type == GGUF_TYPE_ARRAY) {
        field.element_type = gguf_get_arr_type(ctx.get(), i);
        field.element_count = gguf_get_arr_n(ctx.get(), i);
        // Tokenizer arrays are not architecture inputs; retain explicit
        // summaries.
        field.elements_omitted = field.element_type == GGUF_TYPE_STRING ||
                                 field.key.rfind("tokenizer.", 0) == 0;
        if (!field.elements_omitted) {
          const void *data = gguf_get_arr_data(ctx.get(), i);
          for (size_t j = 0; j < field.element_count; ++j)
            field.elements.push_back(ElementValue(field.element_type, data, j));
        }
      } else
        field.scalar = ScalarValue(ctx.get(), i);
      result.metadata.push_back(std::move(field));
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
      const uint64_t relative = gguf_get_tensor_offset(ctx.get(), i);
      const uint64_t bytes = gguf_get_tensor_size(ctx.get(), i);
      if (result.data_offset > result.file_size ||
          relative > result.file_size - result.data_offset ||
          bytes > result.file_size - result.data_offset - relative) {
        *error = "tensor range exceeds file size: " +
                 Text(gguf_get_tensor_name(ctx.get(), i));
        return false;
      }
      {
        TensorRecord tensor;
        tensor.name = gguf_get_tensor_name(ctx.get(), i);
        const int64_t *shape = gguf_get_tensor_ne(ctx.get(), i);
        std::copy_n(shape, tensor.normalized_extents.size(),
                    tensor.normalized_extents.begin());
        tensor.storage_type = gguf_get_tensor_type(ctx.get(), i);
        tensor.storage_name =
            ggml_type_name(gguf_get_tensor_type(ctx.get(), i));
        tensor.absolute_offset = result.data_offset + relative;
        tensor.byte_size = bytes;
        result.tensors.push_back(std::move(tensor));
      }
    }
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
bool ReadGgufInfo(FILE *file, GgufInfo *info, std::string *error) {
  if (!info)
    return false;
  GgufCatalog catalog;
  if (!ReadGgufCatalog(file, &catalog, error))
    return false;
  GgufInfo result;
  result.version = catalog.version;
  result.file_size = catalog.file_size;
  result.metadata_count = catalog.metadata.size();
  result.tensor_count = catalog.tensors.size();
  result.alignment = catalog.alignment;
  result.data_offset = catalog.data_offset;
  result.bytes_read = catalog.bytes_read;
  result.name = Text(catalog.name.c_str());
  result.architecture = Text(catalog.architecture.c_str());
  for (const auto &field : catalog.metadata) {
    if (result.metadata.size() == kReportMetadataLimit)
      break;
    result.metadata.push_back({Text(field.key.c_str()), DisplayValue(field)});
  }
  for (const auto &tensor : catalog.tensors) {
    if (result.tensors.size() == kReportTensorLimit)
      break;
    result.tensors.push_back({Text(tensor.name.c_str()),
                              tensor.normalized_extents, tensor.storage_name,
                              tensor.absolute_offset, tensor.byte_size});
  }
  *info = std::move(result);
  return true;
}
std::string FormatGgufInfo(const GgufInfo &info) {
  std::ostringstream out;
  out << "GGUF version: " << info.version
      << "\nModel name: " << (info.name.empty() ? "(not provided)" : info.name)
      << "\nArchitecture: " << info.architecture
      << "\nMetadata entries: " << info.metadata_count
      << "\nTensors: " << info.tensor_count
      << "\nFile bytes: " << info.file_size << "\nAlignment: " << info.alignment
      << "\nData offset: " << info.data_offset
      << "\nMetadata bytes read: " << info.bytes_read
      << " (includes header validation; no tensor payload)\n";
  for (const auto &field : info.metadata)
    out << "  " << field.key << " = " << field.value << '\n';
  if (info.metadata_count > info.metadata.size())
    out << "  ... " << info.metadata_count - info.metadata.size()
        << " more metadata entries\n";
  out << "Tensor ranges (absolute file bytes; shapes padded to four "
         "dimensions):\n";
  for (const auto &tensor : info.tensors) {
    out << "  " << tensor.name << " [" << tensor.shape[0] << ','
        << tensor.shape[1] << ',' << tensor.shape[2] << ',' << tensor.shape[3]
        << "] " << tensor.storage_type << " bytes=[" << tensor.offset << ','
        << tensor.offset + tensor.size << ")\n";
  }
  if (info.tensor_count > info.tensors.size())
    out << "  ... " << info.tensor_count - info.tensors.size()
        << " more tensors\n";
  return out.str();
}
} // namespace hrx::loomify_gguf
