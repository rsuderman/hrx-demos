// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/gguf_info.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <sstream>

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
  *value = Text(gguf_get_val_str(ctx, id));
  return true;
}

std::string MetadataValue(const gguf_context *ctx, int64_t id) {
  std::ostringstream out;
  switch (gguf_get_kv_type(ctx, id)) {
  case GGUF_TYPE_UINT8:
    out << unsigned(gguf_get_val_u8(ctx, id));
    break;
  case GGUF_TYPE_INT8:
    out << int(gguf_get_val_i8(ctx, id));
    break;
  case GGUF_TYPE_UINT16:
    out << gguf_get_val_u16(ctx, id);
    break;
  case GGUF_TYPE_INT16:
    out << gguf_get_val_i16(ctx, id);
    break;
  case GGUF_TYPE_UINT32:
    out << gguf_get_val_u32(ctx, id);
    break;
  case GGUF_TYPE_INT32:
    out << gguf_get_val_i32(ctx, id);
    break;
  case GGUF_TYPE_UINT64:
    out << gguf_get_val_u64(ctx, id);
    break;
  case GGUF_TYPE_INT64:
    out << gguf_get_val_i64(ctx, id);
    break;
  case GGUF_TYPE_FLOAT32:
    out << gguf_get_val_f32(ctx, id);
    break;
  case GGUF_TYPE_FLOAT64:
    out << gguf_get_val_f64(ctx, id);
    break;
  case GGUF_TYPE_BOOL:
    out << (gguf_get_val_bool(ctx, id) ? "true" : "false");
    break;
  case GGUF_TYPE_STRING:
    out << '"' << Text(gguf_get_val_str(ctx, id)) << '"';
    break;
  case GGUF_TYPE_ARRAY:
    out << "array<" << gguf_type_name(gguf_get_arr_type(ctx, id)) << ">["
        << gguf_get_arr_n(ctx, id) << ']';
    break;
  default:
    out << "unknown";
    break;
  }
  return out.str();
}
} // namespace

bool ReadGgufInfo(FILE *file, GgufInfo *info, std::string *error) {
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
    GgufInfo result;
    result.version = gguf_get_version(ctx.get());
    result.file_size = static_cast<uint64_t>(length);
    result.metadata_count = static_cast<uint64_t>(gguf_get_n_kv(ctx.get()));
    result.tensor_count = static_cast<uint64_t>(gguf_get_n_tensors(ctx.get()));
    result.alignment = gguf_get_alignment(ctx.get());
    result.data_offset = gguf_get_data_offset(ctx.get());
    result.bytes_read = reader.bytes + sizeof(header);
    if (!StringField(ctx.get(), "general.architecture", true,
                     &result.architecture, error) ||
        !StringField(ctx.get(), "general.name", false, &result.name, error))
      return false;
    for (int64_t i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
      if (result.metadata.size() < kReportMetadataLimit) {
        result.metadata.push_back(
            {Text(gguf_get_key(ctx.get(), i)), MetadataValue(ctx.get(), i)});
      }
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
      if (result.tensors.size() < kReportTensorLimit) {
        TensorInfo tensor;
        tensor.name = Text(gguf_get_tensor_name(ctx.get(), i));
        const int64_t *shape = gguf_get_tensor_ne(ctx.get(), i);
        std::copy_n(shape, tensor.shape.size(), tensor.shape.begin());
        tensor.storage_type =
            ggml_type_name(gguf_get_tensor_type(ctx.get(), i));
        tensor.offset = result.data_offset + relative;
        tensor.size = bytes;
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
