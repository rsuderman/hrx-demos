// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_GGUF_INFO_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_GGUF_INFO_H_

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace hrx::loomify_gguf {

struct MetadataInfo {
  std::string key;
  std::string value;
};
struct TensorInfo {
  std::string name;
  // ggml exposes four extents, padding absent dimensions with one.
  std::array<int64_t, 4> shape;
  std::string storage_type;
  uint64_t offset;
  uint64_t size;
};
struct GgufInfo {
  uint32_t version = 0;
  uint64_t file_size = 0;
  uint64_t metadata_count = 0;
  uint64_t tensor_count = 0;
  uint64_t alignment = 0;
  uint64_t data_offset = 0;
  uint64_t bytes_read = 0;
  std::string name;
  std::string architecture;
  std::vector<MetadataInfo> metadata;
  std::vector<TensorInfo> tensors;
};

// Borrows a seekable file and reads only its bounded metadata catalog. The
// caller disables stdio buffering before any I/O to prevent payload prefetch.
// Rewinds
// the file; the caller retains ownership. On failure, leaves |info| unchanged.
// The returned value owns its strings; it does not retain the ggml context.
bool ReadGgufInfo(FILE *file, GgufInfo *info, std::string *error);
std::string FormatGgufInfo(const GgufInfo &info);

} // namespace hrx::loomify_gguf
#endif
