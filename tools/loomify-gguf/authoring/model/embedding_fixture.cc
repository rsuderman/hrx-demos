// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model/embedding_fixture.h"
#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/parameter_index.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
namespace hrx::loomify_gguf::testing {
namespace {
void Number(std::vector<uint8_t> &b, uint64_t n, unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    b.push_back(n >> (i * 8));
}
void String(std::vector<uint8_t> &b, const std::string &s) {
  Number(b, s.size(), 8);
  b.insert(b.end(), s.begin(), s.end());
}
struct Tensor {
  std::string name;
  std::vector<uint64_t> dims;
  uint64_t offset;
};
bool Ok(iree_status_t status, std::string *error) {
  if (iree_status_is_ok(status))
    return true;
  *error =
      "IREE parameter index error " + std::to_string(iree_status_code(status));
  iree_status_fprint(stderr, status);
  iree_status_ignore(status);
  return false;
}
} // namespace
std::vector<uint8_t> EmbeddingGguf(uint32_t width, uint32_t vocab) {
  const std::vector<std::pair<std::string, uint32_t>> metadata = {
      {"block_count", 1},
      {"embedding_length", width},
      {"context_length", 16},
      {"feed_forward_length", width * 2},
      {"attention.head_count", 1},
      {"attention.head_count_kv", 1},
      {"attention.key_length", width},
      {"attention.value_length", width},
      {"vocab_size", vocab}};
  std::vector<Tensor> tensors;
  uint64_t offset = 0;
  const auto tensor = [&](std::string name, std::vector<uint64_t> dims) {
    tensors.push_back({std::move(name), dims, offset});
    uint64_t count = 4;
    for (auto n : dims)
      count *= n;
    offset = (offset + count + 31) / 32 * 32;
  };
  tensor("output_norm.weight", {width});
  tensor("token_embd.weight", {width, vocab});
  tensor("output.weight", {width, vocab});
  for (const char *name : {"attn_norm", "ffn_norm"})
    tensor(std::string("blk.0.") + name + ".weight", {width});
  for (const char *name : {"attn_q", "attn_k", "attn_v", "attn_output"})
    tensor(std::string("blk.0.") + name + ".weight", {width, width});
  for (const char *name : {"ffn_gate", "ffn_up"})
    tensor(std::string("blk.0.") + name + ".weight", {width, width * 2});
  tensor("blk.0.ffn_down.weight", {width * 2, width});
  std::vector<uint8_t> b = {'G', 'G', 'U', 'F'};
  Number(b, 3, 4);
  Number(b, tensors.size(), 8);
  Number(b, metadata.size() + 3, 8);
  String(b, "general.architecture");
  Number(b, 8, 4);
  String(b, "llama");
  String(b, "general.alignment");
  Number(b, 4, 4);
  Number(b, 32, 4);
  for (const auto &m : metadata) {
    String(b, "llama." + m.first);
    Number(b, 4, 4);
    Number(b, m.second, 4);
  }
  String(b, "llama.attention.layer_norm_rms_epsilon");
  Number(b, 12, 4);
  const double epsilon = 1e-5;
  uint64_t bits;
  std::memcpy(&bits, &epsilon, 8);
  Number(b, bits, 8);
  for (const auto &t : tensors) {
    String(b, t.name);
    Number(b, t.dims.size(), 4);
    for (auto d : t.dims)
      Number(b, d, 8);
    Number(b, 0, 4);
    Number(b, t.offset, 8);
  }
  while (b.size() % 32)
    b.push_back(0);
  const size_t origin = b.size();
  b.resize(origin + offset, 0xA5);
  for (const auto &t : tensors) {
    uint64_t count = 1;
    for (auto d : t.dims)
      count *= d;
    for (uint64_t i = 0; i < count; ++i) {
      const float value = t.name == "token_embd.weight"
                              ? 10 + 100 * (i / width) + (i % width)
                              : -1000.0f - static_cast<float>(i);
      uint32_t f;
      std::memcpy(&f, &value, 4);
      for (unsigned j = 0; j < 4; ++j)
        b[origin + t.offset + i * 4 + j] = f >> (j * 8);
    }
  }
  return b;
}
bool ReadFixture(const std::vector<uint8_t> &bytes, model::GgufCatalog *catalog,
                 model::ModelDescriptor *descriptor, std::string *error) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(),
                                                     std::fclose);
  if (!file || std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0 ||
      std::fwrite(bytes.data(), 1, bytes.size(), file.get()) != bytes.size()) {
    *error = "unable to write external GGUF fixture";
    return false;
  }
  return model::ReadGgufCatalog(file.get(), catalog, error) &&
         model::LoadModelDescriptor(*catalog, descriptor, error);
}
bool ReadNamedF32(const std::vector<uint8_t> &bytes, const std::string &name,
                  std::vector<float> *values, std::string *error) {
  iree_io_parameter_index_t *raw_index = nullptr;
  if (!Ok(iree_io_parameter_index_create(iree_allocator_system(), &raw_index),
          error))
    return false;
  std::unique_ptr<iree_io_parameter_index_t,
                  decltype(&iree_io_parameter_index_release)>
      index(raw_index, iree_io_parameter_index_release);
  iree_io_file_handle_t *raw_file = nullptr;
  if (!Ok(iree_io_file_handle_wrap_host_allocation(
              IREE_IO_FILE_ACCESS_READ,
              iree_make_byte_span(const_cast<uint8_t *>(bytes.data()),
                                  bytes.size()),
              iree_io_file_handle_release_callback_null(),
              iree_allocator_system(), &raw_file),
          error))
    return false;
  std::unique_ptr<iree_io_file_handle_t, decltype(&iree_io_file_handle_release)>
      file(raw_file, iree_io_file_handle_release);
  if (!Ok(iree_io_parse_gguf_index(file.get(), index.get(),
                                   iree_allocator_system()),
          error))
    return false;
  const iree_io_parameter_index_entry_t *entry = nullptr;
  if (!Ok(iree_io_parameter_index_lookup(
              index.get(), iree_make_cstring_view(name.c_str()), &entry),
          error))
    return false;
  if (entry->type != IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE ||
      entry->length % 4) {
    *error = "expected file-backed F32 tensor";
    return false;
  }
  const auto primitive =
      iree_io_file_handle_primitive(entry->storage.file.handle);
  if (primitive.type != IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION ||
      entry->storage.file.offset >
          primitive.value.host_allocation.data_length ||
      entry->length > primitive.value.host_allocation.data_length -
                          entry->storage.file.offset) {
    *error = "invalid tensor source range";
    return false;
  }
  values->resize(entry->length / 4);
  const uint8_t *p =
      primitive.value.host_allocation.data + entry->storage.file.offset;
  for (size_t i = 0; i < values->size(); ++i) {
    uint32_t f = uint32_t(p[i * 4]) | uint32_t(p[i * 4 + 1]) << 8 |
                 uint32_t(p[i * 4 + 2]) << 16 | uint32_t(p[i * 4 + 3]) << 24;
    std::memcpy(&(*values)[i], &f, 4);
  }
  return true;
}
} // namespace hrx::loomify_gguf::testing
