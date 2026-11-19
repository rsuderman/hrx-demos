// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <fstream>
#include "iree/base/api.h"
#include "loomc/module.h"
#include "loomc/link.h"
#include "loomc/iree.h"
#include "loomc/target/configured.h"
#include "loomc/target/cmd/program.h"
#include "loom/runtime/command/program.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/parameter_index.h"

static void Check(loomc_status_t status) {
  if (!loomc_status_is_ok(status)) {
    auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", (int)m.size, m.data);
    loomc_status_free(status);
    std::exit(1);
  }
}
static void Result(loomc_result_t *r) {
  if (!r || !loomc_result_succeeded(r)) {
    for (size_t i = 0; r && i < loomc_result_diagnostic_count(r); ++i) {
      auto *d = loomc_result_diagnostic_at(r, i);
      std::fprintf(stderr, "%.*s\n", (int)d->message.size, d->message.data);
    }
    std::exit(2);
  }
  loomc_result_release(r);
}
int main(int argc, char **argv) {
  if (argc < 3)
    return 1;
  auto a = loomc_allocator_system();
  loomc_target_environment_t *env = nullptr;
  Check(loomc_target_environment_create_configured(a, &env));
  loomc_context_target_options_t t = {};
  t.structure_size = sizeof(t);
  t.target_environment = env;
  loomc_context_options_t co = {};
  co.structure_size = sizeof(co);
  co.next = &t;
  t.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  co.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
  loomc_context_t *c = nullptr;
  Check(loomc_context_create(&co, a, &c));
  loomc_workspace_t *w = nullptr;
  Check(loomc_workspace_create(nullptr, a, &w));
  std::vector<loomc_module_t *> modules;
  std::vector<loomc_link_module_provider_t> providers;
  for (int i = 2; i < argc; ++i) {
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    Check(loomc_module_deserialize_from_path(
        c, w, loomc_make_cstring_view(argv[i]), nullptr, a, &m, &r));
    Result(r);
    modules.push_back(m);
    loomc_link_module_provider_t p = {};
    p.module = m;
    p.role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
    providers.push_back(p);
  }
  loomc_linker_t *l = nullptr;
  Check(loomc_linker_create(c, nullptr, a, &l));
  auto root = loomc_make_cstring_view(argv[1]);
  loomc_link_options_t lo = {};
  lo.structure_size = sizeof(lo);
  lo.mode = LOOMC_LINK_MODE_LINK;
  lo.module_providers = providers.data();
  lo.module_provider_count = providers.size();
  lo.root_symbols = &root;
  lo.root_symbol_count = 1;
  loomc_module_t *linked = nullptr;
  loomc_result_t *r = nullptr;
  Check(loomc_link_module(l, w, &lo, &linked, &r));
  Result(r);
  loomc_cmd_program_module_options_t o = {};
  o.structure_size = sizeof(o);
  o.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  o.root_symbols = &root;
  o.root_symbol_count = 1;
  loomc_product_t *product = nullptr;
  Check(loomc_cmd_program_product_build_module(w, linked, &o, a, &product, &r));
  Result(r);
  loomc_cmd_program_t p = {};
  if (!loomc_cmd_program_product_program_at(product, 0, &p))
    return 3;
  loom_cmd_program_t native = {};
  loomc_byte_span_t bytes = {};
  if (!loomc_byte_sequence_try_get_contiguous_span(p.artifact.contents, &bytes))
    return 5;
  auto status = loom_cmd_program_parse(
      iree_make_const_byte_span(bytes.data, bytes.data_length), &native);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return 4;
  }
  iree_io_parameter_index_t *index = nullptr;
  const char *gguf_path = std::getenv("PARAMETER_GGUF");
  if (const char *path = std::getenv("PARAMETER_GGUF")) {
    iree_io_file_handle_t *file = nullptr;
    status = iree_io_file_handle_open(IREE_IO_FILE_MODE_READ,
                                      iree_make_cstring_view(path),
                                      iree_allocator_from_loomc(a), &file);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      return 6;
    }
    status =
        iree_io_parameter_index_create(iree_allocator_from_loomc(a), &index);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      return 6;
    }
    status =
        iree_io_parse_gguf_index(file, index, iree_allocator_from_loomc(a));
    iree_io_file_handle_release(file);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      return 6;
    }
  }
  std::ifstream source_file, packed_file;
  size_t packed_header = 0;
  if (const char *path = std::getenv("VERIFY_PACKED_NPY")) {
    if (!gguf_path)
      return 9;
    source_file.open(gguf_path, std::ios::binary);
    packed_file.open(path, std::ios::binary);
    unsigned char header[12] = {};
    packed_file.read((char *)header, 12);
    if (std::memcmp(header, "\x93NUMPY", 6) != 0 || header[6] != 1)
      return 9;
    packed_header = 10 + header[8] + 256u * header[9];
  }
  for (uint32_t i = 0; i < native.parameter_roots.count; ++i) {
    auto x = loom_cmd_program_parameter_root_at(&native, i);
    std::fprintf(stderr, "ROOT %u length=%llu alignment=%llu\n", i,
                 (unsigned long long)x.required_byte_length,
                 (unsigned long long)x.minimum_alignment);
  }
  for (uint32_t i = 0; i < native.parameters.count; ++i) {
    auto x = loom_cmd_program_parameter_at(&native, i);
    std::printf("%.*s\t%llu\t%llu\t%u", (int)x.key.size, x.key.data,
                (unsigned long long)x.byte_offset,
                (unsigned long long)x.byte_length, x.fixed_buffer_index);
    if (index) {
      const iree_io_parameter_index_entry_t *entry = nullptr;
      status = iree_io_parameter_index_lookup(index, x.key, &entry);
      if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        return 7;
      }
      if (entry->type != IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE ||
          entry->length != x.byte_length)
        return 8;
      std::printf("\t%llu", (unsigned long long)entry->storage.file.offset);
      if (packed_header) {
        if (x.fixed_buffer_index)
          return 10;
        std::vector<char> source_bytes(1048576), packed_bytes(1048576);
        source_file.seekg(entry->storage.file.offset);
        packed_file.seekg(packed_header + x.byte_offset);
        uint64_t remaining = x.byte_length;
        while (remaining) {
          auto n = std::min<uint64_t>(remaining, source_bytes.size());
          source_file.read(source_bytes.data(), n);
          packed_file.read(packed_bytes.data(), n);
          if (!source_file || !packed_file ||
              std::memcmp(source_bytes.data(), packed_bytes.data(), n) != 0) {
            std::fprintf(
                stderr,
                "PACKED MISMATCH %.*s root_offset=%llu source_offset=%llu\n",
                (int)x.key.size, x.key.data, (unsigned long long)x.byte_offset,
                (unsigned long long)entry->storage.file.offset);
            return 11;
          }
          remaining -= n;
        }
      }
    }
    std::puts("");
  }
  if (packed_header)
    std::fprintf(stderr,
                 "PACKED VERIFIED %u named payloads against actual GGUF\n",
                 native.parameters.count);
  iree_io_parameter_index_release(index);
  loomc_product_release(product);
  loomc_module_release(linked);
  loomc_linker_release(l);
  for (auto *m : modules)
    loomc_module_release(m);
  loomc_workspace_release(w);
  loomc_context_release(c);
  loomc_target_environment_release(env);
}
