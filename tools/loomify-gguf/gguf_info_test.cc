// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/gguf_info.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {
using hrx::loomify_gguf::GgufInfo;
using hrx::loomify_gguf::ReadGgufInfo;
void Require(bool value, const char *label) {
  if (!value) {
    std::cerr << "FAIL: " << label << '\n';
    std::exit(1);
  }
}
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
bool Inspect(const std::string &bytes, GgufInfo *info, std::string *error) {
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(),
                                                     std::fclose);
  Require(bool(file), "temporary file");
  Require(std::setvbuf(file.get(), nullptr, _IONBF, 0) == 0,
          "unbuffered input");
  Require(std::fwrite(bytes.data(), 1, bytes.size(), file.get()) ==
              bytes.size(),
          "write fixture");
  return ReadGgufInfo(file.get(), info, error);
}
void Reject(const std::string &bytes, const char *label) {
  GgufInfo info;
  info.name = "unchanged";
  std::string error;
  Require(!Inspect(bytes, &info, &error), label);
  Require(!error.empty(), "actionable diagnostic");
  Require(info.name == "unchanged",
          "failure does not publish partial metadata");
}
} // namespace

int main() {
  GgufInfo info;
  std::string error;
  const auto bytes = Fixture(40);
  Require(Inspect(bytes, &info, &error), error.c_str());
  Require(info.version == 3 && info.architecture == "llama", "identity");
  Require(info.tensor_count == 40 && info.tensors.size() == 32,
          "bounded tensor list");
  Require(info.tensors[0].shape[0] == 4 && info.tensors[0].shape[1] == 2,
          "shape");
  Require(info.tensors[0].offset == info.data_offset &&
              info.tensors[0].size == 32,
          "range");
  Require(info.bytes_read <= info.data_offset + 24, "no payload reads");
  const auto report = hrx::loomify_gguf::FormatGgufInfo(info);
  Require(report.find("private-token-content") == std::string::npos,
          "array contents omitted");
  Require(report.find("array<str>[200]") != std::string::npos,
          "array size reported");
  Require(report.find("8 more tensors") != std::string::npos, "omitted count");
  auto bad_magic = bytes;
  bad_magic[0] = 'X';
  Reject(bad_magic, "bad magic");
  auto bad_version = bytes;
  bad_version[4] = 99;
  Reject(bad_version, "bad version");
  Reject(bytes.substr(0, 12), "truncated header");
  Reject(bytes.substr(0, 100), "truncated metadata");
  Reject(bytes.substr(0, bytes.size() - 1), "truncated tensor payload span");
  Reject(Fixture(1, 4, 2, false), "required architecture");
  Reject(Fixture(1, 4, 2, true, true), "mistyped alignment does not abort");
  Reject(Fixture(1, 0), "zero dimension does not divide by zero");
  Reject(Fixture(1, UINT64_MAX), "negative/overflow dimension");
  Reject(Fixture(1, 4, 0), "zero rank");
  Reject(Fixture(1, 4, 5), "oversized rank");
  Reject(Fixture(1, 4, 2, true, false, UINT64_MAX), "bad range");
  auto counts = bytes;
  for (unsigned i = 8; i < 16; ++i)
    counts[i] = static_cast<char>(255);
  Reject(counts, "bounded count");
  std::cout << "GGUF information checks passed\n";
}
