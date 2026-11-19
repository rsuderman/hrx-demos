// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/gguf_catalog.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {
std::string JsonString(const std::string &value) {
  std::ostringstream escaped;
  escaped << '"';
  for (unsigned char byte : value) {
    if (byte < 0x20) {
      escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << unsigned(byte);
    } else {
      std::ostringstream quoted;
      quoted << std::quoted(std::string(1, static_cast<char>(byte)));
      const auto text = quoted.str();
      escaped << text.substr(1, text.size() - 2);
    }
  }
  escaped << '"';
  return escaped.str();
}
int Fail(const std::string &message) {
  std::cerr << "emit_tensor_facts: " << message << '\n';
  return 1;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3)
    return Fail("usage: emit_tensor_facts GGUF OUTPUT_JSON");
  const std::string asset = argv[1];
  const std::string output_path = argv[2];
  if (asset.empty() || output_path.empty())
    return Fail("GGUF source and JSON output paths must be nonempty");
  if (asset == output_path)
    return Fail("JSON output must differ from immutable GGUF source");
  std::error_code equivalent_error;
  if (std::filesystem::equivalent(asset, output_path, equivalent_error))
    return Fail("JSON output aliases immutable GGUF source");
  std::unique_ptr<FILE, decltype(&std::fclose)> file(
      std::fopen(asset.c_str(), "rb"), std::fclose);
  if (!file)
    return Fail("cannot open GGUF source: " + asset);
  if (std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0)
    return Fail("cannot configure unbuffered GGUF input");
  hrx::model::GgufCatalog catalog;
  std::string error;
  if (!hrx::model::ReadGgufCatalog(file.get(), &catalog, &error))
    return Fail("GGUF header: " + error);
  std::ostringstream json;
  json << "{\n  \"source_asset\": " << JsonString(asset)
       << ",\n  \"file_size\": " << catalog.file_size
       << ",\n  \"architecture\": " << JsonString(catalog.architecture)
       << ",\n  \"name\": " << JsonString(catalog.name)
       << ",\n  \"tensors\": {\n";
  for (size_t index = 0; index < catalog.tensors.size(); ++index) {
    const auto &tensor = catalog.tensors[index];
    json << "    " << JsonString(tensor.name)
         << ": {\"name\": " << JsonString(tensor.name)
         << ", \"tag\": " << tensor.storage_type
         << ", \"offset\": " << tensor.absolute_offset
         << ", \"bytes\": " << tensor.byte_size << ", \"dims\": [";
    for (size_t dim = 0; dim < tensor.normalized_extents.size(); ++dim)
      json << (dim ? ", " : "") << tensor.normalized_extents[dim];
    json << "], \"rank\": " << tensor.rank << "}"
         << (index + 1 == catalog.tensors.size() ? "\n" : ",\n");
  }
  json << "  }\n}\n";
  if (!json)
    return Fail("cannot serialize JSON facts");
  std::ofstream output(output_path, std::ios::out | std::ios::trunc);
  if (!output)
    return Fail("cannot open JSON output: " + output_path);
  output << json.str();
  if (!output)
    return Fail("cannot write JSON output: " + output_path);
  output.close();
  if (!output)
    return Fail("cannot close JSON output: " + output_path);
  return 0;
}
