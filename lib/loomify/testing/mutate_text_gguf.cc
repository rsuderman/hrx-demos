// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Test-only sparse metadata fixture. Never copies or reads checkpoint weights.
#include "tools/model/gguf_catalog.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <variant>
namespace fs = std::filesystem;
namespace {
bool Replace(std::string* text, const std::string& before, const std::string& after) {
  const auto at = text->find(before);
  if (at == std::string::npos || text->find(before, at + before.size()) != std::string::npos) return false;
  text->replace(at, before.size(), after); return true;
}
}
int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: mutate_text_gguf INPUT_GGUF NEW_GGUF NEW_TEMPLATE CASE\n";
    return 1;
  }
  if (fs::exists(argv[2]) || fs::exists(argv[3])) return 1;
  const auto close_file = [](FILE* file) { if (file) std::fclose(file); };
  std::unique_ptr<FILE, decltype(close_file)> input(std::fopen(argv[1], "rb"), close_file);
  if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0)) return 1;
  hrx::model::GgufCatalog catalog; std::string error;
  if (!hrx::model::ReadGgufCatalog(input.get(), &catalog, &error)) {
    std::cerr << error << '\n'; return 1;
  }
  const auto* field = hrx::model::FindMetadata(catalog, "tokenizer.chat_template");
  const auto* original = field ? std::get_if<std::string>(&field->scalar) : nullptr;
  if (!original) return 1;
  std::string changed = *original;
  const std::string kind = argv[4]; bool replaced = false;
  if (kind == "tool-prose")
    replaced = Replace(&changed, "Reminder:", "Source-specific protocol reminder:");
  else if (kind == "effort-default")
    replaced = Replace(&changed, "reasoning_effort|default('xhigh')", "reasoning_effort|default('low')");
  else if (kind == "effort-medium")
    replaced = Replace(&changed, "reasoning_effort|default('xhigh')", "reasoning_effort|default('medium')");
  else if (kind == "llama-date")
    replaced = Replace(&changed, "26 Jul 2024", "10 Nov 2026");
  else if (kind == "llama-prefix")
    replaced = Replace(&changed, "Cutting Knowledge Date: December 2023\\n",
                       "Knowledge snapshot: May 2025\\n") &&
               Replace(&changed, "Today Date: ", u8"Current date café: ");
  else if (kind == "reject-branch")
    replaced = Replace(&changed, "if tools and tools is iterable", "if tools or tools is iterable");
  else if (kind == "reject-marker")
    replaced = Replace(&changed, "<function=example_function_name>", "<function=changed_name>");
  if (!replaced) return 1;
  std::string header(catalog.data_offset, '\0');
  if (std::fseek(input.get(), 0, SEEK_SET) ||
      std::fread(header.data(), 1, header.size(), input.get()) != header.size()) return 1;
  const auto at = header.find(*original);
  if (at == std::string::npos || at < 8 || header.find(*original, at + original->size()) != std::string::npos) return 1;
  uint64_t length = 0;
  for (unsigned i=0; i<8; ++i) length |= uint64_t(static_cast<unsigned char>(header[at-8+i])) << (8*i);
  if (length != original->size()) return 1;
  header.replace(at, original->size(), changed);
  for (unsigned i=0; i<8; ++i) header[at-8+i] = char((uint64_t(changed.size()) >> (8*i)) & 255);
  const auto delta = int64_t(changed.size()) - int64_t(original->size());
  const auto end = uint64_t(int64_t(catalog.header_end) + delta);
  const auto offset = (end + catalog.alignment - 1) / catalog.alignment * catalog.alignment;
  header.resize(offset, '\0');
  std::ofstream fixture(argv[2], std::ios::binary), text(argv[3], std::ios::binary);
  fixture.write(header.data(), header.size()); text.write(changed.data(), changed.size());
  fixture.close(); text.close();
  std::error_code ec;
  if (!fixture || !text) return 1;
  fs::resize_file(argv[2], catalog.file_size + offset - catalog.data_offset, ec);
  if (ec) { std::cerr << ec.message() << '\n'; return 1; }
  std::cout << "Sparse metadata-only fixture; payload is holes and is not valid for inference.\n";
}
