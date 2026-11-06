// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/qwen/package.h"
#include "tools/loomify-gguf/serving/qwen/testing/fixture.h"
#include <cstdio>
#include <fstream>
int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  hrx::loomify_gguf::ModelPackageOptions options;
  options.source_root = argv[1];
  options.prefill_capacity = 128;
  options.context_capacity = 512;
  hrx::loomify_gguf::detail::CompanionSources sources;
  std::string error;
  if (!hrx::loomify_gguf::RenderQwenCompanions(
          options, hrx::loomify_gguf::testing::Geometry(), &sources, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  // Only import binding metadata changes for valid-input component callbacks.
  // The production prepare function and geometry/storage bodies are unchanged.
  for (const char *name : {"input", "prepare", "json"}) {
    std::string token = std::string("import(\"") + name + "\") ";
    size_t pos = 0;
    while ((pos = sources.prepare.find(token, pos)) != std::string::npos)
      sources.prepare.erase(pos, token.size());
  }
  std::ofstream file(argv[2]);
  file << sources.prepare;
  file.close();
  return file ? 0 : 1;
}
