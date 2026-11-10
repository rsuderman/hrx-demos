// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures.h"
#include "lib/serving/library_resources.h"
namespace hrx::loomify_gguf {
const ArchitectureFiles *SelectArchitecture(std::string_view family) {
  static const ArchitectureFiles entries[] = {
      {"llama", resources::kLlamaImplementation,
       resources::kLlamaPreparation,
       resources::kLlamaExecution,
       resources::kWeights,
       "loomify-llama", {"generated.llama_epoch"}},
      {"qwen35", resources::kQwenImplementation,
       resources::kQwenPreparation,
       resources::kQwenExecution,
       resources::kWeights,
       "loomify-qwen35", {"generated.qwen_epoch"}}};
  for (const auto &entry : entries)
    if (family == entry.family) return &entry;
  return nullptr;
}
} // namespace hrx::loomify_gguf
