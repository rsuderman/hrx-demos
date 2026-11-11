// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures.h"
#include "lib/serving/library_resources.h"
namespace hrx::loomify_gguf {
const ArchitectureFiles *SelectArchitecture(std::string_view family) {
  static const ArchitectureFiles entries[] = {
      {"qwen3", resources::kQwen3Implementation,
       resources::kQwen3Preparation, resources::kQwen3Execution, resources::kWeights,
       "loomify-qwen3", {"generated.qwen3_epoch"},
       {RequiredOperation::kScaledF16ProjectionV1}},
      {"llama", resources::kLlamaImplementation,
       resources::kLlamaPreparation,
       resources::kLlamaExecution,
       resources::kWeights,
       "loomify-llama", {"generated.llama_epoch"}},
      {"gemma3", resources::kGemmaImplementation,
       resources::kGemmaPreparation,
       resources::kGemmaExecution,
       resources::kWeights,
       "loomify-gemma3", {"generated.gemma3_epoch"}},
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
