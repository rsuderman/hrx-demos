// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures.h"
#include "lib/loomify/architectures/llama.h"
#include "lib/loomify/architectures/lfm3.h"
#include "lib/loomify/architectures/qwen2.h"
#include "lib/loomify/architectures/qwen3.h"
#include "lib/loomify/architectures/gemma3.h"
#include "lib/loomify/architectures/qwen35.h"
#include "lib/serving/library_resources.h"
namespace hrx::loomify_gguf {
const ArchitectureFiles *SelectArchitecture(std::string_view family) {
  static const ArchitectureFiles entries[] = {
      {"lfm3",
       resources::kLfm3Implementation,
       resources::kLfm3Preparation,
       resources::kLfm3Execution,
       resources::kWeights,
       "loomify-lfm3",
       {"generated.lfm3_epoch"},
       {},
       internal::EmitLfm3,
       PublicationLayout::kTextAndTurn,
       262144},
      {"qwen2",
       resources::kQwen2Implementation,
       resources::kQwen2Preparation,
       resources::kQwen2Execution,
       resources::kWeights,
       "loomify-qwen2",
       {"generated.qwen2_epoch"},
       {RequiredOperation::kScaledF16ProjectionV1},
       internal::EmitQwen2,
       PublicationLayout::kTextAndTurn,
       262144},
      {"qwen3",
       resources::kQwen3Implementation,
       resources::kQwen3Preparation,
       resources::kQwen3Execution,
       resources::kWeights,
       "loomify-qwen3",
       {"generated.qwen3_epoch"},
       {RequiredOperation::kScaledF16ProjectionV1},
       internal::EmitQwen3,
       PublicationLayout::kTextAndTurn,
       262144},
      {"llama",
       resources::kLlamaImplementation,
       resources::kLlamaPreparation,
       resources::kLlamaExecution,
       resources::kWeights,
       "loomify-llama",
       {"generated.llama_epoch"},
       {},
       internal::EmitLlama,
       PublicationLayout::kThreeStops,
       131072},
      {"gemma3",
       resources::kGemmaImplementation,
       resources::kGemmaPreparation,
       resources::kGemmaExecution,
       resources::kWeights,
       "loomify-gemma3",
       {"generated.gemma3_epoch"},
       {},
       internal::EmitGemma,
       PublicationLayout::kEosAndTurn,
       262144},
      {"qwen35",
       resources::kQwenImplementation,
       resources::kQwenPreparation,
       resources::kQwenExecution,
       resources::kWeights,
       "loomify-qwen35",
       {"generated.qwen_epoch"},
       {RequiredOperation::kAttentionPrefillH256V1},
       internal::EmitQwen,
       PublicationLayout::kHostManaged,
       262144}};
  for (const auto &entry : entries)
    if (family == entry.family)
      return &entry;
  return nullptr;
}
} // namespace hrx::loomify_gguf
