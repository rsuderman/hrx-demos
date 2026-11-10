// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/architectures.h"
namespace hrx::loomify_gguf {
const ArchitectureFiles *SelectArchitecture(std::string_view family) {
  static const ArchitectureFiles entries[] = {
      {"llama", "lib/architectures/llama/implementation.loombc",
       "lib/architectures/llama/prepare_library.loombc",
       "lib/architectures/llama/execution.loombc",
       "lib/serving/weights.loom",
       "loomify-llama", {"generated.llama_prefill", "generated.llama_decode", "generated.llama_epoch"}},
      {"qwen35", "lib/architectures/qwen35/implementation.loombc",
       "lib/architectures/qwen35/prepare_library.loombc",
       "lib/architectures/qwen35/execution.loombc",
       "lib/serving/weights.loom",
       "loomify-qwen35", {"generated.qwen_epoch"}}};
  for (const auto &entry : entries)
    if (family == entry.family) return &entry;
  return nullptr;
}
} // namespace hrx::loomify_gguf
