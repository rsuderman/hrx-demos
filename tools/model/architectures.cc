// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/architectures.h"
#include "tools/model/architectures/llama.h"
#include "tools/model/architectures/qwen2.h"
#include "tools/model/architectures/qwen3.h"
#include "tools/model/architectures/gemma3.h"
#include "tools/model/architectures/qwen35.h"
namespace hrx::model::internal {
const ModelArchitecture *SelectModelArchitecture(std::string_view family) {
  static const ModelArchitecture *const entries[] = {
      &kLlamaArchitecture, &kQwen2Architecture, &kQwen3Architecture,
      &kGemma3Architecture, &kQwen35Architecture};
  for (const auto *entry : entries)
    if (family == entry->family)
      return entry;
  return nullptr;
}
} // namespace hrx::model::internal
