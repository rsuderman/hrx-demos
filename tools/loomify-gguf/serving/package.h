// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_SERVING_PACKAGE_H_
#define HRX_SOLUTIONS_LOOMIFY_SERVING_PACKAGE_H_
#include "loomc/module.h"
#include "tools/model/model_descriptor.h"
#include <cstdint>
#include <string>
namespace hrx::loomify_gguf {
inline constexpr char kLlamaPrefillRoot[] = "generated.llama_prefill";
inline constexpr char kLlamaDecodeRoot[] = "generated.llama_decode";
inline constexpr char kLlamaEpochRoot[] = "generated.llama_epoch";

struct ModelPackageOptions {
  std::string source_root;
  std::string output_directory;
  std::string gguf;
  std::string tokenizer;
  uint64_t context_capacity = 512;
  uint64_t prefill_capacity = 128;
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_BYTECODE;
};
// Writes into a fresh directory. A model document is published only after all
// providers serialize successfully. External assets are referenced, never
// copied. The module must contain all three admitted roots and numerical
// providers.
bool WriteLlamaPackage(const model::ModelDescriptor &,
                       const ModelPackageOptions &, const loomc_module_t *,
                       std::string *error);
} // namespace hrx::loomify_gguf
#endif
