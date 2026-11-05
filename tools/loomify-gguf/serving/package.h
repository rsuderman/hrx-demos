// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_SERVING_PACKAGE_H_
#define HRX_SOLUTIONS_LOOMIFY_SERVING_PACKAGE_H_
#include "loomc/module.h"
#include "tools/model/model_descriptor.h"
#include <cstdint>
#include <string>
namespace hrx::loomify_gguf {
struct LlamaPackageOptions {
  std::string source_root;
  std::string output_directory;
  std::string gguf;
  std::string tokenizer;
  std::string prefill_root = "generated.llama_prefill";
  std::string decode_root = "generated.llama_decode";
  std::string epoch_root = "generated.llama_epoch";
  uint64_t context_capacity = 512;
  uint64_t prefill_capacity = 128;
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_BYTECODE;
};
// Writes into a fresh directory. A model document is published only after all
// providers serialize successfully. External assets are referenced, never
// copied. The module must contain all three admitted roots and numerical
// providers.
bool WriteLlamaPackage(const model::ModelDescriptor &,
                       const LlamaPackageOptions &, const loomc_module_t *,
                       std::string *error);
} // namespace hrx::loomify_gguf
#endif
