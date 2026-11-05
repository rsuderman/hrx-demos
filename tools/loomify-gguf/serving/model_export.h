// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_SERVING_MODEL_EXPORT_H_
#define HRX_SOLUTIONS_LOOMIFY_SERVING_MODEL_EXPORT_H_
#include "tools/loomify-gguf/serving/package.h"
namespace hrx::loomify_gguf {
// Uses the sole descriptor-driven model recipe, then wraps it for the existing
// one-row ordinary packed-epoch Serve interface. Caller owns returned module.
bool BuildLlamaServeModule(const model::ModelDescriptor &,
                           const LlamaPackageOptions &, loomc_context_t *,
                           loomc_workspace_t *, loomc_module_t **,
                           std::string *error);
} // namespace hrx::loomify_gguf
#endif
