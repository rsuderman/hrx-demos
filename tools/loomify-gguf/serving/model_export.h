// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_SERVING_MODEL_EXPORT_H_
#define HRX_SOLUTIONS_LOOMIFY_SERVING_MODEL_EXPORT_H_
#include "tools/loomify-gguf/serving/package.h"
namespace hrx::loomify_gguf {
// Extracts data-only GGUF configuration, selects numerical providers and links
// maintained Loom model/epoch programs. Success owns the module and its
// admitted layout together; failure leaves the result unchanged.
bool BuildLlamaServeModule(const model::ModelDescriptor &,
                           const ModelPackageOptions &, loomc_context_t *,
                           loomc_workspace_t *, ConfiguredLlamaModel *,
                           std::string *error);
} // namespace hrx::loomify_gguf
#endif
