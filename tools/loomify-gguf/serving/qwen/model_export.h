// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_QWEN_MODEL_EXPORT_H_
#define HRX_SOLUTIONS_LOOMIFY_QWEN_MODEL_EXPORT_H_
#include "tools/loomify-gguf/serving/qwen/package.h"
namespace hrx::loomify_gguf {
// Preloads the actual owned registry closure, then calls the sole descriptor-
// driven Qwen recipe. No isolated-to-epoch or alternate model recipe is used.
// Caller owns the returned module; failure returns null and leaves geometry
// unchanged. Geometry comes from the exact successful native load.
bool BuildQwenServeModule(const model::ModelDescriptor &,
                          const ModelPackageOptions &, loomc_context_t *,
                          loomc_workspace_t *, loomc_module_t **,
                          dispatch::QwenNativeGeometry *, std::string *error);
} // namespace hrx::loomify_gguf
#endif
