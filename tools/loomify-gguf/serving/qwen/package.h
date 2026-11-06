// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_QWEN_PACKAGE_H_
#define HRX_SOLUTIONS_LOOMIFY_QWEN_PACKAGE_H_
#include "lib/dispatch/qwen.h"
#include "tools/loomify-gguf/serving/package_internal.h"
namespace hrx::loomify_gguf {
inline constexpr char kQwenEpochRoot[] = "generated.qwen_epoch";
// Geometry comes from the successful common native loader for these exact
// settings. Rendering does not derive model configuration or numerical routes.
// Failure leaves sources unchanged.
bool RenderQwenCompanions(const ModelPackageOptions &,
                          const dispatch::QwenNativeGeometry &,
                          detail::CompanionSources *sources,
                          std::string *error);
// Writes the independently owned Qwen role modules and existing Serve document.
// The caller supplies the complete generated epoch module; checkpoint/tokenizer
// are external. No numerical model is constructed by this file-publication API.
bool WriteQwenPackage(const model::ModelDescriptor &,
                      const ModelPackageOptions &,
                      const dispatch::QwenNativeGeometry &,
                      const loomc_module_t *, std::string *error);
} // namespace hrx::loomify_gguf
#endif
