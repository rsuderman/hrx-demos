// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_SERVING_PACKAGE_INTERNAL_H_
#define HRX_SOLUTIONS_LOOMIFY_SERVING_PACKAGE_INTERNAL_H_
#include "tools/loomify-gguf/serving/package.h"
namespace hrx::loomify_gguf::detail {
// File-publication helpers shared by the two model-owned companion renderers.
// These strings are ready-to-write provider contents, not a model schema.
struct CompanionSources {
  std::string prepare, control, weights;
};
// Renders the production cold records directly from the admitted build layout.
bool RenderLlamaCompanions(const ModelPackageOptions &,
                           const LlamaLinkedLayout &, CompanionSources *,
                           std::string *error);
bool ReadText(const std::string &path, std::string *, std::string *error);
void ReplaceToken(std::string &, const std::string &key,
                  const std::string &value);
std::string Hex(const std::string &);
bool WriteServingPackage(const ModelPackageOptions &, const std::string &name,
                         const CompanionSources &, const loomc_module_t *,
                         std::string *error);
} // namespace hrx::loomify_gguf::detail
#endif
