// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_LINKING_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_LINKING_H_
#include "loomc/module.h"
#include <string>
#include <vector>
namespace hrx::loomify_gguf {
struct ArchitectureFiles;
struct PlatformFiles;
// The common numerical composition path. Platform operation products are
// required BUILD-owned inputs, selected before native materialization.
bool LinkNumericalProducts(loomc_context_t *, loomc_workspace_t *,
                           const std::string &library_root,
                           const std::string &configuration,
                           const ArchitectureFiles &, const PlatformFiles &,
                           loomc_module_t **out, std::string *error);
// Private native operation; only the selected products are read. Returns one
// owning strict LINK result. On failure leaves *out unchanged.
bool LinkConfiguredProducts(loomc_context_t *, loomc_workspace_t *,
                           const std::string &configuration,
                           const std::vector<std::string> &product_paths,
                           const std::vector<const char *> &roots,
                           bool normalize_numerical, loomc_module_t **out,
                           std::string *error);
} // namespace hrx::loomify_gguf
#endif
