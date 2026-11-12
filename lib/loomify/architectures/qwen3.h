// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_QWEN3_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_QWEN3_H_
#include "lib/loomify/configuration.h"
namespace hrx::loomify_gguf::internal {
bool EmitQwen3(const model::ModelDescriptor &, const std::string &,
               const ConfigurationSettings &, ConfigurationFacts *,
               std::string *);
} // namespace hrx::loomify_gguf::internal
#endif
