// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_SPARK2_5_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_SPARK2_5_H_
#include "lib/loomify/configuration.h"
namespace hrx::loomify_gguf::internal {
bool EmitSpark25(const model::ModelDescriptor &, const std::string &,
                 const ConfigurationSettings &, ConfigurationFacts *,
                 std::string *);
}
#endif
