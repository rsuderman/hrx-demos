// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_MODEL_QWEN_CONFIGURATION_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_QWEN_CONFIGURATION_H_
#include "tools/model/model_descriptor.h"
namespace hrx::model {
// Borrow caller-owned unbuffered seekable file. Output owns all facts.
bool ExtractQwenConfiguration(FILE *, ModelDescriptor *, std::string *error);
} // namespace hrx::model
#endif
