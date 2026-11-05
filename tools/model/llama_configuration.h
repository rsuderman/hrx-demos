// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_MODEL_LLAMA_CONFIGURATION_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_LLAMA_CONFIGURATION_H_
#include "tools/model/model_descriptor.h"
namespace hrx::model {
// Borrows caller-owned unbuffered seekable file; output owns all facts.
bool ExtractLlamaConfiguration(FILE *, ModelDescriptor *, std::string *error);
} // namespace hrx::model
#endif
