// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_PARAMETERS_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_PARAMETERS_H_

#include "lib/dispatch/invocation.h"
#include "lib/llama/structure.h"

namespace hrx::dispatch {
// Registers immutable tensors used by this existing-operation range. The owner
// supplies canonical keys and storage facts; no GGUF offsets become native root
// offsets. The owner and native module/body must outlive the session. Call once
// before emission. Discard the session after any native registration failure.
// Construction validation completes before the first native mutation.
bool BindRegionParameters(const llama::LlamaStructure &owner,
                          const llama::LlamaRegionView &region,
                          loomify_gguf::Value parameters,
                          NativeEmissionSession &session, std::string *error);
} // namespace hrx::dispatch
#endif
