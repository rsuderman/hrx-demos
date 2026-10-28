// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_SOURCE_BUILDER_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_SOURCE_BUILDER_H_

#include <string>

namespace hrx::loomify_gguf {

// Returns deterministic loomcxx source for an empty model entry. The caller
// owns output delivery; constructing source performs no I/O or compilation.
std::string EmitEmptyProgram();

}  // namespace hrx::loomify_gguf

#endif  // HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_SOURCE_BUILDER_H_
