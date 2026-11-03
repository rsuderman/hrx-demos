// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LLAMA_DETAIL_EMBEDDING_STRUCTURE_H_
#define HRX_SOLUTIONS_LIB_LLAMA_DETAIL_EMBEDDING_STRUCTURE_H_

#include "lib/llama/detail/structure_construction.h"

namespace hrx::llama::detail {
// Append one logical embedding into the owner's existing residual and record
// its region. Inputs must already exist. Resolves the unique global
// token_embedding role through the owner; no numerical provider admission or
// implicit export. Discard private construction after failure; graph/parameter
// mutations do not support rollback. No complete or partial model is published
// here.
bool AppendEmbedding(StructureConstruction *owner, std::string *error);
} // namespace hrx::llama::detail
#endif
