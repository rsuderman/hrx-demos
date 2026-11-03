// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LLAMA_DETAIL_ATTENTION_STRUCTURE_H_
#define HRX_SOLUTIONS_LIB_LLAMA_DETAIL_ATTENTION_STRUCTURE_H_
#include "lib/llama/detail/structure_construction.h"
namespace hrx::llama::detail {
struct NormalizedAttentionRegion {
  size_t layer_slot = 0;
  size_t operation_begin = 0, operation_end = 0;
  operations::ValueId residual, key_cache, value_cache;
};
// Logical append only. On failure leave out unchanged and discard construction.
// Records the known attention region; semantic exports belong to composition.
bool AppendNormalizedAttention(StructureConstruction *owner, size_t layer_slot,
                               NormalizedAttentionRegion *out,
                               std::string *error);
} // namespace hrx::llama::detail
#endif
