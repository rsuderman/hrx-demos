// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_QWEN_SPAN_INPUTS_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_QWEN_SPAN_INPUTS_H_

#include "lib/loomify/benchmark/plan.h"

namespace hrx::loomify_gguf::benchmark {

struct QwenSpanControls {
  std::vector<uint8_t> epoch;
  std::vector<uint8_t> rows;
  uint64_t page_map_origin = 0;
  uint64_t physical_capacity = 0;
};

// Shared qwen38 span ABI, directly matching kernels/qwen/common/spans.loom:
// one active descriptor at byte12, compact token begin0/resident row0, no
// selected outputs, and one row's equal anchor/writer/KV origins0 with
// fork=history. Optional page entries at byte2048 reverse the allocated64-token
// pages. The returned padded capacity is a candidate: callers must match actual
// native kernel pool geometry before using this map. Does not choose state
// layer offsets. Rejects malformed workload or excessive allocation before
// changing *out.
bool BuildQwenSpanControls(const Workload &workload, bool paged,
                           uint64_t max_bytes, QwenSpanControls *out,
                           std::string *error, bool *budget_exceeded = nullptr);

} // namespace hrx::loomify_gguf::benchmark
#endif
