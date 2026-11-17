// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/qwen_span_inputs.h"

#include <limits>
#include <utility>

namespace hrx::loomify_gguf::benchmark {
namespace {
void Word(std::vector<uint8_t> &bytes, size_t offset, uint64_t value,
          unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}
} // namespace

bool BuildQwenSpanControls(const Workload &workload, bool paged,
                           uint64_t max_bytes, QwenSpanControls *out,
                           std::string *error, bool *budget_exceeded) {
  if (budget_exceeded)
    *budget_exceeded = false;
  const auto fail = [&](const std::string &message) {
    if (error)
      *error = message;
    return false;
  };
  if (!out || !workload.token_capacity || workload.token_capacity > 512 ||
      !workload.active_tokens ||
      workload.active_tokens > workload.token_capacity ||
      !workload.context_capacity || workload.context_capacity > INT32_MAX ||
      workload.history_tokens > workload.context_capacity ||
      workload.active_tokens >
          workload.context_capacity - workload.history_tokens)
    return fail("invalid Qwen single-span workload");
  // 3-word header,64 five-word descriptors,64 selected row indices;64
  // four-I64 resident records precede the row's optional packed-page map.
  constexpr uint64_t epoch_bytes = 12 + 64 * 20 + 64 * 4;
  constexpr uint64_t row_bytes = 64 * 32;
  const uint64_t pages = (workload.context_capacity + 63) / 64;
  const uint64_t map_bytes = paged ? pages * 4 : 0;
  if (row_bytes + map_bytes > std::numeric_limits<size_t>::max())
    return fail("Qwen single-span control size exceeds host limits");
  if (epoch_bytes > max_bytes || row_bytes > max_bytes - epoch_bytes ||
      map_bytes > max_bytes - epoch_bytes - row_bytes) {
    if (budget_exceeded)
      *budget_exceeded = true;
    return fail("Qwen single-span controls exceed byte budget");
  }
  QwenSpanControls prepared;
  prepared.epoch.resize(epoch_bytes, 0);
  prepared.rows.resize(row_bytes + map_bytes, 0);
  Word(prepared.epoch, 0, workload.active_tokens, 4);
  Word(prepared.epoch, 4, 1, 4);
  Word(prepared.epoch, 12, workload.active_tokens, 4);
  Word(prepared.epoch, 16, workload.history_tokens, 4);
  Word(prepared.epoch, 28, UINT32_MAX, 4);
  Word(prepared.rows, 24, workload.history_tokens, 8);
  if (paged) {
    prepared.page_map_origin = row_bytes;
    prepared.physical_capacity = pages * 64;
    for (uint64_t page = 0; page < pages; ++page)
      Word(prepared.rows, row_bytes + page * 4, pages - 1 - page, 4);
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::benchmark
