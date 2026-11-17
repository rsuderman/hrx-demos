// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace hrx::loomify_gguf::benchmark {
namespace {
using Ref = loomc_cmd_buffer_ref_t;
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
bool Add(uint64_t a, uint64_t b, uint64_t *out) {
  if (b > UINT64_MAX - a)
    return false;
  *out = a + b;
  return true;
}
bool Mul(uint64_t a, uint64_t b, uint64_t *out) {
  if (a && b > UINT64_MAX / a)
    return false;
  *out = a * b;
  return true;
}
bool Overlaps(const BufferExtent &a, const BufferExtent &b) {
  return a.binding.role == b.binding.role &&
         a.binding.root_index == b.binding.root_index &&
         (a.binding.byte_offset <= b.binding.byte_offset
              ? b.binding.byte_offset - a.binding.byte_offset < a.byte_length
              : a.binding.byte_offset - b.binding.byte_offset < b.byte_length);
}
bool Slice(Ref ref, uint64_t offset, uint64_t length, Ref *out) {
  if ((ref.byte_length != UINT64_MAX &&
       (offset > ref.byte_length || length > ref.byte_length - offset)) ||
      !Add(ref.byte_offset, offset, &ref.byte_offset))
    return false;
  ref.byte_length = length;
  *out = ref;
  return true;
}
std::vector<uint8_t> Words(std::initializer_list<uint32_t> values) {
  std::vector<uint8_t> bytes;
  for (const auto value : values)
    for (unsigned lane = 0; lane < 4; ++lane)
      bytes.push_back(static_cast<uint8_t>(value >> (lane * 8)));
  return bytes;
}
bool F32(Ref ref, uint64_t count, uint64_t seed,
         std::vector<PreparedRoot> *roots, std::string *error) {
  std::vector<uint8_t> bytes(static_cast<size_t>(count) * 4);
  for (size_t i = 0; i < count; ++i) {
    // Small, nonzero dyadic values keep QK accumulation bounded. Absolute
    // positions give overlapping read-only views consistent initialization.
    const uint64_t ordinal = i + ref.byte_offset / 4 + seed % 16;
    const float value = static_cast<float>((ordinal % 8) + 1) / 1024.0f *
                        ((ordinal / 8) % 2 ? -1.0f : 1.0f);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned lane = 0; lane < 4; ++lane)
      bytes[i * 4 + lane] = static_cast<uint8_t>(bits >> (lane * 8));
  }
  return SetInputBytes(ref, bytes, roots, error);
}
bool F16(Ref ref, uint64_t count, uint64_t seed,
         std::vector<PreparedRoot> *roots, std::string *error) {
  // These are exact dyadic F16 values, not an approximate float converter.
  constexpr uint16_t values[] = {0x2400, 0x2800, 0x2c00, 0x3000,
                                 0xa400, 0xa800, 0xac00, 0xb000};
  std::vector<uint8_t> bytes(static_cast<size_t>(count) * 2);
  for (size_t i = 0; i < count; ++i) {
    const auto bits = values[(i + ref.byte_offset / 2 + seed % 8) % 8];
    bytes[i * 2] = static_cast<uint8_t>(bits);
    bytes[i * 2 + 1] = static_cast<uint8_t>(bits >> 8);
  }
  return SetInputBytes(ref, bytes, roots, error);
}
} // namespace

// Ordinary/factored decoder ABI, shared by current numerical architectures.
// KV publication only writes cache; attention only reads cache. Neither needs
// restoration when their authored disjoint-input/output contract holds.
PreparationStatus PrepareAttentionCase(const Plan &plan,
                                       const Invocation &invocation,
                                       uint64_t max_input_bytes,
                                       PreparedCase *out, std::string *error) {
  const auto fail = [&](const std::string &message) {
    Fail(error, message);
    return PreparationStatus::kFailure;
  };
  const auto unsupported = [&](const std::string &message) {
    Fail(error, message);
    return PreparationStatus::kUnsupported;
  };
  const auto &name = invocation.kernel_symbol;
  const bool publish = name == "llm.decoder.publish_kv_h128" ||
                       name == "llm.decoder.publish_kv_h256";
  const bool gated = name == "llm.attention.gated_window_rows_h128" ||
                     name == "llm.attention.gated_window_rows_h256";
  if (!publish && !gated && name != "llm.decoder.attention_rows_h128")
    return PreparationStatus::kUnsupported;
  if (!out)
    return fail("prepared attention case output is required");
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
    return unsupported("attention recipe needs resolved direct launch counts");
  const uint64_t h = name.find("h256") != std::string::npos ? 256 : 128;
  const size_t scalar_count = publish ? 7 : gated ? 10 : 9;
  const size_t buffer_count = gated ? 6 : 5;
  if (invocation.arguments.size() != scalar_count + buffer_count ||
      invocation.command.argument_count != invocation.arguments.size())
    return fail("attention native ABI argument count differs");
  const auto &args = invocation.arguments;
  const size_t offset_begin = publish ? 4 : gated ? 7 : 6;
  for (size_t i = 0; i < scalar_count; ++i) {
    const auto kind = i >= offset_begin    ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                      : !publish && i == 5 ? LOOMC_CMD_ARGUMENT_KIND_B32
                                           : LOOMC_CMD_ARGUMENT_KIND_INDEX;
    if (args[i].kind != kind)
      return fail("attention native scalar ABI differs");
  }
  for (size_t i = scalar_count; i < args.size(); ++i)
    if (args[i].kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER)
      return fail("attention requires native buffer arguments");
  const auto *grid = invocation.command.payload.dispatch.workgroup_count;
  const uint64_t rows = grid[1];
  const uint64_t query_heads = grid[0];
  const uint64_t key_heads = publish ? query_heads : args[0].value.scalar_bits;
  const size_t layer_arg = publish ? 0 : 1;
  const uint64_t layer = args[layer_arg].value.scalar_bits;
  const uint64_t layers = args[layer_arg + 1].value.scalar_bits;
  const uint64_t capacity = args[layer_arg + 2].value.scalar_bits;
  const uint64_t pool = args[layer_arg + 3].value.scalar_bits;
  const uint64_t cache_base = args[offset_begin].value.scalar_bits;
  const uint64_t control_origin = args[offset_begin + 1].value.scalar_bits;
  const uint64_t map_origin = args[offset_begin + 2].value.scalar_bits;
  if (!rows || rows > 512 || grid[2] != 1 || !query_heads ||
      query_heads > 128 || !key_heads || key_heads > 128 ||
      query_heads % key_heads || !layers || layers > 64 || layer >= layers ||
      !capacity || capacity > 262144 || pool > 4194304 || (pool && pool % 64) ||
      cache_base % 2 || control_origin % 4 || (pool && map_origin % 4) ||
      !plan.workload.active_tokens || plan.workload.active_tokens > rows ||
      plan.workload.history_tokens >= capacity ||
      plan.workload.active_tokens > capacity - plan.workload.history_tokens)
    return fail("attention requires valid heads, cache and active positions");
  if (!publish) {
    if (args[5].value.scalar_bits > UINT32_MAX)
      return fail("attention scale exceeds its selected B32 payload");
    const uint32_t bits = static_cast<uint32_t>(args[5].value.scalar_bits);
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    if (!std::isfinite(scale))
      return fail("attention fixture requires a finite selected scale");
  }
  const uint64_t physical = pool ? pool : capacity;
  const uint64_t map_count = pool ? (capacity + 63) / 64 : 0;
  const uint64_t live_pages =
      (plan.workload.history_tokens + plan.workload.active_tokens + 63) / 64;
  if (pool && live_pages > pool / 64)
    return unsupported("attention live prefix exceeds the selected page pool");
  uint64_t plane_bytes, cache_bytes, epoch_bytes, map_bytes;
  if (!Mul(physical, key_heads * h * 2, &plane_bytes) ||
      !Mul(layer * 2 + 2, plane_bytes, &cache_bytes) ||
      !Add(cache_base, cache_bytes, &cache_bytes) ||
      !Add(control_origin, 20, &epoch_bytes) ||
      !Add(map_count ? map_origin : 0, map_count * 4, &map_bytes))
    return fail("attention input footprint overflows");
  const uint64_t query_bytes = rows * query_heads * h * 4;
  const uint64_t kv_bytes = rows * key_heads * h * 4;
  const auto epoch = args[scalar_count].value.buffer;
  const auto map = args[scalar_count + 1].value.buffer;
  const auto input = args[scalar_count + 2].value.buffer;
  const auto second = args[scalar_count + 3].value.buffer;
  const auto output = args[scalar_count + 4].value.buffer;
  const auto cache =
      publish ? output : args[scalar_count + (gated ? 4 : 3)].value.buffer;
  const auto destination = publish ? cache : args.back().value.buffer;
  std::vector<BufferExtent> extents = {
      {epoch, epoch_bytes, 4},
      {map, std::max<uint64_t>(4, map_bytes), 4},
      {input, publish ? kv_bytes : query_bytes, 4}};
  if (publish)
    extents.push_back({second, kv_bytes, 4});
  else if (gated)
    extents.push_back({second, rows * query_heads * 4, 4});
  extents.push_back({cache, cache_bytes, 2});
  if (!publish)
    extents.push_back({destination, query_bytes, 4});
  Ref epoch_values, map_values = map, cache_values;
  if (!Slice(epoch, control_origin, 20, &epoch_values) ||
      (map_count && !Slice(map, map_origin, map_count * 4, &map_values)) ||
      !Slice(cache, cache_base + layer * 2 * plane_bytes, plane_bytes * 2,
             &cache_values))
    return fail("attention view exceeds the selected native binding");
  std::vector<BufferExtent> reads = {
      {epoch_values, 20, 4}, {input, publish ? kv_bytes : query_bytes, 4}};
  if (map_count)
    reads.push_back({map_values, map_count * 4, 4});
  if (publish)
    reads.push_back({second, kv_bytes, 4});
  else {
    reads.push_back({cache_values, plane_bytes * 2, 2});
    if (gated)
      reads.push_back({second, rows * query_heads * 4, 4});
  }
  const BufferExtent writes{publish ? cache_values : destination,
                            publish ? plane_bytes * 2 : query_bytes,
                            publish ? uint64_t{2} : uint64_t{4}};
  for (const auto &read : reads)
    if (Overlaps(read, writes))
      return unsupported("attention recipe requires disjoint read/write spans");
  PreparedCase prepared;
  prepared.setup_recipe_id = publish ? "kv_publish_f16/v1"
                             : gated ? "gated_window_attention_f16/v1"
                                     : "causal_attention_f16/v1";
  prepared.buffer_extents = extents;
  prepared.source_workloads =
      publish ? std::vector<uint64_t>{rows, key_heads}
              : std::vector<uint64_t>{rows, query_heads, key_heads};
  for (size_t i = 0; i < scalar_count; ++i)
    prepared.source_argument_types.push_back(i >= offset_begin    ? "offset"
                                             : !publish && i == 5 ? "f32"
                                                                  : "index");
  prepared.source_argument_types.insert(prepared.source_argument_types.end(),
                                        buffer_count, "buffer");
  bool budget_exceeded = false;
  if (!AllocateInputRoots(extents, max_input_bytes, &prepared.roots, error,
                          &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  if (!SetInputBytes(epoch_values,
                     Words({static_cast<uint32_t>(plan.workload.active_tokens),
                            static_cast<uint32_t>(plan.workload.history_tokens),
                            0, 0, UINT32_MAX}),
                     &prepared.roots, error) ||
      !F32(input, (publish ? kv_bytes : query_bytes) / 4, plan.workload.seed,
           &prepared.roots, error) ||
      !F16(cache_values, cache_values.byte_length / 2, plan.workload.seed + 3,
           &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (publish &&
      !F32(second, kv_bytes / 4, plan.workload.seed, &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (gated && !F32(second, rows * query_heads, plan.workload.seed,
                    &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (map_count) {
    std::vector<uint8_t> bytes;
    for (uint64_t page = 0; page < map_count; ++page) {
      const uint32_t physical_page =
          static_cast<uint32_t>((page + 1) % (pool / 64));
      const auto word = Words({physical_page});
      bytes.insert(bytes.end(), word.begin(), word.end());
    }
    if (!SetInputBytes(map_values, bytes, &prepared.roots, error))
      return PreparationStatus::kFailure;
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
