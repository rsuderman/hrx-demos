// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>

namespace hrx::loomify_gguf::benchmark {
namespace {
PreparationStatus Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return PreparationStatus::kFailure;
}
bool Overlaps(const BufferExtent &a, const BufferExtent &b) {
  return a.binding.role == b.binding.role &&
         a.binding.root_index == b.binding.root_index &&
         (a.binding.byte_offset <= b.binding.byte_offset
              ? b.binding.byte_offset - a.binding.byte_offset < a.byte_length
              : a.binding.byte_offset - b.binding.byte_offset < b.byte_length);
}
void Half(uint16_t value, size_t offset, std::vector<uint8_t> &bytes) {
  bytes[offset] = static_cast<uint8_t>(value);
  bytes[offset + 1] = static_cast<uint8_t>(value >> 8);
}
// For |sum|<=90, sum/128 is exactly representable as normal F16 or zero.
// Each block's ±127 pair fixes amax=127/128, so these bytes also describe the
// canonical packer's exact result for the corresponding dyadic logical input.
uint16_t ScaledSum(int sum) {
  if (!sum)
    return 0;
  const uint16_t sign = sum < 0 ? 0x8000 : 0;
  unsigned magnitude = static_cast<unsigned>(sum < 0 ? -sum : sum);
  unsigned exponent = 0;
  for (unsigned n = magnitude; n > 1; n >>= 1)
    ++exponent;
  return sign | static_cast<uint16_t>((exponent + 8) << 10) |
         static_cast<uint16_t>((magnitude << (10 - exponent)) & 0x3FF);
}
std::vector<uint8_t> PackedInputs(uint64_t capacity, uint64_t width,
                                  uint64_t seed) {
  const uint64_t groups = width / 128;
  std::vector<uint8_t> bytes(capacity * groups * 144);
  for (uint64_t row = 0; row < capacity; ++row)
    for (uint64_t group = 0; group < groups; ++group) {
      const uint64_t base = (row * groups + group) * 144;
      for (unsigned block = 0; block < 4; ++block) {
        int sum = 0;
        for (unsigned j = 0; j < 32; ++j) {
          const int value =
              j == 0   ? 127
              : j == 1 ? -127
                       : static_cast<int>(
                             (seed % 7 + row % 7 + group % 7 + block + j) % 7) -
                             3;
          bytes[base + 16 + block * 32 + j] = static_cast<uint8_t>(value);
          sum += value;
        }
        Half(0x2000, base + block * 4, bytes); // scale=1/128.
        Half(ScaledSum(sum), base + block * 4 + 2, bytes);
      }
    }
  return bytes;
}
} // namespace

PreparationStatus PrepareProjectionCase(const Plan &plan,
                                        const Invocation &invocation,
                                        uint64_t max_input_bytes,
                                        PreparedCase *out, std::string *error) {
  if (invocation.kernel_symbol != "ggml_linear_q4k_q8_1_x4")
    return PreparationStatus::kUnsupported;
  if (!out)
    return Fail(error, "prepared projection case output is required");
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
      invocation.program_ordinal >= plan.programs.size() ||
      invocation.arguments.size() != 6 ||
      invocation.command.argument_count != 6)
    return Fail(error,
                "canonical Q4_K projection requires its direct native ABI");
  for (size_t i = 0; i < 6; ++i)
    if (invocation.arguments[i].kind != (i < 3
                                             ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                                             : LOOMC_CMD_ARGUMENT_KIND_BUFFER))
      return Fail(error, "canonical Q4_K projection argument types differ");
  const uint64_t capacity = invocation.arguments[0].value.scalar_bits;
  const uint64_t width = invocation.arguments[1].value.scalar_bits;
  const uint64_t columns = invocation.arguments[2].value.scalar_bits;
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  if (!capacity || capacity > 2048 || width < 256 || width > 32768 ||
      width % 256 || !columns || columns > 262144 || !grid[0] || !grid[1] ||
      grid[1] > capacity || grid[2] != 1)
    return Fail(error,
                "canonical Q4_K projection geometry/grid is unsupported");
  // Bounds above keep every product below uint64_t and exact source footprints.
  const uint64_t packed_bytes = capacity * (width / 128) * 144;
  const uint64_t weight_bytes = columns * (width / 256) * 144;
  const uint64_t output_bytes = capacity * columns * 4;
  const auto packed = invocation.arguments[3].value.buffer;
  const auto weight = invocation.arguments[4].value.buffer;
  const auto output = invocation.arguments[5].value.buffer;
  const Parameter *parameter =
      FindBenchmarkParameter(plan, invocation, weight, weight_bytes, error);
  if (!parameter)
    return PreparationStatus::kFailure;
  const auto *tensor = FindBenchmarkTensor(plan, parameter->key);
  if (weight.role != LOOMC_CMD_BUFFER_ROLE_FIXED ||
      weight.byte_offset != parameter->byte_offset ||
      parameter->byte_length != weight_bytes || !tensor ||
      tensor->storage_type != 12 || tensor->rank != 2 ||
      tensor->normalized_extents[0] != static_cast<int64_t>(width) ||
      tensor->normalized_extents[1] != static_cast<int64_t>(columns) ||
      tensor->normalized_extents[2] != 1 ||
      tensor->normalized_extents[3] != 1 || tensor->byte_size != weight_bytes)
    return Fail(error,
                "projection requires the exact selected GGUF Q4_K matrix");

  PreparedCase prepared;
  prepared.setup_recipe_id = "q4k_q8_1_x4_dyadic/v1";
  prepared.source_workloads = {grid[1], width, columns};
  prepared.source_argument_types = {"index",  "index",  "index",
                                    "buffer", "buffer", "buffer"};
  prepared.weight_sources = {
      {tensor->name, tensor->absolute_offset, weight_bytes}};
  prepared.buffer_extents = {
      {packed, packed_bytes, 4},
      {weight, weight_bytes,
       std::max<uint64_t>(4, parameter->minimum_alignment)},
      {output, output_bytes, 4}};
  for (size_t i = 0; i < prepared.buffer_extents.size(); ++i)
    for (size_t j = i + 1; j < prepared.buffer_extents.size(); ++j)
      if (Overlaps(prepared.buffer_extents[i], prepared.buffer_extents[j]))
        return Fail(error,
                    "canonical projection requires disjoint argument spans");
  prepared.representative_weight_spans = {prepared.buffer_extents[1]};
  bool budget_exceeded = false;
  if (!AllocateInputRoots(prepared.buffer_extents, max_input_bytes,
                          &prepared.roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  if (!SetInputBytes(packed, PackedInputs(capacity, width, plan.workload.seed),
                     &prepared.roots, error) ||
      !SetTensorInput(weight, plan.source_asset, tensor->absolute_offset,
                      weight_bytes, &prepared.roots, error) ||
      !SetInputBytes(output, std::vector<uint8_t>(output_bytes, 0xA5),
                     &prepared.roots, error))
    return PreparationStatus::kFailure;

  const auto *weight_root = FindInputRoot(prepared.roots, weight);
  const uint64_t begin = weight.byte_offset - weight_root->original_byte_offset;
  for (uint64_t block = 0; block < weight_bytes; block += 144)
    for (unsigned lane = 0; lane < 4; lane += 2) {
      const uint16_t bits =
          weight_root->bytes[begin + block + lane] |
          (static_cast<uint16_t>(weight_root->bytes[begin + block + lane + 1])
           << 8);
      if ((bits & 0x7C00) == 0x7C00)
        return Fail(
            error,
            "GGUF Q4_K scale has nonfinite F16 values; input is not finite");
    }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
