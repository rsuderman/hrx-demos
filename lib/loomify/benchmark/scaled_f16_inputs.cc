// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <array>

namespace hrx::loomify_gguf::benchmark {
namespace {
using Ref = loomc_cmd_buffer_ref_t;
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
void Word(std::vector<uint8_t> &bytes, size_t offset, uint32_t value,
          unsigned width = 4) {
  for (unsigned i = 0; i < width; ++i)
    bytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
}
struct Weight {
  Ref binding;
  const Parameter *parameter;
  const model::TensorRecord *tensor;
  uint64_t bytes;
};
bool FindWeight(const Plan &plan, const Invocation &invocation, Ref binding,
                uint64_t k, uint64_t n, bool q6, Weight *out,
                std::string *error) {
  const uint64_t bytes = n * (k / 256) * (q6 ? 210 : 144);
  const auto *parameter =
      FindBenchmarkParameter(plan, invocation, binding, bytes, error);
  const auto *tensor =
      parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
  if (!tensor || binding.byte_offset != parameter->byte_offset ||
      parameter->byte_length != bytes || tensor->byte_size != bytes ||
      tensor->storage_type != (q6 ? 14u : 12u) || tensor->rank != 2 ||
      tensor->normalized_extents !=
          std::array<int64_t, 4>{static_cast<int64_t>(k),
                                 static_cast<int64_t>(n), 1, 1}) {
    if (error)
      *error = "scaled-F16 projection requires its exact selected GGUF matrix";
    return false;
  }
  *out = {binding, parameter, tensor, bytes};
  return true;
}
bool LoadWeight(const Plan &plan, const Weight &weight, bool q6,
                PreparedCase *prepared, std::string *error) {
  if (!SetTensorInput(weight.binding, plan.source_asset,
                      weight.tensor->absolute_offset, weight.bytes,
                      &prepared->roots, error))
    return false;
  const auto *root = FindInputRoot(prepared->roots, weight.binding);
  const uint64_t begin =
      weight.binding.byte_offset - root->original_byte_offset;
  const uint64_t block_bytes = q6 ? 210 : 144;
  for (uint64_t block = 0; block < weight.bytes; block += block_bytes)
    for (unsigned header = 0; header < (q6 ? 1u : 2u); ++header) {
      const uint64_t offset = begin + block + (q6 ? 208 : 2 * header);
      const uint16_t bits =
          root->bytes[offset] | (uint16_t(root->bytes[offset + 1]) << 8);
      if ((bits & 0x7c00) == 0x7c00) {
        if (error)
          *error = "scaled-F16 GGUF matrix contains a nonfinite encoded scale";
        return false;
      }
    }
  prepared->weight_sources.push_back(
      {weight.tensor->name, weight.tensor->absolute_offset, weight.bytes});
  prepared->representative_weight_spans.push_back(
      {weight.binding, weight.bytes,
       std::max<uint64_t>(4, weight.parameter->minimum_alignment)});
  return true;
}
} // namespace

PreparationStatus PrepareScaledF16Case(const Plan &plan,
                                       const Invocation &invocation,
                                       uint64_t max_input_bytes,
                                       PreparedCase *out, std::string *error) {
  const auto &name = invocation.kernel_symbol;
  const bool prepare = name == "llm.projection.prepare_scaled_f16";
  const bool fused = name == "llm.platform.q4k_scaled_f16_swiglu";
  const bool q6 = name == "llm.platform.q6k_scaled_f16_store";
  if (!prepare && !fused && !q6 && name != "llm.platform.q4k_scaled_f16_store")
    return PreparationStatus::kUnsupported;
  const size_t argument_count = fused ? 9 : 8;
  if (!out ||
      invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
      invocation.arguments.size() != argument_count ||
      invocation.command.argument_count != argument_count)
    return Fail(error,
                "scaled-F16 operation requires its exact direct native ABI");
  for (size_t i = 0; i < argument_count; ++i) {
    const auto kind = i < 3               ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                      : prepare && i == 3 ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                                          : LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    if (invocation.arguments[i].kind != kind)
      return Fail(error, "scaled-F16 native argument carrier differs");
  }
  const auto &args = invocation.arguments;
  const uint64_t rows = args[0].value.scalar_bits;
  const uint64_t k = args[prepare ? 2 : 1].value.scalar_bits;
  const uint64_t columns = args[prepare ? 1 : 2].value.scalar_bits;
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  if (prepare
          ? (!rows || rows > 512 || columns < 64 || columns > 512 ||
             columns % 64 || rows > columns || k < 256 || k > 32768 ||
             k % 256 || grid[0] != columns || grid[1] != 1 || grid[2] != 1)
          : (rows < 64 || rows > 8192 || rows % 64 || k < 256 || k > 65536 ||
             k % 256 || columns < 64 || columns > 262144 || columns % 64 ||
             grid[0] != columns / 64 || grid[1] != rows / 64 || grid[2] != 1))
    return Fail(error,
                "scaled-F16 dimensions/grid violate the selected operation");
  const uint64_t active = plan.workload.active_tokens;
  if (!active || active > rows)
    return Fail(error, "scaled-F16 active count exceeds its source rows");
  const uint64_t count_origin = prepare ? args[3].value.scalar_bits : 0;
  if (count_origin % 4 || count_origin > UINT64_MAX - 12)
    return Fail(error, "scaled-F16 count origin is invalid");
  const auto count = args[prepare ? 4 : 3].value.buffer;
  const auto input = args[prepare ? 5 : 4].value.buffer;
  const auto scales = args[prepare ? 6 : 5].value.buffer;
  const auto output = args.back().value.buffer;
  if (count.byte_offset > UINT64_MAX - count_origin)
    return Fail(error, "scaled-F16 native count offset overflows");
  PreparedCase prepared;
  prepared.setup_recipe_id = prepare ? "prepare_scaled_f16_finite_f32/v1"
                                     : "scaled_f16_dyadic_unit_rows/v1";
  prepared.source_workloads = prepare ? std::vector<uint64_t>{rows, columns, k}
                                      : std::vector<uint64_t>{rows, k, columns};
  prepared.source_argument_types.assign(argument_count, "buffer");
  for (size_t i = 0; i < 3; ++i)
    prepared.source_argument_types[i] = "index";
  if (prepare)
    prepared.source_argument_types[3] = "offset";
  const uint64_t padded = prepare ? columns : rows;
  prepared.buffer_extents = {
      {count, count_origin + 12, 4},
      {input, rows * k * (prepare ? 4 : 2), prepare ? 4u : 2u},
      {scales, padded * 8, 4}};
  std::vector<Weight> weights;
  if (!prepare) {
    for (size_t i = 6; i < argument_count - 1; ++i) {
      Weight weight;
      if (!FindWeight(plan, invocation, args[i].value.buffer, k, columns, q6,
                      &weight, error))
        return PreparationStatus::kFailure;
      weights.push_back(weight);
      prepared.buffer_extents.push_back(
          {weight.binding, weight.bytes,
           std::max<uint64_t>(4, weight.parameter->minimum_alignment)});
    }
  }
  prepared.buffer_extents.push_back(
      {output, prepare ? padded * k * 2 : rows * columns * 4,
       prepare ? 2u : 4u});
  // Synthetic input streams are independent policies, so overlapping distinct
  // reads are unsupported by this fixture recipe. Repeated operands resolving
  // to one actual immutable matrix are known to receive identical bytes.
  auto accesses = prepared.buffer_extents;
  // Validate declared views before applying fixture-policy exclusions,
  // retaining their complete native capacity while using actual accesses for
  // alias tests.
  for (auto &extent : prepared.buffer_extents) {
    if (extent.binding.byte_length != UINT64_MAX) {
      if (extent.byte_length > extent.binding.byte_length)
        return Fail(error, "scaled-F16 footprint exceeds its native view");
      extent.byte_length = extent.binding.byte_length;
    }
    if (extent.binding.byte_offset > UINT64_MAX - extent.byte_length)
      return Fail(error, "scaled-F16 native view overflows its root");
  }
  accesses.front().binding.byte_offset += count_origin;
  accesses.front().byte_length = 12;
  for (size_t i = 0; i < accesses.size(); ++i)
    for (size_t j = i + 1; j < accesses.size(); ++j)
      if (Overlaps(accesses[i], accesses[j])) {
        const bool writes =
            j == accesses.size() - 1 || (prepare && (i == 2 || j == 2));
        const bool same_weight =
            !prepare && i >= 3 && j < accesses.size() - 1 &&
            weights[i - 3].parameter == weights[j - 3].parameter &&
            weights[i - 3].tensor == weights[j - 3].tensor;
        if (writes)
          return Fail(error, "scaled-F16 output/read spans must be disjoint");
        if (!same_weight) {
          if (error)
            *error =
                "scaled-F16 fixture policy does not support overlapping distinct inputs";
          return PreparationStatus::kUnsupported;
        }
      }
  bool budget_exceeded = false;
  if (!AllocateInputRoots(prepared.buffer_extents, max_input_bytes,
                          &prepared.roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  auto control = count;
  control.byte_offset += count_origin;
  control.byte_length = 12;
  std::vector<uint8_t> packet(12);
  Word(packet, 0, static_cast<uint32_t>(active));
  if (!SetInputBytes(control, packet, &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (prepare) {
    if (!SetFiniteF32Input(input, rows * k, plan.workload.seed, &prepared.roots,
                           error))
      return PreparationStatus::kFailure;
  } else {
    // Exact finite dyadics below1 are reachable from prepare_scaled_f16 with
    // scale/inverse=(1,1). Padded rows carry the producer's explicit zero
    // policy.
    constexpr uint16_t values[] = {0x2400, 0x2800, 0x2c00, 0x3000,
                                   0xa400, 0xa800, 0xac00, 0xb000};
    std::vector<uint8_t> halves(rows * k * 2), metadata(rows * 8);
    for (uint64_t row = 0; row < active; ++row) {
      Word(metadata, row * 8, 0x3f800000);
      Word(metadata, row * 8 + 4, 0x3f800000);
      for (uint64_t column = 0; column < k; ++column)
        Word(halves, (row * k + column) * 2,
             values[(row + column + plan.workload.seed % 8) % 8], 2);
    }
    if (!SetInputBytes(input, halves, &prepared.roots, error) ||
        !SetInputBytes(scales, metadata, &prepared.roots, error))
      return PreparationStatus::kFailure;
    for (const auto &weight : weights)
      if (!LoadWeight(plan, weight, q6, &prepared, error))
        return PreparationStatus::kFailure;
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
