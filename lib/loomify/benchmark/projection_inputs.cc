// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <cstring>
#include <iterator>

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
  struct Recipe {
    const char *symbol;
    uint32_t storage_type;
    uint64_t block_values, block_bytes, scale_offset, scale_count, max_width;
    bool controls, quantizer;
  };
  static constexpr Recipe recipes[] = {
      {"ggml_linear_q4k_q8_1_x4", 12, 256, 144, 0, 2, 32768, false, false},
      {"ggml_linear_q5k_q8_1_x4", 13, 256, 176, 0, 2, 32768, false, false},
      {"ggml_linear_q6k_q8_1_x4", 14, 256, 210, 208, 1, 17408, false, false},
      {"ggml_linear_q8_0_q8_1_x4", 8, 32, 34, 0, 1, 32768, false, false},
      {"ggml_linear_q5_0_q8_1_x4", 6, 32, 22, 0, 1, 32768, false, false},
      {"ggml_linear_q4k_q8_1_x4_compat", 12, 256, 144, 0, 2, 32768, true,
       false},
      {"ggml_linear_q6k_q8_1_x4_compat", 14, 256, 210, 208, 1, 17408, true,
       false},
      {"ggml_quantize_q8_1_x4_f32", 0, 128, 0, 0, 0, 32768, false, true},
      {"ggml_quantize_q8_1_x4_qk_compat_f32", 0, 128, 0, 0, 0, 32768, true,
       true}};
  const auto recipe =
      std::find_if(std::begin(recipes), std::end(recipes), [&](const auto &r) {
        return invocation.kernel_symbol == r.symbol;
      });
  if (recipe == std::end(recipes))
    return PreparationStatus::kUnsupported;
  if (!out)
    return Fail(error, "prepared projection case output is required");
  const size_t dimensions = recipe->quantizer ? 2 : 3;
  const size_t input_argument = dimensions + (recipe->controls ? 4 : 0);
  const size_t count = input_argument + (recipe->quantizer ? 2 : 3);
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
      invocation.program_ordinal >= plan.programs.size() ||
      invocation.arguments.size() != count ||
      invocation.command.argument_count != count)
    return Fail(error,
                "quantized projection/quantizer direct native ABI differs");
  PreparedCase prepared;
  for (size_t i = 0; i < count; ++i) {
    const auto kind = i < dimensions || (recipe->controls && i == dimensions)
                          ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                      : recipe->controls && i == dimensions + 1
                          ? LOOMC_CMD_ARGUMENT_KIND_B32
                      : recipe->controls && i == dimensions + 2
                          ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                          : LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    if (invocation.arguments[i].kind != kind)
      return Fail(error,
                  "quantized projection/quantizer argument types differ");
    prepared.source_argument_types.push_back(
        kind == LOOMC_CMD_ARGUMENT_KIND_INDEX    ? "index"
        : kind == LOOMC_CMD_ARGUMENT_KIND_B32    ? "i32"
        : kind == LOOMC_CMD_ARGUMENT_KIND_OFFSET ? "offset"
                                                 : "buffer");
  }
  const auto scalar = [&](size_t i) {
    return invocation.arguments[i].value.scalar_bits;
  };
  const auto buffer = [&](size_t i) {
    return invocation.arguments[i].value.buffer;
  };
  const uint64_t capacity = scalar(0), width = scalar(1);
  const uint64_t columns = recipe->quantizer ? 0 : scalar(2);
  const uint64_t width_multiple = std::max<uint64_t>(128, recipe->block_values);
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  if (!capacity || capacity > 2048 || width < width_multiple ||
      width > recipe->max_width || width % width_multiple || !grid[0] ||
      !grid[1] || grid[2] != 1 ||
      (!recipe->quantizer &&
       (!columns || columns > 262144 || grid[1] > capacity)) ||
      (recipe->quantizer && (grid[1] != 1 || grid[0] % (width / 128) ||
                             grid[0] / (width / 128) > capacity)))
    return Fail(error,
                "quantized projection/quantizer geometry/grid is unsupported");
  const uint64_t launched_rows =
      recipe->quantizer ? grid[0] / (width / 128) : grid[1];
  prepared.source_workloads = {launched_rows, width};
  if (!recipe->quantizer)
    prepared.source_workloads.push_back(columns);
  prepared.setup_recipe_id = std::string(recipe->symbol) + "_dyadic/v1";
  // Keep the original canonical recipe identity for existing Q4 instances.
  if (invocation.kernel_symbol == "ggml_linear_q4k_q8_1_x4")
    prepared.setup_recipe_id = "q4k_q8_1_x4_dyadic/v1";
  const uint64_t packed_bytes = capacity * (width / 128) * 144;
  const uint64_t input_bytes =
      recipe->quantizer ? capacity * width * 4 : packed_bytes;
  const uint64_t output_bytes =
      recipe->quantizer ? packed_bytes : capacity * columns * 4;
  const auto input = buffer(input_argument);
  const auto output = buffer(count - 1);
  const uint64_t weight_bytes =
      recipe->quantizer
          ? 0
          : columns * (width / recipe->block_values) * recipe->block_bytes;
  const model::TensorRecord *tensor = nullptr;
  const Parameter *parameter = nullptr;
  if (!recipe->quantizer) {
    const auto weight = buffer(input_argument + 1);
    parameter =
        FindBenchmarkParameter(plan, invocation, weight, weight_bytes, error);
    if (!parameter)
      return PreparationStatus::kFailure;
    tensor = FindBenchmarkTensor(plan, parameter->key);
    if (weight.role != LOOMC_CMD_BUFFER_ROLE_FIXED ||
        weight.byte_offset != parameter->byte_offset ||
        parameter->byte_length != weight_bytes || !tensor ||
        tensor->storage_type != recipe->storage_type || tensor->rank != 2 ||
        tensor->normalized_extents[0] != static_cast<int64_t>(width) ||
        tensor->normalized_extents[1] != static_cast<int64_t>(columns) ||
        tensor->normalized_extents[2] != 1 ||
        tensor->normalized_extents[3] != 1 || tensor->byte_size != weight_bytes)
      return Fail(
          error,
          "projection requires the exact selected GGUF quantized matrix");
  }
  std::vector<uint8_t> count_bytes;
  loomc_cmd_buffer_ref_t count_binding{};
  bool dynamic_count = false;
  if (recipe->controls) {
    const uint64_t bits = scalar(dimensions + 1);
    const uint32_t raw = static_cast<uint32_t>(bits);
    int32_t fixed;
    std::memcpy(&fixed, &raw, sizeof(fixed));
    if (bits > UINT32_MAX)
      return Fail(error, "projection fixed count exceeds its I32 carrier");
    dynamic_count = fixed < 0;
    const uint64_t begin = scalar(dimensions), origin = scalar(dimensions + 2);
    const uint64_t active =
        fixed < 0 ? plan.workload.active_tokens : static_cast<uint32_t>(fixed);
    if (begin > capacity || active > capacity - begin ||
        (fixed < 0 && (origin > INT64_MAX || origin % 4)))
      return Fail(error, "projection count range/origin is invalid");
    const uint32_t dynamic = static_cast<uint32_t>(plan.workload.active_tokens);
    if (fixed < 0 && plan.workload.active_tokens > INT32_MAX)
      return Fail(error, "projection dynamic count exceeds signed I32");
    count_bytes = {static_cast<uint8_t>(dynamic),
                   static_cast<uint8_t>(dynamic >> 8),
                   static_cast<uint8_t>(dynamic >> 16),
                   static_cast<uint8_t>(dynamic >> 24)};
    auto ref = buffer(dimensions + 3);
    prepared.buffer_extents.push_back({ref, fixed < 0 ? origin + 4 : 4, 4});
    count_binding = ref;
    if (fixed < 0) {
      if (ref.byte_offset > UINT64_MAX - origin)
        return Fail(error, "projection count address overflows");
      count_binding.byte_offset += origin;
    }
    count_binding.byte_length = 4;
  }
  prepared.buffer_extents.push_back({input, input_bytes, 4});
  if (parameter) {
    prepared.buffer_extents.push_back(
        {buffer(input_argument + 1), weight_bytes,
         std::max<uint64_t>(4, parameter->minimum_alignment)});
    prepared.representative_weight_spans.push_back(
        prepared.buffer_extents.back());
    prepared.weight_sources = {
        {tensor->name, tensor->absolute_offset, weight_bytes}};
  }
  prepared.buffer_extents.push_back({output, output_bytes, 4});
  // Nonnegative fixed counts bypass the native count view entirely. Retain its
  // binding/root, but do not impose semantic alias restrictions or initialize
  // its bytes: the selected program may pass a dummy view of input or output.
  const size_t first_semantic = recipe->controls && !dynamic_count ? 1 : 0;
  for (size_t i = first_semantic; i < prepared.buffer_extents.size(); ++i)
    for (size_t j = i + 1; j < prepared.buffer_extents.size(); ++j)
      if (Overlaps(prepared.buffer_extents[i], prepared.buffer_extents[j]))
        return Fail(error, "quantized recipe requires disjoint argument spans");
  bool budget_exceeded = false;
  if (!AllocateInputRoots(prepared.buffer_extents, max_input_bytes,
                          &prepared.roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  if (dynamic_count &&
      !SetInputBytes(count_binding, count_bytes, &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (!(recipe->quantizer
            ? SetFiniteF32Input(input, capacity * width, plan.workload.seed,
                                &prepared.roots, error)
            : SetInputBytes(input,
                            PackedInputs(capacity, width, plan.workload.seed),
                            &prepared.roots, error)) ||
      !SetInputBytes(output, std::vector<uint8_t>(output_bytes, 0xA5),
                     &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (tensor) {
    const auto weight = buffer(input_argument + 1);
    if (!SetTensorInput(weight, plan.source_asset, tensor->absolute_offset,
                        weight_bytes, &prepared.roots, error))
      return PreparationStatus::kFailure;
    const auto *root = FindInputRoot(prepared.roots, weight);
    const uint64_t begin = weight.byte_offset - root->original_byte_offset;
    for (uint64_t block = 0; block < weight_bytes; block += recipe->block_bytes)
      for (uint64_t scale = 0; scale < recipe->scale_count; ++scale) {
        const uint64_t offset =
            begin + block + recipe->scale_offset + scale * 2;
        const uint16_t bits =
            root->bytes[offset] |
            (static_cast<uint16_t>(root->bytes[offset + 1]) << 8);
        if ((bits & 0x7C00) == 0x7C00)
          return Fail(
              error,
              "GGUF quantized scale has nonfinite F16 values; input is not finite");
      }
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
