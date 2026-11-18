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
struct Recipe {
  const char *symbol;
  uint32_t storage_type;
  uint64_t block_values, block_bytes, scale_offset, scale_count, max_width;
  bool controls, quantizer;
  bool gated = false;
};
struct ProjectionLayout {
  const Recipe *recipe = nullptr;
  PreparedCase inputs;
  const model::TensorRecord *tensor = nullptr;
  const model::TensorRecord *up_tensor = nullptr;
  size_t input_argument = 0;
  uint64_t capacity = 0, width = 0, output_bytes = 0, weight_bytes = 0;
  bool dynamic_count = false;
  loomc_cmd_buffer_ref_t count_binding{};
  std::vector<uint8_t> count_bytes;
};
PreparationStatus DescribeProjectionCase(const Plan &plan,
                                         const Invocation &invocation,
                                         ProjectionLayout *out,
                                         std::string *error) {
  static constexpr Recipe recipes[] = {
      {"ggml_linear_q4k_q8_1_x4", 12, 256, 144, 0, 2, 32768, false, false},
      {"llm.projection.q4k_swiglu", 12, 256, 144, 0, 2, 32768, false, false,
       true},
      {"ggml_linear_q5k_q8_1_x4", 13, 256, 176, 0, 2, 32768, false, false},
      {"ggml_linear_q6k_q8_1_x4", 14, 256, 210, 208, 1, 17408, false, false},
      {"ggml_linear_q8_0_q8_1_x4", 8, 32, 34, 0, 1, 32768, false, false},
      {"ggml_linear_q5_0_q8_1_x4", 6, 32, 22, 0, 1, 32768, false, false},
      {"ggml_linear_q4k_q8_1_x4_compat", 12, 256, 144, 0, 2, 32768, true,
       false},
      {"llm.projection.q4k_compat_swiglu", 12, 256, 144, 0, 2, 32768, true,
       false, true},
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
  const size_t count = input_argument + (recipe->quantizer ? 2
                                         : recipe->gated   ? 4
                                                           : 3);
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
  const model::TensorRecord *up_tensor = nullptr;
  for (size_t member = 0;
       !recipe->quantizer && member < (recipe->gated ? 2u : 1u); ++member) {
    const auto weight = buffer(input_argument + 1 + member);
    parameter =
        FindBenchmarkParameter(plan, invocation, weight, weight_bytes, error);
    if (!parameter)
      return PreparationStatus::kFailure;
    const auto *selected_tensor = FindBenchmarkTensor(plan, parameter->key);
    if (member == 0)
      tensor = selected_tensor;
    else
      up_tensor = selected_tensor;
    if (weight.role != LOOMC_CMD_BUFFER_ROLE_FIXED ||
        weight.byte_offset != parameter->byte_offset ||
        parameter->byte_length != weight_bytes || !selected_tensor ||
        selected_tensor->storage_type != recipe->storage_type ||
        selected_tensor->rank != 2 ||
        selected_tensor->normalized_extents[0] != static_cast<int64_t>(width) ||
        selected_tensor->normalized_extents[1] !=
            static_cast<int64_t>(columns) ||
        selected_tensor->normalized_extents[2] != 1 ||
        selected_tensor->normalized_extents[3] != 1 ||
        selected_tensor->byte_size != weight_bytes)
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
    if (invocation.kernel_symbol == "llm.projection.q4k_compat_swiglu" &&
        fixed != -1)
      return Fail(error, "compatible paired SwiGLU requires dynamic count");
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
  for (size_t member = 0; parameter && member < (recipe->gated ? 2u : 1u);
       ++member) {
    const auto ref = buffer(input_argument + 1 + member);
    const auto *selected = member ? up_tensor : tensor;
    const auto *weight_parameter =
        FindBenchmarkParameter(plan, invocation, ref, weight_bytes, error);
    if (!weight_parameter)
      return PreparationStatus::kFailure;
    prepared.buffer_extents.push_back(
        {ref, weight_bytes,
         std::max<uint64_t>(4, weight_parameter->minimum_alignment)});
    prepared.representative_weight_spans.push_back(
        prepared.buffer_extents.back());
    prepared.weight_sources.push_back(
        {selected->name, selected->absolute_offset, weight_bytes});
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
  *out = {recipe,        std::move(prepared), tensor,
          up_tensor,     input_argument,      capacity,
          width,         output_bytes,        weight_bytes,
          dynamic_count, count_binding,       std::move(count_bytes)};
  return PreparationStatus::kReady;
}
PreparationStatus AllocateProjectionRoots(uint64_t max_input_bytes,
                                          PreparedCase *prepared,
                                          std::string *error) {
  bool budget_exceeded = false;
  if (!AllocateInputRoots(prepared->buffer_extents, max_input_bytes,
                          &prepared->roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  return PreparationStatus::kReady;
}
PreparationStatus FillProjectionInputs(const Plan &plan,
                                       const Invocation &invocation,
                                       const ProjectionLayout &layout,
                                       bool fill_shared_input,
                                       PreparedCase *out, std::string *error) {
  const auto *recipe = layout.recipe;
  const auto *tensor = layout.tensor;
  const auto input_argument = layout.input_argument;
  const auto input = invocation.arguments[input_argument].value.buffer;
  const auto output = invocation.arguments.back().value.buffer;
  const auto capacity = layout.capacity, width = layout.width;
  const auto output_bytes = layout.output_bytes,
             weight_bytes = layout.weight_bytes;
  auto &prepared = *out;
  if (layout.dynamic_count &&
      !SetInputBytes(layout.count_binding, layout.count_bytes, &prepared.roots,
                     error))
    return PreparationStatus::kFailure;
  if ((fill_shared_input &&
       !(recipe->quantizer
             ? SetFiniteF32Input(input, capacity * width, plan.workload.seed,
                                 &prepared.roots, error)
             : SetInputBytes(input,
                             PackedInputs(capacity, width, plan.workload.seed),
                             &prepared.roots, error))) ||
      !SetInputBytes(output, std::vector<uint8_t>(output_bytes, 0xA5),
                     &prepared.roots, error))
    return PreparationStatus::kFailure;
  for (size_t member = 0; tensor && member < (recipe->gated ? 2u : 1u);
       ++member) {
    const auto *selected = member ? layout.up_tensor : tensor;
    const auto weight =
        invocation.arguments[input_argument + 1 + member].value.buffer;
    if (!SetTensorInput(weight, plan.source_asset, selected->absolute_offset,
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
  return PreparationStatus::kReady;
}
} // namespace

PreparationStatus PrepareProjectionCase(const Plan &plan,
                                        const Invocation &invocation,
                                        uint64_t max_input_bytes,
                                        PreparedCase *out, std::string *error) {
  ProjectionLayout layout;
  const auto described =
      DescribeProjectionCase(plan, invocation, &layout, error);
  if (described != PreparationStatus::kReady)
    return described;
  if (!out)
    return Fail(error, "prepared projection case output is required");
  const auto allocated =
      AllocateProjectionRoots(max_input_bytes, &layout.inputs, error);
  if (allocated != PreparationStatus::kReady)
    return allocated;
  const auto filled = FillProjectionInputs(plan, invocation, layout, true,
                                           &layout.inputs, error);
  if (filled != PreparationStatus::kReady)
    return filled;
  *out = std::move(layout.inputs);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
PreparationStatus PrepareProjectionPair(const Plan &plan, size_t first_index,
                                        size_t second_index,
                                        uint64_t max_input_bytes,
                                        PreparedSequenceCase *out,
                                        std::string *error) {
  const auto unsupported = [&](const char *message) {
    if (error)
      *error = message;
    return PreparationStatus::kUnsupported;
  };
  if (!out || first_index >= plan.invocations.size() ||
      second_index >= plan.invocations.size())
    return Fail(error, "projection pair requires valid indices and output");
  const auto &first = plan.invocations[first_index];
  const auto &second = plan.invocations[second_index];
  if (first.kernel_symbol != "ggml_linear_q4k_q8_1_x4" ||
      second.kernel_symbol != first.kernel_symbol)
    return unsupported("projection pair supports canonical Q4 only");
  ProjectionLayout layouts[2];
  for (size_t i = 0; i < 2; ++i) {
    const auto status =
        DescribeProjectionCase(plan, i ? second : first, &layouts[i], error);
    if (status != PreparationStatus::kReady)
      return status;
  }
  if (first.program_ordinal != second.program_ordinal ||
      uint64_t(first.command_ordinal) + 1 != second.command_ordinal)
    return unsupported(
        "projection pair requires adjacent commands in the same program");
  const auto &commands = plan.programs[first.program_ordinal].commands;
  if (second.command_ordinal >= commands.size())
    return Fail(error,
                "projection pair is absent from the complete native schedule");
  for (const auto *inv : {&first, &second}) {
    const auto &command = commands[inv->command_ordinal];
    if (command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
        command.argument_count != inv->command.argument_count ||
        command.begins_barrier_wave != inv->command.begins_barrier_wave ||
        command.payload.dispatch.requirement_ordinal !=
            inv->command.payload.dispatch.requirement_ordinal ||
        !std::equal(std::begin(command.payload.dispatch.workgroup_count),
                    std::end(command.payload.dispatch.workgroup_count),
                    std::begin(inv->command.payload.dispatch.workgroup_count)))
      return Fail(error, "projection pair does not match the native schedule");
    if (!command.begins_barrier_wave)
      return unsupported(
          "projection pair requires two recorded serial barrier waves");
  }
  if (first.implementation_index != second.implementation_index ||
      first.request_root_ordinal != second.request_root_ordinal)
    return unsupported(
        "projection pair requires one retained implementation/root");
  if (first.implementation_index >= plan.implementations.size() ||
      first.request_root_ordinal >=
          plan.implementations[first.implementation_index].roots.size())
    return Fail(error,
                "projection pair retained implementation/root is missing");
  for (size_t i = 0; i < 3; ++i)
    if (first.arguments[i].value.scalar_bits !=
            second.arguments[i].value.scalar_bits ||
        first.command.payload.dispatch.workgroup_count[i] !=
            second.command.payload.dispatch.workgroup_count[i])
      return unsupported(
          "projection pair requires identical scalar/grid geometry");
  const auto a = first.arguments[3].value.buffer;
  const auto b = second.arguments[3].value.buffer;
  if (a.role != b.role || a.root_index != b.root_index ||
      a.byte_offset != b.byte_offset || a.byte_length != b.byte_length)
    return unsupported(
        "projection pair requires the exact shared packed input");
  PreparedCase prepared = layouts[0].inputs;
  prepared.setup_recipe_id = "q4k_q8_1_x4_pair_dyadic/v1";
  prepared.measurement_subject = "sequence";
  const auto &other = layouts[1].inputs;
  prepared.buffer_extents.insert(prepared.buffer_extents.end(),
                                 other.buffer_extents.begin(),
                                 other.buffer_extents.end());
  prepared.weight_sources.insert(prepared.weight_sources.end(),
                                 other.weight_sources.begin(),
                                 other.weight_sources.end());
  prepared.representative_weight_spans.insert(
      prepared.representative_weight_spans.end(),
      other.representative_weight_spans.begin(),
      other.representative_weight_spans.end());
  for (const size_t written : {size_t{2}, size_t{5}})
    for (size_t i = 0; i < prepared.buffer_extents.size(); ++i)
      if (i != written && Overlaps(prepared.buffer_extents[written],
                                   prepared.buffer_extents[i]))
        return unsupported(
            "projection pair outputs overlap a read or another output");
  const auto allocated =
      AllocateProjectionRoots(max_input_bytes, &prepared, error);
  if (allocated != PreparationStatus::kReady)
    return allocated;
  for (size_t i = 0; i < 2; ++i) {
    const auto filled = FillProjectionInputs(
        plan, i ? second : first, layouts[i], i == 0, &prepared, error);
    if (filled != PreparationStatus::kReady)
      return filled;
  }
  PreparedSequenceCase result;
  result.invocation_indices = {first_index, second_index};
  result.inputs = std::move(prepared);
  *out = std::move(result);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
