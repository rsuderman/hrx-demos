// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include "lib/loomify/benchmark/qwen_span_inputs.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace hrx::loomify_gguf::benchmark {
namespace {
float Float(const PreparedRoot &root, uint64_t offset) {
  uint32_t bits = 0;
  for (unsigned i = 0; i < 4; ++i)
    bits |= static_cast<uint32_t>(root.bytes[offset + i]) << (8 * i);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
void AppendFloat(float value, std::vector<uint8_t> *bytes) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (unsigned i = 0; i < 4; ++i)
    bytes->push_back(static_cast<uint8_t>(bits >> (8 * i)));
}
bool Overlaps(const BufferExtent &a, const BufferExtent &b) {
  return a.binding.role == b.binding.role &&
         a.binding.root_index == b.binding.root_index &&
         (a.binding.byte_offset <= b.binding.byte_offset
              ? b.binding.byte_offset - a.binding.byte_offset < a.byte_length
              : a.binding.byte_offset - b.binding.byte_offset < b.byte_length);
}
struct Weight {
  size_t argument;
  uint64_t elements;
  const Parameter *parameter;
  const model::TensorRecord *tensor;
};
bool BindWeight(const Plan &plan, const Invocation &invocation, size_t argument,
                uint64_t elements, std::vector<Weight> *weights,
                std::string *error) {
  const auto &ref = invocation.arguments[argument].value.buffer;
  const auto *parameter =
      FindBenchmarkParameter(plan, invocation, ref, elements * 4, error);
  const auto *tensor =
      parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
  if (!parameter || !tensor || tensor->storage_type != 0 ||
      tensor->byte_size != elements * 4 ||
      parameter->byte_length != elements * 4 ||
      parameter->byte_offset != ref.byte_offset) {
    if (error && parameter)
      *error = "GDN requires exact selected F32 parameter spans";
    return false;
  }
  weights->push_back({argument, elements, parameter, tensor});
  return true;
}
bool LoadWeight(const Plan &plan, const Invocation &invocation,
                const Weight &weight, PreparedCase *prepared,
                std::string *error) {
  const auto &ref = invocation.arguments[weight.argument].value.buffer;
  if (!SetTensorInput(ref, plan.source_asset, weight.tensor->absolute_offset,
                      weight.elements * 4, &prepared->roots, error))
    return false;
  const auto *root = FindInputRoot(prepared->roots, ref);
  const auto begin = ref.byte_offset - root->original_byte_offset;
  for (uint64_t i = 0; i < weight.elements; ++i)
    if (!std::isfinite(Float(*root, begin + i * 4))) {
      if (error)
        *error = "GDN learned F32 parameter contains nonfinite values";
      return false;
    }
  prepared->weight_sources.push_back({weight.tensor->name,
                                      weight.tensor->absolute_offset,
                                      weight.elements * 4});
  prepared->representative_weight_spans.push_back(
      {ref, weight.elements * 4,
       std::max<uint64_t>(4, weight.parameter->minimum_alignment)});
  return true;
}
} // namespace

// Preparation follows the configured span operation, not a Qwen model graph.
// Synthetic recurrent Q/K rows have inverse lengths derived from those same
// rows; they are not claimed to be captured model activations or an oracle.
PreparationStatus PrepareGdnCase(const Plan &plan, const Invocation &invocation,
                                 uint64_t max_input_bytes, PreparedCase *out,
                                 std::string *error) {
  const auto fail = [&](const std::string &message) {
    if (error)
      *error = message;
    return PreparationStatus::kFailure;
  };
  const bool convolution = invocation.kernel_symbol == "qwen38_gdn_conv_spans";
  const bool recurrent =
      invocation.kernel_symbol == "qwen38_gdn_recurrent_spans";
  if (!convolution && !recurrent)
    return PreparationStatus::kUnsupported;
  if (!out)
    return fail("GDN prepared case output is required");
  const size_t count = convolution ? 9 : 11;
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
      invocation.program_ordinal >= plan.programs.size() ||
      invocation.command.argument_count != count ||
      invocation.arguments.size() != count ||
      invocation.arguments[2].kind != LOOMC_CMD_ARGUMENT_KIND_OFFSET)
    return fail("GDN selected native ABI differs from the span contract");
  for (size_t i = 0; i < count; ++i)
    if (i != 2 && !(convolution && i == 3) &&
        invocation.arguments[i].kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER)
      return fail("GDN span buffer argument has the wrong native carrier");
  if (convolution) {
    if (invocation.arguments[3].kind != LOOMC_CMD_ARGUMENT_KIND_B32 ||
        invocation.arguments[3].value.scalar_bits > UINT32_MAX)
      return fail("GDN convolution requires exact F32 epsilon bits");
    const uint32_t bits =
        static_cast<uint32_t>(invocation.arguments[3].value.scalar_bits);
    float epsilon;
    std::memcpy(&epsilon, &bits, sizeof(epsilon));
    if (!std::isfinite(epsilon) || epsilon <= 0)
      return fail("GDN convolution epsilon must be finite positive F32");
  }
  if (!plan.model.ssm)
    return fail("GDN model descriptor has no recurrent geometry");
  const float norm_epsilon = static_cast<float>(plan.model.rms_epsilon);
  if (recurrent && (!std::isfinite(norm_epsilon) || norm_epsilon <= 0))
    return fail(
        "GDN recurrent input preparation requires the model's F32 normalization epsilon");
  const auto &ssm = *plan.model.ssm;
  if (ssm.state_width != 128 || ssm.key_heads != 16 || !ssm.value_heads ||
      ssm.value_heads > 48 || ssm.value_heads % ssm.key_heads ||
      ssm.inner_width != ssm.state_width * ssm.value_heads ||
      ssm.convolution_kernel != 4)
    return fail(
        "GDN descriptor geometry differs from the maintained span kernels");
  const uint64_t qkv_width =
      2 * ssm.key_heads * ssm.state_width + ssm.inner_width;
  const uint64_t history_elements = 3 * qkv_width;
  const uint64_t state_elements = ssm.state_width * ssm.inner_width;
  if (ssm.convolution_history_elements != history_elements ||
      ssm.recurrent_state_elements != state_elements)
    return fail("GDN descriptor state/history extents are inconsistent");
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  const uint64_t spans = convolution ? grid[1] : grid[2];
  if (!spans || spans > 64 ||
      grid[0] != (convolution ? ssm.key_heads : ssm.value_heads) ||
      (convolution ? grid[2] != 1 : grid[1] != ssm.state_width / 4))
    return fail("GDN selected grid differs from its descriptor geometry");
  QwenSpanControls controls;
  bool budget_exceeded = false;
  // Unpaged controls have a fixed 3,596-byte setup footprint. Validate these
  // and the selected metadata before applying the complete roots+seed budget
  // below, so a small budget does not mask an invalid selected native view.
  if (!BuildQwenSpanControls(plan.workload, false, UINT64_MAX, &controls,
                             error))
    return PreparationStatus::kFailure;
  const uint64_t active = plan.workload.active_tokens;
  const uint64_t layer_origin = invocation.arguments[2].value.scalar_bits;
  const uint64_t live_elements =
      convolution ? history_elements : state_elements;
  const uint64_t live_bytes = live_elements * 4;
  if (layer_origin > INT64_MAX || layer_origin % 4 ||
      layer_origin > UINT64_MAX - live_bytes)
    return fail("GDN state origin/footprint is invalid");
  std::vector<uint64_t> lengths(count, 0);
  lengths[0] = controls.epoch.size();
  lengths[1] = controls.rows.size();
  const size_t qkv_argument = convolution ? 4 : 3;
  const size_t state_argument = convolution ? 6 : 9;
  lengths[qkv_argument] = active * qkv_width * 4;
  lengths[state_argument] = layer_origin + live_bytes;
  std::vector<Weight> weights;
  if (convolution) {
    lengths[5] = 4 * qkv_width * 4;
    lengths[7] = active * qkv_width * 4;
    lengths[8] = active * 2 * ssm.key_heads * 4;
    if (!BindWeight(plan, invocation, 5, 4 * qkv_width, &weights, error))
      return PreparationStatus::kFailure;
    const auto &tensor = *weights[0].tensor;
    if (tensor.rank != 2 || tensor.normalized_extents[0] != 4 ||
        tensor.normalized_extents[1] != static_cast<int64_t>(qkv_width))
      return fail("GDN convolution taps have the wrong F32 tensor shape");
  } else {
    lengths[4] = active * 2 * ssm.key_heads * 4;
    lengths[5] = lengths[6] = active * ssm.value_heads * 4;
    lengths[7] = lengths[8] = ssm.value_heads * 4;
    lengths[10] = active * ssm.inner_width * 4;
    if (!BindWeight(plan, invocation, 7, ssm.value_heads, &weights, error) ||
        !BindWeight(plan, invocation, 8, ssm.value_heads, &weights, error))
      return PreparationStatus::kFailure;
    for (const auto &weight : weights)
      if (weight.tensor->rank != 1 || weight.tensor->normalized_extents[0] !=
                                          static_cast<int64_t>(ssm.value_heads))
        return fail(
            "GDN recurrent coefficients have the wrong F32 tensor shape");
  }
  PreparedCase prepared;
  prepared.setup_recipe_id =
      convolution ? "gdn_conv_spans_f32/v1" : "gdn_recurrent_spans_f32/v1";
  prepared.measurement_subject = "reset_plus_kernel";
  prepared.reset_policy = "copy_seed_each_invocation";
  prepared.source_workloads = {spans};
  prepared.source_argument_types.assign(count, "buffer");
  prepared.source_argument_types[2] = "offset";
  if (convolution)
    prepared.source_argument_types[3] = "f32";
  for (size_t i = 0; i < count; ++i)
    if (lengths[i]) {
      // Preserve native fixed view extents (including capacity padding). Only
      // remaining-range roots need an operation-owned concrete realization.
      const auto &ref = invocation.arguments[i].value.buffer;
      if (ref.byte_length != UINT64_MAX) {
        if (lengths[i] > ref.byte_length)
          return fail("GDN required span exceeds its selected native view");
        lengths[i] = ref.byte_length;
      }
      uint64_t alignment = 4;
      if (i == 1)
        alignment = 8;
      for (const auto &weight : weights)
        if (weight.argument == i)
          alignment = std::max<uint64_t>(alignment,
                                         weight.parameter->minimum_alignment);
      prepared.buffer_extents.push_back({ref, lengths[i], alignment});
    }
  for (size_t i = 0; i < prepared.buffer_extents.size(); ++i)
    for (size_t j = i + 1; j < prepared.buffer_extents.size(); ++j)
      if (Overlaps(prepared.buffer_extents[i], prepared.buffer_extents[j]))
        return fail("GDN span recipe requires disjoint live argument spans");
  if (!AllocateInputRoots(
          prepared.buffer_extents,
          max_input_bytes >= live_bytes ? max_input_bytes - live_bytes : 0,
          &prepared.roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  if (!SetInputBytes(invocation.arguments[0].value.buffer, controls.epoch,
                     &prepared.roots, error) ||
      !SetInputBytes(invocation.arguments[1].value.buffer, controls.rows,
                     &prepared.roots, error) ||
      !SetFiniteF32Input(invocation.arguments[qkv_argument].value.buffer,
                         active * qkv_width, plan.workload.seed,
                         &prepared.roots, error))
    return PreparationStatus::kFailure;
  for (const auto &weight : weights)
    if (!LoadWeight(plan, invocation, weight, &prepared, error))
      return PreparationStatus::kFailure;
  auto live = invocation.arguments[state_argument].value.buffer;
  if (live.byte_offset > UINT64_MAX - layer_origin)
    return fail("GDN layer state offset overflows its native root");
  live.byte_offset += layer_origin;
  live.byte_length = live_bytes;
  const bool initialized =
      plan.workload.history_tokens
          ? SetFiniteF32Input(live, live_elements, plan.workload.seed + 17,
                              &prepared.roots, error)
          : SetInputBytes(live, std::vector<uint8_t>(live_bytes, 0),
                          &prepared.roots, error);
  if (!initialized ||
      !CaptureStateResets({{live, live_bytes, 4}}, prepared.roots,
                          &prepared.resets, error))
    return PreparationStatus::kFailure;
  if (recurrent) {
    const auto &qkv = invocation.arguments[3].value.buffer;
    const auto *qkv_root = FindInputRoot(prepared.roots, qkv);
    const uint64_t begin = qkv.byte_offset - qkv_root->original_byte_offset;
    std::vector<uint8_t> inverse;
    inverse.reserve(active * 2 * ssm.key_heads * 4);
    for (uint64_t row = 0; row < active; ++row)
      for (uint64_t head = 0; head < ssm.key_heads; ++head)
        for (uint64_t key = 0; key < 2; ++key) {
          double square_sum = 0;
          for (uint64_t lane = 0; lane < ssm.state_width; ++lane) {
            const auto element = row * qkv_width +
                                 key * ssm.key_heads * ssm.state_width +
                                 head * ssm.state_width + lane;
            const float value = Float(*qkv_root, begin + element * 4);
            square_sum += static_cast<double>(value) * value;
          }
          AppendFloat(static_cast<float>(
                          1.0 / std::max(static_cast<double>(norm_epsilon),
                                         std::sqrt(square_sum))),
                      &inverse);
        }
    if (!SetInputBytes(invocation.arguments[4].value.buffer, inverse,
                       &prepared.roots, error) ||
        !SetFiniteF32Input(invocation.arguments[5].value.buffer,
                           active * ssm.value_heads, plan.workload.seed + 31,
                           &prepared.roots, error) ||
        !SetFiniteF32Input(invocation.arguments[6].value.buffer,
                           active * ssm.value_heads, plan.workload.seed + 47,
                           &prepared.roots, error))
      return PreparationStatus::kFailure;
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
