// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace hrx::loomify_gguf::benchmark {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
bool Add(uint64_t a, uint64_t b, uint64_t *out) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    return false;
  *out = a + b;
  return true;
}
bool Overlaps(const loomc_cmd_buffer_ref_t &a, uint64_t a_length,
              const loomc_cmd_buffer_ref_t &b, uint64_t b_length) {
  return a.role == b.role && a.root_index == b.root_index &&
         (a.byte_offset <= b.byte_offset
              ? b.byte_offset - a.byte_offset < a_length
              : a.byte_offset - b.byte_offset < b_length);
}
std::vector<uint8_t> Words(std::initializer_list<uint32_t> words) {
  std::vector<uint8_t> result;
  for (uint32_t word : words)
    for (unsigned i = 0; i < 4; ++i)
      result.push_back(static_cast<uint8_t>(word >> (i * 8)));
  return result;
}
bool Finite(const PreparedRoot &root, uint64_t offset, uint64_t byte_length) {
  for (uint64_t i = 0; i < byte_length; i += 4) {
    uint32_t bits = 0;
    for (unsigned lane = 0; lane < 4; ++lane)
      bits |= static_cast<uint32_t>(root.bytes[offset + i + lane])
              << (lane * 8);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value))
      return false;
  }
  return true;
}
} // namespace

PreparationStatus PrepareStatefulCase(const Plan &plan,
                                      const Invocation &invocation,
                                      uint64_t max_input_bytes,
                                      PreparedCase *out, std::string *error) {
  const auto fail = [&](const std::string &message) {
    Fail(error, message);
    return PreparationStatus::kFailure;
  };
  if (!out)
    return fail("prepared stateful case output is required");
  if (invocation.kernel_symbol != "llm.convolution.shortconv7") {
    Fail(error,
         "unsupported stateful operation recipe: " + invocation.kernel_symbol);
    return PreparationStatus::kUnsupported;
  }
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
      invocation.program_ordinal >= plan.programs.size() ||
      invocation.arguments.size() != 10 ||
      invocation.arguments[0].kind != LOOMC_CMD_ARGUMENT_KIND_INDEX ||
      invocation.arguments[1].kind != LOOMC_CMD_ARGUMENT_KIND_INDEX ||
      invocation.arguments[2].kind != LOOMC_CMD_ARGUMENT_KIND_OFFSET ||
      invocation.arguments[3].kind != LOOMC_CMD_ARGUMENT_KIND_OFFSET)
    return fail("shortconv native ABI differs from the operation contract");
  for (size_t i = 4; i < 10; ++i)
    if (invocation.arguments[i].kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER)
      return fail("shortconv requires six buffer arguments");
  const auto &conv = plan.model.short_convolution;
  const uint64_t rows = invocation.arguments[0].value.scalar_bits;
  const uint64_t width = invocation.arguments[1].value.scalar_bits;
  const uint64_t history_origin = invocation.arguments[2].value.scalar_bits;
  const uint64_t control_origin = invocation.arguments[3].value.scalar_bits;
  if (!conv || conv->kernel != 7 || conv->history_rows != 6 ||
      conv->width != width || plan.model.embedding_width != width || !rows ||
      rows > 512 || width < 256 || width > 32768 || width % 256 ||
      rows != plan.workload.token_capacity || !plan.workload.active_tokens ||
      plan.workload.active_tokens > rows ||
      plan.workload.history_tokens > INT32_MAX ||
      plan.workload.history_tokens > plan.workload.context_capacity ||
      plan.workload.active_tokens >
          plan.workload.context_capacity - plan.workload.history_tokens ||
      history_origin > INT64_MAX || control_origin > INT64_MAX ||
      history_origin % 4 || control_origin % 4)
    return fail("shortconv requires matching geometry and valid history");
  const uint64_t history_bytes = 6 * width * 4;
  const uint64_t weight_bytes = width * 7 * 4;
  uint64_t state_bytes, epoch_bytes;
  if (!Add(history_origin, history_bytes, &state_bytes) ||
      !Add(control_origin, 20, &epoch_bytes))
    return fail("shortconv state or control extent overflows");
  std::vector<BufferExtent> extents = {
      {invocation.arguments[4].value.buffer, 12, 4},
      {invocation.arguments[5].value.buffer, epoch_bytes, 4},
      {invocation.arguments[6].value.buffer, rows * 3 * width * 4, 4},
      {invocation.arguments[7].value.buffer, weight_bytes, 4},
      {invocation.arguments[8].value.buffer, state_bytes, 4},
      {invocation.arguments[9].value.buffer, rows * width * 4, 4}};
  for (size_t i = 0; i < extents.size(); ++i)
    for (size_t j = i + 1; j < extents.size(); ++j)
      if (Overlaps(extents[i].binding, extents[i].byte_length,
                   extents[j].binding, extents[j].byte_length))
        return fail("shortconv requires disjoint argument spans");
  const auto &weight = extents[3].binding;
  const auto *parameter =
      FindBenchmarkParameter(plan, invocation, weight, weight_bytes, error);
  if (!parameter)
    return PreparationStatus::kFailure;
  const auto *tensor = FindBenchmarkTensor(plan, parameter->key);
  if (parameter->byte_offset != weight.byte_offset ||
      parameter->byte_length != weight_bytes || !tensor ||
      tensor->storage_type != 0 || tensor->byte_size != weight_bytes ||
      tensor->rank != 2 || tensor->normalized_extents[0] != 7 ||
      tensor->normalized_extents[1] != static_cast<int64_t>(width) ||
      tensor->normalized_extents[2] != 1 || tensor->normalized_extents[3] != 1)
    return fail("shortconv requires exact selected GGUF F32 tap weights");
  extents[3].minimum_alignment =
      std::max<uint64_t>(4, parameter->minimum_alignment);
  auto history = extents[4].binding;
  auto epoch = extents[1].binding;
  if (!Add(history.byte_offset, history_origin, &history.byte_offset) ||
      !Add(epoch.byte_offset, control_origin, &epoch.byte_offset))
    return fail("shortconv scalar origin overflows its root");
  history.byte_length = history_bytes;
  epoch.byte_length = 20;
  PreparedCase prepared;
  prepared.source_workloads = {rows, width};
  prepared.source_argument_types = {"index",  "index",  "offset", "offset",
                                    "buffer", "buffer", "buffer", "buffer",
                                    "buffer", "buffer"};
  prepared.setup_recipe_id = "shortconv7_f32/v1";
  prepared.measurement_subject = "reset_plus_kernel";
  prepared.reset_policy = "copy_seed_each_invocation";
  prepared.weight_sources = {
      {tensor->name, tensor->absolute_offset, weight_bytes}};
  prepared.buffer_extents = extents;
  prepared.representative_weight_spans = {extents[3]};
  bool budget_exceeded = false;
  if (!AllocateInputRoots(extents,
                          max_input_bytes >= history_bytes
                              ? max_input_bytes - history_bytes
                              : 0,
                          &prepared.roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  if (!SetInputBytes(
          extents[0].binding,
          Words({static_cast<uint32_t>(plan.workload.active_tokens), 0, 0}),
          &prepared.roots, error) ||
      !SetInputBytes(
          epoch,
          Words({static_cast<uint32_t>(plan.workload.active_tokens),
                 static_cast<uint32_t>(plan.workload.history_tokens), 0, 0, 0}),
          &prepared.roots, error) ||
      !SetFiniteF32Input(extents[2].binding, rows * 3 * width,
                         plan.workload.seed, &prepared.roots, error) ||
      !SetTensorInput(weight, plan.source_asset, tensor->absolute_offset,
                      weight_bytes, &prepared.roots, error) ||
      !SetInputBytes(extents[5].binding,
                     std::vector<uint8_t>(rows * width * 4, 0xA5),
                     &prepared.roots, error))
    return PreparationStatus::kFailure;
  // Position zero supplies six logical zeros in the kernel without reading
  // old state; continued positions consume the initialized history instead.
  if (!(plan.workload.history_tokens
            ? SetFiniteF32Input(history, 6 * width, plan.workload.seed + 17,
                                &prepared.roots, error)
            : SetInputBytes(history, std::vector<uint8_t>(history_bytes, 0),
                            &prepared.roots, error)))
    return PreparationStatus::kFailure;
  const auto *weight_root = FindInputRoot(prepared.roots, weight);
  if (!Finite(*weight_root,
              weight.byte_offset - weight_root->original_byte_offset,
              weight_bytes))
    return fail("shortconv GGUF taps contain nonfinite values");
  if (!CaptureStateResets({{history, history_bytes, 4}}, prepared.roots,
                          &prepared.resets, error))
    return PreparationStatus::kFailure;
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
