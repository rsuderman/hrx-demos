// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>

namespace hrx::loomify_gguf::benchmark {
namespace {
PreparationStatus Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return PreparationStatus::kFailure;
}
PreparationStatus Unsupported(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return PreparationStatus::kUnsupported;
}
bool Product(uint64_t a, uint64_t b, uint64_t *out) {
  if (b && a > UINT64_MAX / b)
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
bool SameSpan(const BufferExtent &a, const BufferExtent &b) {
  return a.binding.role == b.binding.role &&
         a.binding.root_index == b.binding.root_index &&
         a.binding.byte_offset == b.binding.byte_offset &&
         a.byte_length == b.byte_length;
}
struct CountInput {
  size_t argument;
  uint64_t origin;
  std::vector<uint32_t> words;
};
struct Input {
  BufferExtent span;
  bool exact_output_alias = true;
};
struct Layout {
  uint64_t capacity = 0;
  uint64_t width = 0;
  uint64_t read_begin = 0;
  size_t output = 0;
  std::vector<size_t> activations;
  std::vector<CountInput> counts;
  std::vector<uint64_t> workloads;
  std::vector<std::string> types;
  std::optional<size_t> weight;
  std::string recipe;
  bool clear = false;
};
bool Signature(const Invocation &invocation,
               const std::vector<std::string> &types) {
  if (invocation.arguments.size() != types.size() ||
      invocation.command.argument_count != types.size())
    return false;
  for (size_t i = 0; i < types.size(); ++i) {
    const auto expected = types[i] == "buffer" ? LOOMC_CMD_ARGUMENT_KIND_BUFFER
                          : types[i] == "offset"
                              ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                              : LOOMC_CMD_ARGUMENT_KIND_INDEX;
    if (invocation.arguments[i].kind != expected)
      return false;
  }
  return true;
}
std::vector<uint8_t> Bytes(const std::vector<uint32_t> &words) {
  std::vector<uint8_t> bytes;
  for (uint32_t value : words)
    for (unsigned i = 0; i < 4; ++i)
      bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
  return bytes;
}
bool FiniteWeight(const PreparedRoot &root, uint64_t offset, uint64_t count) {
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t bits = 0;
    for (unsigned lane = 0; lane < 4; ++lane)
      bits |= uint32_t(root.bytes[offset + i * 4 + lane]) << (lane * 8);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value))
      return false;
  }
  return true;
}
// These row operations have unequal input/output widths or multiple outputs;
// keep their small ABI-specific setup separate from the equal-width recipes.
PreparationStatus PrepareF32RowsCase(const Plan &plan,
                                     const Invocation &invocation,
                                     uint64_t max_input_bytes,
                                     PreparedCase *out, std::string *error) {
  const bool prefix =
      invocation.kernel_symbol == "llm.projection.prepare_f32_prefix";
  const bool dense =
      invocation.kernel_symbol == "llm.projection.f32_rows_kernel";
  const bool split = invocation.kernel_symbol == "llm.decoder.split_qkv";
  if (!prefix && !dense && !split)
    return PreparationStatus::kUnsupported;
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
    return Unsupported(error, "F32 row recipe requires concrete direct counts");
  PreparedCase prepared;
  prepared.source_argument_types = {"index", "index", "buffer", "buffer",
                                    "buffer"};
  if (dense)
    prepared.source_argument_types.push_back("buffer");
  if (split) {
    prepared.source_argument_types.push_back("buffer");
    prepared.source_argument_types.push_back("buffer");
  }
  if (!Signature(invocation, prepared.source_argument_types))
    return Fail(error, "F32 row native ABI differs");
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  const uint64_t rows = grid[0];
  const uint64_t first = invocation.arguments[0].value.scalar_bits;
  const uint64_t second = invocation.arguments[1].value.scalar_bits;
  const uint64_t capacity = prefix ? first : rows;
  if (!rows || rows > 512 || grid[1] != 1 || grid[2] != 1 || !capacity ||
      capacity > 512 || rows > capacity || !plan.workload.active_tokens ||
      plan.workload.active_tokens > rows ||
      (dense && (!first || first > 32768 || !second || second > 32768)) ||
      (prefix && (second < 256 || second > 32768 || second % 256)) ||
      (split && (first < 256 || first > 32768 || first % 256 || second < 256 ||
                 second > 32768 || second % 256)))
    return Fail(error, "F32 row geometry/activity violates source contract");
  const uint64_t input_width = prefix  ? second
                               : dense ? first
                                       : first + 2 * second;
  const auto buffer = [&](size_t i) {
    return invocation.arguments[i].value.buffer;
  };
  prepared.setup_recipe_id = "f32_rows/" + invocation.kernel_symbol + "/v1";
  prepared.source_workloads = {rows};
  if (prefix)
    prepared.source_workloads.push_back(second);
  if (split) {
    prepared.source_workloads.push_back(first);
    prepared.source_workloads.push_back(second);
  }
  uint64_t input_elements, input_bytes;
  if (!Product(capacity, input_width, &input_elements) ||
      !Product(input_elements, 4, &input_bytes))
    return Fail(error, "F32 row input footprint overflows");
  prepared.buffer_extents = {{buffer(2), 12, 4}, {buffer(3), input_bytes, 4}};
  const model::TensorRecord *tensor = nullptr;
  const Parameter *parameter = nullptr;
  if (dense) {
    const uint64_t weight_bytes = first * second * 4;
    parameter = FindBenchmarkParameter(plan, invocation, buffer(4),
                                       weight_bytes, error);
    tensor = parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
    if (!tensor || buffer(4).role != LOOMC_CMD_BUFFER_ROLE_FIXED ||
        buffer(4).byte_offset != parameter->byte_offset ||
        parameter->byte_length != weight_bytes || tensor->storage_type != 0 ||
        tensor->rank != 2 || tensor->normalized_extents[0] != int64_t(first) ||
        tensor->normalized_extents[1] != int64_t(second) ||
        tensor->normalized_extents[2] != 1 ||
        tensor->normalized_extents[3] != 1 || tensor->byte_size != weight_bytes)
      return Fail(error, "dense F32 requires the exact selected GGUF matrix");
    prepared.buffer_extents.push_back(
        {buffer(4), weight_bytes,
         std::max<uint64_t>(4, parameter->minimum_alignment)});
    prepared.representative_weight_spans = {prepared.buffer_extents.back()};
    prepared.weight_sources = {
        {tensor->name, tensor->absolute_offset, tensor->byte_size}};
  }
  const size_t first_output = dense ? 5 : 4;
  for (size_t i = first_output; i < invocation.arguments.size(); ++i) {
    const uint64_t width = prefix   ? second
                           : dense  ? second
                           : i == 4 ? first
                                    : second;
    prepared.buffer_extents.push_back({buffer(i), capacity * width * 4, 4});
  }
  // All spans remain in native buffer-operand order. The dense kernel
  // explicitly assumes noalias; split output overlaps are not a proven safe
  // fixture layout. Prefix exact in-place copies are safe and reset only their
  // active input rows.
  bool inplace = false;
  for (size_t i = 0; i < prepared.buffer_extents.size(); ++i) {
    const auto &extent = prepared.buffer_extents[i];
    const auto &binding = extent.binding;
    if (binding.byte_offset % 4 ||
        binding.byte_offset > UINT64_MAX - extent.byte_length ||
        (binding.byte_length != UINT64_MAX &&
         extent.byte_length > binding.byte_length) ||
        (binding.role != LOOMC_CMD_BUFFER_ROLE_FIXED &&
         binding.role != LOOMC_CMD_BUFFER_ROLE_REBINDABLE))
      return Fail(error, "F32 row binding is invalid or unaligned");
    for (size_t j = i + 1; j < prepared.buffer_extents.size(); ++j) {
      const auto &a = prepared.buffer_extents[i];
      const auto &b = prepared.buffer_extents[j];
      if (!Overlaps(a, b))
        continue;
      if (prefix && i == 1 && j == 2 && SameSpan(a, b)) {
        inplace = true;
        continue;
      }
      return Unsupported(error, "F32 row recipe requires disjoint spans except "
                                "an exact prefix input/output alias");
    }
  }
  bool budget_exceeded = false;
  if (!AllocateInputRoots(prepared.buffer_extents, max_input_bytes,
                          &prepared.roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  if (!SetFiniteF32Input(buffer(3), input_elements, plan.workload.seed,
                         &prepared.roots, error))
    return PreparationStatus::kFailure;
  for (size_t i = first_output; i < invocation.arguments.size(); ++i) {
    if (inplace)
      continue;
    const auto &extent = prepared.buffer_extents[i - 2];
    if (!SetFiniteF32Input(buffer(i), extent.byte_length / 4,
                           plan.workload.seed + 1, &prepared.roots, error))
      return PreparationStatus::kFailure;
  }
  if (!SetInputBytes(
          buffer(2),
          Bytes({static_cast<uint32_t>(plan.workload.active_tokens), 0, 0}),
          &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (tensor) {
    const auto binding = buffer(4);
    if (!SetTensorInput(binding, plan.source_asset, tensor->absolute_offset,
                        tensor->byte_size, &prepared.roots, error))
      return PreparationStatus::kFailure;
    const auto *root = FindInputRoot(prepared.roots, binding);
    if (!FiniteWeight(*root, binding.byte_offset - root->original_byte_offset,
                      tensor->byte_size / 4))
      return Fail(error, "GGUF dense weight has nonfinite F32 values");
  }
  if (inplace) {
    BufferExtent active = prepared.buffer_extents[1];
    active.byte_length = plan.workload.active_tokens * input_width * 4;
    active.binding.byte_length = active.byte_length;
    if (!CaptureStateResets({active}, prepared.roots, &prepared.resets, error,
                            max_input_bytes, &budget_exceeded))
      return budget_exceeded ? PreparationStatus::kUnsupported
                             : PreparationStatus::kFailure;
    prepared.measurement_subject = "reset_plus_kernel";
    prepared.reset_policy = "copy_seed_each_invocation";
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace

PreparationStatus PrepareElementwiseCase(const Plan &plan,
                                         const Invocation &invocation,
                                         uint64_t max_input_bytes,
                                         PreparedCase *out,
                                         std::string *error) {
  if (!out)
    return Fail(error, "prepared elementwise case output is required");
  const auto &name = invocation.kernel_symbol;
  if (name == "llm.projection.prepare_f32_prefix" ||
      name == "llm.projection.f32_rows_kernel" ||
      name == "llm.decoder.split_qkv")
    return PrepareF32RowsCase(plan, invocation, max_input_bytes, out, error);
  const bool residual = name == "qwen38_residual_add_f32_prefill" ||
                        name == "qwen38_residual_add_f32_decode" ||
                        name == "llm_ffn_residual_add_f32_decode" ||
                        name == "llm_ffn_residual_add_f32_prefill" ||
                        name == "llm_ffn_residual_add_f32_prefill_active" ||
                        name == "llm.decoder.residual" ||
                        name == "gemma3_residual" ||
                        name == "llama32_attention_residual";
  const bool gate = name == "llm_ffn_swiglu_f32_decode" ||
                    name == "llm_ffn_swiglu_f32_prefill" ||
                    name == "llm_ffn_swiglu_f32_prefill_active" ||
                    name == "llm.projection.active_swiglu" ||
                    name == "gemma3_geglu";
  const bool bias = name == "llm.decoder.add_bias_active";
  const bool clear = name == "qwen08_gdn_clear_raw" ||
                     name == "qwen38_attention_clear_attended";
  if (!residual && !gate && !bias && !clear)
    return PreparationStatus::kUnsupported;
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
    return Unsupported(error,
                       "elementwise recipe requires concrete direct counts");
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  if (!grid[0] || !grid[1] || grid[2] != 1)
    return Fail(error, "elementwise native grid is invalid");
  Layout layout;
  const auto scalar = [&](size_t i) {
    return invocation.arguments[i].value.scalar_bits;
  };
  // Each branch describes a maintained operation ABI, never a model-family
  // routing choice. The shared tail handles footprints, aliases and resets.
  if (name == "qwen38_residual_add_f32_prefill") {
    layout.types = {"index", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "residual prefill native ABI differs");
    layout.capacity = scalar(0);
    layout.width = plan.model.embedding_width;
    layout.activations = {1, 2};
    layout.output = 3;
    layout.workloads = {grid[1]};
    if (layout.width % 1024 || grid[0] != layout.width / 1024)
      return Fail(error, "residual prefill width differs from selected grid");
  } else if (name == "qwen38_residual_add_f32_decode") {
    layout.types = {"buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types) || grid[1] != 1)
      return Fail(error, "residual decode native ABI/grid differs");
    layout.capacity = 1;
    layout.width = uint64_t(grid[0]) * 1024;
    layout.activations = {0, 1};
    layout.output = 2;
  } else if (name == "llm_ffn_residual_add_f32_decode" ||
             name == "llm_ffn_residual_add_f32_prefill") {
    const bool prefill = name == "llm_ffn_residual_add_f32_prefill";
    layout.types =
        prefill
            ? std::vector<std::string>{"index", "index", "buffer", "buffer",
                                       "buffer"}
            : std::vector<std::string>{"index", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "FFN residual native ABI differs");
    // The ceiling grid includes tail lanes. The launch ABI carries the true
    // row stride; grid.y is the launched row count, not the buffer capacity.
    layout.width = scalar(0);
    layout.capacity = prefill ? scalar(1) : 1;
    if (layout.width < 1024 || layout.width > 32768 || layout.width % 256 ||
        grid[0] != (layout.width + 1023) / 1024 || (!prefill && grid[1] != 1))
      return Fail(error, "FFN residual width/rows differ from source grid");
    const size_t first_buffer = prefill ? 2 : 1;
    layout.activations = {first_buffer, first_buffer + 1};
    layout.output = first_buffer + 2;
    layout.workloads = prefill ? std::vector<uint64_t>{grid[1], layout.width}
                               : std::vector<uint64_t>{layout.width};
  } else if (name == "llm.decoder.residual" || bias) {
    layout.types = {"index", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "count-aware elementwise native ABI differs");
    layout.capacity = grid[1];
    layout.width = scalar(0);
    layout.counts = {
        {1, 0, {static_cast<uint32_t>(plan.workload.active_tokens), 0, 0}}};
    layout.activations =
        bias ? std::vector<size_t>{3} : std::vector<size_t>{2, 3};
    layout.output = 3;
    layout.workloads = {grid[1], layout.width};
    if (bias)
      layout.weight = 2;
    const uint64_t tile = bias ? 256 : 1024;
    if ((bias && layout.width % 256) ||
        grid[0] != (layout.width + tile - 1) / tile)
      return Fail(error, "count-aware elementwise width differs from grid");
  } else if (name == "gemma3_residual") {
    layout.types = {"buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "residual native ABI differs");
    layout.capacity = grid[1];
    layout.width = plan.model.embedding_width;
    layout.counts = {
        {0, 0, {static_cast<uint32_t>(plan.workload.active_tokens)}}};
    layout.activations = {1, 2};
    layout.output = 2;
    layout.workloads = {grid[1]};
    if (grid[0] != (layout.width + 1023) / 1024)
      return Fail(error, "residual width differs from selected grid");
  } else if (name == "llama32_attention_residual") {
    layout.types = {"index", "offset", "index", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "attention residual native ABI differs");
    layout.capacity = grid[1];
    layout.width = plan.model.embedding_width;
    const uint64_t begin = scalar(2);
    layout.read_begin = begin;
    if (begin >= layout.capacity ||
        plan.workload.active_tokens > layout.capacity - begin ||
        plan.workload.history_tokens > INT32_MAX ||
        plan.workload.history_tokens > plan.workload.context_capacity ||
        plan.workload.active_tokens >
            plan.workload.context_capacity - plan.workload.history_tokens ||
        (scalar(0) == 1 && plan.workload.active_tokens != 1))
      return Unsupported(
          error, "attention residual activity does not fit selected span");
    const uint32_t history =
        static_cast<uint32_t>(plan.workload.history_tokens);
    layout.counts = {{3, scalar(1),
                      scalar(0) == 1 ? std::vector<uint32_t>{history, 0, 0}
                                     : std::vector<uint32_t>{
                                           static_cast<uint32_t>(
                                               plan.workload.active_tokens),
                                           history, 0}}};
    layout.activations = {4, 5};
    layout.output = 5;
    layout.workloads = {grid[1]};
    if (layout.width % 1024 || grid[0] != layout.width / 1024)
      return Fail(error, "attention residual width differs from selected grid");
  } else if (name == "llm_ffn_residual_add_f32_prefill_active") {
    layout.types = {"index",  "index",  "index",  "offset",
                    "buffer", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "active residual native ABI differs");
    layout.width = scalar(0);
    layout.capacity = scalar(1);
    if (layout.width < 1024 || layout.width > 32768 || layout.width % 256 ||
        grid[0] != (layout.width + 1023) / 1024)
      return Fail(error, "active residual width differs from source grid");
    layout.read_begin = scalar(2);
    if (scalar(2) >= layout.capacity ||
        plan.workload.active_tokens > layout.capacity - scalar(2))
      return Unsupported(error,
                         "active residual count exceeds selected row span");
    layout.counts = {
        {4, scalar(3), {static_cast<uint32_t>(plan.workload.active_tokens)}}};
    layout.activations = {5, 6};
    layout.output = 7;
    layout.workloads = {grid[1], layout.width};
  } else if (name == "llm_ffn_swiglu_f32_decode") {
    layout.types = {"index", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types) || grid[1] != 1)
      return Fail(error, "SwiGLU decode native ABI/grid differs");
    layout.capacity = 1;
    layout.width = scalar(0);
    layout.activations = {1, 2};
    layout.output = 3;
    layout.workloads = {layout.width};
  } else if (name == "llm_ffn_swiglu_f32_prefill") {
    layout.types = {"index", "index", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "SwiGLU prefill native ABI differs");
    layout.width = scalar(0);
    layout.capacity = scalar(1);
    layout.activations = {2, 3};
    layout.output = 4;
    layout.workloads = {grid[1], layout.width};
  } else if (name == "llm_ffn_swiglu_f32_prefill_active") {
    layout.types = {"index",  "index",  "index",  "offset",
                    "buffer", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "active SwiGLU native ABI differs");
    layout.width = scalar(0);
    layout.capacity = scalar(1);
    layout.read_begin = scalar(2);
    if (scalar(2) >= layout.capacity ||
        plan.workload.active_tokens > layout.capacity - scalar(2))
      return Unsupported(error,
                         "active SwiGLU count exceeds selected row span");
    layout.counts = {
        {4, scalar(3), {static_cast<uint32_t>(plan.workload.active_tokens)}}};
    layout.activations = {5, 6};
    layout.output = 7;
    layout.workloads = {grid[1], layout.width};
  } else if (name == "llm.projection.active_swiglu") {
    layout.types = {"index", "index", "buffer", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "matrix SwiGLU native ABI differs");
    layout.capacity = scalar(0);
    layout.width = scalar(1);
    if (layout.capacity > 16)
      return Fail(error, "matrix SwiGLU capacity exceeds source contract");
    layout.counts = {
        {2, 0, {static_cast<uint32_t>(plan.workload.active_tokens), 0, 0}}};
    layout.activations = {3, 4};
    layout.output = 5;
    layout.workloads = {grid[1], layout.width};
  } else if (name == "gemma3_geglu") {
    layout.types = {"index", "buffer", "buffer", "buffer", "buffer"};
    if (!Signature(invocation, layout.types))
      return Fail(error, "GeGLU native ABI differs");
    layout.capacity = grid[1];
    layout.width = scalar(0);
    layout.counts = {
        {1, 0, {static_cast<uint32_t>(plan.workload.active_tokens)}}};
    layout.activations = {2, 3};
    layout.output = 4;
    layout.workloads = {grid[1], layout.width};
  } else {
    layout.types = {"index", "buffer"};
    if (!Signature(invocation, layout.types) || grid[1] != 1)
      return Fail(error, "clear native ABI/grid differs");
    layout.capacity = scalar(0);
    layout.output = 1;
    layout.clear = true;
    if (name == "qwen08_gdn_clear_raw") {
      if (!plan.model.ssm)
        return Fail(error, "GDN clear requires configured inner width");
      layout.width = plan.model.ssm->inner_width;
      layout.workloads = {plan.workload.token_capacity};
      uint64_t elements;
      if (!Product(plan.workload.token_capacity, layout.width, &elements) ||
          !elements || elements > UINT64_MAX - 255 ||
          grid[0] != (elements + 255) / 256)
        return Fail(error, "GDN clear logical grid differs");
    } else {
      if (plan.model.layers.empty() ||
          !Product(plan.model.layers.front().query_heads,
                   plan.model.key_head_width, &layout.width))
        return Fail(error, "attention clear requires configured query width");
      layout.workloads = {grid[0]};
      if (grid[0] > layout.capacity)
        return Fail(error, "attention clear rows exceed runtime capacity");
    }
  }
  if (!layout.capacity || layout.capacity > 512 || !layout.width ||
      (!clear && (grid[1] > layout.capacity || !plan.workload.active_tokens ||
                  plan.workload.active_tokens > layout.capacity)) ||
      (residual &&
       (layout.width < 1024 || layout.width > 32768 || layout.width % 4)) ||
      ((gate || bias) &&
       (layout.width < 256 || layout.width > 32768 || layout.width % 256)) ||
      (gate && grid[0] != layout.width / 256))
    return Fail(error,
                "elementwise capacity/width/activity violates source contract");
  uint64_t elements, activation_bytes;
  if (!Product(layout.capacity, layout.width, &elements) ||
      !Product(elements, 4, &activation_bytes))
    return Fail(error, "elementwise footprint overflows");
  PreparedCase prepared;
  prepared.setup_recipe_id = "elementwise/" + name + "/v1";
  prepared.source_workloads = layout.workloads;
  prepared.source_argument_types = layout.types;
  const auto argument = [&](size_t i) {
    return invocation.arguments[i].value.buffer;
  };
  std::vector<std::optional<BufferExtent>> extents(invocation.arguments.size());
  std::vector<Input> reads;
  for (size_t index : layout.activations) {
    BufferExtent extent{argument(index), activation_bytes, 4};
    extents[index] = extent;
    reads.push_back({extent, true});
  }
  const BufferExtent output{argument(layout.output), activation_bytes, 4};
  extents[layout.output] = output;
  for (const auto &count : layout.counts) {
    if (count.origin % 4 || count.origin > UINT64_MAX - count.words.size() * 4)
      return Fail(error, "elementwise count origin overflows");
    const auto binding = argument(count.argument);
    extents[count.argument] =
        BufferExtent{binding, count.origin + count.words.size() * 4, 4};
    auto actual = binding;
    if (actual.byte_offset > UINT64_MAX - count.origin)
      return Fail(error, "elementwise count address overflows");
    actual.byte_offset += count.origin;
    actual.byte_length = count.words.size() * 4;
    reads.push_back({{actual, actual.byte_length, 4}, false});
  }
  const model::TensorRecord *tensor = nullptr;
  if (layout.weight) {
    const auto binding = argument(*layout.weight);
    const uint64_t bytes = layout.width * 4;
    const auto *parameter =
        FindBenchmarkParameter(plan, invocation, binding, bytes, error);
    tensor = parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
    if (!tensor || binding.role != LOOMC_CMD_BUFFER_ROLE_FIXED ||
        binding.byte_offset != parameter->byte_offset ||
        parameter->byte_length != bytes || tensor->storage_type != 0 ||
        tensor->rank != 1 || tensor->byte_size != bytes ||
        tensor->normalized_extents[0] != static_cast<int64_t>(layout.width))
      return Fail(error, "bias requires the exact selected GGUF F32 vector");
    BufferExtent extent{binding, bytes,
                        std::max<uint64_t>(4, parameter->minimum_alignment)};
    extents[*layout.weight] = extent;
    prepared.representative_weight_spans = {extent};
    reads.push_back({extent, false});
    prepared.weight_sources = {
        {tensor->name, tensor->absolute_offset, tensor->byte_size}};
  }
  // Emission binds buffer operands in the selected source ABI order, which is
  // independent of their initialization/read/write roles.
  for (size_t i = 0; i < invocation.arguments.size(); ++i)
    if (invocation.arguments[i].kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
      if (!extents[i])
        return Fail(error, "elementwise buffer has no declared footprint");
      prepared.buffer_extents.push_back(*extents[i]);
    }
  std::vector<BufferExtent> mutations;
  for (const auto &input : reads)
    if (Overlaps(input.span, output)) {
      if (!input.exact_output_alias || !SameSpan(input.span, output))
        return Unsupported(
            error,
            "elementwise recipe does not support partial output/read overlap");
      // Restore only rows this invocation can read and overwrite. Padded rows
      // and unrelated bytes of a shared root are not state-reset traffic.
      const uint64_t read_rows =
          layout.read_begin < grid[1]
              ? std::min<uint64_t>(layout.counts.empty()
                                       ? grid[1]
                                       : plan.workload.active_tokens,
                                   grid[1] - layout.read_begin)
              : 0;
      if (read_rows) {
        BufferExtent live = input.span;
        live.binding.byte_offset += layout.read_begin * layout.width * 4;
        live.byte_length = read_rows * layout.width * 4;
        live.binding.byte_length = live.byte_length;
        mutations.push_back(live);
      }
    }
  for (const auto &extent : prepared.buffer_extents)
    if (extent.binding.byte_offset % 4)
      return Fail(error, "elementwise F32/control binding is unaligned");
  bool budget_exceeded = false;
  if (!AllocateInputRoots(prepared.buffer_extents, max_input_bytes,
                          &prepared.roots, error, &budget_exceeded))
    return budget_exceeded ? PreparationStatus::kUnsupported
                           : PreparationStatus::kFailure;
  for (size_t index : layout.activations) {
    const auto binding = argument(index);
    const auto *root = FindInputRoot(prepared.roots, binding);
    const uint64_t shift =
        (binding.byte_offset - root->original_byte_offset) / 4;
    if (!SetFiniteF32Input(binding, elements,
                           (plan.workload.seed % 61 + shift % 61) % 61,
                           &prepared.roots, error))
      return PreparationStatus::kFailure;
  }
  if (layout.clear &&
      !SetFiniteF32Input(output.binding, elements, plan.workload.seed,
                         &prepared.roots, error))
    return PreparationStatus::kFailure;
  for (const auto &count : layout.counts) {
    auto binding = argument(count.argument);
    binding.byte_offset += count.origin;
    binding.byte_length = count.words.size() * 4;
    if (!SetInputBytes(binding, Bytes(count.words), &prepared.roots, error))
      return PreparationStatus::kFailure;
  }
  if (tensor) {
    const auto binding = argument(*layout.weight);
    if (!SetTensorInput(binding, plan.source_asset, tensor->absolute_offset,
                        tensor->byte_size, &prepared.roots, error))
      return PreparationStatus::kFailure;
    const auto *root = FindInputRoot(prepared.roots, binding);
    if (!FiniteWeight(*root, binding.byte_offset - root->original_byte_offset,
                      layout.width))
      return Fail(error,
                  "GGUF bias has nonfinite values; fixture is not finite");
  }
  if (!mutations.empty()) {
    if (!CaptureStateResets(mutations, prepared.roots, &prepared.resets, error,
                            max_input_bytes, &budget_exceeded))
      return budget_exceeded ? PreparationStatus::kUnsupported
                             : PreparationStatus::kFailure;
    prepared.measurement_subject = "reset_plus_kernel";
    prepared.reset_policy = "copy_seed_each_invocation";
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
