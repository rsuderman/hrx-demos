// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace hrx::loomify_gguf::benchmark {
namespace {
using Kind = loomc_cmd_argument_kind_t;
constexpr size_t kAbsent = std::numeric_limits<size_t>::max();
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
PreparationStatus Allocate(PreparedCase *prepared, uint64_t max_bytes,
                           std::string *error) {
  bool budget_exceeded = false;
  if (AllocateInputRoots(prepared->buffer_extents, max_bytes, &prepared->roots,
                         error, &budget_exceeded))
    return PreparationStatus::kReady;
  return budget_exceeded
             ? Unsupported(error,
                           "case input footprint exceeds its explicit budget")
             : PreparationStatus::kFailure;
}
bool Abi(const Invocation &invocation, const std::vector<std::string> &types) {
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
      invocation.command.argument_count != types.size() ||
      invocation.arguments.size() != types.size())
    return false;
  for (size_t i = 0; i < types.size(); ++i) {
    const Kind kind = types[i] == "index"    ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                      : types[i] == "offset" ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                      : types[i] == "f32"    ? LOOMC_CMD_ARGUMENT_KIND_B32
                                             : LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    if (invocation.arguments[i].kind != kind)
      return false;
  }
  return true;
}
uint64_t Scalar(const Invocation &invocation, size_t ordinal) {
  return invocation.arguments[ordinal].value.scalar_bits;
}
loomc_cmd_buffer_ref_t Buffer(const Invocation &invocation, size_t ordinal) {
  return invocation.arguments[ordinal].value.buffer;
}
bool Grid(const Invocation &invocation, uint64_t x, uint64_t y = 1,
          uint64_t z = 1) {
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  return grid[0] == x && grid[1] == y && grid[2] == z;
}
bool Add(uint64_t a, uint64_t b, uint64_t *out) {
  if (b > UINT64_MAX - a)
    return false;
  *out = a + b;
  return true;
}
bool Overlaps(const BufferExtent &a, const BufferExtent &b) {
  return a.binding.role == b.binding.role &&
         a.binding.root_index == b.binding.root_index &&
         (a.binding.byte_offset <= b.binding.byte_offset
              ? b.binding.byte_offset - a.binding.byte_offset < a.byte_length
              : a.binding.byte_offset - b.binding.byte_offset < b.byte_length);
}
void Word(std::vector<uint8_t> &bytes, size_t offset, uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}
std::vector<uint8_t> Count(uint64_t active) {
  std::vector<uint8_t> bytes(12);
  Word(bytes, 0, static_cast<uint32_t>(active));
  return bytes;
}
bool SetAt(const loomc_cmd_buffer_ref_t &binding, uint64_t offset,
           const std::vector<uint8_t> &bytes, std::vector<PreparedRoot> *roots,
           std::string *error) {
  auto shifted = binding;
  if (!Add(binding.byte_offset, offset, &shifted.byte_offset) ||
      (binding.byte_length != UINT64_MAX &&
       (offset > binding.byte_length ||
        bytes.size() > binding.byte_length - offset))) {
    if (error)
      *error = "basic control range exceeds its native buffer";
    return false;
  }
  shifted.byte_length = bytes.size();
  return SetInputBytes(shifted, bytes, roots, error);
}
bool PositiveEpsilon(const Invocation &invocation, size_t ordinal) {
  if (ordinal == kAbsent)
    return true;
  const uint64_t raw = Scalar(invocation, ordinal);
  if (raw > UINT32_MAX)
    return false;
  const uint32_t bits = static_cast<uint32_t>(raw);
  float epsilon;
  std::memcpy(&epsilon, &bits, sizeof(epsilon));
  return std::isfinite(epsilon) && epsilon > 0;
}
bool FiniteF32(const PreparedRoot &root, uint64_t begin, uint64_t length) {
  for (uint64_t i = 0; i < length; i += 4) {
    uint32_t bits = 0;
    for (unsigned lane = 0; lane < 4; ++lane)
      bits |= static_cast<uint32_t>(root.bytes[begin + i + lane]) << (lane * 8);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value))
      return false;
  }
  return true;
}
bool Gamma(const Plan &plan, const Invocation &invocation,
           const loomc_cmd_buffer_ref_t &binding, uint64_t width,
           PreparedCase *prepared, std::string *error) {
  const uint64_t length = width * 4;
  const auto *parameter =
      FindBenchmarkParameter(plan, invocation, binding, length, error);
  const auto *tensor =
      parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
  if (!tensor || binding.role != LOOMC_CMD_BUFFER_ROLE_FIXED ||
      tensor->storage_type != 0 || tensor->byte_size != length ||
      parameter->byte_length != length ||
      binding.byte_offset != parameter->byte_offset) {
    if (error)
      *error = "RMS gamma must be the exact GGUF F32 width tensor";
    return false;
  }
  prepared->weight_sources = {{tensor->name, tensor->absolute_offset, length}};
  return true;
}

// Each row describes a maintained operation ABI, not a model architecture.
struct RmsRecipe {
  std::vector<std::string> types;
  std::vector<uint64_t> workloads;
  uint64_t capacity = 0;
  uint64_t width = 0;
  uint64_t weight_width = 0;
  uint64_t output_rows = 0;
  uint64_t begin = 0;
  uint64_t count_origin = 0;
  uint64_t maximum_capacity = 512;
  size_t input = 0;
  size_t weight = 0;
  size_t output = 0;
  size_t epsilon = kAbsent;
  size_t control = kAbsent;
  size_t gate = kAbsent;
  enum Control { kNone, kCount, kOutputSpans } control_kind = kNone;
  std::string recipe_id;
};

PreparationStatus Rms(const Plan &plan, const Invocation &invocation,
                      uint64_t max_bytes, PreparedCase *out,
                      std::string *error) {
  const auto &name = invocation.kernel_symbol;
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  RmsRecipe r;
  r.width = r.weight_width = plan.model.embedding_width;
  r.recipe_id = "rms_f32/v1";
  // Validate types before reading any native union payload.
  if (name == "qwen38_rmsnorm_f32_prefill") {
    r.types = {"index", "f32", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types))
      return Fail(error,
                  "prefill RMS native ABI does not match its operation recipe");
    r.capacity = Scalar(invocation, 0);
    r.input = 2;
    r.weight = 3;
    r.output = 4;
    r.epsilon = 1;
    r.output_rows = r.capacity;
    r.workloads = {grid[1]};
    r.recipe_id = "rms_f32_prefill/v1";
    if (!r.capacity || r.capacity > 512 || !grid[1] || grid[1] > r.capacity ||
        !Grid(invocation, 1, grid[1]))
      return Fail(
          error,
          "prefill RMS grid/capacity violates the source operation contract");
  } else if (name == "qwen38_rmsnorm_f32_decode") {
    r.types = {"f32", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types) || !Grid(invocation, 1))
      return Fail(error, "decode RMS requires its one-row native ABI");
    r.capacity = r.output_rows = 1;
    r.epsilon = 0;
    r.input = 1;
    r.weight = 2;
    r.output = 3;
  } else if (name == "llama32_rmsnorm_f32") {
    r.types = {"index", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types))
      return Fail(error, "row RMS native ABI differs");
    r.capacity = r.output_rows = Scalar(invocation, 0);
    r.maximum_capacity = 2048;
    r.input = 1;
    r.weight = 2;
    r.output = 3;
    r.workloads = {grid[1]};
    if (!grid[1] || grid[1] > r.capacity || !Grid(invocation, 1, grid[1]))
      return Fail(error, "row RMS grid/capacity differs");
  } else if (name == "llm.decoder.rmsnorm_active") {
    r.types = {"index", "index", "f32", "buffer", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types))
      return Fail(error, "active RMS native ABI differs");
    r.capacity = r.output_rows = Scalar(invocation, 0);
    r.width = r.weight_width = Scalar(invocation, 1);
    r.epsilon = 2;
    r.control = 3;
    r.input = 4;
    r.weight = 5;
    r.output = 6;
    r.control_kind = RmsRecipe::kCount;
    r.workloads = {grid[1], r.width};
    if (!grid[1] || grid[1] > r.capacity || !Grid(invocation, 1, grid[1]))
      return Fail(error, "active RMS grid/capacity differs");
  } else if (name == "llama32_rmsnorm_active_f32" ||
             name == "gemma3_rmsnorm_active_f32" ||
             name == "llama32_output_norm_last_f32" ||
             name == "gemma3_output_norm_last_f32") {
    r.types = {"index",  "index",  "offset", "buffer",
               "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types))
      return Fail(error, "counted RMS native ABI differs");
    r.capacity = Scalar(invocation, 0);
    r.begin = Scalar(invocation, 1);
    r.count_origin = Scalar(invocation, 2);
    r.control = 3;
    r.input = 4;
    r.weight = 5;
    r.output = 6;
    r.control_kind = RmsRecipe::kCount;
    const bool last = name == "llama32_output_norm_last_f32" ||
                      name == "gemma3_output_norm_last_f32";
    r.maximum_capacity = last ? 512 : 2048;
    r.output_rows = last ? 1 : r.capacity;
    r.workloads = {last ? r.capacity : grid[1]};
    if ((last && !Grid(invocation, 1)) ||
        (!last &&
         (!grid[1] || grid[1] > r.capacity || !Grid(invocation, 1, grid[1]))))
      return Fail(error, "counted RMS grid/capacity differs");
  } else if (name == "llm.decoder.output_norm_last") {
    r.types = {"index",  "index",  "index",  "offset", "f32",
               "buffer", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types) || !Grid(invocation, 1))
      return Fail(error, "last RMS native ABI differs");
    r.capacity = Scalar(invocation, 0);
    r.width = r.weight_width = Scalar(invocation, 1);
    r.begin = Scalar(invocation, 2);
    r.count_origin = Scalar(invocation, 3);
    r.epsilon = 4;
    r.control = 5;
    r.input = 6;
    r.weight = 7;
    r.output = 8;
    r.output_rows = 1;
    r.control_kind = RmsRecipe::kCount;
    r.workloads = {r.capacity, r.width};
  } else if (name == "qwen38_rmsnorm_f32_prefill_last") {
    r.types = {"buffer", "f32", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types) || !Grid(invocation, 1))
      return Fail(error, "last RMS native ABI differs");
    r.capacity = 512;
    r.output_rows = 1;
    r.control = 0;
    r.epsilon = 1;
    r.input = 2;
    r.weight = 3;
    r.output = 4;
    r.control_kind = RmsRecipe::kCount;
  } else if (name == "qwen38_output_normalize_spans") {
    r.types = {"buffer", "f32", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types) || !grid[0] || grid[0] > 64 ||
        !Grid(invocation, grid[0]))
      return Fail(error, "output-span RMS native ABI differs");
    r.capacity = 512;
    r.output_rows = grid[0];
    r.control = 0;
    r.epsilon = 1;
    r.input = 2;
    r.weight = 3;
    r.output = 4;
    r.control_kind = RmsRecipe::kOutputSpans;
    r.workloads = {grid[0]};
  } else if (name == "qwen38_gdn_norm_gate_prefill") {
    r.types = {"index", "f32", "buffer", "buffer", "buffer", "buffer"};
    if (!Abi(invocation, r.types) || !plan.model.ssm ||
        plan.model.ssm->state_width != 128 || !plan.model.ssm->value_heads ||
        plan.model.ssm->value_heads > 48 ||
        plan.model.ssm->inner_width != plan.model.ssm->value_heads * 128 ||
        !grid[1] || !Grid(invocation, plan.model.ssm->value_heads, grid[1]))
      return Fail(error,
                  "GDN norm/gate requires its selected head geometry and ABI");
    r.capacity = r.output_rows = Scalar(invocation, 0);
    r.width = plan.model.ssm->inner_width;
    r.weight_width = 128;
    r.epsilon = 1;
    r.input = 2;
    r.weight = 3;
    r.gate = 4;
    r.output = 5;
    r.workloads = {grid[1]};
    r.recipe_id = "gdn_norm_gate_f32/v1";
    if (grid[1] > r.capacity)
      return Fail(error, "GDN norm/gate grid exceeds runtime capacity");
  } else {
    return PreparationStatus::kUnsupported;
  }
  if (r.epsilon == kAbsent &&
      (!std::isfinite(plan.model.rms_epsilon) || plan.model.rms_epsilon <= 0))
    return Fail(error, "configured RMS epsilon must be finite and positive");
  if (!r.capacity || r.capacity > r.maximum_capacity || !r.width ||
      r.width > 32768 || !r.weight_width || r.weight_width > 32768 ||
      !r.output_rows || r.output_rows > 2048 ||
      !PositiveEpsilon(invocation, r.epsilon))
    return Fail(error, "RMS geometry or epsilon must be finite positive F32");
  if (r.control_kind != RmsRecipe::kNone &&
      (!plan.workload.active_tokens || r.begin >= r.capacity ||
       plan.workload.active_tokens > r.capacity - r.begin ||
       r.count_origin % 4))
    return Fail(error, "RMS count/begin is outside the selected row domain");
  uint64_t control_bytes = 0;
  if (r.control_kind == RmsRecipe::kCount &&
      !Add(r.count_origin, 12, &control_bytes))
    return Fail(error, "RMS count origin overflows");
  if (r.control_kind == RmsRecipe::kOutputSpans)
    control_bytes = 1548;
  const uint64_t input_bytes = r.capacity * r.width * 4;
  const uint64_t weight_bytes = r.weight_width * 4;
  const uint64_t output_bytes = r.output_rows * r.width * 4;
  PreparedCase prepared;
  prepared.setup_recipe_id = r.recipe_id;
  prepared.source_workloads = std::move(r.workloads);
  prepared.source_argument_types = std::move(r.types);
  const auto input = Buffer(invocation, r.input);
  const auto weight = Buffer(invocation, r.weight);
  const auto output = Buffer(invocation, r.output);
  const BufferExtent input_extent{input, input_bytes, 4};
  const BufferExtent weight_extent{weight, weight_bytes, 4};
  const BufferExtent output_extent{output, output_bytes, 4};
  if (Overlaps(input_extent, weight_extent) ||
      Overlaps(input_extent, output_extent) ||
      Overlaps(weight_extent, output_extent))
    return Fail(error, "RMS operation requires disjoint argument spans");
  if (r.gate != kAbsent) {
    const BufferExtent gate{Buffer(invocation, r.gate), input_bytes, 4};
    for (const auto &extent : {input_extent, weight_extent, output_extent})
      if (Overlaps(gate, extent))
        return Fail(error, "GDN norm/gate operation requires disjoint spans");
  }
  if (!Gamma(plan, invocation, weight, r.weight_width, &prepared, error))
    return PreparationStatus::kFailure;
  const auto *parameter =
      FindBenchmarkParameter(plan, invocation, weight, weight_bytes, error);
  // Preserve buffer ABI order without conflating aliases by root identity.
  for (size_t i = 0; i < invocation.arguments.size(); ++i) {
    if (invocation.arguments[i].kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER)
      continue;
    const uint64_t length = i == r.input || i == r.gate ? input_bytes
                            : i == r.weight             ? weight_bytes
                            : i == r.output             ? output_bytes
                                                        : control_bytes;
    prepared.buffer_extents.push_back(
        {Buffer(invocation, i), length,
         i == r.weight ? std::max<uint64_t>(4, parameter->minimum_alignment)
                       : 4});
  }
  prepared.representative_weight_spans = {
      {weight, weight_bytes,
       std::max<uint64_t>(4, parameter->minimum_alignment)}};
  const auto allocation = Allocate(&prepared, max_bytes, error);
  if (allocation != PreparationStatus::kReady)
    return allocation;
  if (!SetFiniteF32Input(input, r.capacity * r.width, plan.workload.seed,
                         &prepared.roots, error) ||
      !SetTensorInput(weight, plan.source_asset,
                      prepared.weight_sources[0].source_offset, weight_bytes,
                      &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (r.gate != kAbsent &&
      !SetFiniteF32Input(Buffer(invocation, r.gate), r.capacity * r.width,
                         plan.workload.seed + 1, &prepared.roots, error))
    return PreparationStatus::kFailure;
  const auto *gamma = FindInputRoot(prepared.roots, weight);
  if (!FiniteF32(*gamma, weight.byte_offset - gamma->original_byte_offset,
                 weight_bytes))
    return Fail(
        error,
        "GGUF gamma has nonfinite values; fixture is not qualified finite");
  if (r.control_kind == RmsRecipe::kCount &&
      !SetAt(Buffer(invocation, r.control), r.count_origin,
             Count(plan.workload.active_tokens), &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (r.control_kind == RmsRecipe::kOutputSpans) {
    std::vector<uint8_t> bytes(control_bytes);
    Word(bytes, 0, static_cast<uint32_t>(plan.workload.active_tokens));
    Word(bytes, 4, 1);
    Word(bytes, 8, 1);
    Word(bytes, 12, static_cast<uint32_t>(plan.workload.active_tokens));
    Word(bytes, 1292, static_cast<uint32_t>(plan.workload.active_tokens - 1));
    if (!SetInputBytes(Buffer(invocation, r.control), bytes, &prepared.roots,
                       error))
      return PreparationStatus::kFailure;
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}

PreparationStatus Selection(const Plan &plan, const Invocation &invocation,
                            uint64_t max_bytes, PreparedCase *out,
                            std::string *error) {
  const auto &name = invocation.kernel_symbol;
  const bool partial = name == "qwen38_greedy_argmax_partial_rows_f32";
  const bool reduce = name == "qwen38_greedy_argmax_reduce_rows_f32";
  const bool rows_variant = name == "qwen38_greedy_argmax_rows_f32";
  if (!partial && !reduce && !rows_variant &&
      name != "qwen38_greedy_argmax_f32")
    return PreparationStatus::kUnsupported;
  const std::vector<std::string> types =
      partial || reduce ? std::vector<std::string>{"index", "index", "buffer",
                                                   "buffer", "buffer"}
      : rows_variant
          ? std::vector<std::string>{"index", "index", "buffer", "buffer"}
          : std::vector<std::string>{"index", "buffer", "buffer"};
  if (!Abi(invocation, types))
    return Fail(error, "argmax native ABI differs from its source contract");
  const uint64_t rows =
      rows_variant || partial || reduce ? Scalar(invocation, 0) : 1;
  const uint64_t vocabulary =
      Scalar(invocation, rows_variant || partial || reduce ? 1 : 0);
  if (!rows || rows > 512 || vocabulary < 256 || vocabulary > 262144 ||
      !plan.model.vocabulary_size || vocabulary > plan.model.vocabulary_size)
    return Fail(error, "argmax dimensions exceed the source contract");
  const uint64_t tiles = (vocabulary + 4095) / 4096;
  if (!(partial ? Grid(invocation, tiles, rows) : Grid(invocation, rows)))
    return Fail(error, "argmax grid differs from its logical dimensions");
  const size_t input_ordinal = rows_variant || partial || reduce ? 2 : 1;
  const auto input = Buffer(invocation, input_ordinal);
  const auto first_output = Buffer(invocation, input_ordinal + 1);
  PreparedCase prepared;
  prepared.setup_recipe_id =
      reduce ? "argmax_partial_winners_f32/v1" : "argmax_logits_f32/v1";
  prepared.source_argument_types = types;
  prepared.source_workloads = rows_variant || partial || reduce
                                  ? std::vector<uint64_t>{rows, vocabulary}
                                  : std::vector<uint64_t>{vocabulary};
  const uint64_t input_columns = reduce ? tiles : vocabulary;
  prepared.buffer_extents = {{input, rows * input_columns * 4, 4},
                             {first_output,
                              rows *
                                  (partial  ? tiles
                                   : reduce ? tiles
                                            : 1) *
                                  4,
                              4}};
  if (partial || reduce)
    prepared.buffer_extents.push_back({Buffer(invocation, input_ordinal + 2),
                                       rows * (partial ? tiles : 1) * 4, 4});
  for (size_t i = 0; i < prepared.buffer_extents.size(); ++i)
    for (size_t j = i + 1; j < prepared.buffer_extents.size(); ++j)
      if (Overlaps(prepared.buffer_extents[i], prepared.buffer_extents[j]))
        return Fail(error, "argmax requires disjoint input/output spans");
  const auto allocation = Allocate(&prepared, max_bytes, error);
  if (allocation != PreparationStatus::kReady)
    return allocation;
  if (!SetFiniteF32Input(input, rows * input_columns, plan.workload.seed,
                         &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (reduce) {
    // Valid tile winners: each ID belongs to its tile, including a tail tile.
    std::vector<uint8_t> ids(rows * tiles * 4);
    for (uint64_t row = 0; row < rows; ++row)
      for (uint64_t tile = 0; tile < tiles; ++tile)
        Word(ids, (row * tiles + tile) * 4,
             static_cast<uint32_t>(
                 std::min<uint64_t>(tile * 4096 + 17, vocabulary - 1)));
    if (!SetInputBytes(first_output, ids, &prepared.roots, error))
      return PreparationStatus::kFailure;
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}

PreparationStatus Snapshot(const Plan &plan, const Invocation &invocation,
                           uint64_t max_bytes, PreparedCase *out,
                           std::string *error) {
  if (invocation.kernel_symbol != "llm.epoch.snapshot_count")
    return PreparationStatus::kUnsupported;
  const std::vector<std::string> types{"index", "offset", "buffer", "buffer"};
  if (!Abi(invocation, types) || !Grid(invocation, 1) ||
      Scalar(invocation, 1) % 4 || !plan.workload.active_tokens ||
      plan.workload.active_tokens > INT32_MAX)
    return Fail(
        error, "snapshot count requires a valid kind, origin and active count");
  uint64_t length;
  if (!Add(Scalar(invocation, 1), 12, &length))
    return Fail(error, "snapshot count origin overflows");
  PreparedCase prepared;
  prepared.setup_recipe_id = "epoch_snapshot_count/v1";
  prepared.source_argument_types = types;
  const auto packet = Buffer(invocation, 2), count = Buffer(invocation, 3);
  prepared.buffer_extents = {{packet, length, 4}, {count, 12, 4}};
  if (Overlaps(prepared.buffer_extents[0], prepared.buffer_extents[1]))
    return Unsupported(error, "snapshot count with overlapping packet/output");
  const auto allocation = Allocate(&prepared, max_bytes, error);
  if (allocation != PreparationStatus::kReady)
    return allocation;
  if (!SetAt(packet, Scalar(invocation, 1), Count(plan.workload.active_tokens),
             &prepared.roots, error))
    return PreparationStatus::kFailure;
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}

PreparationStatus Publication(const Plan &plan, const Invocation &invocation,
                              uint64_t max_bytes, PreparedCase *out,
                              std::string *error) {
  const auto &name = invocation.kernel_symbol;
  const bool selected_ids = name == "qwen08_publish_selected_ids";
  const bool epoch =
      name == "llama32_publish_epoch_id" || name == "gemma3_publish_epoch_id" ||
      name == "qwen2_publish_epoch_id" || name == "qwen3_publish_epoch_id" ||
      name == "lfm3_publish_epoch_id" || name == "spark2_5_publish_epoch_id";
  if (!selected_ids && !epoch)
    return PreparationStatus::kUnsupported;
  const std::vector<std::string> types{"buffer", "buffer", "buffer"};
  if (!Abi(invocation, types) || !Grid(invocation, 1))
    return Fail(error,
                "publication native ABI differs from its source contract");
  if (!plan.model.vocabulary_size || plan.model.vocabulary_size > INT32_MAX ||
      !plan.workload.active_tokens || plan.workload.active_tokens > INT32_MAX ||
      plan.workload.history_tokens > INT32_MAX)
    return Fail(error, "publication requires a signed-I32 vocabulary");
  uint32_t terminal = 0, winner = 0;
  if (epoch) {
    if (!plan.publication_tokens ||
        plan.publication_tokens->raw_stops.empty() ||
        plan.publication_tokens->terminal_index >=
            plan.publication_tokens->raw_stops.size())
      return Fail(
          error,
          "publication requires the existing GGUF source terminal facts");
    const auto &facts = *plan.publication_tokens;
    const int32_t raw_terminal = facts.raw_stops[facts.terminal_index].id;
    if (raw_terminal < 0 ||
        static_cast<uint64_t>(raw_terminal) >= plan.model.vocabulary_size)
      return Fail(error, "GGUF publication terminal exceeds the vocabulary");
    terminal = static_cast<uint32_t>(raw_terminal);
    while (winner < plan.model.vocabulary_size &&
           std::any_of(facts.raw_stops.begin(), facts.raw_stops.end(),
                       [&](const TokenIdentity &stop) {
                         return stop.id == static_cast<int32_t>(winner);
                       }))
      ++winner;
    if (winner == plan.model.vocabulary_size)
      return Fail(error, "publication has no nonterminal vocabulary identity");
  } else {
    winner = static_cast<uint32_t>(plan.model.vocabulary_size - 1);
  }
  PreparedCase prepared;
  prepared.setup_recipe_id = "selected_id_publication/v1";
  prepared.source_argument_types = types;
  if (selected_ids)
    prepared.source_workloads = {1};
  const auto packet = Buffer(invocation, 0), ids = Buffer(invocation, 1),
             output = Buffer(invocation, 2);
  prepared.buffer_extents = {{packet, epoch ? 2064u : 32u, 4},
                             {ids, epoch ? 16u : 4u, 4},
                             {output, 64, 4}};
  for (size_t i = 0; i < prepared.buffer_extents.size(); ++i)
    for (size_t j = i + 1; j < prepared.buffer_extents.size(); ++j)
      if (Overlaps(prepared.buffer_extents[i], prepared.buffer_extents[j]))
        return Unsupported(error,
                           "publication with overlapping source/output spans");
  const auto allocation = Allocate(&prepared, max_bytes, error);
  if (allocation != PreparationStatus::kReady)
    return allocation;
  std::vector<uint8_t> descriptor(epoch ? 2064 : 32);
  Word(descriptor, 0, static_cast<uint32_t>(plan.workload.active_tokens));
  Word(descriptor, 4, 1);
  Word(descriptor, 8, 1); // Exactly one live selection, not a count-zero no-op.
  Word(descriptor, 12, static_cast<uint32_t>(plan.workload.active_tokens));
  Word(descriptor, 16, static_cast<uint32_t>(plan.workload.history_tokens));
  Word(descriptor, 28, 3); // Physical output slot differs from logical ordinal.
  if (epoch)
    Word(descriptor, 2060, terminal);
  std::vector<uint8_t> selected(epoch ? 16 : 4);
  Word(selected, 0, winner);
  if (!SetInputBytes(packet, descriptor, &prepared.roots, error) ||
      !SetInputBytes(ids, selected, &prepared.roots, error))
    return PreparationStatus::kFailure;
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}

PreparationStatus Embedding(const Plan &plan, const Invocation &invocation,
                            uint64_t max_bytes, PreparedCase *out,
                            std::string *error) {
  const auto &name = invocation.kernel_symbol;
  const bool ordinary = name == "llm.embedding.ordinary_f32_kernel";
  const bool generic_epoch = name == "llm.token_embedding_epoch_f32";
  const bool gemma_q8 = name == "gemma3_token_embedding_epoch_q8_f32";
  const bool qwen_q4 = name == "qwen38_token_embedding_q4k_f32_prefill";
  const bool qwen_q6 = name == "qwen38_token_embedding_q6k_f16_prefill";
  if (!ordinary && !generic_epoch && !gemma_q8 && !qwen_q4 && !qwen_q6)
    return PreparationStatus::kUnsupported;
  std::vector<std::string> types =
      ordinary ? std::vector<std::string>{"index",  "index",  "index",
                                          "offset", "buffer", "buffer",
                                          "buffer", "buffer", "buffer"}
      : generic_epoch
          ? std::vector<std::string>{"index",  "index",  "index",  "buffer",
                                     "buffer", "buffer", "buffer", "buffer"}
      : gemma_q8 ? std::vector<std::string>(5, "buffer")
                 : std::vector<std::string>(3, "buffer");
  const auto &grid = invocation.command.payload.dispatch.workgroup_count;
  if (!Abi(invocation, types) || !grid[0] ||
      grid[0] > (generic_epoch || gemma_q8 ? 2048u : 512u) ||
      !Grid(invocation, grid[0]))
    return Fail(error, "embedding requires its direct row-grid ABI");
  const uint64_t rows = grid[0];
  const uint64_t width = ordinary || generic_epoch ? Scalar(invocation, 0)
                                                   : plan.model.embedding_width;
  const uint64_t vocabulary = ordinary || generic_epoch
                                  ? Scalar(invocation, 1)
                                  : plan.model.vocabulary_size;
  const uint64_t tag = ordinary || generic_epoch ? Scalar(invocation, 2)
                       : gemma_q8                ? 8
                       : qwen_q4                 ? 12
                                                 : 14;
  const uint64_t block_values = tag == 8 ? 32 : 256;
  const uint64_t block_bytes = tag == 8    ? 34
                               : tag == 12 ? 144
                               : tag == 13 ? 176
                               : tag == 14 ? 210
                                           : 0;
  if (!width || width > 32768 || width % block_values || !block_bytes ||
      !vocabulary || vocabulary > (ordinary ? 262144u : uint64_t(INT32_MAX)) ||
      width != plan.model.embedding_width ||
      vocabulary != plan.model.vocabulary_size ||
      !plan.workload.active_tokens || plan.workload.active_tokens > rows)
    return Fail(
        error,
        "embedding shape/tag/active count differs from its table contract");
  if ((ordinary || generic_epoch) && tag != 12 && tag != 13 && tag != 14)
    return Fail(
        error,
        "ordinary/epoch embedding supports the selected QK table encodings");
  const uint64_t table_bytes = width / block_values * block_bytes * vocabulary;
  const size_t epoch_i = ordinary        ? 4
                         : generic_epoch ? 3
                         : gemma_q8      ? 0
                                         : kAbsent;
  const size_t count_i = ordinary        ? 5
                         : generic_epoch ? 4
                         : gemma_q8      ? 1
                                         : kAbsent;
  const size_t ids_i = ordinary ? 6 : generic_epoch ? 5 : gemma_q8 ? 2 : 0;
  const size_t weight_i = ids_i + 1, output_i = ids_i + 2;
  const uint64_t origin = ordinary ? Scalar(invocation, 3) : 12;
  uint64_t epoch_bytes = 0;
  if (epoch_i != kAbsent && (origin % 4 || !Add(origin, 20, &epoch_bytes)))
    return Fail(error, "embedding descriptor origin is invalid");
  const auto weight = Buffer(invocation, weight_i);
  const auto *parameter =
      FindBenchmarkParameter(plan, invocation, weight, table_bytes, error);
  const auto *tensor =
      parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
  if (!tensor || weight.role != LOOMC_CMD_BUFFER_ROLE_FIXED ||
      weight.byte_offset != parameter->byte_offset ||
      parameter->byte_length != table_bytes || tensor->storage_type != tag ||
      tensor->rank != 2 ||
      tensor->normalized_extents[0] != static_cast<int64_t>(width) ||
      tensor->normalized_extents[1] != static_cast<int64_t>(vocabulary) ||
      tensor->normalized_extents[2] != 1 ||
      tensor->normalized_extents[3] != 1 || tensor->byte_size != table_bytes)
    return Fail(
        error,
        "embedding requires its exact full GGUF table, encoding and shape");
  PreparedCase prepared;
  prepared.setup_recipe_id = "embedding_table_f32/v1";
  prepared.source_argument_types = std::move(types);
  prepared.source_workloads = {rows};
  prepared.weight_sources = {
      {tensor->name, tensor->absolute_offset, table_bytes}};
  const uint64_t id_count = qwen_q4 || qwen_q6 ? 512 : rows;
  if (epoch_i != kAbsent)
    prepared.buffer_extents.push_back(
        {Buffer(invocation, epoch_i), epoch_bytes, 4});
  if (count_i != kAbsent)
    prepared.buffer_extents.push_back({Buffer(invocation, count_i), 12, 4});
  prepared.buffer_extents.push_back(
      {Buffer(invocation, ids_i), id_count * 4, 4});
  prepared.buffer_extents.push_back(
      {weight, table_bytes,
       std::max<uint64_t>(4, parameter->minimum_alignment)});
  prepared.buffer_extents.push_back(
      {Buffer(invocation, output_i), rows * width * 4, 4});
  prepared.representative_weight_spans = {
      prepared.buffer_extents[prepared.buffer_extents.size() - 2]};
  // Output cannot overlap any table, ID or control bytes it still reads.
  const auto &output_extent = prepared.buffer_extents.back();
  for (size_t i = 0; i + 1 < prepared.buffer_extents.size(); ++i)
    if (Overlaps(prepared.buffer_extents[i], output_extent))
      return Unsupported(error,
                         "embedding with overlapping input/output spans");
  const auto allocation = Allocate(&prepared, max_bytes, error);
  if (allocation != PreparationStatus::kReady)
    return allocation;
  if (!SetTensorInput(weight, plan.source_asset, tensor->absolute_offset,
                      table_bytes, &prepared.roots, error))
    return PreparationStatus::kFailure;
  // Every quantization header is a defined finite scale. These are actual
  // checkpoint bytes; a bad scale is rejected rather than silently repaired.
  const auto *table = FindInputRoot(prepared.roots, weight);
  const uint64_t base = weight.byte_offset - table->original_byte_offset;
  for (uint64_t block = 0; block < table_bytes; block += block_bytes) {
    const uint64_t d = base + block + (tag == 14 ? 208 : 0);
    const uint16_t bits =
        table->bytes[d] | (uint16_t(table->bytes[d + 1]) << 8);
    if ((bits & 0x7C00) == 0x7C00)
      return Fail(error,
                  "embedding GGUF table has a nonfinite quantization scale");
    if ((tag == 12 || tag == 13) &&
        ((table->bytes[d + 2] | (uint16_t(table->bytes[d + 3]) << 8)) &
         0x7C00) == 0x7C00)
      return Fail(error,
                  "embedding GGUF table has a nonfinite quantization offset");
  }
  std::vector<uint8_t> ids(id_count * 4);
  for (uint64_t row = 0; row < id_count; ++row)
    Word(ids, row * 4,
         static_cast<uint32_t>(
             row % 2 ? vocabulary - 1 : std::min<uint64_t>(1, vocabulary - 1)));
  if (!SetInputBytes(Buffer(invocation, ids_i), ids, &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (count_i != kAbsent && !SetInputBytes(Buffer(invocation, count_i),
                                           Count(plan.workload.active_tokens),
                                           &prepared.roots, error))
    return PreparationStatus::kFailure;
  if (epoch_i != kAbsent) {
    std::vector<uint8_t> descriptor(20);
    Word(descriptor, 0, static_cast<uint32_t>(plan.workload.active_tokens));
    if (plan.workload.history_tokens > INT32_MAX)
      return Fail(error, "embedding history is outside its I32 epoch contract");
    Word(descriptor, 4, static_cast<uint32_t>(plan.workload.history_tokens));
    // A single declared span starts at token index zero. The inactive tail
    // remains valid data but is not selected by its count packet.
    if (!SetAt(Buffer(invocation, epoch_i), origin, descriptor, &prepared.roots,
               error))
      return PreparationStatus::kFailure;
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace

PreparationStatus PrepareBasicCase(const Plan &plan,
                                   const Invocation &invocation,
                                   uint64_t max_input_bytes, PreparedCase *out,
                                   std::string *error) {
  if (!out)
    return Fail(error, "prepared benchmark case output is required");
  using Preparer =
      PreparationStatus (*)(const Plan &, const Invocation &, uint64_t,
                            PreparedCase *, std::string *);
  for (const Preparer prepare :
       {Rms, Selection, Snapshot, Publication, Embedding}) {
    std::string reason;
    const auto status =
        prepare(plan, invocation, max_input_bytes, out, &reason);
    if (status != PreparationStatus::kUnsupported || !reason.empty()) {
      if (error)
        *error = std::move(reason);
      return status;
    }
  }
  // An unrecognized family leaves an earlier preparer's specific reason intact.
  return PreparationStatus::kUnsupported;
}
} // namespace hrx::loomify_gguf::benchmark
