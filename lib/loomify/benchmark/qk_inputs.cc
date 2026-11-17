// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hrx::loomify_gguf::benchmark {
namespace {
using Ref = loomc_cmd_buffer_ref_t;
bool Overlap(Ref a, uint64_t al, Ref b, uint64_t bl) {
  return al && bl && a.role == b.role && a.root_index == b.root_index &&
         (a.byte_offset <= b.byte_offset ? b.byte_offset - a.byte_offset < al
                                         : a.byte_offset - b.byte_offset < bl);
}
bool Slice(Ref ref, uint64_t origin, uint64_t length, Ref *out) {
  if (origin > INT64_MAX || ref.byte_offset > UINT64_MAX - origin ||
      (ref.byte_length != UINT64_MAX &&
       (origin > ref.byte_length || length > ref.byte_length - origin)))
    return false;
  ref.byte_offset += origin;
  ref.byte_length = length;
  *out = ref;
  return true;
}
std::vector<uint8_t> Words(std::initializer_list<uint32_t> words) {
  std::vector<uint8_t> bytes;
  for (auto word : words)
    for (unsigned i = 0; i < 4; ++i)
      bytes.push_back(static_cast<uint8_t>(word >> (i * 8)));
  return bytes;
}
float Float(const PreparedRoot &root, uint64_t offset) {
  uint32_t bits = 0;
  for (unsigned i = 0; i < 4; ++i)
    bits |= uint32_t(root.bytes[offset + i]) << (i * 8);
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}
bool FiniteScalar(const loomc_cmd_argument_t &arg, bool positive) {
  if (arg.kind != LOOMC_CMD_ARGUMENT_KIND_B32 ||
      arg.value.scalar_bits > UINT32_MAX)
    return false;
  uint32_t bits = static_cast<uint32_t>(arg.value.scalar_bits);
  float value;
  std::memcpy(&value, &bits, 4);
  return std::isfinite(value) && (!positive || value > 0);
}
struct Weight {
  size_t argument;
  uint64_t elements;
  const Parameter *parameter;
  const model::TensorRecord *tensor;
};
bool BindWeight(const Plan &plan, const Invocation &invocation, size_t argument,
                uint64_t elements, Weight *out, std::string *error) {
  const auto &ref = invocation.arguments[argument].value.buffer;
  const auto *parameter =
      FindBenchmarkParameter(plan, invocation, ref, elements * 4, error);
  const auto *tensor =
      parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
  if (!parameter || !tensor || tensor->storage_type != 0 ||
      tensor->byte_size != elements * 4 || tensor->rank != 1 ||
      tensor->normalized_extents[0] != static_cast<int64_t>(elements) ||
      parameter->byte_offset != ref.byte_offset ||
      parameter->byte_length != elements * 4) {
    if (error)
      *error = "QK recipe requires an exact selected rank-one F32 parameter";
    return false;
  }
  *out = {argument, elements, parameter, tensor};
  return true;
}
bool LoadWeight(const Plan &plan, const Invocation &inv, const Weight &weight,
                PreparedCase *prepared, std::string *error,
                bool factors = false, bool at_least_one = false) {
  const auto &ref = inv.arguments[weight.argument].value.buffer;
  if (!SetTensorInput(ref, plan.source_asset, weight.tensor->absolute_offset,
                      weight.elements * 4, &prepared->roots, error))
    return false;
  const auto *root = FindInputRoot(prepared->roots, ref);
  for (uint64_t i = 0; i < weight.elements; ++i) {
    const float value =
        Float(*root, ref.byte_offset - root->original_byte_offset + i * 4);
    if (!std::isfinite(value) ||
        (factors && (at_least_one ? value < 1 : value <= 0))) {
      if (error)
        *error = "QK learned parameter contains invalid/nonfinite F32 values";
      return false;
    }
  }
  prepared->weight_sources.push_back({weight.tensor->name,
                                      weight.tensor->absolute_offset,
                                      weight.elements * 4});
  prepared->representative_weight_spans.push_back(
      {ref, weight.elements * 4,
       std::max<uint64_t>(4, weight.parameter->minimum_alignment)});
  return true;
}
// A deliberately narrow producer check, not a graph evaluator. Any intervening
// command touching this view invalidates this recipe's unit-factor proof.
bool UnitProducer(const Plan &plan, const Invocation &inv, Ref ref) {
  if (inv.program_ordinal >= plan.programs.size())
    return false;
  const auto &commands = plan.programs[inv.program_ordinal].commands;
  if (inv.command_ordinal >= commands.size())
    return false;
  for (size_t c = inv.command_ordinal; c-- > 0;) {
    const auto &command = commands[c];
    if (command.kind == LOOMC_CMD_COMMAND_KIND_BARRIER_EXECUTION)
      continue;
    const Invocation *prior = nullptr;
    for (const auto &candidate : plan.invocations)
      if (candidate.program_ordinal == inv.program_ordinal &&
          candidate.command_ordinal == c) {
        prior = &candidate;
        break;
      }
    if (!prior)
      return false;
    bool touches = false;
    for (const auto &arg : prior->arguments)
      if (arg.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER &&
          Overlap(ref, 256, arg.value.buffer, arg.value.buffer.byte_length))
        touches = true;
    if (!touches)
      continue;
    if (prior->kernel_symbol != "llama32_unit_rope_factors" ||
        prior->command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
        prior->arguments.size() != 1 || prior->command.argument_count != 1 ||
        prior->arguments[0].kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER)
      return false;
    const auto *grid = prior->command.payload.dispatch.workgroup_count;
    const auto &output = prior->arguments[0].value.buffer;
    return grid[0] == 1 && grid[1] == 1 && grid[2] == 1 &&
           output.role == ref.role && output.root_index == ref.root_index &&
           output.byte_offset == ref.byte_offset && output.byte_length >= 256;
  }
  return false;
}
} // namespace

PreparationStatus PrepareRotaryFactors(const Plan &plan, const Invocation &inv,
                                       size_t argument, bool at_least_one,
                                       PreparedCase *prepared,
                                       std::string *error) {
  const auto fail = [&](const char *message) {
    if (error)
      *error = message;
    return PreparationStatus::kFailure;
  };
  if (!prepared || argument >= inv.arguments.size() ||
      inv.arguments[argument].kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER)
    return fail("invalid rotary factor argument");
  const auto &ref = inv.arguments[argument].value.buffer;
  if (ref.role == LOOMC_CMD_BUFFER_ROLE_FIXED) {
    Weight weight;
    if (!BindWeight(plan, inv, argument, 64, &weight, error) ||
        !LoadWeight(plan, inv, weight, prepared, error, true, at_least_one))
      return PreparationStatus::kFailure;
    return PreparationStatus::kReady;
  }
  if (!UnitProducer(plan, inv, ref)) {
    if (error)
      *error =
          "rotary factors have no selected GGUF parameter or proven unit producer";
    return PreparationStatus::kUnsupported;
  }
  std::vector<uint8_t> units;
  for (unsigned i = 0; i < 64; ++i) {
    const auto word = Words({0x3f800000});
    units.insert(units.end(), word.begin(), word.end());
  }
  return SetInputBytes(ref, units, &prepared->roots, error)
             ? PreparationStatus::kReady
             : PreparationStatus::kFailure;
}

PreparationStatus PrepareQkCase(const Plan &plan, const Invocation &inv,
                                uint64_t budget, PreparedCase *out,
                                std::string *error) {
  const auto &name = inv.kernel_symbol;
  const bool norm = name == "llm.decoder.prepare_qk_h128";
  const bool bias = name == "llm.decoder.prepare_qk_bias_h128";
  const bool factored = name == "llm.attention.prepare_qk_factored_h128";
  const bool raw = name == "llm.attention.prepare_qk_raw_h256";
  const bool gemma = name == "gemma3_prepare_qk";
  const bool llama = name == "llama32_rotary_query";
  if (!norm && !bias && !factored && !raw && !gemma && !llama)
    return PreparationStatus::kUnsupported;
  const auto fail = [&](const char *message) {
    if (error)
      *error = message;
    return PreparationStatus::kFailure;
  };
  const auto unsupported = [&](const char *message) {
    if (error)
      *error = message;
    return PreparationStatus::kUnsupported;
  };
  if (!out || inv.program_ordinal >= plan.programs.size())
    return fail("QK recipe needs valid output/program");
  if (inv.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
    return unsupported("QK recipe needs resolved direct counts");
  const size_t scalars = norm       ? 4
                         : bias     ? 3
                         : factored ? 5
                         : raw      ? 4
                         : gemma    ? 2
                                    : 3;
  const size_t buffers = factored ? 5 : (raw ? 3 : 4);
  const auto &args = inv.arguments;
  if (args.size() != scalars + buffers ||
      inv.command.argument_count != args.size())
    return fail("QK native argument count differs");
  std::vector<std::string> types;
  if (norm)
    types = {"index", "f32", "f32", "offset"};
  if (bias)
    types = {"index", "f32", "offset"};
  if (factored)
    types = {"index", "f32", "f32", "i1", "offset"};
  if (raw)
    types = {"index", "index", "f32", "offset"};
  if (gemma)
    types = {"index", "offset"};
  if (llama)
    types = {"index", "offset", "index"};
  for (size_t i = 0; i < scalars; ++i) {
    const auto kind = types[i] == "f32"      ? LOOMC_CMD_ARGUMENT_KIND_B32
                      : types[i] == "offset" ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                      : types[i] == "i1"     ? LOOMC_CMD_ARGUMENT_KIND_B8
                                             : LOOMC_CMD_ARGUMENT_KIND_INDEX;
    if (args[i].kind != kind)
      return fail("QK native scalar carrier differs");
    if (types[i] == "f32" &&
        !FiniteScalar(args[i], (norm || factored) && i == 1))
      return fail("QK epsilon/rotary scalar is invalid");
    if (types[i] == "i1" && args[i].value.scalar_bits > 1)
      return fail("QK factor flag must be exact i1");
  }
  for (size_t i = scalars; i < args.size(); ++i)
    if (args[i].kind != LOOMC_CMD_ARGUMENT_KIND_BUFFER)
      return fail("QK native buffer carrier differs");
  const auto *grid = inv.command.payload.dispatch.workgroup_count;
  const uint64_t rows = grid[1], heads = grid[0],
                 h = (raw || gemma) ? 256 : 128;
  const uint64_t active = plan.workload.active_tokens,
                 history = plan.workload.history_tokens;
  const uint64_t capacity = (gemma || llama) ? plan.workload.context_capacity
                                             : args[0].value.scalar_bits;
  if (!rows || rows > 512 || rows != plan.workload.token_capacity || !heads ||
      heads > (gemma ? 4 : 128) || grid[2] != 1 || !active || active > rows ||
      !capacity || capacity > (llama ? 131072 : 262144) ||
      history >= capacity || active > capacity - history || history > INT32_MAX)
    return fail("QK launch geometry/positions are invalid");
  if (raw && (args[1].value.scalar_bits < 2 ||
              args[1].value.scalar_bits > 256 || args[1].value.scalar_bits % 2))
    return fail("QK raw rotary width is invalid");
  uint64_t begin = 0;
  if (gemma) {
    const uint64_t layer = args[0].value.scalar_bits;
    if (layer >= plan.model.layers.size() || layer >= 64 ||
        plan.model.key_head_width != 256 ||
        (heads != plan.model.layers[layer].query_heads &&
         heads != plan.model.layers[layer].kv_heads))
      return fail("Gemma QK layer/head configuration differs");
  }
  if (llama) {
    begin = args[2].value.scalar_bits;
    if (plan.model.key_head_width != 128 || plan.model.layers.empty() ||
        heads != plan.model.layers[0].query_heads ||
        args[0].value.scalar_bits > 1 ||
        (args[0].value.scalar_bits == 1 && active != 1) || begin >= 512 ||
        active > 512 - begin)
      return fail("Llama QK controls/storage geometry differs");
  }
  const size_t origin_arg = llama ? 1 : scalars - 1;
  const uint64_t origin = args[origin_arg].value.scalar_bits;
  if (origin > INT64_MAX || origin % 4 ||
      origin > UINT64_MAX - (llama ? 12 : 20))
    return fail("QK control origin is invalid");
  const auto control = args[scalars].value.buffer;
  const auto output = args.back().value.buffer;
  const uint64_t storage_rows = llama ? begin + active : rows;
  const uint64_t matrix_bytes = storage_rows * heads * h * 4;
  std::vector<uint64_t> lengths(args.size(), 0);
  lengths[scalars] = origin + (llama ? 12 : 20);
  lengths[scalars + 1] = matrix_bytes;
  lengths.back() = matrix_bytes;
  std::vector<Weight> weights;
  bool use_factors = factored && args[3].value.scalar_bits;
  if (norm || bias || factored || gemma) {
    Weight weight;
    const uint64_t elements = bias ? heads * h : h;
    if (!BindWeight(plan, inv, scalars + 2, elements, &weight, error))
      return PreparationStatus::kFailure;
    weights.push_back(weight);
    lengths[scalars + 2] = elements * 4;
  }
  if (factored)
    lengths[scalars + 3] = use_factors ? 256 : 4;
  if (llama)
    lengths[scalars + 1] = 256; // Llama places factors before projection.
  const size_t projection_index = llama ? 5 : scalars + 1;
  if (llama)
    lengths[projection_index] = matrix_bytes;
  uint64_t factor_alignment = 4;
  const size_t factor_index = llama ? 4 : scalars + 3;
  if ((llama || use_factors) &&
      args[factor_index].value.buffer.role == LOOMC_CMD_BUFFER_ROLE_FIXED) {
    Weight factor;
    if (!BindWeight(plan, inv, factor_index, 64, &factor, error))
      return PreparationStatus::kFailure;
    factor_alignment =
        std::max<uint64_t>(4, factor.parameter->minimum_alignment);
  }
  auto semantic_lengths = lengths;
  if (factored && !use_factors)
    semantic_lengths[factor_index] = 0; // The source never reads this argument.
  PreparedCase prepared;
  prepared.setup_recipe_id = "qk_rotary_f32/v1";
  prepared.source_workloads =
      llama ? std::vector<uint64_t>{rows} : std::vector<uint64_t>{rows, heads};
  prepared.source_argument_types = types;
  prepared.source_argument_types.insert(prepared.source_argument_types.end(),
                                        buffers, "buffer");
  for (size_t i = scalars; i < args.size(); ++i) {
    const auto &ref = args[i].value.buffer;
    if (ref.byte_length != UINT64_MAX) {
      if (lengths[i] > ref.byte_length)
        return fail("QK required span exceeds native binding");
      lengths[i] = ref.byte_length;
    }
    uint64_t alignment = 4;
    for (const auto &weight : weights)
      if (weight.argument == i)
        alignment =
            std::max<uint64_t>(alignment, weight.parameter->minimum_alignment);
    if ((llama && i == 4) || (use_factors && i == scalars + 3)) {
      alignment = std::max(alignment, factor_alignment);
    }
    prepared.buffer_extents.push_back({ref, lengths[i], alignment});
  }
  // These initializers represent different semantic values. Sharing an input
  // root is valid, but overlapping semantic spans could overwrite packet or
  // learned bytes during setup. This recipe excludes those layouts; it is not
  // a kernel-wide alias restriction. Ignore unused native-view padding here.
  for (size_t i = scalars; i < args.size(); ++i)
    for (size_t j = i + 1; j < args.size(); ++j)
      if (Overlap(args[i].value.buffer, semantic_lengths[i],
                  args[j].value.buffer, semantic_lengths[j]))
        return unsupported(
            "QK recipe requires disjoint semantic argument spans");
  bool over = false;
  if (!AllocateInputRoots(prepared.buffer_extents, budget, &prepared.roots,
                          error, &over))
    return over ? PreparationStatus::kUnsupported : PreparationStatus::kFailure;
  Ref descriptor;
  if (!Slice(control, origin, llama ? 12 : 20, &descriptor))
    return fail("QK descriptor exceeds binding");
  auto packet = llama ? (args[0].value.scalar_bits == 1
                             ? Words({static_cast<uint32_t>(history), 0, 0})
                             : Words({static_cast<uint32_t>(active),
                                      static_cast<uint32_t>(history), 0}))
                      : Words({static_cast<uint32_t>(active),
                               static_cast<uint32_t>(history), 0, 0, 0});
  if (!SetInputBytes(descriptor, packet, &prepared.roots, error) ||
      !SetFiniteF32Input(args[projection_index].value.buffer,
                         storage_rows * heads * h, plan.workload.seed,
                         &prepared.roots, error) ||
      !SetInputBytes(output, std::vector<uint8_t>(matrix_bytes, 0xA5),
                     &prepared.roots, error))
    return PreparationStatus::kFailure;
  for (const auto &weight : weights)
    if (!LoadWeight(plan, inv, weight, &prepared, error))
      return PreparationStatus::kFailure;
  if (llama || use_factors) {
    const auto status = PrepareRotaryFactors(plan, inv, llama ? 4 : scalars + 3,
                                             llama, &prepared, error);
    if (status != PreparationStatus::kReady)
      return status;
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
}
} // namespace hrx::loomify_gguf::benchmark
