// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include "lib/loomify/benchmark/qwen_span_inputs.h"

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

// The remaining wrappers carry the same numerical layouts through distinct
// packet ABIs. This prepares those exact arguments; it does not rebuild a
// graph.
PreparationStatus PrepareCacheSpanCase(const Plan &plan,
                                       const Invocation &invocation,
                                       uint64_t max_bytes, PreparedCase *out,
                                       std::string *error) {
  const auto &name = invocation.kernel_symbol;
  const bool llama_publish = name == "llama32_publish_kv";
  const bool llama_attention = name == "llama32_attention_rows";
  const bool gemma_publish = name == "gemma3_publish_kv";
  const bool gemma_attention = name == "gemma3_attention_rows";
  const bool span_query = name == "qwen38_attention_query_prepare_spans";
  const bool span_publish = name == "qwen38_attention_kv_prepare_spans";
  const bool span_attention = name == "qwen38_attention_spans_f16_wmma";
  const bool llama = llama_publish || llama_attention;
  const bool span = span_query || span_publish || span_attention;
  if (!llama && !span && !gemma_publish && !gemma_attention)
    return PreparationStatus::kUnsupported;
  const auto failure = [&](const std::string &message) {
    Fail(error, message);
    return PreparationStatus::kFailure;
  };
  const auto unsupported = [&](const std::string &message) {
    Fail(error, message);
    return PreparationStatus::kUnsupported;
  };
  if (!out)
    return failure("cache/span output is required");
  std::vector<std::string> types;
  if (llama)
    types = {"index", "offset", "index", "offset", "index", "offset"};
  else if (!span)
    types = {"index", "offset", "offset", "offset"};
  if (!span)
    types.insert(types.end(), llama_publish ? 6 : 5, "buffer");
  else if (span_query)
    types = {"buffer", "f32", "buffer", "buffer", "buffer", "buffer"};
  else if (span_publish)
    types = {"buffer", "buffer", "offset", "f32",
             "buffer", "buffer", "buffer", "buffer"};
  else
    types = {"buffer", "buffer", "offset", "buffer",
             "buffer", "buffer", "buffer"};
  if (invocation.command.kind != LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
    return unsupported("cache/span recipe requires a resolved direct launch");
  if (invocation.arguments.size() != types.size() ||
      invocation.command.argument_count != types.size())
    return failure("cache/span native argument count differs");
  for (size_t i = 0; i < types.size(); ++i) {
    const auto kind = types[i] == "buffer"   ? LOOMC_CMD_ARGUMENT_KIND_BUFFER
                      : types[i] == "index"  ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                      : types[i] == "offset" ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                                             : LOOMC_CMD_ARGUMENT_KIND_B32;
    if (invocation.arguments[i].kind != kind)
      return failure("cache/span native argument carrier differs");
  }
  const auto scalar = [&](size_t i) {
    return invocation.arguments[i].value.scalar_bits;
  };
  const auto buffer = [&](size_t i) {
    return invocation.arguments[i].value.buffer;
  };
  const auto &w = plan.workload;
  if (!w.token_capacity || w.token_capacity > 512 || !w.active_tokens ||
      w.active_tokens > w.token_capacity || !w.context_capacity ||
      w.context_capacity > (llama ? 131072u : 262144u) ||
      w.history_tokens >= w.context_capacity ||
      w.active_tokens > w.context_capacity - w.history_tokens)
    return failure(
        "cache/span workload has invalid active or causal positions");
  const auto first_attention =
      std::find_if(plan.model.layers.begin(), plan.model.layers.end(),
                   [](const auto &layer) {
                     return layer.kind == model::LayerKind::kAttention;
                   });
  if (first_attention == plan.model.layers.end())
    return failure("cache/span selected descriptor has no attention geometry");
  const uint64_t h = llama ? 128 : 256;
  const uint64_t qheads = first_attention->query_heads;
  const uint64_t kvheads = first_attention->kv_heads;
  if (plan.model.key_head_width != h || plan.model.value_head_width != h ||
      !qheads || qheads > (span ? 24u : 128u) || !kvheads ||
      kvheads > (span ? 4u : 128u) || qheads % kvheads ||
      (!span && (plan.model.layers.size() > 64 ||
                 scalar(0) >= plan.model.layers.size())))
    return failure("cache/span selected model head or layer geometry differs");
  if ((llama && plan.model.embedding_width != qheads * h) ||
      (!llama && !span && (qheads != 4 || kvheads != 1)))
    return failure(
        "legacy attention descriptor differs from its selected widths");
  for (const auto &layer : plan.model.layers)
    if (layer.kind == model::LayerKind::kAttention &&
        (layer.query_heads != qheads || layer.kv_heads != kvheads))
      return failure(
          "cache/span recipe requires uniform selected attention heads");
  const auto *grid = invocation.command.payload.dispatch.workgroup_count;
  const uint64_t expected_x = span_attention ? (w.token_capacity + 15) / 16
                              : (llama_publish || gemma_publish || span_publish)
                                  ? kvheads
                                  : qheads;
  const uint64_t expected_y = span_attention ? qheads : w.token_capacity;
  if (grid[0] != expected_x || grid[1] != expected_y ||
      (!span && grid[2] != 1) || (span && (!grid[2] || grid[2] > 64)))
    return failure(
        "cache/span selected launch differs from operation geometry");
  const uint64_t pool = (w.context_capacity + 63) / 64 * 64;
  const uint64_t qbytes = w.token_capacity * qheads * h * 4;
  const uint64_t kvbytes = w.token_capacity * kvheads * h * 4;
  const uint64_t plane = pool * kvheads * h * 2;
  PreparedCase prepared;
  prepared.setup_recipe_id = "cache_span_attention/v1/" + name;
  prepared.source_argument_types = types;
  prepared.source_workloads =
      span ? std::vector<uint64_t>{w.token_capacity, grid[2]}
           : std::vector<uint64_t>{w.token_capacity};
  std::vector<uint64_t> sizes(types.size(), 0), aligns(types.size(), 4);
  std::vector<BufferExtent> f32, f16, reads, writes;
  std::vector<std::pair<Ref, std::vector<uint8_t>>> packets;
  const auto view = [&](size_t argument, uint64_t origin, uint64_t bytes,
                        uint64_t alignment, Ref *result) {
    uint64_t end, absolute_end;
    if (!Add(origin, bytes, &end) ||
        !Slice(buffer(argument), origin, bytes, result) ||
        !Add(result->byte_offset, bytes, &absolute_end))
      return false;
    sizes[argument] = std::max(sizes[argument], end);
    aligns[argument] = alignment;
    return true;
  };
  const auto read_f32 = [&](size_t argument, uint64_t origin, uint64_t bytes) {
    Ref ref;
    if (!view(argument, origin, bytes, 4, &ref))
      return false;
    f32.push_back({ref, bytes, 4});
    reads.push_back({ref, bytes, 4});
    return true;
  };
  const auto write = [&](size_t argument, uint64_t origin, uint64_t bytes,
                         uint64_t alignment) {
    Ref ref;
    if (!view(argument, origin, bytes, alignment, &ref))
      return false;
    writes.push_back({ref, bytes, alignment});
    return true;
  };
  const auto packet = [&](size_t argument, uint64_t origin,
                          std::vector<uint8_t> bytes) {
    Ref ref;
    if (!view(argument, origin, bytes.size(), 4, &ref))
      return false;
    reads.push_back({ref, bytes.size(), 4});
    packets.emplace_back(ref, std::move(bytes));
    return true;
  };
  size_t weight_argument = SIZE_MAX;
  uint64_t weight_elements = 0;
  uint64_t cache_origin = 0;
  if (span) {
    QwenSpanControls controls;
    // Native workload bounds cap this metadata packet below20KiB. Complete
    // input-budget admission happens only after ABI/view/parameter preflight.
    if (!BuildQwenSpanControls(w, !span_query, UINT64_MAX, &controls, error))
      return PreparationStatus::kFailure;
    if (!span_query && controls.physical_capacity != pool)
      return failure("Qwen span controls do not match selected physical pool");
    if (!packet(0, 0, std::move(controls.epoch)) ||
        (!span_query && !packet(1, 0, std::move(controls.rows))))
      return failure("Qwen span control binding is too short");
    if (!span_attention) {
      const size_t epsilon = span_query ? 1 : 3;
      if (scalar(epsilon) > UINT32_MAX)
        return failure("Qwen span epsilon exceeds B32 carrier");
      const uint32_t bits = static_cast<uint32_t>(scalar(epsilon));
      float value;
      std::memcpy(&value, &bits, 4);
      if (!std::isfinite(value) || value <= 0)
        return failure("Qwen span epsilon must be finite positive F32");
      weight_argument = span_query ? 3 : 6;
      weight_elements = h;
    }
    if (span_query) {
      if (!read_f32(2, 0, qbytes * 2) || !write(4, 0, qbytes, 4) ||
          !write(5, 0, qbytes, 4))
        return failure("Qwen query/gate projection binding is too short");
    } else {
      cache_origin = scalar(2);
      if (cache_origin % 2)
        return failure("Qwen layer origin is not F16 aligned");
      if (span_publish) {
        if (!read_f32(4, 0, kvbytes) || !read_f32(5, 0, kvbytes) ||
            !write(7, cache_origin, plane * 2, 2))
          return failure("Qwen publication binding is too short");
      } else {
        Ref cache, gate;
        if (!read_f32(3, 0, qbytes) || !view(4, 0, qbytes, 4, &gate) ||
            !view(5, cache_origin, plane * 2, 2, &cache) ||
            !write(6, 0, qbytes, 4))
          return failure("Qwen WMMA binding is too short");
        f16.push_back({cache, plane * 2, 2});
        reads.push_back(f16.back());
        reads.push_back({gate, qbytes, 4});
      }
    }
  } else {
    const uint64_t control_origin = scalar(llama ? 3 : 2);
    const uint64_t map_origin = scalar(llama ? 5 : 3);
    const uint64_t begin = llama ? scalar(4) : 0;
    const uint64_t kind = llama ? scalar(2) : 0;
    if (kind > 1 || (kind == 1 && w.active_tokens != 1) || begin >= 512 ||
        w.active_tokens > 512 - begin || begin > w.token_capacity ||
        w.active_tokens > w.token_capacity - begin || control_origin % 4 ||
        map_origin % 4 || scalar(1) % 2)
      return failure("legacy attention control/origin geometry differs");
    const size_t first_buffer = llama ? 6 : 4;
    const auto words =
        llama ? (kind ? Words({static_cast<uint32_t>(w.history_tokens), 0, 0})
                      : Words({static_cast<uint32_t>(w.active_tokens),
                               static_cast<uint32_t>(w.history_tokens), 0}))
              : Words({static_cast<uint32_t>(w.active_tokens),
                       static_cast<uint32_t>(w.history_tokens), 0, 0,
                       UINT32_MAX});
    if (!packet(first_buffer, control_origin, words))
      return failure("legacy attention control binding is too short");
    std::vector<uint8_t> map;
    for (uint64_t page = 0; page < pool / 64; ++page) {
      const auto word =
          Words({static_cast<uint32_t>((page + 1) % (pool / 64))});
      map.insert(map.end(), word.begin(), word.end());
    }
    if (!packet(first_buffer + 1, map_origin, std::move(map)) ||
        !Add(scalar(1), scalar(0) * 2 * plane, &cache_origin))
      return failure("legacy attention map/cache range overflows");
    const bool publication = llama_publish || gemma_publish;
    const size_t input = first_buffer + (llama_publish ? 3 : 2);
    if (!read_f32(input, 0, publication ? kvbytes : qbytes))
      return failure("legacy projected input binding is too short");
    if (publication) {
      if (!read_f32(input + 1, 0, kvbytes) ||
          !write(input + 2, cache_origin, plane * 2, 2))
        return failure("legacy KV publication binding is too short");
      if (llama_publish) {
        weight_argument = 8;
        weight_elements = 64;
      }
    } else {
      Ref cache;
      if (!view(input + 1, cache_origin, plane * 2, 2, &cache) ||
          !write(input + 2, 0, qbytes, 4))
        return failure("legacy attention cache/output binding is too short");
      f16.push_back({cache, plane * 2, 2});
      reads.push_back(f16.back());
    }
  }
  const Parameter *parameter = nullptr;
  const model::TensorRecord *tensor = nullptr;
  if (weight_argument != SIZE_MAX) {
    Ref ref;
    if (!view(weight_argument, 0, weight_elements * 4, 4, &ref))
      return failure("attention parameter binding is too short");
    reads.push_back({ref, weight_elements * 4, 4});
    if (!llama_publish || ref.role == LOOMC_CMD_BUFFER_ROLE_FIXED) {
      parameter = FindBenchmarkParameter(plan, invocation, ref,
                                         weight_elements * 4, error);
      tensor = parameter ? FindBenchmarkTensor(plan, parameter->key) : nullptr;
      if (!parameter)
        return PreparationStatus::kFailure;
      aligns[weight_argument] =
          std::max<uint64_t>(4, parameter->minimum_alignment);
      if (!tensor || tensor->storage_type != 0 || tensor->rank != 1 ||
          tensor->normalized_extents[0] !=
              static_cast<int64_t>(weight_elements) ||
          tensor->byte_size != weight_elements * 4 ||
          parameter->byte_length != weight_elements * 4 ||
          parameter->byte_offset != ref.byte_offset)
        return failure(
            "attention parameter must be its exact selected rank-one F32 tensor");
    }
  }
  // Fixture policies are distinct even when selected native read views alias.
  // Only common-policy F32/F32 initialization and byte-identical packet overlap
  // are proven consistent here. This is a recipe limit, not a kernel limit.
  std::vector<BufferExtent> packet_spans;
  for (const auto &entry : packets)
    packet_spans.push_back({entry.first, entry.second.size(), 4});
  const auto conflicts_with = [](const BufferExtent &a,
                                 const std::vector<BufferExtent> &others) {
    return std::any_of(others.begin(), others.end(),
                       [&](const auto &b) { return Overlaps(a, b); });
  };
  for (size_t i = 0; i < packets.size(); ++i) {
    if (conflicts_with(packet_spans[i], f32) ||
        conflicts_with(packet_spans[i], f16))
      return unsupported(
          "attention control and numerical input fixture policies overlap");
    for (size_t j = i + 1; j < packets.size(); ++j) {
      if (!Overlaps(packet_spans[i], packet_spans[j]))
        continue;
      const uint64_t begin =
          std::max(packets[i].first.byte_offset, packets[j].first.byte_offset);
      const uint64_t end =
          std::min(packets[i].first.byte_offset + packets[i].second.size(),
                   packets[j].first.byte_offset + packets[j].second.size());
      for (uint64_t offset = begin; offset < end; ++offset)
        if (packets[i].second[offset - packets[i].first.byte_offset] !=
            packets[j].second[offset - packets[j].first.byte_offset])
          return unsupported(
              "attention control fixture packets disagree on aliased bytes");
    }
  }
  for (const auto &extent : f32)
    if (conflicts_with(extent, f16))
      return unsupported("attention F32 and F16 fixture policies overlap");
  if (weight_argument != SIZE_MAX) {
    const BufferExtent weight{buffer(weight_argument), weight_elements * 4, 4};
    if (conflicts_with(weight, packet_spans) || conflicts_with(weight, f32) ||
        conflicts_with(weight, f16))
      return unsupported(
          "attention learned/rotary parameter and synthetic fixture policies overlap");
  }
  for (const auto &read : reads)
    for (const auto &written : writes)
      if (Overlaps(read, written))
        return unsupported(
            "cache/span recipe requires disjoint read/write spans");
  for (size_t i = 0; i < writes.size(); ++i)
    for (size_t j = i + 1; j < writes.size(); ++j)
      if (Overlaps(writes[i], writes[j]))
        return unsupported("cache/span outputs overlap");
  // Unlike ordinary row wrappers, the selected WMMA wrapper explicitly promises
  // no alias among every buffer argument, including its read-only inputs.
  for (size_t i = 0; i < types.size(); ++i)
    if (types[i] == "buffer") {
      if (!sizes[i])
        return failure("cache/span buffer extent was not prepared");
      prepared.buffer_extents.push_back({buffer(i), sizes[i], aligns[i]});
    }
  if (span_attention)
    for (size_t i = 0; i < prepared.buffer_extents.size(); ++i)
      for (size_t j = i + 1; j < prepared.buffer_extents.size(); ++j)
        if (Overlaps(prepared.buffer_extents[i], prepared.buffer_extents[j]))
          return unsupported("selected Qwen WMMA noalias inputs overlap");
  bool budget = false;
  if (!AllocateInputRoots(prepared.buffer_extents, max_bytes, &prepared.roots,
                          error, &budget))
    return budget ? PreparationStatus::kUnsupported
                  : PreparationStatus::kFailure;
  for (const auto &entry : packets)
    if (!SetInputBytes(entry.first, entry.second, &prepared.roots, error))
      return PreparationStatus::kFailure;
  for (const auto &extent : f32)
    if (!F32(extent.binding, extent.byte_length / 4, w.seed, &prepared.roots,
             error))
      return PreparationStatus::kFailure;
  // Qwen's channel-wise gate input already holds sigmoid probabilities.
  if (span_attention) {
    std::vector<uint8_t> gate(qbytes);
    for (uint64_t i = 0; i < qbytes; i += 4) {
      const uint32_t half = 0x3f000000;
      for (unsigned byte = 0; byte < 4; ++byte)
        gate[i + byte] = static_cast<uint8_t>(half >> (byte * 8));
    }
    if (!SetInputBytes(buffer(4), gate, &prepared.roots, error))
      return PreparationStatus::kFailure;
  }
  for (const auto &extent : f16)
    if (!F16(extent.binding, extent.byte_length / 2, w.seed + 3,
             &prepared.roots, error))
      return PreparationStatus::kFailure;
  if (llama_publish) {
    const auto status =
        PrepareRotaryFactors(plan, invocation, 8, true, &prepared, error);
    if (status != PreparationStatus::kReady)
      return status;
  } else if (tensor) {
    const auto ref = buffer(weight_argument);
    if (!SetTensorInput(ref, plan.source_asset, tensor->absolute_offset,
                        weight_elements * 4, &prepared.roots, error))
      return PreparationStatus::kFailure;
    const auto *root = FindInputRoot(prepared.roots, ref);
    const auto begin = ref.byte_offset - root->original_byte_offset;
    for (uint64_t i = 0; i < weight_elements; ++i) {
      uint32_t bits = 0;
      for (unsigned byte = 0; byte < 4; ++byte)
        bits |= static_cast<uint32_t>(root->bytes[begin + i * 4 + byte])
                << (byte * 8);
      float value;
      std::memcpy(&value, &bits, 4);
      if (!std::isfinite(value))
        return failure("Qwen attention gamma contains nonfinite values");
    }
    prepared.weight_sources.push_back(
        {tensor->name, tensor->absolute_offset, weight_elements * 4});
    prepared.representative_weight_spans.push_back(
        {ref, weight_elements * 4,
         std::max<uint64_t>(4, parameter->minimum_alignment)});
  }
  *out = std::move(prepared);
  if (error)
    error->clear();
  return PreparationStatus::kReady;
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
    return PrepareCacheSpanCase(plan, invocation, max_input_bytes, out, error);
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
