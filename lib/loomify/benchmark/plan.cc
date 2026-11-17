// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/plan.h"
#include "lib/loomify/benchmark/inputs.h"
#include "lib/loomify/architectures.h"
#include "loom/util/digest.h"
#include "loomc/source.h"
#include <filesystem>
#include <algorithm>
#include <array>
#include <map>
#include <utility>

namespace hrx::loomify_gguf::benchmark {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
bool Status(loomc_status_t status, std::string *error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok && error) {
    const auto message = loomc_status_message(status);
    error->assign(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Result(loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result))
    return true;
  std::string message = "native benchmark command planning failed";
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto *diagnostic = loomc_result_diagnostic_at(result, i);
      message.append("\n").append(diagnostic->message.data,
                                  diagnostic->message.size);
    }
  return Fail(error, message);
}
void Number(std::string &out, uint64_t n) {
  for (unsigned i = 0; i < 8; ++i)
    out.push_back(char(n >> (8 * i)));
}
void Field(std::string &out, const std::string &value) {
  Number(out, value.size());
  out.append(value);
}
std::string HexDigest(const loom_sha256_t &hash) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (auto byte : hash.bytes) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}
std::string DigestBytes(const void *data, size_t size) {
  loom_sha256_t hash;
  loom_sha256_compute(iree_make_const_byte_span(data, size), &hash);
  return HexDigest(hash);
}
std::string Digest(const std::string &value) {
  return DigestBytes(value.data(), value.size());
}
void Buffer(std::string &out, const loomc_cmd_buffer_ref_t &ref) {
  Number(out, ref.role);
  Number(out, ref.root_index);
  Number(out, ref.byte_offset);
  Number(out, ref.byte_length);
}
void Argument(std::string &out, const loomc_cmd_argument_t &arg) {
  Number(out, arg.kind);
  if (arg.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER)
    Buffer(out, arg.value.buffer);
  else
    Number(out, arg.value.scalar_bits);
}
bool IsDispatch(loomc_cmd_command_kind_t kind) {
  return kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
         kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_STATIC ||
         kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC;
}
loomc_status_t Collect(void *user_data, loomc_request_t *request) {
  auto *requests =
      static_cast<std::vector<std::shared_ptr<loomc_request_t>> *>(user_data);
  requests->emplace_back(request, loomc_request_release);
  return loomc_ok_status();
}
std::string WorkloadKey(const Workload &workload) {
  std::string key;
  Field(key, "loomify-benchmark-workload-v1");
  Field(key, workload.phase);
  Number(key, workload.token_capacity);
  Number(key, workload.active_tokens);
  Number(key, workload.context_capacity);
  Number(key, workload.history_tokens);
  Field(key, workload.platform);
  Number(key, workload.seed);
  Field(key, workload.weight_policy);
  return key;
}
bool Validate(const Workload &workload, std::string *error) {
  if (workload.phase != "prefill" && workload.phase != "decode")
    return Fail(error, "benchmark phase must be prefill or decode");
  if (!workload.active_tokens ||
      workload.active_tokens > workload.token_capacity ||
      workload.history_tokens > workload.context_capacity ||
      workload.active_tokens >
          workload.context_capacity - workload.history_tokens)
    return Fail(error,
                "benchmark active tokens/history exceed declared capacities");
  if (workload.phase == "decode" &&
      (workload.token_capacity != 1 || workload.active_tokens != 1))
    return Fail(error,
                "decode requires token capacity and active tokens of one");
  if (workload.weight_policy != "representative" &&
      workload.weight_policy != "all")
    return Fail(error,
                "benchmark weights policy must be representative or all");
  return true;
}
} // namespace

bool BuildCaseIdentity(const Plan &plan, const Invocation &invocation,
                       const PreparedCase &prepared, CaseIdentity *out,
                       std::string *error) {
  if (!out || invocation.implementation_index >= plan.implementations.size() ||
      prepared.setup_recipe_id.empty() ||
      prepared.measurement_subject.empty() || prepared.reset_policy.empty())
    return Fail(
        error,
        "case identity requires implementation and declared input/reset policy");
  if (invocation.arguments.size() != invocation.command.argument_count)
    return Fail(error,
                "case identity argument count does not match native dispatch");
  using RootKey = std::pair<uint32_t, uint32_t>;
  using Span = std::pair<uint64_t, uint64_t>; // root-relative begin/end
  std::map<RootKey, const PreparedRoot *> by_key;
  for (const auto &root : prepared.roots)
    if (!by_key.emplace(RootKey{root.role, root.root_index}, &root).second)
      return Fail(error, "case identity contains duplicate prepared roots");
  std::map<RootKey, uint32_t> ordinals;
  std::vector<const PreparedRoot *> ordered;
  std::string logical;
  Field(logical, "loomify-benchmark-case-logical-v1");
  Field(logical, plan.workload_id);
  Field(logical, prepared.setup_recipe_id);
  Field(logical, prepared.measurement_subject);
  Field(logical, prepared.reset_policy);
  Field(logical, "fixture-tail-layout-v1");
  Number(logical, kBenchmarkTailGuardBytes);
  auto append_buffer = [&](const loomc_cmd_buffer_ref_t &ref,
                           std::string &destination) -> bool {
    const RootKey key{ref.role, ref.root_index};
    const auto found = by_key.find(key);
    if (found == by_key.end())
      return Fail(error, "case identity is missing a prepared buffer root");
    const auto &root = *found->second;
    if (ref.byte_offset < root.original_byte_offset ||
        ref.byte_offset - root.original_byte_offset > root.bytes.size())
      return Fail(error, "case identity buffer offset exceeds prepared root");
    const auto offset = ref.byte_offset - root.original_byte_offset;
    const uint64_t length = ref.byte_length == UINT64_MAX
                                ? root.bytes.size() - offset
                                : ref.byte_length;
    if (length > root.bytes.size() - offset)
      return Fail(error, "case identity buffer length exceeds prepared root");
    auto [entry, added] = ordinals.emplace(key, ordered.size());
    if (added)
      ordered.push_back(&root);
    Number(destination, ref.role);
    Number(destination, entry->second);
    Number(destination, offset);
    Number(destination, length);
    return true;
  };
  Number(logical, prepared.source_workloads.size());
  for (auto workload : prepared.source_workloads)
    Number(logical, workload);
  Number(logical, prepared.source_argument_types.size());
  for (const auto &type : prepared.source_argument_types)
    Field(logical, type);
  Number(logical, invocation.arguments.size());
  size_t buffer_index = 0;
  for (const auto &argument : invocation.arguments) {
    Number(logical, argument.kind);
    if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
      if (buffer_index >= prepared.buffer_extents.size())
        return Fail(error, "case identity is missing an emitted buffer extent");
      const auto &extent = prepared.buffer_extents[buffer_index++];
      auto binding = argument.value.buffer;
      if (extent.binding.role != binding.role ||
          extent.binding.root_index != binding.root_index ||
          extent.binding.byte_offset != binding.byte_offset ||
          (binding.byte_length != UINT64_MAX &&
           extent.byte_length > binding.byte_length))
        return Fail(
            error,
            "emitted buffer extent does not match selected native argument");
      binding.byte_length = extent.byte_length;
      if (!append_buffer(binding, logical))
        return false;
      Number(logical, extent.minimum_alignment);
    } else {
      Number(logical, argument.value.scalar_bits);
    }
  }
  if (buffer_index != prepared.buffer_extents.size())
    return Fail(error, "case identity has extra emitted buffer extents");
  const auto operation_roots = ordinals;
  std::string physical;
  Number(physical, invocation.command.kind);
  const bool indirect = invocation.command.kind ==
                            LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_STATIC ||
                        invocation.command.kind ==
                            LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC;
  if (invocation.command.kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT) {
    for (auto count : invocation.command.payload.dispatch.workgroup_count)
      Number(physical, count);
  } else if (indirect) {
    std::string count_binding;
    if (!append_buffer(
            invocation.command.payload.dispatch.workgroup_count_buffer,
            count_binding))
      return false;
    Field(physical, count_binding);
    Field(logical, "indirect-count-binding");
    Field(logical, count_binding);
  } else {
    return Fail(error, "case identity requires a selected dispatch");
  }
  Number(logical, prepared.resets.size());
  for (const auto &reset : prepared.resets) {
    if (reset.seed.bytes.empty() ||
        reset.target.byte_length != reset.seed.bytes.size() ||
        !reset.seed.minimum_alignment)
      return Fail(error,
                  "case reset requires an exact target span and seed bytes");
    if (!append_buffer(reset.target, logical))
      return false;
    Number(logical, reset.seed.minimum_alignment);
    Field(logical,
          DigestBytes(reset.seed.bytes.data(), reset.seed.bytes.size()));
  }
  std::map<RootKey, std::vector<Span>> masked_ranges;
  for (const auto &extent : prepared.representative_weight_spans) {
    const auto &ref = extent.binding;
    const RootKey key{ref.role, ref.root_index};
    if (ref.role != LOOMC_CMD_BUFFER_ROLE_FIXED)
      return Fail(error,
                  "representative weights must use immutable fixed roots");
    if (!extent.byte_length || !operation_roots.count(key) ||
        (ref.byte_length != UINT64_MAX && extent.byte_length > ref.byte_length))
      return Fail(error,
                  "representative weight range is not a selected input extent");
    bool selected_extent = false;
    for (const auto &buffer : prepared.buffer_extents)
      if (ref.role == buffer.binding.role &&
          ref.root_index == buffer.binding.root_index &&
          ref.byte_offset >= buffer.binding.byte_offset &&
          ref.byte_offset - buffer.binding.byte_offset <= buffer.byte_length &&
          extent.byte_length <=
              buffer.byte_length -
                  (ref.byte_offset - buffer.binding.byte_offset))
        selected_extent = true;
    if (!selected_extent)
      return Fail(error, "representative weights exceed emitted input extents");
    const auto &root = *by_key.at(key);
    if (ref.byte_offset < root.original_byte_offset)
      return Fail(error, "representative weight range has no prepared storage");
    const auto begin = ref.byte_offset - root.original_byte_offset;
    if (begin > root.bytes.size() ||
        extent.byte_length > root.bytes.size() - begin)
      return Fail(error,
                  "representative weight range exceeds prepared storage");
    const uint64_t end = begin + extent.byte_length;
    auto overlaps = [&](const loomc_cmd_buffer_ref_t &other) {
      return ref.role == other.role && ref.root_index == other.root_index &&
             (ref.byte_offset <= other.byte_offset
                  ? other.byte_offset - ref.byte_offset < extent.byte_length
                  : ref.byte_offset - other.byte_offset < other.byte_length);
    };
    if (indirect &&
        overlaps(invocation.command.payload.dispatch.workgroup_count_buffer))
      return Fail(error, "representative weights overlap launch controls");
    for (const auto &reset : prepared.resets)
      if (overlaps(reset.target))
        return Fail(error,
                    "representative weights overlap mutable reset state");
    auto initialized = root.initialized_ranges;
    std::sort(initialized.begin(), initialized.end());
    uint64_t cursor = begin;
    for (const auto &[offset, length] : initialized) {
      if (offset > cursor)
        break;
      if (offset > root.bytes.size() || length > root.bytes.size() - offset)
        return Fail(error, "prepared weight initialization range is invalid");
      cursor = std::max(cursor, offset + length);
      if (cursor >= end)
        break;
    }
    if (cursor < end)
      return Fail(error, "representative weight bytes are not initialized");
    masked_ranges[key].emplace_back(begin, end);
  }
  std::string mask_policy;
  std::map<uint32_t, std::string> mask_records;
  for (auto &[key, spans] : masked_ranges) {
    std::sort(spans.begin(), spans.end());
    std::vector<Span> merged;
    for (const auto &span : spans)
      if (!merged.empty() && span.first <= merged.back().second)
        merged.back().second = std::max(merged.back().second, span.second);
      else
        merged.push_back(span);
    spans = std::move(merged);
    auto &record = mask_records[operation_roots.at(key)];
    Number(record, key.first);
    Number(record, operation_roots.at(key));
    Number(record, spans.size());
    for (const auto &[begin, end] : spans) {
      Number(record, begin);
      Number(record, end - begin);
    }
  }
  for (const auto &[ordinal, record] : mask_records)
    mask_policy += record;
  for (const auto &[key, root] : by_key)
    if (!ordinals.count(key)) {
      ordinals.emplace(key, ordered.size());
      ordered.push_back(root);
    }
  auto masked_digest = [&](const PreparedRoot &root,
                           const std::vector<Span> &spans) {
    loom_sha256_state_t state;
    loom_sha256_initialize(&state);
    uint64_t cursor = 0;
    static constexpr std::array<uint8_t, 4096> zeros{};
    for (const auto &[begin, end] : spans) {
      loom_sha256_update(&state,
                         iree_make_const_byte_span(root.bytes.data() + cursor,
                                                   begin - cursor));
      uint64_t remaining = end - begin;
      while (remaining) {
        const auto length = std::min<uint64_t>(remaining, zeros.size());
        loom_sha256_update(&state,
                           iree_make_const_byte_span(zeros.data(), length));
        remaining -= length;
      }
      cursor = end;
    }
    loom_sha256_update(&state,
                       iree_make_const_byte_span(root.bytes.data() + cursor,
                                                 root.bytes.size() - cursor));
    loom_sha256_t digest;
    loom_sha256_finalize(&state, &digest);
    return HexDigest(digest);
  };
  std::string inputs;
  Field(inputs, "loomify-benchmark-input-policy-v1");
  Field(inputs, logical);
  Number(inputs, ordered.size());
  std::string representative_inputs = inputs;
  for (const auto *root : ordered) {
    std::string descriptor;
    Number(descriptor, root->role);
    Number(descriptor, root->minimum_alignment);
    Number(descriptor, root->bytes.size());
    const auto digest = DigestBytes(root->bytes.data(), root->bytes.size());
    inputs += descriptor;
    Field(inputs, digest);
    representative_inputs += descriptor;
    const auto spans = masked_ranges.find({root->role, root->root_index});
    Field(representative_inputs, spans == masked_ranges.end()
                                     ? digest
                                     : masked_digest(*root, spans->second));
  }
  auto finalize = [&](const std::string &input_data) {
    CaseIdentity identity;
    identity.input_policy_digest = Digest(input_data);
    auto comparison = logical;
    Field(comparison, identity.input_policy_digest);
    identity.logical_case_key = Digest(comparison);
    std::string concrete;
    Field(concrete, "loomify-benchmark-case-v1");
    Field(concrete, identity.logical_case_key);
    Field(concrete, plan.implementations[invocation.implementation_index]
                        .implementation_id);
    Number(concrete, invocation.request_root_ordinal);
    Field(concrete, physical);
    identity.case_id = Digest(concrete);
    return identity;
  };
  auto identity = finalize(inputs);
  if (!masked_ranges.empty()) {
    std::string representative;
    Field(representative, "loomify-benchmark-weight-group-v1");
    Field(representative, finalize(representative_inputs).case_id);
    Field(representative, mask_policy);
    identity.representative_group_key = Digest(representative);
  }
  *out = std::move(identity);
  if (error)
    error->clear();
  return true;
}

bool BuildBenchmarkPlan(const std::string &gguf_path, const Workload &workload,
                        const std::string &library_root,
                        loomc_context_t *context, loomc_workspace_t *workspace,
                        Plan *out, std::string *error) {
  if (!out || !context || !workspace)
    return Fail(error, "benchmark plan requires context, workspace and output");
  if (!Validate(workload, error))
    return false;
  std::error_code ec;
  const auto source_asset = std::filesystem::canonical(gguf_path, ec);
  if (ec)
    return Fail(error, "cannot resolve benchmark GGUF: " + ec.message());
  std::unique_ptr<GgufModelBuilder> builder;
  if (!GgufModelBuilder::Open(source_asset.string(), &builder, error))
    return false;
  GgufBuildOptions options;
  options.library_root = library_root;
  options.prefill_capacity = workload.token_capacity;
  options.context_capacity = workload.context_capacity;
  options.platform = workload.platform;
  std::unique_ptr<GgufConfiguration> configuration;
  std::unique_ptr<BuiltGgufModule> module;
  if (!builder->ConfigureNumerical(options, &configuration, error) ||
      !configuration->BuildModule(context, workspace, &module, error))
    return false;
  const auto *architecture = SelectArchitecture(builder->descriptor().family);
  if (!architecture)
    return Fail(error,
                "benchmark model has no selected numerical architecture");
  std::vector<loomc_string_view_t> roots;
  for (const auto *root : architecture->roots)
    roots.push_back(loomc_make_cstring_view(root));
  std::vector<std::shared_ptr<loomc_request_t>> requests;
  loomc_cmd_program_module_options_t native_options{};
  native_options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  native_options.structure_size = sizeof(native_options);
  native_options.provider_name = loomc_make_cstring_view("loomify-benchmark");
  native_options.root_symbols = roots.data();
  native_options.root_symbol_count = roots.size();
  native_options.request_sink = {Collect, &requests};
  loomc_product_t *raw_product = nullptr;
  loomc_result_t *raw_result = nullptr;
  const auto status = loomc_cmd_program_product_build_module(
      workspace, module->module(), &native_options, loomc_allocator_system(),
      &raw_product, &raw_result);
  std::unique_ptr<loomc_product_t, decltype(&loomc_product_release)> product(
      raw_product, loomc_product_release);
  std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)> result(
      raw_result, loomc_result_release);
  if (!Status(status, error) || !Result(result.get(), error))
    return false;
  if (!product)
    return Fail(error, "native benchmark planning returned no product");
  // Provisional parent bindings are committed only after planning succeeded.
  Plan prepared;
  prepared.workload = workload;
  prepared.workload_id = Digest(WorkloadKey(workload));
  prepared.model = builder->descriptor();
  prepared.publication_tokens = builder->publication_tokens();
  prepared.source_asset = source_asset.string();
  using BoundRoot = std::pair<uint32_t, uint32_t>;
  std::map<uint32_t, BoundRoot> requirements;
  std::map<std::string, uint32_t> implementation_ids;
  for (const auto &request : requests) {
    const auto bytes =
        loomc_source_contents(loomc_request_source(request.get()));
    std::string key;
    Field(key, "loomify-benchmark-implementation-v1");
    Field(key, workload.platform);
    Number(key, bytes.data_length);
    key.append(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
    Implementation implementation;
    implementation.request = request;
    const auto root_count = loomc_request_root_count(request.get());
    for (size_t i = 0; i < root_count; ++i) {
      loomc_request_root_t root;
      if (!loomc_request_root_at(request.get(), i, &root))
        return Fail(error, "native benchmark request root is unavailable");
      implementation.roots.push_back(root);
      Number(key, root.module_ordinal);
      Number(key, root.symbol_ordinal);
    }
    implementation.implementation_id = Digest(key);
    auto [id, inserted] = implementation_ids.emplace(
        implementation.implementation_id, prepared.implementations.size());
    const uint32_t index = id->second;
    if (inserted)
      prepared.implementations.push_back(std::move(implementation));
    for (size_t i = 0; i < loomc_request_binding_count(request.get()); ++i) {
      loomc_request_binding_t binding;
      if (!loomc_request_binding_at(request.get(), i, &binding) ||
          binding.root_ordinal >= root_count)
        return Fail(error, "native benchmark request binding is invalid");
      if (!requirements
               .emplace(binding.requirement_ordinal,
                        BoundRoot{index, binding.root_ordinal})
               .second)
        return Fail(error,
                    "native benchmark requirement has multiple bindings");
    }
  }
  std::string manifest_key;
  Field(manifest_key, "loomify-benchmark-selected-plan-v1");
  Field(manifest_key, prepared.workload_id);
  const auto program_count =
      loomc_cmd_program_product_program_count(product.get());
  for (size_t p = 0; p < program_count; ++p) {
    loomc_cmd_program_t native_program;
    Program program;
    if (!loomc_cmd_program_product_program_at(product.get(), p,
                                              &native_program) ||
        !loomc_cmd_program_product_layout(product.get(), p, &program.layout))
      return Fail(error, "native benchmark program layout is unavailable");
    program.root_symbol.assign(native_program.symbol.data,
                               native_program.symbol.size);
    Field(manifest_key, program.root_symbol);
    Number(manifest_key, program.layout.fixed_buffer_count);
    Number(manifest_key, program.layout.rebindable_binding_count);
    Number(manifest_key, program.layout.executable_count);
    Number(manifest_key, program.layout.entry_count);
    for (const auto &storage :
         {program.layout.transient, program.layout.launch_counts}) {
      Number(manifest_key, storage.binding_index);
      Number(manifest_key, storage.required_byte_length);
      Number(manifest_key, storage.minimum_alignment);
    }
    Number(manifest_key, program.layout.launch_binding_count);
    Number(manifest_key, program.layout.parameter_root_count);
    Number(manifest_key, program.layout.parameter_count);
    Number(manifest_key, program.layout.command_count);
    for (size_t i = 0; i < program.layout.launch_binding_count; ++i) {
      loomc_cmd_launch_binding_t row;
      if (!loomc_cmd_program_product_launch_binding_at(product.get(), p, i,
                                                       &row))
        return Fail(error, "native benchmark launch binding is unavailable");
      program.launch_bindings.push_back(row);
      Number(manifest_key, row.role);
      Number(manifest_key, row.root_index);
    }
    for (size_t i = 0; i < program.layout.parameter_root_count; ++i) {
      loomc_cmd_parameter_root_t row;
      if (!loomc_cmd_program_product_parameter_root_at(product.get(), p, i,
                                                       &row))
        return Fail(error, "native benchmark parameter root is unavailable");
      program.parameter_roots.push_back(row);
      Number(manifest_key, row.fixed_buffer_index);
      Number(manifest_key, row.required_byte_length);
      Number(manifest_key, row.minimum_alignment);
    }
    for (size_t i = 0; i < program.layout.parameter_count; ++i) {
      loomc_cmd_parameter_t row;
      if (!loomc_cmd_program_product_parameter_at(product.get(), p, i, &row))
        return Fail(error, "native benchmark parameter is unavailable");
      Parameter parameter{std::string(row.key.data, row.key.size),
                          row.fixed_buffer_index, row.byte_offset,
                          row.byte_length, row.minimum_alignment};
      Field(manifest_key, parameter.key);
      Number(manifest_key, parameter.fixed_buffer_index);
      Number(manifest_key, parameter.byte_offset);
      Number(manifest_key, parameter.byte_length);
      Number(manifest_key, parameter.minimum_alignment);
      program.parameters.push_back(std::move(parameter));
    }
    for (size_t c = 0; c < program.layout.command_count; ++c) {
      loomc_cmd_command_t command;
      if (!loomc_cmd_program_product_command_at(product.get(), p, c, &command))
        return Fail(error, "native benchmark command is unavailable");
      program.commands.push_back(command);
      Number(manifest_key, command.kind);
      Number(manifest_key, command.begins_barrier_wave);
      if (!IsDispatch(command.kind)) {
        if (command.kind == LOOMC_CMD_COMMAND_KIND_COPY) {
          Buffer(manifest_key, command.payload.copy.source);
          Buffer(manifest_key, command.payload.copy.target);
        } else if (command.kind == LOOMC_CMD_COMMAND_KIND_FILL) {
          Buffer(manifest_key, command.payload.fill.target);
          Number(manifest_key, command.payload.fill.pattern);
          Number(manifest_key, command.payload.fill.pattern_length);
        } else if (command.kind == LOOMC_CMD_COMMAND_KIND_TIMESTAMP) {
          Buffer(manifest_key, command.payload.timestamp.target);
        }
        continue;
      }
      const auto binding =
          requirements.find(command.payload.dispatch.requirement_ordinal);
      if (binding == requirements.end())
        return Fail(error,
                    "selected dispatch lacks a published kernel request");
      Invocation invocation;
      invocation.program_ordinal = p;
      invocation.command_ordinal = c;
      invocation.invocation_id =
          "p" + std::to_string(p) + "/c" + std::to_string(c);
      invocation.implementation_index = binding->second.first;
      invocation.request_root_ordinal = binding->second.second;
      invocation.command = command;
      loomc_cmd_entry_requirement_t entry;
      if (!loomc_cmd_program_product_entry_requirement_at(
              product.get(), command.payload.dispatch.requirement_ordinal,
              &entry))
        return Fail(error, "native benchmark kernel symbol is unavailable");
      invocation.kernel_symbol.assign(entry.symbol.data, entry.symbol.size);
      Field(manifest_key, invocation.kernel_symbol);
      Field(manifest_key,
            prepared.implementations[invocation.implementation_index]
                .implementation_id);
      Number(manifest_key, invocation.request_root_ordinal);
      if (command.kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT)
        for (auto count : command.payload.dispatch.workgroup_count)
          Number(manifest_key, count);
      else
        Buffer(manifest_key, command.payload.dispatch.workgroup_count_buffer);
      Number(manifest_key, command.argument_count);
      for (size_t a = 0; a < command.argument_count; ++a) {
        loomc_cmd_argument_t argument;
        if (!loomc_cmd_program_product_argument_at(product.get(), p, c, a,
                                                   &argument))
          return Fail(error, "native benchmark argument is unavailable");
        invocation.arguments.push_back(argument);
        Argument(manifest_key, argument);
      }
      prepared.invocations.push_back(std::move(invocation));
    }
    prepared.programs.push_back(std::move(program));
  }
  if (prepared.invocations.empty())
    return Fail(error, "selected numerical program contains no dispatches");
  prepared.selection_id = Digest(manifest_key);
  *out = std::move(prepared);
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf::benchmark
