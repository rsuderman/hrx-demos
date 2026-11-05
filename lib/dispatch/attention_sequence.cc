// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/attention_sequence.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include "loomc/source.h"
#include <algorithm>
#include <memory>
#include <string_view>
namespace hrx::dispatch {
using namespace loomify_gguf;
namespace {
loomc_status_t Error(const char *message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
}
bool Valid(const AttentionNativeConfig &c) {
  const auto physical = c.pool_capacity ? c.pool_capacity : c.cache_capacity;
  return c.layer_count && c.cache_capacity && c.cache_capacity <= 131072 &&
         c.pool_capacity <= 4194304 && c.pool_capacity % 64 == 0 &&
         c.layer_count <= uint64_t(INT64_MAX) / (physical * 1024 * 4);
}
} // namespace
std::vector<NativeSource> AttentionSequenceSources() {
  std::vector<NativeSource> entries;
  const std::string prefix = "lib/kernels/llama/";
  for (const char *symbol :
       {"llama32_rotary_query", "llama32_publish_kv", "llama32_attention_rows",
        "llama32_attention_residual"}) {
    // Load inspects only native kind/declaration, source closure and roots.
    NativeSource e{};
    e.kind = Kind::Kernel;
    e.symbol = e.interface_symbol = symbol;
    e.interface_file = prefix + "attention/attention_native_interface.loom";
    for (const char *file :
         {"attention/attention_prepare.loom", "attention/attention_native.loom",
          "attention/rotary.loom", "attention/attention_core.loom",
          "attention/attention_map.loom",
          "attention/attention_materialized.loom", "rows.loom", "config.loom",
          "output/head_count.loom"})
      e.provider_files.push_back(prefix + file);
    e.required_config = {
        "llama.model.layer_count",      "llama.model.query_heads",
        "llama.model.kv_heads",         "llama.model.head_size",
        "llama.model.kv_size",          "llama.attention.cache_capacity",
        "llama.attention.pool_capacity"};
    entries.push_back(std::move(e));
  }
  return entries;
}
loomc_status_t LoadAttentionSequence(loomc_context_t *context,
                                     loomc_workspace_t *workspace,
                                     const std::string &root,
                                     const std::string &capacity_config,
                                     loomc_module_t **out,
                                     loomc_result_t **result) {
  auto entries = AttentionSequenceSources();
  for (auto &entry : entries) {
    entry.provider_files.push_back(
        "lib/kernels/llama/attention/attention_config.loom");
    entry.provider_files.push_back(capacity_config);
  }
  return LoadNative(context, workspace, entries, root, out, result);
}
loomc_status_t CreateAttentionConfiguration(loomc_context_t *context,
                                            loomc_workspace_t *workspace,
                                            const AttentionNativeConfig &c,
                                            loomc_module_t **out,
                                            loomc_result_t **result) {
  if (out)
    *out = nullptr;
  if (result)
    *result = nullptr;
  if (!context || !workspace || !out || !result || !Valid(c))
    return Error("invalid native attention configuration");
  std::string text;
  auto define = [&](const char *name, uint64_t value) {
    text += "config.def @" + std::string(name) + " = " + std::to_string(value) +
            " : index\n";
  };
  define("llama.model.layer_count", c.layer_count);
  define("llama.model.query_heads", 24);
  define("llama.model.kv_heads", 8);
  define("llama.model.head_size", 128);
  define("llama.model.kv_size", 1024);
  define("llama.attention.cache_capacity", c.cache_capacity);
  define("llama.attention.pool_capacity", c.pool_capacity);
  loomc_source_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  options.identifier = loomc_make_cstring_view("llama-configuration.loom");
  options.contents = loomc_make_byte_span(text.data(), text.size());
  loomc_source_t *source = nullptr;
  auto status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  if (!loomc_status_is_ok(status))
    return status;
  status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), out,
      result);
  loomc_source_release(source);
  return status;
}
loomc_status_t
LoadAttentionSequence(loomc_context_t *context, loomc_workspace_t *workspace,
                      const std::string &root, const AttentionNativeConfig &c,
                      loomc_module_t **out, loomc_result_t **result) {
  if (out)
    *out = nullptr;
  if (result)
    *result = nullptr;
  if (!out || !result)
    return Error("native attention load requires outputs");
  loomc_module_t *configuration = nullptr;
  auto status = CreateAttentionConfiguration(context, workspace, c,
                                             &configuration, result);
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> owned(
      configuration, loomc_module_release);
  if (!loomc_status_is_ok(status) || !*result ||
      !loomc_result_succeeded(*result))
    return status;
  loomc_result_release(*result);
  *result = nullptr;
  return LoadNative(context, workspace, AttentionSequenceSources(), root,
                    configuration, out, result);
}
loomc_status_t
LoadLlamaNative(loomc_context_t *context, loomc_workspace_t *workspace,
                const Registry &registry, const std::vector<Id> &operations,
                const std::string &root, const AttentionNativeConfig &config,
                loomc_module_t **out, loomc_result_t **result) {
  if (out)
    *out = nullptr;
  if (result)
    *result = nullptr;
  if (!out || !result)
    return Error("native Llama load requires outputs");
  auto sources = AttentionSequenceSources();
  Registry validation;
  std::string error;
  std::vector<Id> seen;
  for (auto id : operations) {
    if (std::find(seen.begin(), seen.end(), id) != seen.end())
      continue;
    seen.push_back(id);
    const auto entries = registry.Implementations(id);
    if (entries.empty())
      return Error("missing dispatch registration");
    for (const auto &entry : entries) {
      if (!validation.Add(entry, &error))
        return Error(error.c_str());
      sources.push_back({entry.symbol, entry.kind, entry.interface_file,
                         entry.provider_files, entry.required_config,
                         entry.interface_symbol,
                         entry.builder.required_definitions});
    }
  }
  loomc_module_t *configuration = nullptr;
  auto status = CreateAttentionConfiguration(context, workspace, config,
                                             &configuration, result);
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> owned(
      configuration, loomc_module_release);
  if (!loomc_status_is_ok(status) || !*result ||
      !loomc_result_succeeded(*result))
    return status;
  loomc_result_release(*result);
  *result = nullptr;
  return LoadNative(context, workspace, sources, root, configuration, out,
                    result);
}
loomc_status_t CheckAttentionSequence(ModuleBuilder &module,
                                      CommandProgramBuilder &program,
                                      const AttentionSequenceOptions &o,
                                      AttentionNativeConfig *configuration) {
  if (!o.row_capacity || o.row_capacity > 512 ||
      o.token_begin >= o.row_capacity ||
      o.control_origin > uint64_t(INT64_MAX) - 12 || o.map_origin > INT64_MAX ||
      o.cache_base > INT64_MAX ||
      (o.phase != operations::Phase::Prefill &&
       o.phase != operations::Phase::Decode) ||
      (o.phase == operations::Phase::Decode &&
       (o.row_capacity != 1 || o.token_begin != 0)))
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "invalid native Llama attention sequence options"));
#define TRY(call)                                                              \
  do {                                                                         \
    auto status = (call);                                                      \
    if (!loomc_status_is_ok(status))                                           \
      return status;                                                           \
  } while (0)
  // The loaded definition set is the single configuration authority. Check it
  // before allocating/staging and use its actual count, never a host literal28.
  TRY(program.InspectNative({}, [&](const loom_module_t *native,
                                    const std::vector<loom_value_id_t> &) {
    AttentionNativeConfig c{};
    c.layer_count = 0;
    unsigned found = 0;
    bool geometry = true;
    for (size_t i = 0; i < native->symbols.count; ++i) {
      const auto &symbol = native->symbols.entries[i];
      if (symbol.name_id == LOOM_STRING_ID_INVALID || !symbol.defining_op ||
          !loom_config_def_isa(symbol.defining_op))
        continue;
      auto n = loom_string_table_get(&native->strings, symbol.name_id);
      std::string_view name(n.data, n.size);
      auto attr = loom_config_def_value(symbol.defining_op);
      auto read = [&](uint64_t *out, unsigned bit) {
        if (attr.kind != LOOM_ATTR_I64 || attr.i64 < 0)
          geometry = false;
        else
          *out = uint64_t(attr.i64);
        found |= bit;
      };
      if (name == "llama.model.layer_count")
        read(&c.layer_count, 1);
      if (name == "llama.attention.cache_capacity")
        read(&c.cache_capacity, 2);
      if (name == "llama.attention.pool_capacity")
        read(&c.pool_capacity, 4);
      unsigned bit = 8;
      for (auto pair :
           {std::pair<const char *, int64_t>{"llama.model.query_heads", 24},
            {"llama.model.kv_heads", 8},
            {"llama.model.head_size", 128},
            {"llama.model.kv_size", 1024},
            {"llama.model.hidden_size", 3072}}) {
        if (name == pair.first) {
          geometry &= attr.kind == LOOM_ATTR_I64 && attr.i64 == pair.second;
          found |= bit;
        }
        bit <<= 1;
      }
    }
    const bool valid = geometry && found == 255 && Valid(c);
    const auto physical = c.pool_capacity ? c.pool_capacity : c.cache_capacity;
    const bool offsets =
        valid &&
        o.cache_base <= uint64_t(INT64_MAX) - c.layer_count * physical * 4096 &&
        (!c.pool_capacity ||
         o.map_origin <=
             uint64_t(INT64_MAX) - ((c.cache_capacity + 63) / 64) * 4);
    if (valid && offsets && o.layer < c.layer_count && configuration)
      *configuration = c;
    return valid && offsets && o.layer < c.layer_count
               ? iree_ok_status()
               : iree_make_status(
                     IREE_STATUS_INVALID_ARGUMENT,
                     "attention options disagree with loaded configuration");
  }));
  const auto index = module.Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto offset = module.Scalar(LOOM_SCALAR_TYPE_OFFSET);
  const auto buffer = module.Buffer();
  KernelRef rotary, publish, causal;
  TRY(module.Kernel("llama32_rotary_query", &rotary));
  TRY(module.Kernel("llama32_publish_kv", &publish));
  TRY(module.Kernel("llama32_attention_rows", &causal));
  TRY(module.CheckKernelInterface(
      rotary, {index}, {index, offset, index, buffer, buffer, buffer, buffer}));
  TRY(module.CheckKernelInterface(publish, {index},
                                  {index, offset, index, offset, index, offset,
                                   buffer, buffer, buffer, buffer, buffer,
                                   buffer}));
  TRY(module.CheckKernelInterface(causal, {index},
                                  {index, offset, index, offset, index, offset,
                                   buffer, buffer, buffer, buffer, buffer}));
  return loomc_ok_status();
#undef TRY
}
loomc_status_t BuildAttentionSequence(ModuleBuilder &module,
                                      CommandProgramBuilder &program,
                                      const AttentionSequenceOptions &o,
                                      const AttentionSequenceBindings &b,
                                      Value *attention) {
  if (!attention)
    return Error("attention output is required");
#define TRY(call)                                                              \
  do {                                                                         \
    auto status = (call);                                                      \
    if (!loomc_status_is_ok(status))                                           \
      return status;                                                           \
  } while (0)
  TRY(CheckAttentionSequence(module, program, o, nullptr));
  KernelRef rotary, publish, causal;
  TRY(module.Kernel("llama32_rotary_query", &rotary));
  TRY(module.Kernel("llama32_publish_kv", &publish));
  TRY(module.Kernel("llama32_attention_rows", &causal));
  CommandValues values(program);
  Value rows, layer, kind, control_origin, begin, cache_base, map_origin;
  TRY(values.Constant(o.row_capacity, LOOM_SCALAR_TYPE_INDEX, &rows));
  TRY(values.Constant(o.layer, LOOM_SCALAR_TYPE_INDEX, &layer));
  TRY(values.Constant(o.phase == operations::Phase::Decode,
                      LOOM_SCALAR_TYPE_INDEX, &kind));
  TRY(values.Constant(o.control_origin, LOOM_SCALAR_TYPE_OFFSET,
                      &control_origin));
  TRY(values.Constant(o.token_begin, LOOM_SCALAR_TYPE_INDEX, &begin));
  TRY(values.Constant(o.cache_base, LOOM_SCALAR_TYPE_OFFSET, &cache_base));
  TRY(values.Constant(o.map_origin, LOOM_SCALAR_TYPE_OFFSET, &map_origin));
  Value size, query, output;
  TRY(values.Constant(o.row_capacity * 3072 * 4, LOOM_SCALAR_TYPE_OFFSET,
                      &size));
  TRY(values.Allocate(size, 256, &query));
  TRY(values.Allocate(size, 256, &output));
  TRY(program.Invoke(rotary, {rows},
                     {kind, control_origin, begin, b.control, b.factors,
                      b.query_projection, query}));
  TRY(program.Invoke(publish, {rows},
                     {layer, cache_base, kind, control_origin, begin,
                      map_origin, b.control, b.map, b.factors, b.key_projection,
                      b.value_projection, b.cache}));
  TRY(program.Invoke(causal, {rows},
                     {layer, cache_base, kind, control_origin, begin,
                      map_origin, b.control, b.map, query, b.cache, output}));
  *attention = output;
  return loomc_ok_status();
#undef TRY
}
} // namespace hrx::dispatch
