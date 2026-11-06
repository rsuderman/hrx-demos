// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/qwen/model_export.h"
#include "lib/qwen/program.h"
#include <array>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
bool Check(loomc_status_t status, std::string *error) {
  bool ok = loomc_status_is_ok(status);
  if (!ok && error) {
    auto m = loomc_status_message(status);
    error->assign(m.data, m.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Result(loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (error) {
    error->clear();
    if (!result)
      *error = "native operation returned no diagnostic result";
    else
      for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
        auto d = loomc_result_diagnostic_at(result, i);
        error->append(d->message.data, d->message.size);
        error->push_back('\n');
      }
  }
  return false;
}
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
} // namespace
bool BuildQwenServeModule(const model::ModelDescriptor &descriptor,
                          const ModelPackageOptions &o,
                          loomc_context_t *context,
                          loomc_workspace_t *workspace, loomc_module_t **out,
                          dispatch::QwenNativeGeometry *out_geometry,
                          std::string *error) {
  if (out)
    *out = nullptr;
  if (!out || !out_geometry || !context || !workspace ||
      descriptor.family != "qwen35" || !o.prefill_capacity ||
      o.prefill_capacity > 512 || o.context_capacity < o.prefill_capacity ||
      o.context_capacity > 262144)
    return Fail(error, "invalid bounded Qwen serving build configuration");
  dispatch::QwenNativeOptions native;
  native.token_capacity = o.prefill_capacity;
  native.cache_capacity = o.context_capacity;
  native.pool_capacity = ((o.context_capacity + 63) / 64) * 64;
  native.span_capacity = native.selected_output_capacity = native.row_capacity =
      1;
  const auto registry = dispatch::Registry::Qwen08B();
  dispatch::QwenNativeGeometry geometry;
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  auto status = dispatch::LoadQwenNative(
      context, workspace, registry, qwen::QwenProgramDispatches(), descriptor,
      native, o.source_root, &geometry, &definitions, &result);
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)>
      definitions_owner(definitions, &loomc_module_release);
  std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)> load_result(
      result, &loomc_result_release);
  if (!Check(status, error) || !Result(result, error))
    return false;
  std::unique_ptr<ModuleBuilder> module;
  if (!Check(ModuleBuilder::Create(context, workspace, definitions, &module),
             error))
    return false;
  qwen::QwenProgramSettings settings;
  settings.native = native;
  settings.source_asset = o.gguf;
  qwen::QwenProgram program;
  if (!qwen::DefineQwenProgram(descriptor, settings, registry, *module,
                               kQwenEpochRoot, &program, error))
    return false;
  // Match native model requirements to the existing serving allocations. The
  // fixed checkpoint root is supplied by Serve's named parameter bank.
  const auto blocks = (o.context_capacity + 63) / 64;
  const std::array<const char *, 7> roles = {
      "parameters",     "residual",  "epoch",     "rows",
      "resident_arena", "input_ids", "output_ids"};
  const std::array<uint64_t, 7> capacities = {0,
                                              geometry.residual_bytes,
                                              4192,
                                              geometry.map_origin + 4 * blocks,
                                              geometry.private_row_stride +
                                                  geometry.kv_pool_bytes,
                                              4096,
                                              64};
  if (program.buffers.size() != roles.size())
    return Fail(error, "Qwen model must expose exactly seven serving buffers");
  for (size_t i = 0; i < roles.size(); ++i)
    if (program.buffers[i].name != roles[i] ||
        (i && program.buffers[i].minimum_bytes > capacities[i]))
      return Fail(
          error,
          "Qwen model buffer role/footprint differs from serving storage");
  result = nullptr;
  status = module->LinkAndVerify({program.program}, out, &result);
  std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)> link_result(
      result, &loomc_result_release);
  if (!Check(status, error) || !Result(result, error)) {
    if (*out) {
      loomc_module_release(*out);
      *out = nullptr;
    }
    return false;
  }
  *out_geometry = geometry;
  return true;
}
} // namespace hrx::loomify_gguf
