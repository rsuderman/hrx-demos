// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Finite observer for one command comparison. Execution remains in the public
// HAL/VM profiles; only selected post-execute byte spans are copied here.
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/heap_pool.h"
#include "iree/io/stdio_stream.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loom/tooling/execution/hal/scenario_profile.h"
#include "loom/tooling/target/vm/testbench.h"
#include "loom/tooling/testbench/scenario/executor.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/link.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/configured.h"

namespace {
namespace fs = std::filesystem;
iree_status_t Invalid(const char *message) {
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message);
}
struct Options {
  std::string input, library, scenario, output;
  std::vector<size_t> arguments;
};
iree_status_t Parse(int argc, char **argv, Options &out) {
  std::vector<char *> device_args{argv[0]};
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("--device=", 0) == 0) {
      if (arg != "--device=amdgpu" || device_args.size() != 1)
        return Invalid("exactly one --device=amdgpu is required");
      device_args.push_back(argv[i]);
      continue;
    }
    const auto equal = arg.find('=');
    std::string key = arg.substr(0, equal);
    std::string value = equal == std::string::npos ? "" : arg.substr(equal + 1);
    std::string *slot = nullptr;
    if (key == "--library")
      slot = &out.library;
    else if (key == "--case")
      slot = &out.scenario;
    else if (key == "--capture-output-root")
      slot = &out.output;
    else if (key == "--capture-argument") {
      size_t index = 0;
      const auto parsed =
          std::from_chars(value.data(), value.data() + value.size(), index);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() || index == 0)
        return Invalid(
            "capture index must be a positive integer; arg0 is immutable parameters");
      for (size_t old : out.arguments)
        if (old == index)
          return Invalid("duplicate capture index");
      out.arguments.push_back(index);
      continue;
    } else if (arg.rfind("--", 0) == 0 || !out.input.empty()) {
      return Invalid("unknown option or multiple inputs");
    } else {
      out.input = arg;
      continue;
    }
    if (!slot->empty() || value.empty())
      return Invalid("missing or duplicate option");
    *slot = value;
  }
  if (out.input.empty() || out.library.empty() || out.scenario.empty() ||
      out.output.empty() || out.arguments.empty() || device_args.size() != 2 ||
      !fs::path(out.output).is_absolute())
    return Invalid(
        "explicit input/library/case/device/absolute output and captures required");
  if (out.scenario[0] == '@')
    out.scenario.erase(0, 1);
  int flag_argc = static_cast<int>(device_args.size());
  device_args.push_back(nullptr);
  char **flag_argv = device_args.data();
  return iree_flags_parse(IREE_FLAGS_PARSE_MODE_DEFAULT, &flag_argc,
                          &flag_argv);
}
iree_status_t Result(iree_status_t status, loomc_result_t *result) {
  bool accepted = false;
  if (iree_status_is_ok(status))
    status = loom_tooling_cli_print_loomc_result(stderr, result, &accepted);
  loomc_result_release(result);
  if (iree_status_is_ok(status) && !accepted)
    return Invalid("compiler rejected input");
  return status;
}
struct Observer {
  const Options *options;
  loom_testbench_prepared_product_t original{};
  bool called = false;
  static void Destroy(void *data) {
    auto *self = static_cast<Observer *>(data);
    if (self->original.destroy)
      self->original.destroy(self->original.user_data);
    delete self;
  }
  static iree_status_t
  Execute(void *data, const loom_testbench_invocation_plan_t *invocation,
          iree_host_size_t count, loom_testbench_product_call_t *calls) {
    auto &self = *static_cast<Observer *>(data);
    if (self.called || count != 1)
      return Invalid("one physical target call required");
    self.called = true;
    for (size_t index : self.options->arguments) {
      if (index >= invocation->input_count)
        return Invalid("capture index out of range");
      const auto &value = calls[0].arguments[index];
      const auto &binding = value.buffer;
      if (value.kind != LOOM_TESTBENCH_VALUE_KIND_BUFFER || !binding.buffer ||
          !binding.buffer_view ||
          iree_hal_buffer_view_shape_rank(binding.buffer_view) != 1 ||
          iree_hal_buffer_view_element_type(binding.buffer_view) !=
              IREE_HAL_ELEMENT_TYPE_SINT_8 ||
          iree_hal_buffer_view_encoding_type(binding.buffer_view) !=
              IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR)
        return Invalid("capture requires a dense rank-one i8 buffer argument");
    }
    IREE_RETURN_IF_ERROR(self.original.vtable->execute(
        self.original.user_data, invocation, count, calls));
    // The executor owns host-pool value graphs. HAL stages and synchronously
    // reads back into those graphs before returning. No device pointer escapes.
    for (size_t index : self.options->arguments) {
      const auto &binding = calls[0].arguments[index].buffer;
      iree_hal_buffer_mapping_t mapping{};
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
          binding.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
          IREE_HAL_MEMORY_ACCESS_READ, 0, binding.byte_offset,
          binding.byte_length, &mapping));
      iree_status_t status = iree_hal_buffer_mapping_invalidate_range(
          &mapping, 0, binding.byte_length);
      if (iree_status_is_ok(status)) {
        std::string header =
            "{'descr': '|i1', 'fortran_order': False, 'shape': (" +
            std::to_string(binding.byte_length) + ",), }";
        header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
        header += '\n';
        const unsigned char prefix[] = {
            0x93,
            'N',
            'U',
            'M',
            'P',
            'Y',
            1,
            0,
            static_cast<unsigned char>(header.size() & 255),
            static_cast<unsigned char>(header.size() >> 8)};
        fs::path path = fs::path(self.options->output) /
                        ("product0_call0_after_arg" + std::to_string(index) +
                         ".npy.pending");
        FILE *file = std::fopen(path.c_str(), "wbx");
        bool good = file != nullptr;
        if (file) {
          good =
              std::fwrite(prefix, 1, sizeof(prefix), file) == sizeof(prefix) &&
              std::fwrite(header.data(), 1, header.size(), file) ==
                  header.size() &&
              std::fwrite(mapping.contents.data, 1, binding.byte_length,
                          file) == binding.byte_length;
          if (std::fclose(file) != 0)
            good = false;
        }
        if (!good)
          status =
              iree_make_status(IREE_STATUS_DATA_LOSS, "capture write failed");
      }
      status = iree_status_join(status, iree_hal_buffer_unmap_range(&mapping));
      IREE_RETURN_IF_ERROR(status);
    }
    return iree_ok_status();
  }
};
const loom_testbench_product_vtable_t kObserverVtable{Observer::Execute,
                                                      nullptr};
struct Resources {
  iree_allocator_t allocator = iree_allocator_system();
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  loomc_compiler_t *compiler = nullptr;
  loomc_linker_t *linker = nullptr;
  loomc_module_t *input = nullptr, *library = nullptr, *linked = nullptr;
  iree_arena_block_pool_t blocks{};
  iree_arena_allocator_t arena{};
  loom_vm_testbench_t vm{};
  loom_run_hal_testbench_context_t hal{};
  loom_testbench_compilation_t compilation{};
  loom_run_hal_testbench_scenario_profile_t profile{};
  loom_run_hal_target_route_t route{};
  iree_hal_pool_t *pool = nullptr;
  loom_testbench_scenario_configuration_values_t configuration{};
  loom_testbench_prepared_scenario_configuration_t prepared{};
  loom_testbench_scenario_trial_executor_t executor{};
  bool vm_initialized = false, hal_initialized = false,
       values_initialized = false;
  bool prepared_initialized = false, executor_initialized = false;
  Resources() {
    iree_arena_block_pool_initialize(64 * 1024, allocator, &blocks);
    iree_arena_initialize(&blocks, &arena);
  }
  ~Resources() {
    if (executor_initialized)
      loom_testbench_scenario_trial_executor_deinitialize(&executor);
    if (prepared_initialized)
      loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
    if (values_initialized)
      loom_testbench_scenario_configuration_values_deinitialize(&configuration);
    iree_hal_pool_release(pool);
    if (hal_initialized)
      loom_run_hal_testbench_context_deinitialize(&hal);
    if (vm_initialized)
      loom_vm_testbench_deinitialize(&vm);
    iree_arena_deinitialize(&arena);
    loomc_module_release(linked);
    loomc_module_release(library);
    loomc_module_release(input);
    loomc_linker_release(linker);
    loomc_compiler_release(compiler);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
    iree_arena_block_pool_deinitialize(&blocks);
  }
};
iree_status_t OpenRead(void *, iree_string_view_t path,
                       iree_io_stream_t **stream) {
  // Current drivers provide absolute fixture names (including inherited FDs).
  if (path.size == 0 || path.data[0] != '/')
    return Invalid("absolute fixture paths required");
  return iree_io_stdio_stream_open(IREE_IO_STDIO_STREAM_MODE_READ, path,
                                   iree_allocator_system(), stream);
}
iree_status_t Run(const Options &options) {
  Resources r;
  auto ca = loomc_allocator_system();
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_target_environment_create_configured(ca, &r.environment)));
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.structure_size = sizeof(target);
  target.target_environment = r.environment;
  loomc_context_options_t context{};
  context.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
  context.structure_size = sizeof(context);
  context.next = &target;
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_context_create(&context, ca, &r.context)));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_workspace_create(nullptr, ca, &r.workspace)));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_compiler_create(r.context, nullptr, ca, &r.compiler)));
  loomc_result_t *result = nullptr;
  auto status = iree_status_from_loomc(loomc_module_deserialize_from_path(
      r.context, r.workspace, loomc_make_cstring_view(options.input.c_str()),
      nullptr, ca, &r.input, &result));
  IREE_RETURN_IF_ERROR(Result(status, result));
  result = nullptr;
  status = iree_status_from_loomc(loomc_module_deserialize_from_path(
      r.context, r.workspace, loomc_make_cstring_view(options.library.c_str()),
      nullptr, ca, &r.library, &result));
  IREE_RETURN_IF_ERROR(Result(status, result));
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(
      loomc_linker_create(r.context, nullptr, ca, &r.linker)));
  loomc_link_module_provider_t providers[2]{};
  providers[0].module = r.input;
  providers[1].module = r.library;
  providers[0].provider_name = loomc_make_cstring_view(options.input.c_str());
  providers[1].provider_name = loomc_make_cstring_view(options.library.c_str());
  providers[0].role = providers[1].role = LOOMC_LINK_PROVIDER_ROLE_INPUT;
  loomc_link_options_t link{};
  link.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  link.structure_size = sizeof(link);
  link.mode = LOOMC_LINK_MODE_MERGE;
  link.module_providers = providers;
  link.module_provider_count = 2;
  result = nullptr;
  status = iree_status_from_loomc(
      loomc_link_module(r.linker, r.workspace, &link, &r.linked, &result));
  IREE_RETURN_IF_ERROR(Result(status, result));
  loomc_module_interop_view_t view{};
  result = nullptr;
  status = iree_status_from_loomc(
      loomc_module_get_interop_view(r.linked, ca, &view, &result));
  IREE_RETURN_IF_ERROR(Result(status, result));
  loom_testbench_plan_options_t plan_options{};
  loom_testbench_plan_options_initialize(&plan_options);
  loom_testbench_module_plan_t plan{};
  IREE_RETURN_IF_ERROR(
      loom_testbench_plan_module(view.module, &plan_options, &r.arena, &plan));
  const loom_testbench_scenario_plan_t *scenario = nullptr;
  for (size_t i = 0; i < plan.scenario_count; ++i)
    if (iree_string_view_equal(
            plan.scenarios[i].name,
            iree_make_cstring_view(options.scenario.c_str()))) {
      if (scenario)
        return Invalid("ambiguous scenario");
      scenario = &plan.scenarios[i];
    }
  if (!scenario || scenario->issue_count || scenario->is_configured ||
      scenario->configuration_count != 1 || scenario->trial_count != 1)
    return Invalid("one unconfigured scenario with one trial required");
  const auto &trial = scenario->trials[0];
  if (trial.issue_count || trial.trial_count != 1 ||
      trial.action.kind != LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE ||
      trial.action.target.kind != LOOM_TESTBENCH_INVOCATION_COMMAND_PROGRAM ||
      trial.action.target.workload_count || trial.action.target.result_count ||
      trial.action.oracle.kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL ||
      trial.action.expects_device_events)
    return Invalid(
        "one unspecialized command versus function comparison required");
  for (size_t index : options.arguments)
    if (index >= trial.action.target.input_count)
      return Invalid("capture index out of range before execution");
  for (size_t i = 0; i < trial.recipe_step_count; ++i)
    if (trial.recipe_steps[i].kind ==
        LOOM_TESTBENCH_TRIAL_RECIPE_STEP_GENERATOR)
      return Invalid("authored generators unsupported by this finite observer");
  // Reserve an empty directory before any execution. Pending files are never
  // published on comparison/readback/write errors, and old results are refused.
  std::error_code ec;
  const fs::path output(options.output);
  if (fs::is_symlink(fs::symlink_status(output, ec)))
    return Invalid("output may not be a symlink");
  ec.clear();
  if (!fs::exists(output, ec))
    fs::create_directory(output, ec);
  if (ec || !fs::is_directory(output, ec) || !fs::is_empty(output, ec) || ec)
    return Invalid("output must be a new or empty writable directory");
  auto &compilation = r.compilation;
  compilation.context = r.context;
  compilation.compiler = r.compiler;
  compilation.workspace = r.workspace;
  compilation.module = r.linked;
  status = loom_vm_testbench_initialize(r.environment, r.allocator, &r.vm);
  r.vm_initialized = true;
  IREE_RETURN_IF_ERROR(status);
  r.route = {iree_make_cstring_view("amdgpu"),
             loomc_amdgpu_iree_hal_target_provider()};
  loom_run_hal_testbench_context_initialize(r.environment, &r.route, 1,
                                            r.allocator, &r.hal);
  r.hal_initialized = true;
  const auto &symbol =
      view.module->symbols.entries[trial.action.target.callee_ref.symbol_id];
  const auto root = loomc_string_view_from_iree(
      loom_string_table_get(&view.module->strings, symbol.name_id));
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_add_module_runtime_requirements(
          &r.hal, r.linked, nullptr, &root, 1));
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(&r.hal));
  loom_run_hal_testbench_actual_provider_options_t provider{};
  provider.context = &r.hal;
  provider.compilation = &compilation;
  provider.module = r.linked;
  provider.native_module = view.module;
  auto &profile = r.profile;
  loom_run_hal_testbench_scenario_profile_initialize(
      iree_make_cstring_view("observer-amdgpu"), &provider, view.source_table,
      loom_diagnostic_sink_t{}, &profile);
  loom_testbench_scenario_execution_options_t execution{};
  loom_testbench_scenario_execution_options_initialize(&execution);
  execution.target =
      loom_run_hal_testbench_scenario_execution_profile(&profile);
  execution.oracle = loom_vm_testbench_execution_profile(
      &r.vm, &compilation, view.source_table, loom_diagnostic_sink_t{},
      loom_testbench_compile_result_callback_t{});
  const iree_hal_pool_host_access_t access{
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT};
  IREE_RETURN_IF_ERROR(
      iree_hal_heap_pool_create(access, r.allocator, r.allocator, &r.pool));
  loom_testbench_value_materializer_options_t materializer{};
  loom_testbench_value_materializer_options_initialize(&materializer);
  materializer.target_pool = r.pool;
  materializer.open_read_file = {OpenRead, nullptr};
  status = loom_testbench_scenario_configuration_values_initialize(
      view.module, scenario, r.allocator, &r.configuration);
  r.values_initialized = true;
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_configuration_values_materialize(
      &materializer, loom_testbench_entropy_root(0), 0, &r.configuration));
  status = loom_testbench_prepare_scenario_configuration(
      &execution, &r.configuration, &r.prepared);
  r.prepared_initialized = true;
  IREE_RETURN_IF_ERROR(status);
  if (r.prepared.trial_count != 1)
    return Invalid("one prepared product required");
  auto &product = r.prepared.trials[0].target;
  auto *observer = new Observer{&options, product, false};
  product.user_data = observer;
  product.vtable = &kObserverVtable;
  product.destroy = Observer::Destroy;
  status = loom_testbench_scenario_trial_executor_initialize(
      &r.prepared, 0, LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS,
      &materializer, 1, &r.executor);
  r.executor_initialized = true;
  IREE_RETURN_IF_ERROR(status);
  loom_testbench_scenario_trial_result_list_t results{};
  IREE_RETURN_IF_ERROR(
      loom_testbench_run_scenario_trial_batch(&r.executor, 0, 1, &results));
  if (!observer->called || results.count != 1 || !results.values[0].passed)
    return Invalid("original comparison failed or target did not execute");
  for (size_t index : options.arguments) {
    const auto filename =
        "product0_call0_after_arg" + std::to_string(index) + ".npy";
    fs::rename(output / (filename + ".pending"), output / filename, ec);
    if (ec)
      return iree_make_status(IREE_STATUS_DATA_LOSS, "cannot publish capture");
  }
  return iree_ok_status();
}
} // namespace
int main(int argc, char **argv) {
  Options options;
  auto status = Parse(argc, argv, options);
  if (iree_status_is_ok(status))
    status = Run(options);
  if (iree_status_is_ok(status))
    return 0;
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return 1;
}
