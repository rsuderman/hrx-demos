// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Manual actual-GGUF stateful consumer fixture; the public CLI owns user flags.
#include "lib/loomify/benchmark/stateful_inputs.h"

#include "loomc/target/amdgpu.h"
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    std::cerr << std::string(message.data, message.size) << '\n';
  }
  loomc_status_free(status);
  return ok;
}
bool Number(const char *text, uint64_t *out) {
  if (!text || !*text || text[0] == '-')
    return false;
  try {
    size_t consumed = 0;
    *out = std::stoull(text, &consumed);
    return text[consumed] == '\0';
  } catch (...) {
    return false;
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4 && argc != 8) {
    std::cerr << "Usage: emit_selected_state GGUF LIBRARY_ROOT NEW_OUTPUT "
                 "[PHASE CAPACITY ACTIVE HISTORY]\n";
    return 2;
  }
  namespace benchmark = hrx::loomify_gguf::benchmark;
  benchmark::Workload workload;
  workload.phase = "decode";
  workload.token_capacity = workload.active_tokens = 1;
  workload.history_tokens = 17;
  if (argc == 8) {
    workload.phase = argv[4];
    if (!Number(argv[5], &workload.token_capacity) ||
        !Number(argv[6], &workload.active_tokens) ||
        !Number(argv[7], &workload.history_tokens)) {
      std::cerr << "invalid numerical workload\n";
      return 2;
    }
  }
  loomc_target_environment_t *raw_environment = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &raw_environment)))
    return 1;
  std::unique_ptr<loomc_target_environment_t,
                  decltype(&loomc_target_environment_release)>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *raw_context = nullptr;
  if (!Ok(loomc_context_create(&options, loomc_allocator_system(),
                               &raw_context)))
    return 1;
  std::unique_ptr<loomc_context_t, decltype(&loomc_context_release)> context(
      raw_context, loomc_context_release);
  loomc_workspace_t *raw_workspace = nullptr;
  if (!Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &raw_workspace)))
    return 1;
  std::unique_ptr<loomc_workspace_t, decltype(&loomc_workspace_release)>
      workspace(raw_workspace, loomc_workspace_release);
  benchmark::Plan plan;
  std::string error;
  if (!benchmark::BuildBenchmarkPlan(argv[1], workload, argv[2], context.get(),
                                     workspace.get(), &plan, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::vector<benchmark::PreparedInvocationCase> cases;
  for (size_t i = 0; i < plan.invocations.size() && cases.size() < 2; ++i) {
    const auto &invocation = plan.invocations[i];
    if (invocation.kernel_symbol != "llm.convolution.shortconv7")
      continue;
    benchmark::PreparedCase prepared;
    if (benchmark::PrepareStatefulCase(
            plan, invocation, UINT64_C(64) * 1024 * 1024, &prepared, &error) !=
        benchmark::PreparationStatus::kReady) {
      std::cerr << error << '\n';
      return 1;
    }
    std::cout << "selected=" << invocation.invocation_id
              << " kernel=" << invocation.kernel_symbol
              << " weight=" << prepared.weight_sources[0].key
              << " history=" << plan.workload.history_tokens
              << " reset_bytes=" << prepared.resets[0].target.byte_length
              << '\n';
    cases.push_back({i, std::move(prepared)});
  }
  if (cases.empty()) {
    std::cerr << "selected plan has no supported stateful invocation\n";
    return 1;
  }
  const auto output = std::filesystem::path(argv[3]);
  std::error_code ec;
  if (!std::filesystem::create_directory(output, ec)) {
    std::cerr << "new output directory required: " << ec.message() << '\n';
    return 1;
  }
  std::vector<benchmark::EmittedCase> emitted;
  if (!benchmark::EmitPreparedBenchmarkBundle(plan, std::move(cases), output,
                                              context.get(), workspace.get(),
                                              &emitted, &error)) {
    std::cerr << error << '\n';
    std::filesystem::remove_all(output, ec);
    if (ec)
      std::cerr << "failed to remove owned output: " << ec.message() << '\n';
    return 1;
  }
  std::cout << "cases=" << emitted.size() << '\n';
  for (const auto &item : emitted)
    std::cout << "case_id=" << item.case_id
              << " benchmark=" << item.benchmark_symbol
              << " subject=" << item.measurement_subject
              << " arithmetic_check=" << item.arithmetic_check << '\n';
  return 0;
}
