// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
// Manual real-asset consumer fixture. The public exporter/CLI owns user flags.
#include "lib/loomify/benchmark/bundle.h"

#include <iostream>
#include <memory>
#include <set>
#include <string>

#include "loomc/target/amdgpu.h"

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
} // namespace

int main(int argc, char **argv) {
  if (argc != 4 && argc != 6) {
    std::cerr
        << "Usage: emit_selected_norm GGUF LIBRARY_ROOT NEW_OUTPUT [PHASE CAPACITY]\n";
    return 2;
  }
  namespace benchmark = hrx::loomify_gguf::benchmark;
  benchmark::Workload workload;
  workload.phase = "decode";
  workload.token_capacity = workload.active_tokens = 1;
  workload.history_tokens = 5;
  if (argc == 6) {
    workload.phase = argv[4];
    try {
      size_t consumed = 0;
      workload.token_capacity = std::stoull(argv[5], &consumed);
      if (argv[5][consumed] != '\0')
        return 2;
    } catch (...) {
      std::cerr << "invalid capacity\n";
      return 2;
    }
    workload.active_tokens = workload.token_capacity;
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
  std::set<std::string> expected_cases;
  for (size_t i = 0; i < plan.invocations.size() && cases.size() < 2; ++i) {
    const auto &invocation = plan.invocations[i];
    if (invocation.kernel_symbol != "qwen38_rmsnorm_f32_prefill")
      continue;
    std::cout << "selected=" << invocation.invocation_id << '\n'
              << "kernel=" << invocation.kernel_symbol << '\n'
              << "runtime_capacity="
              << invocation.arguments[0].value.scalar_bits << '\n'
              << "grid="
              << invocation.command.payload.dispatch.workgroup_count[0] << ','
              << invocation.command.payload.dispatch.workgroup_count[1] << ','
              << invocation.command.payload.dispatch.workgroup_count[2] << '\n';
    benchmark::PreparedCase prepared;
    if (benchmark::PrepareStatelessCase(
            plan, invocation, UINT64_C(64) * 1024 * 1024, &prepared, &error) !=
        benchmark::PreparationStatus::kReady) {
      std::cerr << error << '\n';
      return 1;
    }
    benchmark::CaseIdentity identity;
    if (!benchmark::BuildCaseIdentity(plan, invocation, prepared, &identity,
                                      &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    expected_cases.insert(identity.case_id);
    cases.push_back({i, std::move(prepared)});
  }
  if (cases.size() != 2) {
    std::cerr << "selected plan requires two supported RMS occurrences\n";
    return 1;
  }
  // An exact repeated request must not duplicate a benchmark record or fixture.
  cases.push_back(cases.front());
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
  if (emitted.size() != expected_cases.size()) {
    std::cerr << "aggregate changed distinct-case count\n";
    return 1;
  }
  std::cout << "cases=" << emitted.size() << '\n';
  for (const auto &item : emitted)
    std::cout << "case_id=" << item.case_id << '\n'
              << "input_policy_digest=" << item.input_policy_digest << '\n'
              << "benchmark=" << item.benchmark_symbol << '\n'
              << "timed_dispatches=" << item.timed_dispatches.size() << '\n'
              << "arithmetic_check=" << item.arithmetic_check << '\n';
  return 0;
}
