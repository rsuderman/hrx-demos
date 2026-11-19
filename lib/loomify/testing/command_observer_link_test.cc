// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// O1 only: exercise the public profile setup and retain executor/oracle symbols
// so shared-SDK linkage is checked before adding command observation.
#include "loom/tooling/execution/hal/scenario_profile.h"
#include "loom/tooling/target/vm/testbench.h"
#include "loom/tooling/testbench/scenario/executor.h"

int main() {
  loom_run_hal_testbench_actual_provider_options_t provider_options = {};
  loom_run_hal_testbench_scenario_profile_t hal_profile = {};
  loom_run_hal_testbench_scenario_profile_initialize(
      iree_make_cstring_view("observer-link-check"), &provider_options, nullptr,
      loom_diagnostic_sink_t{}, &hal_profile);
  const auto execution_profile =
      loom_run_hal_testbench_scenario_execution_profile(&hal_profile);
  loom_testbench_scenario_execution_options_t options = {};
  loom_testbench_scenario_execution_options_initialize(&options);
  // Volatile function pointers make these real link requirements even with
  // optimization; no scenario, compiler, device or runtime is initialized.
  auto volatile executor = &loom_testbench_run_scenario_trial_batch;
  auto volatile vm_initialize = &loom_vm_testbench_initialize;
  auto volatile vm_deinitialize = &loom_vm_testbench_deinitialize;
  return execution_profile.prepare && executor && vm_initialize &&
                 vm_deinitialize
             ? 0
             : 1;
}
