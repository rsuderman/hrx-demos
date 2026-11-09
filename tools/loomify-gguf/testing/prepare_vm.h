// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TESTING_PREPARE_VM_H_
#define HRX_SOLUTIONS_TESTING_PREPARE_VM_H_
#include "loomc/loomc.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace hrx::loomify_gguf::testing {
struct PreparationRequest {
  int64_t prefill = 128, context = 512, pool = 512, rows = 1, pins = 0;
  int32_t draft_depth = 0;
  std::vector<std::array<int64_t, 2>> shapes = {{1, 1}, {128, 1}};
  std::string weights = "fixture.gguf";
  int32_t options_type = 2;
  int64_t options_fields = 0;
};
struct PreparationStage {
  std::string root;
  int64_t tag;
  int32_t id;
};
struct PreparationParameter {
  int32_t stage, binding, source;
  std::string directory, path, policy;
};
struct PreparationObservation {
  std::vector<int32_t> requirements;
  std::vector<PreparationStage> stages;
  std::vector<PreparationParameter> parameters;
  int64_t effective_prefill = 0;
  int32_t shared = 0;
  // ABI result order: state, terminal, allocations, views, origins, draft,
  // geometry.
  std::array<std::vector<uint8_t>, 7> buffers;
};
// Compile a fresh linked module through the existing VM target and invoke
// actual
// @prepare with observing input/json/prepare native imports. False requirements
// abort the invocation; observations remain available. Does not mock record
// logic. Context must include the VM target; module is borrowed and may be
// lowered.
bool RunPreparation(loomc_target_environment_t *environment,
                    loomc_context_t *context, loomc_workspace_t *workspace,
                    loomc_module_t *module, const PreparationRequest &request,
                    PreparationObservation *out, std::string *error);
} // namespace hrx::loomify_gguf::testing
#endif
