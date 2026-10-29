// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LOOMIFY_LLAMA_STEP_H_
#define HRX_SOLUTIONS_LOOMIFY_LLAMA_STEP_H_

#include <string_view>

#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "tools/model/model_descriptor.h"

namespace hrx::loomify_gguf {

enum class LlamaStepKind { kPrefill, kDecode };

// Construct every numerical stage of a one-token step for the supported Llama
// 3.2 3B Q4_K_M profile. Numerical kernel/template definitions and their
// matching config/target definitions must already be present in module. No
// model command body is imported or called by this recipe.
//
// Seven explicit bindings match the existing isolated decode layout:
// immutable parameters, residual, control, KV map, KV cache, token IDs,
// progress. Prefill control is {active_count=1, absolute_position,
// canonical_eos}; decode control is {absolute_position, generated_count,
// canonical_eos}. Existing finish publication transitions prefill to decode
// layout. Cache/map and progress lifetimes belong to the caller. Each
// invocation consumes one token, publishes the next greedy token, and updates
// control/progress via the existing numerical finish adapter. This is a
// computational root, not a serving adapter or a qualification claim for the
// preexisting numerical providers.
loomc_status_t DefineLlamaStep(const model::ModelDescriptor &model,
                               ModuleBuilder &module, LlamaStepKind kind,
                               std::string_view name, ProgramRef *out);

} // namespace hrx::loomify_gguf

#endif // HRX_SOLUTIONS_LOOMIFY_LLAMA_STEP_H_
