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
// rms_norm is chosen by the caller in this same module. It must accept one
// index workload and (index, buffer, buffer, buffer) launch arguments,
// implementing weighted F32 RMSNorm for this qualified profile. The recipe
// invokes it directly for both per-layer normalizations and the final output
// normalization.
loomc_status_t DefineLlamaStep(const model::ModelDescriptor &model,
                               ModuleBuilder &module, KernelRef rms_norm,
                               LlamaStepKind kind, std::string_view name,
                               ProgramRef *out);

// Author a distinct numerical kernel through native Loom builders, reusing the
// existing llm.rmsnorm.row template motif rather than duplicating its
// arithmetic. The supported profile is width3072, epsilon1e-5, F32
// input/gamma/output and disjoint buffers, with one workgroup per row and 256
// invocations per workgroup. Workload and launch row counts must agree and lie
// in [1,2048], matching the existing numerical adapter contract.
// The motif declaration/providers must already be
// available to the module.
loomc_status_t DefineLlamaRmsNorm(ModuleBuilder &module, std::string_view name,
                                  KernelRef *out);

} // namespace hrx::loomify_gguf

#endif // HRX_SOLUTIONS_LOOMIFY_LLAMA_STEP_H_
