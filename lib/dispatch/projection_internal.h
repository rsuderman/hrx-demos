// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_PROJECTION_INTERNAL_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_PROJECTION_INTERNAL_H_
#include "lib/dispatch/registry.h"
namespace hrx::dispatch::detail {
// Shared native staging, not a public model-level routing API. An attention
// residual provider supplies no output hint so its original residual stays
// live.
loomc_status_t StageProjection(const Registration &, BuildContext &,
                               uint64_t input_width, uint64_t output_width,
                               const NativeOperand &input,
                               const NativeOperand &weight,
                               loomify_gguf::Value existing_output,
                               bool has_existing_output,
                               loomify_gguf::Value *out);
} // namespace hrx::dispatch::detail
#endif
