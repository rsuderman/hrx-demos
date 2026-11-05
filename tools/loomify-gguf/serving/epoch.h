// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_SERVING_EPOCH_H_
#define HRX_SOLUTIONS_LOOMIFY_SERVING_EPOCH_H_
#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include <cstdint>
#include <string_view>
namespace hrx::loomify_gguf {
// epoch.loom must be available in module. Prefill has the numerical
// seven-buffer ABI and uses the configured pool/map/cache offsets within the
// serving arena. Caller retains both program references in this same module
// until final linking.
loomc_status_t DefineLlamaEpoch(ModuleBuilder &module, ProgramRef prefill,
                                uint64_t token_capacity, std::string_view name,
                                ProgramRef *out);
} // namespace hrx::loomify_gguf
#endif
