// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LLAMA_PROGRAM_H_
#define HRX_SOLUTIONS_LIB_LLAMA_PROGRAM_H_
#include "lib/dispatch/attention_sequence.h"
#include "lib/llama/structure.h"
namespace hrx::llama {
struct LlamaProgramSettings {
  operations::Phase phase = operations::Phase::Prefill;
  // pool_capacity is physical token capacity (multiple of 64), zero for dense.
  uint64_t row_capacity = 0, cache_capacity = 0, pool_capacity = 0,
           cache_base = 0, map_origin = 0;
  std::string source_asset;
};
struct LlamaBufferRequirement {
  std::string name;
  uint64_t minimum_bytes = 0;
};
struct LlamaProgram {
  std::shared_ptr<const LlamaStructure> structure;
  loomify_gguf::ProgramRef program;
  std::vector<LlamaBufferRequirement> buffers;
};
std::vector<dispatch::Id> LlamaProgramDispatches();
// Caller preloads LoadLlamaNative with matching layer/cache/pool configuration.
// Failure leaves out unchanged; discard mutable module after native failure.
bool DefineLlamaProgram(const model::ModelDescriptor &,
                        const LlamaProgramSettings &,
                        const dispatch::Registry &,
                        loomify_gguf::ModuleBuilder &, const std::string &name,
                        LlamaProgram *, std::string *error);
} // namespace hrx::llama
#endif
