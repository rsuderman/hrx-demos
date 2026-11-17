// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_STATEFUL_INPUTS_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_STATEFUL_INPUTS_H_

#include "lib/loomify/benchmark/bundle.h"

namespace hrx::loomify_gguf::benchmark {

// First operation-owned stateful recipe: seven-tap short convolution with a
// nonzero history position, actual GGUF F32 taps and deterministic finite BCX/
// six-row history. Other operations report unsupported, never all-zero
// surrogates. The result is reset_plus_kernel/copy_seed_each_invocation and
// math unchecked. This prepares inputs only; the shared emitter owns replay
// proof/publication.
PreparationStatus PrepareStatefulCase(const Plan &plan,
                                      const Invocation &invocation,
                                      uint64_t max_input_bytes,
                                      PreparedCase *out, std::string *error);
PreparationStatus PrepareGdnCase(const Plan &plan, const Invocation &invocation,
                                 uint64_t max_input_bytes, PreparedCase *out,
                                 std::string *error);
PreparationStatus PrepareAttentionCase(const Plan &plan,
                                       const Invocation &invocation,
                                       uint64_t max_input_bytes,
                                       PreparedCase *out, std::string *error);
PreparationStatus PrepareQkCase(const Plan &plan, const Invocation &invocation,
                                uint64_t max_input_bytes, PreparedCase *out,
                                std::string *error);

// Populate an allocated 64-F32 rotary divisor view from its exact GGUF
// parameter or a proven selected unit-factor producer. Unknown provenance is
// unsupported; malformed known data is failure. No synthetic learned factors.
PreparationStatus
PrepareRotaryFactors(const Plan &plan, const Invocation &invocation,
                     size_t argument_index, bool require_at_least_one,
                     PreparedCase *prepared, std::string *error);

} // namespace hrx::loomify_gguf::benchmark
#endif
