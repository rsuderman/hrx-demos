// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

namespace hrx::loomify_gguf::benchmark {
PreparationStatus PrepareStatelessCase(const Plan &plan,
                                       const Invocation &invocation,
                                       uint64_t max_input_bytes,
                                       PreparedCase *out, std::string *error) {
  using Preparer =
      PreparationStatus (*)(const Plan &, const Invocation &, uint64_t,
                            PreparedCase *, std::string *);
  const Preparer preparers[] = {PrepareBasicCase, PrepareProjectionCase,
                                PrepareElementwiseCase, PrepareScaledF16Case};
  if (error)
    error->clear();
  for (const auto prepare : preparers) {
    const auto status = prepare(plan, invocation, max_input_bytes, out, error);
    if (status != PreparationStatus::kUnsupported || (error && !error->empty()))
      return status;
  }
  if (error)
    *error = "unsupported operation recipe: " + invocation.kernel_symbol;
  return PreparationStatus::kUnsupported;
}
} // namespace hrx::loomify_gguf::benchmark
