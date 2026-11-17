// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_BUNDLE_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_BENCHMARK_BUNDLE_H_

#include "lib/loomify/benchmark/inputs.h"
#include "lib/loomify/benchmark/plan.h"
#include <array>

namespace hrx::loomify_gguf::benchmark {

enum class PreparationStatus { kReady, kUnsupported, kFailure };

struct EmittedDispatch {
  std::string kernel_symbol;
  bool reset = false;
  std::array<uint32_t, 3> workgroup_count{};
};

struct EmittedCase {
  std::string invocation_id;
  // Ordered original model origins; one leaf or the admitted two-launch pair.
  std::vector<std::string> covered_origin_ids;
  std::string case_id;
  std::string logical_case_key;
  std::string input_policy_digest;
  std::string benchmark_symbol;
  std::string setup_recipe_id;
  std::string measurement_subject;
  std::string reset_policy;
  std::string arithmetic_check;
  std::vector<WeightSource> weight_sources;
  std::vector<PreparedRoot> roots;
  std::vector<StateReset> resets;
  std::vector<EmittedDispatch> timed_dispatches;
  // Runtime safety assertions, separate from independent arithmetic checking.
  std::vector<std::string> safety_checks;
};

struct PreparedInvocationCase {
  size_t invocation_index = 0;
  PreparedCase inputs;
};

// Bounded same-shape canonical Q4 pair. Both members use identical source
// workloads/scalar ABI; inputs owns merged roots and six ABI buffer views in
// member order (packed, weight, output). This is not arbitrary composition.
struct PreparedSequenceCase {
  std::array<size_t, 2> invocation_indices{};
  PreparedCase inputs;
};

PreparationStatus PrepareProjectionPair(const Plan &plan, size_t first_index,
                                        size_t second_index,
                                        uint64_t max_input_bytes,
                                        PreparedSequenceCase *out,
                                        std::string *error);

// Shared metadata lookup only. Operation recipes retain encoding/shape/layout
// validation; SetTensorInput remains the sole exact-range fixture reader.
const model::TensorRecord *FindBenchmarkTensor(const Plan &plan,
                                               const std::string &key);
const Parameter *FindBenchmarkParameter(const Plan &plan,
                                        const Invocation &invocation,
                                        const loomc_cmd_buffer_ref_t &binding,
                                        uint64_t byte_length,
                                        std::string *error);

// Operation-owned recipes prepare exact final-ABI views and known logical
// launch geometry; unsupported operations remain explicitly uncovered.
PreparationStatus PrepareStatelessCase(const Plan &plan,
                                       const Invocation &invocation,
                                       uint64_t max_input_bytes,
                                       PreparedCase *out, std::string *error);
PreparationStatus PrepareProjectionCase(const Plan &plan,
                                        const Invocation &invocation,
                                        uint64_t max_input_bytes,
                                        PreparedCase *out, std::string *error);

PreparationStatus PrepareScaledF16Case(const Plan &plan,
                                       const Invocation &invocation,
                                       uint64_t max_input_bytes,
                                       PreparedCase *out, std::string *error);

PreparationStatus PrepareElementwiseCase(const Plan &plan,
                                         const Invocation &invocation,
                                         uint64_t max_input_bytes,
                                         PreparedCase *out, std::string *error);

PreparationStatus PrepareBasicCase(const Plan &plan,
                                   const Invocation &invocation,
                                   uint64_t max_input_bytes, PreparedCase *out,
                                   std::string *error);

// Emit genuine selected cases as ONE native-linked module plus relative
// fixtures into an already-owned EMPTY staging directory. The caller owns
// publication and failure cleanup. Optional sequences admit only the bounded
// canonical projection pair; ordinary singletons keep their existing behavior.
// Cases/fixtures/implementations deduplicate
// by exact identities; each distinct case retains its benchmark symbol.
// No private command-bytecode decoder,
// source-kernel copies, compiler subprocess or GPU execution. Direct check.case
// launches compile the retained selected closure without singleton command
// reclassification. Native LINK preserves the retained public implementation
// and resolves private dependencies; a generation-only command proves exact
// ABI/grid. Raw request/root provenance remains authoritative. Safety
// checks do not constitute an arithmetic oracle. Output may be partial.
bool EmitPreparedBenchmarkBundle(
    const Plan &plan, std::vector<PreparedInvocationCase> cases,
    const std::filesystem::path &empty_staging_directory,
    loomc_context_t *context, loomc_workspace_t *workspace,
    std::vector<EmittedCase> *out, std::string *error,
    std::vector<PreparedSequenceCase> sequences = {});

} // namespace hrx::loomify_gguf::benchmark
#endif
