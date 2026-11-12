// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_SERVING_TESTING_ORDINARY_PREPARE_HARNESS_H_
#define HRX_SOLUTIONS_LIB_SERVING_TESTING_ORDINARY_PREPARE_HARNESS_H_

#include "tools/model/model_descriptor.h"
#include <array>
#include <cstdint>
#include <vector>

namespace hrx::loomify_gguf::testing {

// Local suites supply descriptor inputs and independent record expectations.
struct OrdinaryPreparationCase {
  const char *name;
  model::ModelDescriptor (*model)();
  const char *prepare_source;
  const char *prepare_bytecode;
  int64_t hidden, layers, kv;
  const char *epoch_root;
  const char *terminal_marker;
};
using Capacity = std::array<int64_t, 2>;

void CheckPreparationRecords(const OrdinaryPreparationCase &,
                             const std::vector<Capacity> &);
void CheckEffectivePrefill(const OrdinaryPreparationCase &);
void CheckSingleRowStageOrdinals(const OrdinaryPreparationCase &);
void CheckRejectedPreparationRequests(const OrdinaryPreparationCase &);
void CheckMissingPreparationConfig(const OrdinaryPreparationCase &);

} // namespace hrx::loomify_gguf::testing
#endif // HRX_SOLUTIONS_LIB_SERVING_TESTING_ORDINARY_PREPARE_HARNESS_H_
