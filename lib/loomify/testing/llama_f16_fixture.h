// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_TESTING_LLAMA_F16_FIXTURE_H_
#define HRX_SOLUTIONS_LOOMIFY_TESTING_LLAMA_F16_FIXTURE_H_
#include "lib/loomify/testing/configuration_fixture.h"
namespace hrx::loomify_gguf::testing {
// Existing Llama geometry and role order, with raw half matrices and F32 norms.
// Recompute physical layout once, preserving the tied output alias by name.
inline model::ModelDescriptor LlamaF16ModelFacts() {
  auto descriptor = LlamaModelFacts();
  std::map<std::string, model::TensorRecord> identities;
  uint64_t next = 4096;
  auto retag = [&](model::TensorBinding &binding) {
    auto &tensor = binding.tensor;
    auto old = identities.find(tensor.name);
    if (old != identities.end()) {
      tensor = old->second;
      return;
    }
    if (tensor.rank == 2)
      tensor.storage_type = 1;
    if (!model::EncodedTensorByteSize(
            tensor.storage_type, tensor.normalized_extents[0],
            tensor.normalized_extents[1], &tensor.byte_size))
      std::abort();
    tensor.absolute_offset = next;
    next += (tensor.byte_size + 255) / 256 * 256;
    identities.emplace(tensor.name, tensor);
  };
  for (auto &binding : descriptor.globals)
    retag(binding);
  for (auto &layer : descriptor.layers)
    for (auto &binding : layer.tensors)
      retag(binding);
  return descriptor;
}
} // namespace hrx::loomify_gguf::testing
#endif
