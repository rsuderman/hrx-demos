// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/parameters.h"

#include <algorithm>
#include <limits>

namespace hrx::dispatch {
namespace {
using namespace operations;
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
bool SameTensor(const model::TensorRecord &a, const model::TensorRecord &b) {
  return a.name == b.name && a.rank == b.rank &&
         a.normalized_extents == b.normalized_extents &&
         a.storage_type == b.storage_type &&
         a.absolute_offset == b.absolute_offset && a.byte_size == b.byte_size;
}
// Admission for the independently owned first-model providers. Quantized sizes
// are encoded block lengths, not decoded F32 allocation sizes.
bool StorageMatches(const TensorType &type, const model::TensorRecord &tensor) {
  if (type.elementType() != ElementType::F32 ||
      (type.rank() != 1 && type.rank() != 2) ||
      tensor.normalized_extents[2] != 1 || tensor.normalized_extents[3] != 1 ||
      tensor.normalized_extents[0] <= 0 || tensor.normalized_extents[1] <= 0 ||
      type.dimension(0) !=
          static_cast<uint64_t>(tensor.normalized_extents[0]) ||
      (type.rank() == 1
           ? tensor.normalized_extents[1] != 1
           : type.dimension(1) !=
                 static_cast<uint64_t>(tensor.normalized_extents[1])))
    return false;
  uint64_t block = 1, bytes = 4;
  if (type.rank() == 1) {
    if (tensor.storage_type != 0 || type.encoding() != Encoding::Dense ||
        type.layout() != Layout::RowMajor)
      return false;
  } else {
    if (type.layout() != Layout::GgmlWeightRows)
      return false;
    if (tensor.storage_type == 12 && type.encoding() == Encoding::Q4_K) {
      block = 256;
      bytes = 144;
    } else if (tensor.storage_type == 14 && type.encoding() == Encoding::Q6_K) {
      block = 256;
      bytes = 210;
    } else if (tensor.storage_type == 8 && type.encoding() == Encoding::Q8_0) {
      block = 32;
      bytes = 34;
    } else {
      return false;
    }
  }
  if (type.dimension(0) % block)
    return false;
  uint64_t blocks = type.dimension(0) / block;
  const uint64_t rows = type.rank() == 1 ? 1 : type.dimension(1);
  if (blocks > std::numeric_limits<uint64_t>::max() / rows)
    return false;
  blocks *= rows;
  return blocks <= std::numeric_limits<uint64_t>::max() / bytes &&
         tensor.byte_size == blocks * bytes &&
         tensor.byte_size <=
             std::numeric_limits<uint64_t>::max() - tensor.absolute_offset;
}
} // namespace
bool BindRegionParameters(const llama::LlamaStructure &owner,
                          const llama::LlamaRegionView &region,
                          loomify_gguf::Value parameters,
                          NativeEmissionSession &session, std::string *error) {
  if (&session.graph() != &owner.graph())
    return Fail(error, "parameter binding requires the session's graph owner");
  const auto &operations = owner.graph().invocations();
  if (region.operation_begin >= region.operation_end ||
      region.operation_end > operations.size())
    return Fail(error, "parameter binding requires a valid operation range");
  std::vector<ValueId> used;
  for (size_t i = region.operation_begin; i < region.operation_end; ++i)
    for (const auto &binding : operations[i].bindings())
      if (std::find(used.begin(), used.end(), binding.value()) == used.end())
        used.push_back(binding.value());
  struct Selected {
    ValueId value;
    const model::TensorBinding *binding;
  };
  std::vector<Selected> selected;
  for (const auto &association : owner.parameters()) {
    if (std::find(used.begin(), used.end(), association.value) == used.end())
      continue;
    const auto *binding = owner.parameter(association.coordinate);
    const auto *value = owner.graph().value(association.value);
    if (!binding || !value || !value->tensorType())
      return Fail(error, "invalid owned parameter association");
    const auto &key = binding->tensor.name;
    if (key.empty() || key.find('\0') != std::string::npos ||
        key.find('{') != std::string::npos ||
        key.find('}') != std::string::npos ||
        !StorageMatches(*value->tensorType(), binding->tensor))
      return Fail(error,
                  "parameter " + binding->role + " (" + key +
                      ") has incompatible key/encoding/extents/byte span");
    bool found = false;
    for (const auto &prior : selected) {
      if (prior.value == association.value) {
        if (!SameTensor(prior.binding->tensor, binding->tensor))
          return Fail(error, "conflicting repeated parameter identity: " + key);
        found = true;
      } else if (prior.binding->tensor.name == key) {
        return Fail(error,
                    "parameter key has multiple logical identities: " + key);
      }
    }
    if (!found)
      selected.push_back({association.value, binding});
  }
  // All semantic parameter ports require owned associations. This prevents a
  // missing association from quietly becoming a caller-provided boundary.
  for (size_t i = region.operation_begin; i < region.operation_end; ++i)
    for (const auto &binding : operations[i].bindings()) {
      const auto port = binding.port();
      if (port != Port::Gamma && port != Port::Weight &&
          port != Port::GateWeight && port != Port::UpWeight &&
          port != Port::Factors)
        continue;
      if (std::none_of(selected.begin(), selected.end(), [&](const auto &item) {
            return item.value == binding.value();
          }))
        return Fail(error, std::string("missing owned parameter for ") +
                               PortName(port));
    }
  for (const auto &item : selected) {
    auto status =
        session.Parameter(item.value, parameters, item.binding->tensor.name);
    if (!loomc_status_is_ok(status)) {
      const auto message = loomc_status_message(status);
      const std::string copy(message.data ? message.data : "", message.size);
      loomc_status_free(status);
      return Fail(error,
                  "parameter " + item.binding->tensor.name + ": " + copy);
    }
  }
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::dispatch
