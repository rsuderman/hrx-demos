// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures/ffn_binding.h"
#include "lib/loomify/configuration_internal.h"

namespace hrx::loomify_gguf::internal {
bool BindFfnProjections(Producer &producer, const model::LayerDescriptor &layer,
                        uint64_t hidden,
                        std::initializer_list<uint32_t> down_formats,
                        std::vector<std::vector<uint64_t>> &rows) {
  const auto width = layer.feed_forward_width;
  if (!producer.Tensor(layer.tensors, "ffn_gate", hidden, width, {},
                       layer.index) ||
      !producer.Tensor(layer.tensors, "ffn_up", hidden, width, {},
                       layer.index) ||
      !producer.Tensor(layer.tensors, "ffn_down", width, hidden, down_formats,
                       layer.index))
    return false;
  const auto &gate = *Role(layer.tensors, "ffn_gate");
  const auto &up = *Role(layer.tensors, "ffn_up");
  const auto &down = *Role(layer.tensors, "ffn_down");
  CanonicalProjectionFacts gate_facts, up_facts;
  if (!CheckCanonicalProjection(gate, hidden, width, &gate_facts,
                                producer.error) ||
      !CheckCanonicalProjection(up, hidden, width, &up_facts, producer.error))
    return false;
  rows.push_back({width, gate.storage_type, up.storage_type, down.storage_type,
                  gate.byte_size, up.byte_size, down.byte_size});
  producer.Projection(gate, hidden, width);
  producer.Projection(up, hidden, width);
  producer.Projection(down, width, hidden);
  return true;
}
} // namespace hrx::loomify_gguf::internal
