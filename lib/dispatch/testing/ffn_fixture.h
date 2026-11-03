// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_TESTING_FFN_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_TESTING_FFN_FIXTURE_H_
#include "lib/llama/detail/ffn_structure.h"
#include "lib/llama/detail/structure_construction.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"
namespace hrx::llama::detail {
// Explicit incomplete fixture retention; never a partial complete-model API.
class StructureTestPeer {
public:
  static std::shared_ptr<const LlamaStructure>
  Retain(StructureConstruction &construction) { return construction.owner_; }
};
} // namespace hrx::llama::detail
namespace hrx::dispatch::testing {
inline std::shared_ptr<const llama::LlamaStructure>
FfnFixture(model::ModelDescriptor facts, operations::Phase phase, uint64_t rows,
           std::string *error, size_t layer = 0) {
  using namespace operations;
  OperationRegistry registry;
  if (!DefineRmsNorm(&registry, error) ||
      !DefineGatedMatmul(&registry, error) ||
      !DefineLinearResidual(&registry, error))
    return {};
  auto construction = llama::detail::StructureConstruction::Create(
      std::move(facts), {phase, rows, 32, 64, "ffn-checkpoint"}, registry, error);
  if (!construction || !construction->createInputs(error))
    return {};
  auto &owner = construction->structure();
  if (layer >= owner.facts().layers.size())
    return {};
  const std::array<const char *, 4> roles{
      "post_norm", "ffn_gate", "ffn_up", "ffn_down"};
  std::array<ValueId, 4> weights;
  for (size_t r = 0; r < roles.size(); ++r) {
    const auto &tensors = owner.facts().layers[layer].tensors;
    size_t slot = 0;
    while (slot < tensors.size() && tensors[slot].role != roles[r])
      ++slot;
    if (slot == tensors.size() ||
        !construction->resolve({llama::ParameterScope::Layer, layer, slot},
                               &weights[r], error))
      return {};
  }
  llama::detail::NormalizedFfnRegion ffn;
  const auto &inputs = owner.inputs();
  if (!llama::detail::AppendNormalizedFfnResolved(
          &construction->graph(), owner.facts(), layer, inputs.residual,
          weights, inputs.row_begin, inputs.row_count, &ffn, error) ||
      !construction->record({llama::LlamaRegionKind::Ffn, layer, 1},
                            ffn.operation_begin, ffn.operation_end, error) ||
      !construction->exportValue(inputs.residual, error))
    return {};
  return llama::detail::StructureTestPeer::Retain(*construction);
}
inline model::ModelDescriptor FfnFacts(uint64_t hidden = 3072,
                                      uint64_t ffn = 8192) {
  model::ModelDescriptor facts;
  facts.family = "llama";
  facts.embedding_width = hidden;
  facts.vocabulary_size = 128;
  facts.key_head_width = facts.value_head_width = 64;
  facts.rms_epsilon = 1e-5;
  model::LayerDescriptor layer;
  layer.index = 7; // Deliberately different from descriptor vector slot zero.
  layer.query_heads = hidden / 64;
  layer.kv_heads = 2;
  layer.feed_forward_width = ffn;
  auto parameter = [&](std::string role, std::string name, uint32_t storage,
                       uint64_t input, uint64_t output, uint64_t bytes) {
    model::TensorBinding binding;
    binding.role = std::move(role);
    auto &tensor = binding.tensor;
    tensor.name = std::move(name);
    tensor.storage_type = storage;
    tensor.normalized_extents = {int64_t(input), int64_t(output), 1, 1};
    tensor.rank = output == 1 ? 1 : 2;
    tensor.absolute_offset = 4096;
    for (const auto &p : layer.tensors)
      tensor.absolute_offset += p.tensor.byte_size;
    tensor.byte_size = bytes;
    layer.tensors.push_back(std::move(binding));
  };
  parameter("post_norm", "blk.7.ffn_norm.weight", 0, hidden, 1, hidden * 4);
  parameter("ffn_gate", "blk.7.ffn_gate.weight", 12, hidden, ffn,
            hidden * ffn / 256 * 144);
  parameter("ffn_up", "blk.7.ffn_up.weight", 12, hidden, ffn,
            hidden * ffn / 256 * 144);
  parameter("ffn_down", "blk.7.ffn_down.weight", 14, ffn, hidden,
            hidden * ffn / 256 * 210);
  facts.layers.push_back(std::move(layer));
  return facts;
}
} // namespace hrx::dispatch::testing
#endif
