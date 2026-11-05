// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_DISPATCH_TESTING_ATTENTION_FIXTURE_H_
#define HRX_SOLUTIONS_LIB_DISPATCH_TESTING_ATTENTION_FIXTURE_H_
#include "lib/llama/detail/attention_structure.h"
#include "lib/llama/detail/ffn_structure.h"
#include "lib/operations/attention_operations.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"
#include <functional>
namespace hrx::dispatch::testing {
inline std::unique_ptr<llama::detail::StructureConstruction> AttentionFixture(
    uint64_t rows, uint64_t capacity, operations::Phase phase,
    std::string *error, uint64_t layers = 2,
    std::function<void(model::ModelDescriptor &, llama::LlamaBuildOptions &)>
        configure = {},
    bool with_ffn = false) {
  using namespace operations;
  model::ModelDescriptor facts;
  facts.family = "llama";
  facts.embedding_width = 3072;
  facts.vocabulary_size = 128256;
  facts.key_head_width = facts.value_head_width = 128;
  facts.rms_epsilon = 1e-5;
  facts.rope.width = 128;
  facts.rope.base = 500000;
  uint64_t next_source_offset = 4096;
  auto weight = [&](std::string role, std::string name, uint32_t storage,
                    uint64_t in, uint64_t out = 1) {
    model::TensorBinding b;
    b.role = std::move(role);
    b.tensor.name = std::move(name);
    b.tensor.storage_type = storage;
    b.tensor.normalized_extents = {int64_t(in), int64_t(out), 1, 1};
    b.tensor.rank = out == 1 ? 1 : 2;
    b.tensor.absolute_offset = next_source_offset;
    b.tensor.byte_size =
        storage == 0 ? in * 4 : in / 256 * out * (storage == 12 ? 144 : 210);
    next_source_offset += (b.tensor.byte_size + 255) / 256 * 256;
    return b;
  };
  facts.globals.push_back(
      weight("rope_frequencies", "rope_freqs.weight", 0, 64));
  for (uint64_t i = 0; i < layers; ++i) {
    model::LayerDescriptor l;
    l.index = 7 + i * 4;
    l.query_heads = 24;
    l.kv_heads = 8;
    l.feed_forward_width = 8192;
    const auto prefix = "blk." + std::to_string(l.index);
    l.tensors = {weight("input_norm", prefix + ".attn_norm.weight", 0, 3072),
                 weight("query", prefix + ".attn_q.weight", 12, 3072, 3072),
                 weight("key", prefix + ".attn_k.weight", 14, 3072, 1024),
                 weight("value", prefix + ".attn_v.weight", 14, 3072, 1024),
                 weight("attention_output", prefix + ".attn_output.weight", 12,
                        3072, 3072)};
    if (with_ffn) {
      l.tensors.push_back(
          weight("post_norm", prefix + ".ffn_norm.weight", 0, 3072));
      l.tensors.push_back(
          weight("ffn_gate", prefix + ".ffn_gate.weight", 12, 3072, 8192));
      l.tensors.push_back(
          weight("ffn_up", prefix + ".ffn_up.weight", 12, 3072, 8192));
      l.tensors.push_back(
          weight("ffn_down", prefix + ".ffn_down.weight", 12, 8192, 3072));
    }
    facts.layers.push_back(std::move(l));
  }
  OperationRegistry registry;
  if (!DefineGatedMatmul(&registry, error) ||
      !DefineRmsNorm(&registry, error) || !DefineLinear(&registry, error) ||
      !DefineLinearResidual(&registry, error) ||
      !DefineRope(&registry, error) ||
      !DefineRotatePublishKV(&registry, error) ||
      !DefineCausalAttention(&registry, error))
    return nullptr;
  llama::LlamaBuildOptions options{phase, rows, capacity, 128, "fixture"};
  options.attention_policy = NumericalPolicy::LlamaPacketMixed;
  if (configure)
    configure(facts, options);
  auto construction = llama::detail::StructureConstruction::Create(
      std::move(facts), options, registry, error);
  if (!construction || !construction->createInputs(error))
    return nullptr;
  for (uint64_t i = 0; i < layers; ++i) {
    llama::detail::NormalizedAttentionRegion region;
    if (!llama::detail::AppendNormalizedAttention(construction.get(), i,
                                                  &region, error))
      return nullptr;
    if (with_ffn) {
      std::array<ValueId, 4> weights;
      for (size_t j = 0; j < 4; ++j)
        if (!construction->resolve({llama::ParameterScope::Layer, i, 5 + j},
                                   &weights[j], error))
          return nullptr;
      llama::detail::NormalizedFfnRegion ffn;
      if (!llama::detail::AppendNormalizedFfnResolved(
              &construction->graph(), construction->structure().facts(), i,
              construction->structure().inputs().residual, weights,
              construction->structure().inputs().row_begin,
              construction->structure().inputs().row_count, &ffn, error) ||
          !construction->record({llama::LlamaRegionKind::Ffn, i, 1},
                                ffn.operation_begin, ffn.operation_end,
                                error) ||
          !construction->record({llama::LlamaRegionKind::Layer, i, 1},
                                region.operation_begin, ffn.operation_end,
                                error))
        return nullptr;
    }
  }
  return construction;
}
} // namespace hrx::dispatch::testing
#endif
