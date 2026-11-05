// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/llama/detail/attention_structure.h"
#include "lib/llama/detail/embedding_structure.h"
#include "lib/llama/detail/ffn_structure.h"
#include "lib/operations/attention_operations.h"
#include "lib/operations/control_operations.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"

namespace hrx::llama {
bool BuildLlamaStructure(model::ModelDescriptor facts,
                         const LlamaBuildOptions &options,
                         std::shared_ptr<const LlamaStructure> *out,
                         std::string *error) {
  using namespace operations;
  if (!out || !options.history_capacity) {
    if (error)
      *error = "complete Llama requires output and history capacity";
    return false;
  }
  OperationRegistry registry;
  if (!DefineEmbedding(&registry, error) || !DefineRmsNorm(&registry, error) ||
      !DefineLinear(&registry, error) || !DefineRope(&registry, error) ||
      !DefineRotatePublishKV(&registry, error) ||
      !DefineCausalAttention(&registry, error) ||
      !DefineGatedMatmul(&registry, error) ||
      !DefineLinearResidual(&registry, error) ||
      !DefineOutputHead(&registry, error) ||
      !DefinePublishToken(&registry, error))
    return false;
  auto owner = detail::StructureConstruction::Create(std::move(facts), options,
                                                     registry, error);
  if (!owner || !owner->createInputs(error))
    return false;
  auto &g = owner->graph();
  const auto &s = owner->structure();
  const auto &f = s.facts();
  const auto &in = s.inputs();
  auto resolve = [&](ParameterScope scope, size_t layer, const char *role,
                     ValueId *value) {
    const auto &bindings =
        scope == ParameterScope::Global ? f.globals : f.layers[layer].tensors;
    size_t count = 0, slot = 0;
    for (size_t i = 0; i < bindings.size(); ++i)
      if (bindings[i].role == role) {
        ++count;
        slot = i;
      }
    if (count != 1) {
      if (error)
        *error = std::string("requires exactly one tensor role: ") + role;
      return false;
    }
    return owner->resolve({scope, layer, slot}, value, error);
  };
  if (!detail::AppendEmbedding(owner.get(), error))
    return false;
  for (size_t layer = 0; layer < f.layers.size(); ++layer) {
    size_t begin = g.invocations().size();
    detail::NormalizedAttentionRegion attention;
    if (!detail::AppendNormalizedAttention(owner.get(), layer, &attention,
                                           error))
      return false;
    size_t ffn_begin = g.invocations().size();
    std::array<ValueId, 4> weights;
    const char *roles[] = {"post_norm", "ffn_gate", "ffn_up", "ffn_down"};
    for (size_t i = 0; i < 4; ++i)
      if (!resolve(ParameterScope::Layer, layer, roles[i], &weights[i]))
        return false;
    detail::NormalizedFfnRegion ffn;
    if (!detail::AppendNormalizedFfnResolved(&g, f, layer, in.residual, weights,
                                             in.row_begin, in.row_count, &ffn,
                                             error) ||
        !owner->record({LlamaRegionKind::Ffn, layer, 1}, ffn_begin,
                       g.invocations().size(), error) ||
        !owner->record({LlamaRegionKind::Layer, layer, 1}, begin,
                       g.invocations().size(), error))
      return false;
  }
  ValueId gamma, weight;
  if (!resolve(ParameterScope::Global, 0, "output_norm", &gamma) ||
      !resolve(ParameterScope::Global, 0, "output", &weight))
    return false;
  size_t begin = g.invocations().size();
  if (!g.operation(OpCode::OutputHead)
           .bind(Port::Input, in.residual)
           .bind(Port::Gamma, gamma)
           .bind(Port::Weight, weight)
           .bind(Port::Control, in.count_control)
           .bind(Port::Output, in.token)
           .bind(Port::RowBegin, in.row_begin)
           .bind(Port::RowCount, in.row_count)
           .attributes(OutputHeadAttrs{f.embedding_width, f.vocabulary_size,
                                       f.rms_epsilon})
           .append(error) ||
      !g.operation(OpCode::PublishToken)
           .bind(Port::Token, in.token)
           .bind(Port::Control, in.control)
           .bind(Port::History, in.history)
           .bind(Port::Progress, in.progress)
           .attributes(PublishTokenAttrs{options.phase,
                                         options.history_capacity,
                                         TerminalPolicy::StopOnTokenIds,
                                         {128001, 128008, 128009}})
           .append(error) ||
      !owner->record({LlamaRegionKind::Output, 0, 0}, begin,
                     g.invocations().size(), error))
    return false;
  for (auto value : {in.control, in.history, in.progress})
    if (!owner->exportValue(value, error))
      return false;
  for (const auto &layer : in.layers)
    if (!owner->exportValue(layer.key_cache, error) ||
        !owner->exportValue(layer.value_cache, error))
      return false;
  *out = owner->owner_;
  return true;
}
} // namespace hrx::llama
