// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "lib/llama/detail/ffn_structure.h"

#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace hrx::llama::detail {
namespace {
using namespace operations;

bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}

bool EncodingForStorage(uint32_t storage, Encoding *encoding) {
  // GGUF numeric storage tags; no ggml implementation/header dependency.
  switch (storage) {
  case 8:
    *encoding = Encoding::Q8_0;
    return true;
  case 12:
    *encoding = Encoding::Q4_K;
    return true;
  case 14:
    *encoding = Encoding::Q6_K;
    return true;
  default:
    return false;
  }
}

bool Extents(const model::TensorRecord &tensor, uint64_t input,
             uint64_t output) {
  return tensor.normalized_extents ==
         std::array<int64_t, 4>{static_cast<int64_t>(input),
                                static_cast<int64_t>(output), 1, 1};
}
} // namespace

bool AppendNormalizedFfn(OperationGraph *graph,
                         const model::ModelDescriptor &facts, size_t layer_slot,
                         ValueId residual, NormalizedFfnRegion *region,
                         std::string *error) {
  if (!graph || !region || !error)
    return Fail(error,
                "normalized FFN requires graph, region and error outputs");
  const auto prefix =
      "Llama FFN layer slot " + std::to_string(layer_slot) + ": ";
  if (facts.family != "llama" || layer_slot >= facts.layers.size())
    return Fail(error, prefix + "requires Llama facts and an existing layer");
  const auto &layer = facts.layers[layer_slot];
  const auto width = facts.embedding_width;
  const auto ffn_width = layer.feed_forward_width;
  if (layer.kind != model::LayerKind::kAttention || !width || !ffn_width ||
      width > uint64_t(std::numeric_limits<int64_t>::max()) ||
      ffn_width > uint64_t(std::numeric_limits<int64_t>::max()) ||
      !std::isfinite(facts.rms_epsilon) || facts.rms_epsilon <= 0)
    return Fail(error, prefix + "invalid dense geometry or RMS epsilon");
  const auto *description = graph->value(residual);
  const auto *input = description ? description->tensorType() : nullptr;
  if (!input || input->rank() != 2 || !input->dimension(0) ||
      input->dimension(1) != width ||
      input->elementType() != ElementType::F32 ||
      input->encoding() != Encoding::Dense ||
      input->layout() != Layout::RowMajor)
    return Fail(error,
                prefix + "residual must be an owned dense F32 [rows,width]");
  const uint64_t rows = input->dimension(0);
  for (auto opcode :
       {OpCode::RmsNorm, OpCode::GatedMatmul, OpCode::LinearResidual})
    if (!graph->definition(opcode))
      return Fail(error, prefix + "missing definition " + OpCodeName(opcode));

  constexpr std::array<const char *, 4> roles = {"post_norm", "ffn_gate",
                                                 "ffn_up", "ffn_down"};
  std::array<size_t, 4> slots{};
  std::array<Encoding, 4> encodings{};
  for (size_t role = 0; role < roles.size(); ++role) {
    size_t matches = 0;
    for (size_t slot = 0; slot < layer.tensors.size(); ++slot)
      if (layer.tensors[slot].role == roles[role]) {
        slots[role] = slot;
        ++matches;
      }
    if (matches != 1)
      return Fail(error,
                  prefix + roles[role] + " requires exactly one binding");
    const auto &tensor = layer.tensors[slots[role]].tensor;
    if (tensor.name.empty())
      return Fail(error,
                  prefix + roles[role] + " requires source tensor identity");
    const auto in_width = role == 3 ? ffn_width : width;
    const auto out_width = role == 0 ? 1 : role == 3 ? width : ffn_width;
    if (!Extents(tensor, in_width, out_width))
      return Fail(error, prefix + roles[role] + " has incompatible GGML axes");
    if (role == 0) {
      if (tensor.storage_type != 0)
        return Fail(error, prefix + "post_norm requires F32 storage");
      encodings[role] = Encoding::Dense;
    } else if (!EncodingForStorage(tensor.storage_type, &encodings[role])) {
      return Fail(error, prefix + roles[role] +
                             " requires Q4_K, Q6_K or Q8_0 storage (tag " +
                             std::to_string(tensor.storage_type) + ")");
    }
  }

  NormalizedFfnRegion staged;
  staged.layer_slot = layer_slot;
  staged.operation_begin = graph->invocations().size();
  staged.residual = residual;
  std::array<ValueId, 4> weights;
  for (size_t role = 0; role < roles.size(); ++role) {
    auto type = role == 0 ? TensorType::vector(width, ElementType::F32)
                          : TensorType::matrix(role == 3 ? ffn_width : width,
                                               role == 3 ? width : ffn_width,
                                               ElementType::F32)
                                .setEncoding(encodings[role])
                                .setLayout(Layout::GgmlWeightRows);
    weights[role] = graph->tensor(type, error);
    if (!weights[role].valid())
      return false;
    staged.parameters.push_back({weights[role], layer_slot, slots[role]});
  }
  staged.normalized =
      graph->tensor(TensorType::matrix(rows, width, ElementType::F32), error);
  staged.activation = graph->tensor(
      TensorType::matrix(rows, ffn_width, ElementType::F32), error);
  if (!staged.normalized.valid() || !staged.activation.valid())
    return false;
  if (!graph->operation(OpCode::RmsNorm)
           .bind(Port::Input, residual)
           .bind(Port::Gamma, weights[0])
           .bind(Port::Output, staged.normalized)
           .attributes(RmsNormAttrs{width, facts.rms_epsilon})
           .append(error) ||
      !graph->operation(OpCode::GatedMatmul)
           .bind(Port::Input, staged.normalized)
           .bind(Port::GateWeight, weights[1])
           .bind(Port::UpWeight, weights[2])
           .bind(Port::Output, staged.activation)
           .attributes(GatedMatmulAttrs{width, ffn_width, Activation::SiLU})
           .append(error) ||
      !graph->operation(OpCode::LinearResidual)
           .bind(Port::Input, staged.activation)
           .bind(Port::Weight, weights[3])
           .bind(Port::Residual, residual)
           .attributes(LinearResidualAttrs{ffn_width, width})
           .append(error)) {
    *error = prefix + *error;
    return false;
  }
  staged.operation_end = graph->invocations().size();
  *region = std::move(staged);
  return true;
}

} // namespace hrx::llama::detail
