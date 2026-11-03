// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/shapes/result.h"

#include "lib/operations/ffn_operations.h"
#include "lib/operations/attention_operations.h"
#include "lib/operations/control_operations.h"
#include "lib/operations/tensor_operations.h"
#include <limits>

namespace hrx::dispatch::shapes {
namespace {
using namespace operations;
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}

// Bound logical element/byte arithmetic to the signed native index domain.
// Quantized TensorTypes describe decoded F32 axes, not GGUF storage bytes.
bool CheckSize(const TensorType &type, std::string *error) {
  const auto max = uint64_t(std::numeric_limits<int64_t>::max());
  uint64_t size = 1;
  for (size_t i = 0; i < type.rank(); ++i) {
    const auto dim = type.dimension(i);
    if (!dim || dim > max / size)
      return Fail(error, "logical tensor element count is not representable");
    size *= dim;
  }
  uint64_t bytes = 0;
  switch (type.elementType()) {
  case ElementType::F32:
  case ElementType::I32:
    bytes = 4;
    break;
  case ElementType::F16:
    bytes = 2;
    break;
  case ElementType::I8:
    bytes = 1;
    break;
  case ElementType::I64:
    bytes = 8;
    break;
  }
  return (bytes && size <= max / bytes) ||
         Fail(error, "logical tensor byte count is not representable");
}
bool Define(OpCode opcode, OperationRegistry *registry, std::string *error) {
  switch (opcode) {
  case OpCode::OutputHead:
    return DefineOutputHead(registry, error);
  case OpCode::Embedding:
    return DefineEmbedding(registry, error);
  case OpCode::Rope:
    return DefineRope(registry, error);
  case OpCode::CausalAttention:
    return DefineCausalAttention(registry, error);
  case OpCode::RotatePublishKV:
    return DefineRotatePublishKV(registry, error);
  case OpCode::PublishToken:
    return DefinePublishToken(registry, error);
  case OpCode::RmsNorm:
    return DefineRmsNorm(registry, error);
  case OpCode::Linear:
    return DefineLinear(registry, error);
  case OpCode::GatedMatmul:
    return DefineGatedMatmul(registry, error);
  case OpCode::LinearResidual:
    return DefineLinearResidual(registry, error);
  default:
    return Fail(error, "operation has no dispatch result shape rule");
  }
}
} // namespace

bool InferResult(const operations::OperationGraph &graph,
                 operations::OpCode opcode,
                 const operations::OperationAttrs &attributes,
                 const std::vector<Operand> &inputs, Result *out,
                 std::string *error) {
  if (!out || !error)
    return Fail(error, "result or error output is null");
  OperationRegistry registry;
  if (!Define(opcode, &registry, error))
    return false;
  OperationGraph checked(registry);
  const auto *definition = checked.definition(opcode);
  auto invocation = checked.operation(opcode);
  const TensorType *input = nullptr;
  Result staged;
  if (opcode == OpCode::LinearResidual)
    staged.port = Port::Residual;
  staged.has_result = opcode != OpCode::RotatePublishKV &&
                      opcode != OpCode::PublishToken;
  if (staged.has_result)
    staged.access = definition->port(staged.port)->access();
  std::vector<std::pair<ValueId, ValueId>> copied;
  for (const auto &operand : inputs) {
    const auto *port = definition->port(operand.port);
    const auto *value = graph.value(operand.value);
    if (!port || port->access() == Access::Write || !value)
      return Fail(error, "unknown input port, caller output or foreign value");
    ValueId clone;
    for (const auto &entry : copied)
      if (entry.first == operand.value)
        clone = entry.second;
    if (const auto *type = value->tensorType()) {
      if (!CheckSize(*type, error))
        return false;
      if (operand.port == Port::Weight || operand.port == Port::GateWeight ||
          operand.port == Port::UpWeight) {
        if (type->encoding() != Encoding::Dense &&
            type->encoding() != Encoding::Q4_K &&
            type->encoding() != Encoding::Q6_K &&
            type->encoding() != Encoding::Q8_0)
          return Fail(error, "unsupported dispatch weight encoding");
      }
      if (!clone.valid())
        clone = checked.tensor(*type, error);
      if (operand.port == Port::Input || operand.port == Port::TokenIds ||
          operand.port == Port::Query)
        input = type;
      if (staged.has_result && operand.port == Port::Residual) {
        staged.type = *type;
        staged.alias = operand.value;
      }
    } else if (const auto *scalar = value->scalarType()) {
      if (!clone.valid())
        clone = checked.scalar(*scalar, error);
    } else {
      return Fail(error, "dispatch shape inputs cannot be state handles");
    }
    if (!clone.valid())
      return false;
    copied.emplace_back(operand.value, clone);
    invocation.bind(operand.port, clone);
  }
  if (!staged.has_result) {
    std::visit([&](const auto &a) { invocation.attributes(a); }, attributes);
    if (!invocation.append(error))
      return false;
    *out = std::move(staged);
    error->clear();
    return true;
  }
  if (!input)
    return Fail(error, "dispatch shape inference requires a tensor input");
  uint64_t width = 0;
  TensorType result_type;
  if (std::holds_alternative<OutputHeadAttrs>(attributes) &&
      opcode == OpCode::OutputHead) {
    result_type = TensorType::vector(1, ElementType::I32);
    width = 1;
  } else if (const auto *a = std::get_if<EmbeddingAttrs>(&attributes)) {
    if (opcode == OpCode::Embedding)
      width = a->embedding_width;
  } else if (std::holds_alternative<RopeAttrs>(attributes) &&
             opcode == OpCode::Rope) {
    result_type = *input;
    width = input->dimension(1);
  } else if (const auto *a = std::get_if<CausalAttentionAttrs>(&attributes)) {
    if (opcode == OpCode::CausalAttention && a->value_width &&
        a->query_heads <= std::numeric_limits<uint64_t>::max() / a->value_width)
      width = a->query_heads * a->value_width;
  } else if (const auto *a = std::get_if<RmsNormAttrs>(&attributes)) {
    if (opcode == OpCode::RmsNorm)
      width = a->width;
  } else if (const auto *a = std::get_if<LinearAttrs>(&attributes)) {
    if (opcode == OpCode::Linear)
      width = a->output_width;
  } else if (const auto *a = std::get_if<GatedMatmulAttrs>(&attributes)) {
    if (opcode == OpCode::GatedMatmul)
      width = a->ffn_width;
  } else if (std::holds_alternative<LinearResidualAttrs>(attributes) &&
             opcode == OpCode::LinearResidual) {
    width = staged.type.dimension(1);
  }
  if (!width)
    return Fail(error, "missing result width or incompatible attributes");
  if (staged.access == Access::Write) {
    staged.type = result_type.rank() ? result_type :
        TensorType::matrix(input->dimension(0), width, ElementType::F32);
    if (!CheckSize(staged.type, error))
      return false;
    auto result = checked.tensor(staged.type, error);
    if (!result.valid())
      return false;
    invocation.bind(staged.port, result);
  }
  std::visit([&](const auto &a) { invocation.attributes(a); }, attributes);
  if (!invocation.append(error))
    return false;
  *out = std::move(staged);
  error->clear();
  return true;
}
} // namespace hrx::dispatch::shapes
