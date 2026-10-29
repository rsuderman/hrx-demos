// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "lib/operations/ffn_operations.h"

namespace hrx::operations {
namespace {

bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}

bool CheckTensor(const BoundOperands &operands, Port port, uint64_t rows,
                 uint64_t width, Encoding encoding, Layout layout,
                 std::string *error) {
  const auto *type = operands.tensor(port);
  if (!type || type->rank() != 2 || type->elementType() != ElementType::F32 ||
      type->encoding() != encoding || type->layout() != layout)
    return Fail(error, std::string(PortName(port)) +
                           " has an incompatible tensor representation");
  if (type->dimension(0) != rows || type->dimension(1) != width)
    return Fail(error, std::string(PortName(port)) +
                           " has an incompatible matrix shape");
  return true;
}

bool VerifyGatedMatmul(const BoundOperands &operands,
                       const GatedMatmulAttrs &attrs, std::string *error) {
  if (!error)
    return false;
  if (!attrs.input_width || !attrs.ffn_width)
    return Fail(error, "gated_matmul widths must be positive");
  if (attrs.activation != Activation::SiLU)
    return Fail(error, "gated_matmul requires SiLU activation");
  const auto *input = operands.tensor(Port::Input);
  if (!input || !input->dimension(0))
    return Fail(error, "gated_matmul requires positive input rows");
  const auto rows = input->dimension(0);
  if (!CheckTensor(operands, Port::Input, rows, attrs.input_width,
                   Encoding::Dense, Layout::RowMajor, error) ||
      !CheckTensor(operands, Port::GateWeight, attrs.input_width,
                   attrs.ffn_width, Encoding::Q8_0, Layout::GgmlWeightRows,
                   error) ||
      !CheckTensor(operands, Port::UpWeight, attrs.input_width, attrs.ffn_width,
                   Encoding::Q8_0, Layout::GgmlWeightRows, error) ||
      !CheckTensor(operands, Port::Output, rows, attrs.ffn_width,
                   Encoding::Dense, Layout::RowMajor, error))
    return false;
  if (operands.sameValue(Port::Output, Port::Input) ||
      operands.sameValue(Port::Output, Port::GateWeight) ||
      operands.sameValue(Port::Output, Port::UpWeight))
    return Fail(error, "gated_matmul output must differ from read operands");
  error->clear();
  return true;
}

bool VerifyLinearResidual(const BoundOperands &operands,
                          const LinearResidualAttrs &attrs,
                          std::string *error) {
  if (!error)
    return false;
  if (!attrs.input_width || !attrs.output_width)
    return Fail(error, "linear_residual widths must be positive");
  const auto *input = operands.tensor(Port::Input);
  if (!input || !input->dimension(0))
    return Fail(error, "linear_residual requires positive input rows");
  const auto rows = input->dimension(0);
  if (!CheckTensor(operands, Port::Input, rows, attrs.input_width,
                   Encoding::Dense, Layout::RowMajor, error) ||
      !CheckTensor(operands, Port::Weight, attrs.input_width,
                   attrs.output_width, Encoding::Q8_0, Layout::GgmlWeightRows,
                   error) ||
      !CheckTensor(operands, Port::Residual, rows, attrs.output_width,
                   Encoding::Dense, Layout::RowMajor, error))
    return false;
  if (operands.sameValue(Port::Input, Port::Weight) ||
      operands.sameValue(Port::Input, Port::Residual) ||
      operands.sameValue(Port::Weight, Port::Residual))
    return Fail(error, "linear_residual operands must have distinct handles");
  error->clear();
  return true;
}

} // namespace

bool DefineGatedMatmul(OperationRegistry *registry, std::string *error) {
  if (!registry || !error)
    return Fail(error, "registry is null or error output is missing");
  return registry->define(OpCode::GatedMatmul)
      .port(Port::Input, ValueKind::Tensor, Access::Read)
      .port(Port::GateWeight, ValueKind::Tensor, Access::Read)
      .port(Port::UpWeight, ValueKind::Tensor, Access::Read)
      .port(Port::Output, ValueKind::Tensor, Access::Write)
      .attributes<GatedMatmulAttrs>()
      .verify(VerifyGatedMatmul)
      .finish(error);
}

bool DefineLinearResidual(OperationRegistry *registry, std::string *error) {
  if (!registry || !error)
    return Fail(error, "registry is null or error output is missing");
  return registry->define(OpCode::LinearResidual)
      .port(Port::Input, ValueKind::Tensor, Access::Read)
      .port(Port::Weight, ValueKind::Tensor, Access::Read)
      .port(Port::Residual, ValueKind::Tensor, Access::ReadWrite)
      .attributes<LinearResidualAttrs>()
      .verify(VerifyLinearResidual)
      .finish(error);
}

} // namespace hrx::operations
