// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/tensor_operations.h"
#include "lib/operations/detail/checks.h"
#include <cmath>
#include <limits>
namespace hrx::operations {
namespace {
using namespace detail;
bool Embedding(const BoundOperands &o, const EmbeddingAttrs &a,
               std::string *error) {
  if (!a.vocabulary_size || !a.embedding_width)
    return Fail(error, "embedding dimensions must be positive");
  auto n = o.tensor(Port::TokenIds)->dimension(0);
  return Vector(o, Port::TokenIds, n, ElementType::I32, error) &&
         Weight(o, Port::Weight, a.embedding_width, a.vocabulary_size, error) &&
         Matrix(o, Port::Output, n, a.embedding_width, ElementType::F32,
                Encoding::Dense, Layout::RowMajor, error) &&
         Distinct(o, error, Port::TokenIds, Port::Weight, Port::Output);
}
bool Norm(const BoundOperands &o, const RmsNormAttrs &a, std::string *error) {
  if (!a.width || !std::isfinite(a.epsilon) || a.epsilon <= 0)
    return Fail(error, "RMSNorm width/epsilon must be positive and finite");
  auto n = o.tensor(Port::Input)->dimension(0);
  return Matrix(o, Port::Input, n, a.width, ElementType::F32, Encoding::Dense,
                Layout::RowMajor, error) &&
         Vector(o, Port::Gamma, a.width, ElementType::F32, error) &&
         Matrix(o, Port::Output, n, a.width, ElementType::F32, Encoding::Dense,
                Layout::RowMajor, error) &&
         Distinct(o, error, Port::Output, Port::Input, Port::Gamma);
}
bool OutputHead(const BoundOperands &o, const OutputHeadAttrs &a,
                std::string *error) {
  if (!a.width || !a.vocabulary_size ||
      a.vocabulary_size > uint64_t(std::numeric_limits<int32_t>::max()) + 1 ||
      !std::isfinite(a.epsilon) ||
      a.epsilon <= 0 || a.row_selection != RowSelection::LastActive)
    return Fail(error, "output head requires positive dimensions/epsilon and last-active selection");
  auto rows = o.tensor(Port::Input)->dimension(0);
  return Matrix(o, Port::Input, rows, a.width, ElementType::F32,
                Encoding::Dense, Layout::RowMajor, error) &&
         Vector(o, Port::Gamma, a.width, ElementType::F32, error) &&
         Vector(o, Port::Control, 3, ElementType::I32, error) &&
         Weight(o, Port::Weight, a.width, a.vocabulary_size, error) &&
         Vector(o, Port::Output, 1, ElementType::I32, error) &&
         ((!o.value(Port::RowBegin) && !o.value(Port::RowCount)) ||
          RowControls(o, error)) &&
         Distinct(o, error, Port::Output, Port::Input, Port::Gamma, Port::Weight, Port::Control);
}
bool Quantize(const BoundOperands &o, const QuantizeAttrs &a,
              std::string *error) {
  if (!a.width || a.width % 128 ||
      (a.policy != PackingPolicy::Canonical &&
       a.policy != PackingPolicy::QkCompat))
    return Fail(
        error,
        "quantize requires width multiple of 128 and known packing policy");
  auto n = o.tensor(Port::Input)->dimension(0);
  return Matrix(o, Port::Input, n, a.width, ElementType::F32, Encoding::Dense,
                Layout::RowMajor, error) &&
         Matrix(o, Port::Output, n, a.width, ElementType::F32, Encoding::Q8_1X4,
                Layout::PackedQ8_1X4Rows, error) &&
         RowControls(o, error) && Distinct(o, error, Port::Input, Port::Output);
}
bool Linear(const BoundOperands &o, const LinearAttrs &a, std::string *error) {
  if (!a.input_width || !a.output_width ||
      a.numerical_policy != NumericalPolicy::F32Accumulate)
    return Fail(error,
                "linear requires positive widths and known numerical policy");
  const auto *input = o.tensor(Port::Input);
  bool dense = input->encoding() == Encoding::Dense &&
               input->layout() == Layout::RowMajor &&
               (input->elementType() == ElementType::F32 ||
                input->elementType() == ElementType::F16);
  bool packed = input->encoding() == Encoding::Q8_1X4 &&
                input->layout() == Layout::PackedQ8_1X4Rows &&
                input->elementType() == ElementType::F32 &&
                a.input_width % 128 == 0;
  if (input->rank() != 2 || input->dimension(1) != a.input_width ||
      (!dense && !packed))
    return Fail(error, "linear input shape/representation mismatch");
  return Weight(o, Port::Weight, a.input_width, a.output_width, error) &&
         Matrix(o, Port::Output, input->dimension(0), a.output_width,
                ElementType::F32, Encoding::Dense, Layout::RowMajor, error) &&
         Distinct(o, error, Port::Input, Port::Weight, Port::Output);
}
} // namespace
bool DefineEmbedding(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::Embedding)
             .port(Port::TokenIds, ValueKind::Tensor, Access::Read,
                   AccessDomain::ActiveRows)
             .port(Port::Weight, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write,
                   AccessDomain::ActiveRows)
             .optionalPort(Port::RowBegin, ValueKind::Scalar, Access::Read)
             .optionalPort(Port::RowCount, ValueKind::Scalar, Access::Read)
             .attributes<EmbeddingAttrs>()
             .verify(Embedding)
             .finish(e);
}
bool DefineRmsNorm(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::RmsNorm)
             .port(Port::Input, ValueKind::Tensor, Access::Read,
                   AccessDomain::ActiveRows)
             .port(Port::Gamma, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write,
                   AccessDomain::ActiveRows)
             .optionalPort(Port::RowBegin, ValueKind::Scalar, Access::Read)
             .optionalPort(Port::RowCount, ValueKind::Scalar, Access::Read)
             .attributes<RmsNormAttrs>()
             .verify(Norm)
             .finish(e);
}
bool DefineOutputHead(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::OutputHead)
             .port(Port::Input, ValueKind::Tensor, Access::Read,
                   AccessDomain::ActiveRows)
             .port(Port::Control, ValueKind::Tensor, Access::Read)
             .port(Port::Gamma, ValueKind::Tensor, Access::Read)
             .port(Port::Weight, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write)
             .optionalPort(Port::RowBegin, ValueKind::Scalar, Access::Read)
             .optionalPort(Port::RowCount, ValueKind::Scalar, Access::Read)
             .attributes<OutputHeadAttrs>()
             .verify(OutputHead)
             .finish(e);
}
bool DefineQuantize(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::Quantize)
             .port(Port::Input, ValueKind::Tensor, Access::Read,
                   AccessDomain::ActiveRows)
             .port(Port::Output, ValueKind::Tensor, Access::Write,
                   AccessDomain::ActiveRows)
             .port(Port::RowBegin, ValueKind::Scalar, Access::Read)
             .port(Port::RowCount, ValueKind::Scalar, Access::Read)
             .attributes<QuantizeAttrs>()
             .verify(Quantize)
             .finish(e);
}
bool DefineLinear(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::Linear)
             .port(Port::Input, ValueKind::Tensor, Access::Read,
                   AccessDomain::ActiveRows)
             .port(Port::Weight, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write,
                   AccessDomain::ActiveRows)
             .optionalPort(Port::RowBegin, ValueKind::Scalar, Access::Read)
             .optionalPort(Port::RowCount, ValueKind::Scalar, Access::Read)
             .attributes<LinearAttrs>()
             .verify(Linear)
             .finish(e);
}
} // namespace hrx::operations
