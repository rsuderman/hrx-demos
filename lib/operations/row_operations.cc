// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/row_operations.h"
#include "lib/operations/detail/checks.h"
#include <limits>
namespace hrx::operations {
namespace {
using namespace detail;
bool Copy(const BoundOperands &o, const CopyRowsAttrs &a, std::string *e) {
  if (!a.width || (a.selection != RowSelection::Range &&
                   a.selection != RowSelection::LastActive))
    return Fail(e, "copy_rows requires width and known row selection");
  const auto *x = o.tensor(Port::Input), *y = o.tensor(Port::Output);
  if (x->rank() != 2 || y->rank() != 2 || x->dimension(1) != a.width ||
      y->dimension(1) != a.width || x->elementType() != y->elementType() ||
      x->encoding() != Encoding::Dense || y->encoding() != Encoding::Dense ||
      x->layout() != Layout::RowMajor || y->layout() != Layout::RowMajor ||
      (a.selection == RowSelection::LastActive && y->dimension(0) != 1))
    return Fail(e, "copy_rows dense shape/type or selected output mismatch");
  return RowControls(o, e) && Distinct(o, e, Port::Input, Port::Output);
}
bool ArgMax(const BoundOperands &o, const ArgMaxAttrs &a, std::string *e) {
  if (!a.width || a.axis != 1 || a.tie_break != TieBreak::LowestIndex)
    return Fail(e,
                "argmax requires width, matrix width-axis and known tie break");
  if (a.width > uint64_t(std::numeric_limits<int32_t>::max()) + 1)
    return Fail(e, "argmax result indices must fit I32");
  auto n = o.tensor(Port::Input)->dimension(0);
  return Matrix(o, Port::Input, n, a.width, ElementType::F32, Encoding::Dense,
                Layout::RowMajor, e) &&
         Vector(o, Port::Output, n, ElementType::I32, e) &&
         Distinct(o, e, Port::Input, Port::Output);
}
} // namespace
bool DefineCopyRows(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::CopyRows)
             .port(Port::Input, ValueKind::Tensor, Access::Read,
                   AccessDomain::CopyRowsSource)
             .port(Port::Output, ValueKind::Tensor, Access::Write,
                   AccessDomain::CopyRowsDestination)
             .port(Port::RowBegin, ValueKind::Scalar, Access::Read)
             .port(Port::RowCount, ValueKind::Scalar, Access::Read)
             .attributes<CopyRowsAttrs>()
             .verify(Copy)
             .finish(e);
}
bool DefineArgMax(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::ArgMax)
             .port(Port::Input, ValueKind::Tensor, Access::Read)
             .port(Port::Output, ValueKind::Tensor, Access::Write)
             .attributes<ArgMaxAttrs>()
             .verify(ArgMax)
             .finish(e);
}
} // namespace hrx::operations
