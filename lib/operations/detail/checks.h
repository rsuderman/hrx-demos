// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_OPERATIONS_DETAIL_CHECKS_H_
#define HRX_SOLUTIONS_LIB_OPERATIONS_DETAIL_CHECKS_H_
#include "lib/operations/operation_graph.h"
namespace hrx::operations::detail {
inline bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
inline bool Ready(OperationRegistry *registry, std::string *error) {
  return registry && error ? true
                           : Fail(error, "registry or error output is null");
}
inline bool Tensor(const BoundOperands &o, Port p, ElementType e,
                   Encoding encoding, Layout layout, size_t rank,
                   std::string *error) {
  const auto *t = o.tensor(p);
  return t && t->rank() == rank && t->elementType() == e &&
                 t->encoding() == encoding && t->layout() == layout
             ? true
             : Fail(error, std::string(PortName(p)) +
                               " tensor representation/rank mismatch");
}
inline bool Vector(const BoundOperands &o, Port p, uint64_t n, ElementType e,
                   std::string *error) {
  return Tensor(o, p, e, Encoding::Dense, Layout::RowMajor, 1, error) &&
         (o.tensor(p)->dimension(0) == n ||
          Fail(error, std::string(PortName(p)) + " vector shape mismatch"));
}
inline bool Matrix(const BoundOperands &o, Port p, uint64_t rows,
                   uint64_t width, ElementType e, Encoding encoding,
                   Layout layout, std::string *error) {
  return Tensor(o, p, e, encoding, layout, 2, error) &&
         ((o.tensor(p)->dimension(0) == rows &&
           o.tensor(p)->dimension(1) == width) ||
          Fail(error, std::string(PortName(p)) + " matrix shape mismatch"));
}
inline bool Heads(const BoundOperands &o, Port p, uint64_t rows, uint64_t heads,
                  uint64_t width, ElementType e, std::string *error) {
  return Tensor(o, p, e, Encoding::Dense, Layout::RowMajor, 3, error) &&
         ((o.tensor(p)->dimension(0) == rows &&
           o.tensor(p)->dimension(1) == heads &&
           o.tensor(p)->dimension(2) == width) ||
          Fail(error, std::string(PortName(p)) + " head shape mismatch"));
}
inline bool RowControls(const BoundOperands &o, std::string *error) {
  const auto *begin = o.value(Port::RowBegin), *count = o.value(Port::RowCount);
  return begin && count && begin->scalarType() && count->scalarType() &&
                 *begin->scalarType() == ScalarType::I64 &&
                 *count->scalarType() == ScalarType::I64
             ? true
             : Fail(error, "row controls must be I64 scalars");
}
inline bool Weight(const BoundOperands &o, Port p, uint64_t input,
                   uint64_t output, std::string *error) {
  const auto *t = o.tensor(p);
  if (!t || t->rank() != 2 || t->layout() != Layout::GgmlWeightRows ||
      t->encoding() == Encoding::Q8_1X4 ||
      (t->elementType() != ElementType::F32 &&
       !(t->encoding() == Encoding::Dense &&
         t->elementType() == ElementType::F16)))
    return Fail(error, "weight representation mismatch");
  return (t->dimension(0) == input && t->dimension(1) == output) ||
         Fail(error, "weight axes mismatch");
}
template <typename... P>
bool Distinct(const BoundOperands &o, std::string *error, P... ports) {
  std::vector<Port> all;
  (all.push_back(ports), ...);
  for (size_t i = 0; i < all.size(); ++i)
    for (size_t j = i + 1; j < all.size(); ++j)
      if (o.sameValue(all[i], all[j]))
        return Fail(error, "operands require distinct handles");
  return true;
}
} // namespace hrx::operations::detail
#endif
