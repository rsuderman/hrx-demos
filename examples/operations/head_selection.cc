// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "examples/operations/example_utils.h"
#include "lib/operations/row_operations.h"
#include "lib/operations/tensor_operations.h"
using namespace hrx::operations;
using namespace hrx::operations::example;
bool Build(std::string *e) {
  OperationRegistry r;
  if (!DefineCopyRows(&r, e) || !DefineRmsNorm(&r, e) || !DefineLinear(&r, e) ||
      !DefineArgMax(&r, e))
    return false;
  OperationGraph g(r);
  auto source = M(g, 8, 128, e), one = M(g, 1, 128, e),
       gamma = V(g, 128, ElementType::F32, e), normalized = M(g, 1, 128, e);
  auto weight = M(g, 128, 256, e, ElementType::F32, Encoding::Q8_0,
                  Layout::GgmlWeightRows),
       logits = M(g, 1, 256, e), ids = V(g, 1, ElementType::I32, e);
  auto begin = g.scalar(ScalarType::I64, e),
       count = g.scalar(ScalarType::I64, e);
  CopyRowsAttrs copy;
  copy.width = 128;
  copy.selection = RowSelection::LastActive;
  RmsNormAttrs norm;
  norm.width = 128;
  LinearAttrs linear;
  linear.input_width = 128;
  linear.output_width = 256;
  ArgMaxAttrs argmax;
  argmax.width = 256;
  if (!g.operation(OpCode::CopyRows)
           .bind(Port::Input, source)
           .bind(Port::Output, one)
           .bind(Port::RowBegin, begin)
           .bind(Port::RowCount, count)
           .attributes(copy)
           .append(e))
    return false;
  if (!g.operation(OpCode::RmsNorm)
           .bind(Port::Input, one)
           .bind(Port::Gamma, gamma)
           .bind(Port::Output, normalized)
           .attributes(norm)
           .append(e))
    return false;
  if (!g.operation(OpCode::Linear)
           .bind(Port::Input, normalized)
           .bind(Port::Weight, weight)
           .bind(Port::Output, logits)
           .attributes(linear)
           .append(e))
    return false;
  if (!g.operation(OpCode::ArgMax)
           .bind(Port::Input, logits)
           .bind(Port::Output, ids)
           .attributes(argmax)
           .append(e))
    return false;
  return Print(g, 4, e);
}
int main() {
  std::string error;
  if (!Build(&error)) {
    std::cerr << error << '\n';
    return 1;
  }
  return 0;
}
