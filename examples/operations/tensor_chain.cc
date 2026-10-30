// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "examples/operations/example_utils.h"
#include "lib/operations/tensor_operations.h"
using namespace hrx::operations;
using namespace hrx::operations::example;
bool Build(std::string *e) {
  OperationRegistry r;
  if (!DefineEmbedding(&r, e) || !DefineRmsNorm(&r, e) ||
      !DefineQuantize(&r, e) || !DefineLinear(&r, e))
    return false;
  OperationGraph g(r);
  auto ids = V(g, 4, ElementType::I32, e),
       emb = M(g, 128, 256, e, ElementType::F32, Encoding::Q8_0,
               Layout::GgmlWeightRows);
  auto x = M(g, 4, 128, e), gamma = V(g, 128, ElementType::F32, e),
       n = M(g, 4, 128, e);
  auto packed = M(g, 4, 128, e, ElementType::F32, Encoding::Q8_1X4,
                  Layout::PackedQ8_1X4Rows);
  auto weight = M(g, 128, 64, e, ElementType::F32, Encoding::Q8_0,
                  Layout::GgmlWeightRows),
       y = M(g, 4, 64, e);
  auto begin = g.scalar(ScalarType::I64, e),
       count = g.scalar(ScalarType::I64, e);
  EmbeddingAttrs embedding;
  embedding.embedding_width = 128;
  embedding.vocabulary_size = 256;
  RmsNormAttrs norm;
  norm.width = 128;
  QuantizeAttrs quant;
  quant.width = 128;
  LinearAttrs linear;
  linear.input_width = 128;
  linear.output_width = 64;
  if (!g.operation(OpCode::Embedding)
           .bind(Port::TokenIds, ids)
           .bind(Port::Weight, emb)
           .bind(Port::Output, x)
           .attributes(embedding)
           .append(e))
    return false;
  if (!g.operation(OpCode::RmsNorm)
           .bind(Port::Input, x)
           .bind(Port::Gamma, gamma)
           .bind(Port::Output, n)
           .attributes(norm)
           .append(e))
    return false;
  if (!g.operation(OpCode::Quantize)
           .bind(Port::Input, n)
           .bind(Port::Output, packed)
           .bind(Port::RowBegin, begin)
           .bind(Port::RowCount, count)
           .attributes(quant)
           .append(e))
    return false;
  if (!g.operation(OpCode::Linear)
           .bind(Port::Input, packed)
           .bind(Port::Weight, weight)
           .bind(Port::Output, y)
           .attributes(linear)
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
