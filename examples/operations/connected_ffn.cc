// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <iostream>
#include <string>

#include "lib/operations/ffn_operations.h"

namespace {
using namespace hrx::operations;

bool BuildExample(std::string *error) {
  OperationRegistry registry;
  if (!DefineGatedMatmul(&registry, error) ||
      !DefineLinearResidual(&registry, error))
    return false;

  OperationGraph graph(registry);
  const uint64_t token_capacity = 8;
  const uint64_t hidden_width = 4096;
  const uint64_t ffn_width = 14336;
  auto normalized_type =
      TensorType::matrix(token_capacity, hidden_width, ElementType::F32);
  auto hidden_type =
      TensorType::matrix(token_capacity, ffn_width, ElementType::F32);
  auto gate_type =
      TensorType::matrix(hidden_width, ffn_width, ElementType::F32);
  gate_type.setEncoding(Encoding::Q8_0);
  gate_type.setLayout(Layout::GgmlWeightRows);
  auto down_type =
      TensorType::matrix(ffn_width, hidden_width, ElementType::F32);
  down_type.setEncoding(Encoding::Q8_0);
  down_type.setLayout(Layout::GgmlWeightRows);

  auto normalized = graph.tensor(normalized_type, error);
  if (!normalized.valid())
    return false;
  auto hidden = graph.tensor(hidden_type, error);
  if (!hidden.valid())
    return false;
  auto gate_weight = graph.tensor(gate_type, error);
  if (!gate_weight.valid())
    return false;
  auto up_weight = graph.tensor(gate_type, error);
  if (!up_weight.valid())
    return false;
  auto down_weight = graph.tensor(down_type, error);
  if (!down_weight.valid())
    return false;
  auto residual = graph.tensor(normalized_type, error);
  if (!residual.valid())
    return false;

  GatedMatmulAttrs gate_attrs;
  gate_attrs.input_width = hidden_width;
  gate_attrs.ffn_width = ffn_width;
  gate_attrs.activation = Activation::SiLU;
  if (!graph.operation(OpCode::GatedMatmul)
           .bind(Port::Input, normalized)
           .bind(Port::GateWeight, gate_weight)
           .bind(Port::UpWeight, up_weight)
           .bind(Port::Output, hidden)
           .attributes(gate_attrs)
           .append(error))
    return false;

  LinearResidualAttrs down_attrs;
  down_attrs.input_width = ffn_width;
  down_attrs.output_width = hidden_width;
  if (!graph.operation(OpCode::LinearResidual)
           .bind(Port::Input, hidden)
           .bind(Port::Weight, down_weight)
           .bind(Port::Residual, residual)
           .attributes(down_attrs)
           .append(error))
    return false;

  const auto &calls = graph.invocations();
  if (graph.values().size() != 6 || calls.size() != 2 ||
      calls[0].binding(Port::Output)->value() !=
          calls[1].binding(Port::Input)->value() ||
      graph.definition(calls[1].opcode())->port(Port::Residual)->access() !=
          Access::ReadWrite) {
    *error = "example graph is not connected as expected";
    return false;
  }
  for (const auto &call : calls)
    std::cout << OpCodeName(call.opcode()) << '\n';
  std::cout.flush();
  if (!std::cout) {
    *error = "writing example output failed";
    return false;
  }
  return true;
}
} // namespace

int main() {
  std::string error;
  if (!BuildExample(&error)) {
    std::cerr << "connected_ffn: " << error << '\n';
    return 1;
  }
  return 0;
}
