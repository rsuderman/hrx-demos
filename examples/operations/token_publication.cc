// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "examples/operations/example_utils.h"
#include "lib/operations/control_operations.h"
#include "lib/operations/row_operations.h"
using namespace hrx::operations;
using namespace hrx::operations::example;
bool Build(std::string *e) {
  OperationRegistry r;
  if (!DefineArgMax(&r, e) || !DefinePublishToken(&r, e))
    return false;
  OperationGraph g(r);
  auto logits = M(g, 1, 256, e), token = V(g, 1, ElementType::I32, e),
       control = V(g, 3, ElementType::I32, e);
  auto history = V(g, 65, ElementType::I32, e),
       progress = V(g, 8, ElementType::I32, e);
  ArgMaxAttrs argmax;
  argmax.width = 256;
  PublishTokenAttrs publication;
  publication.history_capacity = 64;
  publication.phase = Phase::Prefill;
  publication.terminal_policy = TerminalPolicy::StopOnTokenIds;
  publication.terminal_ids.push_back(2);
  if (!g.operation(OpCode::ArgMax)
           .bind(Port::Input, logits)
           .bind(Port::Output, token)
           .attributes(argmax)
           .append(e))
    return false;
  if (!g.operation(OpCode::PublishToken)
           .bind(Port::Token, token)
           .bind(Port::Control, control)
           .bind(Port::History, history)
           .bind(Port::Progress, progress)
           .attributes(publication)
           .append(e))
    return false;
  return Print(g, 2, e);
}
int main() {
  std::string error;
  if (!Build(&error)) {
    std::cerr << error << '\n';
    return 1;
  }
  return 0;
}
