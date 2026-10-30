// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/control_operations.h"
#include "lib/operations/row_operations.h"
#include "lib/operations/testing/operation_test_util.h"
#include <limits>
namespace hrx::operations {
namespace {
using namespace test;
TEST(ControlOperations, ArgMaxPublicationPoliciesAreCopiedAndEffectsExplicit) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefineArgMax(&r, &e));
  ASSERT_TRUE(DefinePublishToken(&r, &e));
  OperationGraph g(r);
  auto logits = M(g, 1, 7), token = V(g, 1, ElementType::I32),
       control = V(g, 3, ElementType::I32), history = V(g, 9, ElementType::I32),
       progress = V(g, 8, ElementType::I32);
  ArgMaxAttrs argmax;
  argmax.width = 7;
  ASSERT_TRUE(g.operation(OpCode::ArgMax)
                  .bind(Port::Input, logits)
                  .bind(Port::Output, token)
                  .attributes(argmax)
                  .append(&e));
  PublishTokenAttrs a;
  a.history_capacity = 8;
  a.terminal_policy = TerminalPolicy::StopOnTokenIds;
  a.terminal_ids.push_back(2);
  auto b = g.operation(OpCode::PublishToken);
  b.bind(Port::Token, token)
      .bind(Port::Control, control)
      .bind(Port::History, history)
      .bind(Port::Progress, progress)
      .attributes(a);
  a.terminal_ids[0] = 6;
  ASSERT_TRUE(b.append(&e)) << e;
  EXPECT_EQ(std::get<PublishTokenAttrs>(g.invocations()[1].attributes())
                .terminal_ids[0],
            2);
  EXPECT_EQ(g.invocations()[0].binding(Port::Output)->value(),
            g.invocations()[1].binding(Port::Token)->value());
  EXPECT_EQ(r.find(OpCode::PublishToken)->port(Port::Control)->access(),
            Access::ReadWrite);
  EXPECT_EQ(r.find(OpCode::PublishToken)->port(Port::History)->access(),
            Access::ReadWrite);
  EXPECT_EQ(r.find(OpCode::PublishToken)->port(Port::Progress)->access(),
            Access::ReadWrite);
  a.phase = Phase::Decode;
  a.terminal_policy = TerminalPolicy::None;
  a.terminal_ids.clear();
  ASSERT_TRUE(g.operation(OpCode::PublishToken)
                  .bind(Port::Token, token)
                  .bind(Port::Control, control)
                  .bind(Port::History, history)
                  .bind(Port::Progress, progress)
                  .attributes(a)
                  .append(&e));
  EXPECT_EQ(g.invocations().size(), 3);
}
TEST(ControlOperations, SchemaPolicyAndCapacityFailuresRollBack) {
  OperationRegistry r;
  std::string e;
  ASSERT_TRUE(DefinePublishToken(&r, &e));
  OperationGraph g(r);
  auto token = V(g, 1, ElementType::I32), control = V(g, 3, ElementType::I32),
       history = V(g, 9, ElementType::I32),
       progress = V(g, 8, ElementType::I32);
  PublishTokenAttrs a;
  a.history_capacity = 8;
  auto append = [&](ValueId t, ValueId c, ValueId h, ValueId p) {
    return g.operation(OpCode::PublishToken)
        .bind(Port::Token, t)
        .bind(Port::Control, c)
        .bind(Port::History, h)
        .bind(Port::Progress, p)
        .attributes(a)
        .append(&e);
  };
  ASSERT_TRUE(append(token, control, history, progress)) << e;
  EXPECT_FALSE(append(V(g, 1, ElementType::I64), control, history, progress));
  EXPECT_FALSE(append(token, V(g, 4, ElementType::I32), history, progress));
  EXPECT_FALSE(append(token, control, V(g, 8, ElementType::I32), progress));
  EXPECT_FALSE(append(token, control, history, M(g, 1, 8, ElementType::I32)));
  a.history_capacity = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(append(token, control, history, progress));
  EXPECT_NE(e.find("overflow"), std::string::npos);
  a.history_capacity = 0;
  EXPECT_FALSE(append(token, control, history, progress));
  a.history_capacity = 8;
  a.phase = static_cast<Phase>(99);
  EXPECT_FALSE(append(token, control, history, progress));
  a.phase = Phase::Prefill;
  a.terminal_policy = static_cast<TerminalPolicy>(99);
  EXPECT_FALSE(append(token, control, history, progress));
  a.terminal_policy = TerminalPolicy::StopOnTokenIds;
  EXPECT_FALSE(append(token, control, history, progress));
  a.terminal_ids.push_back(-1);
  EXPECT_FALSE(append(token, control, history, progress));
  a.terminal_ids[0] = 2;
  a.terminal_policy = TerminalPolicy::None;
  EXPECT_FALSE(append(token, control, history, progress));
  a.terminal_ids.clear();
  a.history_capacity = 2;
  EXPECT_FALSE(append(token, control, control, progress));
  EXPECT_NE(e.find("distinct"), std::string::npos);
  EXPECT_EQ(g.invocations().size(), 1);
}
} // namespace
} // namespace hrx::operations
