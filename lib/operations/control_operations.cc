// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/operations/control_operations.h"
#include "lib/operations/detail/checks.h"
#include <algorithm>
#include <limits>
namespace hrx::operations {
namespace {
using namespace detail;
bool Publish(const BoundOperands &o, const PublishTokenAttrs &a,
             std::string *e) {
  if (!a.history_capacity ||
      a.history_capacity == std::numeric_limits<uint64_t>::max())
    return Fail(
        e,
        "history capacity must be positive and capacity+1 must not overflow");
  if (a.phase != Phase::Prefill && a.phase != Phase::Decode)
    return Fail(e, "unknown publication phase");
  if (a.terminal_policy != TerminalPolicy::None &&
      a.terminal_policy != TerminalPolicy::StopOnTokenIds)
    return Fail(e, "unknown terminal policy");
  if ((a.terminal_policy == TerminalPolicy::None && !a.terminal_ids.empty()) ||
      (a.terminal_policy == TerminalPolicy::StopOnTokenIds &&
       a.terminal_ids.empty()) ||
      std::any_of(a.terminal_ids.begin(), a.terminal_ids.end(),
                  [](int32_t id) { return id < 0; }))
    return Fail(e, "terminal policy/token IDs mismatch");
  return Vector(o, Port::Token, 1, ElementType::I32, e) &&
         Vector(o, Port::Control, 3, ElementType::I32, e) &&
         Vector(o, Port::History, a.history_capacity + 1, ElementType::I32,
                e) &&
         Vector(o, Port::Progress, 8, ElementType::I32, e) &&
         Distinct(o, e, Port::Token, Port::Control, Port::History,
                  Port::Progress);
}
} // namespace
bool DefinePublishToken(OperationRegistry *r, std::string *e) {
  return Ready(r, e) &&
         r->define(OpCode::PublishToken)
             .port(Port::Token, ValueKind::Tensor, Access::Read)
             .port(Port::Control, ValueKind::Tensor, Access::ReadWrite)
             .port(Port::History, ValueKind::Tensor, Access::ReadWrite)
             .port(Port::Progress, ValueKind::Tensor, Access::ReadWrite)
             .attributes<PublishTokenAttrs>()
             .verify(Publish)
             .finish(e);
}
} // namespace hrx::operations
