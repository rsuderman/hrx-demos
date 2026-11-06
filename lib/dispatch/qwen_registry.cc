// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/qwen_output.h"
namespace hrx::dispatch {
std::vector<Registration> QwenCommonRegistrations() {
  return QwenOutputRegistrations();
}
} // namespace hrx::dispatch
