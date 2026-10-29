// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_LOOMIFY_COMMAND_VALUES_H_
#define HRX_SOLUTIONS_LOOMIFY_COMMAND_VALUES_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "tools/loomify-gguf/authoring/command/command_program_builder.h"

namespace hrx::loomify_gguf {

// Planning values and storage operations in one command-program body. Values
// retain the program's owner/scope checks, including inside structured regions.
class CommandValues {
public:
  explicit CommandValues(CommandProgramBuilder &program) : program_(program) {}

  loomc_status_t Constant(int64_t value, loom_scalar_type_t type, Value *out);
  loomc_status_t Config(std::string_view name, Value *out);
  loomc_status_t Cast(Value input, loom_scalar_type_t result_type, Value *out);
  loomc_status_t Add(Value lhs, Value rhs, Value *out);
  loomc_status_t Multiply(Value lhs, Value rhs, Value *out);
  loomc_status_t Divide(Value lhs, Value rhs, Value *out);
  loomc_status_t ShiftLeft(Value lhs, Value rhs, Value *out);
  loomc_status_t BitAnd(Value lhs, Value rhs, Value *out);
  loomc_status_t Scale(Value index, Value byte_stride, Value *out);
  loomc_status_t Equal(Value lhs, Value rhs, Value *out);
  loomc_status_t NotEqual(Value lhs, Value rhs, Value *out);
  loomc_status_t Select(Value condition, Value if_true, Value if_false,
                        Value *out);
  loomc_status_t Allocate(Value byte_length, int64_t alignment, Value *out);

  // A logical immutable parameter requirement, not a GGUF file-offset lookup.
  // Each '{}' is substituted by one index value. Dimensions are index values;
  // the resulting flattened/rank-2 view retains their dependent SSA identities.
  loomc_status_t Parameter(Value source, std::string_view key_pattern,
                           const std::vector<Value> &substitutions,
                           loom_scalar_type_t element_type,
                           const std::vector<Value> &dimensions, Value *out);

private:
  CommandProgramBuilder &program_;
};

} // namespace hrx::loomify_gguf
#endif // HRX_SOLUTIONS_LOOMIFY_COMMAND_VALUES_H_
