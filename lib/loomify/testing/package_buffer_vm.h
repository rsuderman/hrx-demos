// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TESTING_PACKAGE_BUFFER_VM_H_
#define HRX_SOLUTIONS_TESTING_PACKAGE_BUFFER_VM_H_
#include <string>
namespace hrx::loomify_gguf::testing {
// Invoke one actual package host root with ABI () -> buffer. Optionally reload
// the serialized control through native bytecode before compiling it to the VM.
bool InvokePackageBuffer(const std::string &path, const char *root,
                         bool bytecode, std::string *value, std::string *error);
} // namespace hrx::loomify_gguf::testing
#endif
