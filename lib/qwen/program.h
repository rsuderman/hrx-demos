// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_QWEN_PROGRAM_H_
#define HRX_SOLUTIONS_LIB_QWEN_PROGRAM_H_
#include "lib/dispatch/qwen.h"
#include "lib/qwen/structure.h"
namespace hrx::qwen {
struct QwenProgramSettings {
  dispatch::QwenNativeOptions native;
  std::string source_asset;
};
struct QwenBufferRequirement {
  std::string name;
  uint64_t minimum_bytes = 0;
};
struct QwenProgram {
  std::shared_ptr<const QwenStructure> structure;
  loomify_gguf::ProgramRef program;
  dispatch::QwenNativeGeometry geometry;
  std::vector<QwenBufferRequirement> buffers;
};
std::vector<dispatch::Id> QwenProgramDispatches();
// Preload matching real numerical providers with LoadQwenNative. Output is
// unchanged on failure; discard the mutable native module after staging
// failure.
bool DefineQwenProgram(const model::ModelDescriptor &,
                       const QwenProgramSettings &, const dispatch::Registry &,
                       loomify_gguf::ModuleBuilder &, const std::string &name,
                       QwenProgram *, std::string *error);
} // namespace hrx::qwen
#endif
