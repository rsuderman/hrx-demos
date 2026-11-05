// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/epoch.h"
#include "tools/loomify-gguf/authoring/commands/command_values.h"
#include <algorithm>
namespace hrx::loomify_gguf {
#define TRY(expr)                                                              \
  do {                                                                         \
    auto status = (expr);                                                      \
    if (!loomc_status_is_ok(status))                                           \
      return status;                                                           \
    loomc_status_free(status);                                                 \
  } while (0)
loomc_status_t DefineLlamaEpoch(ModuleBuilder &module, ProgramRef prefill,
                                uint64_t capacity, std::string_view name,
                                ProgramRef *out) {
  if (!out || capacity < 1 || capacity > 512)
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "epoch requires capacity in [1,512] and output");
  KernelRef unpack, scatter;
  TRY(module.Kernel("loomify_epoch_unpack", &unpack));
  TRY(module.Kernel("loomify_epoch_scatter", &scatter));
  std::unique_ptr<CommandProgramBuilder> program;
  TRY(module.DefineProgram(name, {}, 7, &program));
  CommandValues values(*program);
  auto allocate = [&](uint64_t bytes, Value *out) {
    Value length;
    auto status = values.Constant(bytes, LOOM_SCALAR_TYPE_OFFSET, &length);
    if (!loomc_status_is_ok(status))
      return status;
    loomc_status_free(status);
    return values.Allocate(length, 256, out);
  };
  Value control, history, progress, tokens;
  TRY(allocate(12, &control));
  TRY(allocate(4 * std::max<uint64_t>(capacity, 129), &history));
  TRY(allocate(32, &progress));
  TRY(program->IndexConstant(capacity, &tokens));
  TRY(program->Invoke(unpack, {tokens},
                      {tokens, program->Binding(2), program->Binding(5),
                       control, history, progress}));
  TRY(program->Call(prefill, {},
                    {program->Binding(0), program->Binding(1), control,
                     program->Binding(3), program->Binding(4), history,
                     progress}));
  TRY(program->Invoke(scatter, {},
                      {program->Binding(2), history, program->Binding(6)}));
  TRY(program->Finish());
  *out = program->Ref();
  return loomc_ok_status();
}
#undef TRY
} // namespace hrx::loomify_gguf
