// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <memory>

#include "tools/loomify-gguf/authoring/commands/command_values.h"

namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", static_cast<int>(message.size),
                 message.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", static_cast<int>(d->message.size),
                   d->message.data);
    }
  return false;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr,
                 "usage: emit_body_example NUMERICAL.loom OUTPUT.loombc\n");
    return 1;
  }
  loomc_context_t *context = nullptr;
  if (!Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)))
    return 1;
  Owner<loomc_context_t, loomc_context_release> context_owner(context);
  loomc_workspace_t *workspace = nullptr;
  if (!Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_owner(workspace);
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(loomc_module_deserialize_from_path(
          context, workspace, loomc_make_cstring_view(argv[1]), nullptr,
          loomc_allocator_system(), &definitions, &result)))
    return 1;
  Owner<loomc_module_t, loomc_module_release> definitions_owner(definitions);
  Owner<loomc_result_t, loomc_result_release> parse_result(result);
  if (!Succeeded(result))
    return 1;

  using namespace hrx::loomify_gguf;
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  KernelRef add, scale;
  if (!Ok(module->Kernel("add_bias", &add)) ||
      !Ok(module->Kernel("scale_result", &scale)))
    return 1;
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module->DefineProgram("body.example", {LOOM_SCALAR_TYPE_INDEX}, 3,
                                &program)))
    return 1;
  CommandValues values(*program);
  Value count, stride, byte_length, scratch, extent, bias, zero, predicate;
  if (!Ok(values.Config("example.width", &count)) ||
      !Ok(values.Constant(4, LOOM_SCALAR_TYPE_OFFSET, &stride)) ||
      !Ok(values.Scale(count, stride, &byte_length)) ||
      !Ok(values.Allocate(byte_length, 256, &scratch)) ||
      !Ok(values.Constant(1, LOOM_SCALAR_TYPE_INDEX, &extent)) ||
      !Ok(values.Constant(0, LOOM_SCALAR_TYPE_INDEX, &zero)) ||
      !Ok(values.Parameter(program->Binding(0), "blk.{}.bias", {zero},
                           LOOM_SCALAR_TYPE_I32, {extent}, &bias)) ||
      !Ok(program->Invoke(add, {count},
                          {count, bias, program->Binding(1), scratch})) ||
      !Ok(values.Equal(program->Specialization(0), zero, &predicate)))
    return 1;
  auto scaled = [&](int factor, CommandProgramBuilder &body) {
    Value scalar;
    auto status =
        CommandValues(body).Constant(factor, LOOM_SCALAR_TYPE_I32, &scalar);
    if (!loomc_status_is_ok(status))
      return status;
    return body.Invoke(scale, {count},
                       {count, scalar, scratch, program->Binding(2)});
  };
  if (!Ok(program->If(
          predicate,
          [&](CommandProgramBuilder &body) { return scaled(2, body); },
          [&](CommandProgramBuilder &body) { return scaled(3, body); })))
    return 1;
  auto root = program->Ref();
  if (!Ok(program->Finish()))
    return 1;
  loomc_module_t *output = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify({root}, &output, &result)))
    return 1;
  Owner<loomc_module_t, loomc_module_release> output_owner(output);
  Owner<loomc_result_t, loomc_result_release> link_result(result);
  if (!Succeeded(result))
    return 1;
  loomc_module_serialize_options_t options = {};
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  return Ok(loomc_module_serialize_to_path(output, &options,
                                           loomc_make_cstring_view(argv[2]),
                                           loomc_allocator_system()))
             ? 0
             : 1;
}
