// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

// Emit two host-authored programs from the supplied numerical kernel module.
// This example uses the same public builder API as a model author.
#include <cstdio>
#include <memory>
#include <string>

#include "tools/loomify-gguf/authoring/command/command_program_builder.h"

namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;

bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", static_cast<int>(message.size),
                 message.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result) {
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", static_cast<int>(d->message.size),
                   d->message.data);
    }
  }
  return false;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: emit_example KERNELS.loom OUTPUT.loombc\n");
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
  loomc_module_t *module = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(loomc_module_deserialize_from_path(
          context, workspace, loomc_make_cstring_view(argv[1]), nullptr,
          loomc_allocator_system(), &module, &result)))
    return 1;
  Owner<loomc_result_t, loomc_result_release> parse_result(result);
  Owner<loomc_module_t, loomc_module_release> owner(module);
  if (!Succeeded(result))
    return 1;
  using namespace hrx::loomify_gguf;
  std::unique_ptr<ModuleBuilder> authored;
  if (!Ok(ModuleBuilder::Create(context, workspace, module, &authored)))
    return 1;
  KernelRef add;
  if (!Ok(authored->Kernel("add_one", &add)))
    return 1;
  std::vector<RootRef> roots;
  // Pass kernels directly. Each definition shares the module's references;
  // there is no child wrapper or module clone for an individual invocation.
  for (const char *variant : {"two", "three"}) {
    KernelRef scale;
    if (!Ok(authored->Kernel(std::string("times_") + variant, &scale)))
      return 1;
    std::unique_ptr<CommandProgramBuilder> program;
    if (!Ok(authored->DefineProgram(std::string("example.times_") + variant,
                                    {LOOM_SCALAR_TYPE_INDEX}, 3, &program)))
      return 1;
    auto n = program->Specialization(0);
    auto input = program->Binding(0);
    auto intermediate = program->Binding(1);
    auto output = program->Binding(2);
    if (!Ok(program->Invoke(add, {n}, {n, input, intermediate})) ||
        !Ok(program->Invoke(scale, {n}, {n, intermediate, output})) ||
        !Ok(program->Finish()))
      return 1;
    roots.push_back(program->Ref());
  }
  module = nullptr;
  result = nullptr;
  if (!Ok(authored->LinkAndVerify(roots, &module, &result)))
    return 1;
  Owner<loomc_result_t, loomc_result_release> link_result(result);
  Owner<loomc_module_t, loomc_module_release> output(module);
  if (!Succeeded(result))
    return 1;
  loomc_module_serialize_options_t options = {};
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  if (!Ok(loomc_module_serialize_to_path(output.get(), &options,
                                         loomc_make_cstring_view(argv[2]),
                                         loomc_allocator_system())))
    return 1;
  return 0;
}
