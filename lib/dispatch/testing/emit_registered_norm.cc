// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/operations/tensor_operations.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <memory>
using namespace hrx::dispatch;
using namespace hrx::operations;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto *d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  return false;
}

} // namespace
int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  loomc_target_environment_t *environment = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &environment)))
    return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t context_options{};
  context_options.next = &target;
  if (!Ok(loomc_context_create(&context_options, loomc_allocator_system(),
                               &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  auto registry = Registry::Llama3B();
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(Load(context, workspace, registry, {Id::RmsNorm, Id::DownPrefill},
               argv[1], &definitions, &result)) ||
      !Succeeded(result))
    return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module->DefineProgram("registered.norm", {}, 3, &program)))
    return 1;
  OperationRegistry ops;
  std::string error;
  if (!DefineRmsNorm(&ops, &error))
    return 1;
  OperationGraph graph(ops);
  auto x = graph.tensor(TensorType::matrix(2, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  if (!Ok(author.Bind(x, program->Binding(0))) ||
      !Ok(author.Bind(gamma, program->Binding(1))))
    return 1;
  ValueId y;
  if (!Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                        {{Port::Input, x}, {Port::Gamma, gamma}}, &y)))
    return 1;
  Value produced;
  if (!Ok(author.Native(y, &produced)))
    return 1;
  // Consume the actual internally allocated producer value with the existing
  // residual-add kernel. Adding the original input exposes norm errors rather
  // than canceling normalization scale, while keeping the result observable.
  CommandValues values(*program);
  Value rows, hidden;
  if (!Ok(values.Constant(2, LOOM_SCALAR_TYPE_INDEX, &rows)) ||
      !Ok(values.Constant(3072, LOOM_SCALAR_TYPE_INDEX, &hidden)))
    return 1;
  KernelRef observe;
  if (!Ok(module->Kernel("llm_ffn_residual_add_f32_prefill", &observe)) ||
      !Ok(program->Invoke(
          observe, {rows, hidden},
          {rows, program->Binding(0), produced, program->Binding(2)})) ||
      !Ok(program->Finish()))
    return 1;
  loomc_module_t *linked = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)) ||
      !Succeeded(result))
    return 1;
  loomc_result_release(result);
  loomc_module_serialize_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  bool ok = Ok(loomc_module_serialize_to_path(linked, &options,
                                              loomc_make_cstring_view(argv[2]),
                                              loomc_allocator_system()));
  loomc_module_release(linked);
  program.reset();
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
  return ok ? 0 : 1;
}
