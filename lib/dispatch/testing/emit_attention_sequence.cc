// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/attention_sequence.h"
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
  if (!Ok(LoadAttentionSequence(
          context, workspace, argv[1],
          "lib/dispatch/testing/testdata/attention_capacity.loom", &definitions,
          &result)) ||
      !Succeeded(result))
    return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  std::vector<RootRef> roots;
  for (bool decode : {false, true}) {
    if (!Ok(module->DefineProgram(decode ? "native.attention.decode"
                                         : "native.attention",
                                  {}, 8, &program)))
      return 1;
    AttentionSequenceBindings bindings{program->Binding(0), program->Binding(1),
                                       program->Binding(2), program->Binding(3),
                                       program->Binding(4), program->Binding(5),
                                       program->Binding(6)};
    AttentionSequenceOptions sequence_options;
    sequence_options.phase = decode ? Phase::Decode : Phase::Prefill;
    sequence_options.layer = 1;
    sequence_options.cache_base = 16;
    Value produced;
    if (!Ok(BuildAttentionSequence(*module, *program, sequence_options,
                                   bindings, &produced)))
      return 1;
    CommandValues values(*program);
    Value one, zero, origin;
    if (!Ok(values.Constant(1, LOOM_SCALAR_TYPE_INDEX, &one)) ||
        !Ok(values.Constant(0, LOOM_SCALAR_TYPE_INDEX, &zero)) ||
        !Ok(values.Constant(0, LOOM_SCALAR_TYPE_OFFSET, &origin)))
      return 1;
    KernelRef observe;
    if (!Ok(module->Kernel("llama32_attention_residual", &observe)) ||
        !Ok(program->Invoke(observe, {one},
                            {zero, origin, zero, bindings.control, produced,
                             program->Binding(7)})) ||
        !Ok(program->Finish()))
      return 1;
    roots.push_back(program->Ref());
    program.reset();
  }
  loomc_module_t *linked = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify(roots, &linked, &result)) || !Succeeded(result))
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
