// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/target/amdgpu.h"
#include "tools/loomify-gguf/serving/epoch.h"
#include <cstdio>
#include <memory>
using namespace hrx::loomify_gguf;
bool Ok(loomc_status_t s) {
  bool ok = loomc_status_is_ok(s);
  if (!ok) {
    auto m = loomc_status_message(s);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(s);
  return ok;
}
bool Result(loomc_result_t *r) {
  bool ok = r && loomc_result_succeeded(r);
  if (!ok && r)
    for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
      auto d = loomc_result_diagnostic_at(r, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  return ok;
}
int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  loomc_target_environment_t *env = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &env)))
    return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = env;
  loomc_context_options_t options{};
  options.next = &target;
  if (!Ok(loomc_context_create(&options, loomc_allocator_system(), &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  std::unique_ptr<ModuleBuilder> module;
  for (int i = 1; i < 2; ++i) {
    loomc_module_t *provider = nullptr;
    loomc_result_t *r = nullptr;
    if (!Ok(loomc_module_deserialize_from_path(
            context, workspace, loomc_make_cstring_view(argv[i]), nullptr,
            loomc_allocator_system(), &provider, &r)) ||
        !Result(r))
      return 1;
    loomc_result_release(r);
    bool ok =
        module
            ? Ok(module->AddProvider(provider))
            : Ok(ModuleBuilder::Create(context, workspace, provider, &module));
    loomc_module_release(provider);
    if (!ok)
      return 1;
  }
  std::vector<RootRef> roots;
  for (uint64_t capacity : {8, 256}) {
    auto name = "test.epoch" + std::to_string(capacity);
    std::unique_ptr<CommandProgramBuilder> p;
    if (!Ok(module->DefineProgram(name + ".prefill", {}, 7, &p)))
      return 1;
    Value count;
    KernelRef compute;
    if (!Ok(p->IndexConstant(capacity, &count)) ||
        !Ok(module->Kernel("test.epoch_compute", &compute)) ||
        !Ok(p->Invoke(compute, {count},
                      {count, p->Binding(1), p->Binding(2), p->Binding(4),
                       p->Binding(5)})) ||
        !Ok(p->Finish()))
      return 1;
    auto prefill = p->Ref();
    p.reset();
    ProgramRef epoch;
    if (!Ok(DefineLlamaEpoch(*module, prefill, capacity, name + ".single",
                             &epoch)))
      return 1;
    for (int kind = 0; kind < 3; ++kind) {
      if (!Ok(module->DefineProgram(name + "." + std::to_string(kind), {}, 7,
                                    &p)))
        return 1;
      std::vector<Value> bindings;
      for (unsigned i = 0; i < 7; ++i)
        bindings.push_back(p->Binding(i));
      KernelRef continuation, initialize;
      Value choice;
      if (!Ok(p->IndexConstant(kind, &choice)) ||
          !Ok(module->Kernel("test.epoch_initialize", &initialize)) ||
          !Ok(p->Invoke(initialize, {},
                        {choice, p->Binding(2), p->Binding(5)})) ||
          !Ok(module->Kernel("test.epoch_continue", &continuation)) ||
          !Ok(p->Call(epoch, {}, bindings)) ||
          !Ok(p->Invoke(continuation, {}, {p->Binding(2), p->Binding(5)})) ||
          !Ok(p->Call(epoch, {}, bindings)) || !Ok(p->Finish()))
        return 1;
      roots.push_back(p->Ref());
      p.reset();
    }
  }
  loomc_module_t *out = nullptr;
  loomc_result_t *r = nullptr;
  if (!Ok(module->LinkAndVerify(roots, &out, &r)) || !Result(r))
    return 1;
  loomc_module_serialize_options_t s{};
  s.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  bool ok = Ok(loomc_module_serialize_to_path(
      out, &s, loomc_make_cstring_view(argv[2]), loomc_allocator_system()));
  loomc_result_release(r);
  loomc_module_release(out);
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(env);
  return ok ? 0 : 1;
}
