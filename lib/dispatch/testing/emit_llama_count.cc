// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/operations/control_operations.h"
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
  loomc_context_options_t options{};
  options.next = &target;
  if (!Ok(loomc_context_create(&options, loomc_allocator_system(), &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  auto registry = Registry::Llama3B();
  auto entry = *registry.Find(Id::PublishPrefill);
  entry.provider_files.push_back(
      "lib/dispatch/testing/testdata/count_observe.loom");
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(Load(context, workspace, entry, argv[1], &definitions, &result)) ||
      !Succeeded(result))
    return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  std::vector<RootRef> roots;
  for (bool decode : {false, true}) {
    const std::string name =
        decode ? "native.count.decode" : "native.count.prefill";
    std::unique_ptr<CommandProgramBuilder> program;
    if (!Ok(module->DefineProgram(name + ".step", {}, 5, &program)))
      return 1;
    OperationRegistry defs;
    std::string error;
    if (!DefinePublishToken(&defs, &error))
      return 1;
    OperationGraph graph(defs);
    auto raw = graph.tensor(TensorType::vector(1, ElementType::I32), &error);
    auto packet = graph.tensor(TensorType::vector(3, ElementType::I32), &error);
    auto history =
        graph.tensor(TensorType::vector(129, ElementType::I32), &error);
    auto progress =
        graph.tensor(TensorType::vector(8, ElementType::I32), &error);
    NativeLlamaRows rows;
    rows.row_begin = graph.scalar(ScalarType::I64, &error);
    rows.row_count = graph.scalar(ScalarType::I64, &error);
    rows.count_control =
        graph.tensor(TensorType::vector(3, ElementType::I32), &error);
    rows.row_capacity = decode ? 1 : 8;
    rows.phase = decode ? Phase::Decode : Phase::Prefill;
    rows.packet = program->Binding(1);
    PublishTokenAttrs attrs{rows.phase,
                            128,
                            TerminalPolicy::StopOnTokenIds,
                            {128001, 128008, 128009}};
    if (!graph.operation(OpCode::PublishToken)
             .bind(Port::Token, raw)
             .bind(Port::Control, packet)
             .bind(Port::History, history)
             .bind(Port::Progress, progress)
             .attributes(attrs)
             .append(&error))
      return 1;
    NativeEmissionSession session(registry, graph, *module, *program);
    if (!Ok(session.Bind(raw, program->Binding(0))) ||
        !Ok(session.Bind(packet, program->Binding(1))) ||
        !Ok(session.Bind(history, program->Binding(2))) ||
        !Ok(session.Bind(progress, program->Binding(3))) ||
        !Ok(session.LlamaRows(rows)))
      return 1;
    if (!Ok(session.EmitExisting(decode ? Id::PublishDecode
                                        : Id::PublishPrefill,
                                 graph.invocations()[0])))
      return 1;
    Value count;
    KernelRef observe;
    if (!Ok(session.Native(rows.count_control, &count)) ||
        !Ok(module->Kernel("test.observe_count", &observe)) ||
        !Ok(program->Invoke(observe, {}, {count, program->Binding(4)})) ||
        !Ok(program->Finish()))
      return 1;
    auto step = program->Ref();
    program.reset();
    if (!Ok(module->DefineProgram(name, {}, 6, &program)))
      return 1;
    std::vector<Value> bindings;
    for (unsigned i = 0; i < 5; ++i)
      bindings.push_back(program->Binding(i));
    KernelRef initialize;
    if (!Ok(module->Kernel(decode ? "test.init_decode" : "test.init_prefill",
                           &initialize)) ||
        !Ok(program->Invoke(initialize, {}, {program->Binding(1)})) ||
        !Ok(program->Call(step, {}, bindings)))
      return 1;
    if (!decode) {
      KernelRef reset;
      if (!Ok(module->Kernel("test.reset_prefill", &reset)) ||
          !Ok(program->Invoke(reset, {}, {program->Binding(1)})))
        return 1;
    }
    bindings[4] = program->Binding(5);
    if (!Ok(program->Call(step, {}, bindings)) || !Ok(program->Finish()))
      return 1;
    roots.push_back(program->Ref());
    program.reset();
  }
  loomc_module_t *linked = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify(roots, &linked, &result)) || !Succeeded(result))
    return 1;
  loomc_result_release(result);
  loomc_module_serialize_options_t serialized{};
  serialized.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  bool ok = Ok(loomc_module_serialize_to_path(linked, &serialized,
                                              loomc_make_cstring_view(argv[2]),
                                              loomc_allocator_system()));
  loomc_module_release(linked);
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
  return ok ? 0 : 1;
}
