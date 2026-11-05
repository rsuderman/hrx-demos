// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"
#include "loomc/iree.h"
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
  if (!Ok(LoadLlamaNative(context, workspace, registry, {Id::AttentionOutput},
                          argv[1], AttentionNativeConfig{1, 512, 0},
                          &definitions, &result)) ||
      !Succeeded(result))
    return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  std::vector<RootRef> roots;
  for (bool q4 : {true, false})
    for (bool decode : {false, true}) {
      const std::string name = std::string("attention.output.") +
                               (q4 ? "q4" : "q6") +
                               (decode ? ".decode" : ".prefill");
      if (!Ok(module->DefineProgram(name, {}, 4, &program)))
        return 1;
      OperationRegistry ops;
      std::string error;
      if (!DefineLinearResidual(&ops, &error))
        return 1;
      OperationGraph graph(ops);
      const uint64_t capacity = decode ? 1 : 8;
      auto x = graph.tensor(
          TensorType::matrix(capacity, 3072, ElementType::F32), &error);
      auto residual = graph.tensor(
          TensorType::matrix(capacity, 3072, ElementType::F32), &error);
      auto type = TensorType::matrix(3072, 3072, ElementType::F32);
      type.setLayout(Layout::GgmlWeightRows)
          .setEncoding(q4 ? Encoding::Q4_K : Encoding::Q6_K);
      auto w = graph.tensor(type, &error);
      NativeLlamaRows rows;
      rows.row_begin = graph.scalar(ScalarType::I64, &error);
      rows.row_count = graph.scalar(ScalarType::I64, &error);
      rows.row_capacity = capacity;
      rows.token_begin = decode ? 0 : 2;
      rows.phase = decode ? Phase::Decode : Phase::Prefill;
      rows.packet = program->Binding(2);
      if (!graph.operation(OpCode::LinearResidual)
               .bind(Port::Input, x)
               .bind(Port::Weight, w)
               .bind(Port::Residual, residual)
               .bind(Port::RowBegin, rows.row_begin)
               .bind(Port::RowCount, rows.row_count)
               .attributes(LinearResidualAttrs{3072, 3072})
               .append(&error))
        return 1;
      NativeEmissionSession session(registry, graph, *module, *program);
      if (!Ok(session.LlamaRows(rows)) ||
          !Ok(session.Parameter(w, program->Binding(0), "weight")) ||
          !Ok(session.Bind(x, program->Binding(1))) ||
          !Ok(session.Bind(residual, program->Binding(3))) ||
          !Ok(session.EmitExisting(Id::AttentionOutput,
                                   graph.invocations()[0])))
        return 1;
      Value alias;
      if (!Ok(session.Native(residual, &alias)) ||
          !Ok(program->InspectNative(
              {alias, program->Binding(3)},
              [](const loom_module_t *,
                 const std::vector<loom_value_id_t> &ids) {
                return ids[0] == ids[1]
                           ? iree_ok_status()
                           : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                              "lost residual alias");
              })) ||
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
