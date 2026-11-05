// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/operations/control_operations.h"
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
  if (argc != 4)
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
  if (!Ok(LoadLlamaNative(context, workspace, registry,
                          {Id::Embedding, Id::OutputHead, Id::PublishPrefill,
                           Id::PublishDecode},
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
  for (bool decode : {false, true}) {
    const std::string name = decode ? "head.rows.decode" : "head.rows.prefill";
    if (!Ok(module->DefineProgram(name, {}, 7, &program)))
      return 1;
    OperationRegistry ops;
    std::string error;
    if (!DefinePublishToken(&ops, &error))
      return 1;
    OperationGraph graph(ops);
    auto tensor = [&](uint64_t n) {
      return graph.tensor(TensorType::vector(n, ElementType::I32), &error);
    };
    auto raw = tensor(1), packet = tensor(3), history = tensor(129),
         progress = tensor(8);
    NativeLlamaRows rows;
    rows.row_begin = graph.scalar(ScalarType::I64, &error);
    rows.row_count = graph.scalar(ScalarType::I64, &error);
    rows.count_control = tensor(3);
    rows.row_capacity = decode ? 1 : 8;
    rows.token_begin = decode ? 0 : 2;
    rows.phase = decode ? Phase::Decode : Phase::Prefill;
    rows.packet = program->Binding(2);
    if (!graph.operation(OpCode::PublishToken)
             .bind(Port::Token, raw)
             .bind(Port::Control, packet)
             .bind(Port::History, history)
             .bind(Port::Progress, progress)
             .attributes(PublishTokenAttrs{rows.phase,
                                           128,
                                           TerminalPolicy::StopOnTokenIds,
                                           {128001, 128008, 128009}})
             .append(&error))
      return 1;
    NativeEmissionSession session(registry, graph, *module, *program);
    if (!Ok(session.Bind(raw, program->Binding(3))) ||
        !Ok(session.Bind(packet, program->Binding(2))) ||
        !Ok(session.Bind(history, program->Binding(4))) ||
        !Ok(session.Bind(progress, program->Binding(5))) ||
        !Ok(session.LlamaRows(rows)) ||
        !Ok(session.EmitExisting(decode ? Id::PublishDecode
                                        : Id::PublishPrefill,
                                 graph.invocations()[0])))
      return 1;
    // Publication has overwritten the original packet; consumers must retain
    // the invocation's independent count snapshot, never reread that packet.
    Value count;
    if (!Ok(session.Native(rows.count_control, &count)))
      return 1;
    KernelRef norm;
    if (!Ok(module->Kernel("llama32_output_norm_last_f32", &norm)))
      return 1;
    CommandValues values(*program);
    Value capacity, begin, origin;
    if (!Ok(values.Constant(rows.row_capacity, LOOM_SCALAR_TYPE_INDEX,
                            &capacity)) ||
        !Ok(values.Constant(rows.token_begin, LOOM_SCALAR_TYPE_INDEX,
                            &begin)) ||
        !Ok(values.Constant(0, LOOM_SCALAR_TYPE_OFFSET, &origin)) ||
        !Ok(program->Invoke(norm, {capacity},
                            {capacity, begin, origin, count,
                             program->Binding(0), program->Binding(1),
                             program->Binding(6)})) ||
        !Ok(program->Finish()))
      return 1;
    roots.push_back(program->Ref());
    program.reset();
  }
  // Actual registered head -> publication, distinct from the snapshot stress
  // roots above. The tied synthetic tensor has the full admitted vocabulary.
  for (bool decode : {false, true}) {
    const std::string name = decode ? "head.full.decode" : "head.full.prefill";
    if (!Ok(module->DefineProgram(name, {}, 7, &program)))
      return 1;
    OperationRegistry ops;
    std::string error;
    if (!DefineEmbedding(&ops, &error) || !DefineOutputHead(&ops, &error) ||
        !DefinePublishToken(&ops, &error))
      return 1;
    OperationGraph graph(ops);
    auto tensor = [&](uint64_t n, ElementType type) {
      return graph.tensor(TensorType::vector(n, type), &error);
    };
    auto input = graph.tensor(
        TensorType::matrix(decode ? 1 : 8, 3072, ElementType::F32), &error);
    auto gamma = tensor(3072, ElementType::F32);
    auto weight_type = TensorType::matrix(3072, 128256, ElementType::F32);
    weight_type.setEncoding(Encoding::Q6_K).setLayout(Layout::GgmlWeightRows);
    auto weight = graph.tensor(weight_type, &error);
    auto token = tensor(1, ElementType::I32),
         packet = tensor(3, ElementType::I32),
         history = tensor(129, ElementType::I32),
         progress = tensor(8, ElementType::I32);
    NativeLlamaRows rows;
    rows.row_begin = graph.scalar(ScalarType::I64, &error);
    rows.row_count = graph.scalar(ScalarType::I64, &error);
    rows.count_control = tensor(3, ElementType::I32);
    rows.row_capacity = decode ? 1 : 8;
    rows.token_begin = decode ? 0 : 2;
    rows.phase = decode ? Phase::Decode : Phase::Prefill;
    rows.packet = program->Binding(3);
    auto ids = tensor(rows.row_capacity, ElementType::I32);
    if (!graph.operation(OpCode::Embedding)
             .bind(Port::TokenIds, ids)
             .bind(Port::Weight, weight)
             .bind(Port::Output, input)
             .bind(Port::RowBegin, rows.row_begin)
             .bind(Port::RowCount, rows.row_count)
             .attributes(EmbeddingAttrs{128256, 3072})
             .append(&error) ||
        !graph.operation(OpCode::OutputHead)
             .bind(Port::Input, input)
             .bind(Port::Gamma, gamma)
             .bind(Port::Weight, weight)
             .bind(Port::Control, rows.count_control)
             .bind(Port::Output, token)
             .attributes(OutputHeadAttrs{3072, 128256})
             .append(&error) ||
        !graph.operation(OpCode::PublishToken)
             .bind(Port::Token, token)
             .bind(Port::Control, packet)
             .bind(Port::History, history)
             .bind(Port::Progress, progress)
             .attributes(PublishTokenAttrs{rows.phase,
                                           128,
                                           TerminalPolicy::StopOnTokenIds,
                                           {128001, 128008, 128009}})
             .append(&error))
      return 1;
    NativeEmissionSession session(registry, graph, *module, *program);
    if (!Ok(session.Bind(ids, program->Binding(4))) ||
        !Ok(session.Bind(input, program->Binding(1))) ||
        !Ok(session.Bind(gamma, program->Binding(2))) ||
        !Ok(session.Parameter(weight, program->Binding(0),
                              "token_embd.weight")) ||
        !Ok(session.Bind(packet, program->Binding(3))) ||
        !Ok(session.Bind(history, program->Binding(4))) ||
        !Ok(session.Bind(progress, program->Binding(5))) ||
        !Ok(session.Bind(token, program->Binding(6))) ||
        !Ok(session.LlamaRows(rows)) ||
        !Ok(session.EmitExisting(Id::Embedding, graph.invocations()[0])) ||
        !Ok(session.EmitExisting(Id::OutputHead, graph.invocations()[1])) ||
        !Ok(session.EmitExisting(decode ? Id::PublishDecode
                                        : Id::PublishPrefill,
                                 graph.invocations()[2])) ||
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
  bool ok = true;
  for (int format = 0; format < 2; ++format) {
    loomc_module_serialize_options_t options{};
    options.format =
        format ? LOOMC_SOURCE_FORMAT_TEXT : LOOMC_SOURCE_FORMAT_BYTECODE;
    ok &= Ok(loomc_module_serialize_to_path(
        linked, &options, loomc_make_cstring_view(argv[2 + format]),
        loomc_allocator_system()));
  }
  loomc_module_release(linked);
  program.reset();
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
  return ok ? 0 : 1;
}
