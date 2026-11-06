// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/qwen_output.h"
#include "lib/operations/qwen_operations.h"
#include "lib/operations/tensor_operations.h"
#include "lib/qwen/testing/model_fixture.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <cstdlib>
using namespace hrx;
using namespace hrx::dispatch;
using namespace hrx::operations;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t s) {
  bool good = loomc_status_is_ok(s);
  if (!good) {
    auto m = loomc_status_message(s);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(s);
  return good;
}
bool Success(loomc_result_t *r) {
  if (r && loomc_result_succeeded(r))
    return true;
  if (r)
    for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
      auto d = loomc_result_diagnostic_at(r, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  return false;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 5)
    return 1;
  uint64_t tokens = std::strtoull(argv[2], nullptr, 10);
  if (tokens != 1 && tokens != 4)
    return 1;
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
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
  auto facts = qwen::testing::ModelFacts();
  for (auto &layer : facts.layers)
    for (auto &binding : layer.tensors)
      if (binding.role == "key") {
        auto &t = binding.tensor;
        t.storage_type = 14;
        t.byte_size = uint64_t(t.normalized_extents[0]) / 256 *
                      t.normalized_extents[1] * 210;
      }
  Registry registry;
  std::string error;
  for (auto registration : QwenOutputRegistrations())
    if (!registry.Add(std::move(registration), &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  QwenNativeGeometry geometry;
  if (!Ok(LoadQwenNative(context, workspace, registry,
                         {Id::Embedding, Id::QwenEpochHead}, facts,
                         {tokens, 1, 1, 96, 128, 1}, argv[1], &geometry,
                         &definitions, &result)) ||
      !Success(result))
    return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  loomc_module_t *fixture = nullptr;
  result = nullptr;
  const auto fixture_path = std::string(argv[1]) +
                            "/lib/dispatch/testing/qwen/output/nontrivial.loom";
  if (!Ok(loomc_module_deserialize_from_path(
          context, workspace, loomc_make_cstring_view(fixture_path.c_str()),
          nullptr, loomc_allocator_system(), &fixture, &result)) ||
      !Success(result))
    return 1;
  loomc_result_release(result);
  if (!Ok(module->AddProvider(fixture)))
    return 1;
  loomc_module_release(fixture);
  std::vector<RootRef> roots;
  {
    std::unique_ptr<CommandProgramBuilder> program;
    if (!Ok(module->DefineProgram("qwen.output.init", {}, 1, &program)))
      return 1;
    CommandValues values(*program);
    Value hidden, bytes, gamma, weight;
    if (!Ok(values.Constant(1024, LOOM_SCALAR_TYPE_INDEX, &hidden)) ||
        !Ok(values.Constant(208588800, LOOM_SCALAR_TYPE_INDEX, &bytes)) ||
        !Ok(values.Parameter(program->Binding(0), "output_norm.weight", {},
                             LOOM_SCALAR_TYPE_F32, {hidden}, &gamma)) ||
        !Ok(values.Parameter(program->Binding(0), "token_embd.weight", {},
                             LOOM_SCALAR_TYPE_I8, {bytes}, &weight)))
      return 1;
    KernelRef initializer;
    if (!Ok(module->DeclareKernel("qwen_output_init_tied_bank", {},
                                  {module->Buffer(), module->Buffer()},
                                  &initializer)) ||
        !Ok(program->Invoke(initializer, {}, {weight, gamma})) ||
        !Ok(program->Finish()))
      return 1;
    roots.push_back(program->Ref());
  }

  for (bool head : {false, true}) {
    OperationRegistry ops;
    if (!DefineEmbedding(&ops, &error) || !DefineQwenEpochHead(&ops, &error))
      return 1;
    OperationGraph graph(ops);
    auto weight =
        graph.tensor(TensorType::matrix(1024, 248320, ElementType::F32)
                         .setEncoding(Encoding::Q6_K)
                         .setLayout(Layout::GgmlWeightRows),
                     &error);
    auto input = graph.tensor(
        TensorType::matrix(tokens, 1024, ElementType::F32), &error);
    auto gamma =
        graph.tensor(TensorType::vector(1024, ElementType::F32), &error);
    auto ids =
        graph.tensor(TensorType::vector(tokens, ElementType::I32), &error);
    auto epoch =
        graph.tensor(TensorType::vector(1048, ElementType::I32), &error);
    auto output =
        head ? graph.tensor(TensorType::vector(1, ElementType::I32), &error)
             : input;
    if (head) {
      if (!graph.operation(OpCode::QwenEpochHead)
               .bind(Port::Input, input)
               .bind(Port::Gamma, gamma)
               .bind(Port::Weight, weight)
               .bind(Port::Epoch, epoch)
               .bind(Port::Output, output)
               .attributes(QwenEpochHeadAttrs{1024, 248320, 1e-6, tokens, 1, 1})
               .append(&error))
        return 1;
    } else if (!graph.operation(OpCode::Embedding)
                    .bind(Port::TokenIds, ids)
                    .bind(Port::Weight, weight)
                    .bind(Port::Output, output)
                    .attributes(EmbeddingAttrs{248320, 1024})
                    .append(&error))
      return 1;
    std::unique_ptr<CommandProgramBuilder> program;
    std::string name =
        std::string(head ? "qwen.output.head.t" : "qwen.output.embedding.t") +
        std::to_string(tokens);
    if (!Ok(module->DefineProgram(name, {}, head ? 4 : 3, &program)))
      return 1;
    NativeEmissionSession session(registry, graph, *module, *program);
    if (!Ok(session.Parameter(weight, program->Binding(0),
                              "token_embd.weight")))
      return 1;
    if (head) {
      if (!Ok(session.Parameter(gamma, program->Binding(0),
                                "output_norm.weight")) ||
          !Ok(session.Bind(input, program->Binding(1))) ||
          !Ok(session.Bind(epoch, program->Binding(2))) ||
          !Ok(session.Bind(output, program->Binding(3))))
        return 1;
    } else if (!Ok(session.Bind(ids, program->Binding(1))) ||
               !Ok(session.Bind(output, program->Binding(2))))
      return 1;
    if (!Ok(session.EmitExisting(head ? Id::QwenEpochHead : Id::Embedding,
                                 graph.invocations()[0])) ||
        !Ok(program->Finish()))
      return 1;
    roots.push_back(program->Ref());
  }
  {
    OperationRegistry ops;
    if (!DefineEmbedding(&ops, &error) || !DefineQwenEpochHead(&ops, &error))
      return 1;
    OperationGraph graph(ops);
    auto weight =
        graph.tensor(TensorType::matrix(1024, 248320, ElementType::F32)
                         .setEncoding(Encoding::Q6_K)
                         .setLayout(Layout::GgmlWeightRows),
                     &error);
    auto gamma =
        graph.tensor(TensorType::vector(1024, ElementType::F32), &error);
    auto ids =
        graph.tensor(TensorType::vector(tokens, ElementType::I32), &error);
    auto residual = graph.tensor(
        TensorType::matrix(tokens, 1024, ElementType::F32), &error);
    auto epoch =
        graph.tensor(TensorType::vector(1048, ElementType::I32), &error);
    auto output = graph.tensor(TensorType::vector(1, ElementType::I32), &error);
    if (!graph.operation(OpCode::Embedding)
             .bind(Port::TokenIds, ids)
             .bind(Port::Weight, weight)
             .bind(Port::Output, residual)
             .attributes(EmbeddingAttrs{248320, 1024})
             .append(&error) ||
        !graph.operation(OpCode::QwenEpochHead)
             .bind(Port::Input, residual)
             .bind(Port::Gamma, gamma)
             .bind(Port::Weight, weight)
             .bind(Port::Epoch, epoch)
             .bind(Port::Output, output)
             .attributes(QwenEpochHeadAttrs{1024, 248320, 1e-6, tokens, 1, 1})
             .append(&error))
      return 1;
    std::unique_ptr<CommandProgramBuilder> program;
    if (!Ok(module->DefineProgram(
            "qwen.output.chain.t" + std::to_string(tokens), {}, 4, &program)))
      return 1;
    NativeEmissionSession session(registry, graph, *module, *program);
    if (!Ok(session.Parameter(weight, program->Binding(0),
                              "token_embd.weight")) ||
        !Ok(session.Parameter(gamma, program->Binding(0),
                              "output_norm.weight")) ||
        !Ok(session.Bind(ids, program->Binding(1))) ||
        !Ok(session.Bind(epoch, program->Binding(2))) ||
        !Ok(session.Bind(output, program->Binding(3))) ||
        !Ok(session.EmitExisting(Id::Embedding, graph.invocations()[0])))
      return 1;
    Value fresh;
    if (!Ok(session.Native(residual, &fresh)) ||
        !Ok(program->InspectNative(
            {fresh}, [](const loom_module_t *m,
                        const std::vector<loom_value_id_t> &ids) {
              auto *allocation =
                  loom_value_def_op(loom_module_value(m, ids[0]));
              if (!allocation || !loom_buffer_alloca_isa(allocation))
                return iree_make_status(
                    IREE_STATUS_INVALID_ARGUMENT,
                    "fresh embedding result must own scratch");
              auto *constant = loom_value_def_op(loom_module_value(
                  m, loom_buffer_alloca_byte_length(allocation)));
              if (!constant || !loom_index_constant_isa(constant) ||
                  loom_attr_as_i64(loom_index_constant_value(constant)) !=
                      2097152)
                return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                        "fresh embedding scratch must cover "
                                        "actual512row head footprint");
              return iree_ok_status();
            })))
      return 1;
    if (!Ok(session.EmitExisting(Id::QwenEpochHead, graph.invocations()[1])) ||
        !Ok(program->Finish()))
      return 1;
    roots.push_back(program->Ref());
  }
  loomc_module_t *linked = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify(roots, &linked, &result)) || !Success(result))
    return 1;
  loomc_result_release(result);
  for (bool text : {false, true}) {
    loomc_module_serialize_options_t o{};
    o.format = text ? LOOMC_SOURCE_FORMAT_TEXT : LOOMC_SOURCE_FORMAT_BYTECODE;
    if (!Ok(loomc_module_serialize_to_path(
            linked, &o, loomc_make_cstring_view(argv[text ? 4 : 3]),
            loomc_allocator_system())))
      return 1;
  }
  loomc_module_release(linked);
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
  return 0;
}
