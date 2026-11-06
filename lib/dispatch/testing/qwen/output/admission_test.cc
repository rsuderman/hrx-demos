// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen.h"
#include "lib/dispatch/qwen_output.h"
#include "lib/operations/qwen_operations.h"
#include "lib/operations/tensor_operations.h"
#include "lib/qwen/testing/model_fixture.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu.h"
#include "gtest/gtest.h"
#include <cstdlib>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
::testing::AssertionResult Ok(loomc_status_t s) {
  bool good = loomc_status_is_ok(s);
  auto m = loomc_status_message(s);
  std::string message(m.data ? m.data : "", m.size);
  loomc_status_free(s);
  return good ? ::testing::AssertionSuccess()
              : (::testing::AssertionFailure() << message);
}
size_t CountOps(const loom_region_t *r) {
  size_t n = 0;
  if (r)
    for (size_t b = 0; b < r->block_count; ++b)
      for (auto *op = r->blocks[b]->first_op; op; op = op->next_op) {
        ++n;
        for (size_t i = 0; i < op->region_count; ++i)
          n += CountOps(loom_op_regions(op)[i]);
      }
  return n;
}
TEST(QwenOutputAdmission,
     KnownOutputAliasesRejectBeforeMutationAndRecoverSameOwner) {
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &environment)));
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t options{};
  options.next = &target;
  ASSERT_TRUE(
      Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
  ASSERT_TRUE(Ok(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  auto facts = qwen::testing::ModelFacts();
  for (auto &l : facts.layers)
    for (auto &b : l.tensors)
      if (b.role == "key") {
        b.tensor.storage_type = 14;
        b.tensor.byte_size = uint64_t(b.tensor.normalized_extents[0]) / 256 *
                             b.tensor.normalized_extents[1] * 210;
      }
  Registry registry;
  std::string error;
  for (auto r : QwenOutputRegistrations())
    ASSERT_TRUE(registry.Add(std::move(r), &error)) << error;
  ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
  ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
  auto root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
              std::getenv("TEST_WORKSPACE");
  loomc_module_t *definitions = nullptr;
  loomc_result_t *load = nullptr;
  QwenNativeGeometry geometry;
  ASSERT_TRUE(Ok(LoadQwenNative(
      context, workspace, registry, {Id::Embedding, Id::QwenEpochHead}, facts,
      {4, 1, 1, 96, 128, 1}, root, &geometry, &definitions, &load)));
  ASSERT_TRUE(load && loomc_result_succeeded(load));
  loomc_result_release(load);
  for (bool head : {false, true})
    for (size_t alias = 0; alias < (head ? 3u : 2u); ++alias) {
      std::unique_ptr<ModuleBuilder> module;
      ASSERT_TRUE(
          Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
      std::unique_ptr<CommandProgramBuilder> program;
      ASSERT_TRUE(Ok(module->DefineProgram("recover", {}, 5, &program)));
      OperationRegistry ops;
      ASSERT_TRUE(DefineEmbedding(&ops, &error));
      ASSERT_TRUE(DefineQwenEpochHead(&ops, &error));
      OperationGraph graph(ops);
      auto weight =
          graph.tensor(TensorType::matrix(1024, 248320, ElementType::F32)
                           .setEncoding(Encoding::Q6_K)
                           .setLayout(Layout::GgmlWeightRows),
                       &error);
      auto input =
          graph.tensor(head ? TensorType::matrix(4, 1024, ElementType::F32)
                            : TensorType::vector(4, ElementType::I32),
                       &error);
      auto gamma =
               graph.tensor(TensorType::vector(1024, ElementType::F32), &error),
           epoch =
               graph.tensor(TensorType::vector(1048, ElementType::I32), &error);
      OperationAttrs attrs =
          head ? OperationAttrs(QwenEpochHeadAttrs{1024, 248320, 1e-6, 4, 1, 1})
               : OperationAttrs(EmbeddingAttrs{248320, 1024});
      std::vector<shapes::Operand> logical =
          head ? std::vector<shapes::Operand>{{Port::Input, input},
                                              {Port::Gamma, gamma},
                                              {Port::Weight, weight},
                                              {Port::Epoch, epoch}}
               : std::vector<shapes::Operand>{{Port::TokenIds, input},
                                              {Port::Weight, weight}};
      shapes::Result shape;
      ASSERT_TRUE(shapes::InferResult(
          graph, head ? OpCode::QwenEpochHead : OpCode::Embedding, attrs,
          logical, &shape, &error))
          << error;
      InvocationRequest request{graph, attrs, logical, shape};
      const auto *entry = registry.Resolve(
          head ? Id::QwenEpochHead : Id::Embedding, request, "", &error);
      ASSERT_NE(entry, nullptr) << error;
      std::vector<NativeOperand> native =
          head ? std::vector<NativeOperand>{{Port::Input, input,
                                             program->Binding(1), ""},
                                            {Port::Gamma, gamma,
                                             program->Binding(0),
                                             "output_norm.weight"},
                                            {Port::Weight, weight,
                                             program->Binding(0),
                                             "token_embd.weight"},
                                            {Port::Epoch, epoch,
                                             program->Binding(2), ""}}
               : std::vector<NativeOperand>{
                     {Port::TokenIds, input, program->Binding(1), ""},
                     {Port::Weight, weight, program->Binding(0),
                      "token_embd.weight"}};
      CommandValues values(*program);
      BuildContext build{request, *module, *program, values,
                         native,  {},      false,    program->Binding(alias),
                         true};
      size_t before_values = 0, before_ops = 0;
      ASSERT_TRUE(Ok(
          program->InspectNative({}, [&](const loom_module_t *m,
                                         const std::vector<loom_value_id_t> &) {
            before_values = m->values.count;
            before_ops = CountOps(m->body);
            return iree_ok_status();
          })));
      Value output = program->Binding(4);
      auto rejected = entry->builder.build(*entry, build, &output);
      EXPECT_FALSE(loomc_status_is_ok(rejected));
      loomc_status_free(rejected);
      EXPECT_FALSE(build.staged);
      ASSERT_TRUE(Ok(program->InspectNative(
          {output, program->Binding(4)},
          [&](const loom_module_t *m, const std::vector<loom_value_id_t> &ids) {
            EXPECT_EQ(m->values.count, before_values);
            EXPECT_EQ(CountOps(m->body), before_ops);
            EXPECT_EQ(ids[0], ids[1]);
            return iree_ok_status();
          })));
      build.output = program->Binding(3);
      ASSERT_TRUE(Ok(entry->builder.build(*entry, build, &output)));
      EXPECT_TRUE(build.staged);
      ASSERT_TRUE(Ok(program->InspectNative(
          {output, program->Binding(3)},
          [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
            EXPECT_EQ(ids[0], ids[1]);
            return iree_ok_status();
          })));
      ASSERT_TRUE(Ok(program->Finish()));
    }
  loomc_module_release(definitions);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
}
} // namespace
} // namespace hrx::dispatch
