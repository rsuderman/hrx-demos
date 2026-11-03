// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loomc/compile.h"
#include "loomc/interop.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
namespace hrx::dispatch {
namespace {
std::string Message(loomc_status_t status) {
  auto view = loomc_status_message(status);
  std::string text(view.data ? view.data : "", view.size);
  loomc_status_free(status);
  return text;
}
::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << Message(status);
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  if (!result)
    return "no result";
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->message.data, d->message.size);
    text += '\n';
  }
  return text;
}
loomc_status_t Dump(const loomc_module_t *module, std::string *text) {
  loomc_module_serialize_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
  options.structure_size = sizeof(options);
  options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
  loomc_source_t *source = nullptr;
  auto status = loomc_module_serialize_text_to_source(
      module, &options, loomc_allocator_system(), &source);
  if (loomc_status_is_ok(status)) {
    auto bytes = loomc_source_contents(source);
    text->assign(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
    loomc_source_release(source);
  }
  return status;
}
class EmbeddingTest : public ::testing::Test {
protected:
  void SetUp() override {
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
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
           std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    loomc_target_environment_release(environment);
  }
  loomc_module_t *LoadEntry(const Registration &entry) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status = Load(context, workspace, entry, root, &module, &result);
    EXPECT_TRUE(Ok(status));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    return module;
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
using namespace operations;
using namespace loomify_gguf;

void Compile(loomc_context_t *context, loomc_workspace_t *workspace,
             loomc_target_environment_t *environment, loomc_module_t *module,
             const char *name) {
  std::vector<loomc_request_t *> requests;
  loomc_target_profile_t *profile = nullptr;
  ASSERT_TRUE(Ok(loomc_target_profile_select(
      environment, loomc_make_cstring_view("amdgpu:gfx1100"),
      loomc_allocator_system(), &profile)));
  auto root = loomc_make_cstring_view(name);
  loomc_cmd_program_module_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
  options.structure_size = sizeof(options);
  options.root_symbols = &root;
  options.root_symbol_count = 1;
  options.kernel_target_profile = profile;
  options.request_sink.publish = [](void *user, loomc_request_t *request) {
    static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
    return loomc_ok_status();
  };
  options.request_sink.user_data = &requests;
  loomc_product_t *product = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(loomc_cmd_program_product_build_module(
      workspace, module, &options, loomc_allocator_system(), &product,
      &result)));
  EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  if (result)
    loomc_result_release(result);
  if (product)
    loomc_product_release(product);
  EXPECT_FALSE(requests.empty());
  loomc_compiler_t *compiler = nullptr;
  ASSERT_TRUE(Ok(loomc_compiler_create(context, nullptr,
                                       loomc_allocator_system(), &compiler)));
  for (auto *request : requests) {
    loomc_module_t *leaf = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, loomc_request_source(request), nullptr,
        loomc_allocator_system(), &leaf, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    loomc_compile_artifact_options_t compile{};
    compile.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
    compile.structure_size = sizeof(compile);
    compile.target_profile = profile;
    EXPECT_TRUE(
        Ok(loomc_compile_artifact(compiler, workspace, nullptr, leaf, &compile,
                                  loomc_allocator_system(), &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result && loomc_result_succeeded(result)) {
      EXPECT_GT(loomc_result_artifact_count(result), 0u);
    }
    if (result)
      loomc_result_release(result);
    loomc_module_release(leaf);
    loomc_request_release(request);
  }
  loomc_compiler_release(compiler);
  loomc_target_profile_release(profile);
}

TEST_F(EmbeddingTest,
       AllocatesEmbeddingAndChainsOwnedNormWithTiedWeightIdentity) {
  auto registry = Registry::Llama3B();
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(
      Ok(Load(context, workspace, registry, {Id::Embedding, Id::RmsNorm}, root,
              &definitions, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("embedding_norm", {}, 3, &program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineEmbedding(&ops, &error));
  ASSERT_TRUE(DefineRmsNorm(&ops, &error));
  OperationGraph graph(ops);
  auto ids = graph.tensor(TensorType::vector(2, ElementType::I32), &error);
  auto type = TensorType::matrix(3072, 128256, ElementType::F32);
  type.setEncoding(Encoding::Q6_K).setLayout(Layout::GgmlWeightRows);
  auto weight = graph.tensor(type, &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(author.Bind(ids, program->Binding(0))));
  ASSERT_TRUE(
      Ok(author.Parameter(weight, program->Binding(1), "token_embd.weight")));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(2))));
  ValueId embedded, repeated, norm;
  for (auto *output : {&embedded, &repeated})
    ASSERT_TRUE(Ok(author.Invoke(
        Id::Embedding, EmbeddingAttrs{128256, 3072},
        {{Port::TokenIds, ids}, {Port::Weight, weight}}, output)));
  ASSERT_TRUE(Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                               {{Port::Input, embedded}, {Port::Gamma, gamma}},
                               &norm)));
  EXPECT_NE(embedded, repeated);
  EXPECT_NE(norm, embedded);
  EXPECT_EQ(graph.value(embedded)->tensorType()->dimension(0), 2u);
  EXPECT_EQ(graph.value(embedded)->tensorType()->dimension(1), 3072u);
  ASSERT_EQ(graph.invocations().size(), 3u);
  EXPECT_EQ(graph.invocations()[0].binding(Port::Weight)->value(), weight);
  EXPECT_EQ(graph.invocations()[1].binding(Port::Weight)->value(), weight);
  EXPECT_EQ(graph.invocations()[2].binding(Port::Input)->value(), embedded);
  Value embedded_native, repeated_native, norm_native;
  ASSERT_TRUE(Ok(author.Native(embedded, &embedded_native)));
  ASSERT_TRUE(Ok(author.Native(repeated, &repeated_native)));
  ASSERT_TRUE(Ok(author.Native(norm, &norm_native)));
  ASSERT_TRUE(Ok(program->InspectNative(
      {program->Binding(0), embedded_native, repeated_native, norm_native},
      [](const loom_module_t *m, const std::vector<loom_value_id_t> &ids) {
        EXPECT_NE(ids[0], ids[1]);
        EXPECT_NE(ids[1], ids[2]);
        EXPECT_NE(ids[1], ids[3]);
        size_t launches = 0;
        for (size_t i = 0; i < m->symbols.count; ++i) {
          const auto *def = m->symbols.entries[i].defining_op;
          if (!def || !loom_command_program_def_isa(def))
            continue;
          auto *body =
              loom_region_const_entry_block(loom_command_program_def_body(def));
          auto *serial = loom_region_const_entry_block(
              loom_command_serial_body(body->first_op));
          for (auto *op = serial->first_op; op; op = op->next_op) {
            if (!loom_kernel_launch_isa(op))
              continue;
            auto args = loom_kernel_launch_arguments(op);
            EXPECT_EQ(args.count, 4u);
            if (args.count == 4 && launches < 3) {
              EXPECT_EQ(args.values[1], launches < 2 ? ids[0] : ids[1]);
              EXPECT_EQ(args.values[3], ids[launches + 1]);
            }
            ++launches;
          }
        }
        EXPECT_EQ(launches, 3u);
        return iree_ok_status();
      })));
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::string text;
  ASSERT_TRUE(Ok(Dump(linked, &text)));
  EXPECT_NE(text.find("token_embd.weight"), std::string::npos);
  EXPECT_NE(text.find("buffer.alloc"), std::string::npos);
  Compile(context, workspace, environment, linked, "embedding_norm");
  loomc_module_release(linked);
}
TEST_F(EmbeddingTest, ShapesAndSelectionRejectUnsupportedInputs) {
  auto registry = Registry::Llama3B();
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineEmbedding(&ops, &error));
  OperationGraph graph(ops);
  auto type = TensorType::matrix(3072, 128256, ElementType::F32);
  type.setEncoding(Encoding::Q6_K).setLayout(Layout::GgmlWeightRows);
  auto weight = graph.tensor(type, &error);
  for (uint64_t rows : {1, 7, 512, 513}) {
    auto ids = graph.tensor(TensorType::vector(rows, ElementType::I32), &error);
    std::vector<shapes::Operand> inputs{{Port::TokenIds, ids},
                                        {Port::Weight, weight}};
    shapes::Result result;
    ASSERT_TRUE(shapes::InferResult(graph, OpCode::Embedding,
                                    EmbeddingAttrs{128256, 3072}, inputs,
                                    &result, &error))
        << error;
    EXPECT_EQ(result.type.dimension(0), rows);
    EXPECT_EQ(result.type.dimension(1), 3072u);
    EXPECT_FALSE(result.alias.valid());
    OperationAttrs attrs = EmbeddingAttrs{128256, 3072};
    InvocationRequest request{graph, attrs, inputs, result};
    EXPECT_EQ(registry.Resolve(Id::Embedding, request, "", &error) != nullptr,
              rows <= 512);
    auto begin = graph.scalar(ScalarType::I64, &error);
    inputs.push_back({Port::RowBegin, begin});
    EXPECT_EQ(registry.Resolve(Id::Embedding, request, "", &error), nullptr);
    inputs.back().port = Port::RowCount;
    EXPECT_EQ(registry.Resolve(Id::Embedding, request, "", &error), nullptr);
  }
  const auto *registration = registry.Find(Id::Embedding);
  ASSERT_NE(registration, nullptr);
  for (const auto &operand : registration->operands)
    if (operand.port == Port::Weight) {
      EXPECT_EQ(operand.access, Access::Read);
    }
  auto empty = graph.tensor(TensorType::vector(0, ElementType::I32), &error);
  EXPECT_FALSE(empty.valid());
  shapes::Result unchanged;
  EXPECT_FALSE(shapes::InferResult(
      graph, OpCode::Embedding, EmbeddingAttrs{128256, 3072},
      {{Port::TokenIds, empty}, {Port::Weight, weight}}, &unchanged, &error));
  auto float_ids =
      graph.tensor(TensorType::vector(2, ElementType::F32), &error);
  EXPECT_FALSE(shapes::InferResult(
      graph, OpCode::Embedding, EmbeddingAttrs{128256, 3072},
      {{Port::TokenIds, float_ids}, {Port::Weight, weight}}, &unchanged,
      &error));
  auto bad_ids =
      graph.tensor(TensorType::matrix(2, 1, ElementType::I32), &error);
  shapes::Result result;
  EXPECT_FALSE(shapes::InferResult(
      graph, OpCode::Embedding, EmbeddingAttrs{128256, 3072},
      {{Port::TokenIds, bad_ids}, {Port::Weight, weight}}, &result, &error));
  auto ids = graph.tensor(TensorType::vector(2, ElementType::I32), &error);
  EXPECT_FALSE(shapes::InferResult(
      graph, OpCode::Embedding, EmbeddingAttrs{128256, 1024},
      {{Port::TokenIds, ids}, {Port::Weight, weight}}, &result, &error));
  for (auto encoding : {Encoding::Q4_K, Encoding::Q8_0}) {
    type.setEncoding(encoding);
    auto other = graph.tensor(type, &error);
    std::vector<shapes::Operand> inputs{{Port::TokenIds, ids},
                                        {Port::Weight, other}};
    ASSERT_TRUE(shapes::InferResult(graph, OpCode::Embedding,
                                    EmbeddingAttrs{128256, 3072}, inputs,
                                    &result, &error));
    OperationAttrs attrs = EmbeddingAttrs{128256, 3072};
    InvocationRequest request{graph, attrs, inputs, result};
    EXPECT_EQ(registry.Resolve(Id::Embedding, request, "", &error), nullptr);
  }
}
} // namespace
} // namespace hrx::dispatch
