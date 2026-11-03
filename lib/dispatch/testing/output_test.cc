// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/operations/control_operations.h"
#include "lib/operations/tensor_operations.h"
#include "loomc/compile.h"
#include "loomc/iree.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "gtest/gtest.h"
#include <cstdlib>
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
class OutputTest : public ::testing::Test {
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
TEST_F(OutputTest, LoadsOwnedFusedAndPublicationDefinitions) {
  auto registry = Registry::Llama3B();
  for (auto id : {Id::OutputHead, Id::PublishPrefill, Id::PublishDecode}) {
    auto *entry = registry.Find(id);
    ASSERT_NE(entry, nullptr);
    auto *m = LoadEntry(*entry);
    ASSERT_NE(m, nullptr);
    std::string text;
    ASSERT_TRUE(Ok(Dump(m, &text)));
    if (id == Id::OutputHead) {
      EXPECT_NE(text.find("llm_output_last_greedy"), std::string::npos);
      EXPECT_NE(text.find("llama32_output_norm_last_f32"), std::string::npos);
      EXPECT_NE(text.find("qwen38_greedy_argmax_reduce_rows_f32"),
                std::string::npos);
    } else
      EXPECT_NE(text.find("128008"), std::string::npos);
    loomc_module_release(m);
  }
}
TEST_F(OutputTest, HeadAllocatesTokenAndRejectsUnsupportedFacts) {
  auto registry = Registry::Llama3B();
  auto *m = LoadEntry(*registry.Find(Id::OutputHead));
  ASSERT_NE(m, nullptr);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, m, &module)));
  loomc_module_release(m);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("head", {}, 4, &program)));
  OperationRegistry defs;
  std::string e;
  ASSERT_TRUE(DefineOutputHead(&defs, &e));
  OperationGraph graph(defs);
  auto x = graph.tensor(TensorType::matrix(4, 3072, ElementType::F32), &e);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &e);
  auto wt = TensorType::matrix(3072, 128256, ElementType::F32);
  wt.setEncoding(Encoding::Q6_K);
  wt.setLayout(Layout::GgmlWeightRows);
  auto control = graph.tensor(TensorType::vector(3, ElementType::I32), &e);
  auto weight = graph.tensor(wt, &e);
  ASSERT_TRUE(weight.valid()) << e;
  InvocationProgram author(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
  ASSERT_TRUE(
      Ok(author.Parameter(weight, program->Binding(2), "token_embd.weight")));
  ASSERT_TRUE(Ok(author.Bind(control, program->Binding(3))));
  std::vector<shapes::Operand> inputs{{Port::Input, x},
                                      {Port::Gamma, gamma},
                                      {Port::Weight, weight},
                                      {Port::Control, control}};
  OutputHeadAttrs attrs{3072, 128256};
  // The current native checkpoint is tied. Reject an untied named parameter
  // before allocation or native staging even when its tensor shape matches.
  shapes::Result result_shape;
  ASSERT_TRUE(shapes::InferResult(graph, OpCode::OutputHead, attrs, inputs,
                                  &result_shape, &e))
      << e;
  InvocationRequest request{graph, attrs, inputs, result_shape};
  CommandValues values(*program);
  std::vector<NativeOperand> untied{
      {Port::Input, x, program->Binding(0), {}},
      {Port::Gamma, gamma, program->Binding(1), {}},
      {Port::Weight, weight, program->Binding(2), "output.weight"},
      {Port::Control, control, program->Binding(3), {}}};
  BuildContext unsupported{
      request, *module, *program, values, untied, program->Binding(3), true};
  Value unused;
  auto failure =
      registry.Find(Id::OutputHead)
          ->builder.build(*registry.Find(Id::OutputHead), unsupported, &unused);
  EXPECT_FALSE(loomc_status_is_ok(failure));
  loomc_status_free(failure);
  EXPECT_FALSE(unsupported.staged);
  ValueId token;
  auto bad = attrs;
  bad.epsilon = 1e-4;
  auto rejected = author.Invoke(Id::OutputHead, bad, inputs, &token);
  EXPECT_FALSE(loomc_status_is_ok(rejected));
  loomc_status_free(rejected);
  EXPECT_EQ(graph.values().size(), 4u);
  EXPECT_TRUE(graph.invocations().empty());
  ASSERT_TRUE(Ok(author.Invoke(Id::OutputHead, attrs, inputs, &token)));
  EXPECT_EQ(graph.values().size(), 5u);
  ASSERT_TRUE(token.valid());
  EXPECT_EQ(graph.value(token)->tensorType()->elementType(), ElementType::I32);
  EXPECT_EQ(graph.value(token)->tensorType()->rank(), 1u);
  EXPECT_EQ(graph.value(token)->tensorType()->dimension(0), 1u);
  Value native;
  ASSERT_TRUE(Ok(author.Native(token, &native)));
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  std::string text;
  ASSERT_TRUE(Ok(Dump(linked, &text)));
  EXPECT_NE(text.find("buffer.alloca"), std::string::npos);
  EXPECT_NE(text.find("llm.output.last_greedy"), std::string::npos);
  loomc_result_release(result);
  loomc_module_release(linked);
}
TEST_F(OutputTest, PublicationEffectsRetainExplicitStateAndNoOutputAllocation) {
  for (auto phase : {Phase::Prefill, Phase::Decode})
    for (bool existing : {false, true}) {
      auto registry = Registry::Llama3B();
      auto id =
          phase == Phase::Prefill ? Id::PublishPrefill : Id::PublishDecode;
      auto *m = LoadEntry(*registry.Find(id));
      ASSERT_NE(m, nullptr);
      std::unique_ptr<ModuleBuilder> module;
      ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, m, &module)));
      loomc_module_release(m);
      std::unique_ptr<CommandProgramBuilder> program;
      ASSERT_TRUE(Ok(module->DefineProgram("publish", {}, 4, &program)));
      OperationRegistry defs;
      std::string e;
      ASSERT_TRUE(DefinePublishToken(&defs, &e));
      OperationGraph graph(defs);
      auto token = graph.tensor(TensorType::vector(1, ElementType::I32), &e);
      auto control = graph.tensor(TensorType::vector(3, ElementType::I32), &e);
      auto history =
          graph.tensor(TensorType::vector(129, ElementType::I32), &e);
      auto progress = graph.tensor(TensorType::vector(8, ElementType::I32), &e);
      PublishTokenAttrs attrs{
          phase, 128, TerminalPolicy::StopOnTokenIds, {128001, 128008, 128009}};
      std::vector<shapes::Operand> inputs{{Port::Token, token},
                                          {Port::Control, control},
                                          {Port::History, history},
                                          {Port::Progress, progress}};
      if (existing) {
        ASSERT_TRUE(graph.operation(OpCode::PublishToken)
                        .bind(Port::Token, token)
                        .bind(Port::Control, control)
                        .bind(Port::History, history)
                        .bind(Port::Progress, progress)
                        .attributes(attrs)
                        .append(&e))
            << e;
        NativeEmissionSession session(registry, graph, *module, *program);
        for (size_t i = 0; i < inputs.size(); ++i)
          ASSERT_TRUE(Ok(session.Bind(inputs[i].value, program->Binding(i))));
        ASSERT_TRUE(Ok(session.EmitExisting(id, graph.invocations().front())));
      } else {
        InvocationProgram author(registry, graph, *module, *program);
        for (size_t i = 0; i < inputs.size(); ++i)
          ASSERT_TRUE(Ok(author.Bind(inputs[i].value, program->Binding(i))));
        ValueId output;
        auto rejected = author.Invoke(id, attrs, inputs, &output);
        EXPECT_FALSE(loomc_status_is_ok(rejected));
        loomc_status_free(rejected);
        EXPECT_TRUE(graph.invocations().empty());
        auto wrong = attrs;
        wrong.history_capacity = 64;
        rejected = author.InvokeEffect(id, wrong, inputs);
        EXPECT_FALSE(loomc_status_is_ok(rejected));
        loomc_status_free(rejected);
        ASSERT_TRUE(Ok(author.InvokeEffect(id, attrs, inputs)));
        EXPECT_EQ(graph.invocations().size(), 1u);
        EXPECT_EQ(graph.values().size(), 4u);
        EXPECT_FALSE(output.valid());
        for (size_t i = 0; i < inputs.size(); ++i) {
          Value native;
          ASSERT_TRUE(Ok(author.Native(inputs[i].value, &native)));
          ASSERT_TRUE(Ok(program->InspectNative(
              {native, program->Binding(i)},
              [](const loom_module_t *,
                 const std::vector<loom_value_id_t> &ids) {
                EXPECT_EQ(ids[0], ids[1]);
                return iree_ok_status();
              })));
        }
      }
      ASSERT_TRUE(Ok(program->Finish()));
      loomc_module_t *linked = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(
          Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      std::string text;
      ASSERT_TRUE(Ok(Dump(linked, &text)));
      EXPECT_EQ(text.find("buffer.alloca"), std::string::npos);
      EXPECT_NE(text.find("llama32_finish_isolated"), std::string::npos);
      loomc_result_release(result);
      loomc_module_release(linked);
    }
}
} // namespace
} // namespace hrx::dispatch
