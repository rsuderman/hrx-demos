// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
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
class CombinedChainTest : public ::testing::Test {
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

TEST_F(CombinedChainTest, AllocatedEmbeddingNormProjectionValuesChainAcrossFamilies) {
  auto registry = Registry::Llama3B();
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(Load(context, workspace, registry,
      {Id::Embedding, Id::RmsNorm, Id::Projection, Id::OutputHead,
       Id::PublishPrefill, Id::PublishDecode, Id::GateUpPrefill, Id::GateUpDecode,
       Id::DownPrefill, Id::DownDecode, Id::Rope, Id::RotatePublishKV,
       Id::CausalAttention}, root, &definitions, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("combined_chain", {}, 3, &program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineEmbedding(&ops, &error));
  ASSERT_TRUE(DefineRmsNorm(&ops, &error));
  ASSERT_TRUE(DefineLinear(&ops, &error));
  OperationGraph graph(ops);
  auto ids = graph.tensor(TensorType::vector(3, ElementType::I32), &error);
  auto type = TensorType::matrix(3072, 128256, ElementType::F32);
  type.setEncoding(Encoding::Q6_K).setLayout(Layout::GgmlWeightRows);
  auto embedding_weight = graph.tensor(type, &error);
  type = TensorType::matrix(3072, 13, ElementType::F32);
  type.setEncoding(Encoding::Q4_K).setLayout(Layout::GgmlWeightRows);
  auto projection_weight = graph.tensor(type, &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(author.Bind(ids, program->Binding(0))));
  ASSERT_TRUE(Ok(author.Parameter(embedding_weight, program->Binding(1), "token_embd.weight")));
  ASSERT_TRUE(Ok(author.Parameter(projection_weight, program->Binding(1), "projection.weight")));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(2))));
  ValueId embedded, normalized, projected;
  ASSERT_TRUE(Ok(author.Invoke(Id::Embedding, EmbeddingAttrs{128256,3072},
      {{Port::TokenIds,ids},{Port::Weight,embedding_weight}},&embedded)));
  ASSERT_TRUE(Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
      {{Port::Input,embedded},{Port::Gamma,gamma}},&normalized)));
  ASSERT_TRUE(Ok(author.Invoke(Id::Projection, LinearAttrs{3072,13},
      {{Port::Input,normalized},{Port::Weight,projection_weight}},&projected)));
  ASSERT_EQ(graph.invocations().size(),3u);
  EXPECT_EQ(graph.invocations()[1].binding(Port::Input)->value(),embedded);
  EXPECT_EQ(graph.invocations()[2].binding(Port::Input)->value(),normalized);
  EXPECT_EQ(graph.value(projected)->tensorType()->dimension(0),3u);
  EXPECT_EQ(graph.value(projected)->tensorType()->dimension(1),13u);
  Value first,second,third;
  ASSERT_TRUE(Ok(author.Native(embedded,&first)));
  ASSERT_TRUE(Ok(author.Native(normalized,&second)));
  ASSERT_TRUE(Ok(author.Native(projected,&third)));
  ASSERT_TRUE(Ok(program->InspectNative({first,second,third},
      [](const loom_module_t *,const std::vector<loom_value_id_t>& ids) {
        EXPECT_NE(ids[0],ids[1]); EXPECT_NE(ids[1],ids[2]); EXPECT_NE(ids[0],ids[2]);
        return iree_ok_status();
      })));
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked=nullptr;
  result=nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()},&linked,&result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::string text;
  ASSERT_TRUE(Ok(Dump(linked, &text)));
  EXPECT_NE(text.find("llama32_token_embedding_q6k_f32"), std::string::npos);
  EXPECT_NE(text.find("ggml_linear_q4k_q8_1_x4_compat"), std::string::npos);
  Compile(context,workspace,environment,linked,"combined_chain");
  loomc_module_release(linked);
}
} // namespace
} // namespace hrx::dispatch
