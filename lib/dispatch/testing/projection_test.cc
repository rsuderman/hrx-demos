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
class ProjectionTest : public ::testing::Test {
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
TEST_F(ProjectionTest, ProjectionAllocatesPacksLinksAndCompilesBothEncodings) {
  auto registry = Registry::Llama3B();
  EXPECT_EQ(registry.Implementations(Id::Projection).size(), 2u);
  for (auto encoding : {Encoding::Q4_K, Encoding::Q6_K}) {
    for (uint64_t rows : {1, 3, 9}) {
      auto entry = *registry.Find(Id::Projection);
      auto *definitions = LoadEntry(entry);
      ASSERT_NE(definitions, nullptr);
      std::unique_ptr<ModuleBuilder> module;
      ASSERT_TRUE(
          Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
      loomc_module_release(definitions);
      std::unique_ptr<CommandProgramBuilder> program;
      ASSERT_TRUE(Ok(module->DefineProgram("projection", {}, 2, &program)));
      OperationRegistry ops;
      std::string error;
      ASSERT_TRUE(DefineLinear(&ops, &error));
      OperationGraph graph(ops);
      auto x =
          graph.tensor(TensorType::matrix(rows, 256, ElementType::F32), &error);
      auto wt = TensorType::matrix(256, 13, ElementType::F32);
      wt.setEncoding(encoding).setLayout(Layout::GgmlWeightRows);
      auto w = graph.tensor(wt, &error);
      InvocationProgram author(registry, graph, *module, *program);
      ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
      ASSERT_TRUE(
          Ok(author.Parameter(w, program->Binding(1), "projection.weight")));
      ValueId y;
      ASSERT_TRUE(Ok(author.Invoke(Id::Projection, LinearAttrs{256, 13},
                                   {{Port::Input, x}, {Port::Weight, w}}, &y)));
      ASSERT_TRUE(y.valid());
      EXPECT_EQ(graph.value(y)->tensorType()->dimension(0), rows);
      EXPECT_EQ(graph.value(y)->tensorType()->dimension(1), 13u);
      ASSERT_TRUE(Ok(program->Finish()));
      loomc_module_t *linked = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(
          Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      std::string text;
      ASSERT_TRUE(Ok(Dump(linked, &text)));
      EXPECT_NE(text.find("projection.weight"), std::string::npos);
      EXPECT_NE(text.find("buffer.alloca"), std::string::npos);
      EXPECT_NE(text.find(encoding == Encoding::Q4_K
                              ? "ggml_linear_q4k_q8_1_x4_compat"
                              : "ggml_linear_q6k_q8_1_x4_compat"),
                std::string::npos);
      Compile(context, workspace, environment, linked, "projection");
      loomc_module_release(linked);
    }
  }
}
TEST_F(ProjectionTest, ProjectionRejectsUnsupportedRepresentationAndCapacity) {
  auto registry = Registry::Llama3B();
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineLinear(&ops, &error));
  for (auto encoding : {Encoding::Dense, Encoding::Q4_K, Encoding::Q6_K}) {
    for (uint64_t width : {256, 17664, 33024}) {
      OperationGraph graph(ops);
      auto x =
          graph.tensor(TensorType::matrix(1, width, ElementType::F32), &error);
      auto wt = TensorType::matrix(width, 13, ElementType::F32);
      wt.setEncoding(encoding).setLayout(Layout::GgmlWeightRows);
      auto w = graph.tensor(wt, &error);
      LinearAttrs attrs{width, 13};
      std::vector<shapes::Operand> inputs{{Port::Input, x}, {Port::Weight, w}};
      shapes::Result result;
      ASSERT_TRUE(shapes::InferResult(graph, OpCode::Linear, attrs, inputs,
                                      &result, &error));
      OperationAttrs attr = attrs;
      InvocationRequest request{graph, attr, inputs, result};
      auto *selected = registry.Resolve(Id::Projection, request, {}, &error);
      const bool supported = encoding != Encoding::Dense && width <= 32768 &&
                             (encoding != Encoding::Q6_K || width <= 17408);
      EXPECT_EQ(selected != nullptr, supported) << width << " " << error;
      if (selected) {
        EXPECT_EQ(selected->variant,
                  encoding == Encoding::Q4_K ? "q4_k" : "q6_k");
      }
      EXPECT_EQ(
          registry.Resolve(Id::Projection, request, "nonexistent", &error),
          nullptr);
    }
  }
}
} // namespace
} // namespace hrx::dispatch
