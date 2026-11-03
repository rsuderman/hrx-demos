// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/llama/detail/ffn_structure.h"
#include "lib/llama/detail/structure_construction.h"
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
class InvocationTest : public ::testing::Test {
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

void Save(loomc_module_t *module, const std::string &name) {
  const auto *directory = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (!directory)
    return;
  auto path = std::string(directory) + "/" + name + ".loom";
  EXPECT_TRUE(Ok(loomc_module_serialize_to_path(
      module, nullptr, loomc_make_cstring_view(path.c_str()),
      loomc_allocator_system())));
  loomc_module_serialize_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
  options.structure_size = sizeof(options);
  options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  path = std::string(directory) + "/" + name + ".loombc";
  EXPECT_TRUE(Ok(loomc_module_serialize_to_path(
      module, &options, loomc_make_cstring_view(path.c_str()),
      loomc_allocator_system())));
}

TEST_F(InvocationTest, NormCreatesActualDisjointNativeOutputAndCompiles) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  ASSERT_NE(definitions, nullptr);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("norm", {}, 2, &program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineRmsNorm(&ops, &error));
  OperationGraph graph(ops);
  auto x = graph.tensor(TensorType::matrix(2, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
  ValueId y;
  ASSERT_TRUE(Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                               {{Port::Input, x}, {Port::Gamma, gamma}}, &y)));
  EXPECT_NE(y, x);
  EXPECT_NE(y, gamma);
  ASSERT_NE(graph.value(y), nullptr);
  EXPECT_EQ(graph.value(y)->tensorType()->dimension(0), 2u);
  EXPECT_EQ(graph.value(y)->tensorType()->dimension(1), 3072u);
  ASSERT_EQ(graph.invocations().size(), 1u);
  EXPECT_EQ(graph.invocations()[0].binding(Port::Output)->value(), y);
  Value native;
  ASSERT_TRUE(Ok(author.Native(y, &native)));
  // Inspect the same scoped buffer wired to the actual kernel invocation.
  ASSERT_TRUE(Ok(program->InspectNative(
      {program->Binding(0), program->Binding(1), native},
      [](const loom_module_t *module, const std::vector<loom_value_id_t> &ids) {
        EXPECT_NE(ids[2], ids[0]);
        EXPECT_NE(ids[2], ids[1]);
        bool found = false;
        for (size_t i = 0; i < module->symbols.count; ++i) {
          const auto *def = module->symbols.entries[i].defining_op;
          if (!def || !loom_command_program_def_isa(def))
            continue;
          auto *body =
              loom_region_const_entry_block(loom_command_program_def_body(def));
          auto *block = loom_region_const_entry_block(
              loom_command_serial_body(body->first_op));
          for (auto *op = block->first_op; op; op = op->next_op) {
            if (!loom_kernel_launch_isa(op))
              continue;
            auto args = loom_kernel_launch_arguments(op);
            EXPECT_EQ(args.count, 4u);
            EXPECT_EQ(args.values[1], ids[0]);
            EXPECT_EQ(args.values[2], ids[1]);
            EXPECT_EQ(args.values[3], ids[2]);
            found = true;
          }
        }
        EXPECT_TRUE(found);
        return iree_ok_status();
      })));
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::string text;
  ASSERT_TRUE(Ok(Dump(linked, &text)));
  EXPECT_NE(text.find("buffer.alloc"), std::string::npos);
  EXPECT_NE(text.find("kernel.launch"), std::string::npos);
  Save(linked, "norm");
  Compile(context, workspace, environment, linked, "norm");
  loomc_module_release(linked);
}

TEST_F(InvocationTest, WholeFfnBothPhasesComposeOutputsAndResidualAlias) {
  for (bool decode : {false, true})
    for (auto down_encoding : {Encoding::Q4_K, Encoding::Q6_K}) {
      auto registry = Registry::Llama3B();
      std::vector<Registration> entries{
          *registry.Find(Id::RmsNorm),
          *registry.Find(decode ? Id::GateUpDecode : Id::GateUpPrefill),
          *registry.Find(decode ? Id::DownDecode : Id::DownPrefill)};
      loomc_module_t *definitions = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(
          Ok(Load(context, workspace, entries, root, &definitions, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      std::unique_ptr<ModuleBuilder> module;
      ASSERT_TRUE(
          Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
      loomc_module_release(definitions);
      std::unique_ptr<CommandProgramBuilder> program;
      ASSERT_TRUE(Ok(module->DefineProgram("ffn", {}, 4, &program)));
      OperationRegistry ops;
      std::string error;
      ASSERT_TRUE(DefineRmsNorm(&ops, &error));
      ASSERT_TRUE(DefineGatedMatmul(&ops, &error));
      ASSERT_TRUE(DefineLinearResidual(&ops, &error));
      OperationGraph graph(ops);
      const uint64_t rows = decode ? 1 : 2;
      auto x = graph.tensor(TensorType::matrix(rows, 3072, ElementType::F32),
                            &error);
      auto gamma =
          graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
      auto weight = [&](uint64_t input, uint64_t output, Encoding encoding) {
        auto type = TensorType::matrix(input, output, ElementType::F32);
        type.setLayout(Layout::GgmlWeightRows).setEncoding(encoding);
        return graph.tensor(type, &error);
      };
      auto gate = weight(3072, 8192, Encoding::Q4_K);
      auto up = weight(3072, 8192, Encoding::Q4_K);
      auto down = weight(8192, 3072, down_encoding);
      InvocationProgram author(registry, graph, *module, *program);
      ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
      ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
      ASSERT_TRUE(Ok(author.Parameter(gate, program->Binding(2),
                                      "blk.0.ffn_gate.weight")));
      ASSERT_TRUE(
          Ok(author.Parameter(up, program->Binding(2), "blk.0.ffn_up.weight")));
      ASSERT_TRUE(Ok(author.Parameter(down, program->Binding(2),
                                      "blk.0.ffn_down.weight")));
      ASSERT_TRUE(Ok(author.Control(program->Binding(3))));
      ValueId normalized, activation, output;
      ASSERT_TRUE(Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                                   {{Port::Input, x}, {Port::Gamma, gamma}},
                                   &normalized)));
      ASSERT_TRUE(Ok(author.Invoke(entries[1].id, GatedMatmulAttrs{3072, 8192},
                                   {{Port::Input, normalized},
                                    {Port::GateWeight, gate},
                                    {Port::UpWeight, up}},
                                   &activation)));
      EXPECT_EQ(graph.value(activation)->tensorType()->dimension(0), rows);
      EXPECT_EQ(graph.value(activation)->tensorType()->dimension(1), 8192u);
      ASSERT_TRUE(
          Ok(author.Invoke(entries[2].id, LinearResidualAttrs{8192, 3072},
                           {{Port::Input, activation},
                            {Port::Weight, down},
                            {Port::Residual, x}},
                           &output)));
      EXPECT_EQ(output, x);
      EXPECT_NE(activation, normalized);
      EXPECT_EQ(graph.invocations().size(), 3u);
      Value native;
      ASSERT_TRUE(Ok(author.Native(output, &native)));
      ASSERT_TRUE(Ok(program->InspectNative(
          {native, program->Binding(0)},
          [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
            EXPECT_EQ(ids[0], ids[1]);
            return iree_ok_status();
          })));
      ASSERT_TRUE(Ok(program->Finish()));
      loomc_module_t *linked = nullptr;
      result = nullptr;
      ASSERT_TRUE(
          Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      std::string text;
      ASSERT_TRUE(Ok(Dump(linked, &text)));
      EXPECT_NE(text.find("template.apply"), std::string::npos);
      EXPECT_NE(text.find(entries[1].symbol), std::string::npos);
      EXPECT_NE(text.find(entries[2].symbol), std::string::npos);
      Save(linked, std::string(decode ? "decode" : "prefill") +
                       (down_encoding == Encoding::Q4_K ? "-q4" : "-q6"));
      Compile(context, workspace, environment, linked, "ffn");
      loomc_module_release(linked);
    }
}

TEST_F(InvocationTest,
       RejectsBadShapesOwnersMissingRegistrationAndCallerOutputBeforeMutation) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  ASSERT_NE(definitions, nullptr);
  std::unique_ptr<ModuleBuilder> module, foreign_module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  ASSERT_TRUE(Ok(
      ModuleBuilder::Create(context, workspace, definitions, &foreign_module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program, foreign_program;
  ASSERT_TRUE(Ok(module->DefineProgram("negative", {}, 2, &program)));
  ASSERT_TRUE(
      Ok(foreign_module->DefineProgram("foreign", {}, 1, &foreign_program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineRmsNorm(&ops, &error));
  OperationGraph graph(ops), foreign_graph(ops);
  auto x = graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  auto bad_gamma =
      graph.tensor(TensorType::vector(64, ElementType::F32), &error);
  auto foreign = foreign_graph.tensor(
      TensorType::matrix(1, 3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  EXPECT_FALSE(Ok(author.Bind(foreign, program->Binding(0))));
  EXPECT_FALSE(Ok(author.Bind(x, foreign_program->Binding(0))));
  ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
  ValueId output = x;
  auto reject = [&](Id id, RmsNormAttrs attrs,
                    std::vector<shapes::Operand> inputs) {
    EXPECT_FALSE(Ok(author.Invoke(id, attrs, inputs, &output)));
    EXPECT_EQ(output, x);
    EXPECT_TRUE(graph.invocations().empty());
  };
  reject(Id::RmsNorm, {3072}, {{Port::Input, foreign}, {Port::Gamma, gamma}});
  reject(Id::RmsNorm, {3072}, {{Port::Input, x}, {Port::Gamma, bad_gamma}});
  reject(Id::RmsNorm, {3072},
         {{Port::Input, x}, {Port::Gamma, gamma}, {Port::Output, x}});
  reject(Id::RmsNorm, {3072, 0.1}, {{Port::Input, x}, {Port::Gamma, gamma}});
  auto begin = graph.scalar(ScalarType::I64, &error);
  auto count = graph.scalar(ScalarType::I64, &error);
  reject(Id::RmsNorm, {3072},
         {{Port::Input, x},
          {Port::Gamma, gamma},
          {Port::RowBegin, begin},
          {Port::RowCount, count}});
  Registry empty;
  InvocationProgram missing(empty, graph, *module, *program);
  EXPECT_FALSE(
      Ok(missing.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                        {{Port::Input, x}, {Port::Gamma, gamma}}, &output)));
  Registry wrong;
  auto entry = *registry.Find(Id::RmsNorm);
  entry.symbol = "missing_norm";
  ASSERT_TRUE(wrong.Add(entry, &error));
  InvocationProgram bad_registration(wrong, graph, *module, *program);
  ASSERT_TRUE(Ok(bad_registration.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(bad_registration.Bind(gamma, program->Binding(1))));
  EXPECT_FALSE(Ok(bad_registration.Invoke(
      Id::RmsNorm, RmsNormAttrs{3072}, {{Port::Input, x}, {Port::Gamma, gamma}},
      &output)));
  EXPECT_TRUE(graph.invocations().empty());
  ASSERT_TRUE(
      Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                       {{Port::Input, x}, {Port::Gamma, gamma}}, &output)));
  EXPECT_NE(output, x);
  ASSERT_TRUE(Ok(program->Finish()));
}

TEST_F(InvocationTest, ScopedResultsDoNotEscapeTheirNativeBody) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("scope", {}, 2, &program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineRmsNorm(&ops, &error));
  OperationGraph graph(ops);
  auto x = graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
  ValueId nested;
  ASSERT_TRUE(Ok(program->Serial([&](CommandProgramBuilder &) {
    return author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                         {{Port::Input, x}, {Port::Gamma, gamma}}, &nested);
  })));
  Value native;
  EXPECT_FALSE(Ok(author.Native(nested, &native)));
  ValueId ignored;
  EXPECT_FALSE(Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                                {{Port::Input, nested}, {Port::Gamma, gamma}},
                                &ignored)));
  EXPECT_EQ(graph.invocations().size(), 1u);
  ASSERT_TRUE(Ok(program->Finish()));
}

TEST_F(InvocationTest, CommandFamilyChecksOwnerCountAndDependentTypes) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::GateUpDecode));
  ASSERT_NE(definitions, nullptr);
  std::unique_ptr<ModuleBuilder> module, foreign;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &foreign)));
  loomc_module_release(definitions);
  CommandFamilyRef family, other;
  ASSERT_TRUE(Ok(module->CommandFamily("llm.ffn.gate_up_decode", &family)));
  ASSERT_TRUE(Ok(foreign->CommandFamily("llm.ffn.gate_up_decode", &other)));
  EXPECT_FALSE(Ok(module->CommandFamily("missing", &other)));
  ASSERT_TRUE(Ok(foreign->CommandFamily("llm.ffn.gate_up_decode", &other)));
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("apply", {}, 1, &program)));
  EXPECT_TRUE(Ok(module->CheckCommandFamilyInterface(family, family)));
  EXPECT_FALSE(Ok(module->CheckCommandFamilyInterface(family, other)));
  EXPECT_FALSE(Ok(program->Apply(other, {})));
  EXPECT_FALSE(Ok(program->Apply(family, {})));
  CommandValues values(*program);
  Value index, boolean, wrong_bytes, bytes, view;
  ASSERT_TRUE(Ok(values.Constant(3072, LOOM_SCALAR_TYPE_INDEX, &index)));
  ASSERT_TRUE(Ok(values.Constant(1, LOOM_SCALAR_TYPE_I1, &boolean)));
  ASSERT_TRUE(Ok(values.Constant(14155776, LOOM_SCALAR_TYPE_INDEX, &bytes)));
  ASSERT_TRUE(Ok(values.Constant(64, LOOM_SCALAR_TYPE_INDEX, &wrong_bytes)));
  ASSERT_TRUE(Ok(values.Parameter(program->Binding(0), "wrong", {},
                                  LOOM_SCALAR_TYPE_I8, {wrong_bytes}, &view)));
  EXPECT_FALSE(Ok(program->Apply(
      family, {index, index, boolean, boolean, boolean, boolean, bytes, index,
               view, view, program->Binding(0), program->Binding(0)})));
  // Recoverable preflight leaves the existing owner usable.
  ASSERT_TRUE(Ok(program->Finish()));
}
TEST_F(InvocationTest, MutationFailureDisposesExistingNativeOwner) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("failure", {}, 2, &program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(
      ops.define(OpCode::RmsNorm)
          .port(Port::Input, ValueKind::Tensor, Access::Read)
          .port(Port::Gamma, ValueKind::Tensor, Access::Read)
          .port(Port::Output, ValueKind::Tensor, Access::Write)
          .attributes<RmsNormAttrs>()
          .verify(+[](const BoundOperands &, const RmsNormAttrs &,
                      std::string *error) {
            *error = "fixture rejects graph publication after native staging";
            return false;
          })
          .finish(&error));
  OperationGraph graph(ops);
  auto x = graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
  ValueId output = x;
  EXPECT_FALSE(
      Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                       {{Port::Input, x}, {Port::Gamma, gamma}}, &output)));
  EXPECT_EQ(output, x);
  EXPECT_TRUE(graph.invocations().empty());
  EXPECT_FALSE(Ok(program->Finish()));
  Value ignored;
  EXPECT_FALSE(Ok(author.Native(x, &ignored)));
  EXPECT_FALSE(Ok(author.Bind(output, program->Binding(0))));
}

TEST_F(InvocationTest,
       RejectsNormAttributesDisagreeingWithActualNativeConfiguration) {
  for (bool change_width : {false, true}) {
    auto registry = Registry::Llama3B();
    auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    loomc_module_release(definitions);
    ASSERT_TRUE(Ok(module->BuildNative([&](loom_module_t *native,
                                           loom_builder_t *) {
      const std::string expected =
          change_width ? "llama.model.hidden_size" : "llama.model.rms_epsilon";
      for (size_t i = 0; i < native->symbols.count; ++i) {
        const auto &symbol = native->symbols.entries[i];
        auto name = loom_string_table_get(&native->strings, symbol.name_id);
        if (std::string(name.data, name.size) == expected)
          return loom_op_set_attr(native, symbol.defining_op, 1,
                                  change_width ? loom_attr_i64(2048)
                                               : loom_attr_f64(0.1));
      }
      return iree_make_status(IREE_STATUS_NOT_FOUND, "missing fixture config");
    })));
    std::unique_ptr<CommandProgramBuilder> program;
    ASSERT_TRUE(Ok(module->DefineProgram("config", {}, 2, &program)));
    OperationRegistry ops;
    std::string error;
    ASSERT_TRUE(DefineRmsNorm(&ops, &error));
    OperationGraph graph(ops);
    auto x =
        graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
    auto gamma =
        graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
    InvocationProgram author(registry, graph, *module, *program);
    ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
    ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
    ValueId output;
    EXPECT_FALSE(
        Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                         {{Port::Input, x}, {Port::Gamma, gamma}}, &output)));
    EXPECT_FALSE(output.valid());
    EXPECT_EQ(graph.values().size(), 2u);
    EXPECT_TRUE(graph.invocations().empty());
    ASSERT_TRUE(Ok(program->Finish()));
  }
}

TEST_F(InvocationTest,
       IncompatibleSelectedInterfacesRejectBeforeStagingAndRecover) {
  auto registry = Registry::Llama3B();
  std::vector<Registration> entries{*registry.Find(Id::RmsNorm),
                                    *registry.Find(Id::GateUpDecode)};
  entries[0].provider_files.push_back(
      "lib/dispatch/testing/testdata/incompatible_family.loom");
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(
      Ok(Load(context, workspace, entries, root, &definitions, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("interface_recovery", {}, 3, &program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineRmsNorm(&ops, &error));
  ASSERT_TRUE(DefineGatedMatmul(&ops, &error));
  OperationGraph graph(ops);
  auto x = graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  auto weight_type = TensorType::matrix(3072, 8192, ElementType::F32);
  weight_type.setLayout(Layout::GgmlWeightRows).setEncoding(Encoding::Q4_K);
  auto gate = graph.tensor(weight_type, &error),
       up = graph.tensor(weight_type, &error);
  auto norm_entry = *registry.Find(Id::RmsNorm);
  norm_entry.symbol = "ggml_quantize_q8_1_x4_qk_compat_f32";
  auto family_entry = *registry.Find(Id::GateUpDecode);
  family_entry.symbol = "invocation.incompatible_gate";
  Registry bad_registry;
  ASSERT_TRUE(bad_registry.Add(norm_entry, &error));
  ASSERT_TRUE(bad_registry.Add(family_entry, &error));
  InvocationProgram bad(bad_registry, graph, *module, *program);
  ASSERT_TRUE(Ok(bad.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(bad.Bind(gamma, program->Binding(1))));
  ASSERT_TRUE(Ok(bad.Parameter(gate, program->Binding(2), "gate")));
  ASSERT_TRUE(Ok(bad.Parameter(up, program->Binding(2), "up")));
  size_t native_value_count = 0;
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        native_value_count = native->values.count;
        return iree_ok_status();
      })));
  ValueId output = x;
  EXPECT_FALSE(
      Ok(bad.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                    {{Port::Input, x}, {Port::Gamma, gamma}}, &output)));
  EXPECT_FALSE(Ok(bad.Invoke(
      Id::GateUpDecode, GatedMatmulAttrs{3072, 8192},
      {{Port::Input, x}, {Port::GateWeight, gate}, {Port::UpWeight, up}},
      &output)));
  EXPECT_EQ(output, x);
  EXPECT_EQ(graph.values().size(), 4u);
  EXPECT_TRUE(graph.invocations().empty());
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        EXPECT_EQ(native->values.count, native_value_count);
        return iree_ok_status();
      })));
  // Both failures preserve the same author, graph, body and module. Replace
  // only the selected catalog and continue through the same invocation object.
  bad_registry = registry;
  ValueId normalized, activation;
  ASSERT_TRUE(
      Ok(bad.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
                    {{Port::Input, x}, {Port::Gamma, gamma}}, &normalized)));
  ASSERT_TRUE(Ok(bad.Invoke(Id::GateUpDecode, GatedMatmulAttrs{3072, 8192},
                            {{Port::Input, normalized},
                             {Port::GateWeight, gate},
                             {Port::UpWeight, up}},
                            &activation)));
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  Compile(context, workspace, environment, linked, "interface_recovery");
  loomc_module_release(linked);
}

struct ExistingFfnFixture {
  std::unique_ptr<llama::detail::StructureConstruction> construction;
  llama::detail::NormalizedFfnRegion region;
  static ExistingFfnFixture Create(uint64_t rows, Phase phase, Encoding down,
                                   std::string *error) {
    ExistingFfnFixture fixture;
    OperationRegistry ops;
    if (!DefineRmsNorm(&ops, error) || !DefineGatedMatmul(&ops, error) ||
        !DefineLinearResidual(&ops, error))
      return fixture;
    model::ModelDescriptor facts;
    facts.family = "llama";
    facts.embedding_width = 3072;
    facts.vocabulary_size = 128256;
    facts.rms_epsilon = 1e-5;
    facts.key_head_width = 128;
    facts.value_head_width = 128;
    model::LayerDescriptor layer;
    layer.index = 7;
    layer.query_heads = 24;
    layer.kv_heads = 8;
    layer.feed_forward_width = 8192;
    auto tensor = [](const char *role, const char *key, uint32_t storage,
                     int64_t in, int64_t out) {
      model::TensorBinding binding;
      binding.role = role;
      binding.tensor.name = key;
      binding.tensor.storage_type = storage;
      binding.tensor.rank = out == 1 ? 1 : 2;
      binding.tensor.normalized_extents = {in, out, 1, 1};
      binding.tensor.absolute_offset = 4096;
      binding.tensor.byte_size =
          storage == 0 ? in * out * 4
                       : in * out / 256 * (storage == 12 ? 144 : 210);
      return binding;
    };
    layer.tensors = {
        tensor("post_norm", "blk.7.ffn_norm.weight", 0, 3072, 1),
        tensor("ffn_gate", "blk.7.ffn_gate.weight", 12, 3072, 8192),
        tensor("ffn_up", "blk.7.ffn_up.weight", 12, 3072, 8192),
        tensor("ffn_down", "blk.7.ffn_down.weight",
               down == Encoding::Q4_K ? 12 : 14, 8192, 3072)};
    facts.layers.push_back(std::move(layer));
    fixture.construction = llama::detail::StructureConstruction::Create(
        std::move(facts), {phase, rows, 16, 32, "fixture"}, ops, error);
    if (!fixture.construction || !fixture.construction->createInputs(error))
      return {};
    std::array<ValueId, 4> weights;
    for (size_t i = 0; i < weights.size(); ++i)
      if (!fixture.construction->resolve({llama::ParameterScope::Layer, 0, i},
                                         &weights[i], error))
        return {};
    const auto &owner = fixture.construction->structure();
    const auto &input = owner.inputs();
    if (!llama::detail::AppendNormalizedFfnResolved(
            &fixture.construction->graph(), owner.facts(), 0, input.residual,
            weights, input.row_begin, input.row_count, &fixture.region, error))
      return {};
    return fixture;
  }
};

TEST_F(InvocationTest,
       ExistingConstOwnerFfnPreservesIdentitiesAndCompilesBothPhases) {
  for (auto phase : {Phase::Prefill, Phase::Decode})
    for (auto down : {Encoding::Q4_K, Encoding::Q6_K}) {
      std::string error;
      auto fixture = ExistingFfnFixture::Create(1, phase, down, &error);
      ASSERT_TRUE(fixture.construction) << error;
      const auto &owner = fixture.construction->structure();
      const auto &graph = owner.graph();
      const auto values = graph.values().size(),
                 invocations = graph.invocations().size();
      std::vector<ValueId> identities;
      for (const auto &value : graph.values())
        identities.push_back(value.id());
      auto registry = Registry::Llama3B();
      const auto gate =
          phase == Phase::Decode ? Id::GateUpDecode : Id::GateUpPrefill;
      const auto projection =
          phase == Phase::Decode ? Id::DownDecode : Id::DownPrefill;
      std::vector<Registration> entries{*registry.Find(Id::RmsNorm),
                                        *registry.Find(gate),
                                        *registry.Find(projection)};
      loomc_module_t *definitions = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(
          Ok(Load(context, workspace, entries, root, &definitions, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      std::unique_ptr<ModuleBuilder> module;
      ASSERT_TRUE(
          Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
      loomc_module_release(definitions);
      std::unique_ptr<CommandProgramBuilder> program;
      ASSERT_TRUE(Ok(module->DefineProgram("existing_ffn", {}, 4, &program)));
      NativeEmissionSession session(registry, graph, *module, *program);
      EXPECT_EQ(&session.graph(), &owner.graph());
      ASSERT_TRUE(Ok(session.SpecializeOneRow(owner.inputs().row_begin,
                                              owner.inputs().row_count)));
      EXPECT_TRUE(session.one_row_specialized());
      ASSERT_TRUE(
          Ok(session.Bind(owner.inputs().residual, program->Binding(0))));
      // Named gamma staging belongs to the next parameter checkpoint. This test
      // exercises only const graph/output ownership with an explicit gamma
      // boundary.
      const auto gamma = fixture.region.parameters[0].value;
      ASSERT_TRUE(Ok(session.Bind(gamma, program->Binding(1))));
      for (size_t i = 1; i < fixture.region.parameters.size(); ++i) {
        auto value = fixture.region.parameters[i].value;
        const auto *parameter =
            owner.parameter({llama::ParameterScope::Layer, 0, i});
        ASSERT_NE(parameter, nullptr);
        ASSERT_TRUE(Ok(session.Parameter(value, program->Binding(2),
                                         parameter->tensor.name)));
      }
      ASSERT_TRUE(Ok(session.Control(program->Binding(3))));
      std::array<Id, 3> selected{Id::RmsNorm, gate, projection};
      for (size_t i = 0; i < selected.size(); ++i)
        ASSERT_TRUE(
            Ok(session.EmitExisting(selected[i], graph.invocations()[i])));
      EXPECT_FALSE(
          Ok(session.EmitExisting(projection, graph.invocations()[2])));
      EXPECT_EQ(graph.values().size(), values);
      EXPECT_EQ(graph.invocations().size(), invocations);
      for (size_t i = 0; i < identities.size(); ++i)
        EXPECT_EQ(graph.values()[i].id(), identities[i]);
      EXPECT_EQ(graph.invocations()[0].binding(Port::Output)->value(),
                fixture.region.normalized);
      EXPECT_EQ(graph.invocations()[1].binding(Port::Output)->value(),
                fixture.region.activation);
      EXPECT_EQ(graph.invocations()[2].binding(Port::Residual)->value(),
                owner.inputs().residual);
      Value normalized, activation, residual;
      ASSERT_TRUE(Ok(session.Native(fixture.region.normalized, &normalized)));
      ASSERT_TRUE(Ok(session.Native(fixture.region.activation, &activation)));
      ASSERT_TRUE(Ok(session.Native(fixture.region.residual, &residual)));
      ASSERT_TRUE(Ok(program->InspectNative(
          {normalized, activation, residual, program->Binding(0)},
          [](const loom_module_t *, const std::vector<loom_value_id_t> &ids) {
            EXPECT_NE(ids[0], ids[1]);
            EXPECT_NE(ids[0], ids[2]);
            EXPECT_NE(ids[1], ids[2]);
            EXPECT_EQ(ids[2], ids[3]);
            return iree_ok_status();
          })));
      ASSERT_TRUE(Ok(program->Finish()));
      loomc_module_t *linked = nullptr;
      result = nullptr;
      ASSERT_TRUE(
          Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      Save(linked, std::string(phase == Phase::Decode ? "existing-decode"
                                                      : "existing-prefill") +
                       (down == Encoding::Q4_K ? "-q4" : "-q6"));
      Compile(context, workspace, environment, linked, "existing_ffn");
      loomc_module_release(linked);
    }
}

TEST_F(InvocationTest,
       OneRowCapacityAndRuntimeControlsDoNotImplySpecialization) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("specialization", {}, 2, &program)));
  std::string error;
  auto fixture =
      ExistingFfnFixture::Create(1, Phase::Decode, Encoding::Q6_K, &error);
  auto foreign =
      ExistingFfnFixture::Create(1, Phase::Decode, Encoding::Q6_K, &error);
  ASSERT_TRUE(fixture.construction) << error;
  ASSERT_TRUE(foreign.construction);
  const auto &owner = fixture.construction->structure();
  NativeEmissionSession session(registry, owner.graph(), *module, *program);
  ASSERT_TRUE(Ok(session.Bind(owner.inputs().residual, program->Binding(0))));
  ASSERT_TRUE(Ok(
      session.Bind(fixture.region.parameters[0].value, program->Binding(1))));
  size_t before = 0;
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        before = native->values.count;
        return iree_ok_status();
      })));
  EXPECT_FALSE(session.one_row_specialized());
  EXPECT_FALSE(
      Ok(session.EmitExisting(Id::RmsNorm, owner.graph().invocations()[0])));
  EXPECT_FALSE(Ok(session.SpecializeOneRow(
      foreign.construction->structure().inputs().row_begin,
      owner.inputs().row_count)));
  EXPECT_FALSE(Ok(session.SpecializeOneRow(owner.inputs().row_begin,
                                           owner.inputs().row_begin)));
  ASSERT_TRUE(Ok(session.SpecializeOneRow(owner.inputs().row_begin,
                                          owner.inputs().row_count)));
  EXPECT_FALSE(Ok(session.EmitExisting(
      Id::RmsNorm,
      foreign.construction->structure().graph().invocations()[0])));
  auto copied = owner.graph().invocations()[0];
  EXPECT_FALSE(Ok(session.EmitExisting(Id::RmsNorm, copied)));
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        EXPECT_EQ(native->values.count, before);
        return iree_ok_status();
      })));
  ASSERT_TRUE(
      Ok(session.EmitExisting(Id::RmsNorm, owner.graph().invocations()[0])));
  EXPECT_FALSE(Ok(session.SpecializeOneRow(owner.inputs().row_begin,
                                           owner.inputs().row_count)));
  ASSERT_TRUE(Ok(program->Finish()));
}

TEST_F(InvocationTest,
       OneRowSpecializationRejectsLargerCapacityBeforeMutation) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("capacity", {}, 2, &program)));
  std::string error;
  auto fixture =
      ExistingFfnFixture::Create(2, Phase::Prefill, Encoding::Q6_K, &error);
  ASSERT_TRUE(fixture.construction) << error;
  const auto &owner = fixture.construction->structure();
  NativeEmissionSession session(registry, owner.graph(), *module, *program);
  ASSERT_TRUE(Ok(session.Bind(owner.inputs().residual, program->Binding(0))));
  ASSERT_TRUE(Ok(
      session.Bind(fixture.region.parameters[0].value, program->Binding(1))));
  ASSERT_TRUE(Ok(session.SpecializeOneRow(owner.inputs().row_begin,
                                          owner.inputs().row_count)));
  size_t before = 0;
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        before = native->values.count;
        return iree_ok_status();
      })));
  EXPECT_FALSE(
      Ok(session.EmitExisting(Id::RmsNorm, owner.graph().invocations()[0])));
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        EXPECT_EQ(native->values.count, before);
        return iree_ok_status();
      })));
  EXPECT_EQ(owner.graph().invocations().size(), 3u);
  ASSERT_TRUE(Ok(program->Finish()));
}

TEST_F(InvocationTest,
       ExistingOutputShapeDisagreementRejectsWithoutNativeMutation) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("existing_shape", {}, 2, &program)));
  OperationRegistry ops;
  std::string error;
  // A permissive external definition makes an invalid output reachable for
  // this adapter test; native emission still uses built-in result inference.
  ASSERT_TRUE(ops.define(OpCode::RmsNorm)
                  .port(Port::Input, ValueKind::Tensor, Access::Read)
                  .port(Port::Gamma, ValueKind::Tensor, Access::Read)
                  .port(Port::Output, ValueKind::Tensor, Access::Write)
                  .attributes<RmsNormAttrs>()
                  .verify(+[](const BoundOperands &, const RmsNormAttrs &,
                              std::string *) { return true; })
                  .finish(&error));
  OperationGraph graph(ops);
  auto x = graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  auto wrong =
      graph.tensor(TensorType::matrix(1, 64, ElementType::F32), &error);
  ASSERT_TRUE(graph.operation(OpCode::RmsNorm)
                  .attributes(RmsNormAttrs{3072})
                  .bind(Port::Input, x)
                  .bind(Port::Gamma, gamma)
                  .bind(Port::Output, wrong)
                  .append(&error));
  NativeEmissionSession session(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(session.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(session.Bind(gamma, program->Binding(1))));
  size_t before = 0;
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        before = native->values.count;
        return iree_ok_status();
      })));
  EXPECT_FALSE(Ok(session.EmitExisting(Id::RmsNorm, graph.invocations()[0])));
  ASSERT_TRUE(
      Ok(program->InspectNative({}, [&](const loom_module_t *native,
                                        const std::vector<loom_value_id_t> &) {
        EXPECT_EQ(native->values.count, before);
        return iree_ok_status();
      })));
  EXPECT_EQ(graph.values().size(), 3u);
  EXPECT_EQ(graph.invocations().size(), 1u);
  ASSERT_TRUE(Ok(program->Finish()));
}

TEST_F(InvocationTest, ExistingWritesReuseLogicalStorageAndFeedLaterReads) {
  for (bool boundary_output : {false, true}) {
    auto registry = Registry::Llama3B();
    auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    loomc_module_release(definitions);
    std::unique_ptr<CommandProgramBuilder> program;
    ASSERT_TRUE(Ok(module->DefineProgram("existing_rewrite", {}, 4, &program)));
    OperationRegistry ops;
    std::string error;
    ASSERT_TRUE(DefineRmsNorm(&ops, &error));
    OperationGraph graph(ops);
    auto matrix = [&]() {
      return graph.tensor(TensorType::matrix(1, 3072, ElementType::F32),
                          &error);
    };
    auto first = matrix(), second = matrix(), rewritten = matrix(),
         consumed = matrix();
    auto gamma =
        graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
    auto append = [&](ValueId input, ValueId output) {
      return graph.operation(OpCode::RmsNorm)
          .attributes(RmsNormAttrs{3072})
          .bind(Port::Input, input)
          .bind(Port::Gamma, gamma)
          .bind(Port::Output, output)
          .append(&error);
    };
    ASSERT_TRUE(append(first, rewritten));
    ASSERT_TRUE(append(second, rewritten));
    ASSERT_TRUE(append(rewritten, consumed));
    NativeEmissionSession immutable(registry, graph, *module, *program);
    ASSERT_TRUE(Ok(immutable.Bind(first, program->Binding(0))));
    ASSERT_TRUE(Ok(immutable.Bind(gamma, program->Binding(2))));
    ASSERT_TRUE(Ok(immutable.Parameter(rewritten, program->Binding(3),
                                       "immutable_output")));
    EXPECT_FALSE(
        Ok(immutable.EmitExisting(Id::RmsNorm, graph.invocations()[0])));
    NativeEmissionSession session(registry, graph, *module, *program);
    ASSERT_TRUE(Ok(session.Bind(first, program->Binding(0))));
    ASSERT_TRUE(Ok(session.Bind(second, program->Binding(1))));
    ASSERT_TRUE(Ok(session.Bind(gamma, program->Binding(2))));
    if (boundary_output) {
      ASSERT_TRUE(Ok(session.Bind(rewritten, program->Binding(3))));
    }
    ASSERT_TRUE(Ok(session.EmitExisting(Id::RmsNorm, graph.invocations()[0])));
    Value written_once;
    ASSERT_TRUE(Ok(session.Native(rewritten, &written_once)));
    ASSERT_TRUE(Ok(session.EmitExisting(Id::RmsNorm, graph.invocations()[1])));
    Value written_twice;
    ASSERT_TRUE(Ok(session.Native(rewritten, &written_twice)));
    ASSERT_TRUE(Ok(session.EmitExisting(Id::RmsNorm, graph.invocations()[2])));
    Value result;
    ASSERT_TRUE(Ok(session.Native(consumed, &result)));
    ASSERT_TRUE(Ok(program->InspectNative(
        {written_once, written_twice, result, program->Binding(3)},
        [&](const loom_module_t *native,
            const std::vector<loom_value_id_t> &ids) {
          EXPECT_EQ(ids[0], ids[1]);
          EXPECT_NE(ids[0], ids[2]);
          if (boundary_output) {
            EXPECT_EQ(ids[0], ids[3]);
          }
          size_t launches = 0;
          for (size_t i = 0; i < native->symbols.count; ++i) {
            auto *def = native->symbols.entries[i].defining_op;
            if (!def || !loom_command_program_def_isa(def))
              continue;
            auto *body = loom_region_const_entry_block(
                loom_command_program_def_body(def));
            auto *block = loom_region_const_entry_block(
                loom_command_serial_body(body->first_op));
            for (auto *op = block->first_op; op; op = op->next_op) {
              if (!loom_kernel_launch_isa(op))
                continue;
              auto args = loom_kernel_launch_arguments(op);
              if (launches < 2) {
                EXPECT_EQ(args.values[3], ids[0]);
              } else {
                EXPECT_EQ(args.values[1], ids[0]);
                EXPECT_EQ(args.values[3], ids[2]);
              }
              ++launches;
            }
          }
          EXPECT_EQ(launches, 3u);
          return iree_ok_status();
        })));
    EXPECT_EQ(graph.values().size(), 5u);
    EXPECT_EQ(graph.invocations().size(), 3u);
    EXPECT_EQ(graph.invocations()[0].binding(Port::Output)->value(), rewritten);
    EXPECT_EQ(graph.invocations()[1].binding(Port::Output)->value(), rewritten);
    EXPECT_EQ(graph.invocations()[2].binding(Port::Input)->value(), rewritten);
    ASSERT_TRUE(Ok(program->Finish()));
    loomc_module_t *linked = nullptr;
    loomc_result_t *result_module = nullptr;
    ASSERT_TRUE(
        Ok(module->LinkAndVerify({program->Ref()}, &linked, &result_module)));
    ASSERT_TRUE(result_module && loomc_result_succeeded(result_module))
        << Diagnostics(result_module);
    loomc_result_release(result_module);
    Save(linked, boundary_output ? "existing-boundary-rewrite"
                                 : "existing-generated-rewrite");
    Compile(context, workspace, environment, linked, "existing_rewrite");
    loomc_module_release(linked);
  }
}

TEST_F(InvocationTest,
       ProviderF32EpsilonAdmissionPreservesOriginalLogicalAttribute) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("f32_epsilon", {}, 2, &program)));
  OperationRegistry ops;
  std::string error;
  ASSERT_TRUE(DefineRmsNorm(&ops, &error));
  OperationGraph graph(ops);
  auto x = graph.tensor(TensorType::matrix(1, 3072, ElementType::F32), &error);
  auto gamma = graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
  InvocationProgram author(registry, graph, *module, *program);
  ASSERT_TRUE(Ok(author.Bind(x, program->Binding(0))));
  ASSERT_TRUE(Ok(author.Bind(gamma, program->Binding(1))));
  ValueId output;
  EXPECT_FALSE(
      Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072, 1e-4},
                       {{Port::Input, x}, {Port::Gamma, gamma}}, &output)));
  EXPECT_EQ(graph.values().size(), 2u);
  EXPECT_TRUE(graph.invocations().empty());
  const double gguf_f32_epsilon = static_cast<double>(1e-5f);
  ASSERT_NE(gguf_f32_epsilon, 1e-5);
  ASSERT_TRUE(
      Ok(author.Invoke(Id::RmsNorm, RmsNormAttrs{3072, gguf_f32_epsilon},
                       {{Port::Input, x}, {Port::Gamma, gamma}}, &output)));
  ASSERT_EQ(graph.invocations().size(), 1u);
  EXPECT_EQ(std::get<RmsNormAttrs>(graph.invocations()[0].attributes()).epsilon,
            gguf_f32_epsilon);
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *linked = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  Save(linked, "f32-epsilon");
  Compile(context, workspace, environment, linked, "f32_epsilon");
  loomc_module_release(linked);
}

} // namespace
} // namespace hrx::dispatch
