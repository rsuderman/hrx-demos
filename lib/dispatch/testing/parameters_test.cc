// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/parameters.h"
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
#include <array>
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
class ParameterTest : public ::testing::Test {
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

model::TensorBinding Weight(std::string role, std::string key, uint32_t storage,
                            int64_t in, int64_t out, uint64_t bytes) {
  model::TensorBinding binding;
  binding.role = std::move(role);
  binding.tensor.name = std::move(key);
  binding.tensor.storage_type = storage;
  binding.tensor.normalized_extents = {in, out, 1, 1};
  binding.tensor.rank = out == 1 ? 1 : 2;
  binding.tensor.absolute_offset = 4096;
  binding.tensor.byte_size = bytes;
  return binding;
}
model::ModelDescriptor Facts() {
  model::ModelDescriptor facts;
  facts.family = "llama";
  facts.embedding_width = 3072;
  facts.vocabulary_size = 128256;
  facts.key_head_width = facts.value_head_width = 128;
  facts.rms_epsilon = 1e-5;
  model::LayerDescriptor layer;
  layer.index = 7; // Actual descriptor index differs from vector slot zero.
  layer.query_heads = 24;
  layer.kv_heads = 8;
  layer.feed_forward_width = 8192;
  const uint64_t blocks = 3072ull * 8192 / 256;
  layer.tensors = {
      Weight("post_norm", "blk.7.ffn_norm.weight", 0, 3072, 1, 3072 * 4),
      Weight("ffn_gate", "blk.7.ffn_gate.weight", 12, 3072, 8192, blocks * 144),
      Weight("ffn_up", "blk.7.ffn_up.weight", 12, 3072, 8192, blocks * 144),
      Weight("ffn_down", "blk.7.ffn_down.weight", 14, 8192, 3072,
             blocks * 210)};
  facts.layers.push_back(std::move(layer));
  return facts;
}
std::unique_ptr<llama::detail::StructureConstruction>
Construct(model::ModelDescriptor facts, OperationRegistry &ops,
          llama::detail::NormalizedFfnRegion *region, std::string *error) {
  if (!DefineRmsNorm(&ops, error) || !DefineGatedMatmul(&ops, error) ||
      !DefineLinearResidual(&ops, error))
    return nullptr;
  llama::LlamaBuildOptions options{Phase::Decode, 1, 32, 32, "checkpoint"};
  auto construction = llama::detail::StructureConstruction::Create(
      std::move(facts), options, ops, error);
  if (!construction || !construction->createInputs(error))
    return nullptr;
  std::array<ValueId, 4> weights;
  for (size_t i = 0; i < weights.size(); ++i)
    if (!construction->resolve({llama::ParameterScope::Layer, 0, i},
                               &weights[i], error))
      return nullptr;
  const auto &owner = construction->structure();
  if (!llama::detail::AppendNormalizedFfnResolved(
          &construction->graph(), owner.facts(), 0, owner.inputs().residual,
          weights, owner.inputs().row_begin, owner.inputs().row_count, region,
          error))
    return nullptr;
  return construction;
}

TEST_F(ParameterTest, OwnerKeysBindNamedGammaAndEncodedWholeFfn) {
  for (bool tied : {false, true}) {
    SCOPED_TRACE(tied);
    auto registry = Registry::Llama3B();
    std::vector<Registration> entries{*registry.Find(Id::RmsNorm),
                                      *registry.Find(Id::GateUpDecode),
                                      *registry.Find(Id::DownDecode)};
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
    ASSERT_TRUE(Ok(module->DefineProgram("bound_ffn", {}, 2, &program)));
    OperationRegistry ops;
    llama::detail::NormalizedFfnRegion region;
    std::string error;
    auto facts = Facts();
    if (tied)
      facts.layers[0].tensors[2].tensor = facts.layers[0].tensors[1].tensor;
    auto construction = Construct(std::move(facts), ops, &region, &error);
    ASSERT_TRUE(construction) << error;
    const auto &owner = construction->structure();
    NativeEmissionSession session(registry, owner.graph(), *module, *program);
    ASSERT_TRUE(Ok(session.Bind(owner.inputs().residual, program->Binding(1))));
    ASSERT_TRUE(Ok(session.SpecializeOneRow(owner.inputs().row_begin,
                                            owner.inputs().row_count)));
    const llama::LlamaRegionView view{
        region.operation_begin, region.operation_end, {}};
    ASSERT_TRUE(
        BindRegionParameters(owner, view, program->Binding(0), session, &error))
        << error;
    const size_t operation_count = owner.graph().invocations().size();
    const Id selected[] = {Id::RmsNorm, Id::GateUpDecode, Id::DownDecode};
    for (size_t i = 0; i < 3; ++i)
      ASSERT_TRUE(Ok(session.EmitExisting(
          selected[i],
          owner.graph().invocations()[region.operation_begin + i])));
    EXPECT_EQ(owner.graph().invocations().size(), operation_count);
    Value native;
    ASSERT_TRUE(Ok(session.Native(region.residual, &native)));
    ASSERT_TRUE(
        Ok(program->InspectNative({}, [](const loom_module_t *module,
                                         const std::vector<loom_value_id_t> &) {
          bool wired_gamma = false;
          for (size_t i = 0; i < module->symbols.count; ++i) {
            const auto *def = module->symbols.entries[i].defining_op;
            if (!def || !loom_command_program_def_isa(def))
              continue;
            const auto *body = loom_region_const_entry_block(
                loom_command_program_def_body(def));
            const auto *block = loom_region_const_entry_block(
                loom_command_serial_body(body->first_op));
            loom_value_id_t gamma = LOOM_VALUE_ID_INVALID;
            for (const auto *op = block->first_op; op; op = op->next_op) {
              if (loom_command_parameter_isa(op)) {
                const auto key = loom_string_table_get(
                    &module->strings, loom_command_parameter_pattern(op));
                if (std::string(key.data, key.size) ==
                    "blk.7.ffn_norm.weight") {
                  gamma = loom_command_parameter_result(op);
                  const auto type = loom_module_value_type(module, gamma);
                  EXPECT_EQ(loom_type_element_type(type), LOOM_SCALAR_TYPE_F32);
                  EXPECT_EQ(loom_type_rank(type), 1);
                }
              }
              if (loom_kernel_launch_isa(op) &&
                  gamma != LOOM_VALUE_ID_INVALID) {
                const auto args = loom_kernel_launch_arguments(op);
                if (args.count == 4 && args.values[2] == gamma)
                  wired_gamma = true;
              }
            }
          }
          EXPECT_TRUE(wired_gamma);
          return iree_ok_status();
        })));
    ASSERT_TRUE(Ok(program->Finish()));
    loomc_module_t *linked = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    std::string text;
    ASSERT_TRUE(Ok(Dump(linked, &text)));
    for (const auto &binding : owner.facts().layers[0].tensors)
      EXPECT_NE(text.find(binding.tensor.name), std::string::npos);
    EXPECT_EQ(text.find("blk.0."), std::string::npos);
    EXPECT_NE(text.find("f32"), std::string::npos);
    EXPECT_NE(text.find("i8"), std::string::npos);
    Save(linked, "bound_ffn");
    Compile(context, workspace, environment, linked, "bound_ffn");
    loomc_module_release(linked);
  }
}

TEST_F(ParameterTest, RejectsMismatchedEncodedSpanBeforeRegistration) {
  for (bool invalid_range : {false, true}) {
    auto registry = Registry::Llama3B();
    auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
    ASSERT_NE(definitions, nullptr);
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(
        Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
    loomc_module_release(definitions);
    std::unique_ptr<CommandProgramBuilder> program;
    ASSERT_TRUE(Ok(module->DefineProgram("invalid", {}, 1, &program)));
    auto facts = Facts();
    if (!invalid_range)
      --facts.layers[0].tensors[3].tensor.byte_size;
    OperationRegistry ops;
    llama::detail::NormalizedFfnRegion region;
    std::string error;
    auto construction = Construct(std::move(facts), ops, &region, &error);
    ASSERT_TRUE(construction) << error;
    const auto &owner = construction->structure();
    NativeEmissionSession session(registry, owner.graph(), *module, *program);
    llama::LlamaRegionView view{
        region.operation_begin, region.operation_end, {}};
    if (invalid_range)
      ++view.operation_end;
    EXPECT_FALSE(BindRegionParameters(owner, view, program->Binding(0), session,
                                      &error));
    EXPECT_FALSE(error.empty());
    // Semantic preflight failed before even the first norm registration.
    ASSERT_TRUE(Ok(session.Parameter(owner.parameters()[0].value,
                                     program->Binding(0), "still_unbound")));
  }
}
TEST_F(ParameterTest, RejectsForeignOwnerBeforeParameterRegistration) {
  auto registry = Registry::Llama3B();
  auto *definitions = LoadEntry(*registry.Find(Id::RmsNorm));
  ASSERT_NE(definitions, nullptr);
  std::unique_ptr<ModuleBuilder> module;
  ASSERT_TRUE(
      Ok(ModuleBuilder::Create(context, workspace, definitions, &module)));
  loomc_module_release(definitions);
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(module->DefineProgram("foreign", {}, 1, &program)));
  OperationRegistry ops;
  llama::detail::NormalizedFfnRegion region;
  std::string error;
  auto construction = Construct(Facts(), ops, &region, &error);
  ASSERT_TRUE(construction) << error;
  OperationGraph other(ops);
  auto gamma = other.tensor(TensorType::vector(3072, ElementType::F32), &error);
  NativeEmissionSession session(registry, other, *module, *program);
  EXPECT_FALSE(
      BindRegionParameters(construction->structure(),
                           {region.operation_begin, region.operation_end, {}},
                           program->Binding(0), session, &error));
  EXPECT_NE(error.find("graph owner"), std::string::npos);
  ASSERT_TRUE(Ok(session.Parameter(gamma, program->Binding(0), "unbound")));
}
} // namespace
} // namespace hrx::dispatch
