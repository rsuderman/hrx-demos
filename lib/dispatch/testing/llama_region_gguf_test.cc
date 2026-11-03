// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/llama_region.h"
#include "lib/dispatch/testing/ffn_fixture.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/template/ops.h"
#include "loomc/compile.h"
#include "loomc/interop.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "tools/model/gguf_catalog.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
::testing::AssertionResult Ok(loomc_status_t status) {
  auto message = loomc_status_message(status);
  std::string copy(message.data ? message.data : "", message.size);
  bool ok = loomc_status_is_ok(status);
  loomc_status_free(status);
  return ok ? ::testing::AssertionSuccess()
            : ::testing::AssertionFailure() << copy;
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  if (!result)
    return "no result";
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->message.data, d->message.size);
    text += '\n';
  }
  return text;
}
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

int64_t IndexValue(const loom_module_t *module, loom_value_id_t value) {
  const auto *def = loom_value_def_op(loom_module_value(module, value));
  EXPECT_TRUE(def && loom_index_constant_isa(def));
  if (!def || !loom_index_constant_isa(def))
    return -1;
  return loom_attr_as_i64(loom_index_constant_value(def));
}
int64_t ScalarValue(const loom_module_t *module, loom_value_id_t value) {
  const auto *def = loom_value_def_op(loom_module_value(module, value));
  EXPECT_TRUE(def && loom_scalar_constant_isa(def));
  if (!def || !loom_scalar_constant_isa(def))
    return -1;
  return loom_attr_as_i64(loom_scalar_constant_value(def));
}
class RealFfnTest : public ::testing::Test {
protected:
  void SetUp() override {
    // This manual target requires an explicit immutable external input. It
    // never downloads/substitutes weights or silently skips its real gate.
    const char *path = std::getenv("LOOMIFY_TEST_GGUF");
    ASSERT_NE(path, nullptr) << "set LOOMIFY_TEST_GGUF to the known Llama GGUF";
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                    &std::fclose);
    ASSERT_TRUE(file) << path;
    ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
    model::GgufCatalog catalog;
    std::string error;
    ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
    ASSERT_TRUE(model::LoadModelDescriptor(catalog, &facts, &error)) << error;
    ASSERT_EQ(facts.family, "llama");
    ASSERT_EQ(facts.embedding_width, 3072u);
    ASSERT_FALSE(facts.layers.empty());
    ASSERT_EQ(facts.layers[0].feed_forward_width, 8192u);
    ASSERT_EQ(static_cast<float>(facts.rms_epsilon), 1e-5f);
    EXPECT_LT(catalog.bytes_read, catalog.file_size);
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(),
                                      &context)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                        &workspace)));
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
           std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    if (workspace)
      loomc_workspace_release(workspace);
    if (context)
      loomc_context_release(context);
    if (environment)
      loomc_target_environment_release(environment);
  }
  model::ModelDescriptor facts;
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};

TEST_F(RealFfnTest, OwnedGgufRegionEmitsPrefill3Decode4AndCompilesSourceBytecode) {
  for (auto phase : {Phase::Prefill, Phase::Decode}) {
    SCOPED_TRACE(phase == Phase::Decode ? "decode" : "prefill");
    std::string error;
    auto owner = testing::FfnFixture(facts, phase, 1, &error);
    ASSERT_TRUE(owner) << error;
    const auto values = owner->graph().values().size();
    const auto operations = owner->graph().invocations().size();
    auto registry = Registry::Llama3B();
    std::unique_ptr<FfnRegion> region;
    ASSERT_TRUE(FfnRegion::Select(owner, {llama::LlamaRegionKind::Ffn, 0, 1},
                                  registry, &region, &error)) << error;
    // Original names, file offsets and representation survive ownership and
    // region selection; reflection/root placement is a separate later proof.
    for (const auto &association : owner->parameters()) {
      const auto *parameter = owner->parameter(association.coordinate);
      ASSERT_NE(parameter, nullptr);
      const auto &original = facts.layers[association.coordinate.layer_slot]
                                 .tensors[association.coordinate.tensor_slot];
      EXPECT_EQ(parameter->tensor.name, original.tensor.name);
      EXPECT_EQ(parameter->tensor.absolute_offset,
                original.tensor.absolute_offset);
      EXPECT_EQ(parameter->tensor.byte_size, original.tensor.byte_size);
      EXPECT_EQ(parameter->tensor.storage_type, original.tensor.storage_type);
    }
    std::vector<Registration> registrations;
    for (auto id : region->dispatches())
      registrations.push_back(*registry.Find(id));
    loomc_module_t *definitions = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(Load(context, workspace, registrations, root, &definitions,
                        &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    std::unique_ptr<ModuleBuilder> module;
    ASSERT_TRUE(Ok(ModuleBuilder::Create(context, workspace, definitions,
                                         &module)));
    loomc_module_release(definitions);
    std::unique_ptr<CommandProgramBuilder> program;
    const bool decode = phase == Phase::Decode;
    const std::string name = decode ? "gguf_ffn_decode" : "gguf_ffn_prefill";
    ASSERT_TRUE(Ok(module->DefineProgram(name, {}, decode ? 2 : 3, &program)));
    auto foreign_owner = testing::FfnFixture(facts, phase, 1, &error);
    ASSERT_TRUE(foreign_owner) << error;
    NativeEmissionSession foreign(registry, foreign_owner->graph(), *module,
                                  *program);
    EXPECT_FALSE(region->EmitOneRow(foreign, program->Binding(1), &error));
    EXPECT_FALSE(foreign.one_row_specialized());
    NativeEmissionSession session(registry, owner->graph(), *module, *program);
    ASSERT_TRUE(Ok(session.Bind(owner->inputs().residual, program->Binding(0))));
    if (!decode)
      ASSERT_TRUE(Ok(session.Control(program->Binding(2))));
    ASSERT_TRUE(region->EmitOneRow(session, program->Binding(1), &error))
        << error;
    EXPECT_TRUE(session.one_row_specialized());
    EXPECT_EQ(owner->graph().values().size(), values);
    EXPECT_EQ(owner->graph().invocations().size(), operations);
    const auto &ops = owner->graph().invocations();
    const auto normalized = ops[0].binding(Port::Output)->value();
    const auto activation = ops[1].binding(Port::Output)->value();
    Value n, a, residual;
    ASSERT_TRUE(Ok(session.Native(normalized, &n)));
    ASSERT_TRUE(Ok(session.Native(activation, &a)));
    ASSERT_TRUE(Ok(session.Native(owner->inputs().residual, &residual)));
    const std::vector<std::string> expected = decode
        ? std::vector<std::string>{registrations[0].symbol,
              "ggml_quantize_q8_1_x4_qk_compat_f32", registrations[1].symbol,
              registrations[2].symbol}
        : std::vector<std::string>{registrations[0].symbol,
              registrations[1].symbol, registrations[2].symbol};
    ASSERT_TRUE(Ok(program->InspectNative(
        {program->Binding(0), n, a, residual, program->Binding(1)},
        [&](const loom_module_t *native, const std::vector<loom_value_id_t> &ids) {
          EXPECT_EQ(ids[0], ids[3]);
          EXPECT_NE(ids[0], ids[1]);
          EXPECT_NE(ids[0], ids[2]);
          EXPECT_NE(ids[1], ids[2]);
          std::vector<std::string> calls;
          size_t parameters = 0;
          for (size_t i = 0; i < native->symbols.count; ++i) {
            const auto *def = native->symbols.entries[i].defining_op;
            if (!def || !loom_command_program_def_isa(def))
              continue;
            const auto *body = loom_region_const_entry_block(
                loom_command_program_def_body(def));
            const auto *serial = loom_region_const_entry_block(
                loom_command_serial_body(body->first_op));
            for (auto *op = serial->first_op; op; op = op->next_op) {
              if (loom_command_parameter_isa(op)) {
                ++parameters;
                EXPECT_EQ(loom_command_parameter_source(op), ids[4]);
                const auto key = loom_string_table_get(&native->strings,
                    loom_command_parameter_pattern(op));
                const std::string name(key.data, key.size);
                const model::TensorBinding *binding = nullptr;
                for (const auto &association : owner->parameters()) {
                  auto *candidate = owner->parameter(association.coordinate);
                  if (candidate->tensor.name == name)
                    binding = candidate;
                }
                EXPECT_NE(binding, nullptr);
                if (binding) {
                  const auto type = loom_module_value_type(
                      native, loom_command_parameter_result(op));
                  EXPECT_TRUE(loom_type_is_view(type));
                  EXPECT_EQ(loom_type_rank(type), 1);
                  const bool gamma = binding->role == "post_norm";
                  EXPECT_EQ(loom_type_element_type(type),
                            gamma ? LOOM_SCALAR_TYPE_F32 : LOOM_SCALAR_TYPE_I8);
                  const int64_t elements = loom_type_dim_is_dynamic_at(type, 0)
                      ? IndexValue(native, loom_type_dim_value_id_at(type, 0))
                      : loom_type_dim_static_size_at(type, 0);
                  EXPECT_EQ(elements, gamma ? 3072 : binding->tensor.byte_size);
                }
              }
              loom_symbol_ref_t callee{};
              if (loom_kernel_launch_isa(op))
                callee = loom_kernel_launch_callee(op);
              else if (loom_template_apply_isa(op)) {
                const auto operands = loom_template_apply_operands(op);
                EXPECT_EQ(operands.count,
                          decode ? 12u : calls.size() == 1 ? 22u : 20u);
                if (operands.count >= 2) {
                  EXPECT_EQ(IndexValue(native, operands.values[0]), 3072);
                  EXPECT_EQ(IndexValue(native, operands.values[1]), 8192);
                  EXPECT_EQ(ScalarValue(native, operands.values[2]), 1);
                  EXPECT_EQ(ScalarValue(native, operands.values[3]), 1);
                  const bool down = calls.size() == expected.size() - 1;
                  bool q6 = false;
                  if (down)
                    for (const auto &p : facts.layers[0].tensors)
                      if (p.role == "ffn_down")
                        q6 = p.tensor.storage_type == 14;
                  EXPECT_EQ(ScalarValue(native, operands.values[4]), q6 ? 0 : 1);
                  EXPECT_EQ(ScalarValue(native, operands.values[5]), q6 ? 1 : 0);
                }
                callee = loom_template_apply_family(op);
              }
              else
                continue;
              auto symbol = native->symbols.entries[callee.symbol_id];
              auto text = loom_string_table_get(&native->strings, symbol.name_id);
              calls.emplace_back(text.data, text.size);
            }
          }
          EXPECT_EQ(calls, expected);
          EXPECT_EQ(parameters, 4u);
          return iree_ok_status();
        })));
    ASSERT_TRUE(Ok(program->Finish()));
    loomc_module_t *linked = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(module->LinkAndVerify({program->Ref()}, &linked, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options{};
      options.format = format;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          linked, &options, loomc_allocator_system(), &source)));
      if (const char *directory = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
        const auto path = std::string(directory) + "/" + name +
            (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc");
        ASSERT_TRUE(Ok(loomc_module_serialize_to_path(
            linked, &options, loomc_make_cstring_view(path.c_str()),
            loomc_allocator_system())));
      }
      if (format == LOOMC_SOURCE_FORMAT_TEXT) {
        const auto bytes = loomc_source_contents(source);
        const std::string text(reinterpret_cast<const char *>(bytes.data),
                               bytes.data_length);
        for (const auto &association : owner->parameters())
          EXPECT_NE(text.find(owner->parameter(association.coordinate)
                                  ->tensor.name), std::string::npos);
        EXPECT_NE(text.find("f32"), std::string::npos);
        EXPECT_NE(text.find("i8"), std::string::npos);
      }
      loomc_module_t *roundtrip = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context, workspace, source, nullptr, loomc_allocator_system(),
          &roundtrip, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      loomc_result_release(result);
      loomc_source_release(source);
      Compile(context, workspace, environment, roundtrip, name.c_str());
      loomc_module_release(roundtrip);
    }
    loomc_module_release(linked);
  }
}
} // namespace
} // namespace hrx::dispatch
