// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/configuration.h"
#include "tools/loomify-gguf/testing/configuration_fixture.h"
#include "tools/model/qwen_configuration.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ir/context.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loomc/compile.h"
#include "loomc/interop.h"
#include "loomc/link.h"
#include "loomc/pass.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "gtest/gtest.h"
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace hrx::loomify_gguf {
namespace {
using Module = std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)>;
::testing::AssertionResult Ok(loomc_status_t status) {
  const auto view = loomc_status_message(status);
  std::string message(view.data ? view.data : "", view.size);
  const bool ok = loomc_status_is_ok(status);
  loomc_status_free(status);
  return ok ? ::testing::AssertionSuccess() : ::testing::AssertionFailure() << message;
}
std::string Diagnostics(const loomc_result_t *result) {
  std::string text;
  for (size_t i = 0; result && i < loomc_result_diagnostic_count(result); ++i) {
    const auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->message.data, d->message.size);
    text += '\n';
  }
  return text;
}
std::string Text(const loom_module_t *module, loom_string_id_t id) {
  const auto view = loom_string_table_get(&module->strings, id);
  return std::string(view.data, view.size);
}
const loom_op_t *Definition(const loom_module_t *module, const char *name) {
  for (size_t i = 0; i < module->symbols.count; ++i) {
    const auto &entry = module->symbols.entries[i];
    if (entry.name_id != LOOM_STRING_ID_INVALID && Text(module, entry.name_id) == name)
      return entry.defining_op;
  }
  return nullptr;
}
int64_t Constant(const loom_module_t *module, loom_value_id_t value) {
  const auto *op = loom_value_def_op(loom_module_value(module, value));
  EXPECT_TRUE(op && loom_index_constant_isa(op));
  if (!op || !loom_index_constant_isa(op)) return -1;
  const auto attr = loom_index_constant_value(op);
  EXPECT_EQ(attr.kind, LOOM_ATTR_I64);
  return attr.i64;
}
void Walk(const loom_region_t *region, const std::function<void(const loom_op_t *)> &visit) {
  for (size_t b = 0; b < region->block_count; ++b)
    for (const auto *op = region->blocks[b]->first_op; op; op = op->next_op) {
      visit(op);
      for (size_t i = 0; i < op->region_count; ++i) Walk(loom_op_regions(op)[i], visit);
    }
}
class QwenModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(), &environment)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
    ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
    root = std::string(std::getenv("TEST_SRCDIR")) + "/" + std::getenv("TEST_WORKSPACE");
  }
  void TearDown() override {
    if (workspace) loomc_workspace_release(workspace);
    if (context) loomc_context_release(context);
    if (environment) loomc_target_environment_release(environment);
  }
  Module Parse(const std::string &text) {
    loomc_source_options_t options{};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier = loomc_make_cstring_view("qwen-test-config.loom");
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.contents = loomc_make_byte_span(text.data(), text.size());
    loomc_source_t *source = nullptr;
    EXPECT_TRUE(Ok(loomc_source_create(&options, loomc_allocator_system(), &source)));
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(context, workspace, source, nullptr,
        loomc_allocator_system(), &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    if (result) loomc_result_release(result);
    if (source) loomc_source_release(source);
    return Module(module, loomc_module_release);
  }
  Module Load(const std::string &path) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_path(context, workspace,
        loomc_make_cstring_view(path.c_str()), nullptr, loomc_allocator_system(), &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    if (result) loomc_result_release(result);
    return Module(module, loomc_module_release);
  }
  Module Link(loomc_module_t *config, loomc_module_t *implementation, bool succeeds = true) {
    loomc_linker_t *linker = nullptr;
    EXPECT_TRUE(Ok(loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker)));
    loomc_link_module_provider_t providers[] = {
        {config, loomc_make_cstring_view("configuration"), LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {implementation, loomc_make_cstring_view("implementation"), LOOMC_LINK_PROVIDER_ROLE_INPUT}};
    const auto symbol = loomc_make_cstring_view("generated.qwen_epoch");
    loomc_link_options_t options{};
    options.mode = LOOMC_LINK_MODE_LINK;
    options.module_providers = providers;
    options.module_provider_count = 2;
    options.root_symbols = &symbol;
    options.root_symbol_count = 1;
    options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status = loomc_link_module(linker, workspace, &options, &module, &result);
    if (succeeds) {
      EXPECT_TRUE(Ok(status));
      EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      EXPECT_NE(module, nullptr);
    } else {
      const bool failed = !loomc_status_is_ok(status) || !result || !loomc_result_succeeded(result);
      loomc_status_free(status);
      EXPECT_TRUE(failed) << "missing live dependency unexpectedly linked";
      EXPECT_EQ(module, nullptr);
      EXPECT_FALSE(Diagnostics(result).empty());
    }
    if (result) loomc_result_release(result);
    if (linker) loomc_linker_release(linker);
    return Module(module, loomc_module_release);
  }
  void Pass(loomc_module_t *module, const char *pipeline) {
    loomc_pass_program_t *passes = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(context,
        loomc_make_cstring_view(pipeline), nullptr, loomc_allocator_system(), &passes, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_run(passes, workspace, module, nullptr,
        loomc_allocator_system(), &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    if (result) loomc_result_release(result);
    loomc_pass_program_release(passes);
  }
  void InspectCalls(loomc_module_t *module, const model::ModelDescriptor &facts, int64_t tokens) {
    loomc_module_interop_view_t view{};
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(module, loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    const auto *def = Definition(view.module, "generated.qwen_epoch");
    ASSERT_TRUE(def && loom_command_program_def_isa(def));
    const auto *body = loom_region_const_entry_block(loom_command_program_def_body(def));
    ASSERT_EQ(body->arg_count, 7u);
    std::vector<const loom_op_t *> calls, kernels;
    Walk(loom_command_program_def_body(def), [&](const loom_op_t *op) {
      if (loom_command_program_launch_isa(op)) calls.push_back(op);
      if (loom_kernel_launch_isa(op)) kernels.push_back(op);
    });
    ASSERT_EQ(kernels.size(), 1u);
    const auto *embedding = kernels.front();
    EXPECT_EQ(Text(view.module, view.module->symbols.entries[loom_kernel_launch_callee(embedding).symbol_id].name_id),
              "qwen38_token_embedding_q6k_f16_prefill");
    auto workloads = loom_kernel_launch_workloads(embedding);
    ASSERT_EQ(workloads.count, 1u);
    EXPECT_EQ(Constant(view.module, workloads.values[0]), tokens);
    auto args = loom_kernel_launch_arguments(embedding);
    ASSERT_EQ(args.count, 3u);
    EXPECT_EQ(args.values[0], body->arg_ids[5]);
    EXPECT_EQ(args.values[2], body->arg_ids[1]);
    const auto *parameter = loom_value_def_op(loom_module_value(view.module, args.values[1]));
    ASSERT_TRUE(parameter && loom_command_parameter_isa(parameter));
    EXPECT_EQ(Text(view.module, loom_command_parameter_pattern(parameter)), "token_embd.weight");
    EXPECT_EQ(loom_command_parameter_source(parameter), body->arg_ids[0]);
    // Native unrolling preserves both conditional IR branches until callable
    // inlining. These are 24 routing sites, not 48 executed layers.
    std::vector<const loom_op_t *> routes, branches;
    Walk(loom_command_program_def_body(def), [&](const loom_op_t *op) {
      const auto name = loom_op_vtable_name(loom_op_vtable(view.module, op));
      if (std::string(name.data, name.size) == "func.call") routes.push_back(op);
      if (loom_scf_if_isa(op)) branches.push_back(op);
    });
    ASSERT_EQ(routes.size(), facts.layers.size());
    ASSERT_EQ(branches.size(), facts.layers.size());
    ASSERT_EQ(calls.size(), 2 * branches.size() + 1);
    loom_value_id_t gdn_scratch = LOOM_VALUE_ID_INVALID, attention_scratch = LOOM_VALUE_ID_INVALID;
    size_t selected_gdn = 0, selected_attention = 0;
    for (size_t i = 0; i < facts.layers.size(); ++i) {
      SCOPED_TRACE(i);
      const auto *route = routes[i];
      ASSERT_GE(route->attribute_count, 1u);
      const auto callee = loom_op_const_attrs(route)[0];
      ASSERT_EQ(callee.kind, LOOM_ATTR_SYMBOL);
      EXPECT_EQ(Text(view.module, view.module->symbols.entries[callee.symbol.symbol_id].name_id),
                "loomify.config.layer_route");
      ASSERT_EQ(route->operand_count, 1u);
      EXPECT_EQ(Constant(view.module, loom_op_const_operands(route)[0]), int64_t(i));
      ASSERT_EQ(route->result_count, 3u);
      const auto *tuple = loom_op_const_results(route);
      const auto *condition = loom_value_def_op(loom_module_value(view.module, loom_scf_if_condition(branches[i])));
      ASSERT_TRUE(condition && loom_index_cmp_isa(condition));
      EXPECT_EQ(loom_index_cmp_predicate(condition), LOOM_INDEX_CMP_PREDICATE_EQ);
      EXPECT_EQ(loom_index_cmp_lhs(condition), tuple[0]);
      EXPECT_EQ(Constant(view.module, loom_index_cmp_rhs(condition)), 0);
      for (size_t arm = 0; arm < 2; ++arm) {
        const bool is_gdn = arm == 1;
        const auto *region = is_gdn ? loom_scf_if_else_region(branches[i]) : loom_scf_if_then_region(branches[i]);
        ASSERT_NE(region, nullptr);
        std::vector<const loom_op_t *> arm_calls;
        Walk(region, [&](const loom_op_t *op) {
          if (loom_command_program_launch_isa(op)) arm_calls.push_back(op);
        });
        ASSERT_EQ(arm_calls.size(), 1u); // No appended FFN or hidden model call.
        const auto *op = arm_calls.front();
        EXPECT_EQ(calls[2 * i + arm], op);
        const auto callee = loom_command_program_launch_callee(op);
        EXPECT_EQ(Text(view.module, view.module->symbols.entries[callee.symbol_id].name_id),
                  is_gdn ? "qwen38_gdn_layer_prefill" : "qwen38_attention_layer_prefill");
        const auto specs = loom_command_program_launch_specializations(op);
        ASSERT_EQ(specs.count, 6u);
        EXPECT_EQ(Constant(view.module, specs.values[0]), tokens);
        EXPECT_EQ(Constant(view.module, specs.values[1]), tokens);
        EXPECT_EQ(specs.values[2], tuple[1]);
        EXPECT_EQ(specs.values[3], tuple[2]);
        EXPECT_EQ(Constant(view.module, specs.values[4]), 0);
        EXPECT_EQ(Constant(view.module, specs.values[5]), 1);
        const auto bindings = loom_command_program_launch_bindings(op);
        ASSERT_EQ(bindings.count, 6u);
        for (auto pair : {std::pair<size_t, size_t>{0, 0}, {1, 1}, {2, 4}, {3, 2}, {5, 3}})
          EXPECT_EQ(bindings.values[pair.first], body->arg_ids[pair.second]);
        auto &scratch = is_gdn ? gdn_scratch : attention_scratch;
        if (scratch == LOOM_VALUE_ID_INVALID) scratch = bindings.values[4];
        EXPECT_EQ(bindings.values[4], scratch);
        const auto *allocation = loom_value_def_op(loom_module_value(view.module, scratch));
        ASSERT_TRUE(allocation && loom_buffer_alloca_isa(allocation));
        EXPECT_EQ(Constant(view.module, loom_buffer_alloca_byte_length(allocation)), is_gdn ? 256 : 4194304);
        EXPECT_EQ(loom_buffer_alloca_base_alignment(allocation), 256);
      }
      if (facts.layers[i].kind == model::LayerKind::kGatedDeltaNet) {
        ASSERT_TRUE(facts.layers[i].recurrent_ordinal);
        EXPECT_EQ(*facts.layers[i].recurrent_ordinal, selected_gdn++);
      } else {
        ASSERT_TRUE(facts.layers[i].attention_ordinal);
        EXPECT_EQ(*facts.layers[i].attention_ordinal, selected_attention++);
      }
    }
    EXPECT_EQ(selected_gdn, 18u);
    EXPECT_EQ(selected_attention, 6u);
    EXPECT_NE(gdn_scratch, attention_scratch);
    std::printf("Prepared Qwen hierarchy: 24 route sites, 24 attention arms, 24 GDN arms, one embedding and one head; configured selection 6 attention/18 GDN\n");
    const auto *head = calls.back();
    EXPECT_EQ(Text(view.module, view.module->symbols.entries[loom_command_program_launch_callee(head).symbol_id].name_id), "qwen08_epoch_head");
    const auto specs = loom_command_program_launch_specializations(head);
    ASSERT_EQ(specs.count, 2u);
    EXPECT_EQ(Constant(view.module, specs.values[0]), tokens);
    EXPECT_EQ(Constant(view.module, specs.values[1]), 1);
    const auto bindings = loom_command_program_launch_bindings(head);
    ASSERT_EQ(bindings.count, 4u);
    const size_t expected[] = {0, 1, 2, 6};
    for (size_t j = 0; j < 4; ++j) EXPECT_EQ(bindings.values[j], body->arg_ids[expected[j]]);
  }
  void InspectParameters(loomc_module_t *module, const model::ModelDescriptor &facts) {
    loomc_module_t *raw = nullptr;
    ASSERT_TRUE(Ok(loomc_module_clone(module, workspace, loomc_allocator_system(), &raw)));
    Module expanded(raw, loomc_module_release);
    Pass(expanded.get(), "inline-callables,canonicalize");
    loomc_module_interop_view_t view{};
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(expanded.get(), loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    const auto *def = Definition(view.module, "generated.qwen_epoch");
    ASSERT_TRUE(def && loom_command_program_def_isa(def));
    std::set<std::string> observed, expected;
    Walk(loom_command_program_def_body(def), [&](const loom_op_t *op) {
      if (!loom_command_parameter_isa(op)) return;
      auto key = Text(view.module, loom_command_parameter_pattern(op));
      const auto substitutions = loom_command_parameter_substitutions(op);
      for (size_t i = 0; i < substitutions.count; ++i) {
        const auto position = key.find("{}");
        EXPECT_NE(position, std::string::npos);
        if (position != std::string::npos)
          key.replace(position, 2, std::to_string(Constant(view.module, substitutions.values[i])));
      }
      observed.insert(key);
    });
    for (const auto &tensor : facts.globals) expected.insert(tensor.tensor.name);
    for (const auto &layer : facts.layers)
      for (const auto &tensor : layer.tensors) expected.insert(tensor.tensor.name);
    EXPECT_EQ(observed, expected); // Every actual parameter identity, including embedded FFNs.
    // After the native inliner folds route facts, one stateful marker remains
    // for each selected complete layer. Check order AND actual state ordinals.
    std::vector<const loom_op_t *> selected;
    Walk(loom_command_program_def_body(def), [&](const loom_op_t *op) {
      if (!loom_kernel_launch_isa(op)) return;
      const auto name = Text(view.module, view.module->symbols.entries[loom_kernel_launch_callee(op).symbol_id].name_id);
      if (name == "qwen38_gdn_conv_spans" || name == "qwen38_attention_kv_prepare_spans")
        selected.push_back(op);
    });
    ASSERT_EQ(selected.size(), facts.layers.size());
    const auto &ssm = *facts.ssm;
    const uint64_t recurrent_bytes = ssm.recurrent_state_elements * 4;
    const uint64_t history_bytes = ssm.convolution_history_elements * 4;
    for (size_t i = 0; i < facts.layers.size(); ++i) {
      const auto *op = selected[i];
      const bool gdn = facts.layers[i].kind == model::LayerKind::kGatedDeltaNet;
      EXPECT_EQ(Text(view.module, view.module->symbols.entries[loom_kernel_launch_callee(op).symbol_id].name_id),
                gdn ? "qwen38_gdn_conv_spans" : "qwen38_attention_kv_prepare_spans");
      const auto args = loom_kernel_launch_arguments(op);
      ASSERT_GE(args.count, 3u);
      const uint64_t origin = gdn
          ? *facts.layers[i].recurrent_ordinal * (recurrent_bytes + history_bytes) + recurrent_bytes
          : *facts.layers[i].attention_ordinal * 2 * 512 * 1024;
      EXPECT_EQ(Constant(view.module, args.values[2]), int64_t(origin));
    }
  }
  void Compile(loomc_module_t *module) {
    loomc_target_profile_t *profile = nullptr;
    ASSERT_TRUE(Ok(loomc_target_profile_select(environment, loomc_make_cstring_view("amdgpu:gfx1100"),
        loomc_allocator_system(), &profile)));
    const auto symbol = loomc_make_cstring_view("generated.qwen_epoch");
    std::vector<loomc_request_t *> requests;
    loomc_cmd_program_module_options_t options{};
    options.root_symbols = &symbol;
    options.root_symbol_count = 1;
    options.kernel_target_profile = profile;
    options.request_sink.publish = [](void *user, loomc_request_t *request) {
      static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
      return loomc_ok_status();
    };
    options.request_sink.user_data = &requests;
    loomc_product_t *product = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_cmd_program_product_build_module(workspace, module, &options,
        loomc_allocator_system(), &product, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    if (product) loomc_product_release(product);
    ASSERT_FALSE(requests.empty());
    loomc_compiler_t *compiler = nullptr;
    ASSERT_TRUE(Ok(loomc_compiler_create(context, nullptr, loomc_allocator_system(), &compiler)));
    for (auto *request : requests) {
      loomc_module_t *raw = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(context, workspace, loomc_request_source(request),
          nullptr, loomc_allocator_system(), &raw, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      loomc_result_release(result);
      Module leaf(raw, loomc_module_release);
      result = nullptr;
      loomc_compile_artifact_options_t compile{};
      compile.target_profile = profile;
      ASSERT_TRUE(Ok(loomc_compile_artifact(compiler, workspace, nullptr, leaf.get(), &compile,
          loomc_allocator_system(), &result)));
      EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      EXPECT_GT(result ? loomc_result_artifact_count(result) : 0u, 0u);
      if (result) loomc_result_release(result);
      loomc_request_release(request);
    }
    std::printf("Full Qwen24-slot materialization and target compilation: %zu kernel requests\n", requests.size());
    loomc_compiler_release(compiler);
    loomc_target_profile_release(profile);
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_F(QwenModelTest, CompleteSequenceSurvivesBothProviderEncodings) {
  auto facts = testing::QwenModelFacts();
#ifdef LOOMIFY_QWEN_REAL_GGUF
  const char *asset = std::getenv("LOOMIFY_TEST_QWEN_GGUF");
  ASSERT_NE(asset, nullptr) << "real header fixture required; no checkpoint download/substitution";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(asset, "rb"), &std::fclose);
  ASSERT_TRUE(file);
  std::setvbuf(file.get(), nullptr, _IONBF, 0);
  std::string extraction_error;
  ASSERT_TRUE(model::ExtractQwenConfiguration(file.get(), &facts, &extraction_error)) << extraction_error;
#endif
  ConfigurationSettings settings;
  settings.token_capacity = 4;
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(facts, "immutable-gguf", settings, &configuration, &error)) << error;
  auto config = Parse(configuration.source);
  ASSERT_TRUE(config);
  for (const char *name : {"implementation_text.loom", "implementation.loombc"}) {
    SCOPED_TRACE(name);
    auto implementation = Load(root + "/lib/architectures/qwen35/" + name);
    ASSERT_TRUE(implementation);
    auto module = Link(config.get(), implementation.get());
    ASSERT_TRUE(module);
    Pass(module.get(), "canonicalize,unroll-scf-for,canonicalize");
    InspectCalls(module.get(), facts, 4);
    InspectParameters(module.get(), facts);
    Compile(module.get());
  }
}
TEST_F(QwenModelTest, MissingLiveConfigurationAndProvidersFailStrictLink) {
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "immutable-gguf", {}, &configuration, &error)) << error;
  auto implementation = Load(root + "/lib/architectures/qwen35/implementation.loombc");
  ASSERT_TRUE(implementation);
  auto line = configuration.source.find("config.def @runner.qwen38.prefill_token_count");
  ASSERT_NE(line, std::string::npos);
  configuration.source.erase(line, configuration.source.find('\n', line) - line + 1);
  auto missing = Parse(configuration.source);
  ASSERT_TRUE(missing);
  auto rejected = Link(missing.get(), implementation.get(), false);
  EXPECT_FALSE(rejected);
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "immutable-gguf", {}, &configuration, &error)) << error;
  auto config = Parse(configuration.source);
  auto model_only = Load(root + "/lib/architectures/qwen35/model.loom");
  ASSERT_TRUE(config && model_only);
  rejected = Link(config.get(), model_only.get(), false);
  EXPECT_FALSE(rejected);
}
} // namespace
} // namespace hrx::loomify_gguf
