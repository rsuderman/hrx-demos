// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/testing/configuration_fixture.h"
#include "lib/loomify/testing/alternating_qwen_fixture.h"
#include "tools/model/model_descriptor.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ir/context.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loomc/compile.h"
#include "loomc/interop.h"
#include "loomc/link.h"
#include "loomc/pass.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iterator>
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
  return ok ? ::testing::AssertionSuccess()
            : ::testing::AssertionFailure() << message;
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
    if (entry.name_id != LOOM_STRING_ID_INVALID &&
        Text(module, entry.name_id) == name)
      return entry.defining_op;
  }
  return nullptr;
}
int64_t Constant(const loom_module_t *module, loom_value_id_t value) {
  const auto *op = loom_value_def_op(loom_module_value(module, value));
  EXPECT_TRUE(op && loom_index_constant_isa(op));
  if (!op || !loom_index_constant_isa(op))
    return -1;
  const auto attr = loom_index_constant_value(op);
  EXPECT_EQ(attr.kind, LOOM_ATTR_I64);
  return attr.i64;
}
void Walk(const loom_region_t *region,
          const std::function<void(const loom_op_t *)> &visit) {
  for (size_t b = 0; b < region->block_count; ++b)
    for (const auto *op = region->blocks[b]->first_op; op; op = op->next_op) {
      visit(op);
      for (size_t i = 0; i < op->region_count; ++i)
        Walk(loom_op_regions(op)[i], visit);
    }
}
class QwenModelTest : public ::testing::Test {
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
  Module Parse(const std::string &text) {
    loomc_source_options_t options{};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier = loomc_make_cstring_view("qwen-test-config.loom");
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.contents = loomc_make_byte_span(text.data(), text.size());
    loomc_source_t *source = nullptr;
    EXPECT_TRUE(
        Ok(loomc_source_create(&options, loomc_allocator_system(), &source)));
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(), &module,
        &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    if (source)
      loomc_source_release(source);
    return Module(module, loomc_module_release);
  }
  Module Load(const std::string &path) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
        loomc_allocator_system(), &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    return Module(module, loomc_module_release);
  }
  Module Link(loomc_module_t *config, loomc_module_t *implementation,
              bool succeeds = true) {
    loomc_linker_t *linker = nullptr;
    EXPECT_TRUE(Ok(loomc_linker_create(context, nullptr,
                                       loomc_allocator_system(), &linker)));
    auto targets = Load(root + "/lib/platforms/amdgpu/gfx11/targets.loombc");
    auto schedule = Load(
        root +
        "/lib/platforms/amdgpu/gfx11/attention_h256_implementation.loombc");
    loomc_link_module_provider_t providers[] = {
        {config, loomc_make_cstring_view("configuration"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {implementation, loomc_make_cstring_view("implementation"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {targets.get(), loomc_make_cstring_view("platform"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {schedule.get(), loomc_make_cstring_view("attention schedule"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT}};
    const auto symbol = loomc_make_cstring_view("generated.qwen_epoch");
    loomc_link_options_t options{};
    options.mode = LOOMC_LINK_MODE_LINK;
    options.module_providers = providers;
    options.module_provider_count = 4;
    options.root_symbols = &symbol;
    options.root_symbol_count = 1;
    options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status =
        loomc_link_module(linker, workspace, &options, &module, &result);
    if (succeeds) {
      EXPECT_TRUE(Ok(status));
      EXPECT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      EXPECT_NE(module, nullptr);
    } else {
      const bool failed = !loomc_status_is_ok(status) || !result ||
                          !loomc_result_succeeded(result);
      loomc_status_free(status);
      EXPECT_TRUE(failed) << "missing live dependency unexpectedly linked";
      EXPECT_EQ(module, nullptr);
      EXPECT_FALSE(Diagnostics(result).empty());
    }
    if (result)
      loomc_result_release(result);
    if (linker)
      loomc_linker_release(linker);
    return Module(module, loomc_module_release);
  }
  void Pass(loomc_module_t *module, const char *pipeline) {
    loomc_pass_program_t *passes = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(
        context, loomc_make_cstring_view(pipeline), nullptr,
        loomc_allocator_system(), &passes, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_run(passes, workspace, module, nullptr,
                                          loomc_allocator_system(), &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    loomc_pass_program_release(passes);
  }
  void InspectCalls(loomc_module_t *module, const model::ModelDescriptor &facts,
                    int64_t tokens) {
    loomc_module_interop_view_t view{};
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(
        module, loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    const auto *epoch = Definition(view.module, "generated.qwen_epoch");
    ASSERT_TRUE(epoch && loom_command_def_isa(epoch));
    const auto *epoch_body =
        loom_region_const_entry_block(loom_command_def_body(epoch));
    ASSERT_EQ(epoch_body->arg_count, 7u);
    std::vector<const loom_op_t *> boundary;
    Walk(loom_command_def_body(epoch), [&](const loom_op_t *op) {
      if (loom_buffer_alloca_isa(op) || loom_command_launch_isa(op) ||
          loom_kernel_launch_isa(op) || loom_scf_for_isa(op))
        boundary.push_back(op);
    });
    ASSERT_EQ(boundary.size(), 3u);
    ASSERT_TRUE(loom_buffer_alloca_isa(boundary[0]));
    EXPECT_EQ(
        Constant(view.module, loom_buffer_alloca_byte_length(boundary[0])), 16);
    EXPECT_EQ(loom_buffer_alloca_base_alignment(boundary[0]), 256);
    ASSERT_EQ(boundary[0]->result_count, 1u);
    const auto selected_ids = loom_op_const_results(boundary[0])[0];
    ASSERT_TRUE(loom_command_launch_isa(boundary[1]));
    EXPECT_EQ(
        Text(view.module,
             view.module->symbols
                 .entries[loom_command_launch_callee(boundary[1]).symbol_id]
                 .name_id),
        "loomify.qwen.model");
    const auto model_args = loom_command_launch_bindings(boundary[1]);
    ASSERT_EQ(model_args.count, 7u);
    for (size_t i = 0; i < 6; ++i)
      EXPECT_EQ(model_args.values[i], epoch_body->arg_ids[i]);
    EXPECT_EQ(model_args.values[6], selected_ids);
    ASSERT_TRUE(loom_kernel_launch_isa(boundary[2]));
    EXPECT_EQ(
        Text(view.module,
             view.module->symbols
                 .entries[loom_kernel_launch_callee(boundary[2]).symbol_id]
                 .name_id),
        "qwen08_publish_selected_ids");
    const auto publication_args = loom_kernel_launch_arguments(boundary[2]);
    ASSERT_EQ(publication_args.count, 3u);
    EXPECT_EQ(publication_args.values[0], epoch_body->arg_ids[2]);
    EXPECT_EQ(publication_args.values[1], selected_ids);
    EXPECT_EQ(publication_args.values[2], epoch_body->arg_ids[6]);
    const auto *head_def = Definition(view.module, "loomify.qwen.head");
    ASSERT_TRUE(head_def && loom_command_def_isa(head_def));
    const auto *head_body =
        loom_region_const_entry_block(loom_command_def_body(head_def));
    // Native program bodies include two specialization arguments before
    // buffers.
    ASSERT_EQ(head_body->arg_count, 6u);
    size_t reduce_calls = 0;
    Walk(loom_command_def_body(head_def), [&](const loom_op_t *op) {
      if (!loom_kernel_launch_isa(op))
        return;
      const auto name = Text(
          view.module,
          view.module->symbols.entries[loom_kernel_launch_callee(op).symbol_id]
              .name_id);
      EXPECT_NE(name, "qwen08_publish_selected_ids");
      if (name != "qwen38_greedy_argmax_reduce_rows_f32")
        return;
      ++reduce_calls;
      const auto args = loom_kernel_launch_arguments(op);
      ASSERT_EQ(args.count, 5u);
      EXPECT_EQ(args.values[4], head_body->arg_ids[5]);
    });
    EXPECT_EQ(reduce_calls, 1u);
    const auto *def = Definition(view.module, "loomify.qwen.model");
    ASSERT_TRUE(def && loom_command_def_isa(def));
    const auto *body =
        loom_region_const_entry_block(loom_command_def_body(def));
    ASSERT_EQ(body->arg_count, 7u);
    std::vector<const loom_op_t *> calls, kernels;
    Walk(loom_command_def_body(def), [&](const loom_op_t *op) {
      if (loom_command_launch_isa(op))
        calls.push_back(op);
      if (loom_kernel_launch_isa(op))
        kernels.push_back(op);
    });
    ASSERT_EQ(kernels.size(), 1u);
    const auto *embedding = kernels.front();
    EXPECT_EQ(Text(view.module,
                   view.module->symbols
                       .entries[loom_kernel_launch_callee(embedding).symbol_id]
                       .name_id),
              facts.globals.front().tensor.storage_type == 12
                  ? "qwen38_token_embedding_q4k_f32_prefill"
                  : "qwen38_token_embedding_q6k_f16_prefill");
    auto workloads = loom_kernel_launch_workloads(embedding);
    ASSERT_EQ(workloads.count, 1u);
    EXPECT_EQ(Constant(view.module, workloads.values[0]), tokens);
    auto args = loom_kernel_launch_arguments(embedding);
    ASSERT_EQ(args.count, 3u);
    EXPECT_EQ(args.values[0], body->arg_ids[5]);
    EXPECT_EQ(args.values[2], body->arg_ids[1]);
    const auto *parameter =
        loom_value_def_op(loom_module_value(view.module, args.values[1]));
    ASSERT_TRUE(parameter && loom_command_parameter_isa(parameter));
    EXPECT_EQ(Text(view.module, loom_command_parameter_pattern(parameter)),
              "token_embd.weight");
    EXPECT_EQ(loom_command_parameter_source(parameter), body->arg_ids[0]);
    // Native unrolling preserves both conditional IR branches until callable
    // inlining. These are 24 routing sites, not 48 executed layers.
    std::vector<const loom_op_t *> routes, branches;
    Walk(loom_command_def_body(def), [&](const loom_op_t *op) {
      const auto name = loom_op_vtable_name(loom_op_vtable(view.module, op));
      if (std::string(name.data, name.size) == "func.call")
        routes.push_back(op);
      if (loom_scf_if_isa(op))
        branches.push_back(op);
    });
    ASSERT_EQ(routes.size(), facts.layers.size());
    ASSERT_EQ(branches.size(), facts.layers.size());
    ASSERT_EQ(calls.size(), 2 * branches.size() + 1);
    loom_value_id_t gdn_scratch = LOOM_VALUE_ID_INVALID,
                    attention_scratch = LOOM_VALUE_ID_INVALID;
    size_t selected_gdn = 0, selected_attention = 0;
    for (size_t i = 0; i < facts.layers.size(); ++i) {
      SCOPED_TRACE(i);
      const auto *route = routes[i];
      ASSERT_GE(route->attribute_count, 1u);
      const auto callee = loom_op_const_attrs(route)[0];
      ASSERT_EQ(callee.kind, LOOM_ATTR_SYMBOL);
      EXPECT_EQ(
          Text(view.module,
               view.module->symbols.entries[callee.symbol.symbol_id].name_id),
          "loomify.config.layer_route");
      ASSERT_EQ(route->operand_count, 1u);
      EXPECT_EQ(Constant(view.module, loom_op_const_operands(route)[0]),
                int64_t(i));
      ASSERT_EQ(route->result_count, 3u);
      const auto *tuple = loom_op_const_results(route);
      const auto *condition = loom_value_def_op(
          loom_module_value(view.module, loom_scf_if_condition(branches[i])));
      ASSERT_TRUE(condition && loom_index_cmp_isa(condition));
      EXPECT_EQ(loom_index_cmp_predicate(condition),
                LOOM_INDEX_CMP_PREDICATE_EQ);
      EXPECT_EQ(loom_index_cmp_lhs(condition), tuple[0]);
      EXPECT_EQ(Constant(view.module, loom_index_cmp_rhs(condition)), 0);
      for (size_t arm = 0; arm < 2; ++arm) {
        const bool is_gdn = arm == 1;
        const auto *region = is_gdn ? loom_scf_if_else_region(branches[i])
                                    : loom_scf_if_then_region(branches[i]);
        ASSERT_NE(region, nullptr);
        std::vector<const loom_op_t *> arm_calls;
        Walk(region, [&](const loom_op_t *op) {
          if (loom_command_launch_isa(op))
            arm_calls.push_back(op);
        });
        ASSERT_EQ(arm_calls.size(),
                  1u); // No appended FFN or hidden model call.
        const auto *op = arm_calls.front();
        EXPECT_EQ(calls[2 * i + arm], op);
        const auto callee = loom_command_launch_callee(op);
        EXPECT_EQ(Text(view.module,
                       view.module->symbols.entries[callee.symbol_id].name_id),
                  is_gdn ? "qwen38_gdn_layer_prefill"
                         : "qwen38_attention_layer_prefill");
        const auto specs = loom_command_launch_specializations(op);
        ASSERT_EQ(specs.count, 6u);
        EXPECT_EQ(Constant(view.module, specs.values[0]), tokens);
        EXPECT_EQ(Constant(view.module, specs.values[1]), tokens);
        EXPECT_EQ(specs.values[2], tuple[1]);
        EXPECT_EQ(specs.values[3], tuple[2]);
        EXPECT_EQ(Constant(view.module, specs.values[4]), 0);
        EXPECT_EQ(Constant(view.module, specs.values[5]), 1);
        const auto bindings = loom_command_launch_bindings(op);
        ASSERT_EQ(bindings.count, 6u);
        for (auto pair :
             {std::pair<size_t, size_t>{0, 0}, {1, 1}, {2, 4}, {3, 2}, {5, 3}})
          EXPECT_EQ(bindings.values[pair.first], body->arg_ids[pair.second]);
        auto &scratch = is_gdn ? gdn_scratch : attention_scratch;
        if (scratch == LOOM_VALUE_ID_INVALID)
          scratch = bindings.values[4];
        EXPECT_EQ(bindings.values[4], scratch);
        const auto *allocation =
            loom_value_def_op(loom_module_value(view.module, scratch));
        ASSERT_TRUE(allocation && loom_buffer_alloca_isa(allocation));
        EXPECT_EQ(
            Constant(view.module, loom_buffer_alloca_byte_length(allocation)),
            is_gdn ? 256
                   : int64_t(512 *
                             std::max(facts.ssm->inner_width,
                                      facts.layers.front().query_heads *
                                          facts.key_head_width) *
                             4));
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
    const auto gdn_count = std::count_if(
        facts.layers.begin(), facts.layers.end(), [](const auto &layer) {
          return layer.kind == model::LayerKind::kGatedDeltaNet;
        });
    EXPECT_EQ(selected_gdn, gdn_count);
    EXPECT_EQ(selected_attention, facts.layers.size() - gdn_count);
    EXPECT_NE(gdn_scratch, attention_scratch);
    std::printf(
        "Prepared Qwen hierarchy: %zu route sites; selection %zu attention/%zu GDN\n",
        facts.layers.size(), selected_attention, selected_gdn);
    const auto *head = calls.back();
    EXPECT_EQ(Text(view.module,
                   view.module->symbols
                       .entries[loom_command_launch_callee(head).symbol_id]
                       .name_id),
              "loomify.qwen.head");
    const auto specs = loom_command_launch_specializations(head);
    ASSERT_EQ(specs.count, 2u);
    EXPECT_EQ(Constant(view.module, specs.values[0]), tokens);
    EXPECT_EQ(Constant(view.module, specs.values[1]), 1);
    const auto bindings = loom_command_launch_bindings(head);
    ASSERT_EQ(bindings.count, 4u);
    const size_t expected[] = {0, 1, 2, 6};
    for (size_t j = 0; j < 4; ++j)
      EXPECT_EQ(bindings.values[j], body->arg_ids[expected[j]]);
  }
  void InspectParameters(loomc_module_t *module,
                         const model::ModelDescriptor &facts,
                         uint64_t pool = 512) {
    loomc_module_t *raw = nullptr;
    ASSERT_TRUE(Ok(
        loomc_module_clone(module, workspace, loomc_allocator_system(), &raw)));
    Module expanded(raw, loomc_module_release);
    Pass(expanded.get(), "inline-callables,canonicalize");
    loomc_module_interop_view_t view{};
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(
        expanded.get(), loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    const auto *def = Definition(view.module, "generated.qwen_epoch");
    ASSERT_TRUE(def && loom_command_def_isa(def));
    std::set<std::string> observed, expected;
    Walk(loom_command_def_body(def), [&](const loom_op_t *op) {
      if (!loom_command_parameter_isa(op))
        return;
      auto key = Text(view.module, loom_command_parameter_pattern(op));
      const auto substitutions = loom_command_parameter_substitutions(op);
      for (size_t i = 0; i < substitutions.count; ++i) {
        const auto position = key.find("{}");
        EXPECT_NE(position, std::string::npos);
        if (position != std::string::npos)
          key.replace(
              position, 2,
              std::to_string(Constant(view.module, substitutions.values[i])));
      }
      observed.insert(key);
    });
    for (const auto &tensor : facts.globals)
      expected.insert(tensor.tensor.name);
    for (const auto &layer : facts.layers)
      for (const auto &tensor : layer.tensors)
        expected.insert(tensor.tensor.name);
    EXPECT_EQ(
        observed,
        expected); // Every actual parameter identity, including embedded FFNs.
    // One metadata value reaches all four GDN normalization consumers. Inspect
    // actual linked operands, not source spelling or a second preset table.
    size_t rms_calls = 0, l2_calls = 0, gated_calls = 0;
    Walk(loom_command_def_body(def), [&](const loom_op_t *op) {
      if (!loom_kernel_launch_isa(op))
        return;
      const auto name = Text(
          view.module,
          view.module->symbols.entries[loom_kernel_launch_callee(op).symbol_id]
              .name_id);
      size_t epsilon_arg;
      if (name == "qwen38_rmsnorm_f32_prefill") {
        epsilon_arg = 1;
        ++rms_calls;
      } else if (name == "qwen38_gdn_conv_spans") {
        epsilon_arg = 3;
        ++l2_calls;
      } else if (name == "qwen38_gdn_norm_gate_prefill") {
        epsilon_arg = 1;
        ++gated_calls;
      } else
        return;
      const auto args = loom_kernel_launch_arguments(op);
      ASSERT_GT(args.count, epsilon_arg);
      const auto *value = loom_value_def_op(
          loom_module_value(view.module, args.values[epsilon_arg]));
      ASSERT_TRUE(value && loom_scalar_constant_isa(value)) << name;
      const auto attribute = loom_scalar_constant_value(value);
      ASSERT_EQ(attribute.kind, LOOM_ATTR_F64);
      EXPECT_FLOAT_EQ(float(attribute.f64), float(facts.rms_epsilon)) << name;
    });
    const auto recurrent_count = std::count_if(
        facts.layers.begin(), facts.layers.end(), [](const auto &layer) {
          return layer.kind == model::LayerKind::kGatedDeltaNet;
        });
    EXPECT_EQ(rms_calls, 2 * facts.layers.size());
    EXPECT_EQ(l2_calls, recurrent_count);
    EXPECT_EQ(gated_calls, recurrent_count);
    // After the native inliner folds route facts, one stateful marker remains
    // for each selected complete layer. Check order AND actual state ordinals.
    std::vector<const loom_op_t *> selected;
    Walk(loom_command_def_body(def), [&](const loom_op_t *op) {
      if (!loom_kernel_launch_isa(op))
        return;
      const auto name = Text(
          view.module,
          view.module->symbols.entries[loom_kernel_launch_callee(op).symbol_id]
              .name_id);
      if (name == "qwen38_gdn_conv_spans" ||
          name == "qwen38_attention_kv_prepare_spans")
        selected.push_back(op);
    });
    ASSERT_EQ(selected.size(), facts.layers.size());
    const auto &ssm = *facts.ssm;
    const uint64_t recurrent_bytes = ssm.recurrent_state_elements * 4;
    const uint64_t history_bytes = ssm.convolution_history_elements * 4;
    for (size_t i = 0; i < facts.layers.size(); ++i) {
      const auto *op = selected[i];
      const bool gdn = facts.layers[i].kind == model::LayerKind::kGatedDeltaNet;
      EXPECT_EQ(Text(view.module,
                     view.module->symbols
                         .entries[loom_kernel_launch_callee(op).symbol_id]
                         .name_id),
                gdn ? "qwen38_gdn_conv_spans"
                    : "qwen38_attention_kv_prepare_spans");
      const auto args = loom_kernel_launch_arguments(op);
      ASSERT_GE(args.count, 3u);
      const uint64_t origin =
          gdn ? *facts.layers[i].recurrent_ordinal *
                        (recurrent_bytes + history_bytes) +
                    recurrent_bytes
              : *facts.layers[i].attention_ordinal * 2 * pool *
                    (2 * facts.layers.front().kv_heads * facts.key_head_width);
      EXPECT_EQ(Constant(view.module, args.values[2]), int64_t(origin));
    }
  }
  void Compile(loomc_module_t *module) {
    loomc_target_profile_t *profile = nullptr;
    ASSERT_TRUE(Ok(loomc_target_profile_select(
        environment, loomc_make_cstring_view("amdgpu:gfx1100"),
        loomc_allocator_system(), &profile)));
    const auto symbol = loomc_make_cstring_view("generated.qwen_epoch");
    std::vector<loomc_request_t *> requests;
    loomc_cmd_program_module_options_t options{};
    options.root_symbols = &symbol;
    options.root_symbol_count = 1;
    options.target_profile = profile;
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
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    if (product)
      loomc_product_release(product);
    ASSERT_FALSE(requests.empty());
    loomc_compiler_t *compiler = nullptr;
    ASSERT_TRUE(Ok(loomc_compiler_create(context, nullptr,
                                         loomc_allocator_system(), &compiler)));
    for (auto *request : requests) {
      loomc_module_t *raw = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context, workspace, loomc_request_source(request), nullptr,
          loomc_allocator_system(), &raw, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      Module leaf(raw, loomc_module_release);
      result = nullptr;
      loomc_compile_artifact_options_t compile{};
      compile.target_profile = profile;
      ASSERT_TRUE(Ok(
          loomc_compile_artifact(compiler, workspace, nullptr, leaf.get(),
                                 &compile, loomc_allocator_system(), &result)));
      EXPECT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      EXPECT_GT(result ? loomc_result_artifact_count(result) : 0u, 0u);
      if (result)
        loomc_result_release(result);
      loomc_request_release(request);
    }
    std::printf(
        "Full Qwen materialization and target compilation: %zu kernel requests\n",
        requests.size());
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
#ifdef LOOMIFY_QWEN_LARGE_GGUF
  facts = testing::QwenModelFacts(true);
#endif
#ifdef LOOMIFY_QWEN_REAL_GGUF
  const char *asset = std::getenv(
#ifdef LOOMIFY_QWEN_LARGE_GGUF
      "LOOMIFY_TEST_QWEN_27_GGUF"
#else
      "LOOMIFY_TEST_QWEN_GGUF"
#endif
  );
  ASSERT_NE(asset, nullptr)
      << "real header fixture required; no checkpoint download/substitution";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(asset, "rb"),
                                                     &std::fclose);
  ASSERT_TRUE(file);
  std::setvbuf(file.get(), nullptr, _IONBF, 0);
  std::string extraction_error;
  model::GgufCatalog catalog;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &extraction_error))
      << extraction_error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &facts, &extraction_error))
      << extraction_error;
  ASSERT_EQ(facts.family, "qwen35");
#endif
  ConfigurationSettings settings;
  settings.token_capacity = 4;
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(facts, "immutable-gguf", settings,
                                  &configuration, &error))
      << error;
  auto config = Parse(configuration.source);
  ASSERT_TRUE(config);
  for (const char *name :
       {"implementation_text.loom", "implementation.loombc"}) {
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
TEST_F(QwenModelTest, AlteredScalarDataReachesAllGdnConsumers) {
  auto facts = testing::QwenModelFacts();
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(facts, "fixture", {}, &configuration, &error))
      << error;
  // U1 does not widen model admission. Change the linked leaf data only, so a
  // stale literal in any complete-layer consumer is nevertheless observable.
  const auto start =
      configuration.source.find("config.def @qwen.model.rms_epsilon = ");
  ASSERT_NE(start, std::string::npos);
  configuration.source.replace(
      start, configuration.source.find('\n', start) - start,
      "config.def @qwen.model.rms_epsilon = 0.00002 : f32");
  facts.rms_epsilon = 2e-5;
  for (const char *name :
       {"implementation_text.loom", "implementation.loombc"}) {
    auto config = Parse(configuration.source);
    auto implementation = Load(root + "/lib/architectures/qwen35/" + name);
    auto module = Link(config.get(), implementation.get());
    ASSERT_TRUE(module);
    Pass(module.get(), "canonicalize,unroll-scf-for,canonicalize");
    InspectParameters(module.get(), facts);
  }
}
TEST_F(QwenModelTest, IndependentAlternatingGgufUsesUnchangedCompleteBlocks) {
  model::ModelDescriptor facts;
  std::string error;
  ASSERT_TRUE(testing::LoadAlternatingQwenFixture(&facts, nullptr, &error))
      << error;
  ConfigurationFacts configuration;
  ASSERT_TRUE(DeriveConfiguration(facts, "independent.gguf", {3, 65},
                                  &configuration, &error))
      << error;
  for (const char *name :
       {"implementation_text.loom", "implementation.loombc"}) {
    auto config = Parse(configuration.source);
    auto implementation = Load(root + "/lib/architectures/qwen35/" + name);
    auto module = Link(config.get(), implementation.get());
    ASSERT_TRUE(module);
    Pass(module.get(), "canonicalize,unroll-scf-for,canonicalize");
    InspectCalls(module.get(), facts, 3);
    InspectParameters(module.get(), facts, 128);
    Compile(module.get());
  }
}
TEST_F(QwenModelTest, ChangedGgufFactsReachTheNormalConstructor) {
  for (bool tied : {true, false}) {
    model::ModelDescriptor facts;
    std::string error;
    ASSERT_TRUE(testing::LoadAlternatingQwenFixture(&facts, nullptr, &error,
                                                    4096, 8, tied))
        << error;
    ConfigurationFacts configuration;
    ASSERT_TRUE(DeriveConfiguration(facts, "changed.gguf", {3, 65},
                                    &configuration, &error))
        << error;
    for (const char *name :
         {"implementation_text.loom", "implementation.loombc"}) {
      auto config = Parse(configuration.source);
      auto implementation = Load(root + "/lib/architectures/qwen35/" + name);
      auto module = Link(config.get(), implementation.get());
      ASSERT_TRUE(module);
      Pass(module.get(), "canonicalize,unroll-scf-for,canonicalize");
      InspectCalls(module.get(), facts, 3);
      InspectParameters(module.get(), facts, 128);
      // The product path used by the public builder must actually materialize
      // operation predicates; plain LINK alone is insufficient admission.
      loomc_module_t *raw = nullptr;
      ASSERT_TRUE(LinkConfiguredProducts(
          context, workspace, configuration.source,
          {root + "/lib/architectures/qwen35/" + name,
           root + "/lib/platforms/amdgpu/gfx11/targets.loombc",
           root +
               "/lib/platforms/amdgpu/gfx11/attention_h256_implementation.loombc"},
          {"generated.qwen_epoch"}, true, &raw, &error))
          << error;
      Module constructed(raw, loomc_module_release);
      ASSERT_TRUE(constructed);
      Compile(constructed.get());
    }
  }
}
TEST_F(QwenModelTest, UnsupportedQueryEncodingFailsBeforeNumericalPublication) {
  model::ModelDescriptor facts;
  std::string error;
  ASSERT_TRUE(
      testing::LoadAlternatingQwenFixture(&facts, nullptr, &error, 4096, 0))
      << error;
  ConfigurationFacts configuration;
  // F32 is a valid GGUF tensor representation; no F32 matrix provider is
  // implemented by this operation. Reject at the operation, not an E preset.
  ASSERT_TRUE(DeriveConfiguration(facts, "unsupported.gguf", {3, 65},
                                  &configuration, &error))
      << error;
  for (const char *name :
       {"implementation_text.loom", "implementation.loombc"}) {
    loomc_module_t *raw = nullptr;
    EXPECT_FALSE(LinkConfiguredProducts(
        context, workspace, configuration.source,
        {root + "/lib/architectures/qwen35/" + name,
         root + "/lib/platforms/amdgpu/gfx11/targets.loombc",
         root +
             "/lib/platforms/amdgpu/gfx11/attention_h256_implementation.loombc"},
        {"generated.qwen_epoch"}, true, &raw, &error));
    EXPECT_EQ(raw, nullptr);
    EXPECT_FALSE(error.empty());
  }
}
TEST_F(QwenModelTest,
       UnsupportedConvolutionKeyHeadCountFailsBeforePublication) {
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "fixture", {3, 65},
                                  &configuration, &error))
      << error;
  // A linked-data sensitivity probe, not a second coherent GGUF architecture:
  // the production convolution has K-channel assumptions for sixteen heads.
  const std::string prefix = "config.def @qwen.model.gdn_key_heads = ";
  const auto start = configuration.source.find(prefix);
  ASSERT_NE(start, std::string::npos);
  configuration.source.replace(start,
                               configuration.source.find('\n', start) - start,
                               prefix + "8 : index");
  for (const char *name :
       {"implementation_text.loom", "implementation.loombc"}) {
    loomc_module_t *raw = nullptr;
    EXPECT_FALSE(LinkConfiguredProducts(
        context, workspace, configuration.source,
        {root + "/lib/architectures/qwen35/" + name,
         root + "/lib/platforms/amdgpu/gfx11/targets.loombc",
         root +
             "/lib/platforms/amdgpu/gfx11/attention_h256_implementation.loombc"},
        {"generated.qwen_epoch"}, true, &raw, &error));
    EXPECT_EQ(raw, nullptr);
    EXPECT_FALSE(error.empty());
  }
}
TEST_F(QwenModelTest, SeparatePlatformScheduleMaterializesAcrossModules) {
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "immutable-gguf",
                                  {3, 65}, &configuration, &error))
      << error;
  ASSERT_NE(std::getenv("TEST_TMPDIR"), nullptr);
  const std::string temporary = std::getenv("TEST_TMPDIR");
  const auto bytecode =
      root + "/lib/architectures/qwen35/implementation.loombc";
  auto implementation = Load(bytecode);
  const auto source = temporary + "/schedule_free.loom";
  loomc_module_serialize_options_t serialization{};
  serialization.format = LOOMC_SOURCE_FORMAT_TEXT;
  ASSERT_TRUE(Ok(loomc_module_serialize_to_path(
      implementation.get(), &serialization,
      loomc_make_cstring_view(source.c_str()), loomc_allocator_system())));
  for (const auto &input : {source, bytecode}) {
    loomc_module_t *raw = nullptr;
    ASSERT_TRUE(LinkConfiguredProducts(
        context, workspace, configuration.source,
        {input, root + "/lib/platforms/amdgpu/gfx11/targets.loombc",
         root +
             "/lib/platforms/amdgpu/gfx11/attention_h256_implementation.loombc"},
        {"generated.qwen_epoch"}, true, &raw, &error))
        << error;
    Module module(raw, loomc_module_release);
    for (const auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      serialization.format = format;
      const auto path = temporary + (format == LOOMC_SOURCE_FORMAT_TEXT
                                         ? "/separate_schedule.loom"
                                         : "/separate_schedule.loombc");
      ASSERT_TRUE(Ok(loomc_module_serialize_to_path(
          module.get(), &serialization, loomc_make_cstring_view(path.c_str()),
          loomc_allocator_system())));
      auto reloaded = Load(path);
      Compile(reloaded.get());
    }
  }
}
TEST_F(QwenModelTest, ExcludingPlatformSchedulePredicateFailsMaterialization) {
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "immutable-gguf",
                                  {3, 65}, &configuration, &error))
      << error;
  std::ifstream input(
      root + "/lib/platforms/amdgpu/gfx11/attention_h256_schedule.loom");
  ASSERT_TRUE(input);
  std::string schedule((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  const std::string predicate = "where [eq(%c256, 256)]";
  const auto position = schedule.find(predicate);
  ASSERT_NE(position, std::string::npos);
  ASSERT_EQ(schedule.find(predicate, position + predicate.size()),
            std::string::npos);
  schedule.replace(position, predicate.size(), "where [eq(%c256, 257)]");
  ASSERT_NE(std::getenv("TEST_TMPDIR"), nullptr);
  const auto path =
      std::string(std::getenv("TEST_TMPDIR")) + "/excluding_schedule.loom";
  {
    std::ofstream output(path);
    ASSERT_TRUE(output);
    output << schedule;
    output.close();
    ASSERT_TRUE(output);
  }
  loomc_module_t *raw = nullptr;
  EXPECT_FALSE(LinkConfiguredProducts(
      context, workspace, configuration.source,
      {root + "/lib/architectures/qwen35/implementation.loombc",
       root + "/lib/platforms/amdgpu/gfx11/targets.loombc", path},
      {"generated.qwen_epoch"}, true, &raw, &error));
  Module module(raw, loomc_module_release);
  EXPECT_FALSE(module);
  EXPECT_NE(error.find("llm.attention.prefill_wmma_h256"), std::string::npos)
      << error;
}
TEST_F(QwenModelTest, MissingPlatformScheduleFailsFinalMaterialization) {
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "immutable-gguf",
                                  {}, &configuration, &error))
      << error;
  loomc_module_t *raw = nullptr;
  EXPECT_FALSE(LinkConfiguredProducts(
      context, workspace, configuration.source,
      {root + "/lib/architectures/qwen35/implementation.loombc",
       root + "/lib/platforms/amdgpu/gfx11/targets.loombc"},
      {"generated.qwen_epoch"}, true, &raw, &error));
  Module module(raw, loomc_module_release);
  EXPECT_FALSE(module);
  EXPECT_NE(error.find("llm.attention.prefill_wmma_h256"), std::string::npos)
      << error;
}
TEST_F(QwenModelTest, MissingLiveConfigurationAndProvidersFailStrictLink) {
  ConfigurationFacts configuration;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "immutable-gguf",
                                  {}, &configuration, &error))
      << error;
  auto implementation =
      Load(root + "/lib/architectures/qwen35/implementation.loombc");
  ASSERT_TRUE(implementation);
  auto line = configuration.source.find(
      "config.def @runner.qwen38.prefill_token_count");
  ASSERT_NE(line, std::string::npos);
  configuration.source.erase(line,
                             configuration.source.find('\n', line) - line + 1);
  auto missing = Parse(configuration.source);
  ASSERT_TRUE(missing);
  auto rejected = Link(missing.get(), implementation.get(), false);
  EXPECT_FALSE(rejected);
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "immutable-gguf",
                                  {}, &configuration, &error))
      << error;
  auto config = Parse(configuration.source);
  auto model_only = Load(root + "/lib/architectures/qwen35/model.loom");
  ASSERT_TRUE(config && model_only);
  rejected = Link(config.get(), model_only.get(), false);
  EXPECT_FALSE(rejected);
}
} // namespace
} // namespace hrx::loomify_gguf
