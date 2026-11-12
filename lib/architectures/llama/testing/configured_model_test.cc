// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loomc/compile.h"
#include "loomc/interop.h"
#include "loomc/link.h"
#include "loomc/pass.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/configuration_fixture.h"
#include "tools/model/model_descriptor.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
namespace hrx::loomify_gguf {
namespace {
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
             const char *name, bool expect_success = true) {
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
  options.target_profile = profile;
  options.request_sink.publish = [](void *user, loomc_request_t *request) {
    static_cast<std::vector<loomc_request_t *> *>(user)->push_back(request);
    return loomc_ok_status();
  };
  options.request_sink.user_data = &requests;
  loomc_product_t *product = nullptr;
  loomc_result_t *result = nullptr;
  const auto build_status = loomc_cmd_program_product_build_module(
      workspace, module, &options, loomc_allocator_system(), &product, &result);
  const bool built = loomc_status_is_ok(build_status) && result &&
                     loomc_result_succeeded(result) && product;
  loomc_status_free(build_status);
  if (!built) {
    EXPECT_FALSE(expect_success) << Diagnostics(result);
    if (!expect_success)
      EXPECT_NE(Diagnostics(result).find("llm.embedding.epoch"),
                std::string::npos)
          << Diagnostics(result);
    if (result)
      loomc_result_release(result);
    if (product)
      loomc_product_release(product);
    for (auto *request : requests)
      loomc_request_release(request);
    loomc_target_profile_release(profile);
    return;
  }
  if (result)
    loomc_result_release(result);
  if (product)
    loomc_product_release(product);
  EXPECT_FALSE(requests.empty());
  loomc_compiler_t *compiler = nullptr;
  ASSERT_TRUE(Ok(loomc_compiler_create(context, nullptr,
                                       loomc_allocator_system(), &compiler)));
  bool rejected = false;
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
    const auto compile_status =
        loomc_compile_artifact(compiler, workspace, nullptr, leaf, &compile,
                               loomc_allocator_system(), &result);
    const bool accepted = loomc_status_is_ok(compile_status) && result &&
                          loomc_result_succeeded(result);
    loomc_status_free(compile_status);
    if (expect_success) {
      EXPECT_TRUE(accepted) << Diagnostics(result);
    } else if (!accepted) {
      rejected = true;
      EXPECT_NE(Diagnostics(result).find("llm.embedding.epoch"),
                std::string::npos)
          << Diagnostics(result);
    }
    if (result && loomc_result_succeeded(result)) {
      EXPECT_GT(loomc_result_artifact_count(result), 0u);
    }
    if (result)
      loomc_result_release(result);
    loomc_module_release(leaf);
    loomc_request_release(request);
  }
  if (!expect_success)
    EXPECT_TRUE(rejected);
  loomc_compiler_release(compiler);
  loomc_target_profile_release(profile);
}

class ConfiguredLlamaModelTest : public ::testing::Test {
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
    const char *directory = std::getenv("TEST_SRCDIR");
    const char *name = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(directory, nullptr);
    ASSERT_NE(name, nullptr);
    root = std::string(directory) + "/" + name;
  }
  void TearDown() override {
    if (workspace)
      loomc_workspace_release(workspace);
    if (context)
      loomc_context_release(context);
    if (environment)
      loomc_target_environment_release(environment);
  }
  std::string Text(const loom_module_t *module, loom_string_id_t id) {
    auto text = loom_string_table_get(&module->strings, id);
    return std::string(text.data, text.size);
  }
  const loom_op_t *Definition(const loom_module_t *module, const char *name) {
    for (size_t i = 0; i < module->symbols.count; ++i) {
      const auto &symbol = module->symbols.entries[i];
      if (symbol.name_id != LOOM_STRING_ID_INVALID &&
          Text(module, symbol.name_id) == name)
        return symbol.defining_op;
    }
    return nullptr;
  }
  int64_t Constant(const loom_module_t *module, loom_value_id_t value) {
    auto *op = loom_value_def_op(loom_module_value(module, value));
    EXPECT_TRUE(op && loom_index_constant_isa(op));
    if (!op || !loom_index_constant_isa(op))
      return -1;
    auto attribute = loom_index_constant_value(op);
    EXPECT_EQ(attribute.kind, LOOM_ATTR_I64);
    return attribute.i64;
  }
  void InspectPrepared(loomc_module_t *module,
                       const model::ModelDescriptor &descriptor) {
    const size_t layers = descriptor.layers.size();
    const uint64_t hidden = descriptor.embedding_width;
    const uint64_t vocabulary = descriptor.vocabulary_size;
    loomc_module_interop_view_t view{};
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(
        module, loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    const auto *epoch = Definition(view.module, "generated.llama_epoch");
    ASSERT_TRUE(epoch && loom_command_def_isa(epoch));
    const auto *epoch_body =
        loom_region_const_entry_block(loom_command_def_body(epoch));
    ASSERT_EQ(epoch_body->arg_count, 7u);
    std::vector<const loom_op_t *> boundary;
    for (const auto *op = epoch_body->first_op; op; op = op->next_op)
      if (loom_buffer_alloca_isa(op) || loom_command_launch_isa(op) ||
          loom_kernel_launch_isa(op))
        boundary.push_back(op);
    ASSERT_EQ(boundary.size(), 3u);
    ASSERT_TRUE(loom_buffer_alloca_isa(boundary[0]));
    EXPECT_EQ(
        Constant(view.module, loom_buffer_alloca_byte_length(boundary[0])), 16);
    const auto selected_ids = loom_op_const_results(boundary[0])[0];
    ASSERT_TRUE(loom_command_launch_isa(boundary[1]));
    EXPECT_EQ(
        Text(view.module,
             view.module->symbols
                 .entries[loom_command_launch_callee(boundary[1]).symbol_id]
                 .name_id),
        "loomify.llama.model");
    const auto model_args = loom_command_launch_bindings(boundary[1]);
    ASSERT_EQ(model_args.count, 7u);
    for (size_t j = 0; j < 6; ++j)
      EXPECT_EQ(model_args.values[j], epoch_body->arg_ids[j]);
    EXPECT_EQ(model_args.values[6], selected_ids);
    ASSERT_TRUE(loom_kernel_launch_isa(boundary[2]));
    EXPECT_EQ(
        Text(view.module,
             view.module->symbols
                 .entries[loom_kernel_launch_callee(boundary[2]).symbol_id]
                 .name_id),
        "llama32_publish_epoch_id");
    const auto publication_args = loom_kernel_launch_arguments(boundary[2]);
    ASSERT_EQ(publication_args.count, 3u);
    EXPECT_EQ(publication_args.values[0], epoch_body->arg_ids[2]);
    EXPECT_EQ(publication_args.values[1], selected_ids);
    EXPECT_EQ(publication_args.values[2], epoch_body->arg_ids[6]);
    const auto *model = Definition(view.module, "loomify.llama.model");
    ASSERT_TRUE(model && loom_command_def_isa(model));
    const auto *body =
        loom_region_const_entry_block(loom_command_def_body(model));
    ASSERT_EQ(body->arg_count, 7u);
    loom_value_id_t count_value = 0;
    size_t layer = 0;
    int64_t rows = 0;
    for (const auto *op = body->first_op; op; op = op->next_op) {
      if (!loom_command_launch_isa(op))
        continue;
      auto callee = loom_command_launch_callee(op);
      ASSERT_LT(callee.symbol_id, view.module->symbols.count);
      EXPECT_EQ(Text(view.module,
                     view.module->symbols.entries[callee.symbol_id].name_id),
                "loomify.llama.layer");
      auto specializations = loom_command_launch_specializations(op);
      ASSERT_EQ(specializations.count, 4u);
      EXPECT_EQ(Constant(view.module, specializations.values[0]),
                int64_t(layer));
      EXPECT_EQ(Constant(view.module, specializations.values[1]), 0);
      EXPECT_GT(Constant(view.module, specializations.values[2]), 0);
      EXPECT_LE(Constant(view.module, specializations.values[2]), 512);
      if (layer == 0)
        rows = Constant(view.module, specializations.values[2]);
      else
        EXPECT_EQ(Constant(view.module, specializations.values[2]), rows);
      EXPECT_EQ(Constant(view.module, specializations.values[3]), 12);
      auto bindings = loom_command_launch_bindings(op);
      ASSERT_EQ(bindings.count, 6u);
      for (size_t j = 0; j < 5; ++j)
        EXPECT_EQ(bindings.values[j], body->arg_ids[j]);
      const auto *count_alloc =
          loom_value_def_op(loom_module_value(view.module, bindings.values[5]));
      ASSERT_TRUE(count_alloc && loom_buffer_alloca_isa(count_alloc));
      EXPECT_EQ(
          Constant(view.module, loom_buffer_alloca_byte_length(count_alloc)),
          12);
      if (count_value != 0)
        EXPECT_EQ(bindings.values[5], count_value);
      count_value = bindings.values[5];
      ++layer;
    }
    EXPECT_EQ(layer, layers);
    // Inspect native inlining of that prepared call sequence, not maintained
    // source text: concrete parameter identities must follow each layer's ABI.
    loomc_module_t *raw = nullptr;
    ASSERT_TRUE(Ok(
        loomc_module_clone(module, workspace, loomc_allocator_system(), &raw)));
    std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> expanded(
        raw, loomc_module_release);
    loomc_pass_program_t *passes = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(
        context,
        loomc_make_cstring_view(
            "inline-callables,canonicalize,select-templates{rewrite=inline},canonicalize"),
        nullptr, loomc_allocator_system(), &passes, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    std::unique_ptr<loomc_pass_program_t, decltype(&loomc_pass_program_release)>
        owned_passes(passes, loomc_pass_program_release);
    result = nullptr;
    ASSERT_TRUE(
        Ok(loomc_pass_program_run(passes, workspace, expanded.get(), nullptr,
                                  loomc_allocator_system(), &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(
        expanded.get(), loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    const auto *root = Definition(view.module, "generated.llama_epoch");
    ASSERT_TRUE(root && loom_command_def_isa(root));
    std::vector<std::string> keys;
    std::set<std::string> global_keys;
    std::set<std::string> embedding_kernels;
    size_t embedding_count = 0;
    uint32_t embedding_tag = 0;
    for (const auto &binding : descriptor.globals)
      if (binding.role == "token_embedding")
        embedding_tag = binding.tensor.storage_type;
    std::function<void(const loom_region_t *)> collect =
        [&](const loom_region_t *region) {
          for (size_t b = 0; b < region->block_count; ++b)
            for (const auto *op = region->blocks[b]->first_op; op;
                 op = op->next_op) {
              if (loom_command_parameter_isa(op)) {
                auto key =
                    Text(view.module, loom_command_parameter_pattern(op));
                auto substitutions = loom_command_parameter_substitutions(op);
                for (size_t i = 0; i < substitutions.count; ++i) {
                  auto position = key.find("{}");
                  EXPECT_NE(position, std::string::npos);
                  if (position != std::string::npos)
                    key.replace(position, 2,
                                std::to_string(Constant(
                                    view.module, substitutions.values[i])));
                }
                if (key.rfind("blk.", 0) == 0)
                  keys.push_back(std::move(key));
                else
                  global_keys.insert(std::move(key));
              }
              if (loom_kernel_launch_isa(op)) {
                const auto callee = loom_kernel_launch_callee(op);
                const auto name = Text(
                    view.module,
                    view.module->symbols.entries[callee.symbol_id].name_id);
                if (name == "llm.token_embedding_epoch_f32") {
                  ++embedding_count;
                  embedding_kernels.insert(name);
                  const auto workloads = loom_kernel_launch_workloads(op);
                  ASSERT_EQ(workloads.count, 1u);
                  EXPECT_EQ(Constant(view.module, workloads.values[0]), rows);
                  const auto args = loom_kernel_launch_arguments(op);
                  ASSERT_EQ(args.count, 8u);
                  EXPECT_EQ(Constant(view.module, args.values[0]), hidden);
                  EXPECT_EQ(Constant(view.module, args.values[1]), vocabulary);
                  EXPECT_EQ(Constant(view.module, args.values[2]),
                            embedding_tag);
                }
              }
              for (size_t i = 0; i < op->region_count; ++i)
                collect(loom_op_regions(op)[i]);
            }
        };
    collect(loom_command_def_body(root));
    std::vector<std::string> expected;
    for (size_t i = 0; i < layers; ++i)
      for (const char *suffix :
           {"attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
            "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"})
        expected.push_back("blk." + std::to_string(i) + "." + suffix +
                           ".weight");
    EXPECT_EQ(keys, expected);
    std::set<std::string> expected_globals;
    for (const auto &binding : descriptor.globals) {
      expected_globals.insert(binding.tensor.name);
    }
    EXPECT_EQ(global_keys, expected_globals);
    EXPECT_EQ(embedding_count, 1u);
    EXPECT_EQ(embedding_kernels,
              std::set<std::string>{"llm.token_embedding_epoch_f32"});
  }
  using Module =
      std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)>;
  void Link(const ConfigurationFacts &facts, const char *product, Module *out,
            bool success = true, const std::string &extra_source = {},
            const char *entry_symbol = "generated.llama_epoch") {
    loomc_source_options_t source_options{};
    source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
    source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
    source_options.identifier =
        loomc_make_cstring_view("llama-model-test-config");
    // Plain descriptor fixture: explicit publication identities are test data,
    // while the public builder derives these from its real GGUF vocabulary.
    const std::string source_text =
        facts.source + extra_source +
        "config.def @llama.publication.end_text_id = 128001 : i32\n"
        "config.def @llama.publication.end_message_id = 128008 : i32\n"
        "config.def @llama.publication.end_turn_id = 128009 : i32\n";
    source_options.contents =
        loomc_make_byte_span(source_text.data(), source_text.size());
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(Ok(loomc_source_create(&source_options,
                                       loomc_allocator_system(), &source)));
    loomc_module_t *raw_config = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(),
        &raw_config, &result)));
    Module config(raw_config, loomc_module_release);
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    loomc_source_release(source);
    loomc_module_t *raw_product = nullptr;
    result = nullptr;
    auto path = root + "/lib/architectures/llama/" + product;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
        loomc_allocator_system(), &raw_product, &result)));
    Module implementation(raw_product, loomc_module_release);
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    loomc_module_t *raw_targets = nullptr;
    result = nullptr;
    const auto target_path =
        root + "/lib/platforms/amdgpu/gfx11/targets.loombc";
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
        context, workspace, loomc_make_cstring_view(target_path.c_str()),
        nullptr, loomc_allocator_system(), &raw_targets, &result)));
    Module targets(raw_targets, loomc_module_release);
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    loomc_linker_t *raw_linker = nullptr;
    ASSERT_TRUE(Ok(loomc_linker_create(context, nullptr,
                                       loomc_allocator_system(), &raw_linker)));
    std::unique_ptr<loomc_linker_t, decltype(&loomc_linker_release)> linker(
        raw_linker, loomc_linker_release);
    const loomc_link_module_provider_t providers[] = {
        {config.get(), loomc_make_cstring_view("configuration"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {implementation.get(), loomc_make_cstring_view("implementation"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {targets.get(), loomc_make_cstring_view("platform"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT}};
    const loomc_string_view_t roots[] = {loomc_make_cstring_view(entry_symbol)};
    loomc_link_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    options.structure_size = sizeof(options);
    options.mode = LOOMC_LINK_MODE_LINK;
    options.module_providers = providers;
    options.module_provider_count = 3;
    options.root_symbols = roots;
    options.root_symbol_count = 1;
    options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    loomc_module_t *raw = nullptr;
    result = nullptr;
    const auto status =
        loomc_link_module(linker.get(), workspace, &options, &raw, &result);
    Module linked(raw, loomc_module_release);
    const bool accepted = loomc_status_is_ok(status) && result &&
                          loomc_result_succeeded(result) && linked;
    if (success)
      EXPECT_TRUE(accepted) << Diagnostics(result);
    else
      EXPECT_FALSE(accepted);
    loomc_status_free(status);
    if (result)
      loomc_result_release(result);
    if (accepted)
      *out = std::move(linked);
  }
  void Build(const model::ModelDescriptor &descriptor, uint64_t rows,
             bool materialize, bool text_product = false) {
    ConfigurationFacts facts;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(descriptor, "immutable-gguf", {rows, 513},
                                    &facts, &error))
        << error;
    Module built(nullptr, loomc_module_release);
    Link(facts,
         text_product ? "implementation_text.loom" : "implementation.loombc",
         &built);
    ASSERT_TRUE(built);
    loomc_pass_program_t *raw_passes = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(
        context,
        loomc_make_cstring_view("canonicalize,unroll-scf-for,canonicalize"),
        nullptr, loomc_allocator_system(), &raw_passes, &result)));
    std::unique_ptr<loomc_pass_program_t, decltype(&loomc_pass_program_release)>
        passes(raw_passes, loomc_pass_program_release);
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    ASSERT_TRUE(
        Ok(loomc_pass_program_run(passes.get(), workspace, built.get(), nullptr,
                                  loomc_allocator_system(), &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    InspectPrepared(built.get(), descriptor);
    facts = {}; // linked source owns its configuration independently.
    for (auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options{};
      options.format = format;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          built.get(), &options, loomc_allocator_system(), &source)));
      loomc_module_t *loaded = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context, workspace, source, nullptr, loomc_allocator_system(),
          &loaded, &result)));
      Module roundtrip(loaded, loomc_module_release);
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      loomc_source_release(source);
      if (materialize && format == LOOMC_SOURCE_FORMAT_BYTECODE) {
        Compile(context, workspace, environment, loaded,
                "generated.llama_epoch");
      }
    }
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_F(ConfiguredLlamaModelTest, RealGgufMaintainedArchitectureAndEpochRoot) {
  const char *path = std::getenv("LOOMIFY_TEST_GGUF");
  if (!path)
    GTEST_SKIP() << "requires explicit immutable Llama GGUF gate";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     &std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::ModelDescriptor descriptor;
  std::string error;
  model::GgufCatalog catalog;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &descriptor, &error))
      << error;
  ASSERT_EQ(descriptor.family, "llama");
  file.reset();
  ASSERT_EQ(descriptor.layers.size(), 28u);
  auto down = [&](size_t layer) {
    for (const auto &t : descriptor.layers[layer].tensors)
      if (t.role == "ffn_down")
        return t.tensor.storage_type;
    return uint32_t(0);
  };
  EXPECT_EQ(down(0), 14u);
  EXPECT_EQ(down(1), 14u);
  EXPECT_EQ(down(3), 12u);
  Build(descriptor, 1, false, true);
  Build(descriptor, 128, true);
}
TEST_F(ConfiguredLlamaModelTest, FullDescriptorFixtureUsesMaintainedProducts) {
  for (bool text : {false, true})
    Build(testing::LlamaModelFacts(), 128, false, text);
}
TEST_F(ConfiguredLlamaModelTest,
       PlainLlamaUsesOnlyActualBindingsInBothProducts) {
  for (bool text : {false, true})
    for (uint64_t rows : {8u, 64u})
      Build(testing::LlamaPlainModelFacts(), rows, false, text);
}
TEST_F(ConfiguredLlamaModelTest,
       RealHyperclovaxUsesMaintainedArchitectureAndEpochRoot) {
  const char *path = std::getenv("LOOMIFY_TEST_HYPERCLOVAX_GGUF");
  if (!path)
    GTEST_SKIP() << "requires explicit immutable HyperCLOVAX GGUF gate";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &descriptor, &error))
      << error;
  ASSERT_EQ(descriptor.family, "llama");
  ASSERT_EQ(descriptor.layers.size(), 24u);
  EXPECT_FALSE(descriptor.output_tied_to_embedding);
  for (bool text : {false, true})
    for (uint64_t rows : {8u, 64u})
      Build(descriptor, rows, true, text);
}
TEST_F(ConfiguredLlamaModelTest,
       AlteredScalarDataReachesRotaryAndAttentionProviders) {
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::LlamaModelFacts(), "fixture", {},
                                  &facts, &error))
      << error;
  // Scalar leaf sensitivity does not widen full-model geometry admission.
  for (const auto &entry : std::vector<std::pair<std::string, std::string>>{
           {"llama.model.rotary_step", "0.75"},
           {"llama.model.attention_scale", "0.125"}}) {
    const auto start = facts.source.find("config.def @" + entry.first + " = ");
    ASSERT_NE(start, std::string::npos);
    facts.source.replace(start, facts.source.find('\n', start) - start,
                         "config.def @" + entry.first + " = " + entry.second +
                             " : f32");
  }
  for (const char *product :
       {"implementation_text.loom", "implementation.loombc"}) {
    Module module(nullptr, loomc_module_release);
    Link(facts, product, &module);
    ASSERT_TRUE(module);
    // LINK resolves configuration definitions but need not fold config.get.
    // Inspect the same canonicalized representation used by prepared products.
    loomc_pass_program_t *passes = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(
        context, loomc_make_cstring_view("canonicalize"), nullptr,
        loomc_allocator_system(), &passes, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    ASSERT_TRUE(
        Ok(loomc_pass_program_run(passes, workspace, module.get(), nullptr,
                                  loomc_allocator_system(), &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    loomc_pass_program_release(passes);
    loomc_module_interop_view_t view{};
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(
        module.get(), loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_result_release(result);
    for (const auto &entry : std::vector<std::pair<const char *, float>>{
             {"llama32_rotary_query", 0.75f},
             {"llama32_publish_kv", 0.75f},
             {"llama32_attention_rows", 0.125f}}) {
      const auto *kernel = Definition(view.module, entry.first);
      ASSERT_TRUE(kernel && loom_kernel_def_isa(kernel)) << entry.first;
      size_t uses = 0;
      std::function<void(const loom_region_t *)> inspect =
          [&](const loom_region_t *region) {
            for (size_t b = 0; b < region->block_count; ++b)
              for (const auto *op = region->blocks[b]->first_op; op;
                   op = op->next_op) {
                const auto spelling =
                    loom_op_vtable_name(loom_op_vtable(view.module, op));
                if (std::string(spelling.data, spelling.size) ==
                    "template.apply") {
                  for (size_t i = 0; i < op->operand_count; ++i) {
                    const auto *operand = loom_module_value(
                        view.module, loom_op_const_operands(op)[i]);
                    // Buffer and position arguments have no defining operation.
                    // The public IR accessor requires excluding block
                    // arguments.
                    if (loom_value_is_block_arg(operand))
                      continue;
                    const auto *value = loom_value_def_op(operand);
                    if (!value || !loom_scalar_constant_isa(value))
                      continue;
                    const auto attribute = loom_scalar_constant_value(value);
                    if (attribute.kind == LOOM_ATTR_F64 &&
                        float(attribute.f64) == entry.second)
                      ++uses;
                  }
                }
                for (size_t i = 0; i < op->region_count; ++i)
                  inspect(loom_op_regions(op)[i]);
              }
          };
      inspect(loom_kernel_def_body(kernel));
      EXPECT_GT(uses, 0u) << entry.first;
    }
  }
}
TEST_F(ConfiguredLlamaModelTest, EmbeddingOperationAdmitsItsOwnStaticFacts) {
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::LlamaPlainModelFacts(), "asset", {},
                                  &facts, &error))
      << error;
  // Root the real production operation directly. A full-model root would let
  // unrelated unresolved FFN/projection templates mask the intended failure.
  struct Case {
    int hidden;
    int vocabulary;
    int tag;
    bool accepted;
  };
  for (const auto *product :
       {"implementation_text.loom", "implementation.loombc"}) {
    for (const auto c : {Case{512, 3, 12, true}, Case{512, 3, 14, true},
                         Case{512, 3, 13, false}, Case{513, 3, 12, false},
                         Case{512, 0, 12, false}}) {
      const std::string entry =
          "template.decl @llm.embedding.epoch_f32(%r:index,%h:index,%v:index,%t:index,%b:index,%e:buffer,%c:buffer,%i:buffer,%w:view<[%b]xi8>,%o:view<[%r]x[%h]xf32>)\n"
          "command.def @embedding.contract() launch(%e:buffer,%c:buffer,%i:buffer,%w:buffer,%o:buffer) {\n"
          " %r = index.constant 8 : index\n"
          " %h = index.constant " +
          std::to_string(c.hidden) +
          " : index\n"
          " %v = index.constant " +
          std::to_string(c.vocabulary) +
          " : index\n"
          " %t = index.constant " +
          std::to_string(c.tag) +
          " : index\n"
          " %b = index.constant " +
          std::to_string(c.tag == 14 ? 1260 : 864) +
          " : index\n"
          " %z = index.constant 0 : offset\n"
          " %table = buffer.view %w[%z] : buffer -> view<[%b]xi8>\n"
          " %out = buffer.view %o[%z] : buffer -> view<[%r]x[%h]xf32>\n"
          " template.apply<@llm.embedding.epoch_f32>(%r,%h,%v,%t,%b,%e,%c,%i,%table,%out) : (index,index,index,index,index,buffer,buffer,buffer,view<[%b]xi8>,view<[%r]x[%h]xf32>)\n"
          " command.return\n}\n";
      Module module(nullptr, loomc_module_release);
      Link(facts, product, &module, true, entry, "embedding.contract");
      ASSERT_TRUE(module);
      Compile(context, workspace, environment, module.get(),
              "embedding.contract", c.accepted);
    }
  }
}
TEST_F(ConfiguredLlamaModelTest, MissingLiveConfigOrProviderRejects) {
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::LlamaModelFacts(), "asset", {},
                                  &facts, &error))
      << error;
  Module out(nullptr, loomc_module_release);
  Link(facts, "model.loom", &out, false);
  EXPECT_FALSE(out);
  const auto begin = facts.source.find("config.def @llama.model.hidden_size");
  ASSERT_NE(begin, std::string::npos);
  facts.source.erase(begin, facts.source.find('\n', begin) - begin + 1);
  Link(facts, "implementation.loombc", &out, false);
  EXPECT_FALSE(out);
}
} // namespace
} // namespace hrx::loomify_gguf
