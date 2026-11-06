// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/authoring/model-config/model.h"
#include "tools/loomify-gguf/authoring/model-config/providers.h"
#include "tools/model/llama_configuration.h"
#include "loomc/compile.h"
#include "loomc/interop.h"
#include "loomc/pass.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <memory>
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
    ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    const char *directory = std::getenv("TEST_SRCDIR");
    const char *name = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(directory, nullptr);
    ASSERT_NE(name, nullptr);
    root = std::string(directory) + "/" + name;
  }
  void TearDown() override {
    if (workspace) loomc_workspace_release(workspace);
    if (context) loomc_context_release(context);
    if (environment) loomc_target_environment_release(environment);
  }
  std::string Text(const loom_module_t *module, loom_string_id_t id) {
    auto text = loom_string_table_get(&module->strings, id);
    return std::string(text.data, text.size);
  }
  const loom_op_t *Definition(const loom_module_t *module, const char *name) {
    for (size_t i = 0; i < module->symbols.count; ++i) {
      const auto &symbol = module->symbols.entries[i];
      if (symbol.name_id != LOOM_STRING_ID_INVALID && Text(module, symbol.name_id) == name)
        return symbol.defining_op;
    }
    return nullptr;
  }
  int64_t Constant(const loom_module_t *module, loom_value_id_t value) {
    auto *op = loom_value_def_op(loom_module_value(module, value));
    EXPECT_TRUE(op && loom_index_constant_isa(op));
    if (!op || !loom_index_constant_isa(op)) return -1;
    auto attribute = loom_index_constant_value(op);
    EXPECT_EQ(attribute.kind, LOOM_ATTR_I64);
    return attribute.i64;
  }
  void InspectPrepared(loomc_module_t *module, size_t layers) {
    loomc_module_interop_view_t view{};
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(module, loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    const auto *model = Definition(view.module, "loomify.llama.model");
    ASSERT_TRUE(model && loom_command_program_def_isa(model));
    const auto *body = loom_region_const_entry_block(loom_command_program_def_body(model));
    ASSERT_EQ(body->arg_count, 9u);
    size_t layer = 0;
    for (const auto *op = body->first_op; op; op = op->next_op) {
      if (!loom_command_program_launch_isa(op)) continue;
      auto callee = loom_command_program_launch_callee(op);
      ASSERT_LT(callee.symbol_id, view.module->symbols.count);
      EXPECT_EQ(Text(view.module, view.module->symbols.entries[callee.symbol_id].name_id),
                "loomify.llama.layer");
      auto specializations = loom_command_program_launch_specializations(op);
      ASSERT_EQ(specializations.count, 3u);
      EXPECT_EQ(Constant(view.module, specializations.values[0]), int64_t(layer));
      EXPECT_EQ(specializations.values[1], body->arg_ids[0]);
      EXPECT_EQ(specializations.values[2], body->arg_ids[1]);
      auto bindings = loom_command_program_launch_bindings(op);
      ASSERT_EQ(bindings.count, 5u);
      for (size_t j = 0; j < 5; ++j)
        EXPECT_EQ(bindings.values[j], body->arg_ids[j + 2]);
      ++layer;
    }
    EXPECT_EQ(layer, layers);
    // Inspect native inlining of that prepared call sequence, not maintained
    // source text: concrete parameter identities must follow each layer's ABI.
    loomc_module_t *raw = nullptr;
    ASSERT_TRUE(Ok(loomc_module_clone(module, workspace, loomc_allocator_system(), &raw)));
    std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> expanded(raw, loomc_module_release);
    loomc_pass_program_t *passes = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(context,
        loomc_make_cstring_view("inline-callables,canonicalize"), nullptr,
        loomc_allocator_system(), &passes, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    std::unique_ptr<loomc_pass_program_t, decltype(&loomc_pass_program_release)> owned_passes(passes, loomc_pass_program_release);
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_pass_program_run(passes, workspace, expanded.get(), nullptr,
        loomc_allocator_system(), &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_get_interop_view(expanded.get(), loomc_allocator_system(), &view, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    const auto *root = Definition(view.module, "generated.llama_prefill");
    ASSERT_TRUE(root && loom_command_program_def_isa(root));
    std::vector<std::string> keys;
    std::function<void(const loom_region_t *)> collect = [&](const loom_region_t *region) {
      for (size_t b = 0; b < region->block_count; ++b)
        for (const auto *op = region->blocks[b]->first_op; op; op = op->next_op) {
          if (loom_command_parameter_isa(op)) {
            auto key = Text(view.module, loom_command_parameter_pattern(op));
            auto substitutions = loom_command_parameter_substitutions(op);
            for (size_t i = 0; i < substitutions.count; ++i) {
              auto position = key.find("{}");
              EXPECT_NE(position, std::string::npos);
              if (position != std::string::npos)
                key.replace(position, 2, std::to_string(Constant(view.module, substitutions.values[i])));
            }
            if (key.rfind("blk.", 0) == 0) keys.push_back(std::move(key));
          }
          for (size_t i = 0; i < op->region_count; ++i) collect(loom_op_regions(op)[i]);
        }
    };
    collect(loom_command_program_def_body(root));
    std::vector<std::string> expected;
    for (size_t i = 0; i < layers; ++i)
      for (const char *suffix : {"attn_norm", "attn_q", "attn_k", "attn_v",
                                 "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"})
        expected.push_back("blk." + std::to_string(i) + "." + suffix + ".weight");
    EXPECT_EQ(keys, expected);
  }
  void Build(const model::ModelDescriptor &descriptor, uint64_t rows,
             bool materialize) {
    LinkedConfigurationSettings settings;
    settings.token_capacity = rows;
    settings.context_capacity = 513; // Distinguish logical C from pooled P=576.
    LinkedConfiguration configuration;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(CreateLinkedConfiguration(context, workspace, descriptor,
        "immutable-gguf", settings, &configuration, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    result = nullptr;
    ConfiguredLlamaModel built;
    ASSERT_TRUE(Ok(BuildConfiguredLlamaModel(context, workspace, configuration,
        MakeLlamaLinkedProviders(root), &built, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    loomc_result_release(result);
    ASSERT_TRUE(built.module);
    InspectPrepared(built.module.get(), descriptor.layers.size());
    ASSERT_TRUE(configuration.facts.llama_layout);
    EXPECT_EQ(built.layout.cache_base, configuration.facts.llama_layout->cache_base);
    EXPECT_EQ(built.layout.cold_allocations[4].bytes,
              configuration.facts.llama_layout->cold_allocations[4].bytes);
    EXPECT_EQ(built.source_asset, "immutable-gguf");
    EXPECT_EQ(built.layout.token_capacity, rows);
    EXPECT_EQ(built.layout.context_capacity, 513u);
    EXPECT_EQ(built.layout.pool_capacity, 576u);
    if (descriptor.layers.size() == 1 && rows == 1) {
      auto incomplete = MakeLlamaLinkedProviders(root);
      incomplete.providers.erase(std::remove_if(incomplete.providers.begin(),
          incomplete.providers.end(), [](const ConfiguredProvider &provider) {
            return provider.path == "lib/kernels/llama/embedding/embedding.loom";
          }), incomplete.providers.end());
      ConfiguredLlamaModel unchanged;
      unchanged.layout.cache_base = 42;
      result = nullptr;
      auto rejected = BuildConfiguredLlamaModel(context, workspace, configuration,
          incomplete, &unchanged, &result);
      EXPECT_TRUE(!loomc_status_is_ok(rejected) || !result ||
                  !loomc_result_succeeded(result));
      loomc_status_free(rejected);
      if (result) loomc_result_release(result);
      EXPECT_FALSE(unchanged.module);
      EXPECT_EQ(unchanged.layout.cache_base, 42u);
    }
    // Drop the producer to prove the returned module/layout are independent.
    configuration = LinkedConfiguration{};
    for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options{};
      options.format = format;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(built.module.get(), &options,
          loomc_allocator_system(), &source)));
      loomc_module_t *loaded = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(context, workspace,
          source, nullptr, loomc_allocator_system(), &loaded, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      loomc_result_release(result);
      loomc_source_release(source);
      if (materialize && format == LOOMC_SOURCE_FORMAT_BYTECODE) {
        Compile(context, workspace, environment, loaded, "generated.llama_prefill");
        Compile(context, workspace, environment, loaded, "generated.llama_decode");
        Compile(context, workspace, environment, loaded, "generated.llama_epoch");
      }
      loomc_module_release(loaded);
    }
  }
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  std::string root;
};
TEST_F(ConfiguredLlamaModelTest, RealGgufMaintainedArchitectureAndThreeRoots) {
  const char *path = std::getenv("LOOMIFY_TEST_GGUF");
  if (!path) GTEST_SKIP() << "requires explicit immutable Llama GGUF gate";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"), &std::fclose);
  ASSERT_TRUE(file);
  ASSERT_EQ(std::setvbuf(file.get(), nullptr, _IONBF, 0), 0);
  model::ModelDescriptor descriptor;
  std::string error;
  ASSERT_TRUE(model::ExtractLlamaConfiguration(file.get(), &descriptor, &error)) << error;
  file.reset();
  ASSERT_EQ(descriptor.layers.size(), 28u);
  for (size_t count : {1u, 2u, 28u}) {
    auto subset = descriptor;
    subset.layers.resize(count);
    for (uint64_t rows : {1u, 128u}) {
      SCOPED_TRACE(::testing::Message() << "layers=" << count << " rows=" << rows);
      Build(subset, rows, count == 28 && rows == 128);
    }
  }
}
TEST_F(ConfiguredLlamaModelTest, InvalidAdmissionLeavesResultUnchanged) {
  LinkedConfiguration missing;
  ConfiguredLlamaModel output;
  output.layout.cache_base = 42;
  loomc_result_t *result = nullptr;
  auto status = BuildConfiguredLlamaModel(context, workspace, missing,
      MakeLlamaLinkedProviders(root), &output, &result);
  EXPECT_FALSE(loomc_status_is_ok(status));
  loomc_status_free(status);
  EXPECT_FALSE(output.module);
  EXPECT_EQ(output.layout.cache_base, 42u);
  EXPECT_EQ(result, nullptr);
}
} // namespace
} // namespace hrx::loomify_gguf
