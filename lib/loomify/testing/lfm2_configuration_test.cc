// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/lfm2_fixture.h"
#include "tools/model/tensor_names.h"
#include <algorithm>
#include <gtest/gtest.h>
namespace hrx::loomify_gguf {
namespace {
const model::TensorRecord *Role(const std::vector<model::TensorBinding> &b,
                                const std::string &role) {
  for (const auto &v : b)
    if (v.role == role)
      return &v.tensor;
  return nullptr;
}
model::MetadataRecord *Field(model::GgufCatalog &c, const char *key) {
  for (auto &f : c.metadata)
    if (f.key == key)
      return &f;
  return nullptr;
}
void CheckDirectory(const model::ModelDescriptor &m) {
  ASSERT_EQ(m.family, "lfm2");
  ASSERT_EQ(m.layers.size(), 16u);
  ASSERT_TRUE(m.short_convolution);
  EXPECT_EQ(m.short_convolution->kernel, 3u);
  EXPECT_EQ(m.short_convolution->history_rows, 2u);
  EXPECT_EQ(m.embedding_width, 2048u);
  EXPECT_EQ(m.key_head_width, 64u);
  EXPECT_EQ(m.vocabulary_size, 65536u);
  EXPECT_EQ(m.rope.width, 64u);
  EXPECT_EQ(m.rope.base, 1e6);
  EXPECT_EQ(m.rope.scaling_factor, 1);
  EXPECT_DOUBLE_EQ(m.rms_epsilon, 9.999999747378752e-6);
  EXPECT_TRUE(m.output_tied_to_embedding);
  ASSERT_NE(Role(m.globals, "output_norm"), nullptr);
  EXPECT_EQ(Role(m.globals, "output_norm")->name, "token_embd_norm.weight");
  ASSERT_NE(Role(m.globals, "output"), nullptr);
  EXPECT_EQ(Role(m.globals, "output")->name, "token_embd.weight");
  uint32_t a = 0, c = 0;
  size_t tensors =
      2; // Tied output is a second role for the same source tensor.
  for (uint32_t i = 0; i < 16; ++i) {
    const auto &l = m.layers[i];
    EXPECT_EQ(l.index, i);
    EXPECT_EQ(l.query_heads, 32u);
    EXPECT_EQ(l.feed_forward_width, 8192u);
    EXPECT_FALSE(l.recurrent_ordinal);
    EXPECT_EQ(l.attention_window, 0u);
    EXPECT_EQ(Role(l.tensors, "attention_post_norm"), nullptr);
    EXPECT_EQ(Role(l.tensors, "ffn_post_norm"), nullptr);
    ASSERT_NE(Role(l.tensors, "ffn_gate"), nullptr);
    EXPECT_EQ(Role(l.tensors, "ffn_gate")->storage_type, 12u);
    EXPECT_EQ(Role(l.tensors, "ffn_up")->storage_type, 12u);
    const bool down6 = i == 0 || i == 1 || i == 4 || i == 7 || i == 10 ||
                       i == 13 || i == 14 || i == 15;
    EXPECT_EQ(Role(l.tensors, "ffn_down")->storage_type, down6 ? 14u : 12u);
    const bool attention =
        i == 2 || i == 5 || i == 8 || i == 10 || i == 12 || i == 14;
    if (attention) {
      EXPECT_EQ(l.kind, model::LayerKind::kAttention);
      EXPECT_EQ(l.kv_heads, 8u);
      EXPECT_EQ(l.attention_ordinal, a++);
      EXPECT_FALSE(l.convolution_ordinal);
      EXPECT_EQ(Role(l.tensors, "value")->storage_type,
                i == 8 || i == 14 ? 14u : 12u);
      EXPECT_EQ(Role(l.tensors, "query_norm")->normalized_extents[0], 64);
    } else {
      EXPECT_EQ(l.kind, model::LayerKind::kShortConvolution);
      EXPECT_EQ(l.kv_heads, 0u);
      EXPECT_EQ(l.convolution_ordinal, c++);
      EXPECT_FALSE(l.attention_ordinal);
    }
    tensors += l.tensors.size();
  }
  EXPECT_EQ(a, 6u);
  EXPECT_EQ(c, 10u);
  EXPECT_EQ(tensors, 148u);
}
TEST(Lfm2Descriptor, ActualGeometryRolesAndCompactOrdinals) {
  auto catalog = testing::Lfm2Catalog();
  ASSERT_EQ(catalog.tensors.size(), 148u);
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
  CheckDirectory(m);
}
TEST(Lfm2Descriptor, RejectsMissingWrongShapeAndAdditionalGraphTensors) {
  const auto good = testing::Lfm2Catalog();
  for (int trial = 0; trial < 6; ++trial) {
    auto c = good;
    if (trial == 0)
      c.tensors.erase(c.tensors.begin() + 1); // Required final gamma.
    if (trial == 1)
      c.tensors[1].normalized_extents[0] = 1024;
    if (trial == 2)
      Field(c, "lfm2.attention.head_count_kv")->elements.pop_back();
    if (trial == 3)
      Field(c, "lfm2.attention.head_count_kv")->elements[2] = uint64_t(3);
    if (trial == 4) {
      auto t = c.tensors[1];
      t.name = "blk.0.post_attention_norm.weight";
      c.tensors.push_back(t);
    }
    if (trial == 5)
      c.metadata.push_back({"lfm2.attention.sliding_window", 4, uint64_t(128)});
    model::ModelDescriptor m;
    m.name = "preserve";
    std::string error;
    EXPECT_FALSE(model::LoadModelDescriptor(c, &m, &error)) << trial;
    EXPECT_EQ(m.name, "preserve");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Lfm2Configuration, ExactFormatsFinalGammaAndStateExtent) {
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(testing::Lfm2Catalog(), &m, &error));
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture.gguf", {8, 256}, &facts, &error))
      << error;
  for (const char *fragment :
       {"@lfm2.model.hidden_size = 2048 : index",
        "@lfm2.model.attention_layer_count = 6 : index",
        "@lfm2.model.head_size = 64 : index",
        "@lfm2.model.tied_output = 1 : index",
        "@lfm2.model.output_tag = 14 : index",
        "@lfm2.model.embedding_weight_bytes = 110100480 : index",
        "@lfm2.model.attention_scale = 1.250000000e-01",
        "@lfm2.shortconv.history_base = 3147012 : offset",
        "@loomify.serving.trailing_state_bytes = 163840 : index"})
    EXPECT_NE(facts.source.find(fragment), std::string::npos) << fragment;
  EXPECT_EQ(facts.source.find("post_rms"), std::string::npos);
  EXPECT_EQ(facts.source.find("rope_freqs.weight"), std::string::npos);
  EXPECT_EQ(facts.parameters.size(), 149u);
  EXPECT_EQ(std::count_if(facts.parameters.begin(), facts.parameters.end(),
                          [](const auto &p) {
                            return p.tensor.name == "token_embd_norm.weight";
                          }),
            1);
}
TEST(Lfm2Configuration, RejectsConstructedDescriptorErrorsTransactionally) {
  model::ModelDescriptor good;
  std::string error;
  ASSERT_TRUE(
      model::LoadModelDescriptor(testing::Lfm2Catalog(), &good, &error));
  for (int trial = 0; trial < 7; ++trial) {
    auto m = good;
    if (trial == 0)
      m.short_convolution->kernel = 7;
    if (trial == 1)
      m.layers[2].attention_ordinal = 1;
    if (trial == 2)
      m.globals[1].tensor.rank = 2;
    if (trial == 3)
      m.layers[0].tensors[0].tensor.storage_type = 14;
    if (trial == 4)
      m.layers[0].tensors[0].tensor.absolute_offset =
          m.globals[0].tensor.absolute_offset;
    if (trial == 5)
      m.layers[2].kv_heads = 4;
    if (trial == 6)
      --m.layers[0].tensors[2].tensor.byte_size;
    ConfigurationFacts facts;
    facts.source = "preserve";
    EXPECT_FALSE(
        DeriveConfiguration(m, "fixture.gguf", {8, 256}, &facts, &error))
        << trial;
    EXPECT_EQ(facts.source, "preserve");
    EXPECT_TRUE(facts.parameters.empty());
  }
}
} // namespace
} // namespace hrx::loomify_gguf

#ifdef LOOMIFY_LFM2_REAL_ASSETS
#include "lib/loomify/gguf_model_builder.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include <cstdlib>
#include <filesystem>
namespace hrx::loomify_gguf {
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
TEST(Lfm2Actual, BothCheckpointsNumericalOnlySourceAndBytecode) {
  const auto *runfiles = std::getenv("TEST_SRCDIR");
  const auto *workspace_name = std::getenv("TEST_WORKSPACE");
  const auto *temporary = std::getenv("TEST_TMPDIR");
  ASSERT_TRUE(runfiles && workspace_name && temporary);
  const auto root = std::filesystem::path(runfiles) / workspace_name;
  // A public numerical build has no preparation/control/text companion inputs.
  const auto products = std::filesystem::path(temporary) / "numerical-only";
  for (const auto *relative : {"lib/architectures/lfm2/implementation.loombc",
                               "lib/platforms/amdgpu/gfx11/targets.loombc"}) {
    const auto destination = products / relative;
    std::filesystem::create_directories(destination.parent_path());
    std::filesystem::copy_file(root / relative, destination);
  }
  const auto *retained = std::getenv("LOOMIFY_LFM2_OUTPUT_DIR");
  const auto output = retained ? std::filesystem::path(retained)
                               : std::filesystem::path(temporary);
  ASSERT_TRUE(std::filesystem::is_directory(output));
  for (const auto *key :
       {"LOOMIFY_TEST_LFM2_GGUF", "LOOMIFY_TEST_LFM25_GGUF"}) {
    const auto *asset = std::getenv(key);
    ASSERT_TRUE(asset && *asset) << "required actual cached fixture: " << key;
    SCOPED_TRACE(key);
    std::unique_ptr<GgufModelBuilder> builder;
    std::string error;
    ASSERT_TRUE(GgufModelBuilder::Open(asset, &builder, &error)) << error;
    CheckDirectory(builder->descriptor());
    GgufBuildOptions options;
    options.library_root = products.string();
    options.prefill_capacity = 4;
    options.context_capacity = 256;
    std::unique_ptr<GgufConfiguration> configuration;
    ASSERT_TRUE(builder->Configure(options, nullptr, &configuration, &error))
        << error;
    EXPECT_FALSE(configuration->has_package_configuration());
    EXPECT_TRUE(configuration->package_data_source().empty());
    for (const auto *fragment : {"@lfm2.publication.end_text_id = 2 : i32",
                                 "@lfm2.publication.end_turn_id = 7 : i32"})
      EXPECT_NE(configuration->configuration_source().find(fragment),
                std::string::npos);
    loomc_target_environment_t *environment = nullptr;
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment;
    loomc_context_options_t context_options{};
    context_options.next = &target;
    loomc_context_t *context = nullptr;
    ASSERT_TRUE(Ok(loomc_context_create(&context_options,
                                        loomc_allocator_system(), &context)));
    loomc_target_environment_release(environment);
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    std::unique_ptr<BuiltGgufModule> module;
    ASSERT_TRUE(configuration->BuildModule(context, workspace, &module, &error))
        << error;
    const std::string stem =
        std::string(key) == "LOOMIFY_TEST_LFM2_GGUF" ? "lfm2" : "lfm25";
    for (auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      const auto path =
          output /
          (stem + (format == LOOMC_SOURCE_FORMAT_TEXT ? ".loom" : ".loombc"));
      ASSERT_TRUE(module->WriteModule(path.string(), format, &error)) << error;
      loomc_module_t *reloaded = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
          context, workspace, loomc_make_cstring_view(path.c_str()), nullptr,
          loomc_allocator_system(), &reloaded, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result));
      loomc_result_release(result);
      ASSERT_NE(reloaded, nullptr);
      loomc_module_release(reloaded);
    }
    module.reset();
    loomc_workspace_release(workspace);
    loomc_context_release(context);
  }
}
} // namespace
} // namespace hrx::loomify_gguf
#endif
