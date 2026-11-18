// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/qwen2_fixture.h"
#include "tools/model/tensor_names.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#ifndef LOOMIFY_QWEN2_REAL_ASSETS
#include "lib/loomify/linking.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include <filesystem>
#endif
namespace hrx::loomify_gguf {
namespace {
#ifndef LOOMIFY_QWEN2_REAL_ASSETS
TEST(Qwen2ModelRoot, TiedQ8AndQ6HeadsMaterializeTheirActualPolicies) {
  auto ok = [](loomc_status_t status) {
    const bool success = loomc_status_is_ok(status);
    if (!success) {
      const auto message = loomc_status_message(status);
      ADD_FAILURE() << std::string(message.data, message.size);
    }
    loomc_status_free(status);
    return success;
  };
  loomc_target_environment_t *raw_environment = nullptr;
  ASSERT_TRUE(ok(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &raw_environment)));
  std::unique_ptr<loomc_target_environment_t,
                  decltype(&loomc_target_environment_release)>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *raw_context = nullptr;
  ASSERT_TRUE(ok(
      loomc_context_create(&options, loomc_allocator_system(), &raw_context)));
  std::unique_ptr<loomc_context_t, decltype(&loomc_context_release)> context(
      raw_context, loomc_context_release);
  loomc_workspace_t *raw_workspace = nullptr;
  ASSERT_TRUE(ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                        &raw_workspace)));
  std::unique_ptr<loomc_workspace_t, decltype(&loomc_workspace_release)>
      workspace(raw_workspace, loomc_workspace_release);
  const char *srcdir = std::getenv("TEST_SRCDIR");
  const char *workspace_name = std::getenv("TEST_WORKSPACE");
  ASSERT_TRUE(srcdir && workspace_name);
  const auto root = std::filesystem::path(srcdir) / workspace_name;
  for (uint32_t tag : {8u, 14u}) {
    SCOPED_TRACE(tag);
    auto descriptor = testing::qwen2::ModelFacts(1536, 1, 3072);
    uint64_t next = 4096;
    std::map<std::string, model::TensorRecord> aliases;
    auto place = [&](model::TensorRecord &tensor) {
      const auto found = aliases.find(tensor.name);
      if (found != aliases.end()) {
        tensor = found->second;
        return;
      }
      if (tensor.name == "token_embd.weight") {
        tensor.storage_type = tag;
        EXPECT_TRUE(model::EncodedTensorByteSize(
            tag, 1536, descriptor.vocabulary_size, &tensor.byte_size));
      }
      tensor.absolute_offset = next;
      next += (tensor.byte_size + 255) / 256 * 256;
      aliases.emplace(tensor.name, tensor);
    };
    for (auto &binding : descriptor.globals)
      place(binding.tensor);
    for (auto &layer : descriptor.layers)
      for (auto &binding : layer.tensors)
        place(binding.tensor);
    ConfigurationFacts facts;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(descriptor, "root-fixture", {1, 128},
                                    &facts, &error))
        << error;
    // Constructed publication IDs complete the same public epoch entrypoint
    // used by exported models; numerical head selection is independent of them.
    const auto configuration =
        facts.source + "config.def @qwen2.publication.end_text_id = 2 : i32\n"
                       "config.def @qwen2.publication.end_turn_id = 4 : i32\n";
    for (bool bytecode : {false, true}) {
      SCOPED_TRACE(bytecode);
      const auto product =
          root / "lib/architectures/qwen2" /
          (bytecode ? "implementation.loombc" : "implementation_text.loom");
      loomc_module_t *raw_module = nullptr;
      ASSERT_TRUE(LinkConfiguredProducts(
          context.get(), workspace.get(), configuration,
          {product.string(),
           (root / "lib/platforms/amdgpu/gfx11/targets.loom").string(),
           (root /
            "lib/platforms/amdgpu/gfx11/projection/implementation.loombc")
               .string()},
          {"generated.qwen2_epoch"}, true, &raw_module, &error))
          << error;
      std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module(
          raw_module, loomc_module_release);
      loomc_source_t *raw_text = nullptr;
      ASSERT_TRUE(ok(loomc_module_serialize_text_to_source(
          module.get(), nullptr, loomc_allocator_system(), &raw_text)));
      std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)> printed(
          raw_text, loomc_source_release);
      const auto bytes = loomc_source_contents(printed.get());
      const std::string text(reinterpret_cast<const char *>(bytes.data),
                             bytes.data_length);
      EXPECT_NE(
          text.find(tag == 8
                        ? "kernel.launch @ggml_linear_q8_0_q8_1_x4["
                        : "kernel.launch @ggml_linear_q6k_q8_1_x4_compat["),
          std::string::npos);
      if (tag == 8)
        EXPECT_EQ(text.find("kernel.launch @ggml_linear_q6k_q8_1_x4_compat["),
                  std::string::npos);
    }
  }
}
#endif

const model::TensorRecord *
Role(const std::vector<model::TensorBinding> &bindings,
     const std::string &role) {
  auto it = std::find_if(bindings.begin(), bindings.end(),
                         [&](const auto &b) { return b.role == role; });
  return it == bindings.end() ? nullptr : &it->tensor;
}
TEST(Qwen2Descriptor, DefaultsBiasRolesMixedFormatsAndTiedHead) {
  auto catalog = testing::qwen2::Catalog();
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
  EXPECT_EQ(catalog.tensors.size(), 338u);
  EXPECT_EQ(m.family, "qwen2");
  EXPECT_EQ(m.embedding_width, 1536u);
  EXPECT_EQ(m.vocabulary_size, 151936u);
  EXPECT_EQ(m.training_context, 32768u);
  EXPECT_EQ(m.key_head_width, 128u);
  EXPECT_EQ(m.value_head_width, 128u);
  EXPECT_EQ(m.rope.width, 128u);
  EXPECT_EQ(m.rope.base, 1e6);
  EXPECT_EQ(m.rope.scaling_factor, 1);
  EXPECT_TRUE(m.output_tied_to_embedding);
  EXPECT_FALSE(m.ssm);
  ASSERT_EQ(m.layers.size(), 28u);
  for (const auto &layer : m.layers) {
    ASSERT_EQ(layer.tensors.size(), 12u);
    EXPECT_EQ(layer.query_heads, 12u);
    EXPECT_EQ(layer.kv_heads, 2u);
    EXPECT_EQ(layer.feed_forward_width, 8960u);
    EXPECT_EQ(layer.kind, model::LayerKind::kAttention);
    EXPECT_EQ(layer.attention_ordinal, layer.index);
    EXPECT_FALSE(layer.recurrent_ordinal);
    const auto *q = Role(layer.tensors, "query_bias");
    const auto *k = Role(layer.tensors, "key_bias");
    const auto *v = Role(layer.tensors, "value_bias");
    ASSERT_TRUE(q && k && v);
    EXPECT_EQ(q->normalized_extents[0], 1536);
    EXPECT_EQ(k->normalized_extents[0], 256);
    EXPECT_EQ(v->normalized_extents[0], 256);
    EXPECT_EQ(q->storage_type, 0u);
    EXPECT_EQ(Role(layer.tensors, "query_norm"), nullptr);
    EXPECT_EQ(Role(layer.tensors, "key_norm"), nullptr);
  }
  // Explicit transition witnesses; the filename is not an encoding selector.
  for (const auto &[layer, tag] : std::vector<std::pair<size_t, uint32_t>>{
           {1, 14}, {2, 12}, {4, 12}, {5, 14}}) {
    EXPECT_EQ(Role(m.layers[layer].tensors, "value")->storage_type, tag);
    EXPECT_EQ(Role(m.layers[layer].tensors, "ffn_down")->storage_type, tag);
  }
}
TEST(Qwen2Descriptor, RejectsMissingOrWrongBiasAndUnknownGraphFeatures) {
  for (int mutation = 0; mutation < 8; ++mutation) {
    auto catalog = testing::qwen2::Catalog();
    auto bias = std::find_if(
        catalog.tensors.begin(), catalog.tensors.end(),
        [](const auto &t) { return t.name == "blk.0.attn_k.bias"; });
    ASSERT_NE(bias, catalog.tensors.end());
    if (mutation == 0)
      catalog.tensors.erase(bias);
    if (mutation == 1)
      bias->normalized_extents[0] = 128;
    if (mutation == 2) {
      auto extra = *bias;
      extra.name = "blk.0.attn_output.bias";
      catalog.tensors.push_back(extra);
    }
    if (mutation == 3) {
      auto extra = *bias;
      extra.name = "blk.0.attn_q_norm.weight";
      catalog.tensors.push_back(extra);
    }
    if (mutation == 4) {
      auto extra = *bias;
      extra.name = "blk.0.attn_qkv.weight";
      catalog.tensors.push_back(extra);
    }
    if (mutation == 5)
      catalog.metadata.push_back(
          {"qwen2.attention.sliding_window", 4, uint64_t{128}});
    if (mutation == 6)
      catalog.metadata.push_back(
          {"qwen2.rope.scaling.type", 8, std::string("yarn")});
    if (mutation == 7)
      catalog.metadata.push_back(
          {"qwen2.rope.dimension_count", 4, uint64_t{64}});
    model::ModelDescriptor out;
    out.name = "unchanged";
    std::string error;
    EXPECT_FALSE(model::LoadModelDescriptor(catalog, &out, &error)) << mutation;
    EXPECT_EQ(out.name, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen2Configuration, DataOnlyFactsAndAll339BindingsFollowInputs) {
  for (const auto hidden : {1536u, 2048u}) {
    const auto m = testing::qwen2::ModelFacts(hidden);
    ConfigurationFacts out;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(m, "fixture.gguf", {17, 513}, &out, &error))
        << error;
    EXPECT_EQ(out.parameters.size(), 339u);
    const std::vector<std::string> globals{"token_embedding", "output",
                                           "output_norm"};
    const std::vector<std::string> roles{
        "input_norm",       "post_norm", "query_bias", "key_bias",
        "value_bias",       "query",     "key",        "value",
        "attention_output", "ffn_gate",  "ffn_up",     "ffn_down"};
    for (size_t i = 0; i < out.parameters.size(); ++i) {
      const auto &binding = out.parameters[i];
      if (i < globals.size()) {
        EXPECT_EQ(binding.role, globals[i]);
        EXPECT_FALSE(binding.layer);
      } else {
        EXPECT_EQ(binding.role, roles[(i - globals.size()) % roles.size()]);
        EXPECT_EQ(binding.layer, (i - globals.size()) / roles.size());
      }
    }
    for (const auto &binding : out.parameters) {
      const auto *tensor =
          Role(binding.layer ? m.layers[*binding.layer].tensors : m.globals,
               binding.role);
      ASSERT_NE(tensor, nullptr);
      EXPECT_EQ(binding.source_asset, "fixture.gguf");
      EXPECT_EQ(binding.tensor.name, tensor->name);
      EXPECT_EQ(binding.tensor.absolute_offset, tensor->absolute_offset);
      EXPECT_EQ(binding.tensor.byte_size, tensor->byte_size);
      EXPECT_EQ(binding.tensor.storage_type, tensor->storage_type);
      EXPECT_EQ(binding.tensor.normalized_extents, tensor->normalized_extents);
    }
    EXPECT_NE(
        out.source.find("@qwen2.model.hidden_size = " + std::to_string(hidden)),
        std::string::npos);
    EXPECT_NE(
        out.source.find("@qwen2.model.query_size = " + std::to_string(hidden)),
        std::string::npos);
    EXPECT_NE(out.source.find("@qwen2.model.kv_size = 256 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@qwen2.model.layer_count = 28 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@qwen2.attention.pool_capacity = 576 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.serving.hidden_size = " +
                              std::to_string(hidden)),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.serving.kv_size = 256 : index"),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.config.layer_attention"),
              std::string::npos);
    EXPECT_NE(out.source.find("@loomify.config.layer_ffn"), std::string::npos);
    EXPECT_NE(out.source.find(
                  "@qwen38.greedy_argmax.output_capacity = 151936 : index"),
              std::string::npos);
    EXPECT_EQ(out.source.find("qwen28."), std::string::npos);
    EXPECT_EQ(out.source.find("command."), std::string::npos);
    EXPECT_EQ(out.source.find("kernel."), std::string::npos);
  }
  ConfigurationFacts shortened;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::qwen2::ModelFacts(1536, 2, 9216),
                                  "fixture", {}, &shortened, &error))
      << error;
  EXPECT_EQ(shortened.parameters.size(), 27u);
  EXPECT_NE(shortened.source.find("@qwen2.model.layer_count = 2 : index"),
            std::string::npos);
}
TEST(Qwen2Configuration,
     RejectsMalformedConstructedBindingsWithoutReplacingOutput) {
  for (int mutation = 0; mutation < 11; ++mutation) {
    auto m = testing::qwen2::ModelFacts();
    auto bias =
        std::find_if(m.layers[0].tensors.begin(), m.layers[0].tensors.end(),
                     [](const auto &b) { return b.role == "query_bias"; });
    ASSERT_NE(bias, m.layers[0].tensors.end());
    if (mutation == 0)
      bias->tensor.storage_type = 14;
    if (mutation == 1)
      bias->tensor.absolute_offset = uint64_t(INT64_MAX);
    if (mutation == 2)
      bias->tensor.name = "blk.1.attn_q.bias";
    if (mutation == 3)
      m.layers[0].tensors.push_back(*bias);
    if (mutation == 4)
      m.output_tied_to_embedding = false;
    if (mutation == 5)
      m.layers[0].attention_window = 256;
    if (mutation == 6)
      m.rope.scaling_factor = 2;
    if (mutation == 7) {
      auto norm = *bias;
      norm.role = "query_norm";
      m.layers[0].tensors.push_back(norm);
    }
    if (mutation == 8)
      bias->tensor.byte_size -= 4;
    if (mutation == 9)
      m.layers[0].tensors.erase(bias);
    if (mutation == 10)
      bias->tensor.normalized_extents[0] = 128;
    ConfigurationFacts out;
    out.source = "unchanged";
    std::string error;
    EXPECT_FALSE(DeriveConfiguration(m, "fixture", {}, &out, &error))
        << mutation;
    EXPECT_EQ(out.source, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen2Configuration, MixedLayerTablesAndNeutralPreparationFacts) {
  auto m = testing::qwen2::ModelFacts(1536, 6);
  ConfigurationFacts out;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {17, 513}, &out, &error))
      << error;
  for (const auto &[name, value] :
       std::vector<std::pair<const char *, uint64_t>>{{"hidden_size", 1536},
                                                      {"layer_count", 6},
                                                      {"kv_size", 256},
                                                      {"history_capacity", 129},
                                                      {"token_capacity", 17},
                                                      {"cache_capacity", 513},
                                                      {"pool_capacity", 576},
                                                      {"cache_base", 1284},
                                                      {"map_origin", 2048}}) {
    const std::string type =
        std::string(name) == "cache_base" || std::string(name) == "map_origin"
            ? "offset"
            : "index";
    EXPECT_NE(out.source.find("config.def @loomify.serving." +
                              std::string(name) + " = " +
                              std::to_string(value) + " : " + type + "\n"),
              std::string::npos)
        << name;
  }
  EXPECT_NE(out.source.find(
                "@qwen38.greedy_argmax.output_capacity = 151936 : index\n"),
            std::string::npos);
  EXPECT_EQ(out.source.find("@qwen2.greedy_argmax.output_capacity"),
            std::string::npos);
  // Inspect every tuple element, including encoded byte extents, in both
  // lookups.
  const auto check_row = [&](const std::string &lookup, size_t layer,
                             const std::vector<uint64_t> &values) {
    const auto start = out.source.find("@loomify.config." + lookup);
    ASSERT_NE(start, std::string::npos);
    const auto text =
        out.source.substr(start, out.source.find("\n}\n", start) - start);
    const auto variable = layer + 1 == m.layers.size()
                              ? std::string("%v0_")
                              : "%c" + std::to_string(layer) + "_";
    for (size_t j = 0; j < values.size(); ++j)
      EXPECT_NE(text.find(variable + std::to_string(j) + " = index.constant " +
                          std::to_string(values[j]) + " : index\n"),
                std::string::npos)
          << lookup << " layer=" << layer << " field=" << j;
  };
  for (const auto &layer : m.layers) {
    const auto tensor =
        [&](const std::string &role) -> const model::TensorRecord & {
      return std::find_if(layer.tensors.begin(), layer.tensors.end(),
                          [&](const auto &b) { return b.role == role; })
          ->tensor;
    };
    const auto &q = tensor("query"), &k = tensor("key"), &v = tensor("value"),
               &o = tensor("attention_output"), &g = tensor("ffn_gate"),
               &u = tensor("ffn_up"), &d = tensor("ffn_down");
    const uint32_t varied =
        layer.index == 2 || layer.index == 3 || layer.index == 4 ? 12 : 14;
    EXPECT_EQ(v.storage_type, varied);
    EXPECT_EQ(d.storage_type, varied);
    check_row("layer_attention", layer.index,
              {q.storage_type, k.storage_type, v.storage_type, o.storage_type,
               q.byte_size, k.byte_size, v.byte_size, o.byte_size});
    check_row("layer_ffn", layer.index,
              {layer.feed_forward_width, g.storage_type, u.storage_type,
               d.storage_type, g.byte_size, u.byte_size, d.byte_size});
  }
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {256, 513}, &out, &error))
      << error;
  EXPECT_NE(
      out.source.find("@loomify.serving.history_capacity = 256 : index\n"),
      std::string::npos);
  EXPECT_NE(out.source.find("@loomify.serving.cache_base = 1792 : offset\n"),
            std::string::npos);
}
#ifdef LOOMIFY_QWEN2_REAL_ASSETS
TEST(Qwen2ConfigurationRealAssets,
     Actual338TensorsAllRangesAndFormatTransitions) {
  const char *path = std::getenv("LOOMIFY_TEST_QWEN2_GGUF");
  ASSERT_NE(path, nullptr) << "explicit immutable Qwen2 GGUF required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file);
  std::setvbuf(file.get(), nullptr, _IONBF, 0);
  model::GgufCatalog catalog;
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  EXPECT_LE(catalog.bytes_read, catalog.header_end);
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
  ASSERT_EQ(catalog.tensors.size(), 338u);
  ASSERT_EQ(m.layers.size(), 28u);
  EXPECT_EQ(m.embedding_width, 1536u);
  EXPECT_EQ(m.vocabulary_size, 151936u);
  EXPECT_TRUE(m.output_tied_to_embedding);
  ConfigurationFacts out;
  ASSERT_TRUE(DeriveConfiguration(m, path, {128, 512}, &out, &error)) << error;
  ASSERT_EQ(out.parameters.size(), 339u);
  std::map<uint32_t, size_t> physical_formats;
  for (const auto &tensor : catalog.tensors)
    ++physical_formats[tensor.storage_type];
  EXPECT_EQ(physical_formats[0], 141u);
  EXPECT_EQ(physical_formats[12], 168u);
  EXPECT_EQ(physical_formats[14], 29u);
  for (const auto &binding : out.parameters) {
    const auto *tensor = model::FindTensor(catalog, binding.tensor.name);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(binding.tensor.absolute_offset, tensor->absolute_offset);
    EXPECT_EQ(binding.tensor.byte_size, tensor->byte_size);
    EXPECT_EQ(binding.tensor.storage_type, tensor->storage_type);
    EXPECT_EQ(binding.tensor.rank, tensor->rank);
    EXPECT_EQ(binding.tensor.normalized_extents, tensor->normalized_extents);
  }
  for (const auto &[layer, tag] : std::vector<std::pair<size_t, uint32_t>>{
           {1, 14}, {2, 12}, {4, 12}, {5, 14}}) {
    EXPECT_EQ(Role(m.layers[layer].tensors, "value")->storage_type, tag);
    EXPECT_EQ(Role(m.layers[layer].tensors, "ffn_down")->storage_type, tag);
  }
}
#endif
TEST(Qwen2Configuration, IndependentGateAndUpUseTheirOwnEncodedRanges) {
  model::ModelDescriptor model;
  std::string error;
  ASSERT_TRUE(
      model::LoadModelDescriptor(testing::qwen2::Catalog(), &model, &error))
      << error;
  auto &layer = model.layers[1];
  auto up = std::find_if(
      layer.tensors.begin(), layer.tensors.end(),
      [](const auto &binding) { return binding.role == "ffn_up"; });
  ASSERT_NE(up, layer.tensors.end());
  up->tensor.storage_type = 13;
  ASSERT_TRUE(model::EncodedTensorByteSize(13, model.embedding_width,
                                           layer.feed_forward_width,
                                           &up->tensor.byte_size));
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(model, "mixed.gguf", {}, &facts, &error))
      << error;
  const auto actual = std::find_if(
      facts.parameters.begin(), facts.parameters.end(),
      [](const auto &p) { return p.role == "ffn_up" && p.layer == 1; });
  ASSERT_NE(actual, facts.parameters.end());
  EXPECT_EQ(actual->tensor.storage_type, 13u);
  EXPECT_EQ(actual->tensor.byte_size, up->tensor.byte_size);
  EXPECT_EQ(actual->tensor.absolute_offset, up->tensor.absolute_offset);
  EXPECT_EQ(Role(layer.tensors, "ffn_gate")->storage_type, 12u);
  ++up->tensor.byte_size;
  facts.source = "sentinel";
  EXPECT_FALSE(DeriveConfiguration(model, "bad", {}, &facts, &error));
  EXPECT_EQ(facts.source, "sentinel");
}
TEST(Qwen2Configuration, SharedFfnBindingRetainsDownFormatAdmission) {
  auto model = testing::qwen2::ModelFacts();
  auto &layer = model.layers[0];
  auto down = std::find_if(
      layer.tensors.begin(), layer.tensors.end(),
      [](const auto &binding) { return binding.role == "ffn_down"; });
  ASSERT_NE(down, layer.tensors.end());
  down->tensor.storage_type = 13;
  ASSERT_TRUE(model::EncodedTensorByteSize(13, layer.feed_forward_width,
                                           model.embedding_width,
                                           &down->tensor.byte_size));
  ConfigurationFacts facts;
  facts.source = "unchanged";
  std::string error;
  EXPECT_FALSE(DeriveConfiguration(model, "fixture", {}, &facts, &error));
  EXPECT_EQ(error,
            "unsupported tensor name, shape or encoding: " + down->tensor.name);
  EXPECT_EQ(facts.source, "unchanged");
}
} // namespace
} // namespace hrx::loomify_gguf
