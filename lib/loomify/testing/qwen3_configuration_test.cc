// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/qwen3_fixture.h"
#include "tools/model/gguf_catalog.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <map>
#ifndef LOOMIFY_QWEN3_REAL_ASSETS
#include "lib/loomify/linking.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include <filesystem>
#include <sstream>
#endif
namespace hrx::loomify_gguf {
namespace {
#ifndef LOOMIFY_QWEN3_REAL_ASSETS
TEST(Qwen3ModelRoot, TiedAndIndependentHeadsUseTheirActualParameter) {
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
  for (bool tied : {false, true}) {
    for (uint64_t rows : {1u, 8u}) {
      SCOPED_TRACE(tied);
      SCOPED_TRACE(rows);
      const auto descriptor = testing::Qwen3ModelFacts(1024, 1, tied);
      ConfigurationFacts facts;
      std::string error;
      ASSERT_TRUE(DeriveConfiguration(descriptor, "root-fixture", {rows, 128},
                                      &facts, &error))
          << error;
      const auto configuration =
          facts.source +
          "config.def @qwen3.publication.end_text_id = 2 : i32\n"
          "config.def @qwen3.publication.end_turn_id = 4 : i32\n";
      for (bool bytecode : {false, true}) {
        SCOPED_TRACE(bytecode);
        const auto product =
            root / "lib/architectures/qwen3" /
            (bytecode ? "implementation.loombc" : "implementation_text.loom");
        loomc_module_t *raw_module = nullptr;
        ASSERT_TRUE(LinkConfiguredProducts(
            context.get(), workspace.get(), configuration,
            {product.string(),
             (root / "lib/platforms/amdgpu/gfx11/targets.loom").string(),
             (root /
              "lib/platforms/amdgpu/gfx11/projection/implementation.loombc")
                 .string()},
            {"generated.qwen3_epoch"}, true, &raw_module, &error))
            << error;
        std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module(
            raw_module, loomc_module_release);
        loomc_source_t *raw_text = nullptr;
        ASSERT_TRUE(ok(loomc_module_serialize_text_to_source(
            module.get(), nullptr, loomc_allocator_system(), &raw_text)));
        std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)>
            printed(raw_text, loomc_source_release);
        const auto bytes = loomc_source_contents(printed.get());
        const std::string text(reinterpret_cast<const char *>(bytes.data),
                               bytes.data_length);
        // Follow the actual head launch's weight SSA operand to its parameter
        // definition; merely finding both parameter names would miss a
        // hardcoded embedding head. The head is the final compatible-Q6
        // projection.
        const auto launch =
            text.rfind("kernel.launch @ggml_linear_q6k_q8_1_x4_compat[");
        ASSERT_NE(launch, std::string::npos);
        const auto args = text.find("](", launch);
        ASSERT_NE(args, std::string::npos);
        std::istringstream arguments(
            text.substr(args + 2, text.find(')', args) - args - 2));
        std::string weight;
        for (int i = 0; i <= 8; ++i)
          ASSERT_TRUE(bool(std::getline(arguments, weight, ',')));
        weight.erase(0, weight.find_first_not_of(" \t"));
        weight.erase(weight.find_last_not_of(" \t") + 1);
        const auto definition = text.find(weight + " = command.parameter ");
        ASSERT_NE(definition, std::string::npos) << weight;
        const auto parameter =
            text.substr(definition, text.find('\n', definition) - definition);
        EXPECT_NE(parameter.find(tied ? ", \"token_embd.weight\""
                                      : ", \"output.weight\""),
                  std::string::npos)
            << parameter;
      }
    }
  }
}
#endif

TEST(Qwen3Configuration, PreservesTiedAndIndependentOutputBindings) {
  for (bool tied : {false, true}) {
    const auto descriptor = testing::Qwen3ModelFacts(5120, 1, tied);
    ConfigurationFacts facts;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(descriptor, "distinct-head-asset", {},
                                    &facts, &error))
        << error;
    EXPECT_NE(facts.source.find("@qwen3.model.tie_word_embeddings = " +
                                std::to_string(tied) + " : index\n"),
              std::string::npos);
    const auto &expected = descriptor.globals.back().tensor;
    const auto output = std::find_if(
        facts.parameters.begin(), facts.parameters.end(),
        [](const auto &binding) { return binding.role == "output"; });
    ASSERT_NE(output, facts.parameters.end());
    EXPECT_FALSE(output->layer);
    EXPECT_EQ(output->source_asset, "distinct-head-asset");
    EXPECT_EQ(output->tensor.name,
              tied ? "token_embd.weight" : "output.weight");
    EXPECT_EQ(output->tensor.absolute_offset, expected.absolute_offset);
    EXPECT_EQ(output->tensor.byte_size, expected.byte_size);
    EXPECT_EQ(output->tensor.rank, 2u);
    EXPECT_EQ(output->tensor.normalized_extents, expected.normalized_extents);
    EXPECT_EQ(output->tensor.storage_type, 14u);
    EXPECT_EQ(output->tensor.absolute_offset ==
                  descriptor.globals.front().tensor.absolute_offset,
              tied);
  }
}

TEST(Qwen3Configuration, RejectsMalformedIndependentHeadsTransactionally) {
  for (int mutation = 0; mutation < 6; ++mutation) {
    auto descriptor = testing::Qwen3ModelFacts(1024, 1, false);
    auto &head = descriptor.globals.back().tensor;
    if (mutation == 0)
      descriptor.globals.pop_back();
    if (mutation == 1)
      head.storage_type = 12;
    if (mutation == 2)
      head.normalized_extents[1]--;
    if (mutation == 3)
      head.byte_size--;
    if (mutation == 4)
      head.absolute_offset = uint64_t(INT64_MAX);
    if (mutation == 5)
      head.name = "token_embd.weight";
    ConfigurationFacts facts;
    facts.source = "unchanged";
    facts.parameters.push_back({"sentinel", {}, "sentinel", {}});
    std::string error;
    EXPECT_FALSE(DeriveConfiguration(descriptor, "asset", {}, &facts, &error))
        << mutation;
    EXPECT_EQ(facts.source, "unchanged");
    ASSERT_EQ(facts.parameters.size(), 1u);
    EXPECT_EQ(facts.parameters.front().role, "sentinel");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen3Configuration,
     ClassicQ4UsesActualFormatsAndIndependentE2560QueryWidth) {
  auto model = testing::Qwen3ModelFacts(2560);
  uint64_t next = 4096;
  std::map<std::string, model::TensorRecord> aliases;
  auto relocate = [&](model::TensorRecord &tensor) {
    const auto existing = aliases.find(tensor.name);
    if (existing != aliases.end()) {
      tensor = existing->second;
      return;
    }
    ASSERT_TRUE(model::EncodedTensorByteSize(
        tensor.storage_type, tensor.normalized_extents[0],
        tensor.normalized_extents[1], &tensor.byte_size));
    tensor.absolute_offset = next;
    next += (tensor.byte_size + 255) / 256 * 256;
    aliases.emplace(tensor.name, tensor);
  };
  for (auto &binding : model.globals)
    relocate(binding.tensor);
  for (auto &layer : model.layers) {
    layer.query_heads = 32;
    layer.feed_forward_width = 9728;
    for (auto &binding : layer.tensors) {
      auto &tensor = binding.tensor;
      if (tensor.storage_type != 0) {
        tensor.storage_type = binding.role == "ffn_down" ? 3 : 2;
        if (binding.role == "query")
          tensor.normalized_extents[1] = 4096;
        if (binding.role == "attention_output")
          tensor.normalized_extents[0] = 4096;
        if (binding.role == "ffn_gate" || binding.role == "ffn_up")
          tensor.normalized_extents[1] = 9728;
        if (binding.role == "ffn_down")
          tensor.normalized_extents[0] = 9728;
      }
      relocate(tensor);
    }
  }
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(
      DeriveConfiguration(model, "classic-q4", {17, 512}, &facts, &error))
      << error;
  EXPECT_NE(facts.source.find("@qwen3.model.hidden_size = 2560"),
            std::string::npos);
  EXPECT_NE(facts.source.find("@qwen3.model.query_size = 4096"),
            std::string::npos);
  EXPECT_NE(
      facts.source.find("@ggml.linear_q4_0_q8_1_x4.output_capacity = 9728"),
      std::string::npos);
  EXPECT_NE(
      facts.source.find("@ggml.linear_q4_1_q8_1_x4.output_capacity = 2560"),
      std::string::npos);
  for (const auto &binding : facts.parameters) {
    if (!binding.layer)
      continue;
    if (binding.role == "ffn_down")
      EXPECT_EQ(binding.tensor.storage_type, 3u);
    else if (binding.tensor.storage_type != 0)
      EXPECT_EQ(binding.tensor.storage_type, 2u);
  }
}

TEST(Qwen3Configuration, IndependentQueryWidthAndDataOnlyBindings) {
  for (const auto hidden : {1024u, 1536u, 2048u}) {
    const auto m = testing::Qwen3ModelFacts(hidden);
    ConfigurationFacts f;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(m, "asset", {17, 513}, &f, &error))
        << error;
    EXPECT_NE(
        f.source.find("@qwen3.model.hidden_size = " + std::to_string(hidden)),
        std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.model.query_size = 2048"),
              std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.model.vocabulary_size = 151936"),
              std::string::npos);
    EXPECT_NE(f.source.find("@qwen3.attention.pool_capacity = 576"),
              std::string::npos);
    EXPECT_EQ(f.source.find("command."), std::string::npos);
    EXPECT_EQ(f.source.find("kernel."), std::string::npos);
    EXPECT_EQ(f.parameters.size(), 3 + 11 * m.layers.size());
    const std::vector<std::string> globals{"token_embedding", "output",
                                           "output_norm"};
    const std::vector<std::string> roles{
        "input_norm", "post_norm", "query_norm", "key_norm",
        "query",      "key",       "value",      "attention_output",
        "ffn_gate",   "ffn_up",    "ffn_down"};
    for (size_t i = 0; i < f.parameters.size(); ++i) {
      const auto &binding = f.parameters[i];
      if (i < globals.size()) {
        EXPECT_EQ(binding.role, globals[i]);
        EXPECT_FALSE(binding.layer);
      } else {
        EXPECT_EQ(binding.role, roles[(i - globals.size()) % roles.size()]);
        EXPECT_EQ(binding.layer, (i - globals.size()) / roles.size());
      }
    }
    for (const auto &p : f.parameters) {
      const auto &source = p.layer ? m.layers[*p.layer].tensors : m.globals;
      const auto t =
          std::find_if(source.begin(), source.end(),
                       [&](const auto &b) { return b.role == p.role; });
      ASSERT_NE(t, source.end());
      EXPECT_EQ(p.tensor.name, t->tensor.name);
      EXPECT_EQ(p.source_asset, "asset");
      EXPECT_EQ(p.tensor.absolute_offset, t->tensor.absolute_offset);
      EXPECT_EQ(p.tensor.byte_size, t->tensor.byte_size);
      EXPECT_EQ(p.tensor.storage_type, t->tensor.storage_type);
      EXPECT_EQ(p.tensor.normalized_extents, t->tensor.normalized_extents);
    }
  }
}
TEST(Qwen3Configuration,
     RejectsUnsupportedOrInconsistentInputsWithoutReplacingOutput) {
  for (int mutation = 0; mutation < 12; ++mutation) {
    auto m = testing::Qwen3ModelFacts();
    if (mutation == 0)
      m.layers[0].tensors.erase(m.layers[0].tensors.begin() + 2);
    if (mutation == 1)
      m.layers[0].tensors[4].tensor.normalized_extents[1] = 1024;
    if (mutation == 2)
      m.layers[0].query_heads = 8;
    if (mutation == 3)
      m.layers[0].tensors[8].tensor.storage_type = 14;
    if (mutation == 4)
      m.rope.width = 64;
    if (mutation == 5)
      m.globals.back().tensor.storage_type = 12;
    if (mutation == 6)
      m.layers[1].attention_window = 512;
    if (mutation == 7)
      m.embedding_width = 2432;
    if (mutation == 8)
      m.layers[0].tensors[2].tensor.storage_type = 14;
    if (mutation == 9)
      m.layers[0].tensors[3].tensor.normalized_extents[0] = 256;
    if (mutation == 10)
      m.layers[0].tensors[2].tensor.byte_size -= 4;
    if (mutation == 11)
      m.layers[0].tensors.erase(m.layers[0].tensors.begin() + 3);
    ConfigurationFacts out;
    out.source = "unchanged";
    std::string error;
    EXPECT_FALSE(DeriveConfiguration(m, "asset", {}, &out, &error)) << mutation;
    EXPECT_EQ(out.source, "unchanged");
    EXPECT_FALSE(error.empty());
  }
}
TEST(Qwen3Configuration, MixedLayerTablesAndNeutralPreparationFacts) {
  auto m = testing::Qwen3ModelFacts(1024, 6);
  // Match the real Q4/Q6 transitions without making the fixture's filename a
  // selector.
  for (auto &layer : m.layers)
    for (auto &binding : layer.tensors)
      if (binding.role == "value" || binding.role == "ffn_down") {
        binding.tensor.storage_type =
            layer.index == 2 || layer.index == 3 || layer.index == 4 ? 12 : 14;
        ASSERT_TRUE(model::EncodedTensorByteSize(
            binding.tensor.storage_type, binding.tensor.normalized_extents[0],
            binding.tensor.normalized_extents[1], &binding.tensor.byte_size));
      }
  ConfigurationFacts out;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(m, "fixture", {17, 513}, &out, &error))
      << error;
  for (const auto &[name, value] :
       std::vector<std::pair<const char *, uint64_t>>{{"hidden_size", 1024},
                                                      {"layer_count", 6},
                                                      {"kv_size", 1024},
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
  EXPECT_EQ(out.source.find("@qwen3.greedy_argmax.output_capacity"),
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
#ifdef LOOMIFY_QWEN3_REAL_ASSETS
TEST(Qwen3ConfigurationRealAssets, ReadsActualCheckpointAndAllNamedRanges) {
  const char *path = std::getenv("HRX_QWEN3_GGUF");
  ASSERT_NE(path, nullptr) << "explicit immutable Qwen3 GGUF required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file);
  std::setvbuf(file.get(), nullptr, _IONBF, 0);
  model::GgufCatalog catalog;
  model::ModelDescriptor m;
  std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
  ASSERT_EQ(m.family, "qwen3");
  ASSERT_EQ(m.layers.size(), 28u);
  EXPECT_EQ(m.embedding_width, 1024u);
  EXPECT_EQ(m.layers.front().query_heads * m.key_head_width, 2048u);
  ConfigurationFacts out;
  ASSERT_TRUE(DeriveConfiguration(m, path, {128, 512}, &out, &error)) << error;
  EXPECT_EQ(out.parameters.size(), 311u);
  for (const auto &binding : out.parameters) {
    const auto *tensor = model::FindTensor(catalog, binding.tensor.name);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(binding.tensor.absolute_offset, tensor->absolute_offset);
    EXPECT_EQ(binding.tensor.byte_size, tensor->byte_size);
    EXPECT_EQ(binding.tensor.storage_type, tensor->storage_type);
  }
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
