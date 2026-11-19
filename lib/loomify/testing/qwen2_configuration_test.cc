// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/qwen2_fixture.h"
#include "tools/model/tensor_names.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#ifndef LOOMIFY_QWEN2_REAL_ASSETS
#include "lib/loomify/linking.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include <filesystem>
#include <sstream>
#endif
namespace hrx::loomify_gguf {
namespace {
struct HeadCase {
  uint32_t embedding_tag, output_tag;
  bool tied;
};
constexpr std::array<HeadCase, 5> kHeadCases{{
    {8, 8, true},
    {14, 14, true},
    {12, 14, false},
    {8, 14, false},
    {14, 8, false},
}};

model::ModelDescriptor HeadFacts(const HeadCase &head) {
  auto catalog = testing::qwen2::Catalog(1536, 1, 3072);
  auto embedding = std::find_if(
      catalog.tensors.begin(), catalog.tensors.end(),
      [](const auto &tensor) { return tensor.name == "token_embd.weight"; });
  EXPECT_NE(embedding, catalog.tensors.end());
  if (embedding == catalog.tensors.end())
    return {};
  embedding->storage_type = head.embedding_tag;
  if (!head.tied) {
    auto output = *embedding;
    output.name = "output.weight";
    output.storage_type = head.output_tag;
    catalog.tensors.push_back(output);
  }
  // Recompute physical ranges before loading the descriptor, so the reader
  // derives the tie and the two logical roles retain real encoded extents.
  uint64_t offset = 4096;
  for (auto &tensor : catalog.tensors) {
    EXPECT_TRUE(model::EncodedTensorByteSize(
        tensor.storage_type, tensor.normalized_extents[0],
        tensor.normalized_extents[1], &tensor.byte_size));
    tensor.absolute_offset = offset;
    offset += (tensor.byte_size + 255) / 256 * 256;
  }
  model::ModelDescriptor descriptor;
  std::string error;
  EXPECT_TRUE(model::LoadModelDescriptor(catalog, &descriptor, &error))
      << error;
  return descriptor;
}

#ifndef LOOMIFY_QWEN2_REAL_ASSETS
TEST(Qwen2ModelRoot, ActualHeadControlsPhysicalParameterEncodingAndPolicy) {
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
  for (const auto &head : kHeadCases) {
    SCOPED_TRACE(head.embedding_tag);
    SCOPED_TRACE(head.output_tag);
    SCOPED_TRACE(head.tied);
    const auto descriptor = HeadFacts(head);
    for (uint64_t rows : {1u, 128u}) {
      SCOPED_TRACE(rows);
      ConfigurationFacts facts;
      std::string error;
      ASSERT_TRUE(DeriveConfiguration(descriptor, "root-fixture", {rows, 128},
                                      &facts, &error))
          << error;
      // Constructed publication IDs complete the same public epoch entrypoint
      // used by exported models; numerical head selection is independent of
      // them.
      const auto configuration =
          facts.source +
          "config.def @qwen2.publication.end_text_id = 2 : i32\n"
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
        std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)>
            printed(raw_text, loomc_source_release);
        const auto bytes = loomc_source_contents(printed.get());
        const std::string text(reinterpret_cast<const char *>(bytes.data),
                               bytes.data_length);
        const auto arguments = [&](size_t launch) {
          std::vector<std::string> values;
          const auto begin = text.find("](", launch);
          if (begin == std::string::npos)
            return values;
          std::istringstream input(
              text.substr(begin + 2, text.find(')', begin) - begin - 2));
          for (std::string value; std::getline(input, value, ',');) {
            const auto first = value.find_first_not_of(" \t");
            const auto last = value.find_last_not_of(" \t");
            if (first == std::string::npos)
              return std::vector<std::string>{};
            values.push_back(value.substr(first, last - first + 1));
          }
          return values;
        };
        // The head is the final projection. Trace its actual weight operand;
        // finding both table names or a Q6 layer launch would miss the bug.
        const auto launch =
            text.rfind(head.output_tag == 8
                           ? "kernel.launch @ggml_linear_q8_0_q8_1_x4["
                           : "kernel.launch @ggml_linear_q6k_q8_1_x4_compat[");
        ASSERT_NE(launch, std::string::npos);
        const auto projected = arguments(launch);
        const size_t weight_index = head.output_tag == 8 ? 4 : 8;
        ASSERT_EQ(projected.size(), weight_index + 2);
        const auto definition =
            text.find(projected[weight_index] + " = command.parameter ");
        ASSERT_NE(definition, std::string::npos);
        const auto parameter =
            text.substr(definition, text.find('\n', definition) - definition);
        EXPECT_NE(parameter.find(head.tied ? ", \"token_embd.weight\""
                                           : ", \"output.weight\""),
                  std::string::npos)
            << parameter;
        uint64_t output_bytes = 0;
        ASSERT_TRUE(model::EncodedTensorByteSize(
            head.output_tag, descriptor.embedding_width,
            descriptor.vocabulary_size, &output_bytes));
        // The linked type can retain a dimension SSA value instead of a
        // literal. Resolve that actual dimension through its local definition
        // and linked configuration, rather than accepting any byte-view type.
        const std::string type_prefix = " : view<";
        const auto type_begin = parameter.find(type_prefix);
        ASSERT_NE(type_begin, std::string::npos) << parameter;
        const auto dimension_begin = type_begin + type_prefix.size();
        const auto type_end = parameter.find("xi8>", dimension_begin);
        ASSERT_NE(type_end, std::string::npos) << parameter;
        auto extent =
            parameter.substr(dimension_begin, type_end - dimension_begin);
        ASSERT_FALSE(extent.empty());
        if (extent.front() == '[') {
          ASSERT_EQ(extent.back(), ']') << extent;
          const auto value = extent.substr(1, extent.size() - 2);
          const auto command = text.rfind("command.def ", definition);
          ASSERT_NE(command, std::string::npos);
          const auto dimension = text.find(value + " = ", command);
          ASSERT_NE(dimension, std::string::npos) << value;
          ASSERT_LT(dimension, definition) << value;
          const auto rhs_begin = dimension + value.size() + 3;
          const auto rhs =
              text.substr(rhs_begin, text.find('\n', rhs_begin) - rhs_begin);
          const std::string constant_prefix = "index.constant ";
          const std::string config_prefix = "config.get @";
          if (rhs.find(constant_prefix) == 0) {
            extent = rhs.substr(constant_prefix.size());
          } else {
            ASSERT_EQ(rhs.find(config_prefix), 0u) << rhs;
            const auto symbol_end = rhs.find(" : ", config_prefix.size());
            ASSERT_NE(symbol_end, std::string::npos) << rhs;
            const auto symbol = rhs.substr(config_prefix.size(),
                                           symbol_end - config_prefix.size());
            EXPECT_EQ(symbol, "qwen2.model.output_weight_bytes");
            const auto binding = "config.def @" + symbol + " = ";
            const auto bound = text.find(binding);
            ASSERT_NE(bound, std::string::npos) << symbol;
            const auto literal = bound + binding.size();
            extent = text.substr(literal, text.find('\n', literal) - literal);
          }
        }
        std::istringstream extent_input(extent);
        uint64_t resolved_bytes = 0;
        ASSERT_TRUE(bool(extent_input >> resolved_bytes)) << extent;
        EXPECT_EQ(resolved_bytes, output_bytes) << parameter;
        // Both packing and projection must use the output policy, including
        // cases whose embedding would select the opposite packing policy.
        const auto pack = text.rfind(
            head.output_tag == 8
                ? "kernel.launch @ggml_quantize_q8_1_x4_f32["
                : "kernel.launch @ggml_quantize_q8_1_x4_qk_compat_f32[",
            launch);
        ASSERT_NE(pack, std::string::npos);
        const auto packed = arguments(pack);
        ASSERT_FALSE(packed.empty());
        EXPECT_EQ(packed.back(), projected[weight_index - 1]);
      }
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
TEST(Qwen2Configuration, OutputFactsAndBindingsAreIndependentOfEmbedding) {
  for (const auto &head : kHeadCases) {
    SCOPED_TRACE(head.embedding_tag);
    SCOPED_TRACE(head.output_tag);
    SCOPED_TRACE(head.tied);
    const auto descriptor = HeadFacts(head);
    EXPECT_EQ(descriptor.output_tied_to_embedding, head.tied);
    const auto *embedding = Role(descriptor.globals, "token_embedding");
    const auto *output = Role(descriptor.globals, "output");
    ASSERT_TRUE(embedding && output);
    EXPECT_EQ(embedding->storage_type, head.embedding_tag);
    EXPECT_EQ(output->storage_type, head.output_tag);
    EXPECT_EQ(embedding->absolute_offset == output->absolute_offset, head.tied);
    if (head.embedding_tag != head.output_tag)
      EXPECT_NE(embedding->byte_size, output->byte_size);
    ConfigurationFacts facts;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(descriptor, "distinct-tables", {128, 512},
                                    &facts, &error))
        << error;
    for (const auto &[name, value] :
         std::vector<std::pair<const char *, uint64_t>>{
             {"tie_word_embeddings", head.tied},
             {"output_tag", head.output_tag},
             {"output_weight_bytes", output->byte_size},
             {"embedding_tag", head.embedding_tag},
             {"embedding_weight_bytes", embedding->byte_size}}) {
      EXPECT_NE(facts.source.find("config.def @qwen2.model." +
                                  std::string(name) + " = " +
                                  std::to_string(value) + " : index\n"),
                std::string::npos)
          << name;
    }
    const auto bound = std::find_if(
        facts.parameters.begin(), facts.parameters.end(),
        [](const auto &binding) { return binding.role == "output"; });
    ASSERT_NE(bound, facts.parameters.end());
    EXPECT_FALSE(bound->layer);
    EXPECT_EQ(bound->source_asset, "distinct-tables");
    EXPECT_EQ(bound->tensor.name,
              head.tied ? "token_embd.weight" : "output.weight");
    EXPECT_EQ(bound->tensor.storage_type, output->storage_type);
    EXPECT_EQ(bound->tensor.byte_size, output->byte_size);
    EXPECT_EQ(bound->tensor.absolute_offset, output->absolute_offset);
    EXPECT_EQ(bound->tensor.rank, output->rank);
    EXPECT_EQ(bound->tensor.normalized_extents, output->normalized_extents);
    const auto format = head.output_tag == 8 ? "q8_0" : "q6k";
    EXPECT_NE(facts.source.find("@ggml.linear_" + std::string(format) +
                                "_q8_1_x4.output_capacity = " +
                                std::to_string(descriptor.vocabulary_size) +
                                " : index\n"),
              std::string::npos);
  }
}

TEST(Qwen2Configuration, RejectsMalformedIndependentHeadsTransactionally) {
  for (int mutation = 0; mutation < 8; ++mutation) {
    auto descriptor = HeadFacts({12, 14, false});
    auto output = std::find_if(
        descriptor.globals.begin(), descriptor.globals.end(),
        [](const auto &binding) { return binding.role == "output"; });
    ASSERT_NE(output, descriptor.globals.end());
    // Missing physical output.weight alone is valid and derives a tied model.
    // These mutations invalidate the logical role or an untied descriptor.
    if (mutation == 0)
      descriptor.globals.erase(output);
    if (mutation == 1)
      output->tensor.name = "token_embd.weight";
    if (mutation == 2)
      output->tensor.storage_type = 12;
    if (mutation == 3)
      output->tensor.rank = 1;
    if (mutation == 4)
      --output->tensor.normalized_extents[1];
    if (mutation == 5)
      --output->tensor.byte_size;
    if (mutation == 6)
      output->tensor.absolute_offset = uint64_t(INT64_MAX);
    if (mutation == 7)
      descriptor.output_tied_to_embedding = true;
    ConfigurationFacts facts;
    facts.source = "unchanged";
    facts.parameters.push_back({"sentinel", {}, "sentinel", {}});
    std::string error;
    EXPECT_FALSE(DeriveConfiguration(descriptor, "asset", {}, &facts, &error))
        << mutation;
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(facts.source, "unchanged");
    ASSERT_EQ(facts.parameters.size(), 1u);
    EXPECT_EQ(facts.parameters.front().role, "sentinel");
  }
  auto tied = HeadFacts({14, 14, true});
  auto output = std::find_if(tied.globals.begin(), tied.globals.end(),
                             [](const auto &b) { return b.role == "output"; });
  ASSERT_NE(output, tied.globals.end());
  output->tensor.absolute_offset += 256;
  ConfigurationFacts facts;
  facts.source = "unchanged";
  std::string error;
  EXPECT_FALSE(DeriveConfiguration(tied, "asset", {}, &facts, &error));
  EXPECT_NE(error.find("inconsistent aliases"), std::string::npos) << error;
  EXPECT_EQ(facts.source, "unchanged");
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

TEST(Qwen2ConfigurationRealAssets,
     ActualSelectedHeadRetainsPhysicalEncodingAndRange) {
  const char *path = std::getenv("LOOMIFY_TEST_QWEN2_GGUF");
  ASSERT_NE(path, nullptr) << "explicit immutable Qwen2 GGUF required";
  std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"),
                                                     std::fclose);
  ASSERT_TRUE(file);
  std::setvbuf(file.get(), nullptr, _IONBF, 0);
  model::GgufCatalog catalog;
  model::ModelDescriptor descriptor;
  std::string error;
  ASSERT_TRUE(model::ReadGgufCatalog(file.get(), &catalog, &error)) << error;
  EXPECT_LE(catalog.bytes_read, catalog.header_end);
  ASSERT_TRUE(model::LoadModelDescriptor(catalog, &descriptor, &error))
      << error;
  ASSERT_EQ(descriptor.family, "qwen2");
  const auto *embedding = model::FindTensor(catalog, "token_embd.weight");
  const auto *output = model::FindTensor(catalog, "output.weight");
  const bool tied = output == nullptr;
  if (tied)
    output = embedding;
  ASSERT_TRUE(embedding && output);
  EXPECT_EQ(descriptor.output_tied_to_embedding, tied);
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(descriptor, path, {128, 512}, &facts, &error))
      << error;
  const auto bound = std::find_if(
      facts.parameters.begin(), facts.parameters.end(),
      [](const auto &binding) { return binding.role == "output"; });
  ASSERT_NE(bound, facts.parameters.end());
  EXPECT_EQ(bound->tensor.name, output->name);
  EXPECT_EQ(bound->tensor.storage_type, output->storage_type);
  EXPECT_EQ(bound->tensor.absolute_offset, output->absolute_offset);
  EXPECT_EQ(bound->tensor.byte_size, output->byte_size);
  EXPECT_EQ(bound->tensor.rank, output->rank);
  EXPECT_EQ(bound->tensor.normalized_extents, output->normalized_extents);
  EXPECT_NE(facts.source.find("@qwen2.model.tie_word_embeddings = " +
                              std::to_string(tied) + " : index\n"),
            std::string::npos);
  EXPECT_NE(facts.source.find("@qwen2.model.output_tag = " +
                              std::to_string(output->storage_type) +
                              " : index\n"),
            std::string::npos);
  EXPECT_NE(facts.source.find("@qwen2.model.embedding_tag = " +
                              std::to_string(embedding->storage_type) +
                              " : index\n"),
            std::string::npos);
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
