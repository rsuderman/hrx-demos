// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/pass.h"
#include "loomc/link.h"
#include "loomc/source.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/configuration_fixture.h"
#include "lib/loomify/testing/gemma3_fixture.h"
#include "lib/loomify/testing/qwen3_fixture.h"
#include "lib/loomify/testing/alternating_qwen_fixture.h"
#include "tools/model/gguf_catalog.h"
#include "tools/model/model_descriptor.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
namespace hrx::loomify_gguf {
namespace {
auto Model(bool llama) {
  return llama ? testing::LlamaModelFacts() : testing::QwenModelFacts();
}
std::map<std::string, std::pair<std::string, std::string>>
Scalars(const std::string &text) {
  std::map<std::string, std::pair<std::string, std::string>> out;
  const std::regex line(R"(config.def @([^ ]+) = ([^ ]+) : ([a-z0-9]+))");
  for (auto it = std::sregex_iterator(text.begin(), text.end(), line);
       it != std::sregex_iterator(); ++it)
    EXPECT_TRUE(out.emplace((*it)[1].str(),
                            std::make_pair((*it)[2].str(), (*it)[3].str()))
                    .second);
  return out;
}
TEST(ConfigurationFactsTest, OwnsAllNamedRangesAndNoPrograms) {
  for (bool llama : {true, false}) {
    auto m = Model(llama);
    ConfigurationFacts f, again;
    std::string error = "stale";
    ASSERT_TRUE(DeriveConfiguration(m, "asset", {}, &f, &error)) << error;
    EXPECT_TRUE(error.empty());
    ASSERT_TRUE(DeriveConfiguration(m, "asset", {}, &again, &error));
    EXPECT_EQ(f.source, again.source);
    size_t count = m.globals.size();
    for (const auto &l : m.layers)
      count += l.tensors.size();
    ASSERT_EQ(f.parameters.size(), count);
    for (const auto &p : f.parameters) {
      const auto &bindings = p.layer ? m.layers[*p.layer].tensors : m.globals;
      auto b = std::find_if(bindings.begin(), bindings.end(),
                            [&](const auto &b) { return b.role == p.role; });
      ASSERT_NE(b, bindings.end());
      EXPECT_EQ(p.source_asset, "asset");
      EXPECT_EQ(p.tensor.name, b->tensor.name);
      EXPECT_EQ(p.tensor.normalized_extents, b->tensor.normalized_extents);
      EXPECT_EQ(p.tensor.storage_type, b->tensor.storage_type);
      EXPECT_EQ(p.tensor.byte_size, b->tensor.byte_size);
      EXPECT_EQ(p.tensor.absolute_offset, b->tensor.absolute_offset);
      EXPECT_EQ(p.tensor.rank, b->tensor.rank);
    }
    m = {};
    EXPECT_EQ(f.source, again.source);
    EXPECT_FALSE(f.parameters.front().tensor.name.empty());
    EXPECT_EQ(f.source.find("command."), std::string::npos);
    EXPECT_EQ(f.source.find("kernel."), std::string::npos);
    EXPECT_EQ(f.source.find("buffer.alloc"), std::string::npos);
  }
}
TEST(ConfigurationFactsTest, SharedGeometryAtHistoryAndPoolBoundaries) {
  for (bool llama : {true, false})
    for (uint64_t t : {1, 128, 129, 512})
      for (uint64_t c : {512u, 513u, llama ? 131072u : 262144u}) {
        ConfigurationFacts f;
        std::string error;
        ASSERT_TRUE(
            DeriveConfiguration(Model(llama), "asset", {t, c}, &f, &error))
            << error;
        auto s = Scalars(f.source);
        auto value = [&](const char *key, const char *type = "index") {
          EXPECT_EQ(s.at(key).second, type);
          return std::stoull(s.at(key).first);
        };
        EXPECT_EQ(value(llama ? "llama.attention.pool_capacity"
                              : "qwen38.attention.pool_capacity"),
                  (c + 63) / 64 * 64);
        EXPECT_EQ(value(llama ? "llama.attention.cache_capacity"
                              : "qwen38.attention.cache_capacity"),
                  c);
        EXPECT_EQ(value(llama ? "runner.llama32.prefill_token_count"
                              : "runner.qwen38.prefill_token_count"),
                  t);
        if (llama) {
          EXPECT_EQ(value("llama.model.history_capacity"),
                    std::max<uint64_t>(t, 129));
          EXPECT_EQ(value("llama.attention.cache_base", "offset"),
                    768 + 4 * std::max<uint64_t>(t, 129));
          EXPECT_EQ(value("llama.attention.map_origin", "offset"), 2048u);
        } else {
          EXPECT_EQ(value("qwen.model.hidden_capacity_bytes"), 2097152u);
          EXPECT_EQ(value("qwen.model.gdn_history_bytes"), 73728u);
          EXPECT_EQ(value("qwen.model.gdn_recurrent_bytes"), 1048576u);
          EXPECT_EQ(value("qwen.model.gdn_state_bytes"), 20201472u);
          EXPECT_EQ(value("qwen.model.gdn_slot_origin"), 256u);
          EXPECT_EQ(value("qwen.model.private_row_stride"), 20204032u);
          EXPECT_EQ(value("qwen.model.map_origin"), 2048u);
          EXPECT_EQ(value("qwen.model.target_kv_planes"), 12u);
          EXPECT_EQ(value("qwen.model.kv_plane_token_bytes"), 1024u);
          EXPECT_EQ(value("qwen.model.target_kv_token_bytes"), 12288u);
          EXPECT_EQ(s.count("qwen.model.ffn_gate_q4_layers"), 0u);
        }
      }
}
TEST(ConfigurationFactsTest, OrdinaryPreparationUsesSharedTypedFacts) {
  const std::vector<model::ModelDescriptor> models = {
      testing::LlamaModelFacts(), testing::gemma3::ModelFacts(), testing::Qwen3ModelFacts()};
  const uint64_t hidden[] = {3072, 1152, 1024};
  const uint64_t layers[] = {28, 6, 2};
  const uint64_t kv[] = {1024, 256, 1024};
  for (size_t family = 0; family < models.size(); ++family)
    for (uint64_t tokens : {1, 8, 128, 129, 512}) {
      ConfigurationFacts facts;
      std::string error;
      ASSERT_TRUE(DeriveConfiguration(models[family], "fixture", {tokens, 513}, &facts, &error)) << error;
      const auto values = Scalars(facts.source);
      const auto history = std::max<uint64_t>(tokens, 129);
      const std::vector<std::pair<const char *, uint64_t>> expected = {
          {"hidden_size", hidden[family]}, {"layer_count", layers[family]},
          {"kv_size", kv[family]}, {"history_capacity", history},
          {"token_capacity", tokens}, {"cache_capacity", 513}, {"pool_capacity", 576},
          {"cache_base", 768 + 4 * history}, {"map_origin", 2048}};
      for (const auto &[field, value] : expected) {
        const auto &actual = values.at(std::string("loomify.serving.") + field);
        EXPECT_EQ(std::stoull(actual.first), value) << models[family].family << field;
        EXPECT_EQ(actual.second, (std::string(field) == "cache_base" || std::string(field) == "map_origin") ? "offset" : "index");
      }
    }
  ConfigurationFacts hybrid;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(testing::QwenModelFacts(), "fixture", {}, &hybrid, &error)) << error;
  EXPECT_EQ(hybrid.source.find("@loomify.serving."), std::string::npos);
}
TEST(ConfigurationFactsTest, SharedNamesStillRejectMalformedConstructedBindings) {
  for (const auto &original : {testing::LlamaModelFacts(), testing::gemma3::ModelFacts(), testing::Qwen3ModelFacts(), testing::QwenModelFacts()})
    for (int mutation = 0; mutation < 4; ++mutation) {
      auto model = original;
      if (mutation == 0) model.layers[0].tensors[0].tensor.name = "blk.99.attn_norm.weight";
      if (mutation == 1) model.layers[0].tensors.push_back(model.layers[0].tensors[0]);
      if (mutation == 2) {
        auto output = std::find_if(model.globals.begin(), model.globals.end(), [](const auto &b) { return b.role == "output"; });
        ASSERT_NE(output, model.globals.end());
        output->tensor.name = model.output_tied_to_embedding ? "output.weight" : "token_embd.weight";
      }
      if (mutation == 3) model.layers[0].tensors[0].tensor.absolute_offset = uint64_t(INT64_MAX);
      ConfigurationFacts facts;
      facts.source = "unchanged";
      std::string error;
      EXPECT_FALSE(DeriveConfiguration(model, "fixture", {}, &facts, &error)) << original.family << mutation;
      EXPECT_EQ(facts.source, "unchanged");
      EXPECT_FALSE(error.empty());
    }
}
TEST(ConfigurationFactsTest, LlamaScalarFactsPreserveQualifiedF32Bits) {
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(Model(true), "asset", {}, &facts, &error)) << error;
  const auto scalars = Scalars(facts.source);
  const float step = std::stof(scalars.at("llama.model.rotary_step").first);
  uint32_t bits;
  std::memcpy(&bits, &step, sizeof(bits));
  EXPECT_EQ(bits, 1062243009u);
  EXPECT_FLOAT_EQ(std::stof(scalars.at("llama.model.attention_scale").first),
                  0.08838834764831845f);
}
TEST(ConfigurationFactsTest, Qwen27UsesActualFormatsBindingsAnd64LayerLookups) {
  const auto model = testing::QwenModelFacts(true);
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(model, "asset", {512, 513}, &facts, &error)) << error;
  const auto scalars = Scalars(facts.source);
  EXPECT_EQ(scalars.at("qwen.model.base_layers").first, "64");
  EXPECT_EQ(scalars.at("qwen.model.target_kv_planes").first, "32");
  EXPECT_EQ(scalars.at("qwen.model.gdn_state_bytes").first, "156893184");
  EXPECT_EQ(scalars.at("ggml.quantize_q8_1_x4.group_capacity").first, "69632");
  EXPECT_EQ(scalars.at("qwen.model.embedding_storage_type").first, "12");
  EXPECT_EQ(scalars.at("qwen.model.tie_word_embeddings").first, "0");
  EXPECT_EQ(facts.parameters.size(), 851u);
  EXPECT_NE(facts.source.find("@loomify.config.attention_projections"), std::string::npos);
  EXPECT_NE(facts.source.find("@loomify.config.gdn_projections"), std::string::npos);
  EXPECT_EQ(scalars.count("qwen.model.ffn_gate_q4_layers"), 0u);
  EXPECT_EQ(scalars.count("qwen.model.attention_q6_layers"), 0u);
  for (size_t layer : {31u, 32u, 47u, 63u}) {
    const auto name = "blk." + std::to_string(layer) + ".ffn_down.weight";
    auto it = std::find_if(facts.parameters.begin(), facts.parameters.end(), [&](const auto &p) { return p.tensor.name == name; });
    ASSERT_NE(it, facts.parameters.end());
    EXPECT_EQ(it->tensor.storage_type, 12u);
    EXPECT_EQ(it->tensor.byte_size, 50135040u);
  }
  for (auto mutation : {0, 1, 2}) {
    auto bad = model;
    if (mutation == 0) bad.layers[32].tensors.front().tensor.storage_type = 8;
    if (mutation == 1) bad.layers[63].attention_ordinal = 0;
    if (mutation == 2) bad.globals[2].tensor.name = "token_embd.weight";
    ConfigurationFacts unchanged;
    unchanged.source = "sentinel";
    EXPECT_FALSE(DeriveConfiguration(bad, "asset", {}, &unchanged, &error));
    EXPECT_EQ(unchanged.source, "sentinel");
  }
}
TEST(ConfigurationFactsTest, IndependentGgufLayerMapControlsRoutesAndStorage) {
  model::ModelDescriptor model;
  model::GgufCatalog catalog;
  std::string error;
  ASSERT_TRUE(testing::LoadAlternatingQwenFixture(&model, &catalog, &error)) << error;
  ASSERT_EQ(model.layers.size(), 8u);
  EXPECT_EQ(catalog.tensors.size(), 102u);
  EXPECT_LT(catalog.bytes_read, catalog.data_offset + 1);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(model.layers[i].index, i);
    EXPECT_EQ(model.layers[i].kind, i % 2 ? model::LayerKind::kAttention
                                        : model::LayerKind::kGatedDeltaNet);
    EXPECT_EQ(i % 2 ? model.layers[i].attention_ordinal
                    : model.layers[i].recurrent_ordinal, i / 2);
  }
  ConfigurationFacts facts;
  ASSERT_TRUE(DeriveConfiguration(model, "independent.gguf", {3, 65}, &facts, &error)) << error;
  const auto scalars = Scalars(facts.source);
  for (const auto &expected : std::vector<std::pair<const char *, const char *>>{
           {"qwen.model.base_layers", "8"}, {"qwen.model.gdn_layers", "4"},
           {"qwen.model.group_count", "4"}, {"qwen.model.target_kv_planes", "8"},
           {"qwen.model.gdn_state_bytes", "4489216"},
           {"qwen.model.private_row_stride", "4491776"},
           {"qwen.model.target_kv_token_bytes", "8192"}})
    EXPECT_EQ(scalars.at(expected.first).first, expected.second) << expected.first;
  // One physical embedding binding also supplies the tied logical output.
  EXPECT_EQ(facts.parameters.size(), 103u);
  for (const auto &parameter : facts.parameters) {
    const auto *source = model::FindTensor(catalog, parameter.tensor.name);
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(parameter.tensor.absolute_offset, source->absolute_offset);
    EXPECT_EQ(parameter.tensor.byte_size, source->byte_size);
    EXPECT_EQ(parameter.tensor.storage_type, source->storage_type);
  }
  auto no_attention = model;
  no_attention.layers = {model.layers[0]};
  EXPECT_FALSE(DeriveConfiguration(no_attention, "fixture", {}, &facts, &error));
  auto too_many = model;
  too_many.layers.resize(65);
  EXPECT_FALSE(DeriveConfiguration(too_many, "fixture", {}, &facts, &error));
}
TEST(ConfigurationFactsTest, DescriptorFieldsAndFormatsDoNotSelectCheckpointPresets) {
  for (bool tied : {true, false}) {
    model::ModelDescriptor model;
    model::GgufCatalog catalog;
    std::string error;
    ASSERT_TRUE(testing::LoadAlternatingQwenFixture(&model, &catalog, &error,
        4096, 8, tied)) << error;
    ASSERT_EQ(model.embedding_width, 1024u);
    ASSERT_EQ(model.layers.size(), 8u);
    ConfigurationFacts facts;
    ASSERT_TRUE(DeriveConfiguration(model, "changed.gguf", {3, 65}, &facts, &error)) << error;
    const auto scalars = Scalars(facts.source);
    EXPECT_EQ(scalars.at("qwen.model.ffn_size").first, "4096");
    EXPECT_EQ(scalars.at("qwen.model.ffn_row_bytes").first, "16384");
    EXPECT_EQ(scalars.at("qwen.model.ffn_capacity_bytes").first, "8388608");
    EXPECT_EQ(scalars.at("qwen.model.tie_word_embeddings").first, tied ? "1" : "0");
    EXPECT_EQ(scalars.at("ggml.linear_q4k_q8_1_x4.output_capacity").first, "4096");
    EXPECT_EQ(scalars.at("ggml.linear_q5k_q8_1_x4.output_capacity").first, "1024");
    EXPECT_EQ(scalars.at("ggml.linear_q6k_q8_1_x4.output_capacity").first, "248320");
    EXPECT_EQ(scalars.at("ggml.linear_q8_0_q8_1_x4.output_capacity").first, "4096");
    EXPECT_EQ(scalars.at("ggml.quantize_q8_1_x4.group_capacity").first, "16384");
    for (const auto &parameter : facts.parameters) {
      if (parameter.role == "ffn_gate") EXPECT_EQ(parameter.tensor.byte_size, 2359296u);
      if (parameter.role == "ffn_down") EXPECT_EQ(parameter.tensor.byte_size, 2883584u);
      if (parameter.role == "query") {
        EXPECT_EQ(parameter.tensor.storage_type, 8u);
        EXPECT_EQ(parameter.tensor.byte_size, 4456448u);
      }
      if (parameter.role == "output") EXPECT_EQ(parameter.tensor.name, tied ? "token_embd.weight" : "output.weight");
      const auto *source = model::FindTensor(catalog, parameter.tensor.name);
      ASSERT_NE(source, nullptr);
      EXPECT_EQ(parameter.tensor.absolute_offset, source->absolute_offset);
      EXPECT_EQ(parameter.tensor.byte_size, source->byte_size);
    }
    // Global-shape consumers still require uniform block geometry. This is a
    // real storage contract, not a requirement to match another checkpoint.
    auto mixed = model;
    mixed.layers[1].feed_forward_width = 3584;
    for (auto &binding : mixed.layers[1].tensors) {
      if (binding.role == "ffn_gate" || binding.role == "ffn_up") {
        binding.tensor.normalized_extents[1] = 3584;
        binding.tensor.byte_size = 2064384;
      } else if (binding.role == "ffn_down") {
        binding.tensor.normalized_extents[0] = 3584;
        binding.tensor.byte_size = 2523136;
      }
    }
    EXPECT_FALSE(DeriveConfiguration(mixed, "mixed", {}, &facts, &error));
  }
}
TEST(ConfigurationFactsTest, ScalarDataMustSurviveF32Representation) {
  for (bool llama : {true, false}) {
    auto descriptor = Model(llama);
    descriptor.rms_epsilon = 2e-5;
    ConfigurationFacts facts;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(descriptor, "fixture", {}, &facts, &error)) << error;
    const auto key = llama ? "llama.model.rms_epsilon" : "qwen.model.rms_epsilon";
    EXPECT_FLOAT_EQ(std::stof(Scalars(facts.source).at(key).first), 2e-5f);
    for (double value : {1e-100, std::numeric_limits<double>::max()}) {
      descriptor.rms_epsilon = value;
      facts.source = "sentinel";
      EXPECT_FALSE(DeriveConfiguration(descriptor, "fixture", {}, &facts, &error));
      EXPECT_EQ(facts.source, "sentinel");
    }
  }
}
TEST(ConfigurationFactsTest, SharedEncodingSizeUsesQ8BlocksAndPreservesFailureOutput) {
  uint64_t bytes = 7;
  EXPECT_TRUE(model::EncodedTensorByteSize(8, 5120, 48, &bytes));
  EXPECT_EQ(bytes, 261120u);
  EXPECT_TRUE(model::EncodedTensorByteSize(12, 5120, 17408, &bytes));
  EXPECT_EQ(bytes, 50135040u);
  for (auto tag : {8u, 12u, 999u}) {
    bytes = 7;
    EXPECT_FALSE(model::EncodedTensorByteSize(tag, 31, 48, &bytes));
    EXPECT_EQ(bytes, 7u);
  }
  EXPECT_FALSE(model::EncodedTensorByteSize(8, UINT64_MAX - 31, UINT64_MAX, &bytes));
}
TEST(ConfigurationFactsTest, InvalidCapacitiesPreserveOutput) {
  for (bool llama : {true, false}) {
    const uint64_t limit = llama ? 131072 : 262144;
    for (const auto &s :
         std::vector<ConfigurationSettings>{{0, 512},
                                            {513, 1024},
                                            {UINT64_MAX, 512},
                                            {128, 0},
                                            {128, 127},
                                            {128, limit + 1},
                                            {128, UINT64_MAX}}) {
      ConfigurationFacts f;
      f.source = "sentinel";
      std::string error;
      EXPECT_FALSE(DeriveConfiguration(Model(llama), "asset", s, &f, &error));
      EXPECT_EQ(f.source, "sentinel");
      EXPECT_FALSE(error.empty());
    }
  }
}
TEST(ConfigurationFactsTest, MalformedModelFactsPreserveOutput) {
  using Mutation = std::function<void(model::ModelDescriptor &)>;
  std::vector<Mutation> common = {
      [](auto &m) { m.family = "unsupported"; },
      [](auto &m) { m.embedding_width++; },
      [](auto &m) { m.rms_epsilon = std::numeric_limits<double>::quiet_NaN(); },
      [](auto &m) { m.output_tied_to_embedding = false; },
      [](auto &m) { m.globals.back().tensor.absolute_offset++; },
      [](auto &m) { m.layers[1].index = 0; },
      [](auto &m) { m.layers[1].index = 24; },
      [](auto &m) { m.layers[1].feed_forward_width = UINT64_MAX; },
      [](auto &m) { m.layers[0].tensors.erase(m.layers[0].tensors.begin()); },
      [](auto &m) {
        m.layers[0].tensors.push_back(m.layers[0].tensors.front());
      },
      [](auto &m) { m.layers[0].tensors.front().tensor.storage_type = 999; },
      [](auto &m) { m.layers[0].tensors.front().tensor.byte_size++; },
      [](auto &m) {
        m.layers[0].tensors.front().tensor.absolute_offset = UINT64_MAX;
      },
      [](auto &m) {
        m.layers[0].tensors.front().tensor.normalized_extents[0]++;
      },
      [](auto &m) { m.layers[0].kind = static_cast<model::LayerKind>(99); }};
  for (bool llama : {true, false})
    for (const auto &change : common) {
      auto m = Model(llama);
      change(m);
      ConfigurationFacts f;
      f.source = "sentinel";
      std::string error;
      EXPECT_FALSE(DeriveConfiguration(m, "asset", {}, &f, &error));
      EXPECT_EQ(f.source, "sentinel");
      EXPECT_FALSE(error.empty());
    }
  // Layer count is source data; removing a complete final attention layer is
  // no longer a malformed descriptor when the remaining bindings are coherent.
  auto short_llama = Model(true);
  short_llama.layers.pop_back();
  ConfigurationFacts ignored;
  std::string short_error;
  EXPECT_TRUE(DeriveConfiguration(short_llama, "asset", {}, &ignored, &short_error));
  EXPECT_EQ(Scalars(ignored.source).at("llama.model.layer_count").first, "27");
  for (const Mutation &change : std::vector<Mutation>{
           [](auto &m) { m.layers[0].recurrent_ordinal = 18; },
           [](auto &m) { m.layers[0].recurrent_ordinal.reset(); },
           [](auto &m) { m.layers[0].attention_ordinal = 0; },
           [](auto &m) { m.layers[3].attention_ordinal = 6; },
           [](auto &m) { m.ssm->inner_width++; },
           [](auto &m) { m.rope.sections[0]++; },
           [](auto &m) { std::swap(m.layers[0], m.layers[1]); }}) {
    auto m = Model(false);
    change(m);
    ConfigurationFacts f;
    f.source = "sentinel";
    std::string e;
    EXPECT_FALSE(DeriveConfiguration(m, "asset", {}, &f, &e));
    EXPECT_EQ(f.source, "sentinel");
  }
}
std::string Diagnostics(loomc_result_t *r) {
  std::string s;
  if (r)
    for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
      auto *d = loomc_result_diagnostic_at(r, i);
      s.append(d->message.data, d->message.size);
      s += '\n';
    }
  return s;
}
::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status))
    return ::testing::AssertionSuccess();
  auto message = loomc_status_message(status);
  std::string text(message.data ? message.data : "", message.size);
  loomc_status_free(status);
  return ::testing::AssertionFailure() << text;
}
class ConfigurationNativeTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
  }
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
};
void CheckLookupTuples(loomc_context_t *context, loomc_workspace_t *workspace,
                       const model::ModelDescriptor &m, bool llama,
                       bool bytecode = false) {
  ConfigurationSettings settings;
  ConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveConfiguration(m, "asset", settings, &facts, &error))
      << error;
  std::ostringstream source;
  source << facts.source;
  std::map<std::string, std::vector<uint64_t>> expected;
  for (size_t layer = 0; layer < m.layers.size(); ++layer) {
    const auto &bindings = m.layers[layer].tensors;
    auto role = [&](const char *name) -> const model::TensorRecord & {
      return std::find_if(bindings.begin(), bindings.end(),
                          [&](const auto &b) { return b.role == name; })
          ->tensor;
    };
    for (int kind : {0, 1, 2, 3, 4}) {
      const bool attention = kind == 1;
      const bool route = kind == 2;
      const bool qwen_attention = kind == 3;
      const bool qwen_gdn = kind == 4;
      const auto &descriptor = m.layers[layer];
      if ((attention && !llama) || (route && llama) ||
          (qwen_attention && (llama || descriptor.kind != model::LayerKind::kAttention)) ||
          (qwen_gdn && (llama || descriptor.kind != model::LayerKind::kGatedDeltaNet)))
        continue;
      std::vector<uint64_t> tuple;
      if (route) {
        const auto &l = m.layers[layer];
        const bool gdn = l.kind == model::LayerKind::kGatedDeltaNet;
        tuple = {gdn ? 1u : 0u, layer,
                 gdn ? *l.recurrent_ordinal : *l.attention_ordinal};
      } else if (attention || qwen_attention) {
        for (const char *r : {"query", "key", "value", "attention_output"})
          tuple.push_back(role(r).storage_type);
        for (const char *r : {"query", "key", "value", "attention_output"})
          tuple.push_back(role(r).byte_size);
      } else if (qwen_gdn) {
        for (const char *r : {"recurrent_qkv", "recurrent_gate", "alpha", "beta", "recurrent_output"})
          tuple.push_back(role(r).storage_type);
        for (const char *r : {"recurrent_qkv", "recurrent_gate", "alpha", "beta", "recurrent_output"})
          tuple.push_back(role(r).byte_size);
      } else {
        tuple.push_back(m.layers[layer].feed_forward_width);
        for (const char *r : {"ffn_gate", "ffn_up", "ffn_down"})
          tuple.push_back(role(r).storage_type);
        for (const char *r : {"ffn_gate", "ffn_up", "ffn_down"})
          tuple.push_back(role(r).byte_size);
      }
      const char *lookup = route ? "layer_route" : attention ? "layer_attention"
          : qwen_attention ? "attention_projections" : qwen_gdn ? "gdn_projections" : "layer_ffn";
      const std::string name = std::string("probe_") + lookup + "_" + std::to_string(layer);
      const auto lookup_index = qwen_attention ? *descriptor.attention_ordinal
          : qwen_gdn ? *descriptor.recurrent_ordinal : layer;
      expected.emplace(name, tuple);
      std::string types, values;
      for (size_t i = 0; i < tuple.size(); ++i) {
        types += (i ? ", index" : "index");
        values += (i ? ", %v" : "%v") + std::to_string(i);
      }
      source << "func.def public pure @" << name << "() -> (" << types
             << ") {\n"
             << "  %layer = index.constant " << lookup_index << " : index\n  "
             << values << " = func.call pure @loomify.config."
             << lookup << "(%layer) : (" << "index) -> (" << types
             << ")\n  func.return " << values << " : " << types << "\n}\n";
    }
  }
  auto text = source.str();
  loomc_source_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  options.storage = LOOMC_SOURCE_STORAGE_COPY;
  options.identifier = loomc_make_cstring_view("tuple-probes.loom");
  options.contents = loomc_make_byte_span(text.data(), text.size());
  loomc_source_t *input = nullptr;
  ASSERT_TRUE(
      Ok(loomc_source_create(&options, loomc_allocator_system(), &input)));
  loomc_module_t *module = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
      context, workspace, input, nullptr, loomc_allocator_system(), &module,
      &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_source_release(input);
  loomc_result_release(result);
  if (bytecode) {
    loomc_source_t *serialized = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_bytecode_to_source(
        module, nullptr, loomc_allocator_system(), &serialized)));
    loomc_module_release(module);
    module = nullptr;
    result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, serialized, nullptr, loomc_allocator_system(),
        &module, &result)));
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    loomc_source_release(serialized);
    loomc_result_release(result);
  }
  loomc_pass_program_t *passes = nullptr;
  result = nullptr;
  ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view("inline-callables,combine,cse,dce"),
      nullptr, loomc_allocator_system(), &passes, &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  result = nullptr;
  ASSERT_TRUE(Ok(loomc_pass_program_run(passes, workspace, module, nullptr,
                                        loomc_allocator_system(), &result)));
  ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
  loomc_result_release(result);
  loomc_pass_program_release(passes);
  loomc_source_t *printed = nullptr;
  ASSERT_TRUE(Ok(loomc_module_serialize_text_to_source(
      module, nullptr, loomc_allocator_system(), &printed)));
  const auto bytes = loomc_source_contents(printed);
  text.assign(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
  loomc_source_release(printed);
  loomc_module_release(module);
  for (const auto &[name, tuple] : expected) {
    const auto begin = text.find("@" + name + "(");
    ASSERT_NE(begin, std::string::npos) << text;
    const auto end = text.find("}", begin);
    const auto body = text.substr(begin, end - begin);
    EXPECT_EQ(body.find("func.call"), std::string::npos) << body;
    EXPECT_EQ(body.find("scf.if"), std::string::npos) << body;
    std::map<std::string, uint64_t> constants;
    const std::regex constant(R"((%[a-zA-Z0-9_]+) = index.constant ([0-9]+))");
    for (auto i = std::sregex_iterator(body.begin(), body.end(), constant);
         i != std::sregex_iterator(); ++i)
      constants[(*i)[1].str()] = std::stoull((*i)[2].str());
    const auto ret = body.find("func.return ");
    ASSERT_NE(ret, std::string::npos) << body;
    const auto returned = body.substr(ret, body.find(':', ret) - ret);
    const std::regex value(R"(%[a-zA-Z0-9_]+)");
    std::vector<uint64_t> actual;
    for (auto i = std::sregex_iterator(returned.begin(), returned.end(), value);
         i != std::sregex_iterator(); ++i) {
      ASSERT_TRUE(constants.count(i->str())) << body;
      actual.push_back(constants.at(i->str()));
    }
    EXPECT_EQ(actual, tuple) << name << "\n" << body;
  }
}
TEST_F(ConfigurationNativeTest, SpecializesEveryLayerAndRouteTuple) {
  for (bool llama : {true, false})
    for (bool bytecode : {false, true})
      CheckLookupTuples(context, workspace, Model(llama), llama, bytecode);
}
TEST_F(ConfigurationNativeTest, IndependentGgufSpecializesAllEightRoutesAndTuples) {
  model::ModelDescriptor descriptor;
  std::string error;
  ASSERT_TRUE(testing::LoadAlternatingQwenFixture(&descriptor, nullptr, &error)) << error;
  for (bool bytecode : {false, true})
    CheckLookupTuples(context, workspace, descriptor, false, bytecode);
}
TEST_F(ConfigurationNativeTest, ChangedGgufDimensionsAndFormatsReachSourceAndBytecodeLookups) {
  model::ModelDescriptor descriptor;
  std::string error;
  ASSERT_TRUE(testing::LoadAlternatingQwenFixture(&descriptor, nullptr, &error,
      4096, 8, false)) << error;
  for (bool bytecode : {false, true})
    CheckLookupTuples(context, workspace, descriptor, false, bytecode);
}
// Resolve and fold real config consumers through the public native linker. The
// bitcast keeps exact F32 rounding observable after a source/bytecode roundtrip.
TEST_F(ConfigurationNativeTest, GeneratedNumericalScalarsReachLinkedConsumers) {
  for (bool llama : {true, false}) {
    ConfigurationFacts facts;
    std::string error;
    ASSERT_TRUE(DeriveConfiguration(Model(llama), "asset", {}, &facts, &error)) << error;
    std::vector<std::pair<std::string, float>> values = llama
        ? std::vector<std::pair<std::string, float>>{
              {"llama.model.rotary_step", 0.8146172165870667f},
              {"llama.model.attention_scale", 0.08838834764831845f}}
        : std::vector<std::pair<std::string, float>>{{"qwen.model.rms_epsilon", 1e-6f}};
    std::string text = facts.source;
    for (size_t i = 0; i < values.size(); ++i)
      text += "func.def public @scalar_probe_" + std::to_string(i) +
              "() -> (i32) {\n  %value = config.get @" + values[i].first +
              " : f32\n  %bits = scalar.bitcast %value : f32 to i32\n"
              "  func.return %bits : i32\n}\n";
    for (bool bytecode : {false, true}) {
      loomc_source_options_t options{};
      options.format = LOOMC_SOURCE_FORMAT_TEXT;
      options.storage = LOOMC_SOURCE_STORAGE_COPY;
      options.identifier = loomc_make_cstring_view("scalar-facts.loom");
      options.contents = loomc_make_byte_span(text.data(), text.size());
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_source_create(&options, loomc_allocator_system(), &source)));
      loomc_module_t *input = nullptr;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(context, workspace, source, nullptr,
          loomc_allocator_system(), &input, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      loomc_source_release(source);
      loomc_result_release(result);
      if (bytecode) {
        source = nullptr;
        ASSERT_TRUE(Ok(loomc_module_serialize_bytecode_to_source(input, nullptr,
            loomc_allocator_system(), &source)));
        loomc_module_release(input);
        input = nullptr;
        result = nullptr;
        ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(context, workspace, source, nullptr,
            loomc_allocator_system(), &input, &result)));
        ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
        loomc_source_release(source);
        loomc_result_release(result);
      }
      std::vector<std::string> names;
      for (size_t i = 0; i < values.size(); ++i) names.push_back("scalar_probe_" + std::to_string(i));
      std::vector<loomc_string_view_t> roots;
      for (const auto &name : names) roots.push_back(loomc_make_cstring_view(name.c_str()));
      loomc_link_module_provider_t provider{input, loomc_make_cstring_view("facts"), LOOMC_LINK_PROVIDER_ROLE_INPUT};
      loomc_link_options_t link{};
      link.mode = LOOMC_LINK_MODE_LINK;
      link.module_providers = &provider;
      link.module_provider_count = 1;
      link.root_symbols = roots.data();
      link.root_symbol_count = roots.size();
      link.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
      loomc_linker_t *linker = nullptr;
      ASSERT_TRUE(Ok(loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker)));
      loomc_module_t *linked = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_link_module(linker, workspace, &link, &linked, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      loomc_result_release(result);
      loomc_linker_release(linker);
      loomc_module_release(input);
      loomc_pass_program_t *passes = nullptr;
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_pass_program_create_from_pipeline_text(context,
          loomc_make_cstring_view("combine,cse,dce"), nullptr, loomc_allocator_system(), &passes, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      loomc_result_release(result);
      result = nullptr;
      ASSERT_TRUE(Ok(loomc_pass_program_run(passes, workspace, linked, nullptr,
          loomc_allocator_system(), &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
      loomc_result_release(result);
      loomc_pass_program_release(passes);
      source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_text_to_source(linked, nullptr,
          loomc_allocator_system(), &source)));
      auto contents = loomc_source_contents(source);
      std::string printed(reinterpret_cast<const char *>(contents.data), contents.data_length);
      loomc_source_release(source);
      loomc_module_release(linked);
      for (size_t i = 0; i < values.size(); ++i) {
        uint32_t expected;
        std::memcpy(&expected, &values[i].second, sizeof(expected));
        const auto begin = printed.find("@" + names[i] + "(");
        ASSERT_NE(begin, std::string::npos) << printed;
        const auto body = printed.substr(begin, printed.find('}', begin) - begin);
        EXPECT_EQ(body.find("config.get"), std::string::npos) << body;
        EXPECT_NE(body.find("scalar.constant " + std::to_string(expected) + " : i32"),
                  std::string::npos) << body;
      }
    }
  }
}
#ifdef LOOMIFY_CONFIGURATION_REAL_ASSETS
TEST_F(ConfigurationNativeTest, RealHeadersPreserveEveryNamedRangeAndTuple) {
  for (bool llama : {true, false}) {
    const char *path =
        std::getenv(llama ? "LOOMIFY_TEST_GGUF" : "LOOMIFY_TEST_QWEN_GGUF");
    ASSERT_NE(path, nullptr);
    FILE *file = std::fopen(path, "rb");
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(std::setvbuf(file, nullptr, _IONBF, 0), 0);
    model::GgufCatalog catalog;
    std::string error;
    ASSERT_TRUE(model::ReadGgufCatalog(file, &catalog, &error)) << error;
    std::fclose(file);
    EXPECT_LE(catalog.bytes_read, catalog.data_offset);
    EXPECT_LT(catalog.bytes_read, catalog.file_size / 100);
    model::ModelDescriptor m;
    ASSERT_TRUE(model::LoadModelDescriptor(catalog, &m, &error)) << error;
    ConfigurationFacts f;
    ASSERT_TRUE(DeriveConfiguration(m, path, {}, &f, &error)) << error;
    size_t count = m.globals.size();
    for (const auto &l : m.layers)
      count += l.tensors.size();
    EXPECT_EQ(f.parameters.size(), count);
    for (const auto &p : f.parameters) {
      const auto *t = model::FindTensor(catalog, p.tensor.name);
      ASSERT_NE(t, nullptr);
      EXPECT_EQ(p.tensor.absolute_offset, t->absolute_offset);
      EXPECT_EQ(p.tensor.byte_size, t->byte_size);
      EXPECT_EQ(p.tensor.normalized_extents, t->normalized_extents);
    }
    CheckLookupTuples(context, workspace, m, llama);
  }
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
