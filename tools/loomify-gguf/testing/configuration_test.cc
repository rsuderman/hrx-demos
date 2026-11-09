// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/pass.h"
#include "loomc/source.h"
#include "tools/loomify-gguf/configuration.h"
#include "tools/loomify-gguf/testing/configuration_fixture.h"
#include "tools/model/gguf_catalog.h"
#include "tools/model/model_descriptor.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdio>
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
          EXPECT_EQ(s.at("qwen.model.ffn_gate_q4_layers").second, "i32");
        }
      }
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
      [](auto &m) { m.layers.pop_back(); },
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
    for (int kind : {0, 1, 2}) {
      const bool attention = kind == 1;
      const bool route = kind == 2;
      if ((attention && !llama) || (route && llama))
        continue;
      std::vector<uint64_t> tuple;
      if (route) {
        const auto &l = m.layers[layer];
        const bool gdn = l.kind == model::LayerKind::kGatedDeltaNet;
        tuple = {gdn ? 1u : 0u, layer,
                 gdn ? *l.recurrent_ordinal : *l.attention_ordinal};
      } else if (attention) {
        for (const char *r : {"query", "key", "value", "attention_output"})
          tuple.push_back(role(r).storage_type);
        for (const char *r : {"query", "key", "value", "attention_output"})
          tuple.push_back(role(r).byte_size);
      } else {
        tuple.push_back(m.layers[layer].feed_forward_width);
        for (const char *r : {"ffn_gate", "ffn_up", "ffn_down"})
          tuple.push_back(role(r).storage_type);
        for (const char *r : {"ffn_gate", "ffn_up", "ffn_down"})
          tuple.push_back(role(r).byte_size);
      }
      const std::string name = std::string("probe_") +
                               (route       ? "route_"
                                : attention ? "attention_"
                                            : "ffn_") +
                               std::to_string(layer);
      expected.emplace(name, tuple);
      std::string types, values;
      for (size_t i = 0; i < tuple.size(); ++i) {
        types += (i ? ", index" : "index");
        values += (i ? ", %v" : "%v") + std::to_string(i);
      }
      source << "func.def public pure @" << name << "() -> (" << types
             << ") {\n"
             << "  %layer = index.constant " << layer << " : index\n  "
             << values << " = func.call pure @loomify.config.layer_"
             << (route       ? "route"
                 : attention ? "attention"
                             : "ffn")
             << "(%layer) : (index) -> (" << types << ")\n  func.return "
             << values << " : " << types << "\n}\n";
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
