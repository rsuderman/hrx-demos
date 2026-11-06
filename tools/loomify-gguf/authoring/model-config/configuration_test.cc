// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/pass.h"
#include "loomc/source.h"
#include "tools/loomify-gguf/authoring/model-config/configuration.h"
#include "tools/loomify-gguf/authoring/model-config/testing/fixture.h"
#include "tools/model/llama_configuration.h"
#include "tools/model/qwen_configuration.h"
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
using testing::Fixture;
TEST(LinkedConfigurationFactsTest, OwnsRequirementsAndEmitsDeterministicFacts) {
  LinkedConfigurationFacts a, b;
  std::string error;
  {
    auto m = Fixture();
    ASSERT_TRUE(DeriveLinkedConfiguration(m, "asset", {}, &a, &error)) << error;
    ASSERT_TRUE(DeriveLinkedConfiguration(m, "asset", {}, &b, &error)) << error;
    m.globals.clear();
    m.layers.clear();
  }
  EXPECT_EQ(a.source, b.source);
  ASSERT_EQ(a.parameters.size(), 31u);
  EXPECT_EQ(a.parameters[0].source_asset, "asset");
  EXPECT_EQ(a.parameters[0].tensor.name, "token_embd.weight");
  EXPECT_GT(a.parameters[0].tensor.absolute_offset, 0u);
  EXPECT_NE(
      a.source.find("func.def public pure inline @loomify.config.layer_ffn"),
      std::string::npos);
  EXPECT_NE(a.source.find(
                "func.def public pure inline @loomify.config.layer_attention"),
            std::string::npos);
  EXPECT_NE(a.source.find("llama.attention.cache_base = 1284 : offset"),
            std::string::npos);
  EXPECT_EQ(a.source.find("command."), std::string::npos);
  EXPECT_EQ(a.source.find("kernel."), std::string::npos);
  EXPECT_EQ(a.source.find("alloc"), std::string::npos);
}
TEST(LinkedConfigurationFactsTest, DerivesDirectAndColdEnvelopesAtBoundaries) {
  for (uint64_t t : {1, 128, 512})
    for (uint64_t c : {512, 513}) {
      LinkedConfigurationSettings s;
      s.token_capacity = t;
      s.context_capacity = c;
      LinkedConfigurationFacts f;
      std::string error;
      ASSERT_TRUE(DeriveLinkedConfiguration(Fixture(), "asset", s, &f, &error))
          << error;
      const auto &g = *f.llama_layout;
      auto h = std::max<uint64_t>(t, 129), p = (c + 63) / 64 * 64,
           b = 768 + 4 * h;
      EXPECT_EQ(g.pool_capacity, p);
      EXPECT_EQ(g.cache_base, b);
      EXPECT_EQ(g.prefill.cache, b + 4 * 3 * p * 1024);
      EXPECT_EQ(g.decode.cache, g.prefill.cache);
      EXPECT_EQ(g.prefill.residual, 4 * t * 3072);
      EXPECT_EQ(g.decode.residual, 4 * 3072u);
      EXPECT_EQ(g.epoch_minima[5], 4 * t);
      EXPECT_EQ(g.cold_allocations[4].bytes, 4 * h);
      EXPECT_EQ(g.cold_allocations[1].clear_bytes, b);
      EXPECT_EQ(g.cold_allocations[2].bytes, 2064u);
      EXPECT_EQ(g.cold_allocations[5].bytes, 64u);
      EXPECT_EQ(g.cold_allocations[10].bytes, 0u);
      EXPECT_EQ(g.row_views, (std::array<uint64_t, 8>{0, 12, 512, 0, 512, 4 * h,
                                                      512 + 4 * h, 32}));
      EXPECT_EQ(g.origins[2], b);
      EXPECT_EQ(g.origins.back(), 0u);
      EXPECT_EQ(g.pooled_geometry,
                (std::array<uint64_t, 13>{64, 2048, 0, 64, 256, b, 4, 1, 1, b,
                                          6, 2 * p * 1024, 128 * 1024}));
    }
}
TEST(LinkedConfigurationFactsTest,
     RejectsInvalidDescriptorsWithoutChangingOutput) {
  using Mutation = std::function<void(model::ModelDescriptor &)>;
  const Mutation cases[] = {
      [](auto &m) { m.family = "other"; },
      [](auto &m) { m.rms_epsilon = std::numeric_limits<double>::quiet_NaN(); },
      [](auto &m) { m.output_tied_to_embedding = false; },
      [](auto &m) { m.globals.back().tensor.absolute_offset++; },
      [](auto &m) { m.globals.back().tensor.name = "output.weight"; },
      [](auto &m) { m.rope.base = 10000; },
      [](auto &m) { m.layers[1].index = 0; },
      [](auto &m) { m.layers[1].index = 4; },
      [](auto &m) { m.layers[1].feed_forward_width++; },
      [](auto &m) { m.layers[1].tensors.pop_back(); },
      [](auto &m) { m.layers[1].tensors.push_back(m.layers[1].tensors[0]); },
      [](auto &m) { m.layers[1].tensors[1].tensor.storage_type = 14; },
      [](auto &m) { m.layers[1].tensors[1].tensor.byte_size++; },
      [](auto &m) { m.layers[1].tensors[1].tensor.normalized_extents[0]++; },
      [](auto &m) {
        m.layers[1].tensors[1].tensor.absolute_offset = UINT64_MAX;
      },
      [](auto &m) { m.layers.clear(); }};
  for (const auto &change : cases) {
    auto m = Fixture();
    change(m);
    LinkedConfigurationFacts out;
    out.source = "sentinel";
    std::string error;
    EXPECT_FALSE(DeriveLinkedConfiguration(m, "asset", {}, &out, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(out.source, "sentinel");
  }
}
TEST(LinkedConfigurationFactsTest, RejectsInvalidDeploymentSettings) {
  for (uint64_t t : std::array<uint64_t, 3>{0, 513, UINT64_MAX}) {
    LinkedConfigurationSettings s;
    s.token_capacity = t;
    LinkedConfigurationFacts f;
    std::string error;
    EXPECT_FALSE(DeriveLinkedConfiguration(Fixture(), "asset", s, &f, &error));
  }
  for (uint64_t c : std::array<uint64_t, 4>{0, 127, 131073, UINT64_MAX}) {
    LinkedConfigurationSettings s;
    s.context_capacity = c;
    LinkedConfigurationFacts f;
    std::string error;
    EXPECT_FALSE(DeriveLinkedConfiguration(Fixture(), "asset", s, &f, &error));
  }
  LinkedConfigurationSettings s;
  s.pool_capacity = 4097;
  LinkedConfigurationFacts f;
  std::string error;
  EXPECT_FALSE(DeriveLinkedConfiguration(Fixture(), "asset", s, &f, &error));
}
TEST(LinkedConfigurationFactsTest,
     QwenFfnAdmitsOnlyRequiredFactsWithoutCacheState) {
  auto m = Fixture(false);
  LinkedConfigurationSettings s;
  s.profile = LinkedProfile::Qwen08BFfn;
  s.context_capacity = 0; // No artificial cache context is required.
  LinkedConfigurationFacts f;
  std::string error;
  ASSERT_TRUE(DeriveLinkedConfiguration(m, "qwen", s, &f, &error)) << error;
  EXPECT_FALSE(f.llama_layout);
  EXPECT_EQ(f.parameters.size(), 12u);
  EXPECT_EQ(f.source.find("cache"), std::string::npos);
  EXPECT_EQ(f.source.find("layer_attention"), std::string::npos);
  EXPECT_NE(f.source.find("loomify.ffn.profile = 1 : index"),
            std::string::npos);
  auto &up = m.layers[1].tensors[2].tensor;
  up.storage_type = 14;
  up.byte_size = 1024 / 256 * 3584 * 210;
  EXPECT_FALSE(DeriveLinkedConfiguration(m, "qwen", s, &f, &error));
  EXPECT_NE(error.find("matching"), std::string::npos);
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
class LinkedConfigurationNativeTest : public ::testing::Test {
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
TEST_F(LinkedConfigurationNativeTest,
       UnsealedTextAndBytecodeRoundTripRetainFacts) {
  for (bool llama : {true, false})
    for (size_t layers : {1, 3}) {
      LinkedConfigurationSettings s;
      s.profile =
          llama ? LinkedProfile::Llama3BModel : LinkedProfile::Qwen08BFfn;
      LinkedConfiguration cfg;
      loomc_result_t *result = nullptr;
      ASSERT_TRUE(Ok(CreateLinkedConfiguration(context, workspace,
                                               Fixture(llama, layers), "asset",
                                               s, &cfg, &result)));
      ASSERT_TRUE(result && loomc_result_succeeded(result))
          << Diagnostics(result);
      loomc_result_release(result);
      ASSERT_NE(cfg.module, nullptr);
      for (bool bytecode : {false, true}) {
        loomc_source_t *source = nullptr;
        ASSERT_TRUE(Ok(bytecode ? loomc_module_serialize_bytecode_to_source(
                                      cfg.module.get(), nullptr,
                                      loomc_allocator_system(), &source)
                                : loomc_module_serialize_text_to_source(
                                      cfg.module.get(), nullptr,
                                      loomc_allocator_system(), &source)));
        loomc_module_t *roundtrip = nullptr;
        result = nullptr;
        ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
            context, workspace, source, nullptr, loomc_allocator_system(),
            &roundtrip, &result)));
        ASSERT_TRUE(result && loomc_result_succeeded(result))
            << Diagnostics(result);
        loomc_result_release(result);
        loomc_source_release(source);
        loomc_source_t *printed = nullptr;
        ASSERT_TRUE(Ok(loomc_module_serialize_text_to_source(
            roundtrip, nullptr, loomc_allocator_system(), &printed)));
        auto bytes = loomc_source_contents(printed);
        std::string text(reinterpret_cast<const char *>(bytes.data),
                         bytes.data_length);
        EXPECT_NE(text.find("loomify.config.layer_ffn"), std::string::npos);
        EXPECT_NE(text.find("loomify.ffn.profile"), std::string::npos);
        if (llama) {
          EXPECT_NE(text.find("loomify.config.layer_attention"),
                    std::string::npos);
        }
        loomc_source_release(printed);
        loomc_module_release(roundtrip);
      }
      auto *identity = cfg.module.get();
      cfg.facts.source = "sentinel";
      auto bad = Fixture(llama, layers);
      bad.embedding_width = 1;
      result = nullptr;
      auto status = CreateLinkedConfiguration(context, workspace, bad, "asset",
                                              s, &cfg, &result);
      EXPECT_FALSE(loomc_status_is_ok(status));
      loomc_status_free(status);
      EXPECT_EQ(cfg.module.get(), identity);
      EXPECT_EQ(cfg.facts.source, "sentinel");
      EXPECT_EQ(result, nullptr);
    }
}
// Exercise the native inliner and scalar folder, independently of later model
// command construction, so tuple order and per-layer cases are observable.
void CheckLookupTuples(loomc_context_t *context, loomc_workspace_t *workspace,
                       const model::ModelDescriptor &m, bool llama) {
  LinkedConfigurationSettings settings;
  settings.profile =
      llama ? LinkedProfile::Llama3BModel : LinkedProfile::Qwen08BFfn;
  LinkedConfigurationFacts facts;
  std::string error;
  ASSERT_TRUE(DeriveLinkedConfiguration(m, "asset", settings, &facts, &error))
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
    for (bool attention : {false, true}) {
      if (attention && !llama)
        continue;
      std::vector<uint64_t> tuple;
      if (attention) {
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
                               (attention ? "attention_" : "ffn_") +
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
             << (attention ? "attention" : "ffn") << "(%layer) : (index) -> ("
             << types << ")\n  func.return " << values << " : " << types
             << "\n}\n";
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
TEST_F(LinkedConfigurationNativeTest, SpecializesFirstMiddleLastLookupTuples) {
  for (bool llama : {true, false})
    CheckLookupTuples(context, workspace, Fixture(llama), llama);
}
#ifdef LOOMIFY_CONFIGURATION_REAL_ASSETS
TEST_F(LinkedConfigurationNativeTest,
       RealDescriptorsSpecializeEveryLayerTuple) {
  for (bool llama : {true, false}) {
    const char *path =
        std::getenv(llama ? "LOOMIFY_TEST_GGUF" : "LOOMIFY_TEST_QWEN_GGUF");
    ASSERT_NE(path, nullptr);
    FILE *file = std::fopen(path, "rb");
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(std::setvbuf(file, nullptr, _IONBF, 0), 0);
    model::ModelDescriptor descriptor;
    std::string error;
    bool loaded =
        llama ? model::ExtractLlamaConfiguration(file, &descriptor, &error)
              : model::ExtractQwenConfiguration(file, &descriptor, &error);
    std::fclose(file);
    ASSERT_TRUE(loaded) << error;
    SCOPED_TRACE(llama ? "all real Llama layer tuples"
                       : "all real Qwen FFN tuples");
    CheckLookupTuples(context, workspace, descriptor, llama);
  }
}
TEST(LinkedConfigurationRealAssetsTest,
     ProductionDescriptorsRetainExactSourceRanges) {
  for (bool llama : {true, false}) {
    const char *path =
        std::getenv(llama ? "LOOMIFY_TEST_GGUF" : "LOOMIFY_TEST_QWEN_GGUF");
    ASSERT_NE(path, nullptr)
        << "explicit real asset gate requires both GGUF paths";
    FILE *file = std::fopen(path, "rb");
    ASSERT_NE(file, nullptr);
    std::setvbuf(file, nullptr, _IONBF, 0);
    model::ModelDescriptor m;
    std::string error;
    bool loaded = llama ? model::ExtractLlamaConfiguration(file, &m, &error)
                        : model::ExtractQwenConfiguration(file, &m, &error);
    std::fclose(file);
    ASSERT_TRUE(loaded) << error;
    LinkedConfigurationSettings s;
    s.profile = llama ? LinkedProfile::Llama3BModel : LinkedProfile::Qwen08BFfn;
    LinkedConfigurationFacts f;
    ASSERT_TRUE(DeriveLinkedConfiguration(m, path, s, &f, &error)) << error;
    EXPECT_EQ(f.parameters.size(),
              llama ? 4 + 9 * m.layers.size() : 4 * m.layers.size());
    for (const auto &p : f.parameters) {
      const auto &bindings = p.layer ? m.layers[*p.layer].tensors : m.globals;
      auto it = std::find_if(bindings.begin(), bindings.end(),
                             [&](const auto &b) { return b.role == p.role; });
      ASSERT_NE(it, bindings.end());
      EXPECT_EQ(p.tensor.name, it->tensor.name);
      EXPECT_EQ(p.tensor.byte_size, it->tensor.byte_size);
      EXPECT_EQ(p.tensor.absolute_offset, it->tensor.absolute_offset);
    }
  }
}
#endif
} // namespace
} // namespace hrx::loomify_gguf
