// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/status.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"
#include "tools/loomify-gguf/linking.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {
template <typename T, void (*Release)(T*)>
using Owner = std::unique_ptr<T, decltype(Release)>;
std::string Message(loomc_status_t status) {
  auto v = loomc_status_message(status);
  std::string text(v.data ? v.data : "", v.size);
  loomc_status_free(status);
  return text;
}
std::string Runfile(const std::string& path) {
  const char* dir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  return std::string(dir ? dir : ".") + "/" +
      (workspace ? workspace : "_main") + "/" + path;
}
struct Input {
  std::string name;
  int profile = 1;
  int hidden = 2048;
  int ffn = 4096;
  int rows = 1;
  int phase = 1;
  int gate_tag = 12;
  int down_tag = 13;
  int up_tag = 12;
  bool bad_gate_bytes = false;
  bool bad_down_bytes = false;
  bool expected = true;
};
// Fixture byte sizes, not a production admission table.
int Bytes(int tag, int input, int output) {
  if (tag == 8) return input / 32 * output * 34;
  const int block = tag == 12 ? 144 : tag == 13 ? 176 : 210;
  return input / 256 * output * block;
}
std::vector<Input> Inputs() {
  std::vector<Input> result;
  for (int profile : {0, 1}) {
    for (int down_tag : (profile == 0 ? std::vector<int>{12, 14}
                                    : std::vector<int>{13, 14})) {
      for (int rows : {1, 3}) {
        Input v;
        v.name = "positive_" + std::to_string(profile) + "_" +
            std::to_string(down_tag) + "_" + std::to_string(rows);
        v.profile = profile;
        v.hidden = profile == 0 ? 3072 : 2048;
        v.ffn = profile == 0 ? 6144 : 4096;
        v.down_tag = down_tag;
        v.rows = rows;
        v.phase = rows == 1 ? 1 : 0;
        result.push_back(v);
      }
    }
  }
  for (int rows : {1, 3}) {
    Input v;
    v.name = "same_hidden_changed_ffn_" + std::to_string(rows);
    v.hidden = 1024;
    v.rows = rows;
    v.phase = rows == 1 ? 1 : 0;
    result.push_back(v);
  }
  auto bad = [&](const char* name) -> Input& {
    result.push_back(Input{});
    result.back().name = name;
    result.back().expected = false;
    return result.back();
  };
  auto& compat_gate = bad("unsupported_compatible_gate_q5");
  compat_gate.profile = 0;
  compat_gate.gate_tag = compat_gate.up_tag = 13;
  auto& compat_down = bad("unsupported_compatible_down_q5");
  compat_down.profile = 0;
  auto& q8_gate = bad("unsupported_ffn_q8_gate");
  q8_gate.gate_tag = q8_gate.up_tag = 8;
  bad("unknown_down_tag").down_tag = 99;
  bad("wrong_gate_bytes").bad_gate_bytes = true;
  bad("wrong_down_bytes").bad_down_bytes = true;
  bad("hidden_row_alignment").hidden = 1536;
  bad("ffn_block_alignment").ffn = 640;
  auto& q6_limit = bad("q6_input_capability");
  q6_limit.ffn = 17664;
  q6_limit.down_tag = 14;
  bad("decode_more_than_one_row").rows = 3;
  bad("unknown_profile").profile = 2;
  bad("up_tag_mismatch").up_tag = 13;
  return result;
}
using Case = std::tuple<Input, bool>;
class FfnCapabilityTest : public ::testing::TestWithParam<Case> {};
TEST_P(FfnCapabilityTest, ActualFactsReachNativeOperationCapabilities) {
  const Input& v = std::get<0>(GetParam());
  SCOPED_TRACE(v.name);
  const bool bytecode = std::get<1>(GetParam());
  const bool compatible_provider = v.profile == 0;
  const char* family = compatible_provider ? "llama" : "qwen";
  loomc_target_environment_t* raw_environment = nullptr;
  auto status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &raw_environment);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_target_environment_t, loomc_target_environment_release> environment(
      raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t* raw_context = nullptr;
  status = loomc_context_create(&options, loomc_allocator_system(), &raw_context);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_context_t, loomc_context_release> context(raw_context, loomc_context_release);
  loomc_workspace_t* raw_workspace = nullptr;
  status = loomc_workspace_create(nullptr, loomc_allocator_system(), &raw_workspace);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_workspace_t, loomc_workspace_release> workspace(raw_workspace, loomc_workspace_release);
  std::ifstream interface(Runfile("lib/programs/ffn/interface.loom"));
  ASSERT_TRUE(interface);
  std::ostringstream source;
  source << interface.rdbuf();
  source << "template.decl @loomify.ffn.match_up(%gate_tag: index, %up_tag: index, %gate_bytes: index, %up_bytes: index)\n";
  for (const char* format : {"q4k", "q5k", "q6k", "q8_0"})
    source << "config.def @ggml.linear_" << format
           << "_q8_1_x4.token_capacity = 512 : index\n"
           << "config.def @ggml.linear_" << format
           << "_q8_1_x4.output_capacity = 262144 : index\n";
  source << "config.def @ggml.quantize_q8_1_x4.group_capacity = 524288 : index\n"
         << "config.def @" << family << ".model.hidden_size = " << v.hidden << " : index\n"
         << "config.def @" << family << ".model.rms_epsilon = 0.000001 : f32\n";
  if (!compatible_provider)
    source << "config.def @qwen.model.hidden_divisor = " << v.hidden << ".0 : f32\n";
  source << "command.program.def public @capability.root() launch(%residual: buffer, %count: buffer, %gamma_bytes: buffer, %gate_bytes: buffer, %up_bytes: buffer, %down_bytes: buffer) {\n";
  const int gate_bytes = v.bad_gate_bytes ? 1 : Bytes(v.gate_tag, v.hidden, v.ffn);
  const int down_bytes = v.bad_down_bytes ? 1 : Bytes(v.down_tag, v.ffn, v.hidden);
  for (const auto& [key, value] : std::vector<std::pair<const char*, int>>{
      {"hidden", v.hidden}, {"ffn", v.ffn}, {"rows", v.rows},
      {"phase", v.phase}, {"profile", v.profile}, {"gate_tag", v.gate_tag},
      {"up_tag", v.up_tag}, {"down_tag", v.down_tag},
      {"gate_size", gate_bytes}, {"down_size", down_bytes}})
    source << "  %" << key << " = index.constant " << value << " : index\n";
  source << "  %epsilon = scalar.constant 0.000001 : f32\n"
         << "  %origin = index.constant 0 : offset\n"
         << "  %gamma = buffer.view %gamma_bytes[%origin] : buffer -> view<[%hidden]xf32>\n"
         << "  %gate = buffer.view %gate_bytes[%origin] : buffer -> view<[%gate_size]xi8>\n"
         << "  %up = buffer.view %up_bytes[%origin] : buffer -> view<[%gate_size]xi8>\n"
         << "  %down = buffer.view %down_bytes[%origin] : buffer -> view<[%down_size]xi8>\n"
         << "  template.apply<@loomify.ffn.match_up>(%gate_tag, %up_tag, %gate_size, %gate_size) : (index, index, index, index)\n"
         << "  template.apply<@loomify.ffn>(%hidden, %ffn, %rows, %phase, %profile, %epsilon, %gate_tag, %down_tag, %gate_size, %down_size, %residual, %count, %gamma, %gate, %up, %down) : (index, index, index, index, index, f32, index, index, index, index, buffer, buffer, view<[%hidden]xf32>, view<[%gate_size]xi8>, view<[%gate_size]xi8>, view<[%down_size]xi8>)\n"
         << "  command.return\n}\n";
  const auto product = Runfile(std::string("lib/programs/ffn/testing/capability_") +
      (compatible_provider ? "compatible" : "canonical") + "_provider_" +
      (bytecode ? "bc.loombc" : "text.loom"));
  std::string error;
  loomc_module_t* raw_module = nullptr;
  const bool ok = hrx::loomify_gguf::LinkConfiguredProducts(context.get(),
      workspace.get(), source.str(), {product}, {"capability.root"}, true,
      &raw_module, &error);
  Owner<loomc_module_t, loomc_module_release> module(raw_module, loomc_module_release);
  EXPECT_EQ(ok, v.expected) << error;
  if (!v.expected) {
    EXPECT_EQ(raw_module, nullptr);
    EXPECT_NE(error.find("all_rejected"), std::string::npos) << error;
    EXPECT_EQ(error.find("unresolved exact declaration"), std::string::npos) << error;
  }
}
INSTANTIATE_TEST_SUITE_P(ActualConsumers, FfnCapabilityTest,
    ::testing::Combine(::testing::ValuesIn(Inputs()), ::testing::Bool()));
}  // namespace
