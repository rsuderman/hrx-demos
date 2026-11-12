// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual operation admission through normal configured product construction.
// Numerical execution is covered by the accompanying GPU fixtures.
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"
#include "gtest/gtest.h"
#include "lib/loomify/linking.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>
#include <memory>
#include <string>

namespace {
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
std::string Message(loomc_status_t status) {
  auto v = loomc_status_message(status);
  std::string text(v.data ? v.data : "", v.size);
  loomc_status_free(status);
  return text;
}
std::string Diagnostics(loomc_result_t *result) {
  std::string text;
  for (size_t i = 0; result && i < loomc_result_diagnostic_count(result); ++i) {
    const auto *d = loomc_result_diagnostic_at(result, i);
    text.append(d->code.data, d->code.size)
        .append(": ")
        .append(d->message.data, d->message.size)
        .append("\n");
  }
  return text;
}

std::string Runfile(const std::string &relative) {
  const char *directory = std::getenv("TEST_SRCDIR");
  const char *workspace = std::getenv("TEST_WORKSPACE");
  return std::string(directory ? directory : ".") + "/" +
         (workspace ? workspace : "_main") + "/" + relative;
}
struct Input {
  const char *name;
  int policy = 1;
  int tag = 12;
  int width = 768;
  int rows = 1;
  int capacity = 4;
  int bytes = 21168;
  int group_capacity = 524288;
  int output_limit = 262144;
  bool expected = true;
  int columns = 49;
};
std::vector<Input> Inputs() {
  return {
      {"canonical_q4"},
      {"canonical_q5", 1, 13, 768, 1, 4, 25872},
      {"canonical_q6", 1, 14, 768, 1, 4, 30870},
      {"canonical_q8", 1, 8, 768, 1, 4, 39984},
      {"compatible_q4", 0},
      {"compatible_q6", 0, 14, 768, 1, 4, 30870},
      {"canonical_q5_0_minimum", 1, 6, 128, 1, 4, 4312},
      {"canonical_q5_0_gemma_kv", 1, 6, 1152, 1, 4, 202752, 524288, 262144,
       true, 256},
      {"canonical_q5_0_gemma_query", 1, 6, 1152, 1, 4, 811008, 524288, 262144,
       true, 1024},
      {"canonical_q5_0_gemma_ffn", 1, 6, 1152, 3, 4, 5474304, 524288, 262144,
       true, 6912},
      {"unsupported_compatible_q5_0", 0, 6, 1152, 1, 4, 202752, 524288, 262144,
       false, 256},
      {"bad_q5_0_weight_bytes", 1, 6, 1152, 1, 4, 202751, 524288, 262144, false,
       256},
      {"bad_q5_0_alignment", 1, 6, 1120, 1, 4, 37730, 524288, 262144, false},
      {"bad_q5_0_width", 1, 6, 32896, 1, 4, 1108184, 524288, 262144, false},
      {"bad_q5_0_capacity", 1, 6, 1152, 4, 1, 202752, 524288, 262144, false,
       256},
      {"bad_q5_0_packed_capacity", 1, 6, 1152, 1, 4, 202752, 1, 262144, false,
       256},
      {"bad_q5_0_output_capacity", 1, 6, 1152, 1, 4, 202752, 524288, 255, false,
       256},
      {"bad_policy", 2, 12, 768, 1, 4, 21168, 524288, 262144, false},
      {"bad_tag", 1, 99, 768, 1, 4, 21168, 524288, 262144, false},
      {"unsupported_compatible_q5", 0, 13, 768, 1, 4, 25872, 524288, 262144,
       false},
      {"unsupported_compatible_q8", 0, 8, 768, 1, 4, 39984, 524288, 262144,
       false},
      {"bad_q6_width", 1, 14, 17664, 1, 4, 710010, 524288, 262144, false},
      {"bad_q8_alignment", 1, 8, 64, 1, 4, 3332, 524288, 262144, false},
      {"bad_weight_bytes", 1, 12, 768, 1, 4, 1, 524288, 262144, false},
      {"bad_capacity", 1, 12, 768, 4, 1, 21168, 524288, 262144, false},
      {"bad_packed_capacity", 1, 12, 768, 1, 4, 21168, 1, 262144, false},
      {"bad_output_capacity", 1, 12, 768, 1, 4, 21168, 524288, 48, false},
  };
}
using Case = std::tuple<Input, bool>;
class ProjectionMaterializationTest : public ::testing::TestWithParam<Case> {};
TEST_P(ProjectionMaterializationTest,
       NormalConstructionEnforcesActualOperations) {
  const auto &input = std::get<0>(GetParam());
  const bool bytecode = std::get<1>(GetParam());
  SCOPED_TRACE(input.name);
  loomc_target_environment_t *raw_environment = nullptr;
  auto status = loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                       &raw_environment);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *raw_context = nullptr;
  status =
      loomc_context_create(&options, loomc_allocator_system(), &raw_context);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_context_t, loomc_context_release> context(raw_context,
                                                        loomc_context_release);
  loomc_workspace_t *raw_workspace = nullptr;
  status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &raw_workspace);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_workspace_t, loomc_workspace_release> workspace(
      raw_workspace, loomc_workspace_release);
  std::ifstream stream(Runfile("lib/programs/projection/interface.loom"));
  ASSERT_TRUE(stream);
  std::ostringstream source;
  source << stream.rdbuf();
  for (const char *format : {"q4k", "q5k", "q6k", "q8_0", "q5_0"})
    source << "config.def @ggml.linear_" << format
           << "_q8_1_x4.token_capacity = 2048 : index\n"
           << "config.def @ggml.linear_" << format
           << "_q8_1_x4.output_capacity = " << input.output_limit
           << " : index\n";
  source
      << "config.def @ggml.quantize_q8_1_x4.group_capacity = "
      << input.group_capacity << " : index\n"
      << "command.def public @probe.root() launch(%count: buffer, %raw: buffer, %packed: buffer, %weights: buffer, %output: buffer) {\n";
  for (const auto &[key, value] :
       std::vector<std::pair<const char *, int>>{{"policy", input.policy},
                                                 {"tag", input.tag},
                                                 {"rows", input.rows},
                                                 {"capacity", input.capacity},
                                                 {"width", input.width},
                                                 {"cols", input.columns},
                                                 {"bytes", input.bytes},
                                                 {"begin", 0}})
    source << "  %" << key << " = index.constant " << value << " : index\n";
  source
      << "  %origin = index.constant 0 : offset\n"
      << "  %fixed = scalar.constant 1 : i32\n"
      << "  %weight = buffer.view %weights[%origin] : buffer -> view<[%bytes]xi8>\n"
      << "  template.apply<@llm.projection.pack_input>(%policy, %rows, %capacity, %width, %begin, %fixed, %origin, %count, %raw, %packed) : (index, index, index, index, index, i32, offset, buffer, buffer, buffer)\n"
      << "  template.apply<@llm.projection.project_packed>(%policy, %tag, %rows, %capacity, %width, %cols, %bytes, %begin, %fixed, %origin, %count, %packed, %weight, %output) : (index, index, index, index, index, index, index, index, i32, offset, buffer, buffer, view<[%bytes]xi8>, buffer)\n"
      << "  command.return\n}\n";
  std::string error;
  loomc_module_t *raw_module = nullptr;
  const auto product =
      Runfile(std::string("lib/programs/projection/testing/provider_") +
              (bytecode ? "bc.loombc" : "text.loom"));
  const bool succeeded = hrx::loomify_gguf::LinkConfiguredProducts(
      context.get(), workspace.get(), source.str(), {product}, {"probe.root"},
      true, &raw_module, &error);
  Owner<loomc_module_t, loomc_module_release> module(raw_module,
                                                     loomc_module_release);
  EXPECT_EQ(succeeded, input.expected) << error;
  if (!input.expected) {
    EXPECT_NE(error.find("all_rejected"), std::string::npos) << error;
    EXPECT_NE(error.find("llm.projection."), std::string::npos) << error;
    EXPECT_EQ(error.find("unresolved exact declaration"), std::string::npos)
        << error;
    EXPECT_EQ(raw_module, nullptr);
  }
}
INSTANTIATE_TEST_SUITE_P(ActualOperations, ProjectionMaterializationTest,
                         ::testing::Combine(::testing::ValuesIn(Inputs()),
                                            ::testing::Bool()));
} // namespace
