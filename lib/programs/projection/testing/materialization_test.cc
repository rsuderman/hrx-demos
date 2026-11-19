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
#include <utility>
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
  bool prepacked_rows = false;
  bool paired = false;
  int up_tag = 12;
  int up_bytes = 14155776;
  int fixed_count = 1;
  bool matrix_route = false;
};
std::vector<Input> Inputs() {
  std::vector<Input> cases = {
      {"canonical_q4"},
      {"canonical_q5", 1, 13, 768, 1, 4, 25872},
      {"canonical_q6", 1, 14, 768, 1, 4, 30870},
      {"canonical_q8", 1, 8, 768, 1, 4, 39984},
      {"canonical_q6_full_vocabulary", 1, 14, 256, 1, 1, 55063680, 524288,
       262208, true, 262208},
      {"canonical_q8_full_vocabulary", 1, 8, 256, 1, 1, 71320576, 524288,
       262208, true, 262208},
      {"compatible_q6_full_vocabulary_rejected", 0, 14, 256, 1, 1, 55063680,
       524288, 262208, false, 262208},
      {"canonical_q6_past_vocabulary_rejected", 1, 14, 256, 1, 1, 55063890,
       524288, 262208, false, 262209},
      {"canonical_q4_0", 1, 2, 128, 1, 4, 3528},
      {"canonical_q4_1", 1, 3, 128, 1, 4, 3920},
      {"canonical_iq4_xs", 1, 23, 256, 1, 4, 6664},
      {"canonical_iq4_nl_minimum", 1, 20, 128, 1, 4, 3528},
      {"canonical_iq3_s", 1, 21, 256, 1, 4, 5390},
      {"canonical_q3k", 1, 11, 256, 1, 4, 5390},
      {"bad_iq4_nl_wire_extent", 1, 20, 128, 1, 4, 3527, 524288, 262144, false},
      {"bad_iq3_s_wire_extent", 1, 21, 256, 1, 4, 5389, 524288, 262144, false},
      {"bad_q3k_wire_extent", 1, 11, 256, 1, 4, 5391, 524288, 262144, false},
      {"bad_iq4_nl_packet_alignment", 1, 20, 64, 1, 4, 1764, 524288, 262144,
       false},
      {"bad_iq3_s_block_alignment", 1, 21, 128, 1, 4, 2695, 524288, 262144,
       false},
      {"bad_q3k_block_alignment", 1, 11, 128, 1, 4, 2695, 524288, 262144,
       false},
      {"unsupported_compatible_iq4_nl", 0, 20, 128, 1, 4, 3528, 524288, 262144,
       false},
      {"unsupported_compatible_iq3_s", 0, 21, 256, 1, 4, 5390, 524288, 262144,
       false},
      {"unsupported_compatible_q3k", 0, 11, 256, 1, 4, 5390, 524288, 262144,
       false},
      {"bad_q4_0_weight_bytes", 1, 2, 128, 1, 4, 3527, 524288, 262144, false},
      {"bad_q4_1_weight_bytes", 1, 3, 128, 1, 4, 3921, 524288, 262144, false},
      {"bad_iq4_xs_weight_bytes", 1, 23, 256, 1, 4, 6663, 524288, 262144,
       false},
      {"bad_iq4_xs_alignment", 1, 23, 128, 1, 4, 3332, 524288, 262144, false},
      {"unsupported_compatible_q4_0", 0, 2, 128, 1, 4, 3528, 524288, 262144,
       false},
      {"unsupported_compatible_q4_1", 0, 3, 128, 1, 4, 3920, 524288, 262144,
       false},
      {"unsupported_compatible_iq4_xs", 0, 23, 256, 1, 4, 6664, 524288, 262144,
       false},
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
      {"bad_q6_width", 1, 14, 27904, 1, 4, 1121610, 524288, 262144, false},
      {"bad_q8_alignment", 1, 8, 64, 1, 4, 3332, 524288, 262144, false},
      {"bad_weight_bytes", 1, 12, 768, 1, 4, 1, 524288, 262144, false},
      {"bad_capacity", 1, 12, 768, 4, 1, 21168, 524288, 262144, false},
      {"bad_packed_capacity", 1, 12, 768, 1, 4, 21168, 1, 262144, false},
      {"bad_output_capacity", 1, 12, 768, 1, 4, 21168, 524288, 48, false},
      {"packed_f32_ornith_rejected", 1, 0, 4096, 17, 512, 524288, 524288,
       262144, false, 32},
  };
  for (int policy : {0, 1}) {
    for (int width : {17408, 17664, 21504, 22016, 27648}) {
      Input input;
      input.name = "wide_q6_packed_provider";
      input.policy = policy;
      input.tag = 14;
      input.width = width;
      input.rows = input.capacity = 1;
      input.columns = 9;
      input.bytes = width / 256 * 210 * input.columns;
      cases.push_back(input);
      if (width == 27648) {
        input.name = "wide_q6_short_weight_range";
        --input.bytes;
        input.expected = false;
        cases.push_back(input);
      }
    }
  }
  for (int tag : {20, 21, 11}) {
    for (int rows : {1, 17}) {
      Input input;
      input.name = "new_format_matrix_prefix";
      input.tag = tag;
      input.width = tag == 20 ? 128 : 256;
      input.rows = input.capacity = rows;
      input.columns = 7;
      input.bytes = tag == 20 ? 504 : 770;
      input.matrix_route = true;
      cases.push_back(input);
    }
  }
  auto routed = [&](const char *name, int tag, int width, int columns, int rows,
                    int capacity, int bytes, bool expected) {
    Input input;
    input.name = name;
    input.tag = tag;
    input.width = width;
    input.columns = columns;
    input.rows = rows;
    input.capacity = capacity;
    input.bytes = bytes;
    input.expected = expected;
    input.prepacked_rows = true;
    cases.push_back(input);
  };
  routed("routed_f32_ornith", 0, 4096, 32, 17, 512, 524288, true);
  routed("routed_f32_nonsquare", 0, 7, 5, 3, 5, 140, true);
  routed("routed_q5_preserved", 13, 4096, 32, 17, 512, 90112, true);
  routed("routed_q8_preserved", 8, 4096, 32, 17, 512, 139264, true);
  routed("routed_f32_short_weights", 0, 4096, 32, 17, 512, 524284, false);
  routed("routed_f32_long_weights", 0, 4096, 32, 17, 512, 524292, false);
  routed("routed_f16", 1, 4096, 32, 17, 512, 262144, true);
  routed("routed_f16_state", 1, 1024, 16, 8, 17, 32768, true);
  routed("routed_f16_short_weights", 1, 1024, 16, 8, 17, 32766, false);
  routed("routed_f16_long_weights", 1, 1024, 16, 8, 17, 32770, false);
  routed("routed_f16_small_capacity", 1, 1024, 16, 17, 8, 32768, false);
  routed("routed_f16_large_output", 1, 1024, 32769, 8, 17, 67110912, false);
  routed("routed_bad_tag", 99, 4096, 32, 17, 512, 524288, false);
  routed("routed_f32_zero_rows", 0, 4096, 32, 0, 512, 524288, false);
  routed("routed_f32_large_capacity", 0, 4096, 32, 17, 513, 524288, false);
  routed("routed_f32_small_capacity", 0, 4096, 32, 17, 16, 524288, false);
  routed("routed_f32_zero_width", 0, 0, 32, 17, 512, 4, false);
  routed("routed_f32_large_width", 0, 32769, 32, 17, 512, 4194432, false);
  routed("routed_f32_large_output", 0, 4096, 32769, 17, 512, 536887296, false);
  for (int variant = 0; variant < 5; ++variant) {
    Input paired;
    static const char *names[] = {"compatible_pair", "pair_up_format",
                                  "pair_up_extent", "pair_other_rows",
                                  "pair_fixed_count"};
    paired.name = names[variant];
    paired.policy = 0;
    paired.width = 3072;
    paired.columns = 8192;
    paired.rows = paired.capacity = 128;
    paired.bytes = 14155776;
    paired.paired = true;
    paired.fixed_count = -1;
    paired.expected = variant == 0;
    if (variant == 1) {
      paired.up_tag = 14;
      paired.up_bytes = 20643840;
    }
    if (variant == 2)
      --paired.up_bytes;
    if (variant == 3)
      paired.rows = 1;
    if (variant == 4)
      paired.fixed_count = 1;
    cases.push_back(paired);
  }
  return cases;
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
  if (input.matrix_route)
    source
        << "template.decl @llm.projection.matrix(%T:index,%K:index,%N:index,%tag:index,%B:index,%count:buffer,%input:buffer,%weight:view<[%B]xi8>,%output:buffer)\n";
  if (input.paired)
    source
        << "template.decl @llm.ffn.swiglu_packed(%policy:index,%rows:index,%capacity:index,%width:index,%ffn:index,%gt:index,%ut:index,%gb:index,%ub:index,%begin:index,%fixed:i32,%origin:offset,%count:buffer,%packed:buffer,%gate:view<[%gb]xi8>,%up:view<[%ub]xi8>,%output:buffer)\n";
  if (input.prepacked_rows)
    source
        << "template.decl @llm.projection.canonical_rows_prepacked(%rows: index, %capacity: index, %input_width: index, %output_width: index, %tag: index, %weight_bytes: index, %count: buffer, %raw: buffer, %packed: buffer, %weight: view<[%weight_bytes]xi8>, %output: buffer)\n";
  for (const char *format : {"q4k", "q5k", "q6k", "q8_0", "q5_0", "q4_0",
                             "q4_1", "iq4_xs", "iq4_nl", "iq3_s", "q3k"})
    source << "config.def @ggml.linear_" << format
           << "_q8_1_x4.token_capacity = 2048 : index\n"
           << "config.def @ggml.linear_" << format
           << "_q8_1_x4.output_capacity = "
           // Extended vocabulary is a Q6/Q8 contract. Unselected formats
           // retain their existing declaration limits in the complete product.
           << ((input.output_limit > 262144 && std::string(format) != "q6k" &&
                std::string(format) != "q8_0")
                   ? 262144
                   : input.output_limit)
           << " : index\n";
  source
      << "config.def @ggml.quantize_q8_1_x4.group_capacity = "
      << input.group_capacity << " : index\n"
      << "command.def public @probe.root() launch(%count: buffer, %raw: buffer, %packed: buffer, %weights: buffer,"
      << (input.paired ? "%up_weights: buffer," : "") << "%output: buffer) {\n";
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
      << "  %fixed = scalar.constant " << input.fixed_count << " : i32\n"
      << "  %weight = buffer.view %weights[%origin] : buffer -> view<[%bytes]xi8>\n";
  if (input.paired) {
    source
        << "  %ut = index.constant " << input.up_tag << " : index\n"
        << "  %ub = index.constant " << input.up_bytes << " : index\n"
        << "  %up = buffer.view %up_weights[%origin] : buffer -> view<[%ub]xi8>\n"
        << "  template.apply<@llm.ffn.swiglu_packed>(%policy,%rows,%capacity,%width,%cols,%tag,%ut,%bytes,%ub,%begin,%fixed,%origin,%count,%packed,%weight,%up,%output) : (index,index,index,index,index,index,index,index,index,index,i32,offset,buffer,buffer,view<[%bytes]xi8>,view<[%ub]xi8>,buffer)\n";
  } else if (input.matrix_route) {
    source
        << "  template.apply<@llm.projection.matrix>(%rows,%width,%cols,%tag,%bytes,%count,%raw,%weight,%output) : (index,index,index,index,index,buffer,buffer,view<[%bytes]xi8>,buffer)\n";
  } else if (input.prepacked_rows) {
    source
        << "  template.apply<@llm.projection.canonical_rows_prepacked>(%rows, %capacity, %width, %cols, %tag, %bytes, %count, %raw, %packed, %weight, %output) : (index, index, index, index, index, index, buffer, buffer, buffer, view<[%bytes]xi8>, buffer)\n";
  } else {
    source
        << "  template.apply<@llm.projection.pack_input>(%policy, %rows, %capacity, %width, %begin, %fixed, %origin, %count, %raw, %packed) : (index, index, index, index, index, i32, offset, buffer, buffer, buffer)\n"
        << "  template.apply<@llm.projection.project_packed>(%policy, %tag, %rows, %capacity, %width, %cols, %bytes, %begin, %fixed, %origin, %count, %packed, %weight, %output) : (index, index, index, index, index, index, index, index, i32, offset, buffer, buffer, view<[%bytes]xi8>, buffer)\n";
  }
  source << "  command.return\n}\n";
  std::string error;
  loomc_module_t *raw_module = nullptr;
  const auto product =
      Runfile(std::string("lib/programs/projection/testing/provider_") +
              (bytecode ? "bc.loombc" : "text.loom"));
  std::vector<std::string> products = {product};
  if (input.matrix_route) {
    for (const char *name : {"matrix", "prepared_rows"})
      products.push_back(Runfile(std::string("lib/programs/projection/") +
                                 name + (bytecode ? ".loombc" : "_text.loom")));
    products.push_back(
        Runfile("lib/platforms/amdgpu/gfx11/projection/implementation.loombc"));
    // provider_text/provider_bc already owns the concrete target definitions.
  }
  if (input.paired)
    products.push_back(Runfile(std::string("lib/programs/ffn/gate_up") +
                               (bytecode ? ".loombc" : "_text.loom")));
  const bool succeeded = hrx::loomify_gguf::LinkConfiguredProducts(
      context.get(), workspace.get(), source.str(), products, {"probe.root"},
      true, &raw_module, &error);
  Owner<loomc_module_t, loomc_module_release> module(raw_module,
                                                     loomc_module_release);
  EXPECT_EQ(succeeded, input.expected) << error;
  if (succeeded && (input.tag == 20 || input.tag == 21 || input.tag == 11)) {
    loomc_source_t *raw_text = nullptr;
    status = loomc_module_serialize_text_to_source(
        module.get(), nullptr, loomc_allocator_system(), &raw_text);
    ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
    Owner<loomc_source_t, loomc_source_release> printed(raw_text,
                                                        loomc_source_release);
    const auto bytes = loomc_source_contents(printed.get());
    const std::string text(reinterpret_cast<const char *>(bytes.data),
                           bytes.data_length);
    for (const auto &[tag, name] : std::vector<std::pair<int, const char *>>{
             {20, "iq4_nl"}, {21, "iq3_s"}, {11, "q3k"}}) {
      const auto launch =
          std::string("kernel.launch @ggml_linear_") + name + "_q8_1_x4[";
      EXPECT_EQ(text.find(launch) != std::string::npos, input.tag == tag)
          << text;
    }
    if (input.matrix_route) {
      EXPECT_NE(text.find("kernel.launch @llm.projection.prepare_f32_prefix["),
                std::string::npos);
      EXPECT_NE(text.find("kernel.launch @ggml_quantize_q8_1_x4_f32["),
                std::string::npos);
      EXPECT_EQ(text.find("kernel.launch @llm.projection.prepare_scaled_f16["),
                std::string::npos);
    }
  }
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

TEST(ProjectionComponents, SeparateProductsAcceptLateConfiguration) {
  // Each input is already independently merged. Supplying configuration here
  // exercises actual cross-product references, not a flattened source closure.
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode);
    loomc_target_environment_t *raw_environment = nullptr;
    auto status = loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &raw_environment);
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
    Owner<loomc_context_t, loomc_context_release> context(
        raw_context, loomc_context_release);
    loomc_workspace_t *raw_workspace = nullptr;
    status = loomc_workspace_create(nullptr, loomc_allocator_system(),
                                    &raw_workspace);
    ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
    Owner<loomc_workspace_t, loomc_workspace_release> workspace(
        raw_workspace, loomc_workspace_release);
    const auto product = [&](const char *directory, const char *name) {
      return Runfile(std::string(directory) + "/" + name +
                     (bytecode ? ".loombc" : "_text.loom"));
    };
    const auto packed = product("lib/programs/projection", "packed");
    const auto prepared = product("lib/programs/projection", "prepared_rows");
    const auto gate_up = product("lib/programs/ffn", "gate_up");
    const auto matrix = product("lib/programs/projection", "matrix");
    const auto matrix_gate_up = product("lib/programs/ffn", "matrix_gate_up");
    const auto schedule =
        Runfile("lib/platforms/amdgpu/gfx11/projection/implementation.loombc");
    const auto targets = Runfile("lib/platforms/amdgpu/gfx11/targets.loom");
    const std::string matrix_declaration =
        "template.decl @llm.projection.matrix(%T: index, %K: index, %N: index, %tag: index, %B: index, %count: buffer, %input: buffer, %weight: view<[%B]xi8>, %output: buffer)\n";
    const std::string matrix_gate_up_declaration =
        "template.decl @llm.projection.gated_matrix(%T: index, %K: index, %N: index, %gt: index, %ut: index, %GB: index, %UB: index, %count: buffer, %input: buffer, %gate: view<[%GB]xi8>, %up: view<[%UB]xi8>, %output: buffer)\n";
    const std::string prepared_declaration =
        "template.decl @llm.projection.policy_rows(%profile: index, %rows: index, %K: index, %N: index, %tag: index, %B: index, %count: buffer, %input: buffer, %weight: view<[%B]xi8>, %output: buffer)\n";
    const std::string packed_declarations =
        "template.decl @llm.projection.pack_input(%policy: index, %rows: index, %capacity: index, %K: index, %begin: index, %fixed: i32, %origin: offset, %count: buffer, %input: buffer, %packed: buffer)\n"
        "template.decl @llm.projection.project_packed(%policy: index, %tag: index, %rows: index, %capacity: index, %K: index, %N: index, %B: index, %begin: index, %fixed: i32, %origin: offset, %count: buffer, %packed: buffer, %weight: view<[%B]xi8>, %output: buffer)\n";
    const std::string qkv_declaration =
        "template.decl @llm.projection.policy_qkv(%profile: index, %rows: index, %K: index, %Q: index, %KV: index, %qt: index, %kt: index, %vt: index, %QB: index, %KB: index, %VB: index, %count: buffer, %input: buffer, %qweight: view<[%QB]xi8>, %kweight: view<[%KB]xi8>, %vweight: view<[%VB]xi8>, %q: buffer, %k: buffer, %v: buffer)\n";
    const std::string matrix_qkv_declaration =
        "template.decl @llm.projection.matrix_qkv(%rows: index, %K: index, %Q: index, %KV: index, %qt: index, %kt: index, %vt: index, %QB: index, %KB: index, %VB: index, %count: buffer, %input: buffer, %qweight: view<[%QB]xi8>, %kweight: view<[%KB]xi8>, %vweight: view<[%VB]xi8>, %q: buffer, %k: buffer, %v: buffer)\n";
    const std::string gate_up_declaration =
        "template.decl @llm.ffn.swiglu_packed(%policy: index, %rows: index, %capacity: index, %K: index, %N: index, %gt: index, %ut: index, %GB: index, %UB: index, %begin: index, %fixed: i32, %origin: offset, %count: buffer, %packed: buffer, %gate: view<[%GB]xi8>, %up: view<[%UB]xi8>, %output: buffer)\n";
    struct Probe {
      const char *name;
      bool prepared, gate_up, omit_dependency;
      int up_tag;
      int qkv = 0;
      int matrix = 0;
    };
    for (const auto &probe :
         {Probe{"packed_only", false, false, false, 14},
          Probe{"prepared", true, false, false, 14},
          Probe{"gate_up_mixed", false, true, false, 14},
          Probe{"gate_up_fused", false, true, false, 12},
          Probe{"missing_prepared", true, false, true, 14},
          Probe{"missing_gate_up", false, true, true, 14},
          Probe{"qkv_compatible", true, false, false, 14, 1},
          Probe{"qkv_canonical", true, false, false, 14, 2},
          Probe{"qkv_mixed", true, false, false, 14, 3},
          Probe{"qkv_iq_profile0_t16", true, false, false, 14, 4},
          Probe{"qkv_iq_profile1_t16", true, false, false, 14, 5},
          Probe{"qkv_iq_profile0_t17", true, false, false, 14, 6},
          Probe{"qkv_iq_profile1_t17", true, false, false, 14, 7},
          Probe{"matrix_qkv_iq_t16", true, false, false, 14, 8},
          Probe{"matrix_qkv_iq_t17", true, false, false, 14, 9},
          Probe{"matrix", true, false, false, 14, 0, 1},
          Probe{"matrix_gate_up", true, true, false, 14, 0, 2}}) {
      SCOPED_TRACE(probe.name);
      std::ostringstream source;
      source << (probe.qkv >= 8      ? matrix_qkv_declaration
                 : probe.matrix == 2 ? matrix_gate_up_declaration
                 : probe.matrix == 1 ? matrix_declaration
                 : probe.qkv         ? qkv_declaration
                 : probe.gate_up     ? gate_up_declaration
                 : probe.prepared    ? prepared_declaration
                                     : packed_declarations);
      for (const char *format : {"q4k", "q5k", "q6k", "q5_0", "q8_0", "q4_0",
                                 "q4_1", "iq4_xs", "iq4_nl", "iq3_s", "q3k"})
        source << "config.def @ggml.linear_" << format
               << "_q8_1_x4.token_capacity = "
               << (probe.matrix || probe.qkv >= 4 ? 17 : 1) << " : index\n"
               << "config.def @ggml.linear_" << format
               << "_q8_1_x4.output_capacity = 256 : index\n";
      source
          << "config.def @ggml.quantize_q8_1_x4.group_capacity = 34 : index\n"
          << "command.def public @component.root() launch(%count: buffer, %input: buffer, %packed: buffer, %weights: buffer, %up: buffer, %output: buffer, %vweights: buffer, %koutput: buffer, %voutput: buffer) {\n"
          << "  %one = index.constant 1 : index\n"
          << "  %zero = index.constant 0 : index\n"
          << "  %origin = index.constant 0 : offset\n"
          << "  %fixed = scalar.constant -1 : i32\n"
          << "  %K = index.constant 256 : index\n"
          << "  %N = index.constant 256 : index\n"
          << "  %tag = index.constant " << (probe.gate_up ? 12 : 8)
          << " : index\n"
          << "  %B = index.constant " << (probe.gate_up ? 36864 : 69632)
          << " : index\n"
          << "  %weight = buffer.view %weights[%origin] : buffer -> view<[%B]xi8>\n";
      if (probe.matrix) {
        source
            << "  %rows = index.constant 17 : index\n"
            << "  %mt = index.constant 12 : index\n"
            << "  %MB = index.constant 36864 : index\n"
            << "  %mw = buffer.view %weights[%origin] : buffer -> view<[%MB]xi8>\n";
        if (probe.matrix == 2) {
          source
              << "  %mu = buffer.view %up[%origin] : buffer -> view<[%MB]xi8>\n"
              << "  template.apply<@llm.projection.gated_matrix>(%rows, %K, %N, %mt, %mt, %MB, %MB, %count, %input, %mw, %mu, %output) : (index, index, index, index, index, index, index, buffer, buffer, view<[%MB]xi8>, view<[%MB]xi8>, buffer)\n";
        } else {
          source
              << "  template.apply<@llm.projection.matrix>(%rows, %K, %N, %mt, %MB, %count, %input, %mw, %output) : (index, index, index, index, index, buffer, buffer, view<[%MB]xi8>, buffer)\n";
        }
      } else if (probe.qkv) {
        const bool iq = probe.qkv >= 4;
        const bool canonical =
            probe.qkv == 2 || probe.qkv == 5 || probe.qkv == 7;
        const int qt = probe.qkv == 1 || iq ? 12 : 8;
        const int qb = qt == 12 ? 36864 : 69632;
        source
            << "  %profile = index.constant " << (canonical ? 1 : 0)
            << " : index\n"
            << "  %R = index.constant "
            << (probe.qkv == 6 || probe.qkv == 7 || probe.qkv == 9 ? 17
                : iq                                               ? 16
                                                                   : 1)
            << " : index\n"
            << "  %qt = index.constant " << qt << " : index\n"
            << "  %kt = index.constant " << (iq ? 23 : 12) << " : index\n"
            << "  %vt = index.constant 14 : index\n"
            << "  %QB = index.constant " << qb << " : index\n"
            << "  %KB = index.constant " << (iq ? 34816 : 36864) << " : index\n"
            << "  %VB = index.constant 53760 : index\n"
            << "  %qw = buffer.view %weights[%origin] : buffer -> view<[%QB]xi8>\n"
            << "  %kw = buffer.view %up[%origin] : buffer -> view<[%KB]xi8>\n"
            << "  %vw = buffer.view %vweights[%origin] : buffer -> view<[%VB]xi8>\n";
        if (probe.qkv >= 8)
          source
              << "  template.apply<@llm.projection.matrix_qkv>(%R, %K, %N, %N, %qt, %kt, %vt, %QB, %KB, %VB, %count, %input, %qw, %kw, %vw, %output, %koutput, %voutput) : (index, index, index, index, index, index, index, index, index, index, buffer, buffer, view<[%QB]xi8>, view<[%KB]xi8>, view<[%VB]xi8>, buffer, buffer, buffer)\n";
        else
          source
              << "  template.apply<@llm.projection.policy_qkv>(%profile, %R, %K, %N, %N, %qt, %kt, %vt, %QB, %KB, %VB, %count, %input, %qw, %kw, %vw, %output, %koutput, %voutput) : (index, index, index, index, index, index, index, index, index, index, index, buffer, buffer, view<[%QB]xi8>, view<[%KB]xi8>, view<[%VB]xi8>, buffer, buffer, buffer)\n";
      } else if (probe.gate_up) {
        source
            << "  %ut = index.constant " << probe.up_tag << " : index\n"
            << "  %UB = index.constant " << (probe.up_tag == 12 ? 36864 : 53760)
            << " : index\n"
            << "  %up_weight = buffer.view %up[%origin] : buffer -> view<[%UB]xi8>\n"
            << "  template.apply<@llm.ffn.swiglu_packed>(%one, %one, %one, %K, %N, %tag, %ut, %B, %UB, %zero, %fixed, %origin, %count, %packed, %weight, %up_weight, %output) : (index, index, index, index, index, index, index, index, index, index, i32, offset, buffer, buffer, view<[%B]xi8>, view<[%UB]xi8>, buffer)\n";
      } else if (probe.prepared) {
        source
            << "  template.apply<@llm.projection.policy_rows>(%zero, %one, %K, %N, %tag, %B, %count, %input, %weight, %output) : (index, index, index, index, index, index, buffer, buffer, view<[%B]xi8>, buffer)\n";
      } else {
        source
            << "  template.apply<@llm.projection.pack_input>(%one, %one, %one, %K, %zero, %fixed, %origin, %count, %input, %packed) : (index, index, index, index, index, i32, offset, buffer, buffer, buffer)\n"
            << "  template.apply<@llm.projection.project_packed>(%one, %tag, %one, %one, %K, %N, %B, %zero, %fixed, %origin, %count, %packed, %weight, %output) : (index, index, index, index, index, index, index, index, i32, offset, buffer, buffer, view<[%B]xi8>, buffer)\n";
      }
      source << "  command.return\n}\n";
      std::vector<std::string> products{packed, targets};
      if (!probe.omit_dependency && probe.prepared)
        products.push_back(prepared);
      if (!probe.omit_dependency && probe.gate_up)
        products.push_back(gate_up);
      if (probe.matrix || probe.qkv >= 8) {
        products.push_back(matrix);
        products.push_back(schedule);
        if (probe.matrix == 2)
          products.push_back(matrix_gate_up);
      }
      std::string error;
      loomc_module_t *raw_module = nullptr;
      const bool succeeded = hrx::loomify_gguf::LinkConfiguredProducts(
          context.get(), workspace.get(), source.str(), products,
          {"component.root"}, true, &raw_module, &error);
      Owner<loomc_module_t, loomc_module_release> module(raw_module,
                                                         loomc_module_release);
      EXPECT_EQ(succeeded, !probe.omit_dependency) << error;
      if (succeeded && probe.qkv) {
        loomc_source_t *raw_printed = nullptr;
        status = loomc_module_serialize_text_to_source(
            module.get(), nullptr, loomc_allocator_system(), &raw_printed);
        ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
        Owner<loomc_source_t, loomc_source_release> printed(
            raw_printed, loomc_source_release);
        const auto bytes = loomc_source_contents(printed.get());
        const std::string text(reinterpret_cast<const char *>(bytes.data),
                               bytes.data_length);
        const auto count = [&](const std::string &needle) {
          size_t n = 0;
          for (size_t pos = 0;
               (pos = text.find(needle, pos)) != std::string::npos;
               pos += needle.size())
            ++n;
          return n;
        };
        const bool canonical =
            probe.qkv == 2 || probe.qkv == 5 || probe.qkv == 7;
        EXPECT_EQ(count("kernel.launch @ggml_quantize_q8_1_x4_qk_compat_f32["),
                  canonical || probe.qkv >= 8 ? 0u : 1u)
            << text;
        EXPECT_EQ(count("kernel.launch @ggml_quantize_q8_1_x4_f32["),
                  probe.qkv == 8   ? 3u
                  : probe.qkv == 1 ? 0u
                                   : 1u)
            << text;
        EXPECT_EQ(count("kernel.launch @llm.projection.prepare_f32_prefix["),
                  probe.qkv == 8   ? 3u
                  : probe.qkv == 1 ? 0u
                                   : 1u)
            << text;
        if (probe.qkv >= 4)
          EXPECT_EQ(count("kernel.launch @ggml_linear_iq4_xs_q8_1_x4["), 1u)
              << text;
        if (probe.qkv >= 8) {
          EXPECT_EQ(count("kernel.launch @llm.projection.prepare_scaled_f16["),
                    probe.qkv == 9 ? 2u : 0u)
              << text;
          EXPECT_EQ(count("kernel.launch @llm.platform.q4k_scaled_f16_store["),
                    probe.qkv == 9 ? 1u : 0u)
              << text;
          EXPECT_EQ(count("kernel.launch @llm.platform.q6k_scaled_f16_store["),
                    probe.qkv == 9 ? 1u : 0u)
              << text;
        }
      }
      if (probe.omit_dependency) {
        EXPECT_EQ(raw_module, nullptr);
        EXPECT_NE(error.find(probe.prepared ? "policy_rows" : "swiglu_packed"),
                  std::string::npos)
            << error;
      }
    }
  }
}
} // namespace
