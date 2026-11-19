// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Public expert composition contract through the ordinary native product
// linker.
#include "lib/loomify/linking.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/status.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
std::string Message(loomc_status_t status) {
  const auto message = loomc_status_message(status);
  std::string text(message.data ? message.data : "", message.size);
  loomc_status_free(status);
  return text;
}
std::string Runfile(const std::string &relative) {
  const char *directory = std::getenv("TEST_SRCDIR");
  const char *workspace = std::getenv("TEST_WORKSPACE");
  return std::string(directory ? directory : ".") + "/" +
         (workspace ? workspace : "_main") + "/" + relative;
}
struct Case {
  std::string name;
  int capacity = 1;
  int gate_tag = 12;
  int up_tag = 12;
  int down_tag = 12;
  int64_t gate_bytes = 113246208;
  int64_t up_bytes = 113246208;
  int64_t down_bytes = 113246208;
  bool expected = true;
  bool bad_router_shape = false;
  bool bad_gate_rank = false;
  bool matmul = false;
  int width = 768;
  int columns = 2048;
  int experts = 128;
  int assignments_per_token = 8;
  int input_repeat = 1;
};
int64_t WeightBytes(int tag) {
  // Independent fixed-profile wire extents: 128 * 2048 * 768 values.
  switch (tag) {
  case 12:
    return 113246208;
  case 13:
  case 6:
    return 138412032;
  case 14:
    return 165150720;
  case 8:
    return 213909504;
  }
  return 0;
}
std::vector<Case> Cases() {
  std::vector<Case> cases = {
      {"q4"},
      {"q6", 17, 12, 12, 14, 113246208, 113246208, 165150720},
      // The former unsupported-Q5 case still rejects its mismatched Q4 extent.
      {"wrong_down_tag_extent", 1, 12, 12, 13, 113246208, 113246208, 113246208,
       false},
      {"wrong_q4_down_extent", 1, 12, 12, 12, 113246208, 113246208, 165150720,
       false},
      {"wrong_q6_down_extent", 1, 12, 12, 14, 113246208, 113246208, 113246208,
       false},
  };
  for (int gate : {12, 13, 14, 8})
    for (int up : {12, 13, 14, 8})
      cases.push_back(
          {"pair_" + std::to_string(gate) + "_" + std::to_string(up), 1, gate,
           up, 12, WeightBytes(gate), WeightBytes(up), WeightBytes(12)});
  for (int down : {12, 13, 14, 8, 6})
    cases.push_back({"down_" + std::to_string(down), 17, 12, 12, down,
                     WeightBytes(12), WeightBytes(12), WeightBytes(down)});
  cases.push_back({"capacity512", 512});
  for (int capacity : {0, 513}) {
    Case c{"capacity_" + std::to_string(capacity), capacity};
    c.expected = false;
    cases.push_back(c);
  }
  for (int role = 0; role != 3; ++role) {
    for (int tag : {12, 13, 14, 8, 6}) {
      // Q5_0 has a valid encoding but is intentionally down-only.
      Case c{"role_" + std::to_string(role) + "_tag_" + std::to_string(tag)};
      int *slot = role == 0 ? &c.gate_tag : role == 1 ? &c.up_tag : &c.down_tag;
      int64_t *bytes = role == 0   ? &c.gate_bytes
                       : role == 1 ? &c.up_bytes
                                   : &c.down_bytes;
      *slot = tag;
      *bytes = WeightBytes(tag);
      if (role != 2 && tag == 6) {
        c.expected = false;
        cases.push_back(c);
        continue;
      }
      for (int delta : {-1, 1}) {
        Case bad = c;
        bad.name += delta < 0 ? "_short" : "_long";
        (role == 0   ? bad.gate_bytes
         : role == 1 ? bad.up_bytes
                     : bad.down_bytes) += delta;
        bad.expected = false;
        cases.push_back(bad);
      }
    }
    for (int tag : {1, 11, 20, 21, 23, 99}) {
      Case c{"unsupported_role_" + std::to_string(role) + "_tag_" +
             std::to_string(tag)};
      (role == 0 ? c.gate_tag : role == 1 ? c.up_tag : c.down_tag) = tag;
      c.expected = false;
      cases.push_back(c);
    }
    Case overflow{"overflow_role_" + std::to_string(role)};
    (role == 0   ? overflow.gate_bytes
     : role == 1 ? overflow.up_bytes
                 : overflow.down_bytes) = INT64_MAX;
    overflow.expected = false;
    cases.push_back(overflow);
  }
  Case router{"wrong_router_shape"};
  router.expected = false;
  router.bad_router_shape = true;
  cases.push_back(router);
  Case rank{"wrong_gate_view_rank"};
  rank.expected = false;
  rank.bad_gate_rank = true;
  cases.push_back(rank);
  for (bool gate : {false, true}) {
    for (int tag : {12, 13, 14, 8, 6}) {
      Case c{"single_" + std::string(gate ? "gate_" : "down_") +
             std::to_string(tag)};
      c.matmul = true;
      c.down_tag = tag;
      c.down_bytes = WeightBytes(tag);
      c.width = gate ? 2048 : 768;
      c.columns = gate ? 768 : 2048;
      c.input_repeat = gate ? 8 : 1;
      c.expected = !gate || tag != 6;
      cases.push_back(c);
    }
  }
  for (const auto &name :
       {"wrong_K", "wrong_N", "wrong_experts", "wrong_assignments_per_token",
        "wrong_input_repeat"}) {
    Case c{name};
    c.matmul = true;
    c.expected = false;
    if (c.name == "wrong_K")
      c.width = 769;
    if (c.name == "wrong_N")
      c.columns = 2049;
    if (c.name == "wrong_experts")
      c.experts = 64;
    if (c.name == "wrong_assignments_per_token")
      c.assignments_per_token = 4;
    if (c.name == "wrong_input_repeat")
      c.input_repeat = 8;
    cases.push_back(c);
  }
  return cases;
}
TEST(RoutedMaterialization, TypedFirstProfileAndExactExtents) {
  const auto cases = Cases();
  for (bool bytecode : {false, true}) {
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
    for (const auto &input : cases) {
      SCOPED_TRACE(input.name);
      SCOPED_TRACE(bytecode);
      const std::string router_shape =
          input.bad_router_shape ? "64x4096xf32" : "128x2048xf32";
      const std::string gate_shape =
          input.bad_gate_rank ? "128x884736xi8" : "[%GB]xi8";
      std::ostringstream source;
      if (input.matmul)
        source
            << "template.decl @llm.experts.indexed_matmul(%R:index,%K:index,%N:index,%E:index,%A:index,%I:index,%tag:index,%B:index,%count:buffer,%ids:buffer,%packed:buffer,%weight:view<[%B]xi8>,%out:buffer)\n";
      else
        source
            << "template.decl @llm.ffn.direct_routed(%C:index,%GT:index,%UT:index,%DT:index,%GB:index,%UB:index,%DB:index,%count:buffer,%x:buffer,%router:view<128x2048xf32>,%gate:view<[%GB]xi8>,%up:view<[%UB]xi8>,%down:view<[%DB]xi8>,%out:buffer)\n";
      source
          << "command.def public @probe.root() launch(%count:buffer,%x:buffer,%router:buffer,%gate:buffer,%up:buffer,%down:buffer,%out:buffer) {\n"
          << "  %C = index.constant " << input.capacity << " : index\n"
          << "  %GT = index.constant " << input.gate_tag << " : index\n"
          << "  %UT = index.constant " << input.up_tag << " : index\n"
          << "  %DT = index.constant " << input.down_tag << " : index\n"
          << "  %GB = index.constant " << input.gate_bytes << " : index\n"
          << "  %UB = index.constant " << input.up_bytes << " : index\n"
          << "  %DB = index.constant " << input.down_bytes << " : index\n"
          << "  %origin = index.constant 0 : offset\n"
          << "  %rv = buffer.view %router[%origin] : buffer -> view<"
          << router_shape << ">\n"
          << "  %gv = buffer.view %gate[%origin] : buffer -> view<"
          << gate_shape << ">\n"
          << "  %uv = buffer.view %up[%origin] : buffer -> view<[%UB]xi8>\n"
          << "  %dv = buffer.view %down[%origin] : buffer -> view<[%DB]xi8>\n";
      if (input.matmul) {
        source
            << "  %R = index.constant " << 8 * input.capacity << " : index\n"
            << "  %K = index.constant " << input.width << " : index\n"
            << "  %N = index.constant " << input.columns << " : index\n"
            << "  %E = index.constant " << input.experts << " : index\n"
            << "  %A = index.constant " << input.assignments_per_token
            << " : index\n"
            << "  %I = index.constant " << input.input_repeat << " : index\n"
            << "  template.apply<@llm.experts.indexed_matmul>(%R,%K,%N,%E,%A,%I,%DT,%DB,%count,%router,%x,%dv,%out) : (index,index,index,index,index,index,index,index,buffer,buffer,buffer,view<[%DB]xi8>,buffer)\n";
      } else {
        source
            << "  template.apply<@llm.ffn.direct_routed>(%C,%GT,%UT,%DT,%GB,%UB,%DB,%count,%x,%rv,%gv,%uv,%dv,%out) : (index,index,index,index,index,index,index,buffer,buffer,view<"
            << router_shape << ">,view<" << gate_shape
            << ">,view<[%UB]xi8>,view<[%DB]xi8>,buffer)\n";
      }
      source << "  command.return\n}\n";
      std::string error;
      loomc_module_t *raw_module = nullptr;
      const std::vector<std::string> products = {
          Runfile(std::string("lib/programs/ffn/direct_routed") +
                  (bytecode ? ".loombc" : "_text.loom")),
          Runfile("lib/platforms/amdgpu/gfx11/targets.loom")};
      const bool succeeded = hrx::loomify_gguf::LinkConfiguredProducts(
          context.get(), workspace.get(), source.str(), products,
          {"probe.root"}, true, &raw_module, &error);
      Owner<loomc_module_t, loomc_module_release> module(raw_module,
                                                         loomc_module_release);
      EXPECT_EQ(succeeded, input.expected) << error;
      if (!input.expected) {
        EXPECT_EQ(raw_module, nullptr);
        EXPECT_FALSE(error.empty());
      }
    }
  }
}
} // namespace
