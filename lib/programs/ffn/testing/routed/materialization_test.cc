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
  const char *name;
  int capacity = 1;
  int down_tag = 12;
  int gate_bytes = 113246208;
  int up_bytes = 113246208;
  int down_bytes = 113246208;
  bool expected = true;
  bool bad_router_shape = false;
  bool bad_gate_rank = false;
};
TEST(RoutedMaterialization, TypedFirstProfileAndExactExtents) {
  const std::vector<Case> cases = {
      {"q4"},
      {"q6", 17, 14, 113246208, 113246208, 165150720},
      {"capacity512", 512},
      {"capacity0", 0, 12, 113246208, 113246208, 113246208, false},
      {"capacity513", 513, 12, 113246208, 113246208, 113246208, false},
      {"wrong_down_tag", 1, 13, 113246208, 113246208, 113246208, false},
      {"wrong_gate_extent", 1, 12, 113246207, 113246208, 113246208, false},
      {"wrong_up_extent", 1, 12, 113246208, 113246209, 113246208, false},
      {"wrong_q4_down_extent", 1, 12, 113246208, 113246208, 165150720, false},
      {"wrong_q6_down_extent", 1, 14, 113246208, 113246208, 113246208, false},
      {"wrong_router_shape", 1, 12, 113246208, 113246208, 113246208, false,
       true},
      {"wrong_gate_view_rank", 1, 12, 113246208, 113246208, 113246208, false,
       false, true},
  };
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
      source
          << "template.decl @llm.ffn.direct_routed(%C:index,%tag:index,%GB:index,%UB:index,%DB:index,%count:buffer,%x:buffer,%router:view<128x2048xf32>,%gate:view<[%GB]xi8>,%up:view<[%UB]xi8>,%down:view<[%DB]xi8>,%out:buffer)\n"
          << "command.def public @probe.root() launch(%count:buffer,%x:buffer,%router:buffer,%gate:buffer,%up:buffer,%down:buffer,%out:buffer) {\n"
          << "  %C = index.constant " << input.capacity << " : index\n"
          << "  %tag = index.constant " << input.down_tag << " : index\n"
          << "  %GB = index.constant " << input.gate_bytes << " : index\n"
          << "  %UB = index.constant " << input.up_bytes << " : index\n"
          << "  %DB = index.constant " << input.down_bytes << " : index\n"
          << "  %origin = index.constant 0 : offset\n"
          << "  %rv = buffer.view %router[%origin] : buffer -> view<"
          << router_shape << ">\n"
          << "  %gv = buffer.view %gate[%origin] : buffer -> view<"
          << gate_shape << ">\n"
          << "  %uv = buffer.view %up[%origin] : buffer -> view<[%UB]xi8>\n"
          << "  %dv = buffer.view %down[%origin] : buffer -> view<[%DB]xi8>\n"
          << "  template.apply<@llm.ffn.direct_routed>(%C,%tag,%GB,%UB,%DB,%count,%x,%rv,%gv,%uv,%dv,%out) : (index,index,index,index,index,buffer,buffer,view<"
          << router_shape << ">,view<" << gate_shape
          << ">,view<[%UB]xi8>,view<[%DB]xi8>,buffer)\n"
          << "  command.return\n}\n";
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
    }
  }
}
} // namespace
