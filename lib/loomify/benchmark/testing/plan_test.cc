// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/plan.h"
#include "lib/loomify/benchmark/inputs.h"
#include "loomc/source.h"
#include "loomc/interop.h"
#include "loom/ops/op_defs.h"
#include "loomc/target/amdgpu.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <string>

namespace hrx::loomify_gguf::benchmark {
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
class BenchmarkPlanTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(Ok(loomc_target_environment_create_amdgpu(
        loomc_allocator_system(), &environment_)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment_;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(Ok(
        loomc_context_create(&options, loomc_allocator_system(), &context_)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                          &workspace_)));
  }
  void TearDown() override {
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    loomc_target_environment_release(environment_);
  }
  loomc_target_environment_t *environment_ = nullptr;
  loomc_context_t *context_ = nullptr;
  loomc_workspace_t *workspace_ = nullptr;
};
PreparedCase IdentityInputs(const Invocation &invocation,
                            const std::vector<PreparedRoot> &roots,
                            const std::string &reset_policy) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "test.recipe.v1";
  prepared.reset_policy = reset_policy;
  prepared.roots = roots;
  for (const auto &argument : invocation.arguments)
    if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
      const auto &binding = argument.value.buffer;
      uint64_t bytes = binding.byte_length;
      if (bytes == UINT64_MAX)
        for (const auto &root : roots)
          if (root.role == binding.role &&
              root.root_index == binding.root_index)
            bytes = root.bytes.size() -
                    (binding.byte_offset - root.original_byte_offset);
      prepared.buffer_extents.push_back({binding, bytes, 4});
    }
  return prepared;
}
TEST(CaseIdentity, ProjectionPairBindsOrderedOriginalMembersAndMergedInputs) {
  Plan plan;
  plan.workload_id = "pair-workload";
  plan.implementations.push_back({"implementation", {}, {}});
  PreparedCase prepared;
  prepared.setup_recipe_id = "test.projection-pair/v1";
  prepared.measurement_subject = "sequence";
  prepared.reset_policy = "none_overwrite_only";
  for (size_t member = 0; member < 2; ++member) {
    Invocation invocation;
    invocation.command_ordinal = member;
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = 6;
    for (auto &count : invocation.command.payload.dispatch.workgroup_count)
      count = 1;
    for (size_t arg = 0; arg < 6; ++arg) {
      loomc_cmd_argument_t argument{};
      argument.kind = arg < 3 ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                              : LOOMC_CMD_ARGUMENT_KIND_BUFFER;
      if (arg < 3)
        argument.value.scalar_bits = 4;
      else {
        argument.value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                 (member * 3 + arg - 3) * 4, 4};
        prepared.buffer_extents.push_back({argument.value.buffer, 4, 4});
      }
      invocation.arguments.push_back(argument);
    }
    plan.invocations.push_back(std::move(invocation));
  }
  PreparedRoot root;
  root.bytes.resize(28, 1);
  root.minimum_alignment = 4;
  prepared.roots.push_back(root);
  CaseIdentity baseline, changed;
  std::string error;
  ASSERT_TRUE(
      BuildProjectionPairIdentity(plan, {0, 1}, prepared, &baseline, &error))
      << error;
  auto guard_written = prepared;
  guard_written.roots[0].fixture_suffix_bytes = kBenchmarkTailGuardBytes;
  ASSERT_TRUE(BuildProjectionPairIdentity(plan, {0, 1}, guard_written, &changed,
                                          &error));
  EXPECT_EQ(baseline.case_id, changed.case_id);
  prepared.roots[0].bytes[0] ^= 1;
  ASSERT_TRUE(
      BuildProjectionPairIdentity(plan, {0, 1}, prepared, &changed, &error));
  EXPECT_NE(baseline.case_id, changed.case_id);
  prepared.roots[0].bytes[0] ^= 1;
  plan.invocations[1].arguments[5].value.buffer.byte_offset = 24;
  prepared.buffer_extents[5].binding.byte_offset = 24;
  ASSERT_TRUE(
      BuildProjectionPairIdentity(plan, {0, 1}, prepared, &changed, &error));
  EXPECT_NE(baseline.case_id, changed.case_id);
  plan.invocations[1].arguments[0].value.scalar_bits++;
  changed.case_id = "preserved";
  EXPECT_FALSE(
      BuildProjectionPairIdentity(plan, {0, 1}, prepared, &changed, &error));
  EXPECT_EQ(changed.case_id, "preserved");
  plan.invocations[1].arguments[0].value.scalar_bits--;
  EXPECT_FALSE(
      BuildProjectionPairIdentity(plan, {1, 0}, prepared, &changed, &error));
  plan.invocations[1].command_ordinal = 2;
  EXPECT_FALSE(
      BuildProjectionPairIdentity(plan, {0, 1}, prepared, &changed, &error));
}

TEST(CaseIdentity, BindsInputsAbiAndReplayWithoutConfusingImplementation) {
  Plan plan;
  plan.workload_id = "workload";
  plan.implementations.push_back({"implementation-A", {}, {}});
  Invocation invocation;
  invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
  invocation.command.argument_count = 2;
  invocation.command.payload.dispatch.workgroup_count[0] = 1;
  invocation.command.payload.dispatch.workgroup_count[1] = 1;
  invocation.command.payload.dispatch.workgroup_count[2] = 1;
  loomc_cmd_argument_t scalar{};
  scalar.kind = LOOMC_CMD_ARGUMENT_KIND_B32;
  scalar.value.scalar_bits = 0x80000000u;
  invocation.arguments.push_back(scalar);
  loomc_cmd_argument_t buffer{};
  buffer.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
  buffer.value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 3, 100, 16};
  invocation.arguments.push_back(buffer);
  PreparedRoot root;
  root.root_index = 3;
  root.original_byte_offset = 100;
  root.minimum_alignment = 4;
  root.bytes.assign(16, 1);
  std::vector<PreparedRoot> roots{root};
  auto identify = [&](const Plan &candidate_plan, const Invocation &candidate,
                      const std::vector<PreparedRoot> &candidate_roots,
                      const char *reset) {
    CaseIdentity identity;
    std::string error;
    EXPECT_TRUE(BuildCaseIdentity(
        candidate_plan, candidate,
        IdentityInputs(candidate, candidate_roots, reset), &identity, &error))
        << error;
    return identity;
  };
  const auto baseline = identify(plan, invocation, roots, "none");
  EXPECT_FALSE(baseline.case_id.empty());
  auto written_roots = roots;
  written_roots[0].fixture_suffix_bytes = kBenchmarkTailGuardBytes;
  EXPECT_EQ(identify(plan, invocation, written_roots, "none").case_id,
            baseline.case_id);
  auto changed_plan = plan;
  changed_plan.implementations[0].implementation_id = "implementation-B";
  const auto implementation = identify(changed_plan, invocation, roots, "none");
  EXPECT_NE(implementation.case_id, baseline.case_id);
  EXPECT_EQ(implementation.logical_case_key, baseline.logical_case_key);
  auto grid = invocation;
  grid.command.payload.dispatch.workgroup_count[0] = 2;
  const auto schedule = identify(plan, grid, roots, "none");
  EXPECT_NE(schedule.case_id, baseline.case_id);
  EXPECT_EQ(schedule.logical_case_key, baseline.logical_case_key);
  auto moved = invocation;
  moved.arguments[1].value.buffer.root_index = 9;
  moved.arguments[1].value.buffer.byte_offset = 200;
  auto moved_roots = roots;
  moved_roots[0].root_index = 9;
  moved_roots[0].original_byte_offset = 200;
  EXPECT_EQ(identify(plan, moved, moved_roots, "none").case_id,
            baseline.case_id);
  auto aliased = invocation;
  aliased.arguments.push_back(aliased.arguments.back());
  aliased.command.argument_count = 3;
  auto disjoint = aliased;
  disjoint.arguments.back().value.buffer.root_index = 4;
  auto disjoint_roots = roots;
  auto other_root = root;
  other_root.root_index = 4;
  disjoint_roots.push_back(other_root);
  EXPECT_NE(identify(plan, aliased, roots, "none").case_id,
            identify(plan, disjoint, disjoint_roots, "none").case_id);
  auto indirect = invocation;
  indirect.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_STATIC;
  indirect.command.payload.dispatch.workgroup_count_buffer = {
      LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 4, 0, 12};
  auto indirect_roots = roots;
  PreparedRoot count_root;
  count_root.root_index = 4;
  count_root.minimum_alignment = 4;
  count_root.bytes.assign(24, 0);
  count_root.bytes[0] = 1;
  count_root.bytes[4] = count_root.bytes[8] = 1;
  count_root.bytes[12] = 2;
  count_root.bytes[16] = count_root.bytes[20] = 1;
  indirect_roots.push_back(count_root);
  const auto first_counts = identify(plan, indirect, indirect_roots, "none");
  indirect.command.payload.dispatch.workgroup_count_buffer.byte_offset = 12;
  const auto second_counts = identify(plan, indirect, indirect_roots, "none");
  EXPECT_NE(first_counts.logical_case_key, second_counts.logical_case_key);
  EXPECT_NE(first_counts.input_policy_digest,
            second_counts.input_policy_digest);
  auto bits = invocation;
  bits.arguments[0].value.scalar_bits = 0;
  EXPECT_NE(identify(plan, bits, roots, "none").case_id, baseline.case_id);
  EXPECT_NE(identify(plan, bits, roots, "none").input_policy_digest,
            baseline.input_policy_digest);
  auto bytes = roots;
  bytes[0].bytes[0] = 2;
  EXPECT_NE(identify(plan, invocation, bytes, "none").input_policy_digest,
            baseline.input_policy_digest);
  EXPECT_NE(
      identify(plan, invocation, roots, "restore_each_invocation").case_id,
      baseline.case_id);
  auto reset_inputs = IdentityInputs(invocation, roots, "copy_seed");
  reset_inputs.measurement_subject = "reset_plus_kernel";
  StateReset reset;
  reset.target = invocation.arguments[1].value.buffer;
  reset.seed.bytes = roots[0].bytes;
  reset.seed.minimum_alignment = 4;
  reset_inputs.resets = {reset};
  CaseIdentity seeded, changed_seed, changed_target, changed_view,
      changed_workload;
  std::string reset_error;
  ASSERT_TRUE(
      BuildCaseIdentity(plan, invocation, reset_inputs, &seeded, &reset_error));
  reset_inputs.resets[0].seed.bytes[0] ^= 1;
  ASSERT_TRUE(BuildCaseIdentity(plan, invocation, reset_inputs, &changed_seed,
                                &reset_error));
  EXPECT_NE(seeded.input_policy_digest, changed_seed.input_policy_digest);
  reset_inputs.resets[0].seed.bytes = roots[0].bytes;
  reset_inputs.resets[0].target.byte_offset += 4;
  reset_inputs.resets[0].target.byte_length -= 4;
  reset_inputs.resets[0].seed.bytes.resize(12);
  ASSERT_TRUE(BuildCaseIdentity(plan, invocation, reset_inputs, &changed_target,
                                &reset_error));
  EXPECT_NE(seeded.input_policy_digest, changed_target.input_policy_digest);
  auto views = IdentityInputs(invocation, roots, "none");
  views.buffer_extents[0].byte_length = 8;
  ASSERT_TRUE(
      BuildCaseIdentity(plan, invocation, views, &changed_view, &reset_error));
  EXPECT_NE(changed_view.input_policy_digest, baseline.input_policy_digest);
  views = IdentityInputs(invocation, roots, "none");
  views.source_workloads = {8};
  views.source_argument_types = {"i32", "buffer"};
  ASSERT_TRUE(BuildCaseIdentity(plan, invocation, views, &changed_workload,
                                &reset_error));
  EXPECT_NE(changed_workload.input_policy_digest, baseline.input_policy_digest);
  CaseIdentity preserved;
  preserved.case_id = "preserved";
  std::string error;
  EXPECT_FALSE(BuildCaseIdentity(plan, invocation, {}, &preserved, &error));
  EXPECT_EQ(preserved.case_id, "preserved");
}
TEST_F(BenchmarkPlanTest, InvalidProfilesPreserveOutputWithoutOpeningWeights) {
  const std::vector<Workload> invalid = [] {
    std::vector<Workload> result(6);
    result[0].phase = "unknown";
    result[1].active_tokens = 0;
    result[2].history_tokens = result[2].context_capacity + 1;
    result[3].phase = "decode";
    result[4].weight_policy = "unknown";
    result[5].history_tokens = result[5].context_capacity;
    return result;
  }();
  for (const auto &workload : invalid) {
    Plan out;
    out.selection_id = "preserved";
    std::string error;
    EXPECT_FALSE(BuildBenchmarkPlan("missing.gguf", workload, "", context_,
                                    workspace_, &out, &error));
    EXPECT_EQ(out.selection_id, "preserved");
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(error.find("resolve benchmark GGUF"), std::string::npos);
  }
}
TEST_F(BenchmarkPlanTest, CopiesCompleteSelectedScheduleAndOwnsRequests) {
  const char *asset = std::getenv("LOOMIFY_TEST_QWEN_GGUF");
  const char *srcdir = std::getenv("TEST_SRCDIR");
  const char *workspace_name = std::getenv("TEST_WORKSPACE");
  ASSERT_TRUE(asset && srcdir && workspace_name)
      << "explicit immutable Qwen35 asset and runfiles required";
  const auto root = (std::filesystem::path(srcdir) / workspace_name).string();
  std::string previous_workload;
  for (const auto capacity : {1u, 8u}) {
    Workload workload;
    workload.phase = capacity == 1 ? "decode" : "prefill";
    workload.token_capacity = capacity;
    workload.active_tokens = capacity == 1 ? 1 : 3;
    workload.history_tokens = 5;
    Plan plan;
    std::string error;
    ASSERT_TRUE(BuildBenchmarkPlan(asset, workload, root, context_, workspace_,
                                   &plan, &error))
        << error;
    EXPECT_EQ(plan.workload.token_capacity, capacity);
    EXPECT_FALSE(plan.selection_id.empty());
    EXPECT_NE(plan.workload_id, previous_workload);
    previous_workload = plan.workload_id;
    ASSERT_FALSE(plan.invocations.empty());
    size_t dispatches = 0;
    for (const auto &program : plan.programs) {
      EXPECT_EQ(program.commands.size(), program.layout.command_count);
      for (const auto &command : program.commands)
        if (command.kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT ||
            command.kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_STATIC ||
            command.kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC)
          ++dispatches;
    }
    EXPECT_EQ(plan.invocations.size(), dispatches);
    for (const auto &invocation : plan.invocations) {
      ASSERT_LT(invocation.program_ordinal, plan.programs.size());
      ASSERT_LT(invocation.command_ordinal,
                plan.programs[invocation.program_ordinal].commands.size());
      EXPECT_FALSE(invocation.kernel_symbol.empty());
      ASSERT_LT(invocation.implementation_index, plan.implementations.size());
      const auto &implementation =
          plan.implementations[invocation.implementation_index];
      ASSERT_LT(invocation.request_root_ordinal, implementation.roots.size());
      ASSERT_NE(implementation.request, nullptr);
      EXPECT_GT(loomc_source_contents(
                    loomc_request_source(implementation.request.get()))
                    .data_length,
                0u);
      EXPECT_EQ(invocation.arguments.size(), invocation.command.argument_count);
      EXPECT_EQ(invocation.invocation_id,
                "p" + std::to_string(invocation.program_ordinal) + "/c" +
                    std::to_string(invocation.command_ordinal));
    }
  }
}
TEST_F(BenchmarkPlanTest, RecordsActualPublishedRootWorkloadSignatures) {
  const char *srcdir = std::getenv("TEST_SRCDIR");
  const char *workspace_name = std::getenv("TEST_WORKSPACE");
  ASSERT_TRUE(srcdir && workspace_name);
  const auto root = (std::filesystem::path(srcdir) / workspace_name).string();
  std::string rows, convolution_rows;
  struct Profile {
    const char *label;
    const char *variable;
    const char *family;
  };
  const Profile profiles[] = {
      {"qwen35_08", "LOOMIFY_TEST_QWEN_GGUF", "qwen35"},
      {"qwen35_27", "LOOMIFY_TEST_QWEN_27_GGUF", "qwen35"},
      {"llama", "LOOMIFY_TEST_GGUF", "llama"},
      {"qwen3", "HRX_QWEN3_GGUF", "qwen3"},
      {"qwen2", "LOOMIFY_TEST_QWEN2_GGUF", "qwen2"},
      {"gemma3", "LOOMIFY_TEST_GEMMA_GGUF", "gemma3"},
      {"lfm3", "LOOMIFY_TEST_LFM3_GGUF", "lfm3"},
      {"spark2_5", "LOOMIFY_TEST_SPARK_GGUF", "spark2_5"},
  };
  for (const auto &profile : profiles) {
    const char *asset = std::getenv(profile.variable);
    ASSERT_TRUE(asset && *asset)
        << "explicit immutable asset required: " << profile.variable;
    for (const auto capacity : {1u, 8u, 128u}) {
      Workload workload;
      workload.phase = capacity == 1 ? "decode" : "prefill";
      workload.token_capacity = workload.active_tokens = capacity;
      Plan plan;
      std::string error;
      ASSERT_TRUE(BuildBenchmarkPlan(asset, workload, root, context_,
                                     workspace_, &plan, &error))
          << error;
      EXPECT_EQ(plan.model.family, profile.family);
      for (const auto &invocation : plan.invocations) {
        // Investigation output only, never a recipe/admission heuristic.
        if (invocation.kernel_symbol.find("conv") == std::string::npos)
          continue;
        convolution_rows += std::string(profile.label) + "/" + workload.phase +
                            "/T" + std::to_string(capacity) + "/" +
                            invocation.invocation_id + "/" +
                            invocation.kernel_symbol +
                            " kind=" + std::to_string(invocation.command.kind);
        if (invocation.command.kind == LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT) {
          convolution_rows += " grid=";
          for (auto count : invocation.command.payload.dispatch.workgroup_count)
            convolution_rows += std::to_string(count) + ",";
        }
        for (const auto &argument : invocation.arguments) {
          convolution_rows += " arg" + std::to_string(argument.kind) + "=";
          if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
            const auto &buffer = argument.value.buffer;
            convolution_rows += std::to_string(buffer.role) + ":" +
                                std::to_string(buffer.root_index) + ":" +
                                std::to_string(buffer.byte_offset) + ":" +
                                std::to_string(buffer.byte_length);
            if (buffer.role == LOOMC_CMD_BUFFER_ROLE_FIXED)
              for (const auto &parameter :
                   plan.programs[invocation.program_ordinal].parameters)
                if (parameter.fixed_buffer_index == buffer.root_index &&
                    buffer.byte_offset >= parameter.byte_offset &&
                    buffer.byte_offset - parameter.byte_offset <
                        parameter.byte_length)
                  convolution_rows +=
                      "{parameter=" + parameter.key +
                      ",offset=" + std::to_string(parameter.byte_offset) +
                      ",length=" + std::to_string(parameter.byte_length) + "}";
          } else {
            convolution_rows += std::to_string(argument.value.scalar_bits);
          }
        }
        convolution_rows += "\n";
      }
      for (const auto &implementation : plan.implementations) {
        loomc_module_t *raw_module = nullptr;
        loomc_result_t *raw_result = nullptr;
        std::vector<loomc_string_view_t> names(implementation.roots.size());
        const auto status = loomc_module_deserialize_request(
            context_, workspace_, implementation.request.get(),
            loomc_allocator_system(), &raw_module, names.data(), &raw_result);
        std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module(
            raw_module, loomc_module_release);
        std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)> result(
            raw_result, loomc_result_release);
        ASSERT_TRUE(Ok(status));
        ASSERT_TRUE(result && loomc_result_succeeded(result.get()));
        ASSERT_TRUE(module);
        loomc_module_interop_view_t native{};
        raw_result = nullptr;
        const auto view_status = loomc_module_get_interop_view(
            module.get(), loomc_allocator_system(), &native, &raw_result);
        std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)>
            view_result(raw_result, loomc_result_release);
        ASSERT_TRUE(Ok(view_status));
        ASSERT_TRUE(view_result && loomc_result_succeeded(view_result.get()));
        ASSERT_NE(native.module, nullptr);
        for (const auto &name : names) {
          loomc_module_function_t function{};
          ASSERT_TRUE(
              loomc_module_try_lookup_function(module.get(), name, &function));
          ASSERT_TRUE(loomc_module_function_is_kernel_entry(function));
          const auto name_id = loom_module_lookup_string(
              native.module, iree_make_string_view(name.data, name.size));
          const auto symbol_id =
              loom_module_find_symbol(native.module, name_id);
          ASSERT_LT(symbol_id, native.module->symbols.count);
          const auto *operation =
              native.module->symbols.entries[symbol_id].defining_op;
          ASSERT_NE(operation, nullptr);
          const auto workloads =
              loom_kernel_workload_arg_ids(native.module, operation);
          rows += std::string(profile.label) + "/" + workload.phase + "/T" +
                  std::to_string(capacity) + "/" +
                  implementation.implementation_id + "/" +
                  std::string(name.data, name.size) + "=" +
                  std::to_string(workloads.count) + "\n";
        }
      }
    }
  }
  // These are observations, not an assertion that specialization erases every
  // source workload. Consumers still prove their exact selected replay.
  RecordProperty("published_root_workload_counts", rows);
  RecordProperty("selected_convolution_native_abi", convolution_rows);
  EXPECT_FALSE(rows.empty());
  EXPECT_FALSE(convolution_rows.empty());
}

} // namespace
} // namespace hrx::loomify_gguf::benchmark
