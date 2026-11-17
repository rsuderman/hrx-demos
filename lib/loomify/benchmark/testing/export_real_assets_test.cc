// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/export.h"
#include "lib/loomify/benchmark/bundle.h"
#include "lib/loomify/benchmark/plan.h"
#include "lib/loomify/benchmark/report_io.h"
#include "loomc/target/amdgpu.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::benchmark {
namespace {
namespace fs = std::filesystem;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
std::string Read(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}
class BenchmarkExportRealTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *asset = std::getenv("LOOMIFY_TEST_QWEN_GGUF");
    const char *srcdir = std::getenv("TEST_SRCDIR");
    const char *workspace = std::getenv("TEST_WORKSPACE");
    const char *tmp = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(asset && *asset && srcdir && workspace && tmp)
        << "explicit immutable Qwen GGUF and Bazel runfiles are required";
    gguf_ = asset;
    library_root_ = (fs::path(srcdir) / workspace).string();
    directory_ =
        fs::path(tmp) /
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(fs::create_directory(directory_));
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
    workload_.phase = "decode";
    workload_.token_capacity = workload_.active_tokens = 1;
    workload_.history_tokens = 5;
  }
  void TearDown() override {
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    loomc_target_environment_release(environment_);
    std::error_code ignored;
    fs::remove_all(directory_, ignored);
  }
  bool Export(const ExportOptions &options, ExportResult *out,
              std::string *error) {
    return ExportGgufBenchmarks(gguf_, workload_, library_root_, options,
                                context_, workspace_, out, error);
  }
  std::string gguf_, library_root_;
  fs::path directory_;
  Workload workload_;
  loomc_target_environment_t *environment_ = nullptr;
  loomc_context_t *context_ = nullptr;
  loomc_workspace_t *workspace_ = nullptr;
};
TEST_F(BenchmarkExportRealTest,
       PlanOnlyPublishesNoFixturesAndPreservesExistingOutput) {
  ExportOptions options;
  options.max_case_input_bytes = UINT64_C(64) * 1024 * 1024;
  options.plan_only = true;
  options.output = directory_ / "plan.json";
  ExportResult result;
  std::string error;
  ASSERT_TRUE(Export(options, &result, &error)) << error;
  EXPECT_GT(result.selected_invocations, 0u);
  EXPECT_EQ(result.case_count, 0u);
  EXPECT_TRUE(result.manifest_id.empty());
  const auto original = Read(options.output);
  EXPECT_NE(original.find("loomify.benchmark.plan.v1"), std::string::npos);
  EXPECT_EQ(original.find(gguf_), std::string::npos);
  EXPECT_EQ(std::distance(fs::directory_iterator(directory_),
                          fs::directory_iterator()),
            1);
  EXPECT_FALSE(Export(options, &result, &error));
  EXPECT_EQ(Read(options.output), original);
}
TEST_F(BenchmarkExportRealTest, LinksDistinctSelectedArgmaxRequestsTogether) {
  Plan plan;
  std::string error;
  ASSERT_TRUE(BuildBenchmarkPlan(gguf_, workload_, library_root_, context_,
                                 workspace_, &plan, &error))
      << error;
  const std::vector<std::string> kernels = {
      "qwen38_greedy_argmax_partial_rows_f32",
      "qwen38_greedy_argmax_reduce_rows_f32"};
  std::vector<PreparedInvocationCase> cases;
  std::vector<size_t> selected;
  for (const auto &kernel : kernels) {
    size_t index = 0;
    while (index < plan.invocations.size() &&
           plan.invocations[index].kernel_symbol != kernel)
      ++index;
    ASSERT_LT(index, plan.invocations.size()) << kernel;
    const auto &invocation = plan.invocations[index];
    ASSERT_LT(invocation.implementation_index, plan.implementations.size());
    const auto &implementation =
        plan.implementations[invocation.implementation_index];
    ASSERT_NE(implementation.request.get(), nullptr);
    ASSERT_LT(invocation.request_root_ordinal, implementation.roots.size());
    PreparedCase prepared;
    ASSERT_EQ(PrepareStatelessCase(plan, invocation, UINT64_C(64) * 1024 * 1024,
                                   &prepared, &error),
              PreparationStatus::kReady)
        << error;
    selected.push_back(index);
    cases.push_back({index, std::move(prepared)});
  }
  ASSERT_NE(plan.invocations[selected[0]].implementation_index,
            plan.invocations[selected[1]].implementation_index);
  EXPECT_NE(
      plan.implementations[plan.invocations[selected[0]].implementation_index]
          .request.get(),
      plan.implementations[plan.invocations[selected[1]].implementation_index]
          .request.get());
  std::vector<EmittedCase> emitted;
  ASSERT_TRUE(EmitPreparedBenchmarkBundle(plan, std::move(cases), directory_,
                                          context_, workspace_, &emitted,
                                          &error))
      << error;
  ASSERT_EQ(emitted.size(), 2u);
  EXPECT_NE(emitted[0].case_id, emitted[1].case_id);
  EXPECT_NE(emitted[0].benchmark_symbol, emitted[1].benchmark_symbol);
  for (size_t i = 0; i < emitted.size(); ++i) {
    EXPECT_EQ(emitted[i].invocation_id,
              plan.invocations[selected[i]].invocation_id);
    EXPECT_EQ(emitted[i].arithmetic_check, "unchecked");
    bool saw_selected_kernel = false;
    for (const auto &dispatch : emitted[i].timed_dispatches)
      if (!dispatch.reset && dispatch.kernel_symbol == kernels[i])
        saw_selected_kernel = true;
    EXPECT_TRUE(saw_selected_kernel) << kernels[i];
  }
  EXPECT_TRUE(fs::is_regular_file(directory_ / "benchmark.loombc"));
  EXPECT_TRUE(fs::is_regular_file(directory_ / "benchmark.loom"));
  size_t source_modules = 0, bytecode_modules = 0;
  for (const auto &entry : fs::directory_iterator(directory_)) {
    source_modules += entry.path().extension() == ".loom";
    bytecode_modules += entry.path().extension() == ".loombc";
  }
  EXPECT_EQ(source_modules, 1u);
  EXPECT_EQ(bytecode_modules, 1u);
}
TEST_F(BenchmarkExportRealTest, StrictCoverageMatchesActualRunnableInventory) {
  ExportOptions options;
  options.max_case_input_bytes = UINT64_C(64) * 1024 * 1024;
  options.allow_partial = true;
  options.output = directory_ / "partial";
  ExportResult partial;
  std::string error;
  ASSERT_TRUE(Export(options, &partial, &error)) << error;
  options.allow_partial = false;
  options.output = directory_ / "strict";
  ExportResult result;
  result.selection_id = "preserved";
  if (partial.unsupported_invocations) {
    EXPECT_FALSE(Export(options, &result, &error));
    EXPECT_NE(error.find("unsupported selected invocations"),
              std::string::npos);
    EXPECT_EQ(result.selection_id, "preserved");
    EXPECT_FALSE(fs::exists(options.output));
  } else {
    ASSERT_TRUE(Export(options, &result, &error)) << error;
    EXPECT_TRUE(result.complete);
    EXPECT_EQ(result.selected_invocations, partial.selected_invocations);
  }
}
TEST_F(BenchmarkExportRealTest, ExplicitUnsupportedSequenceNeverPublishes) {
  ExportOptions options;
  options.max_case_input_bytes = UINT64_C(64) * 1024 * 1024;
  options.allow_partial = true;
  options.output = directory_ / "bundle";
  ExportResult result;
  result.selection_id = "preserved";
  std::string error;
  options.sequence_pairs = {{{"missing-origin", "p0/c183"}}};
  EXPECT_FALSE(Export(options, &result, &error));
  EXPECT_NE(error.find("unknown selected origin"), std::string::npos);
  EXPECT_EQ(result.selection_id, "preserved");
  EXPECT_TRUE(fs::is_empty(directory_));
  // Reversing the known selected pair cannot describe the native schedule.
  options.sequence_pairs = {{{"p0/c183", "p0/c182"}}};
  EXPECT_FALSE(Export(options, &result, &error));
  EXPECT_NE(error.find("cannot prepare requested sequence"), std::string::npos);
  EXPECT_EQ(result.selection_id, "preserved");
  EXPECT_TRUE(fs::is_empty(directory_));
  options.sequence_pairs = {{{"p0/c182", "p0/c183"}}, {{"p0/c182", "p0/c183"}}};
  EXPECT_FALSE(Export(options, &result, &error));
  EXPECT_NE(error.find("duplicate pairs"), std::string::npos);
  EXPECT_TRUE(fs::is_empty(directory_));
  options.sequence_pairs.resize(1);
  options.plan_only = true;
  EXPECT_FALSE(Export(options, &result, &error));
  EXPECT_NE(error.find("plan-only"), std::string::npos);
  EXPECT_TRUE(fs::is_empty(directory_));
}
TEST_F(BenchmarkExportRealTest, PartialBundleProjectsCompleteOriginInventory) {
  ExportOptions options;
  options.max_case_input_bytes = UINT64_C(64) * 1024 * 1024;
  options.allow_partial = true;
  options.generator_identity = "test-generator";
  options.output = directory_ / "partial";
  ExportResult result;
  std::string error;
  // Short canonical gate/up pairs are fused. Exercise the explicit sequence
  // exporter on the maintained longer-row path, where both leaves still exist.
  workload_.phase = "prefill";
  workload_.token_capacity = workload_.active_tokens = 32;
  workload_.history_tokens = 0;
  Plan plan;
  ASSERT_TRUE(BuildBenchmarkPlan(gguf_, workload_, library_root_, context_,
                                 workspace_, &plan, &error))
      << error;
  for (size_t i = 0; i + 1 < plan.invocations.size(); ++i) {
    const auto &gate = plan.invocations[i];
    const auto &up = plan.invocations[i + 1];
    if (gate.kernel_symbol != "ggml_linear_q4k_q8_1_x4" ||
        up.kernel_symbol != gate.kernel_symbol ||
        gate.program_ordinal != up.program_ordinal ||
        up.command_ordinal != gate.command_ordinal + 1)
      continue;
    const auto *parameter = FindBenchmarkParameter(
        plan, gate, gate.arguments[4].value.buffer, 2064384, &error);
    if (!parameter ||
        parameter->key.find(".ffn_gate.weight") == std::string::npos)
      continue;
    options.sequence_pairs = {{{gate.invocation_id, up.invocation_id}}};
    break;
  }
  ASSERT_EQ(1u, options.sequence_pairs.size());
  for (const auto &id : options.sequence_pairs[0]) {
    const auto invocation = std::find_if(
        plan.invocations.begin(), plan.invocations.end(),
        [&](const Invocation &value) { return value.invocation_id == id; });
    ASSERT_NE(invocation, plan.invocations.end()) << id;
    ASSERT_EQ(invocation->kernel_symbol, "ggml_linear_q4k_q8_1_x4");
    ASSERT_TRUE(invocation->command.begins_barrier_wave);
    ASSERT_GE(invocation->arguments.size(), 3u);
    EXPECT_EQ(invocation->arguments[0].value.scalar_bits, 512u);
    EXPECT_EQ(invocation->arguments[1].value.scalar_bits, 1024u);
    EXPECT_EQ(invocation->arguments[2].value.scalar_bits, 3584u);
    const auto &grid = invocation->command.payload.dispatch.workgroup_count;
    EXPECT_EQ(grid[0], 448u);
    EXPECT_EQ(grid[1], 32u);
    EXPECT_EQ(grid[2], 1u);
  }
  ASSERT_TRUE(Export(options, &result, &error)) << error;
  EXPECT_EQ(result.complete, result.unsupported_invocations == 0);
  EXPECT_GT(result.exported_invocations, 0u);
  EXPECT_EQ(result.exported_invocations + result.unsupported_invocations,
            result.selected_invocations);
  EXPECT_GT(result.case_count, 0u);
  const auto manifest = Read(options.output / "manifest.json");
  EXPECT_NE(manifest.find("\"max_case_input_bytes\":67108864"),
            std::string::npos);
  EXPECT_NE(manifest.find("\"weight_sources\":["), std::string::npos);
  EXPECT_NE(manifest.find("\"safety_checks\":["), std::string::npos);
  EXPECT_NE(manifest.find("\"fixture_payload_byte_length\":"),
            std::string::npos);
  EXPECT_EQ(manifest.find(gguf_), std::string::npos);
  EXPECT_EQ(manifest.find(library_root_), std::string::npos);
  ReportInventory inventory;
  ASSERT_TRUE(ParseBenchmarkReportManifest(manifest, &inventory, &error))
      << error;
  EXPECT_EQ(inventory.manifest_id, result.manifest_id);
  EXPECT_EQ(inventory.invocations.size(), result.selected_invocations);
  EXPECT_EQ(inventory.cases.size(), result.case_count);
  ASSERT_EQ(inventory.sequence_occurrences.size(), 1u);
  const auto &occurrence = inventory.sequence_occurrences[0];
  EXPECT_EQ(occurrence.program_ordinal, 0u);
  const auto selected_origin =
      std::find_if(plan.invocations.begin(), plan.invocations.end(),
                   [&](const Invocation &value) {
                     return value.invocation_id == options.sequence_pairs[0][0];
                   });
  ASSERT_NE(selected_origin, plan.invocations.end());
  EXPECT_EQ(occurrence.command_begin, selected_origin->command_ordinal);
  EXPECT_EQ(occurrence.command_end, selected_origin->command_ordinal + 2);
  EXPECT_EQ(occurrence.covered_invocation_ids,
            (std::vector<std::string>{options.sequence_pairs[0][0],
                                      options.sequence_pairs[0][1]}));
  const auto sequence =
      std::find_if(inventory.cases.begin(), inventory.cases.end(),
                   [&](const ReportCase &value) {
                     return value.case_id == occurrence.case_id;
                   });
  ASSERT_NE(sequence, inventory.cases.end());
  EXPECT_EQ(sequence->measurement_subject, "sequence");
  EXPECT_EQ(sequence->reset_policy, "none_overwrite_only");
  EXPECT_EQ(sequence->arithmetic_check, "unchecked");
  for (const auto &id : occurrence.covered_invocation_ids) {
    const auto origin =
        std::find_if(inventory.invocations.begin(), inventory.invocations.end(),
                     [&](const ReportInvocation &value) {
                       return value.invocation_id == id;
                     });
    ASSERT_NE(origin, inventory.invocations.end());
    EXPECT_FALSE(origin->case_id.empty());
    EXPECT_NE(origin->case_id, sequence->case_id);
  }
  EXPECT_TRUE(fs::is_regular_file(options.output / "benchmark.loombc"));
  EXPECT_TRUE(fs::is_regular_file(options.output / "benchmark.loom"));
  for (const auto &origin : inventory.invocations)
    EXPECT_NE(origin.case_id.empty(), origin.unsupported_reason.empty());
  const auto moved = directory_ / "moved";
  fs::rename(options.output, moved);
  EXPECT_EQ(Read(moved / "manifest.json"), manifest);
  EXPECT_EQ(std::distance(fs::directory_iterator(directory_),
                          fs::directory_iterator()),
            1);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
