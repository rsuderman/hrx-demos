// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/export.h"
#include "iree/base/internal/json.h"
#include "gtest/gtest.h"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>

namespace hrx::loomify_gguf::benchmark {
namespace {
namespace fs = std::filesystem;
std::string Read(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}
bool JsonObject(const std::string &text) {
  auto cursor = iree_make_string_view(text.data(), text.size());
  iree_string_view_t object{};
  const auto consume = iree_json_consume_object(&cursor, &object);
  const bool ok = iree_status_is_ok(consume);
  iree_status_ignore(consume);
  const auto trailing = iree_json_consume_insignificant(&cursor);
  const bool clean = iree_status_is_ok(trailing);
  iree_status_ignore(trailing);
  return ok && clean && !cursor.size;
}
class BenchmarkExportTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *tmp = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(tmp && *tmp);
    directory_ =
        fs::path(tmp) /
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(fs::create_directory(directory_));
  }
  void TearDown() override {
    std::error_code ignored;
    fs::remove_all(directory_, ignored);
  }
  fs::path directory_;
};
TEST_F(BenchmarkExportTest, PlanInspectionPreservesBitsAndSelectedOrder) {
  Plan plan;
  plan.selection_id = "selection";
  plan.workload_id = "workload";
  plan.workload.seed = UINT64_MAX;
  plan.model.family = "example";
  plan.model.name = "model\n\"name";
  plan.source_asset = "/never/embed/a/workstation/model.gguf";
  Program program;
  program.root_symbol = "pipeline";
  loomc_cmd_command_t first{};
  first.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
  first.payload.dispatch.workgroup_count[0] = 2;
  first.payload.dispatch.workgroup_count[1] = 3;
  first.payload.dispatch.workgroup_count[2] = 1;
  program.commands.push_back(first);
  loomc_cmd_command_t second{};
  second.kind = LOOMC_CMD_COMMAND_KIND_BARRIER_EXECUTION;
  program.commands.push_back(second);
  plan.programs.push_back(program);
  Invocation invocation;
  invocation.invocation_id = "p0/c0";
  invocation.kernel_symbol = "selected";
  loomc_cmd_argument_t negative_zero{};
  negative_zero.kind = LOOMC_CMD_ARGUMENT_KIND_B32;
  negative_zero.value.scalar_bits = UINT64_C(0x80000000);
  loomc_cmd_argument_t nan{};
  nan.kind = LOOMC_CMD_ARGUMENT_KIND_B64;
  nan.value.scalar_bits = UINT64_C(0x7ff8000000000123);
  loomc_cmd_argument_t remaining{};
  remaining.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
  remaining.value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 5, 16,
                            UINT64_MAX};
  invocation.arguments = {negative_zero, nan, remaining};
  plan.invocations.push_back(invocation);
  const auto text = BenchmarkPlanJson(plan);
  EXPECT_TRUE(JsonObject(text));
  EXPECT_NE(text.find("0x0000000080000000"), std::string::npos);
  EXPECT_NE(text.find("0x7ff8000000000123"), std::string::npos);
  EXPECT_NE(text.find("\"byte_length\":\"remaining\""), std::string::npos);
  EXPECT_NE(text.find("\"grid\":[2,3,1]"), std::string::npos);
  EXPECT_NE(text.find("18446744073709551615"), std::string::npos);
  EXPECT_NE(text.find("model\\u000a\\\"name"), std::string::npos);
  EXPECT_EQ(text.find("/never/embed"), std::string::npos);
  EXPECT_EQ(text, BenchmarkPlanJson(plan));
}
TEST_F(BenchmarkExportTest, ReportPublicationNeverOverwritesExistingEntries) {
  BenchmarkReport report;
  report.manifest_id = "manifest";
  std::string error;
  const auto file = directory_ / "report.json";
  ASSERT_TRUE(WriteBenchmarkReportFile(report, file, &error)) << error;
  const auto original = Read(file);
  EXPECT_TRUE(JsonObject(original));
  report.manifest_id = "replacement";
  EXPECT_FALSE(WriteBenchmarkReportFile(report, file, &error));
  EXPECT_EQ(Read(file), original);
  const auto existing_directory = directory_ / "existing";
  ASSERT_TRUE(fs::create_directory(existing_directory));
  EXPECT_FALSE(WriteBenchmarkReportFile(report, existing_directory, &error));
  EXPECT_TRUE(fs::is_empty(existing_directory));
  const auto dangling = directory_ / "dangling";
  fs::create_symlink(directory_ / "absent", dangling);
  EXPECT_FALSE(WriteBenchmarkReportFile(report, dangling, &error));
  EXPECT_TRUE(fs::is_symlink(fs::symlink_status(dangling)));
  EXPECT_EQ(std::distance(fs::directory_iterator(directory_),
                          fs::directory_iterator()),
            3);
}
TEST_F(BenchmarkExportTest, ConcurrentReportPublicationHasOneWinner) {
  const auto file = directory_ / "report.json";
  std::atomic<unsigned> ready{0}, successes{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> writers;
  for (unsigned i = 0; i < 8; ++i)
    writers.emplace_back([&, i] {
      BenchmarkReport report;
      report.manifest_id = "writer-" + std::to_string(i);
      std::string error;
      ++ready;
      while (!start.load())
        std::this_thread::yield();
      if (WriteBenchmarkReportFile(report, file, &error))
        ++successes;
    });
  while (ready.load() != writers.size())
    std::this_thread::yield();
  start = true;
  for (auto &writer : writers)
    writer.join();
  EXPECT_EQ(successes.load(), 1u);
  EXPECT_TRUE(JsonObject(Read(file)));
  EXPECT_EQ(std::distance(fs::directory_iterator(directory_),
                          fs::directory_iterator()),
            1);
}
TEST_F(BenchmarkExportTest, MissingOutputParentIsNotCreated) {
  BenchmarkReport report;
  std::string error;
  EXPECT_FALSE(WriteBenchmarkReportFile(
      report, directory_ / "missing" / "report.json", &error));
  EXPECT_FALSE(fs::exists(directory_ / "missing"));
  EXPECT_TRUE(fs::is_empty(directory_));
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
