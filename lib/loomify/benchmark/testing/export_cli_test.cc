// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "gtest/gtest.h"
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

extern char **environ;
namespace hrx::loomify_gguf::benchmark {
namespace {
namespace fs = std::filesystem;
std::string Read(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}
class BenchmarkCliTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *srcdir = std::getenv("TEST_SRCDIR");
    const char *workspace = std::getenv("TEST_WORKSPACE");
    const char *tmp = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(srcdir && workspace && tmp);
    tool_ = fs::path(srcdir) / workspace /
            "tools/loomify-benchmark/loomify-benchmark";
    ASSERT_TRUE(fs::is_regular_file(tool_));
    directory_ =
        fs::path(tmp) /
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(fs::create_directory(directory_));
    stderr_ = directory_ / "stderr.txt";
  }
  void TearDown() override {
    std::error_code ignored;
    fs::remove_all(directory_, ignored);
  }
  int Run(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), tool_.string());
    std::vector<char *> argv;
    for (auto &argument : arguments)
      argv.push_back(argument.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0)
      return -1;
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, stderr_.c_str(),
                                     O_WRONLY | O_CREAT | O_TRUNC, 0600);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                     O_WRONLY, 0);
    pid_t pid = 0;
    const int result = posix_spawn(&pid, tool_.c_str(), &actions, nullptr,
                                   argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (result != 0)
      return -1;
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
      if (errno != EINTR)
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  }
  fs::path tool_, directory_, stderr_;
};
TEST_F(BenchmarkCliTest, InvalidWorkloadsNeverPublishOutput) {
  EXPECT_EQ(Run({"--help"}), 0);
  const auto output = directory_ / "output";
  const auto common = "--output=" + output.string();
  for (const char *bad :
       {"--phase=other", "--tokens=-1", "--active-tokens=18446744073709551616",
        "--history-tokens=-1", "--seed=bad"}) {
    EXPECT_NE(Run({"--gguf=absent.gguf", common, bad}), 0) << Read(stderr_);
    EXPECT_FALSE(fs::exists(output));
  }
  EXPECT_NE(Run({"--gguf=absent.gguf", common, "--phase=decode", "--tokens=8"}),
            0);
  EXPECT_NE(Read(stderr_).find("decode requires"), std::string::npos);
  EXPECT_FALSE(fs::exists(output));
}
TEST_F(BenchmarkCliTest, SequenceFlagsRejectMalformedOrPlanOnlyRequests) {
  const auto output = directory_ / "bundle";
  const auto destination = "--output=" + output.string();
  for (const char *bad :
       {"", "p0/c1", ",p0/c2", "p0/c1,", "p0/c1,p0/c2,p0/c3"}) {
    EXPECT_NE(Run({"--gguf=absent.gguf", destination,
                   std::string("--sequence=") + bad}),
              0);
    EXPECT_NE(Read(stderr_).find("--sequence requires exactly"),
              std::string::npos);
    EXPECT_FALSE(fs::exists(output));
  }
  EXPECT_NE(Run({"--gguf=absent.gguf", destination, "--plan-only",
                 "--sequence=p0/c1,p0/c2"}),
            0);
  EXPECT_NE(Read(stderr_).find("--sequence cannot be used with --plan-only"),
            std::string::npos);
  EXPECT_FALSE(fs::exists(output));
  // Repeatable valid syntax reaches source admission without a parser fallback.
  EXPECT_NE(Run({"--gguf=absent.gguf", destination, "--sequence=p0/c1,p0/c2",
                 "--sequence=p0/c3,p0/c4"}),
            0);
  EXPECT_NE(Read(stderr_).find("GGUF"), std::string::npos);
  EXPECT_EQ(Read(stderr_).find("--sequence requires"), std::string::npos);
  EXPECT_FALSE(fs::exists(output));
}
TEST_F(BenchmarkCliTest, FullUnsignedSeedReachesSourceAdmission) {
  EXPECT_NE(Run({"--gguf=absent.gguf", "--plan-only", "--phase=decode",
                 "--seed=18446744073709551615",
                 "--output=" + (directory_ / "plan.json").string()}),
            0);
  EXPECT_NE(Read(stderr_).find("GGUF"), std::string::npos);
  EXPECT_EQ(Read(stderr_).find("unsigned integer"), std::string::npos);
}
TEST_F(BenchmarkCliTest, ReportImportsMissingMeasurementsAndDoesNotClobber) {
  const std::string manifest_id(64, 'a'), case_id(64, 'c');
  const auto manifest = directory_ / "manifest.json";
  const auto index = directory_ / "index.json";
  const auto output = directory_ / "report.json";
  {
    std::ofstream file(manifest);
    file
        << "{\"format\":\"loomify.benchmark.bundle.v1\",\"manifest_id\":\""
        << manifest_id << "\",\"workload_id\":\"" << std::string(64, 'b')
        << "\",\"phase\":\"decode\",\"cases\":[{\"case_id\":\"" << case_id
        << "\",\"input_policy_digest\":\"" << std::string(64, 'd')
        << "\",\"implementation_id\":\"" << std::string(64, 'e')
        << "\",\"measurement_subject\":\"kernel\",\"arithmetic_check\":\"unchecked\","
           "\"benchmark_symbol\":\"bench_test\"}],"
           "\"invocations\":[{\"invocation_id\":\"p0/c0\",\"case_id\":\""
        << case_id << "\",\"unsupported_reason\":\"\"}]}";
  }
  {
    std::ofstream file(index);
    file
        << "{\"format\":\"loomify.benchmark.result-index.v1\",\"manifest_id\":\""
        << manifest_id << "\",\"results\":[]}";
  }
  const std::vector<std::string> args = {
      "report", "--manifest=" + manifest.string(), "--index=" + index.string(),
      "--output=" + output.string()};
  ASSERT_EQ(Run(args), 0) << Read(stderr_);
  const auto original = Read(output);
  EXPECT_NE(original.find("\"measured_invocations\":0"), std::string::npos);
  EXPECT_NE(original.find("\"selected_invocations\":1"), std::string::npos);
  EXPECT_NE(Run(args), 0);
  EXPECT_EQ(Read(output), original);
  EXPECT_NE(Run({"report", "--manifest=" + manifest.string(),
                 "--index=" + index.string(), "--output=" + output.string(),
                 "--gguf=not-for-report"}),
            0);
  EXPECT_EQ(Read(output), original);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
