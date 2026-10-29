// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "gtest/gtest.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
std::string Read(const std::filesystem::path &path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream), {}};
}

TEST(LlamaStepCli, RejectsInputAliasesBeforeReadingAndPreservesSentinels) {
#if defined(_WIN32)
  GTEST_SKIP() << "This wheel CLI integration gate uses POSIX process APIs";
#else
  const char *emitter = std::getenv("LOOMIFY_STEP_EMITTER");
  const char *temporary = std::getenv("TEST_TMPDIR");
  ASSERT_NE(emitter, nullptr);
  ASSERT_NE(temporary, nullptr);
  const auto directory = std::filesystem::path(temporary) / "output aliases";
  std::filesystem::create_directories(directory);
  const auto checkpoint = directory / "checkpoint.gguf";
  const auto provider = directory / "provider.loombc";
  const auto diagnostic = directory / "stderr";
  {
    std::ofstream(checkpoint) << "checkpoint sentinel";
  }
  {
    std::ofstream(provider) << "provider sentinel";
  }
  for (const auto &input : {checkpoint, provider}) {
    auto hard = input;
    hard += ".hard";
    auto symbolic = input;
    symbolic += ".symbolic";
    std::filesystem::create_hard_link(input, hard);
    std::filesystem::create_symlink(input, symbolic);
    for (const auto &output : {input, hard, symbolic}) {
      const auto child = fork();
      ASSERT_GE(child, 0);
      if (child == 0) {
        const int capture =
            open(diagnostic.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (capture < 0 || dup2(capture, STDERR_FILENO) < 0)
          _exit(125);
        close(capture);
        execl(emitter, emitter, checkpoint.c_str(), provider.c_str(),
              output.c_str(), nullptr);
        _exit(126);
      }
      int status = 0;
      ASSERT_EQ(waitpid(child, &status, 0), child);
      ASSERT_TRUE(WIFEXITED(status));
      EXPECT_EQ(WEXITSTATUS(status), 1);
      EXPECT_NE(Read(diagnostic).find("output aliases an input"),
                std::string::npos);
      EXPECT_EQ(Read(checkpoint), "checkpoint sentinel");
      EXPECT_EQ(Read(provider), "provider sentinel");
    }
  }
#endif
}
} // namespace
