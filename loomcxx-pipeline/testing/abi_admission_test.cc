// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Exercise shared ABI admission against a real imported module, independently
// of the numerical algorithm. The normalization mapping is just the fixture.
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include "gtest/gtest.h"

namespace {
std::string Runfile(const char *relative) {
  return std::string(std::getenv("TEST_SRCDIR")) + "/" +
         std::getenv("TEST_WORKSPACE") + "/" + relative;
}
class AbiAdmissionTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::ifstream input(Runfile("loomcxx-pipeline/llm/normalization.abi"));
    ASSERT_TRUE(input.good());
    mapping_ = std::string(std::istreambuf_iterator<char>(input), {});
    directory_ = std::getenv("TEST_TMPDIR");
  }
  int Assemble(const std::string &mapping) {
    auto mappings = directory_ + "/mapping.abi";
    auto output = directory_ + "/output.loombc";
    std::ofstream(mappings) << mapping;
    auto tool = Runfile("loomcxx-pipeline/support/assemble");
    auto input = Runfile("loomcxx-pipeline/llm/normalization_import.loombc");
    pid_t child = fork();
    if (child == 0) {
      execl(tool.c_str(), tool.c_str(), input.c_str(), output.c_str(),
            mappings.c_str(), static_cast<char *>(nullptr));
      _exit(127);
    }
    if (child < 0) return -1;
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
  }
  std::string Replace(const std::string &from, const std::string &to) {
    std::string text = mapping_;
    auto position = text.find(from);
    EXPECT_NE(position, std::string::npos);
    if (position != std::string::npos) text.replace(position, from.size(), to);
    return text;
  }
  std::string mapping_, directory_;
};
TEST_F(AbiAdmissionTest, AcceptsBothExplicitContracts) {
  EXPECT_EQ(Assemble(mapping_), 0);
}
TEST_F(AbiAdmissionTest, RejectsMissingImportedFamily) {
  EXPECT_EQ(Assemble(Replace("cxx.llm.rmsnorm.row", "missing.family")), 1);
}
TEST_F(AbiAdmissionTest, RejectsIncorrectPointerAdaptation) {
  EXPECT_EQ(Assemble(Replace("zero-origins", "raw")), 1);
}
TEST_F(AbiAdmissionTest, RejectsSameArityScalarMismatch) {
  EXPECT_EQ(Assemble(Replace("index,f32,index", "f32,f32,index")), 1);
}
} // namespace
