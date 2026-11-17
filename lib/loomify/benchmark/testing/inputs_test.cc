// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/inputs.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
loomc_cmd_buffer_ref_t
Ref(uint32_t root, uint64_t offset, uint64_t size,
    loomc_cmd_buffer_role_t role = LOOMC_CMD_BUFFER_ROLE_REBINDABLE) {
  return {role, root, offset, size};
}

class BenchmarkInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *directory = std::getenv("TEST_TMPDIR");
    ASSERT_NE(directory, nullptr);
    path = std::filesystem::path(directory) /
           ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directories(path));
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  std::filesystem::path path;
};

TEST(BenchmarkInputLayout, PreservesRolesAliasesAndRelativeOffsets) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots(
      {{Ref(3, 1032, 16), 16, 16},
       {Ref(3, 1040, 16), 16, 8},
       {Ref(3, 1032, 16, LOOMC_CMD_BUFFER_ROLE_FIXED), 16, 8}},
      64, &roots, &error))
      << error;
  ASSERT_EQ(roots.size(), 2);
  const auto *shared = FindInputRoot(roots, Ref(3, 1032, 16));
  ASSERT_NE(shared, nullptr);
  EXPECT_EQ(shared->original_byte_offset, 1024);
  EXPECT_EQ(shared->bytes.size(), 32);
  EXPECT_EQ(shared->minimum_alignment, 16);
  EXPECT_NE(shared, FindInputRoot(
                        roots, Ref(3, 1032, 16, LOOMC_CMD_BUFFER_ROLE_FIXED)));
}

TEST(BenchmarkInputLayout,
     RejectsOverflowUnboundedAndOversizedBeforeAllocation) {
  std::vector<PreparedRoot> roots(1);
  roots[0].root_index = 99;
  std::string error;
  EXPECT_FALSE(AllocateInputRoots({{Ref(0, UINT64_MAX - 1, 4), 4, 1}}, 1024,
                                  &roots, &error));
  EXPECT_FALSE(
      AllocateInputRoots({{Ref(0, 0, 8), 9, 1}}, 1024, &roots, &error));
  EXPECT_FALSE(
      AllocateInputRoots({{Ref(0, 0, 8), 8, 3}}, 1024, &roots, &error));
  EXPECT_FALSE(AllocateInputRoots({{Ref(0, 0, 8), 8, 1}}, 7, &roots, &error));
  ASSERT_EQ(roots.size(), 1);
  EXPECT_EQ(roots[0].root_index, 99);
}

TEST(BenchmarkInputLayout, BudgetAdmissionIsTypedAndSeparateFromInvalidRanges) {
  std::vector<PreparedRoot> roots(1);
  roots[0].root_index = 99;
  std::string error;
  bool excluded = false;
  EXPECT_FALSE(
      AllocateInputRoots({{Ref(0, 0, 8), 8, 4}}, 7, &roots, &error, &excluded));
  EXPECT_TRUE(excluded);
  ASSERT_EQ(roots.size(), 1);
  EXPECT_EQ(roots[0].root_index, 99);
  EXPECT_FALSE(AllocateInputRoots({{Ref(0, UINT64_MAX - 1, 4), 4, 4}}, 0,
                                  &roots, &error, &excluded));
  EXPECT_FALSE(excluded);
  EXPECT_TRUE(
      AllocateInputRoots({{Ref(0, 0, 8), 8, 4}}, 8, &roots, &error, &excluded));
  EXPECT_FALSE(excluded);
}

TEST(BenchmarkInputLayout, AliasInitializationsMustAgree) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots({{Ref(0, 12, 8), 8, 4}}, 8, &roots, &error));
  ASSERT_TRUE(SetInputBytes(Ref(0, 12, 4), {1, 2, 3, 4}, &roots, &error));
  EXPECT_TRUE(SetInputBytes(Ref(0, 14, 4), {3, 4, 5, 6}, &roots, &error));
  const auto previous = roots[0].bytes;
  EXPECT_FALSE(SetInputBytes(Ref(0, 14, 4), {9, 4, 5, 6}, &roots, &error));
  EXPECT_EQ(previous, roots[0].bytes);
  EXPECT_FALSE(SetInputBytes(Ref(0, 19, 2), {0, 0}, &roots, &error));
}

TEST(BenchmarkInputLayout, SyntheticActivationsAreFiniteNontrivialAndStable) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(
      AllocateInputRoots({{Ref(0, 0, 244), 244, 4}}, 244, &roots, &error));
  ASSERT_TRUE(SetFiniteF32Input(Ref(0, 0, 244), 61, 7, &roots, &error));
  const auto expected = roots[0].bytes;
  EXPECT_TRUE(SetFiniteF32Input(Ref(0, 0, 244), 61, 7, &roots, &error));
  EXPECT_EQ(expected, roots[0].bytes);
  bool positive = false, negative = false;
  for (size_t i = 0; i < 61; ++i) {
    uint32_t bits = 0;
    for (unsigned lane = 0; lane < 4; ++lane)
      bits |= static_cast<uint32_t>(expected[i * 4 + lane]) << (lane * 8);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_LE(std::abs(value), 0.5f);
    positive |= value > 0;
    negative |= value < 0;
  }
  EXPECT_TRUE(positive && negative);
}

TEST_F(BenchmarkInputsTest, TensorFixtureReadsOnlyRequestedSourceSpan) {
  const auto source = path / "input.gguf";
  {
    std::ofstream file(source, std::ios::binary);
    file.write("unrelatedWEIGHTtail", 19);
  }
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots({{Ref(0, 16, 6), 6, 1}}, 6, &roots, &error));
  ASSERT_TRUE(SetTensorInput(Ref(0, 16, 6), source, 9, 6, &roots, &error))
      << error;
  EXPECT_EQ(std::string(roots[0].bytes.begin(), roots[0].bytes.end()),
            "WEIGHT");
  EXPECT_FALSE(SetTensorInput(Ref(0, 16, 6), source, 18, 6, &roots, &error));
  EXPECT_FALSE(
      SetTensorInput(Ref(0, 16, 6), path / "missing", 0, 6, &roots, &error));
}

TEST_F(BenchmarkInputsTest,
       RelativeNpyDeduplicatesContentWithoutAliasingRoots) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots({{Ref(0, 0, 3), 3, 1}, {Ref(1, 0, 3), 3, 1}},
                                 6, &roots, &error));
  for (uint32_t i = 0; i < 2; ++i)
    ASSERT_TRUE(SetInputBytes(Ref(i, 0, 3), {0x80, 1, 0xFF}, &roots, &error));
  ASSERT_TRUE(WriteInputFixtures(path, &roots, &error)) << error;
  EXPECT_EQ(roots[0].fixture_path, roots[1].fixture_path);
  EXPECT_EQ(roots[0].sha256.size(), 64);
  EXPECT_FALSE(std::filesystem::path(roots[0].fixture_path).is_absolute());
  auto npy = path / roots[0].fixture_path;
  std::ifstream file(npy, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  ASSERT_GE(data.size(), 13);
  EXPECT_EQ(data[0], 0x93);
  EXPECT_EQ(std::string(data.begin() + 1, data.begin() + 6), "NUMPY");
  const size_t header_size = data[8] | (static_cast<size_t>(data[9]) << 8);
  EXPECT_EQ((10 + header_size) % 64, 0);
  EXPECT_EQ(std::vector<uint8_t>(data.begin() + 10 + header_size, data.end()),
            roots[0].bytes);
  EXPECT_TRUE(WriteInputFixtures(path, &roots, &error)) << error;
  {
    std::ofstream corrupt(npy, std::ios::binary | std::ios::trunc);
    corrupt << "corrupt";
  }
  EXPECT_FALSE(WriteInputFixtures(path, &roots, &error));
}
TEST_F(BenchmarkInputsTest, GuardedNpyKeepsLogicalBytesAndChecksReusedTail) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots({{Ref(0, 8, 3), 3, 1}}, 3, &roots, &error));
  ASSERT_TRUE(SetInputBytes(Ref(0, 8, 3), {1, 2, 3}, &roots, &error));
  ASSERT_TRUE(
      WriteInputFixtures(path, &roots, &error, kBenchmarkTailGuardBytes))
      << error;
  EXPECT_EQ(roots[0].bytes, (std::vector<uint8_t>{1, 2, 3}));
  EXPECT_EQ(roots[0].original_byte_offset, 8);
  EXPECT_EQ(roots[0].fixture_suffix_bytes, kBenchmarkTailGuardBytes);
  const auto fixture = path / roots[0].fixture_path;
  std::ifstream file(fixture, std::ios::binary);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  ASSERT_GE(data.size(), 10 + 3 + kBenchmarkTailGuardBytes);
  const size_t header_size = data[8] | (static_cast<size_t>(data[9]) << 8);
  EXPECT_EQ(data.size(), 10 + header_size + 3 + kBenchmarkTailGuardBytes);
  EXPECT_EQ(
      std::vector<uint8_t>(data.end() - kBenchmarkTailGuardBytes, data.end()),
      std::vector<uint8_t>(kBenchmarkTailGuardBytes, kBenchmarkTailGuardValue));
  EXPECT_TRUE(
      WriteInputFixtures(path, &roots, &error, kBenchmarkTailGuardBytes))
      << error;
  {
    std::fstream corrupt(fixture,
                         std::ios::binary | std::ios::in | std::ios::out);
    corrupt.seekp(-1, std::ios::end);
    corrupt.put(0);
  }
  EXPECT_FALSE(
      WriteInputFixtures(path, &roots, &error, kBenchmarkTailGuardBytes));
  EXPECT_FALSE(WriteInputFixtures(path, &roots, &error, 1));
}
TEST(BenchmarkInputLayout, ResetBudgetCountsMergedSeedsBeforeAllocation) {
  std::vector<PreparedRoot> roots;
  std::string error;
  ASSERT_TRUE(AllocateInputRoots({{Ref(0, 0, 16), 16, 1}}, 16, &roots, &error));
  ASSERT_TRUE(SetInputBytes(Ref(0, 0, 16), std::vector<uint8_t>(16, 3), &roots,
                            &error));
  std::vector<StateReset> resets(1);
  resets[0].seed.bytes = {9};
  bool excluded = false;
  const std::vector<BufferExtent> overlapping = {{Ref(0, 2, 6), 6, 1},
                                                 {Ref(0, 6, 6), 6, 1}};
  EXPECT_FALSE(
      CaptureStateResets(overlapping, roots, &resets, &error, 25, &excluded));
  EXPECT_TRUE(excluded);
  ASSERT_EQ(resets.size(), 1);
  EXPECT_EQ(resets[0].seed.bytes, (std::vector<uint8_t>{9}));
  ASSERT_TRUE(
      CaptureStateResets(overlapping, roots, &resets, &error, 26, &excluded))
      << error;
  EXPECT_FALSE(excluded);
  ASSERT_EQ(resets.size(), 1);
  EXPECT_EQ(resets[0].seed.bytes.size(), 10);
  EXPECT_FALSE(CaptureStateResets({{Ref(0, 15, 2), 2, 1}}, roots, &resets,
                                  &error, 0, &excluded));
  EXPECT_FALSE(excluded);
  EXPECT_EQ(resets[0].seed.bytes.size(), 10);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
