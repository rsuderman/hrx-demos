// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "gtest/gtest.h"
#include <cstdlib>
TEST(QwenModelCli, KnownModelFormatsAndPublication) {
  EXPECT_EQ(
      std::system(
          "bash tools/loomify-gguf/serving/qwen/testing/cli_model_test.sh"),
      0);
}
