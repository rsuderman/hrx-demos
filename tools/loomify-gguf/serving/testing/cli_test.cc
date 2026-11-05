// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "gtest/gtest.h"
#include <cstdlib>
TEST(LoomifyCli, RejectsIncompleteOrUnsupportedExportsWithoutPublishing) {
  EXPECT_EQ(std::system("bash tools/loomify-gguf/serving/testing/cli_test.sh"),
            0);
}
