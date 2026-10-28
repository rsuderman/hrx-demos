// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <string>

#include "gtest/gtest.h"
#include "tools/loomify-gguf/source_builder.h"

namespace {

TEST(LoomifyFrameworkTest, EmitsModelEntry) {
  const std::string source = hrx::loomify_gguf::EmitEmptyProgram();
  ASSERT_FALSE(source.empty());
  EXPECT_NE(source.find("model_pipeline"), std::string::npos);
}

}  // namespace
