// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "loomcxx-pipeline/support/mapping.h"
#include "gtest/gtest.h"

namespace loomcxx_pipeline {
namespace {
TEST(MappingTest, AdmitsBothDemonstratedAdaptations) {
  std::istringstream input("# explicit public contracts\n"
                           "a pa ca zero-origins index,f32,buffer,buffer\n"
                           "b pb cb raw index,offset,i1,buffer\n");
  std::vector<Mapping> mappings;
  std::string error;
  ASSERT_TRUE(ReadMappings(input, &mappings, &error)) << error;
  ASSERT_EQ(mappings.size(), 2);
  EXPECT_TRUE(mappings[0].pointer_origins);
  EXPECT_FALSE(mappings[1].pointer_origins);
  EXPECT_EQ(mappings[0].types[2], "buffer");
  EXPECT_EQ(mappings[1].types[1], "offset");
}
TEST(MappingTest, RejectsUnsupportedOrAmbiguousMappingWithoutPartialOutput) {
  for (const char *text :
       {"", "a pa ca guessed buffer", "a pa ca raw f64",
        "a pa ca raw index,,buffer", "a pa ca raw buffer,",
        "a pa ca zero-origins index", "a pa ca raw buffer extra",
        "a a ca raw buffer", "a pa ca raw buffer\nb pb ca raw buffer",
        "a pa ca raw buffer\nb pb cb bad buffer"}) {
    std::istringstream input(text);
    std::vector<Mapping> mappings;
    std::string error;
    EXPECT_FALSE(ReadMappings(input, &mappings, &error)) << text;
    EXPECT_TRUE(mappings.empty());
    EXPECT_FALSE(error.empty());
  }
}
} // namespace
} // namespace loomcxx_pipeline
