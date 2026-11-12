// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration.h"
#include "lib/loomify/testing/alternating_qwen_fixture.h"
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  hrx::model::ModelDescriptor descriptor;
  hrx::loomify_gguf::ConfigurationFacts facts;
  std::string error;
  if (!hrx::loomify_gguf::testing::LoadAlternatingQwenFixture(
          &descriptor, nullptr, &error) ||
      !hrx::loomify_gguf::DeriveConfiguration(descriptor, "independent.gguf",
                                              {3, 65}, &facts, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::ofstream output(argv[1], std::ios::binary);
  output << facts.source;
  return output ? 0 : 1;
}
