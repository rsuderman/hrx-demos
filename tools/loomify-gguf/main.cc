// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <cstdio>

#include "iree/base/tooling/flags.h"
#include "tools/loomify-gguf/source_builder.h"

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "loomify-gguf",
      "Print an empty loomcxx model entry to stdout.\n"
      "\n"
      "Usage: loomify-gguf > model.cxx\n"
      "\n"
      "This developer foundation emits source only. GGUF input and model\n"
      "construction will be added in later stages.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc != 1) {
    std::fprintf(stderr,
                 "loomify-gguf: unexpected argument '%s'; see --help\n",
                 argv[1]);
    return 1;
  }

  const auto source = hrx::loomify_gguf::EmitEmptyProgram();
  if (std::fwrite(source.data(), 1, source.size(), stdout) != source.size() ||
      std::fflush(stdout) != 0) {
    std::perror("loomify-gguf: writing source");
    return 1;
  }
  return 0;
}
