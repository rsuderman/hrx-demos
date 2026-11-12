// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Generates the actual package data helper for the maintained control fixture.
#include "lib/loomify/token_policy.h"
#include <fstream>
int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  std::ofstream output(argv[1], std::ios::binary);
  output << hrx::loomify_gguf::EmitTerminalMarkerSource("<|eot_id|>");
  // The complete maintained control also declares its package route identity.
  output
      << "global.rodata.def public @loomify.text.model_name = bytes(\"6c6f6f6d6966792d6c6c616d61\")\n";
  output.close();
  return output ? 0 : 1;
}
