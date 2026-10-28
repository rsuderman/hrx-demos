// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/source_builder.h"

namespace hrx::loomify_gguf {

std::string EmitEmptyProgram() {
  return "[[loom::symbol(\"model.pipeline\")]]\n"
         "void model_pipeline() {\n"
         "}\n";
}

}  // namespace hrx::loomify_gguf
