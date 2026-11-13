// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_COMPILER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_COMPILER_H_

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace hrx::loomify_gguf::jinja {

struct Diagnostic {
  std::string message;
  std::string source_name;
  size_t offset = 0;
  size_t line = 1;
  size_t column = 1;
};

// An owned, source-derived renderer. Helpers identify finite BUILD-owned
// inputs, never paths or implementations supplied by template text.
struct GeneratedInputRenderer {
  std::string source;
  std::string entrypoint;
  std::vector<std::string> helper_requirements;
};

// Compiles the supported template language to direct Loom control flow over an
// explicit JSON environment. Leaves *out unchanged on failure. This operation
// performs no file I/O, template-revision recognition or serving policy
// selection.
bool CompileChatTemplate(std::string_view source, std::string_view source_name,
                         GeneratedInputRenderer *out, Diagnostic *diagnostic);

} // namespace hrx::loomify_gguf::jinja

#endif
