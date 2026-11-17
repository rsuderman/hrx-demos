// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_EMISSION_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_TEXT_JINJA_EMISSION_H_

#include <string>
#include <string_view>

namespace hrx::loomify_gguf::jinja::internal {

inline std::string HexBytes(std::string_view bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

inline void AppendBufferLiteral(std::string_view name, std::string_view bytes,
                                std::string *out) {
  *out += "global.rodata.def @" + std::string(name) + " = bytes(\"" +
          HexBytes(bytes) + "\")\n";
}

// JSON escaping and Loom byte encoding are separate operations: UTF-8 bytes
// remain unchanged in JSON, then the complete JSON document becomes rodata.
inline std::string QuoteJsonString(std::string_view text) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char byte : text) {
    if (byte == '"' || byte == '\\') {
      result += '\\';
      result += static_cast<char>(byte);
    } else if (byte < 32) {
      result += "\\u00";
      result += digits[byte >> 4];
      result += digits[byte & 15];
    } else {
      result += static_cast<char>(byte);
    }
  }
  return result + '"';
}

} // namespace hrx::loomify_gguf::jinja::internal

#endif
