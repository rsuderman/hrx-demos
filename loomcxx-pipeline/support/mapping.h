// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef LOOMCXX_PIPELINE_SUPPORT_MAPPING_H_
#define LOOMCXX_PIPELINE_SUPPORT_MAPPING_H_

#include <istream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace loomcxx_pipeline {
// Deliberately bounded: void device providers, scalar/raw-buffer arguments,
// forwarded unchanged or with a zero byte origin after each buffer argument.
struct Mapping {
  std::string family, provider, imported_family;
  bool pointer_origins = false;
  std::vector<std::string> types;
};

inline bool ReadMappings(std::istream &input, std::vector<Mapping> *mappings,
                         std::string *error) {
  std::vector<Mapping> parsed;
  std::set<std::string> symbols;
  std::string line;
  size_t number = 0;
  while (std::getline(input, line)) {
    ++number;
    line.erase(line.find('#') == std::string::npos ? line.size()
                                                   : line.find('#'));
    std::istringstream fields(line);
    Mapping m;
    if (!(fields >> m.family))
      continue;
    std::string mode, types, extra;
    auto fail = [&](const char *reason) {
      *error = "mapping line " + std::to_string(number) + ": " + reason;
      return false;
    };
    if (!(fields >> m.provider >> m.imported_family >> mode >> types) ||
        (fields >> extra))
      return fail("expected family provider imported-family mode types");
    if (mode != "raw" && mode != "zero-origins")
      return fail("unsupported adaptation (use raw or zero-origins)");
    m.pointer_origins = mode == "zero-origins";
    for (const auto *symbol : {&m.family, &m.provider, &m.imported_family}) {
      if (symbol->empty() ||
          symbol->find_first_not_of(
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._") !=
              std::string::npos)
        return fail("unsupported symbol spelling");
      if (!symbols.insert(*symbol).second)
        return fail("duplicate mapping symbol");
    }
    if (types.empty() || types.back() == ',')
      return fail("empty argument type");
    std::istringstream arguments(types);
    std::string type;
    bool buffer = false;
    while (std::getline(arguments, type, ',')) {
      if (type != "index" && type != "offset" && type != "f32" &&
          type != "i1" && type != "buffer")
        return fail("unsupported argument type");
      buffer |= type == "buffer";
      m.types.push_back(type);
    }
    if (m.pointer_origins && !buffer)
      return fail("zero-origins requires a buffer argument");
    parsed.push_back(std::move(m));
  }
  if (input.bad() || parsed.empty()) {
    *error = "mapping must contain at least one readable declaration";
    return false;
  }
  *mappings = std::move(parsed);
  return true;
}
} // namespace loomcxx_pipeline
#endif
