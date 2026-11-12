// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/text/matching.h"
#include "lib/loomify/chat_templates.h"
#include <algorithm>

namespace hrx::loomify_gguf::text_policy {
std::string_view Fixture(const char *name) {
  const auto *files = chat_templates_create();
  for (size_t i = 0; i < chat_templates_size(); ++i)
    if (std::string_view(files[i].name) == name)
      return {files[i].data, files[i].size};
  return {};
}
bool Utf8(std::string_view value) {
  for (size_t i = 0; i < value.size();) {
    const auto c = static_cast<unsigned char>(value[i++]);
    if (c < 0x80)
      continue;
    unsigned n = 0, cp = 0, minimum = 0;
    if (c >= 0xc2 && c <= 0xdf) {
      n = 1;
      cp = c & 31;
      minimum = 0x80;
    } else if (c >= 0xe0 && c <= 0xef) {
      n = 2;
      cp = c & 15;
      minimum = 0x800;
    } else if (c >= 0xf0 && c <= 0xf4) {
      n = 3;
      cp = c & 7;
      minimum = 0x10000;
    } else
      return false;
    if (n > value.size() - i)
      return false;
    for (unsigned j = 0; j < n; ++j) {
      const auto next = static_cast<unsigned char>(value[i++]);
      if ((next & 0xc0) != 0x80)
        return false;
      cp = (cp << 6) | (next & 63);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
      return false;
  }
  return true;
}
bool Quoted(std::string_view source, size_t begin, std::string *value,
            size_t *end) {
  if (begin >= source.size() || (source[begin] != '\'' && source[begin] != '"'))
    return false;
  const char quote = source[begin];
  std::string decoded;
  for (size_t i = begin + 1; i < source.size(); ++i) {
    char c = source[i];
    if (c == quote) {
      if (!Utf8(decoded))
        return false;
      *value = std::move(decoded);
      *end = i + 1;
      return true;
    }
    if (static_cast<unsigned char>(c) < 32)
      return false;
    if (c == '\\') {
      if (++i == source.size())
        return false;
      switch (source[i]) {
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case '\\':
      case '\'':
      case '"':
        c = source[i];
        break;
      default:
        return false;
      }
    }
    if (decoded.size() == 65536)
      return false;
    decoded.push_back(c);
  }
  return false;
}
bool Tags(std::string_view value, std::vector<std::string> *tags) {
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '>')
      return false;
    if (value[i] != '<')
      continue;
    const auto end = value.find('>', i + 1);
    if (end == std::string_view::npos ||
        value.substr(i + 1, end - i - 1).find('<') != std::string_view::npos)
      return false;
    tags->emplace_back(value.substr(i, end - i + 1));
    i = end;
  }
  return true;
}
bool AddSlot(std::string_view source, const char *name, std::string_view anchor,
             size_t quote_offset, std::vector<Slot> *slots) {
  const auto found = source.find(anchor);
  if (found == std::string_view::npos ||
      source.find(anchor, found + 1) != std::string_view::npos)
    return false;
  Slot slot;
  slot.name = name;
  slot.begin = found + quote_offset;
  if (!Quoted(source, slot.begin, &slot.original, &slot.end))
    return false;
  slots->push_back(std::move(slot));
  return true;
}
bool Capture(std::string_view expected, std::string_view actual,
             std::vector<Slot> slots, Captures *values) {
  std::sort(slots.begin(), slots.end(),
            [](const Slot &a, const Slot &b) { return a.begin < b.begin; });
  size_t prior = 0, cursor = 0;
  for (const auto &slot : slots) {
    const auto fixed = expected.substr(prior, slot.begin - prior);
    if (actual.substr(cursor, fixed.size()) != fixed)
      return false;
    cursor += fixed.size();
    if (cursor >= actual.size() || actual[cursor] != expected[slot.begin])
      return false;
    std::string value;
    size_t end = 0;
    if (!Quoted(actual, cursor, &value, &end))
      return false;
    std::vector<std::string> original_tags, actual_tags;
    if (!Tags(slot.original, &original_tags) || !Tags(value, &actual_tags) ||
        original_tags != actual_tags)
      return false;
    if (!values->emplace(slot.name, std::move(value)).second)
      return false;
    cursor = end;
    prior = slot.end;
  }
  return actual.substr(cursor) == expected.substr(prior);
}
std::string Hex(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : value) {
    result += digits[c >> 4];
    result += digits[c & 15];
  }
  return result;
}
void Datum(ChatPolicySelection *policy, const char *name,
           std::string_view value) {
  policy->configuration_source += "global.rodata.def public @" +
                                  std::string(name) + " = bytes(\"" +
                                  Hex(value) + "\")\n";
  policy->data_roots.push_back(name);
}
} // namespace hrx::loomify_gguf::text_policy
