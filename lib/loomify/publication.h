// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_PUBLICATION_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_PUBLICATION_H_
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace hrx::loomify_gguf {
// Fixed numerical publication slots, independent of source marker spelling.
enum class PublicationLayout {
  kUnspecified = 0,
  kThreeStops,
  kTextAndTurn,
  kEosAndTurn,
  kHostManaged,
};
struct TokenIdentity {
  int32_t id = 0;
  std::string piece;
};
// Source-only raw stop identities consumed by the numerical publication.
struct PublicationTokenFacts {
  std::vector<TokenIdentity> raw_stops;
  size_t terminal_index = 0;
  std::string configuration_source;
};
} // namespace hrx::loomify_gguf
#endif
