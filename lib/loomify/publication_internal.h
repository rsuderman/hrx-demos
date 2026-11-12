// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_PUBLICATION_INTERNAL_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_PUBLICATION_INTERNAL_H_
#include "lib/loomify/publication.h"
#include "tools/model/text_metadata.h"
#include <cstdio>
#include <optional>
#include <string_view>
namespace hrx::loomify_gguf::internal {
const model::DeclaredToken *DeclaredRole(const model::TextMetadata &, std::string_view);
bool ReadTokenIdentity(const model::TextMetadata &, const model::DeclaredToken &,
                       TokenIdentity *, std::string *);
bool DeriveNamedPublication(FILE *, std::string_view config_prefix,
    const model::TextMetadata &, const std::vector<std::string> &pieces,
    const std::vector<std::string> &config_keys, size_t terminal_index,
    size_t declared_eos_index, std::optional<PublicationTokenFacts> *, std::string *);
bool DeriveChatmlPublication(FILE *, std::string_view family,
    const model::TextMetadata &, std::optional<PublicationTokenFacts> *, std::string *);
} // namespace hrx::loomify_gguf::internal
#endif
