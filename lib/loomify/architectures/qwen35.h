// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_QWEN35_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_QWEN35_H_
#include "lib/loomify/configuration.h"
#include "lib/loomify/publication.h"
#include <cstdio>
#include <optional>
#include <string_view>
namespace hrx::model { struct TextMetadata; }
namespace hrx::loomify_gguf::internal {
bool EmitQwen(const model::ModelDescriptor &, const std::string &,
    const ConfigurationSettings &, ConfigurationFacts *, std::string *);
bool DeriveQwenPublication(FILE *, std::string_view family, const model::TextMetadata &,
    std::optional<PublicationTokenFacts> *, std::string *);
} // namespace hrx::loomify_gguf::internal
#endif
