// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_PROVIDERS_H_
#define HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_PROVIDERS_H_
#include "tools/loomify-gguf/authoring/model-config/linker.h"
namespace hrx::loomify_gguf {
// These requests contain one config-free numerical profile, with no model
// sequence or configuration defaults. The caller adds maintained program units
// and exact callable requirements. Source and relocatable forms are equivalent.
ConfiguredLinkRequest MakeLlamaLinkedProviders(const std::string &source_root,
                                               bool relocatable = false);
ConfiguredLinkRequest MakeQwenFfnLinkedProviders(const std::string &source_root,
                                                 bool relocatable = false);
} // namespace hrx::loomify_gguf
#endif
