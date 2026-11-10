// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_GGUF_MODEL_BUILDER_INTERNAL_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_GGUF_MODEL_BUILDER_INTERNAL_H_
#include "tools/loomify-gguf/gguf_model_builder.h"
#include "tools/loomify-gguf/architectures.h"
#include "tools/loomify-gguf/token_policy.h"
namespace hrx::loomify_gguf {
struct GgufModelBuilder::Impl {
  model::ModelDescriptor descriptor;
  model::TextMetadata text_metadata;
  std::optional<PublicationTokenFacts> publication_tokens;
  std::string publication_token_error;
  std::string source_asset;
};
struct BuiltGgufModule::Impl {
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module{nullptr, loomc_module_release};
  std::string configuration_source;
  std::string source_asset;
  GgufBuildOptions options;
  const ArchitectureFiles *architecture = nullptr;
};
} // namespace hrx::loomify_gguf
#endif
