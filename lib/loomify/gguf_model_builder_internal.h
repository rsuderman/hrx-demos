// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_GGUF_MODEL_BUILDER_INTERNAL_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_GGUF_MODEL_BUILDER_INTERNAL_H_
#include "lib/loomify/gguf_model_builder.h"
#include "lib/loomify/architectures.h"
#include "lib/loomify/platforms.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/publication.h"
namespace hrx::loomify_gguf {
// Package implementation owns the complete text state; numerical callers only
// retain its lifetime without a text compiler or tokenizer dependency.
struct PackageConfiguration;
struct GgufModelBuilder::Impl {
  model::ModelDescriptor descriptor;
  model::TextMetadata text_metadata;
  std::optional<PublicationTokenFacts> publication_tokens;
  std::string publication_token_error;
  std::string source_asset;
  std::string source_alias;
};
struct BuiltGgufModule::Impl {
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module{
      nullptr, loomc_module_release};
  std::string configuration_source;
  std::string platform;
};
struct GgufConfiguration::Impl {
  ConfigurationFacts numerical;
  std::string source_asset;
  std::string source_alias;
  GgufBuildOptions options;
  const ArchitectureFiles *architecture = nullptr;
  const PlatformFiles *platform = nullptr;
  std::shared_ptr<PackageConfiguration> package;
};
} // namespace hrx::loomify_gguf
#endif
