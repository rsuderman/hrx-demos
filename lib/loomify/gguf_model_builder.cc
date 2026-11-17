// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/gguf_model_builder_internal.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/linking.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <utility>
namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
} // namespace
GgufModelBuilder::GgufModelBuilder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
GgufModelBuilder::~GgufModelBuilder() = default;
GgufConfiguration::GgufConfiguration(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
GgufConfiguration::~GgufConfiguration() = default;
GgufConfiguration::GgufConfiguration(GgufConfiguration &&) noexcept = default;
GgufConfiguration &
GgufConfiguration::operator=(GgufConfiguration &&) noexcept = default;
const std::string &GgufConfiguration::configuration_source() const {
  return impl_->numerical.source;
}
const std::string &GgufConfiguration::platform() const {
  return impl_->options.platform;
}
bool GgufConfiguration::has_package_configuration() const {
  return bool(impl_->package);
}
BuiltGgufModule::BuiltGgufModule(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
BuiltGgufModule::~BuiltGgufModule() = default;
BuiltGgufModule::BuiltGgufModule(BuiltGgufModule &&) noexcept = default;
BuiltGgufModule &
BuiltGgufModule::operator=(BuiltGgufModule &&) noexcept = default;
const model::ModelDescriptor &GgufModelBuilder::descriptor() const {
  return impl_->descriptor;
}
const model::TextMetadata &GgufModelBuilder::text_metadata() const {
  return impl_->text_metadata;
}
const std::optional<PublicationTokenFacts> &
GgufModelBuilder::publication_tokens() const {
  return impl_->publication_tokens;
}
const std::string &BuiltGgufModule::configuration_source() const {
  return impl_->configuration_source;
}
const std::string &BuiltGgufModule::platform() const { return impl_->platform; }
const loomc_module_t *BuiltGgufModule::module() const {
  return impl_->module.get();
}
bool GgufModelBuilder::Open(const std::string &path,
                            std::unique_ptr<GgufModelBuilder> *out,
                            std::string *error) {
  std::string ignored_error;
  if (!error)
    error = &ignored_error;
  if (!out || path.empty())
    return Fail(error, "GGUF input and builder output are required");
  std::error_code ec;
  auto asset = std::filesystem::canonical(path, ec);
  if (ec)
    return Fail(error, "cannot resolve GGUF input: " + ec.message());
  const auto alias = std::filesystem::absolute(path, ec);
  if (ec)
    return Fail(error, "cannot resolve GGUF input path: " + ec.message());
  std::unique_ptr<FILE, decltype(&std::fclose)> file(
      std::fopen(asset.c_str(), "rb"), &std::fclose);
  if (!file)
    return Fail(error,
                "cannot open GGUF input: " + std::string(std::strerror(errno)));
  if (std::setvbuf(file.get(), nullptr, _IONBF, 0) != 0)
    return Fail(error, "cannot disable GGUF input buffering");
  model::GgufCatalog catalog;
  auto impl = std::make_unique<Impl>();
  if (!model::ReadGgufCatalog(file.get(), &catalog, error) ||
      !model::LoadModelDescriptor(catalog, &impl->descriptor, error) ||
      !model::ReadTextMetadata(file.get(), catalog, &impl->text_metadata,
                               error))
    return false;
  // Source-only publication facts are captured once with the catalog FILE.
  // Unsupported templates/external tokenizers remain package-only concerns.
  // Retain a numerical requirement diagnostic without hiding valid metadata.
  DerivePublicationTokenFacts(file.get(), impl->descriptor.family,
                              impl->text_metadata, &impl->publication_tokens,
                              &impl->publication_token_error);
  impl->source_asset = asset.string();
  impl->source_alias = alias.string();
  if (std::fclose(file.release()) != 0)
    return Fail(error, "cannot close GGUF input");
  auto built =
      std::unique_ptr<GgufModelBuilder>(new GgufModelBuilder(std::move(impl)));
  *out = std::move(built);
  if (error)
    error->clear();
  return true;
}
bool GgufModelBuilder::ConfigureNumerical(
    const GgufBuildOptions &options, std::unique_ptr<GgufConfiguration> *out,
    std::string *error) const {
  std::string ignored_error;
  if (!error)
    error = &ignored_error;
  if (!out)
    return Fail(error, "configuration output is required");
  const auto *platform = SelectPlatform(options.platform);
  if (!platform)
    return Fail(error, "unsupported generation platform: " + options.platform +
                           "; maintained platform: amdgpu:gfx11-generic");
  const auto *architecture = SelectArchitecture(impl_->descriptor.family);
  if (!architecture)
    return Fail(error, "unsupported numerical architecture");
  if (!impl_->publication_token_error.empty())
    return Fail(error, impl_->publication_token_error);
  auto prepared = std::make_unique<GgufConfiguration::Impl>();
  ConfigurationSettings settings{options.prefill_capacity,
                                 options.context_capacity};
  if (!DeriveConfiguration(impl_->descriptor, impl_->source_asset, settings,
                           &prepared->numerical, error))
    return false;
  if (impl_->publication_tokens)
    prepared->numerical.source +=
        impl_->publication_tokens->configuration_source;
  prepared->numerical.source.insert(
      0, "// loomify platform: " + options.platform + "\n");
  prepared->source_asset = impl_->source_asset;
  prepared->source_alias = impl_->source_alias;
  prepared->options = options;
  prepared->architecture = architecture;
  prepared->platform = platform;
  auto configured = std::unique_ptr<GgufConfiguration>(
      new GgufConfiguration(std::move(prepared)));
  *out = std::move(configured);
  error->clear();
  return true;
}
bool GgufConfiguration::BuildModule(loomc_context_t *context,
                                    loomc_workspace_t *workspace,
                                    std::unique_ptr<BuiltGgufModule> *out,
                                    std::string *error) const {
  std::string ignored_error;
  if (!error)
    error = &ignored_error;
  if (!impl_ || !context || !workspace || !out ||
      impl_->options.library_root.empty())
    return Fail(
        error,
        "module build requires context, workspace, library root and output");
  const auto &architecture = *impl_->architecture;
  loomc_module_t *module = nullptr;
  if (!LinkNumericalProducts(context, workspace, impl_->options.library_root,
                             impl_->numerical.source, architecture,
                             *impl_->platform, &module, error))
    return false;
  auto built_impl = std::make_unique<BuiltGgufModule::Impl>();
  built_impl->module.reset(module);
  built_impl->configuration_source = impl_->numerical.source;
  built_impl->platform = impl_->options.platform;
  auto built = std::unique_ptr<BuiltGgufModule>(
      new BuiltGgufModule(std::move(built_impl)));
  *out = std::move(built);
  error->clear();
  return true;
}
bool BuiltGgufModule::WriteModule(const std::string &path,
                                  loomc_source_format_t format,
                                  std::string *error) const {
  if (!impl_ || !impl_->module || path.empty() ||
      (format != LOOMC_SOURCE_FORMAT_TEXT &&
       format != LOOMC_SOURCE_FORMAT_BYTECODE))
    return Fail(
        error, "module output requires a new path and text or bytecode format");
  std::error_code ec;
  const auto existing = std::filesystem::symlink_status(path, ec);
  if (existing.type() != std::filesystem::file_type::not_found && !ec)
    return Fail(error,
                "module output must not already exist or alias an input asset");
  if (ec && ec != std::errc::no_such_file_or_directory)
    return Fail(error, "cannot inspect module output: " + ec.message());
  // Serialize before acquiring the new output path; native failure publishes
  // nothing.
  loomc_module_serialize_options_t options{};
  options.format = format;
  loomc_source_t *raw = nullptr;
  auto status = loomc_module_serialize_to_source(
      impl_->module.get(), &options, loomc_allocator_system(), &raw);
  std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)> source(
      raw, loomc_source_release);
  const bool serialized = loomc_status_is_ok(status);
  if (!serialized) {
    auto message = loomc_status_message(status);
    if (error)
      error->assign(message.data, message.size);
  }
  loomc_status_free(status);
  if (!serialized)
    return false;
  if (!source)
    return Fail(error, "native serializer returned no source");
  FILE *file = std::fopen(path.c_str(), "wbx");
  if (!file)
    return Fail(error, "cannot create new module output: " +
                           std::string(std::strerror(errno)));
  const auto bytes = loomc_source_contents(source.get());
  bool written =
      std::fwrite(bytes.data, 1, bytes.data_length, file) == bytes.data_length;
  if (std::fclose(file) != 0)
    written = false;
  if (!written) {
    ec.clear();
    const bool removed = std::filesystem::remove(path, ec);
    std::string message = "cannot write or close module output";
    if (!removed || ec)
      message += "; cannot remove owned partial output: " + ec.message();
    return Fail(error, message);
  }
  if (error)
    error->clear();
  return true;
}
} // namespace hrx::loomify_gguf
