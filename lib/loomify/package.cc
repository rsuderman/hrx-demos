// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/gguf_model_builder_internal.h"
#include "lib/loomify/linking.h"
#include "lib/loomify/chat_policy.h"
#include "lib/loomify/token_policy.h"
#include "lib/loomify/template_analysis.h"
#include "lib/serving/library_resources.h"
#include <cstdio>
#include <cstdint>
#include "loomc/source.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>
#include <algorithm>

namespace hrx::loomify_gguf {
// Package-only state is deliberately completed outside the numerical library.
struct PackageConfiguration {
  GgufPackageOptions options;
  PreparedTextControl text;
  std::string source_key;
  size_t source_size = 0;
};
namespace {
namespace fs = std::filesystem;
bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
bool ValidPath(const std::string &path) {
  return !path.empty() && path.find('\0') == std::string::npos;
}
bool Status(loomc_status_t status, std::string *error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    Fail(error, std::string(message.data, message.size));
  }
  loomc_status_free(status);
  return ok;
}
bool ReadText(const fs::path &path, std::string *text, std::string *error) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec) || ec)
    return Fail(error, "serving provider must be an existing regular file: " +
                           path.string());
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return Fail(error, "cannot open serving provider: " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (stream.bad())
    return Fail(error, "cannot read serving provider: " + path.string());
  *text = contents.str();
  return true;
}
bool Write(const fs::path &path, const std::string &text, std::string *error) {
  std::ofstream stream(path, std::ios::binary);
  stream << text;
  stream.close();
  return stream
             ? true
             : Fail(error, "cannot write serving provider: " + path.string());
}
std::string Json(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << char(c);
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << unsigned(c);
    else
      out << char(c);
  }
  out << '"';
  return out.str();
}
std::string Hex(std::string_view text) {
  constexpr char digits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(text.size() * 2);
  for (unsigned char c : text) {
    encoded.push_back(digits[c >> 4]);
    encoded.push_back(digits[c & 15]);
  }
  return encoded;
}
bool SerializeText(const loomc_module_t *module, std::string *text,
                   std::string *error) {
  loomc_module_serialize_options_t options{};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  loomc_source_t *raw = nullptr;
  const auto status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &raw);
  std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)> source(
      raw, loomc_source_release);
  if (!Status(status, error))
    return false;
  if (!source)
    return Fail(error, "native serialization returned no source");
  const auto bytes = loomc_source_contents(source.get());
  text->assign(reinterpret_cast<const char *>(bytes.data), bytes.data_length);
  return true;
}
struct CompanionSources {
  std::string prepare, control, weights;
};

// Preserve the existing publication contract: only a newly created directory
// is owned by this call; model.json is closed and renamed into place last.
bool Publish(const fs::path &gguf, const fs::path &tokenizer,
             const GgufBuildOptions &build, const GgufPackageOptions &package,
             const char *name, const CompanionSources &sources,
             const loomc_module_t *module, std::string *error) {
  const fs::path destination(package.output_directory);
  std::error_code ec;
  if (!fs::create_directory(destination, ec) || ec)
    return Fail(error,
                "cannot create new package output directory: " + ec.message());
  const auto cleanup = [&] {
    std::error_code failure;
    fs::remove_all(destination, failure);
    if (failure && error)
      *error += "; partial output cleanup failed: " + failure.message();
    return false;
  };
  if (!Write(destination / "prepare.loom", sources.prepare, error) ||
      !Write(destination / "control.loom", sources.control, error) ||
      !Write(destination / "weights.loom", sources.weights, error))
    return cleanup();
  const std::string model_file = package.format == LOOMC_SOURCE_FORMAT_BYTECODE
                                     ? "model.loombc"
                                     : "model.loom";
  loomc_module_serialize_options_t serialization{};
  serialization.format = package.format;
  if (!Status(loomc_module_serialize_to_path(
                  module, &serialization,
                  loomc_make_cstring_view((destination / model_file).c_str()),
                  loomc_allocator_system()),
              error))
    return cleanup();
  std::ostringstream document;
  document << "{\n  \"name\": " << Json(name) << ",\n  \"definition\": {\n"
           << "    \"kind\": \"text\",\n    \"prepare\": \"prepare.loom\",\n"
           << "    \"control\": \"control.loom\",\n    \"sources\": ["
           << Json(model_file) << "],\n"
           << "    \"weights\": " << Json(gguf.string()) << ",\n"
           << "    \"tokenizer\": " << Json(tokenizer.string()) << ",\n"
           << "    \"rows\": 1,\n    \"max_sessions\": 1,\n"
           << "    \"context_capacity\": " << build.context_capacity << ",\n"
           << "    \"prefill_capacity\": " << build.prefill_capacity << ",\n"
           << "    \"max_tokens\": 16,\n    \"session_cache_capacity\": 0,\n"
           << "    \"prefix_cache_capacity\": 0\n  }\n}\n";
  if (!Write(destination / "model.json.tmp", document.str(), error))
    return cleanup();
  fs::rename(destination / "model.json.tmp", destination / "model.json", ec);
  if (ec) {
    Fail(error, "cannot publish model document: " + ec.message());
    return cleanup();
  }
  if (error)
    error->clear();
  return true;
}
} // namespace

bool ConfigurePackageData(
    const std::string &source_asset, const model::TextMetadata &text_metadata,
    const std::optional<PublicationTokenFacts> &publication_tokens,
    const ArchitectureFiles &files, const GgufBuildOptions &build,
    const GgufPackageOptions &package, PackageConfiguration *out,
    std::string *error) {
  if (!out || !ValidPath(package.tokenizer) || !ValidPath(build.library_root) ||
      (package.format != LOOMC_SOURCE_FORMAT_BYTECODE &&
       package.format != LOOMC_SOURCE_FORMAT_TEXT))
    return Fail(error, "invalid serving package input configuration");
  std::error_code ec;
  const auto gguf = fs::canonical(source_asset, ec);
  if (ec || !fs::is_regular_file(gguf, ec) || ec)
    return Fail(error, "GGUF must be an existing regular file");
  const auto tokenizer = fs::canonical(package.tokenizer, ec);
  if (ec || !fs::is_regular_file(tokenizer, ec) || ec)
    return Fail(error, "tokenizer must be an existing tokenizer.json file");
  // Actual source selection and tokenizer association are package-only.
  // Numerical BuildModule has no chat compiler or tokenizer dependency.
  ChatTemplateSelection selected;
  if (!SelectChatTemplateSource(text_metadata, package.template_name, &selected,
                                error))
    return false;
  std::string tokenizer_json;
  if (!ReadText(tokenizer, &tokenizer_json, error))
    return false;
  std::unique_ptr<FILE, decltype(&std::fclose)> input(
      std::fopen(gguf.c_str(), "rb"), std::fclose);
  if (!input || std::setvbuf(input.get(), nullptr, _IONBF, 0))
    return Fail(error, "cannot open unbuffered GGUF for package token binding");
  const auto scratch_parent = fs::temp_directory_path(ec);
  if (ec)
    return Fail(error,
                "cannot resolve text preparation scratch: " + ec.message());
  // Each CPU preparation owns its unique scratch children and destroys them
  // before returning. Inspection never uses or creates the publication path.
  const jinja::TemplateAnalysisOptions analysis_options{
      build.library_root, scratch_parent.string(), false};
  PreparedTextControl text;
  if (!PrepareGeneratedText(selected, input.get(), text_metadata,
                            publication_tokens, tokenizer_json,
                            analysis_options, &text, error))
    return false;
  if (std::fclose(input.release()) != 0)
    return Fail(error, "cannot close package GGUF token input");

  PackageConfiguration prepared;
  prepared.options = package;
  prepared.options.tokenizer = tokenizer.string();
  text.source +=
      "\nglobal.rodata.def public @loomify.text.model_name = bytes(\"" +
      Hex(files.package_name) + "\")\n";
  for (const char *root :
       {"step", "epoch", "encode_epoch", "publish_epoch", "encode_state",
        "resize_state", "request_defaults", "request_options", "bind_request",
        "model_name", "chat_image_limits", "loomify.text.model_name"})
    if (std::find(text.roots.begin(), text.roots.end(), root) ==
        text.roots.end())
      text.roots.push_back(root);
  for (const char *helper :
       {resources::kTextIdentity, resources::kControlInterface})
    if (std::find(text.helper_requirements.begin(),
                  text.helper_requirements.end(),
                  helper) == text.helper_requirements.end())
      text.helper_requirements.push_back(helper);
  for (const auto &helper : text.helper_requirements) {
    bool declared = false;
    for (const char *product : resources::kAllProducts)
      declared |= helper == product;
    if (!declared)
      return Fail(error, "undeclared generated text helper: " + helper);
    ec.clear();
    if (!fs::is_regular_file(fs::path(build.library_root) / helper, ec) || ec)
      return Fail(error, "missing generated text helper: " + helper);
  }
  prepared.source_key = std::move(selected.source_key);
  prepared.source_size = selected.source_text.size();
  prepared.text = std::move(text);
  *out = std::move(prepared);
  if (error)
    error->clear();
  return true;
}

const std::string &GgufConfiguration::package_data_source() const {
  static const std::string empty;
  return impl_->package ? impl_->package->text.source : empty;
}
bool GgufModelBuilder::Configure(const GgufBuildOptions &options,
                                 const GgufPackageOptions *package,
                                 std::unique_ptr<GgufConfiguration> *out,
                                 std::string *error) const {
  std::string ignored_error;
  if (!error)
    error = &ignored_error;
  if (!out)
    return Fail(error, "configuration output is required");
  std::unique_ptr<GgufConfiguration> configured;
  if (!ConfigureNumerical(options, &configured, error))
    return false;
  if (package) {
    auto data = std::make_shared<PackageConfiguration>();
    if (!ConfigurePackageData(impl_->source_asset, impl_->text_metadata,
                              impl_->publication_tokens,
                              *configured->impl_->architecture, options,
                              *package, data.get(), error))
      return false;
    configured->impl_->package = std::move(data);
  }
  *out = std::move(configured);
  error->clear();
  return true;
}
bool GgufConfiguration::WritePackage(loomc_context_t *context,
                                     loomc_workspace_t *workspace,
                                     std::string *error) const {
  std::string ignored_error;
  if (!error)
    error = &ignored_error;
  if (!impl_ || !impl_->package || !context || !workspace ||
      !ValidPath(impl_->package->options.output_directory))
    return Fail(error, "invalid serving package output configuration");
  if (impl_->options.context_capacity % 64 != 0)
    return Fail(error,
                "Serve package context_capacity must be a multiple of 64");
  const auto &package = impl_->package->options;
  const auto &text = impl_->package->text;
  const fs::path gguf(impl_->source_asset), tokenizer(package.tokenizer);
  std::error_code ec;
  if (!fs::is_regular_file(gguf, ec) || ec)
    return Fail(error, "GGUF must be an existing regular file");
  const fs::path source_alias(impl_->source_alias);
  const auto current_source = fs::canonical(source_alias, ec);
  if (ec || current_source != gguf)
    return Fail(error,
                "GGUF input alias no longer resolves to the opened source");
  if (!fs::is_regular_file(tokenizer, ec) || ec)
    return Fail(error, "tokenizer must be an existing tokenizer.json file");
  if (fs::exists(package.output_directory, ec) || ec)
    return Fail(error, "package output directory must not already exist");

  std::unique_ptr<BuiltGgufModule> numerical;
  if (!BuildModule(context, workspace, &numerical, error))
    return false;
  const auto &files = *impl_->architecture;
  const fs::path library_root(impl_->options.library_root);
  // Weight preparation needs the bound terminal marker, but no input renderer.
  const auto preparation_configuration =
      impl_->numerical.source + text.token_source;
  const auto joined_configuration = impl_->numerical.source + text.source;
  loomc_module_t *raw_prepare = nullptr;
  if (!LinkConfiguredProducts(context, workspace, preparation_configuration,
                              {(library_root / files.preparation).string()},
                              {"prepare"}, false, &raw_prepare, error))
    return false;
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> preparation(
      raw_prepare, loomc_module_release);
  // A cross-unit rodata declaration is not sufficient to retain its definition.
  // Keep this package-owned data alongside the fixed public control functions.
  std::vector<const char *> control_roots;
  for (const auto &root : text.roots)
    control_roots.push_back(root.c_str());
  loomc_module_t *raw_control = nullptr;
  std::vector<std::string> control_products{
      (library_root / files.execution).string(),
      (library_root / resources::kGreedy).string()};
  for (const auto &product : text.helper_requirements) {
    const auto path = (library_root / product).string();
    if (std::find(control_products.begin(), control_products.end(), path) ==
        control_products.end())
      control_products.push_back(path);
  }
  if (!LinkConfiguredProducts(context, workspace, joined_configuration,
                              control_products, control_roots, false,
                              &raw_control, error))
    return false;
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> control(
      raw_control, loomc_module_release);
  CompanionSources companions;
  if (!SerializeText(preparation.get(), &companions.prepare, error) ||
      !SerializeText(control.get(), &companions.control, error) ||
      !ReadText(library_root / files.weights, &companions.weights, error))
    return false;
  // The effective manifest rejects arbitrary metadata. Documentary identity is
  // kept in the existing text provider, without changing the runtime contract.
  companions.control = "// GGUF template " + Json(impl_->package->source_key) +
                       " bytes " + std::to_string(impl_->package->source_size) +
                       "\n" + companions.control;
  return Publish(source_alias, tokenizer, impl_->options, package,
                 files.package_name, companions, numerical->module(), error);
}
} // namespace hrx::loomify_gguf
