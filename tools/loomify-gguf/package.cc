// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/gguf_model_builder_internal.h"
#include "tools/loomify-gguf/linking.h"
#include "loomc/source.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace hrx::loomify_gguf {
namespace {
namespace fs = std::filesystem;
bool Fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}
bool ValidPath(const std::string& path) {
  return !path.empty() && path.find('\0') == std::string::npos;
}
bool Status(loomc_status_t status, std::string* error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    Fail(error, std::string(message.data, message.size));
  }
  loomc_status_free(status);
  return ok;
}
bool ReadText(const fs::path& path, std::string* text, std::string* error) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec) || ec)
    return Fail(error, "serving provider must be an existing regular file: " + path.string());
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return Fail(error, "cannot open serving provider: " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (stream.bad()) return Fail(error, "cannot read serving provider: " + path.string());
  *text = contents.str();
  return true;
}
bool Write(const fs::path& path, const std::string& text, std::string* error) {
  std::ofstream stream(path, std::ios::binary);
  stream << text;
  stream.close();
  return stream ? true : Fail(error, "cannot write serving provider: " + path.string());
}
std::string Json(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << char(c);
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    else out << char(c);
  }
  out << '"';
  return out.str();
}
struct CompanionSources { std::string prepare, control, weights; };

// Preserve the existing publication contract: only a newly created directory
// is owned by this call; model.json is closed and renamed into place last.
// Old serving renderers retire with their final consumers in delivery10.
bool Publish(const fs::path& gguf, const fs::path& tokenizer,
             const GgufBuildOptions& build, const GgufPackageOptions& package,
             const char* name, const CompanionSources& sources,
             const loomc_module_t* module, std::string* error) {
  const fs::path destination(package.output_directory);
  std::error_code ec;
  if (!fs::create_directory(destination, ec) || ec)
    return Fail(error, "cannot create new package output directory: " + ec.message());
  const auto cleanup = [&] {
    std::error_code failure;
    fs::remove_all(destination, failure);
    if (failure && error) *error += "; partial output cleanup failed: " + failure.message();
    return false;
  };
  if (!Write(destination / "prepare.loom", sources.prepare, error) ||
      !Write(destination / "control.loom", sources.control, error) ||
      !Write(destination / "weights.loom", sources.weights, error)) return cleanup();
  const std::string model_file = package.format == LOOMC_SOURCE_FORMAT_BYTECODE
      ? "model.loombc" : "model.loom";
  loomc_module_serialize_options_t serialization{};
  serialization.format = package.format;
  if (!Status(loomc_module_serialize_to_path(module, &serialization,
      loomc_make_cstring_view((destination / model_file).c_str()),
      loomc_allocator_system()), error)) return cleanup();
  std::ostringstream document;
  document << "{\n  \"name\": " << Json(name) << ",\n  \"definition\": {\n"
           << "    \"kind\": \"text\",\n    \"prepare\": \"prepare.loom\",\n"
           << "    \"control\": \"control.loom\",\n    \"sources\": ["
           << Json(model_file) << "],\n"
           << "    \"weights\": " << Json(gguf.string()) << ",\n"
           << "    \"tokenizer\": " << Json(tokenizer.string()) << ",\n"
           << "    \"rows\": 1,\n    \"context_capacity\": " << build.context_capacity << ",\n"
           << "    \"pool_capacity\": " << ((build.context_capacity + 63) / 64) * 64
           << ",\n    \"prefill_capacity\": " << build.prefill_capacity << ",\n"
           << "    \"max_tokens\": 16,\n    \"session_cache_capacity\": 0,\n"
           << "    \"prefix_cache_capacity\": 0\n  }\n}\n";
  if (!Write(destination / "model.json.tmp", document.str(), error)) return cleanup();
  fs::rename(destination / "model.json.tmp", destination / "model.json", ec);
  if (ec) {
    Fail(error, "cannot publish model document: " + ec.message());
    return cleanup();
  }
  if (error) error->clear();
  return true;
}
} // namespace

bool GgufModelBuilder::WritePackage(loomc_context_t* context, loomc_workspace_t* workspace,
    const GgufBuildOptions& build, const GgufPackageOptions& package, std::string* error) const {
  if (!impl_ || !context || !workspace || !ValidPath(package.output_directory) ||
      !ValidPath(package.tokenizer) ||
      (package.format != LOOMC_SOURCE_FORMAT_BYTECODE && package.format != LOOMC_SOURCE_FORMAT_TEXT))
    return Fail(error, "invalid serving package output configuration");
  std::error_code ec;
  const auto gguf = fs::canonical(impl_->source_asset, ec);
  if (ec || !fs::is_regular_file(gguf, ec) || ec)
    return Fail(error, "GGUF must be an existing regular file");
  const auto tokenizer = fs::canonical(package.tokenizer, ec);
  if (ec || !fs::is_regular_file(tokenizer, ec) || ec)
    return Fail(error, "tokenizer must be an existing tokenizer.json file");
  if (fs::exists(package.output_directory, ec) || ec)
    return Fail(error, "package output directory must not already exist");

  std::unique_ptr<BuiltGgufModule> numerical;
  if (!BuildModule(context, workspace, build, &numerical, error)) return false;
  const auto& built = *numerical->impl_;
  const auto& files = *built.architecture;
  const fs::path library_root(built.options.library_root);
  loomc_module_t* raw_prepare = nullptr;
  if (!LinkConfiguredProduct(context, workspace, built.configuration_source,
      (library_root / files.preparation).string(), {"prepare"}, false, &raw_prepare, error)) return false;
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> preparation(raw_prepare, loomc_module_release);
  loomc_module_serialize_options_t serialization{};
  serialization.format = LOOMC_SOURCE_FORMAT_TEXT;
  loomc_source_t* raw_source = nullptr;
  if (!Status(loomc_module_serialize_to_source(preparation.get(), &serialization,
      loomc_allocator_system(), &raw_source), error)) return false;
  std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)> source(raw_source, loomc_source_release);
  const auto contents = loomc_source_contents(source.get());
  CompanionSources companions;
  companions.prepare.assign(reinterpret_cast<const char*>(contents.data), contents.data_length);
  if (!ReadText(library_root / files.control, &companions.control, error) ||
      !ReadText(library_root / files.weights, &companions.weights, error)) return false;
  return Publish(gguf, tokenizer, built.options, package, files.package_name,
      companions, numerical->module(), error);
}
} // namespace hrx::loomify_gguf
