// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/package_internal.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
namespace hrx::loomify_gguf {
namespace detail {
namespace fs = std::filesystem;
bool Fail(std::string *error, const std::string &text) {
  if (error)
    *error = text;
  return false;
}
bool ReadText(const std::string &file, std::string *text, std::string *error) {
  const fs::path path(file);
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return Fail(error, "cannot open serving provider: " + path.string());
  std::ostringstream result;
  result << stream.rdbuf();
  if (stream.bad())
    return Fail(error, "cannot read serving provider: " + path.string());
  *text = result.str();
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
std::string Hex(const std::string &text) {
  std::ostringstream out;
  for (unsigned char c : text)
    out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
  return out.str();
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
void ReplaceToken(std::string &text, const std::string &key,
                  const std::string &value) {
  const std::string token = "{{" + key + "}}";
  size_t begin = 0;
  while ((begin = text.find(token, begin)) != std::string::npos) {
    text.replace(begin, token.size(), value);
    begin += value.size();
  }
}
bool WriteServingPackage(const ModelPackageOptions &o, const std::string &name,
                         const CompanionSources &sources,
                         const loomc_module_t *module, std::string *error) {
  if (!module || o.output_directory.empty() || !o.prefill_capacity ||
      o.prefill_capacity > 512 || o.context_capacity < o.prefill_capacity ||
      (o.format != LOOMC_SOURCE_FORMAT_BYTECODE &&
       o.format != LOOMC_SOURCE_FORMAT_TEXT))
    return Fail(error, "invalid serving package output configuration");
  std::error_code ec;
  fs::path gguf = fs::canonical(o.gguf, ec);
  if (ec || !fs::is_regular_file(gguf, ec) || ec)
    return Fail(error, "GGUF must be an existing regular file");
  fs::path tokenizer = fs::canonical(o.tokenizer, ec);
  if (ec || !fs::is_regular_file(tokenizer, ec) || ec)
    return Fail(error, "tokenizer must be an existing tokenizer.json file");
  const fs::path destination(o.output_directory);
  if (fs::exists(destination, ec) || ec)
    return Fail(error, "package output directory must not already exist");
  if (!fs::create_directory(destination, ec) || ec)
    return Fail(error,
                "cannot create package output directory: " + ec.message());
  auto cleanup = [&] {
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
  const std::string model_file =
      o.format == LOOMC_SOURCE_FORMAT_BYTECODE ? "model.loombc" : "model.loom";
  loomc_module_serialize_options_t serialization{};
  serialization.format = o.format;
  auto status = loomc_module_serialize_to_path(
      module, &serialization,
      loomc_make_cstring_view((destination / model_file).c_str()),
      loomc_allocator_system());
  if (!loomc_status_is_ok(status)) {
    auto message = loomc_status_message(status);
    Fail(error, std::string(message.data, message.size));
    loomc_status_free(status);
    return cleanup();
  }
  loomc_status_free(status);
  std::ostringstream document;
  document << "{\n  \"name\": " << Json(name) << ",\n  \"definition\": {\n"
           << "    \"kind\": \"text\",\n    \"prepare\": \"prepare.loom\",\n"
           << "    \"control\": \"control.loom\",\n    \"sources\": ["
           << Json(model_file) << "],\n"
           << "    \"weights\": " << Json(gguf.string()) << ",\n"
           << "    \"tokenizer\": " << Json(tokenizer.string()) << ",\n"
           << "    \"rows\": 1,\n    \"context_capacity\": "
           << o.context_capacity << ",\n"
           << "    \"pool_capacity\": " << ((o.context_capacity + 63) / 64) * 64
           << ",\n    \"prefill_capacity\": " << o.prefill_capacity << ",\n"
           << "    \"max_tokens\": 16,\n    \"session_cache_capacity\": 0,\n"
           << "    \"prefix_cache_capacity\": 0\n  }\n}\n";
  // A failed write must not leave a model document advertising the package.
  // The directory is new and owned by this invocation; publish its final name
  // only after the complete document has been closed successfully.
  if (!Write(destination / "model.json.tmp", document.str(), error))
    return cleanup();
  fs::rename(destination / "model.json.tmp", destination / "model.json", ec);
  if (ec) {
    Fail(error, "cannot publish model document: " + ec.message());
    return cleanup();
  }
  return true;
}
} // namespace detail
bool WriteLlamaPackage(const model::ModelDescriptor &model,
                       const ModelPackageOptions &o,
                       const loomc_module_t *module, std::string *error) {
  if (!module || model.layers.empty() || !model.embedding_width ||
      !o.prefill_capacity || o.prefill_capacity > 512 ||
      o.context_capacity < o.prefill_capacity || o.context_capacity > 131072) {
    if (error)
      *error = "invalid bounded Llama package configuration";
    return false;
  }
  const auto providers =
      std::filesystem::path(o.source_root) / "tools/loomify-gguf/serving";
  detail::CompanionSources sources;
  if (!detail::ReadText((providers / "prepare.loom.in").string(),
                        &sources.prepare, error) ||
      !detail::ReadText((providers / "control.loom").string(), &sources.control,
                        error) ||
      !detail::ReadText((providers / "weights.loom").string(), &sources.weights,
                        error))
    return false;
  auto &prepare = sources.prepare;
  detail::ReplaceToken(prepare, "HIDDEN",
                       std::to_string(model.embedding_width));
  detail::ReplaceToken(prepare, "FFN",
                       std::to_string(model.layers.front().feed_forward_width));
  detail::ReplaceToken(
      prepare, "KV",
      std::to_string(model.layers.front().kv_heads * model.key_head_width));
  detail::ReplaceToken(prepare, "LAYERS", std::to_string(model.layers.size()));
  detail::ReplaceToken(prepare, "CONTEXT", std::to_string(o.context_capacity));
  detail::ReplaceToken(prepare, "POOL",
                       std::to_string(((o.context_capacity + 63) / 64) * 64));
  detail::ReplaceToken(prepare, "TOKENS", std::to_string(o.prefill_capacity));
  detail::ReplaceToken(
      prepare, "HISTORY",
      std::to_string(std::max<uint64_t>(o.prefill_capacity, 129)));
  detail::ReplaceToken(prepare, "PREFILL_ROOT_HEX",
                       detail::Hex(kLlamaPrefillRoot));
  detail::ReplaceToken(prepare, "DECODE_ROOT_HEX",
                       detail::Hex(kLlamaDecodeRoot));
  detail::ReplaceToken(prepare, "EPOCH_ROOT_HEX", detail::Hex(kLlamaEpochRoot));
  if (prepare.find("{{") != std::string::npos) {
    if (error)
      *error = "unresolved prepare template field";
    return false;
  }
  return detail::WriteServingPackage(o, "loomify-llama", sources, module,
                                     error);
}
} // namespace hrx::loomify_gguf
