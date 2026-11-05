// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/package.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
namespace hrx::loomify_gguf {
namespace {
namespace fs = std::filesystem;
bool Fail(std::string *error, const std::string &text) {
  if (error)
    *error = text;
  return false;
}
bool Read(const fs::path &path, std::string *text, std::string *error) {
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
void Replace(std::string &text, const std::string &key,
             const std::string &value) {
  const std::string token = "{{" + key + "}}";
  size_t begin = 0;
  while ((begin = text.find(token, begin)) != std::string::npos) {
    text.replace(begin, token.size(), value);
    begin += value.size();
  }
}
} // namespace
bool WriteLlamaPackage(const model::ModelDescriptor &model,
                       const LlamaPackageOptions &o,
                       const loomc_module_t *module, std::string *error) {
  if (!module || model.layers.empty() || !model.embedding_width ||
      !o.prefill_capacity || o.prefill_capacity > 512 ||
      o.context_capacity < o.prefill_capacity || o.context_capacity > 131072 ||
      o.output_directory.empty() || o.prefill_root.empty() ||
      o.decode_root.empty() || o.epoch_root.empty() ||
      (o.format != LOOMC_SOURCE_FORMAT_BYTECODE &&
       o.format != LOOMC_SOURCE_FORMAT_TEXT))
    return Fail(error, "invalid bounded Llama package configuration");
  std::error_code ec;
  fs::path gguf = fs::canonical(o.gguf, ec);
  if (ec || !fs::is_regular_file(gguf))
    return Fail(error, "GGUF must be an existing regular file");
  fs::path tokenizer = fs::canonical(o.tokenizer, ec);
  if (ec || !fs::is_regular_file(tokenizer))
    return Fail(error, "tokenizer must be an existing tokenizer.json file");
  const fs::path destination(o.output_directory);
  if (fs::exists(destination, ec) || ec)
    return Fail(error, "package output directory must not already exist");
  const fs::path providers =
      fs::path(o.source_root) / "tools/loomify-gguf/serving";
  std::string prepare, control, weights;
  if (!Read(providers / "prepare.loom.in", &prepare, error) ||
      !Read(providers / "control.loom", &control, error) ||
      !Read(providers / "weights.loom", &weights, error))
    return false;
  Replace(prepare, "HIDDEN", std::to_string(model.embedding_width));
  Replace(prepare, "FFN",
          std::to_string(model.layers.front().feed_forward_width));
  Replace(prepare, "KV",
          std::to_string(model.layers.front().kv_heads * model.key_head_width));
  Replace(prepare, "LAYERS", std::to_string(model.layers.size()));
  Replace(prepare, "CONTEXT", std::to_string(o.context_capacity));
  Replace(prepare, "POOL",
          std::to_string(((o.context_capacity + 63) / 64) * 64));
  Replace(prepare, "TOKENS", std::to_string(o.prefill_capacity));
  Replace(prepare, "HISTORY",
          std::to_string(std::max<uint64_t>(o.prefill_capacity, 129)));
  Replace(prepare, "PREFILL_ROOT_HEX", Hex(o.prefill_root));
  Replace(prepare, "DECODE_ROOT_HEX", Hex(o.decode_root));
  Replace(prepare, "EPOCH_ROOT_HEX", Hex(o.epoch_root));
  if (prepare.find("{{") != std::string::npos)
    return Fail(error, "unresolved prepare template field");
  if (!fs::create_directory(destination, ec) || ec)
    return Fail(error,
                "cannot create package output directory: " + ec.message());
  // Only files created by this call exist here. On failure remove this fresh
  // directory so a model document never advertises partial output.
  auto cleanup = [&] {
    std::error_code ignored;
    fs::remove_all(destination, ignored);
    return false;
  };
  if (!Write(destination / "prepare.loom", prepare, error) ||
      !Write(destination / "control.loom", control, error) ||
      !Write(destination / "weights.loom", weights, error))
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
  document << "{\n  \"name\": \"loomify-llama\",\n  \"definition\": {\n"
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
  if (!Write(destination / "model.json", document.str(), error))
    return cleanup();
  return true;
}
} // namespace hrx::loomify_gguf
