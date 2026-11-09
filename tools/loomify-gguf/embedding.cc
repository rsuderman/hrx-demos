// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/embedding.h"
#include "loomc/link.h"
#include "tools/loomify-gguf/embedding_sources.h"
#include <vector>
#include <cstring>
#include <limits>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
bool Check(loomc_status_t status, std::string *error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    error->assign(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Result(loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result))
    return true;
  *error = "Loom construction failed";
  if (result) {
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto *diagnostic = loomc_result_diagnostic_at(result, i);
      error->append("\n");
      error->append(diagnostic->message.data, diagnostic->message.size);
    }
  }
  return false;
}
bool Validate(const hrx::model::ModelDescriptor &model,
              const hrx::model::TensorRecord **tensor, std::string *error) {
  if (model.family != "llama") {
    *error = "embedding generation currently supports the llama architecture";
    return false;
  }
  for (const auto &binding : model.globals) {
    if (binding.role != "token_embedding")
      continue;
    if (*tensor) {
      *error = "ambiguous token_embedding role";
      return false;
    }
    *tensor = &binding.tensor;
  }
  if (!*tensor) {
    *error = "missing token_embedding role";
    return false;
  }
  const auto &t = **tensor;
  const auto width = model.embedding_width;
  const auto vocab = model.vocabulary_size;
  if (!width || !vocab || width > INT64_MAX || vocab > INT64_MAX ||
      width > uint64_t(INT64_MAX) / vocab / sizeof(float) ||
      t.normalized_extents !=
          std::array<int64_t, 4>{int64_t(width), int64_t(vocab), 1, 1}) {
    *error =
        "token_embedding dimensions do not match embedding/vocabulary size";
    return false;
  }
  if (t.storage_type != 0 || t.storage_name != "f32") {
    *error = "token_embedding requires F32 storage; got " + t.storage_name;
    return false;
  }
  if (t.name != "token_embd.weight" ||
      t.byte_size != width * vocab * sizeof(float)) {
    *error = "token_embedding named parameter or storage extent is invalid";
    return false;
  }
  return true;
}
} // namespace
bool BuildEmbeddingProgram(const hrx::model::ModelDescriptor &model,
                           loomc_context_t *context,
                           loomc_workspace_t *workspace,
                           loomc_module_t **out_module, std::string *error) {
  if (out_module)
    *out_module = nullptr;
  if (!error)
    return false;
  error->clear();
  if (!context || !workspace || !out_module) {
    *error = "context, workspace and module output are required";
    return false;
  }
  const hrx::model::TensorRecord *tensor = nullptr;
  if (!Validate(model, &tensor, error))
    return false;
  const std::string config = "config.def @loomify.embedding.width = " +
      std::to_string(model.embedding_width) + " : index\n" +
      "config.def @loomify.embedding.vocabulary = " +
      std::to_string(model.vocabulary_size) + " : index\n";
  std::vector<Owner<loomc_module_t, loomc_module_release>> modules;
  auto load = [&](const char *name, const void *bytes, size_t size) {
    loomc_source_options_t options{};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier = loomc_make_cstring_view(name);
    options.contents = loomc_make_byte_span(bytes, size);
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t *source = nullptr;
    if (!Check(loomc_source_create(&options, loomc_allocator_system(), &source), error)) return false;
    Owner<loomc_source_t, loomc_source_release> source_owner(source);
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    auto status = loomc_module_deserialize_from_source(context, workspace, source,
        nullptr, loomc_allocator_system(), &module, &result);
    Owner<loomc_module_t, loomc_module_release> owner(module);
    Owner<loomc_result_t, loomc_result_release> result_owner(result);
    if (!Check(status, error) || !Result(result, error)) return false;
    modules.push_back(std::move(owner));
    return true;
  };
  if (!load("embedding-config.loom", reinterpret_cast<const uint8_t *>(config.data()), config.size())) return false;
  const auto *sources = embedding_sources_create();
  for (size_t i = 0; i < embedding_sources_size(); ++i)
    if (!load(sources[i].name, sources[i].data, sources[i].size)) return false;
  std::vector<loomc_link_module_provider_t> providers;
  for (const auto &module : modules)
    providers.push_back({module.get(), {}, LOOMC_LINK_PROVIDER_ROLE_INPUT});
  loomc_linker_t *linker = nullptr;
  if (!Check(loomc_linker_create(context, nullptr, loomc_allocator_system(), &linker), error)) return false;
  Owner<loomc_linker_t, loomc_linker_release> linker_owner(linker);
  const auto root = loomc_make_cstring_view("model.embedding");
  loomc_link_options_t options{};
  options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  options.structure_size = sizeof(options);
  options.mode = LOOMC_LINK_MODE_LINK;
  options.module_providers = providers.data();
  options.module_provider_count = providers.size();
  options.root_symbols = &root;
  options.root_symbol_count = 1;
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  auto status = loomc_link_module(linker, workspace, &options, &output, &result);
  Owner<loomc_module_t, loomc_module_release> output_owner(output);
  Owner<loomc_result_t, loomc_result_release> result_owner(result);
  if (!Check(status, error) || !Result(result, error)) return false;
  *out_module = output_owner.release();
  return true;
}
} // namespace hrx::loomify_gguf
