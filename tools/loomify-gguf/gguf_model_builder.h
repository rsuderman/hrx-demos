// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_GGUF_MODEL_BUILDER_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_GGUF_MODEL_BUILDER_H_
#include "loomc/module.h"
#include "tools/model/model_descriptor.h"
#include <memory>
#include <string>
namespace hrx::loomify_gguf {
struct GgufBuildOptions {
  std::string library_root;
  uint64_t prefill_capacity = 128;
  uint64_t context_capacity = 512;
};
struct GgufPackageOptions {
  std::string output_directory;
  std::string tokenizer;
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_BYTECODE;
};
class BuiltGgufModule final {
public:
  ~BuiltGgufModule();
  BuiltGgufModule(BuiltGgufModule &&) noexcept;
  BuiltGgufModule &operator=(BuiltGgufModule &&) noexcept;
  BuiltGgufModule(const BuiltGgufModule &) = delete;
  BuiltGgufModule &operator=(const BuiltGgufModule &) = delete;
  const std::string &configuration_source() const;
  const loomc_module_t *module() const;
  // Only a new file is accepted. Failure removes this call's partial output;
  // existing files and input assets remain untouched, and this result is usable.
  bool WriteModule(const std::string &new_path, loomc_source_format_t format,
                   std::string *error) const;
private:
  friend class GgufModelBuilder;
  struct Impl;
  explicit BuiltGgufModule(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
class GgufModelBuilder final {
public:
  ~GgufModelBuilder();
  GgufModelBuilder(const GgufModelBuilder &) = delete;
  GgufModelBuilder &operator=(const GgufModelBuilder &) = delete;
  // Reads metadata/tensor ranges only; owns the copied facts and closes input.
  static bool Open(const std::string &gguf_path,
                   std::unique_ptr<GgufModelBuilder> *out, std::string *error);
  const model::ModelDescriptor &descriptor() const;
  // Inputs are borrowed synchronously; shared workspace use is externally
  // synchronized. Failure preserves *out. The result owns its native reference
  // and may outlive this builder and the caller's context/workspace references.
  bool BuildModule(loomc_context_t *, loomc_workspace_t *, const GgufBuildOptions &,
                   std::unique_ptr<BuiltGgufModule> *out, std::string *error) const;
  bool WritePackage(loomc_context_t *, loomc_workspace_t *, const GgufBuildOptions &,
                    const GgufPackageOptions &, std::string *error) const;
private:
  struct Impl;
  explicit GgufModelBuilder(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
} // namespace hrx::loomify_gguf
#endif
