// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_GGUF_MODEL_BUILDER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_GGUF_MODEL_BUILDER_H_
#include "loomc/module.h"
#include "tools/model/model_descriptor.h"
#include "lib/loomify/publication.h"
#include "tools/model/text_metadata.h"
#include <memory>
#include <optional>
#include <string>
namespace hrx::loomify_gguf {
struct GgufBuildOptions {
  std::string library_root;
  uint64_t prefill_capacity = 128;
  uint64_t context_capacity = 512;
  // Maintained generation recipe; Serve independently selects its live device.
  std::string platform = "amdgpu:gfx11-generic";
};
struct GgufPackageOptions {
  std::string output_directory;
  std::string tokenizer;
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_BYTECODE;
  // Optional named entry from this GGUF; never a provider/profile override.
  std::optional<std::string> template_name;
};
class GgufConfiguration;
class BuiltGgufModule final {
public:
  ~BuiltGgufModule();
  BuiltGgufModule(BuiltGgufModule &&) noexcept;
  BuiltGgufModule &operator=(BuiltGgufModule &&) noexcept;
  BuiltGgufModule(const BuiltGgufModule &) = delete;
  BuiltGgufModule &operator=(const BuiltGgufModule &) = delete;
  const std::string &configuration_source() const;
  const std::string &platform() const;
  const loomc_module_t *module() const;
  // Only a new file is accepted. Failure removes this call's partial output;
  // existing files and input assets remain untouched, and this result is
  // usable.
  bool WriteModule(const std::string &new_path, loomc_source_format_t format,
                   std::string *error) const;

private:
  friend class GgufConfiguration;
  struct Impl;
  explicit BuiltGgufModule(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
// Owned source configuration, prepared before Loom compilation or LINK. No
// compiler context, product files or output directory are needed to inspect it.
class GgufConfiguration final {
public:
  ~GgufConfiguration();
  GgufConfiguration(GgufConfiguration &&) noexcept;
  GgufConfiguration &operator=(GgufConfiguration &&) noexcept;
  GgufConfiguration(const GgufConfiguration &) = delete;
  GgufConfiguration &operator=(const GgufConfiguration &) = delete;
  const std::string &configuration_source() const;
  const std::string &platform() const;
  // Complete owned serving supplement, exactly as later consumed by LINK.
  const std::string &package_data_source() const;
  bool has_package_configuration() const;
  // Native inputs are borrowed synchronously; workspace use is externally
  // synchronized. Failure preserves *out. The result owns its native reference
  // and may outlive this configuration and caller's context/workspace
  // references.
  bool BuildModule(loomc_context_t *, loomc_workspace_t *,
                   std::unique_ptr<BuiltGgufModule> *out,
                   std::string *error) const;
  // Uses the package options copied by Configure. Numerical and companion LINK
  // consume the prepared sources, without deriving configuration again.
  bool WritePackage(loomc_context_t *, loomc_workspace_t *,
                    std::string *error) const;

private:
  friend class GgufModelBuilder;
  struct Impl;
  explicit GgufConfiguration(std::unique_ptr<Impl>);
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
  // Owned snapshot with independent issues; inspection needs no native build.
  const model::TextMetadata &text_metadata() const;
  // Source-only facts already derived during Open; no external tokenizer or
  // chat-template admission. Empty for host-managed numerical publication.
  const std::optional<PublicationTokenFacts> &publication_tokens() const;
  // Data-only numerical configuration, including GGUF source publication facts.
  // Links through :gguf_numerical_builder without package text/tokenizer code.
  // Failure preserves *out; all options/facts are copied and success clears
  // error.
  bool ConfigureNumerical(const GgufBuildOptions &,
                          std::unique_ptr<GgufConfiguration> *out,
                          std::string *error) const;
  // nullptr package prepares numerical data only, independently of tokenizer
  // or chat admission. This package-facing method links :gguf_model_builder.
  // Package preparation uses the existing native tokenizer parser and may scan
  // source GGUF vocabulary; it never loads Loom products. Empty resource/output
  // roots are allowed for inspection. All inputs are copied; failure preserves
  // *out, success clears error.
  bool Configure(const GgufBuildOptions &, const GgufPackageOptions *package,
                 std::unique_ptr<GgufConfiguration> *out,
                 std::string *error) const;

private:
  struct Impl;
  explicit GgufModelBuilder(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
} // namespace hrx::loomify_gguf
#endif
