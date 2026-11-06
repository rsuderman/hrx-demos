// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_MODEL_H_
#define HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_MODEL_H_
#include "tools/loomify-gguf/authoring/model-config/configuration.h"
#include "tools/loomify-gguf/authoring/model-config/linker.h"
namespace hrx::loomify_gguf {
struct ConfiguredLlamaModel {
  std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module{
      nullptr, loomc_module_release};
  LlamaLinkedLayout layout;
  std::string source_asset;
};
// The caller supplies the declared numerical provider closure and source
// root. This adds maintained FFN/layer/model/epoch sources and checks all public roots.
// Borrows input synchronously; success owns both the linked module and the exact
// admitted layout. Failure leaves out unchanged and returns native diagnostics.
loomc_status_t BuildConfiguredLlamaModel(
    loomc_context_t *context, loomc_workspace_t *workspace,
    const LinkedConfiguration &configuration,
    const ConfiguredLinkRequest &providers, ConfiguredLlamaModel *out,
    loomc_result_t **result);
} // namespace hrx::loomify_gguf
#endif
