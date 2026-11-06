// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_LINKER_H_
#define HRX_SOLUTIONS_LOOMIFY_MODEL_CONFIG_LINKER_H_

#include "loom/ir/types.h"
#include "loomc/link.h"
#include <string>
#include <vector>

namespace hrx::loomify_gguf {
struct ConfiguredProvider {
  std::string path;
  loomc_link_provider_role_t role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
};
enum class ConfiguredCallableKind { Kernel, CommandProgram };
// Native scalar/buffer types only; groups are workload/arguments for kernels,
// specialization/bindings for command programs. No foreign shaped-type IDs.
struct ConfiguredRoot {
  std::string symbol;
  ConfiguredCallableKind kind = ConfiguredCallableKind::CommandProgram;
  std::vector<loom_type_t> specializations;
  std::vector<loom_type_t> arguments;
};
struct ConfiguredLinkRequest {
  std::string source_root;
  std::vector<ConfiguredProvider> providers;
  std::vector<ConfiguredRoot> roots;
  std::vector<std::string> required_configuration;
  std::vector<std::string> required_definitions;
};
// Borrows all inputs synchronously and never changes their IR. Requires
// exclusive access during native inspection, which may populate SDK caches.
// Returns the actual
// strict LINK module, independently owned by the caller. Both output slots are
// cleared first; diagnosed native failures return OK plus a failed result and
// null module. Scalar/buffer signatures do not constrain runtime scalar values.
loomc_status_t LinkConfiguredModules(loomc_context_t *context,
                                     loomc_workspace_t *workspace,
                                     const loomc_module_t *configuration,
                                     const ConfiguredLinkRequest &request,
                                     loomc_module_t **out_module,
                                     loomc_result_t **out_result);
} // namespace hrx::loomify_gguf
#endif
