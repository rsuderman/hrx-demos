// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_LOADING_MODULE_LOADER_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_LOADING_MODULE_LOADER_H_

#include <string>

#include "loomc/module.h"

namespace hrx::loomify_gguf {

// Reads bytecode only; text with a .loombc suffix is not accepted. The caller
// supplies the context shared with other providers and owns the returned module
// and result references. Check both API status and result_succeeded. On a
// non-OK status both outputs are null; an OK status may have a diagnosed failed
// result. Returned modules/results retain their native ownership independently
// of the temporary input source. Output pointers are required.
loomc_status_t LoadBytecode(loomc_context_t *context,
                            loomc_workspace_t *workspace,
                            loomc_string_view_t path,
                            loomc_module_t **out_module,
                            loomc_result_t **out_result);

// Produces owned canonical text without locations. The string remains valid
// after module release. Clears output before attempting serialization.
loomc_status_t DumpText(const loomc_module_t *module, std::string *out_text);

} // namespace hrx::loomify_gguf

#endif // HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_LOADING_MODULE_LOADER_H_
