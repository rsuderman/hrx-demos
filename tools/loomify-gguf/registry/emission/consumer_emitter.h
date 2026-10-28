// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_EMISSION_CONSUMER_EMITTER_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_EMISSION_CONSUMER_EMITTER_H_

#include "loomc/link.h"
#include "loomc/module.h"

namespace hrx::loomify_gguf {

// Generates and parses a device kernel applying ggml.qk.scale_min and storing
// both results. This is textual Loom IR emission, not command-program
// authoring. On OK, inspect out_result for semantic success. Returned
// references are owned by the caller; input context/workspace are borrowed for
// this call.
loomc_status_t EmitScaleMinProbe(loomc_context_t *context,
                                 loomc_workspace_t *workspace,
                                 loomc_module_t **out_module,
                                 loomc_result_t **out_result);

// Links one invocation-local input against an immutable library-only index.
// All inputs must share context and remain alive for this synchronous call.
// No input/index is mutated. On OK, inspect out_result for semantic success;
// the caller owns the returned module/result references. API failures leave
// both outputs null. An OK semantic failure has a result but no module.
loomc_status_t
LinkConsumer(loomc_context_t *context, loomc_workspace_t *workspace,
             loomc_link_index_t *libraries, const loomc_module_t *input,
             loomc_string_view_t root, loomc_module_t **out_module,
             loomc_result_t **out_result);

} // namespace hrx::loomify_gguf

#endif // HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_REGISTRY_EMISSION_CONSUMER_EMITTER_H_
