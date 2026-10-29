// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_SOURCE_BUILDER_H_
#define HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_SOURCE_BUILDER_H_
#include "loomc/module.h"
#include "tools/model/model_descriptor.h"
#include <string>
namespace hrx::loomify_gguf {
// Construct the bounded token-embedding stage for a structurally admitted Llama
// descriptor. Requires F32 token_embd.weight. The complete module contains the
// model.embedding command root, specialized by token index, with launch buffers
// (named immutable parameters, output). Other model stages are not generated.
// An out-of-vocabulary token writes a zero row. Weights remain external;
// callers supply the normal Loom named-parameter provider when materializing
// the command. Returns one independently owned module on success; clears output
// on failure.
bool BuildEmbeddingProgram(const hrx::model::ModelDescriptor &model,
                           loomc_context_t *context,
                           loomc_workspace_t *workspace,
                           loomc_module_t **out_module, std::string *error);
} // namespace hrx::loomify_gguf
#endif // HRX_SOLUTIONS_TOOLS_LOOMIFY_GGUF_SOURCE_BUILDER_H_
