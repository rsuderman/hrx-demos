// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_ORDINARY_DECODER_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_ARCHITECTURES_ORDINARY_DECODER_H_
#include "lib/loomify/configuration_internal.h"
#include <string_view>

namespace hrx::loomify_gguf::internal {
using ConfigRows = std::vector<std::vector<uint64_t>>;
bool BindOrdinaryDecoderMatrices(Producer &p,
                                 const model::LayerDescriptor &layer,
                                 uint64_t hidden, uint64_t query, uint64_t kv,
                                 ConfigRows &attention, ConfigRows &ffn);
void EmitOrdinaryDecoderFacts(Producer &p, const model::ModelDescriptor &model,
                              const ConfigurationSettings &settings,
                              std::string_view family_prefix, uint64_t query,
                              uint64_t kv, const ConfigRows &attention,
                              const ConfigRows &ffn);
} // namespace hrx::loomify_gguf::internal
#endif
