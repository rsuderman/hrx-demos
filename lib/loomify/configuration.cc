// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/configuration_internal.h"
namespace hrx::loomify_gguf {
bool DeriveConfiguration(const model::ModelDescriptor &m,
                         const std::string &asset,
                         const ConfigurationSettings &s,
                         ConfigurationFacts *out, std::string *error) {
  if (!out || asset.empty())
    return internal::Fail(error,
                "configuration requires output and source asset identity");
  const auto limit = m.family == "llama" ? 131072u : 262144u;
  if (!s.token_capacity || s.token_capacity > 512 ||
      s.context_capacity < s.token_capacity || s.context_capacity > limit)
    return internal::Fail(error, "unsupported configuration token/context capacities");
  if (m.family == "llama")
    return internal::EmitLlama(m, asset, s, out, error);
  if (m.family == "gemma3")
    return internal::EmitGemma(m, asset, s, out, error);
  if (m.family == "qwen3")
    return internal::EmitQwen3(m, asset, s, out, error);
  if (m.family == "qwen35")
    return internal::EmitQwen(m, asset, s, out, error);
  return internal::Fail(error, "unsupported model family");
}
} // namespace hrx::loomify_gguf
