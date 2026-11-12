// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/architectures.h"
namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
} // namespace
bool DeriveConfiguration(const model::ModelDescriptor &m,
                         const std::string &asset,
                         const ConfigurationSettings &s,
                         ConfigurationFacts *out, std::string *error) {
  if (!out || asset.empty())
    return Fail(error,
                "configuration requires output and source asset identity");
  const auto *architecture = SelectArchitecture(m.family);
  if (!architecture)
    return Fail(error, "unsupported model family");
  const auto limit = architecture->context_capacity_limit;
  if (!s.token_capacity || s.token_capacity > 512 ||
      s.context_capacity < s.token_capacity || s.context_capacity > limit)
    return Fail(error, "unsupported configuration token/context capacities");
  return architecture->emit_configuration(m, asset, s, out, error);
}
} // namespace hrx::loomify_gguf
