// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/serving/qwen/package.h"
#include <filesystem>
namespace hrx::loomify_gguf {
namespace {
bool Fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}
} // namespace
bool RenderQwenCompanions(const ModelPackageOptions &o,
                          const dispatch::QwenNativeGeometry &g,
                          detail::CompanionSources *out, std::string *error) {
  if (!out || !o.prefill_capacity || o.prefill_capacity > 512 ||
      o.context_capacity < o.prefill_capacity || o.context_capacity > 262144 ||
      !g.residual_bytes || !g.gdn_total_bytes || !g.gdn_slot_origin ||
      !g.private_row_stride || !g.map_origin || !g.kv_plane_count ||
      !g.kv_plane_token_bytes)
    return Fail(error, "invalid admitted Qwen package settings/geometry");
  const uint64_t pool = ((o.context_capacity + 63) / 64) * 64;
  if (g.kv_plane_stride != pool * g.kv_plane_token_bytes ||
      g.kv_pool_bytes != g.kv_plane_count * g.kv_plane_stride)
    return Fail(error,
                "Qwen package pool differs from admitted native geometry");
  const auto root =
      std::filesystem::path(o.source_root) / "tools/loomify-gguf/serving/qwen";
  detail::CompanionSources sources;
  if (!detail::ReadText((root / "prepare.loom.in").string(), &sources.prepare,
                        error) ||
      !detail::ReadText((root / "control.loom").string(), &sources.control,
                        error) ||
      !detail::ReadText((root / "weights.loom").string(), &sources.weights,
                        error))
    return false;
  auto &prepare = sources.prepare;
  detail::ReplaceToken(prepare, "EPOCH_ROOT_HEX", detail::Hex(kQwenEpochRoot));
  detail::ReplaceToken(prepare, "TOKENS", std::to_string(o.prefill_capacity));
  detail::ReplaceToken(prepare, "CONTEXT", std::to_string(o.context_capacity));
  detail::ReplaceToken(prepare, "POOL", std::to_string(pool));
  detail::ReplaceToken(
      prepare, "KV_TOKEN_BYTES",
      std::to_string(g.kv_plane_count * g.kv_plane_token_bytes));
  detail::ReplaceToken(prepare, "RESIDUAL_BYTES",
                       std::to_string(g.residual_bytes));
  detail::ReplaceToken(prepare, "GDN_BYTES", std::to_string(g.gdn_total_bytes));
  detail::ReplaceToken(prepare, "KV_PLANES", std::to_string(g.kv_plane_count));
  detail::ReplaceToken(prepare, "KV_PLANE_TOKEN_BYTES",
                       std::to_string(g.kv_plane_token_bytes));
  detail::ReplaceToken(prepare, "GDN_ORIGIN",
                       std::to_string(g.gdn_slot_origin));
  detail::ReplaceToken(prepare, "MAP_ORIGIN", std::to_string(g.map_origin));
  detail::ReplaceToken(prepare, "PRIVATE_STRIDE",
                       std::to_string(g.private_row_stride));
  if (prepare.find("{{") != std::string::npos)
    return Fail(error, "unresolved Qwen prepare template field");
  *out = std::move(sources);
  return true;
}
bool WriteQwenPackage(const model::ModelDescriptor &model,
                      const ModelPackageOptions &o,
                      const dispatch::QwenNativeGeometry &geometry,
                      const loomc_module_t *module, std::string *error) {
  if (model.family != "qwen35" || model.layers.empty())
    return Fail(error, "Qwen package requires an admitted Qwen descriptor");
  detail::CompanionSources sources;
  if (!RenderQwenCompanions(o, geometry, &sources, error))
    return false;
  return detail::WriteServingPackage(o, "loomify-qwen-08b", sources, module,
                                     error);
}
} // namespace hrx::loomify_gguf
