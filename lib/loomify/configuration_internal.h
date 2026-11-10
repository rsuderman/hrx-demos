// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_CONFIGURATION_INTERNAL_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_CONFIGURATION_INTERNAL_H_
#include "lib/loomify/configuration.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
namespace hrx::loomify_gguf::internal {
inline bool Fail(std::string *error, const std::string &message) {
  if (error)
    *error = message;
  return false;
}
inline const model::TensorRecord *Role(const std::vector<model::TensorBinding> &items,
                                const std::string &role) {
  const model::TensorRecord *found = nullptr;
  for (const auto &item : items)
    if (item.role == role) {
      if (found)
        return nullptr;
      found = &item.tensor;
    }
  return found;
}
inline bool SameTensor(const model::TensorRecord &a, const model::TensorRecord &b) {
  return a.name == b.name && a.rank == b.rank &&
         a.normalized_extents == b.normalized_extents &&
         a.storage_type == b.storage_type &&
         a.absolute_offset == b.absolute_offset && a.byte_size == b.byte_size;
}
// Keep source checkpoint ranges separate from native origins. Restrict both
// metadata and derived geometry to the native signed index/offset domain.
inline bool Product(uint64_t a, uint64_t b, uint64_t *out) {
  if (b && a > uint64_t(INT64_MAX) / b)
    return false;
  *out = a * b;
  return true;
}
inline bool Sum(uint64_t a, uint64_t b, uint64_t *out) {
  if (a > uint64_t(INT64_MAX) || b > uint64_t(INT64_MAX) - a)
    return false;
  *out = a + b;
  return true;
}
inline bool PositiveF32(double value) {
  return std::isfinite(value) && std::isfinite(float(value)) && float(value) > 0;
}
struct Producer {
  ConfigurationFacts facts;
  std::map<std::string, model::TensorRecord> identities;
  std::ostringstream text;
  std::string asset;
  std::string *error;
  std::map<uint32_t, uint64_t> projection_widths;
  uint64_t packed_input_width = 0;
  void Projection(const model::TensorRecord &tensor, uint64_t input,
                  uint64_t output) {
    projection_widths[tensor.storage_type] =
        std::max(projection_widths[tensor.storage_type], output);
    packed_input_width = std::max(packed_input_width, input);
  }
  void ProjectionCapacities() {
    for (const auto &entry : std::vector<std::pair<uint32_t, const char *>>{
             {12, "ggml.linear_q4k_q8_1_x4."},
             {13, "ggml.linear_q5k_q8_1_x4."},
             {14, "ggml.linear_q6k_q8_1_x4."},
             {8, "ggml.linear_q8_0_q8_1_x4."}}) {
      // The operation owns format support. This only sizes the reused product
      // from its actual tensor consumers; unused products receive a minimal bound.
      Index(std::string(entry.second) + "token_capacity", 512);
      Index(std::string(entry.second) + "output_capacity",
            std::max<uint64_t>(1, projection_widths[entry.first]));
    }
    Index("ggml.quantize_q8_1_x4.group_capacity",
          std::max<uint64_t>(1, 512 * packed_input_width / 128));
  }
  void Index(const std::string &name, uint64_t value,
             const char *type = "index") {
    text << "config.def @" << name << " = " << value << " : " << type << '\n';
  }
  void Real(const std::string &name, float value) {
    text << "config.def @" << name << " = " << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << value
         << " : f32\n";
  }
  bool Tensor(const std::vector<model::TensorBinding> &bindings,
              const std::string &role, const std::string &key, uint64_t in,
              uint64_t out, std::initializer_list<uint32_t> tags,
              std::optional<uint32_t> layer = std::nullopt) {
    const auto *t = Role(bindings, role);
    if (!t)
      return Fail(error, "requires exactly one tensor role: " + role);
    if (t->name != key || t->rank != (out == 1 ? 1u : 2u) ||
        t->normalized_extents !=
            std::array<int64_t, 4>{int64_t(in), int64_t(out), 1, 1} ||
        (tags.size() && std::find(tags.begin(), tags.end(), t->storage_type) == tags.end()))
      return Fail(error, "unsupported tensor name, shape or encoding: " + key);
    uint64_t bytes;
    if (!model::EncodedTensorByteSize(t->storage_type, in, out, &bytes) || bytes > uint64_t(INT64_MAX))
      return Fail(error, "unsupported or overflowing encoded tensor extent: " + key);
    uint64_t end;
    if (bytes != t->byte_size || !Sum(t->absolute_offset, bytes, &end))
      return Fail(error, "invalid encoded tensor byte range: " + key);
    auto [existing, inserted] = identities.emplace(key, *t);
    if (!inserted && !SameTensor(existing->second, *t))
      return Fail(error, "inconsistent aliases for tensor: " + key);
    facts.parameters.push_back({role, layer, asset, *t});
    return true;
  }
  // Scalar cases contain facts only. The layer range is a caller precondition;
  // the final arm represents the last valid layer rather than a new fallback.
  void Lookup(const std::string &name,
              const std::vector<std::vector<uint64_t>> &rows) {
    const auto width = rows.front().size();
    std::string types;
    for (size_t j = 0; j < width; ++j)
      types += (j ? ", index" : "index");
    text << "func.def public pure inline @" << name << "(%layer: index) -> ("
         << types << ") where [range(%layer, 0, " << rows.size() - 1
         << ")] {\n";
    for (size_t j = 0; j < width; ++j)
      text << "  %v0_" << j << " = index.constant " << rows.back()[j]
           << " : index\n";
    size_t version = 0;
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
      text << "  %l" << i << " = index.constant " << i << " : index\n"
           << "  %is" << i << " = index.cmp eq, %layer, %l" << i
           << " : index\n  ";
      for (size_t j = 0; j < width; ++j)
        text << (j ? ", " : "") << "%v" << version + 1 << "_" << j;
      text << " = scf.if %is" << i << " -> (" << types << ") {\n";
      for (size_t j = 0; j < width; ++j)
        text << "    %c" << i << "_" << j << " = index.constant " << rows[i][j]
             << " : index\n";
      text << "    scf.yield ";
      for (size_t j = 0; j < width; ++j)
        text << (j ? ", " : "") << "%c" << i << "_" << j;
      text << " : " << types << "\n  } else {\n    scf.yield ";
      for (size_t j = 0; j < width; ++j)
        text << (j ? ", " : "") << "%v" << version << "_" << j;
      text << " : " << types << "\n  }\n";
      ++version;
    }
    text << "  func.return ";
    for (size_t j = 0; j < width; ++j)
      text << (j ? ", " : "") << "%v" << version << "_" << j;
    text << " : " << types << "\n}\n";
  }
};
bool EmitLlama(const model::ModelDescriptor&, const std::string&,
               const ConfigurationSettings&, ConfigurationFacts*, std::string*);
bool EmitQwen(const model::ModelDescriptor&, const std::string&,
              const ConfigurationSettings&, ConfigurationFacts*, std::string*);
} // namespace hrx::loomify_gguf::internal
#endif
