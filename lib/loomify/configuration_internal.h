// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_LIB_LOOMIFY_CONFIGURATION_INTERNAL_H_
#define HRX_SOLUTIONS_LIB_LOOMIFY_CONFIGURATION_INTERNAL_H_
#include "lib/loomify/configuration.h"
#include "tools/model/tensor_names.h"
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
inline const model::TensorRecord *
Role(const std::vector<model::TensorBinding> &items, const std::string &role) {
  const model::TensorRecord *found = nullptr;
  for (const auto &item : items)
    if (item.role == role) {
      if (found)
        return nullptr;
      found = &item.tensor;
    }
  return found;
}
inline bool SameTensor(const model::TensorRecord &a,
                       const model::TensorRecord &b) {
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
  return std::isfinite(value) && std::isfinite(float(value)) &&
         float(value) > 0;
}
struct CanonicalProjectionFacts {
  uint32_t storage_tag;
  uint64_t weight_bytes, input_width, output_width;
};
struct OrdinaryEmbeddingFacts {
  uint32_t storage_tag;
  uint64_t weight_bytes, hidden_size, vocabulary_size;
};
inline bool CheckEncodedExtent(const model::TensorRecord &tensor,
                               uint64_t input, uint64_t output,
                               std::string *error) {
  uint64_t bytes;
  if (!model::EncodedTensorByteSize(tensor.storage_type, input, output,
                                    &bytes) ||
      bytes > uint64_t(INT64_MAX) || bytes != tensor.byte_size)
    return Fail(error, "invalid encoded tensor extent: " + tensor.name);
  return true;
}
inline bool CheckCanonicalProjection(const model::TensorRecord &tensor,
                                     uint64_t input, uint64_t output,
                                     CanonicalProjectionFacts *facts,
                                     std::string *error) {
  uint64_t block = 0, maximum = 0;
  switch (tensor.storage_type) {
  case 12:
  case 13:
  case 23:
    block = 256;
    maximum = 32768;
    break;
  case 14:
    block = 256;
    maximum = 17408;
    break;
  case 2:
  case 3:
  case 6:
  case 8:
    block = 128;
    maximum = 32768;
    break;
  default:
    return Fail(error,
                "unsupported canonical projection encoding: " + tensor.name);
  }
  const uint64_t output_limit =
      tensor.storage_type == 8 || tensor.storage_type == 14 ? 262208 : 262144;
  if (!facts || input < block || input > maximum || input % block || !output ||
      output > output_limit)
    return Fail(error,
                "unsupported canonical projection geometry: " + tensor.name);
  if (!CheckEncodedExtent(tensor, input, output, error))
    return false;
  *facts = {tensor.storage_type, tensor.byte_size, input, output};
  return true;
}
inline bool CheckOrdinaryEmbedding(const model::TensorRecord &tensor,
                                   uint64_t hidden, uint64_t vocabulary,
                                   OrdinaryEmbeddingFacts *facts,
                                   std::string *error) {
  const uint64_t block = tensor.storage_type == 8 ? 32 : 256;
  const uint64_t vocabulary_limit =
      tensor.storage_type == 8 || tensor.storage_type == 14 ? 262208 : 262144;
  if (!facts ||
      (tensor.storage_type != 8 && tensor.storage_type != 12 &&
       tensor.storage_type != 13 && tensor.storage_type != 14) ||
      hidden < block || hidden > 32768 || hidden % block || !vocabulary ||
      vocabulary > vocabulary_limit)
    return Fail(error, "unsupported ordinary embedding geometry or encoding: " +
                           tensor.name);
  if (!CheckEncodedExtent(tensor, hidden, vocabulary, error))
    return false;
  *facts = {tensor.storage_type, tensor.byte_size, hidden, vocabulary};
  return true;
}
struct LayerRotaryFacts {
  uint64_t window, width;
  float log_step;
};
struct Producer {
  ConfigurationFacts facts;
  std::map<std::string, model::TensorRecord> identities;
  std::ostringstream text;
  const model::ModelDescriptor *model = nullptr;
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
             {2, "ggml.linear_q4_0_q8_1_x4."},
             {3, "ggml.linear_q4_1_q8_1_x4."},
             {6, "ggml.linear_q5_0_q8_1_x4."},
             {12, "ggml.linear_q4k_q8_1_x4."},
             {13, "ggml.linear_q5k_q8_1_x4."},
             {14, "ggml.linear_q6k_q8_1_x4."},
             {8, "ggml.linear_q8_0_q8_1_x4."},
             {23, "ggml.linear_iq4_xs_q8_1_x4."}}) {
      // The operation owns format support. This only sizes the reused product
      // from its actual tensor consumers; unused products receive a minimal
      // bound.
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
  void OrdinaryPreparation(uint64_t hidden, uint64_t layers, uint64_t kv,
                           uint64_t history, uint64_t tokens, uint64_t context,
                           uint64_t pool, uint64_t cache_base,
                           uint64_t map_origin,
                           uint64_t trailing_state_bytes = 0) {
    Index("loomify.serving.hidden_size", hidden);
    Index("loomify.serving.layer_count", layers);
    Index("loomify.serving.kv_size", kv);
    Index("loomify.serving.history_capacity", history);
    Index("loomify.serving.token_capacity", tokens);
    Index("loomify.serving.cache_capacity", context);
    Index("loomify.serving.pool_capacity", pool);
    Index("loomify.serving.cache_base", cache_base, "offset");
    Index("loomify.serving.map_origin", map_origin, "offset");
    Index("loomify.serving.trailing_state_bytes", trailing_state_bytes);
  }
  void Real(const std::string &name, float value) {
    text << "config.def @" << name << " = " << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << value
         << " : f32\n";
  }
  bool Tensor(const std::vector<model::TensorBinding> &bindings,
              const std::string &role, uint64_t in, uint64_t out,
              std::initializer_list<uint32_t> tags,
              std::optional<uint32_t> layer = std::nullopt,
              uint32_t expected_rank = 0) {
    std::string key;
    if (!model || !model::ExpectedTensorName(*model, role, layer, &key))
      return Fail(error, "unsupported tensor role/scope: " + role);
    const auto *t = Role(bindings, role);
    if (!t)
      return Fail(error, "requires exactly one tensor role: " + role);
    if (t->name != key ||
        t->rank != (expected_rank ? expected_rank : (out == 1 ? 1u : 2u)) ||
        t->normalized_extents !=
            std::array<int64_t, 4>{int64_t(in), int64_t(out), 1, 1} ||
        (tags.size() &&
         std::find(tags.begin(), tags.end(), t->storage_type) == tags.end()))
      return Fail(error, "unsupported tensor name, shape or encoding: " + key);
    uint64_t bytes;
    if (!model::EncodedTensorByteSize(t->storage_type, in, out, &bytes) ||
        bytes > uint64_t(INT64_MAX))
      return Fail(error,
                  "unsupported or overflowing encoded tensor extent: " + key);
    uint64_t end;
    if (bytes != t->byte_size || !Sum(t->absolute_offset, bytes, &end))
      return Fail(error, "invalid encoded tensor byte range: " + key);
    auto [existing, inserted] = identities.emplace(key, *t);
    if (!inserted && !SameTensor(existing->second, *t))
      return Fail(error, "inconsistent aliases for tensor: " + key);
    facts.parameters.push_back({role, layer, asset, *t});
    return true;
  }
  bool CanonicalProjection(const std::vector<model::TensorBinding> &bindings,
                           const std::string &role, uint64_t input,
                           uint64_t output, std::optional<uint32_t> layer,
                           CanonicalProjectionFacts *result) {
    const auto *tensor = Role(bindings, role);
    CanonicalProjectionFacts checked;
    if (!result || !tensor)
      return Fail(error, "requires exactly one projection role: " + role);
    if (!CheckCanonicalProjection(*tensor, input, output, &checked, error) ||
        !Tensor(bindings, role, input, output, {}, layer, 2))
      return false;
    Projection(*tensor, input, output);
    *result = checked;
    return true;
  }
  bool OrdinaryEmbedding(const std::vector<model::TensorBinding> &bindings,
                         const std::string &role, uint64_t hidden,
                         uint64_t vocabulary, OrdinaryEmbeddingFacts *result) {
    const auto *tensor = Role(bindings, role);
    OrdinaryEmbeddingFacts checked;
    if (!result || !tensor)
      return Fail(error, "requires exactly one embedding role: " + role);
    if (!CheckOrdinaryEmbedding(*tensor, hidden, vocabulary, &checked, error) ||
        !Tensor(bindings, role, hidden, vocabulary, {}, std::nullopt, 2))
      return false;
    *result = checked;
    return true;
  }
  // The family admits nonempty layer facts; this writes only typed constants.
  void LayerRotaryLookup(const std::vector<LayerRotaryFacts> &rows) {
    text
        << "func.def public pure inline @loomify.config.layer_rotary(%layer: index) -> (index, index, f32) where [range(%layer, 0, "
        << rows.size() - 1 << ")] {\n";
    const auto emit = [&](const std::string &prefix,
                          const LayerRotaryFacts &row, const char *indent) {
      text << indent << "%" << prefix << "w = index.constant " << row.window
           << " : index\n"
           << indent << "%" << prefix << "r = index.constant " << row.width
           << " : index\n"
           << indent << "%" << prefix << "s = scalar.constant "
           << std::scientific
           << std::setprecision(std::numeric_limits<float>::max_digits10)
           << row.log_step << " : f32\n";
    };
    emit("v0_", rows.back(), "  ");
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
      const auto prior = "v" + std::to_string(i) + "_";
      const auto next = "v" + std::to_string(i + 1) + "_";
      const auto current = "c" + std::to_string(i) + "_";
      text << "  %l" << i << " = index.constant " << i << " : index\n"
           << "  %is" << i << " = index.cmp eq, %layer, %l" << i
           << " : index\n  %" << next << "w, %" << next << "r, %" << next
           << "s = scf.if %is" << i << " -> (index, index, f32) {\n";
      emit(current, rows[i], "    ");
      text << "    scf.yield %" << current << "w, %" << current << "r, %"
           << current << "s : index, index, f32\n  } else {\n"
           << "    scf.yield %" << prior << "w, %" << prior << "r, %" << prior
           << "s : index, index, f32\n  }\n";
    }
    const auto last = "v" + std::to_string(rows.size() - 1) + "_";
    text << "  func.return %" << last << "w, %" << last << "r, %" << last
         << "s : index, index, f32\n}\n";
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
} // namespace hrx::loomify_gguf::internal
#endif
