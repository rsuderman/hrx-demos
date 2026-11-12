// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#ifndef HRX_SOLUTIONS_TOOLS_MODEL_MODEL_DESCRIPTOR_INTERNAL_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_MODEL_DESCRIPTOR_INTERNAL_H_
#include "tools/model/model_descriptor.h"
#include "tools/model/tensor_names.h"
#include <cmath>
#include <stdexcept>
#include <string_view>
namespace hrx::model::internal {
// Logical structural bounds are separate from selected numerical support.
constexpr uint64_t kDimensionLimit = 1ULL << 30;
struct Loader {
  const GgufCatalog &catalog;
  std::string prefix;
  [[noreturn]] void Fail(const std::string &message) const {
    throw std::runtime_error(message);
  }
  const MetadataRecord *Field(const std::string &key) const {
    return FindMetadata(catalog, prefix + key);
  }
  uint64_t Integer(const MetadataScalar &value, const std::string &key,
                   bool positive = true) const {
    uint64_t n;
    if (auto v = std::get_if<uint64_t>(&value))
      n = *v;
    else if (auto v = std::get_if<int64_t>(&value); v && *v >= 0)
      n = static_cast<uint64_t>(*v);
    else
      Fail("expected integer metadata: " + prefix + key);
    if ((positive && n == 0) || n > kDimensionLimit)
      Fail("invalid/bounded geometry: " + prefix + key);
    return n;
  }
  uint64_t Int(const std::string &key, std::optional<uint64_t> fallback = {},
               bool positive = true) const {
    const auto *f = Field(key);
    if (!f) {
      if (fallback)
        return *fallback;
      Fail("missing required metadata: " + prefix + key);
    }
    if (f->source_type == 9)
      Fail("expected scalar integer metadata: " + prefix + key);
    return Integer(f->scalar, key, positive);
  }
  double Real(const std::string &key, std::optional<double> fallback = {},
              bool positive = true) const {
    const auto *f = Field(key);
    if (!f) {
      if (fallback)
        return *fallback;
      Fail("missing required metadata: " + prefix + key);
    }
    const auto *v = std::get_if<double>(&f->scalar);
    if (f->source_type == 9 || !v || !std::isfinite(*v) ||
        (positive && *v <= 0))
      Fail("expected finite floating metadata: " + prefix + key);
    return *v;
  }
  std::string String(const std::string &key,
                     const std::string &fallback) const {
    const auto *f = Field(key);
    if (!f)
      return fallback;
    const auto *s = std::get_if<std::string>(&f->scalar);
    if (f->source_type != 8 || !s)
      Fail("expected string metadata: " + prefix + key);
    return *s;
  }
  std::vector<uint64_t>
  Dimensions(const std::string &key, size_t count,
             const std::vector<uint64_t> &fallback = {}) const {
    const auto *f = Field(key);
    if (!f) {
      if (!fallback.empty())
        return fallback;
      Fail("missing required metadata: " + prefix + key);
    }
    if (f->source_type != 9)
      return std::vector<uint64_t>(count, Integer(f->scalar, key));
    if (f->elements_omitted || f->element_count != count ||
        f->elements.size() != count)
      Fail("wrong array length: " + prefix + key);
    std::vector<uint64_t> out;
    for (const auto &v : f->elements)
      out.push_back(Integer(v, key));
    return out;
  }
  void Bind(const ModelDescriptor &model, std::vector<TensorBinding> &out,
            const std::string &role, std::initializer_list<uint64_t> shape,
            std::optional<uint32_t> layer = std::nullopt,
            bool optional = false) const {
    std::string name;
    if (!ExpectedTensorName(model, role, layer, &name))
      Fail("unsupported tensor role/scope: " + role);
    const auto *t = FindTensor(catalog, name);
    if (!t) {
      if (optional)
        return;
      Fail("missing required tensor: " + name);
    }
    std::array<int64_t, 4> expected{1, 1, 1, 1};
    size_t i = 0;
    for (auto extent : shape) {
      if (extent > uint64_t(INT64_MAX))
        Fail("tensor geometry overflow: " + name);
      expected[i++] = extent;
    }
    if (t->normalized_extents != expected)
      Fail("tensor logical shape mismatch: " + name);
    out.push_back({role, *t});
  }
  void Unsupported(const std::string &key) const {
    if (Field(key))
      Fail("unsupported architecture feature: " + prefix + key);
  }
};

uint64_t Product(uint64_t a, uint64_t b);
void ReadDecoderDimensions(const Loader &, ModelDescriptor &);
void ReadRope(const Loader &, ModelDescriptor &, double base);
void BindGlobals(const Loader &, ModelDescriptor &);
void BindFeedForward(const Loader &, const ModelDescriptor &,
                     LayerDescriptor &);
void BindAttention(const Loader &, const ModelDescriptor &, LayerDescriptor &,
                   uint64_t query_width);
void BindHeadNorms(const Loader &, const ModelDescriptor &, LayerDescriptor &);
void LoadLlama(const Loader &, ModelDescriptor &);
void LoadQwen3(const Loader &, ModelDescriptor &);
void LoadQwen2(const Loader &, ModelDescriptor &);
void RejectHeadNorms(const Loader &, std::string_view architecture);
void LoadGemma3(const Loader &, ModelDescriptor &);
void LoadQwen35(const Loader &, ModelDescriptor &);
const char *LlamaTensorSuffix(std::string_view role);
const char *Qwen3TensorSuffix(std::string_view role);
const char *Qwen2TensorSuffix(std::string_view role);
const char *Gemma3TensorSuffix(std::string_view role);
const char *Qwen35TensorSuffix(std::string_view role);
const char *HeadNormTensorSuffix(std::string_view role);
} // namespace hrx::model::internal
#endif
