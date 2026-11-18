// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#ifndef HRX_SOLUTIONS_TOOLS_MODEL_MODEL_DESCRIPTOR_H_
#define HRX_SOLUTIONS_TOOLS_MODEL_MODEL_DESCRIPTOR_H_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "tools/model/gguf_catalog.h"

namespace hrx::model {

enum class LayerKind { kAttention, kGatedDeltaNet, kShortConvolution };

struct TensorBinding {
  std::string role;
  TensorRecord tensor;
};

struct LayerDescriptor {
  uint32_t index = 0;
  LayerKind kind = LayerKind::kAttention;
  std::optional<uint32_t> attention_ordinal;
  std::optional<uint32_t> recurrent_ordinal;
  std::optional<uint32_t> convolution_ordinal;

  uint64_t feed_forward_width = 0;
  uint64_t query_heads = 0;
  uint64_t kv_heads = 0;
  // Effective causal window (zero means full history) and per-layer RoPE base.
  uint64_t attention_window = 0;
  double rope_base = 10000;
  // Zero inherits the model rotary width; otherwise the effective layer width.
  uint64_t rope_width = 0;
  std::vector<TensorBinding> tensors;
};

struct RopeDescriptor {
  uint64_t width = 0;
  uint64_t original_context = 0;
  double base = 10000;
  double scaling_factor = 1;
  std::string scaling_kind = "linear";
  std::array<uint64_t, 4> sections{};
};

struct SsmDescriptor {
  uint64_t convolution_kernel = 0;
  uint64_t inner_width = 0;
  uint64_t state_width = 0;
  uint64_t key_heads = 0;
  uint64_t value_heads = 0;
  uint64_t convolution_history_elements = 0;
  uint64_t recurrent_state_elements = 0;
};

struct ShortConvolutionDescriptor {
  uint64_t width = 0;
  uint64_t kernel = 0;
  uint64_t history_rows = 0;
};

struct ModelDescriptor {
  std::string family;
  std::string name;

  uint64_t embedding_width = 0;
  uint64_t vocabulary_size = 0;
  uint64_t training_context = 0;
  // Physical block_count before removing a trailing next-token predictor.
  // Zero denotes an unspecified count in programmatically built descriptors.
  uint64_t stored_layer_count = 0;
  uint64_t key_head_width = 0;
  uint64_t value_head_width = 0;
  double rms_epsilon = 0;

  RopeDescriptor rope;
  std::optional<SsmDescriptor> ssm;
  std::optional<ShortConvolutionDescriptor> short_convolution;
  bool output_tied_to_embedding = false;
  std::vector<TensorBinding> globals;
  std::vector<LayerDescriptor> layers;
};

// Logical structural admission only: no dtype/backend/pipeline qualification.
// Owns bindings; catalog and output may be destroyed independently. Failure
// leaves output unchanged. Unsupported architecture features are diagnosed.
bool LoadModelDescriptor(const GgufCatalog &catalog, ModelDescriptor *model,
                         std::string *error);

} // namespace hrx::model

#endif // HRX_SOLUTIONS_TOOLS_MODEL_MODEL_DESCRIPTOR_H_
