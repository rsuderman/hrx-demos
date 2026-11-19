// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "tools/model/model_descriptor_internal.h"
#include "tools/model/assembly_registry.h"

namespace hrx::model::internal {
namespace {
const char *Qwen3MoeTensorSuffix(std::string_view role, bool is_layer) {
  if (is_layer) {
    if (role == "expert_router")
      return "ffn_gate_inp.weight";
    if (role == "expert_gate")
      return "ffn_gate_exps.weight";
    if (role == "expert_up")
      return "ffn_up_exps.weight";
    if (role == "expert_down")
      return "ffn_down_exps.weight";
  }
  return Qwen3TensorSuffix(role, is_layer);
}
void LoadQwen3Moe(const Loader &l, ModelDescriptor &m) {
  l.Unsupported("attention.sliding_window");
  ReadDecoderGeometry(l, m);
  if (m.rope.width != m.key_head_width)
    l.Fail("invalid RoPE width");
  ReadRope(l, m, l.Real("rope.freq_base", 10000));
  RoutedExpertsDescriptor experts;
  experts.expert_count = l.Int("expert_count");
  experts.experts_per_token = l.Int("expert_used_count");
  experts.feed_forward_width = l.Int("expert_feed_forward_length");
  if (experts.experts_per_token > experts.expert_count)
    l.Fail("expert_used_count exceeds expert_count");
  for (const char *key :
       {"expert_shared_feed_forward_length", "expert_shared_count",
        "expert_group_count", "expert_group_used_count",
        "leading_dense_block_count"})
    if (l.Int(key, 0, false))
      l.Fail(std::string("unsupported Qwen3MoE feature: ") + key);
  if (l.Int("expert_gating_func", 1) != 1 ||
      l.Real("expert_weights_scale", 1) != 1)
    l.Fail("Qwen3MoE requires normalized softmax routing");
  if (const auto *field = l.Field("expert_weights_norm")) {
    const auto *value = std::get_if<bool>(&field->scalar);
    if (field->source_type != 7 || !value || !*value)
      l.Fail("Qwen3MoE requires normalized selected expert weights");
  }
  for (const auto &tensor : l.catalog.tensors)
    if (tensor.name.find("_shexp") != std::string::npos)
      l.Fail("unsupported shared expert tensor: " + tensor.name);
  m.routed_experts = experts;
  BindGlobals(l, m);
  BindQwen3AttentionLayers(l, m);
  for (auto &layer : m.layers) {
    l.Bind(m, layer.tensors, "expert_router",
           {m.embedding_width, experts.expert_count}, layer.index);
    l.BindExpert(m, layer.tensors, "expert_gate", m.embedding_width,
                 experts.feed_forward_width, experts.expert_count, layer.index);
    l.BindExpert(m, layer.tensors, "expert_up", m.embedding_width,
                 experts.feed_forward_width, experts.expert_count, layer.index);
    l.BindExpert(m, layer.tensors, "expert_down", experts.feed_forward_width,
                 m.embedding_width, experts.expert_count, layer.index);
  }
}
} // namespace
const ModelArchitecture kQwen3MoeArchitecture{"qwen3moe", LoadQwen3Moe,
                                              Qwen3MoeTensorSuffix};
} // namespace hrx::model::internal
