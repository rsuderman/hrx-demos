// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include <map>
#include <set>
#include <string_view>
namespace hrx::dispatch {
loomc_status_t CheckQwenLayerConfiguration(const BuildContext &context) {
  using namespace operations;
  uint32_t layer = 32;
  bool attention = false;
  if (const auto *a =
          std::get_if<QwenGdnLayerAttrs>(&context.request.attributes))
    layer = a->layer_index;
  else if (const auto *a = std::get_if<QwenAttentionLayerAttrs>(
               &context.request.attributes)) {
    layer = a->layer_index;
    attention = true;
  }
  auto error = [] {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen layer graph encodings disagree with loaded selectors");
  };
  if (layer >= 32)
    return loomc_status_from_iree(error());
  auto encoding = [&](Port port) {
    const auto *type = context.request.Type(port);
    return type ? type->encoding() : Encoding::Dense;
  };
  const auto gate = encoding(Port::GateWeight), up = encoding(Port::UpWeight),
             down = encoding(Port::DownWeight);
  if (gate != up ||
      (gate != Encoding::Q4_K && gate != Encoding::Q5_K &&
       gate != Encoding::Q6_K) ||
      (down != Encoding::Q5_K && down != Encoding::Q6_K))
    return loomc_status_from_iree(error());
  const auto query = encoding(Port::QueryWeight),
             key = encoding(Port::KeyWeight);
  if (attention &&
      (query != key || (query != Encoding::Q5_K && query != Encoding::Q6_K)))
    return loomc_status_from_iree(error());
  return context.program.InspectNative(
      {},
      [&](const loom_module_t *module, const std::vector<loom_value_id_t> &) {
        std::map<std::string, uint32_t> masks;
        for (size_t i = 0; i < module->symbols.count; ++i) {
          const auto &symbol = module->symbols.entries[i];
          if (!symbol.defining_op || !loom_config_def_isa(symbol.defining_op))
            continue;
          const auto name =
              loom_string_table_get(&module->strings, symbol.name_id);
          const std::string key(name.data, name.size);
          if (key != "qwen.model.ffn_gate_q4_layers" &&
              key != "qwen.model.ffn_gate_q6_layers" &&
              key != "qwen.model.ffn_down_q6_layers" &&
              key != "qwen.model.attention_q6_layers")
            continue;
          const auto value = loom_config_def_value(symbol.defining_op);
          if (value.kind != LOOM_ATTR_I64 || value.i64 < INT32_MIN ||
              value.i64 > UINT32_MAX)
            return error();
          masks[key] = uint32_t(value.i64);
        }
        auto matches = [&](const char *name, bool expected) {
          auto value = masks.find(name);
          return value != masks.end() &&
                 bool((value->second >> layer) & 1u) == expected;
        };
        if (!matches("qwen.model.ffn_gate_q4_layers", gate == Encoding::Q4_K) ||
            !matches("qwen.model.ffn_gate_q6_layers", gate == Encoding::Q6_K) ||
            !matches("qwen.model.ffn_down_q6_layers", down == Encoding::Q6_K) ||
            (attention && !matches("qwen.model.attention_q6_layers",
                                   query == Encoding::Q6_K)))
          return error();
        return iree_ok_status();
      });
}
loomc_status_t CheckQwenLayerParameters(
    const BuildContext &context,
    const std::vector<std::pair<operations::Port, std::string>> &roles,
    loomify_gguf::Value *parameter_root) {
  auto error = [](const char *message) {
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
  };
  if (!parameter_root || roles.empty())
    return error("Qwen layer requires an explicit parameter role contract");
  std::set<operations::Port> ports;
  std::set<std::string> keys;
  std::vector<loomify_gguf::Value> roots;
  for (const auto &[port, key] : roles) {
    if (key.empty() || !ports.insert(port).second || !keys.insert(key).second)
      return error("Qwen layer parameter roles and keys must be unique");
    const NativeOperand *operand = nullptr;
    for (const auto &input : context.inputs)
      if (input.port == port) {
        if (operand)
          return error("Qwen layer parameter port is bound twice");
        operand = &input;
      }
    if (!operand || operand->parameter != key)
      return error(
          "Qwen layer parameter key does not match its canonical role");
    roots.push_back(operand->native);
  }
  for (const auto &input : context.inputs)
    if (!input.parameter.empty() && !ports.count(input.port))
      return error("Qwen layer has an unexpected named parameter operand");
  auto status = context.program.InspectNative(
      roots,
      [](const loom_module_t *module, const std::vector<loom_value_id_t> &ids) {
        for (auto id : ids)
          if (id != ids.front() ||
              !loom_type_equal(loom_module_value_type(module, id),
                               loom_type_buffer()))
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "Qwen layer weights must share one scoped parameter buffer");
        return iree_ok_status();
      });
  if (loomc_status_is_ok(status))
    *parameter_root = roots.front();
  return status;
}
} // namespace hrx::dispatch
