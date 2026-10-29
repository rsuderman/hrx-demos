// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/model-full/llama_step.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "loom/ops/template/ops.h"
#include "tools/loomify-gguf/authoring/commands/command_values.h"

namespace hrx::loomify_gguf {
namespace {
#define RETURN_IF_ERROR(expr)                                                  \
  do {                                                                         \
    loomc_status_t status = (expr);                                            \
    if (!loomc_status_is_ok(status))                                           \
      return status;                                                           \
  } while (false)

loomc_status_t Invalid(const char *message) {
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT, message);
}
const model::TensorRecord *
Role(const std::vector<model::TensorBinding> &bindings, std::string_view role) {
  const model::TensorRecord *result = nullptr;
  for (const auto &binding : bindings) {
    if (binding.role != role)
      continue;
    if (result)
      return nullptr;
    result = &binding.tensor;
  }
  return result;
}
bool Tensor(const model::TensorRecord *tensor, uint32_t storage, uint64_t width,
            uint64_t rows = 1) {
  if (!tensor || tensor->name.empty() || tensor->storage_type != storage ||
      tensor->normalized_extents[0] != static_cast<int64_t>(width) ||
      tensor->normalized_extents[1] != static_cast<int64_t>(rows) ||
      tensor->normalized_extents[2] != 1 || tensor->normalized_extents[3] != 1)
    return false;
  uint64_t values = width * rows;
  uint64_t bytes =
      storage == 0 ? values * 4 : values / 256 * (storage == 12 ? 144 : 210);
  return tensor->byte_size == bytes;
}
loomc_status_t Validate(const model::ModelDescriptor &model) {
  if (model.family != "llama" || model.embedding_width != 3072 ||
      model.vocabulary_size != 128256 || model.layers.size() != 28 ||
      model.key_head_width != 128 || model.value_head_width != 128 ||
      !model.output_tied_to_embedding || model.ssm || model.rope.width != 128 ||
      model.rope.base != 500000 || model.rope.scaling_factor != 1 ||
      std::abs(model.rms_epsilon - 0.00001) > 1e-10)
    return Invalid("unsupported Llama step profile: numerical providers "
                   "require the qualified 3.2 3B geometry");
  const auto *embedding = Role(model.globals, "token_embedding");
  const auto *output = Role(model.globals, "output");
  if (!Tensor(embedding, 14, 3072, 128256) ||
      !Tensor(output, 14, 3072, 128256) || embedding->name != output->name ||
      !Tensor(Role(model.globals, "output_norm"), 0, 3072) ||
      !Tensor(Role(model.globals, "rope_frequencies"), 0, 64))
    return Invalid("unsupported Llama global tensor role/storage/extent");
  for (size_t i = 0; i < model.layers.size(); ++i) {
    const auto &layer = model.layers[i];
    const auto *value = Role(layer.tensors, "value");
    const auto *down = Role(layer.tensors, "ffn_down");
    if (layer.index != i || layer.kind != model::LayerKind::kAttention ||
        layer.feed_forward_width != 8192 || layer.query_heads != 24 ||
        layer.kv_heads != 8 ||
        !Tensor(Role(layer.tensors, "input_norm"), 0, 3072) ||
        !Tensor(Role(layer.tensors, "post_norm"), 0, 3072) ||
        !Tensor(Role(layer.tensors, "query"), 12, 3072, 3072) ||
        !Tensor(Role(layer.tensors, "key"), 12, 3072, 1024) ||
        !(Tensor(value, 12, 3072, 1024) || Tensor(value, 14, 3072, 1024)) ||
        !Tensor(Role(layer.tensors, "attention_output"), 12, 3072, 3072) ||
        !Tensor(Role(layer.tensors, "ffn_gate"), 12, 3072, 8192) ||
        !Tensor(Role(layer.tensors, "ffn_up"), 12, 3072, 8192) ||
        !(Tensor(down, 12, 8192, 3072) || Tensor(down, 14, 8192, 3072)))
      return Invalid("unsupported Llama layer role/storage/extent");
  }
  return loomc_ok_status();
}

// Native template families are computation/command composition contracts, not
// kernel interfaces. Keep their native operation through the checked scope
// seam.
loomc_status_t Apply(CommandProgramBuilder &program, std::string_view name,
                     const std::vector<Value> &arguments) {
  std::vector<Value> no_results;
  return program.BuildNative(
      arguments,
      [&](loom_module_t *module, loom_builder_t *builder,
          const std::vector<loom_value_id_t> &operands,
          std::vector<loom_value_id_t> *) {
        loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
        for (size_t i = 0; i < module->symbols.count; ++i) {
          const auto &entry = module->symbols.entries[i];
          auto actual = loom_string_table_get(&module->strings, entry.name_id);
          if (std::string_view(actual.data, actual.size) == name) {
            symbol = static_cast<loom_symbol_id_t>(i);
            break;
          }
        }
        if (symbol == LOOM_SYMBOL_ID_INVALID)
          return iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "missing numerical template family: %.*s",
                                  int(name.size()), name.data());
        loom_op_t *op = nullptr;
        return loom_template_apply_build(
            builder, 0, loom_symbol_ref_t{0, symbol}, operands.data(),
            operands.size(), 0, 0, nullptr, 0, nullptr, 0,
            LOOM_LOCATION_UNKNOWN, &op);
      },
      &no_results);
}

class Recipe {
public:
  Recipe(ModuleBuilder &module, CommandProgramBuilder &program)
      : module(module), program(program), values(program) {}
  loomc_status_t Index(uint64_t number, Value *out) {
    return values.Constant(static_cast<int64_t>(number), LOOM_SCALAR_TYPE_INDEX,
                           out);
  }
  loomc_status_t Scratch(uint64_t bytes, Value *out) {
    Value size;
    RETURN_IF_ERROR(values.Constant(bytes, LOOM_SCALAR_TYPE_OFFSET, &size));
    return values.Allocate(size, 256, out);
  }
  loomc_status_t Weight(const model::TensorRecord &tensor, Value *out) {
    Value bytes;
    RETURN_IF_ERROR(Index(tensor.byte_size, &bytes));
    return Weight(tensor, bytes, out);
  }
  loomc_status_t Weight(const model::TensorRecord &tensor, Value bytes,
                        Value *out) {
    return values.Parameter(program.Binding(0), tensor.name, {},
                            LOOM_SCALAR_TYPE_I8, {bytes}, out);
  }
  loomc_status_t Launch(std::string_view name,
                        const std::vector<Value> &workloads,
                        const std::vector<Value> &args) {
    KernelRef kernel;
    RETURN_IF_ERROR(module.Kernel(name, &kernel));
    return program.Invoke(kernel, workloads, args);
  }
  loomc_status_t Pack(Value width, Value input, Value output) {
    return Launch(
        "ggml_quantize_q8_1_x4_qk_compat_f32", {one, width},
        {one, width, zero, count32, origin, program.Binding(2), input, output});
  }
  loomc_status_t Project(const model::TensorRecord &tensor, Value input_width,
                         Value output_width, Value input, Value output) {
    Value weight;
    RETURN_IF_ERROR(Weight(tensor, &weight));
    return Launch(tensor.storage_type == 14 ? "ggml_linear_q6k_q8_1_x4_compat"
                                            : "ggml_linear_q4k_q8_1_x4_compat",
                  {one, input_width, output_width},
                  {one, input_width, output_width, zero, count32, origin,
                   program.Binding(2), input, weight, output});
  }
  loomc_status_t Norm(const model::TensorRecord &tensor, Value input,
                      Value output) {
    Value weight;
    RETURN_IF_ERROR(Weight(tensor, &weight));
    return Launch("llama32_rmsnorm_f32", {one}, {one, input, weight, output});
  }
  ModuleBuilder &module;
  CommandProgramBuilder &program;
  CommandValues values;
  Value one, zero, origin, count32, yes, no, hidden, ffn, kv, kind;
};
} // namespace

loomc_status_t DefineLlamaStep(const model::ModelDescriptor &model,
                               ModuleBuilder &module, LlamaStepKind kind,
                               std::string_view name, ProgramRef *out) {
  if (!out || name.empty())
    return Invalid("missing Llama step name/output");
  *out = {};
  if (kind != LlamaStepKind::kPrefill && kind != LlamaStepKind::kDecode)
    return Invalid("invalid Llama step kind");
  RETURN_IF_ERROR(Validate(model));
  std::unique_ptr<CommandProgramBuilder> program;
  RETURN_IF_ERROR(module.DefineProgram(name, {}, 7, &program));
  Recipe r(module, *program);
  RETURN_IF_ERROR(r.Index(1, &r.one));
  RETURN_IF_ERROR(r.Index(0, &r.zero));
  RETURN_IF_ERROR(r.Index(kind == LlamaStepKind::kDecode ? 1 : 0, &r.kind));
  RETURN_IF_ERROR(r.Index(model.embedding_width, &r.hidden));
  RETURN_IF_ERROR(r.Index(model.layers.front().feed_forward_width, &r.ffn));
  RETURN_IF_ERROR(r.Index(1024, &r.kv));
  RETURN_IF_ERROR(r.values.Constant(0, LOOM_SCALAR_TYPE_OFFSET, &r.origin));
  RETURN_IF_ERROR(r.values.Constant(1, LOOM_SCALAR_TYPE_I32, &r.count32));
  RETURN_IF_ERROR(r.values.Constant(1, LOOM_SCALAR_TYPE_I1, &r.yes));
  RETURN_IF_ERROR(r.values.Constant(0, LOOM_SCALAR_TYPE_I1, &r.no));
  Value normalized, packed, query_projection, key_projection, value_projection;
  Value query, attention, projected, swiglu, raw;
  RETURN_IF_ERROR(r.Scratch(3072 * 4, &normalized));
  RETURN_IF_ERROR(r.Scratch(3072 / 128 * 144, &packed));
  RETURN_IF_ERROR(r.Scratch(3072 * 4, &query_projection));
  RETURN_IF_ERROR(r.Scratch(1024 * 4, &key_projection));
  RETURN_IF_ERROR(r.Scratch(1024 * 4, &value_projection));
  RETURN_IF_ERROR(r.Scratch(3072 * 4, &query));
  RETURN_IF_ERROR(r.Scratch(3072 * 4, &attention));
  RETURN_IF_ERROR(r.Scratch(3072 * 4, &projected));
  RETURN_IF_ERROR(r.Scratch(8192 * 4, &swiglu));
  RETURN_IF_ERROR(r.Scratch(4, &raw));
  Value embedding, embedding_bytes;
  RETURN_IF_ERROR(r.Index(Role(model.globals, "token_embedding")->byte_size,
                          &embedding_bytes));
  RETURN_IF_ERROR(r.Weight(*Role(model.globals, "token_embedding"),
                           embedding_bytes, &embedding));
  RETURN_IF_ERROR(
      r.Launch("llama32_token_embedding_q6k_f32", {r.one},
               {r.one, program->Binding(5), embedding, program->Binding(1)}));
  // RoPE's typed contract is 64xf32 rather than the generic byte-view contract.
  Value factor_count, factors;
  RETURN_IF_ERROR(r.Index(64, &factor_count));
  RETURN_IF_ERROR(r.values.Parameter(
      program->Binding(0), Role(model.globals, "rope_frequencies")->name, {},
      LOOM_SCALAR_TYPE_F32, {factor_count}, &factors));
  Value ffn_bytes, hidden_bytes, four, packed_row_bytes, packed_depth4_bytes;
  RETURN_IF_ERROR(r.Index(8192 * 4, &ffn_bytes));
  RETURN_IF_ERROR(r.Index(3072 * 4, &hidden_bytes));
  RETURN_IF_ERROR(r.Index(4, &four));
  RETURN_IF_ERROR(r.Index(3072 / 128 * 144, &packed_row_bytes));
  RETURN_IF_ERROR(r.Index(4 * 3072 / 128 * 144, &packed_depth4_bytes));
  for (const auto &layer : model.layers) {
    Value layer_index;
    RETURN_IF_ERROR(r.Index(layer.index, &layer_index));
    const auto tensor =
        [&](std::string_view role) -> const model::TensorRecord & {
      return *Role(layer.tensors, role);
    };
    RETURN_IF_ERROR(
        r.Norm(tensor("input_norm"), program->Binding(1), normalized));
    RETURN_IF_ERROR(r.Pack(r.hidden, normalized, packed));
    RETURN_IF_ERROR(r.Project(tensor("query"), r.hidden, r.hidden, packed,
                              query_projection));
    RETURN_IF_ERROR(
        r.Project(tensor("key"), r.hidden, r.kv, packed, key_projection));
    RETURN_IF_ERROR(
        r.Project(tensor("value"), r.hidden, r.kv, packed, value_projection));
    RETURN_IF_ERROR(r.Launch("llama32_rotary_query", {r.one},
                             {r.kind, r.origin, r.zero, program->Binding(2),
                              factors, query_projection, query}));
    RETURN_IF_ERROR(
        r.Launch("llama32_publish_kv", {r.one},
                 {layer_index, r.origin, r.kind, r.origin, r.zero, r.origin,
                  program->Binding(2), program->Binding(3), factors,
                  key_projection, value_projection, program->Binding(4)}));
    RETURN_IF_ERROR(
        r.Launch("llama32_attention_rows", {r.one},
                 {layer_index, r.origin, r.kind, r.origin, r.zero, r.origin,
                  program->Binding(2), program->Binding(3), query,
                  program->Binding(4), attention}));
    RETURN_IF_ERROR(r.Pack(r.hidden, attention, packed));
    RETURN_IF_ERROR(r.Project(tensor("attention_output"), r.hidden, r.hidden,
                              packed, projected));
    RETURN_IF_ERROR(r.Launch("llama32_attention_residual", {r.one},
                             {r.kind, r.origin, r.zero, program->Binding(2),
                              projected, program->Binding(1)}));
    RETURN_IF_ERROR(
        r.Norm(tensor("post_norm"), program->Binding(1), normalized));
    Value gate, up, down, gate_bytes, down_bytes;
    RETURN_IF_ERROR(r.Index(tensor("ffn_gate").byte_size, &gate_bytes));
    RETURN_IF_ERROR(r.Index(tensor("ffn_down").byte_size, &down_bytes));
    RETURN_IF_ERROR(r.Weight(tensor("ffn_gate"), gate_bytes, &gate));
    RETURN_IF_ERROR(r.Weight(tensor("ffn_up"), gate_bytes, &up));
    RETURN_IF_ERROR(r.Weight(tensor("ffn_down"), down_bytes, &down));
    const bool down_q6 = tensor("ffn_down").storage_type == 14;
    if (kind == LlamaStepKind::kDecode) {
      RETURN_IF_ERROR(r.Pack(r.hidden, normalized, packed));
      RETURN_IF_ERROR(Apply(*program, "llm.ffn.gate_up_decode",
                            {r.hidden, r.ffn, r.yes, r.yes, r.yes, r.no,
                             gate_bytes, ffn_bytes, gate, up, packed, swiglu}));
      RETURN_IF_ERROR(
          Apply(*program, "llm.ffn.down_decode",
                {r.hidden, r.ffn, r.yes, r.yes, down_q6 ? r.no : r.yes,
                 down_q6 ? r.yes : r.no, down_bytes, ffn_bytes, hidden_bytes,
                 down, swiglu, program->Binding(1)}));
    } else {
      RETURN_IF_ERROR(Apply(*program, "llm.ffn.gate_up_prefill",
                            {r.hidden,
                             r.ffn,
                             r.yes,
                             r.yes,
                             r.yes,
                             r.no,
                             r.one,
                             r.one,
                             r.one,
                             r.zero,
                             r.origin,
                             r.no,
                             four,
                             ffn_bytes,
                             gate_bytes,
                             packed_row_bytes,
                             packed_depth4_bytes,
                             gate,
                             up,
                             normalized,
                             program->Binding(2),
                             swiglu}));
      RETURN_IF_ERROR(Apply(*program, "llm.ffn.down_prefill",
                            {r.hidden,
                             r.ffn,
                             r.yes,
                             r.yes,
                             down_q6 ? r.no : r.yes,
                             down_q6 ? r.yes : r.no,
                             r.one,
                             r.one,
                             r.one,
                             r.zero,
                             r.origin,
                             r.no,
                             four,
                             ffn_bytes,
                             down_bytes,
                             hidden_bytes,
                             down,
                             program->Binding(2),
                             swiglu,
                             program->Binding(1)}));
    }
  }
  RETURN_IF_ERROR(r.Norm(*Role(model.globals, "output_norm"),
                         program->Binding(1), normalized));
  Value vocabulary, partial_bytes, packed_bytes, logits_bytes;
  RETURN_IF_ERROR(r.Index(model.vocabulary_size, &vocabulary));
  RETURN_IF_ERROR(
      r.Index(((model.vocabulary_size + 4095) / 4096) * 4, &partial_bytes));
  RETURN_IF_ERROR(r.Index(3072 / 128 * 144, &packed_bytes));
  RETURN_IF_ERROR(r.Index(model.vocabulary_size * 4, &logits_bytes));
  RETURN_IF_ERROR(
      Apply(*program, "llm.output.project_greedy",
            {r.one, r.origin, r.origin, r.hidden, vocabulary, partial_bytes,
             packed_bytes, logits_bytes, embedding_bytes, r.no, r.yes,
             embedding, normalized, raw}));
  RETURN_IF_ERROR(r.Launch("llama32_finish_isolated", {},
                           {r.kind, raw, program->Binding(2),
                            program->Binding(5), program->Binding(6)}));
  RETURN_IF_ERROR(program->Finish());
  *out = program->Ref();
  return loomc_ok_status();
}

#undef RETURN_IF_ERROR
} // namespace hrx::loomify_gguf
