// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen_output.h"
#include "lib/dispatch/qwen.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include <algorithm>
#include <map>
#include <string_view>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
bool Fail(std::string *e, const char *message) {
  if (e)
    *e = message;
  return false;
}
bool EmbeddingSupports(const Registration &, const InvocationRequest &r,
                       std::string *e) {
  const auto *a = std::get_if<EmbeddingAttrs>(&r.attributes);
  const auto *weight = r.Type(Port::Weight), *ids = r.Type(Port::TokenIds);
  return a && weight && ids && a->embedding_width == 1024 &&
             a->vocabulary_size == 248320 && ids->dimension(0) >= 1 &&
             ids->dimension(0) <= 512 && weight->encoding() == Encoding::Q6_K ||
         Fail(e, "Qwen embedding requires E1024,V248320,Q6_K and T1..512");
}
bool HeadSupports(const Registration &, const InvocationRequest &r,
                  std::string *e) {
  const auto *a = std::get_if<QwenEpochHeadAttrs>(&r.attributes);
  const auto *weight = r.Type(Port::Weight);
  return a && weight && a->width == 1024 && a->vocabulary_size == 248320 &&
             float(a->epsilon) == 1e-6f && a->token_capacity >= 1 &&
             a->token_capacity <= 512 && a->span_capacity == 1 &&
             a->selected_output_capacity == 1 &&
             weight->encoding() == Encoding::Q6_K ||
         Fail(e, "Qwen epoch head requires E1024,V248320,Q6_K,T1..512,S1,O1");
}
loomc_status_t CheckConfiguration(BuildContext &c, uint64_t tokens) {
  return c.program.InspectNative({}, [&](const loom_module_t *module,
                                         const std::vector<loom_value_id_t> &) {
    std::map<std::string, loom_attribute_t> values;
    for (size_t i = 0; i < module->symbols.count; ++i) {
      const auto &s = module->symbols.entries[i];
      if (!s.defining_op || !loom_config_def_isa(s.defining_op))
        continue;
      auto name = loom_string_table_get(&module->strings, s.name_id);
      values[{name.data, name.size}] = loom_config_def_value(s.defining_op);
    }
    auto index = [&](const char *key, int64_t n) {
      auto v = values.find(key);
      return v != values.end() && v->second.kind == LOOM_ATTR_I64 &&
             v->second.i64 == n;
    };
    auto eps = values.find("qwen.model.rms_epsilon");
    if (!index("qwen.model.hidden_size", 1024) ||
        !index("qwen.model.vocabulary_size", 248320) ||
        !index("qwen.model.embedding_row_bytes", 840) ||
        !index("qwen.model.output_weight_bytes", 208588800) ||
        !index("runner.qwen38.prefill_token_count", tokens) ||
        eps == values.end() || eps->second.kind != LOOM_ATTR_F64 ||
        float(eps->second.f64) != 1e-6f)
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen output configuration disagrees with graph");
    return iree_ok_status();
  });
}
loomc_status_t CheckBuffers(BuildContext &c, const std::vector<Value> &inputs) {
  auto buffers = inputs;
  if (c.has_output)
    buffers.push_back(c.output);
  return c.program.InspectNative(
      buffers,
      [&](const loom_module_t *m, const std::vector<loom_value_id_t> &ids) {
        for (auto id : ids)
          if (!loom_type_is_buffer(loom_module_value_type(m, id)))
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "Qwen output expects scoped buffers");
        if (c.has_output)
          for (size_t i = 0; i + 1 < ids.size(); ++i)
            if (ids[i] == ids.back())
              return iree_make_status(
                  IREE_STATUS_INVALID_ARGUMENT,
                  "Qwen output must not alias a read-only boundary");
        return iree_ok_status();
      });
}
#define TRY(call)                                                              \
  do {                                                                         \
    auto s = (call);                                                           \
    if (!loomc_status_is_ok(s))                                                \
      return s;                                                                \
  } while (0)
loomc_status_t EmbeddingBuild(const Registration &r, BuildContext &c,
                              Value *out) {
  const auto *ids = c.Operand(Port::TokenIds),
             *weight = c.Operand(Port::Weight);
  if (!ids || !weight || !ids->parameter.empty())
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen embedding requires explicit IDs and named weight"));
  Value parameters;
  TRY(CheckQwenLayerParameters(c, {{Port::Weight, "token_embd.weight"}},
                               &parameters));
  auto tokens = c.request.Type(Port::TokenIds)->dimension(0);
  TRY(CheckConfiguration(c, tokens));
  TRY(CheckBuffers(c, {parameters, ids->native}));
  KernelRef kernel;
  TRY(c.module.Kernel(r.symbol, &kernel));
  auto i = c.module.Scalar(LOOM_SCALAR_TYPE_INDEX), b = c.module.Buffer();
  TRY(c.module.CheckKernelInterface(kernel, {i}, {b, b, b}));
  c.staged = true;
  Value count, bytes, view, output = c.output;
  TRY(c.values.Constant(tokens, LOOM_SCALAR_TYPE_INDEX, &count));
  TRY(c.values.Constant(208588800, LOOM_SCALAR_TYPE_INDEX, &bytes));
  TRY(c.values.Parameter(parameters, "token_embd.weight", {},
                         LOOM_SCALAR_TYPE_I8, {bytes}, &view));
  if (!c.has_output)
    TRY(c.Allocate(2097152, &output));
  TRY(c.program.Invoke(kernel, {count}, {ids->native, view, output}));
  *out = output;
  return loomc_ok_status();
}
loomc_status_t HeadBuild(const Registration &r, BuildContext &c, Value *out) {
  const auto &a = std::get<QwenEpochHeadAttrs>(c.request.attributes);
  const auto *input = c.Operand(Port::Input), *epoch = c.Operand(Port::Epoch);
  if (!input || !epoch || !input->parameter.empty() ||
      !epoch->parameter.empty())
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Qwen head requires explicit residual and epoch"));
  Value parameters;
  TRY(CheckQwenLayerParameters(c,
                               {{Port::Gamma, "output_norm.weight"},
                                {Port::Weight, "token_embd.weight"}},
                               &parameters));
  TRY(CheckConfiguration(c, a.token_capacity));
  TRY(CheckBuffers(c, {parameters, input->native, epoch->native}));
  ProgramRef head;
  TRY(c.module.Program(r.symbol, &head));
  auto i = c.module.Scalar(LOOM_SCALAR_TYPE_INDEX), b = c.module.Buffer();
  TRY(c.module.CheckProgramInterface(head, {i, i}, {b, b, b, b}));
  c.staged = true;
  Value tokens, outputs, output = c.output;
  TRY(c.values.Constant(a.token_capacity, LOOM_SCALAR_TYPE_INDEX, &tokens));
  TRY(c.values.Constant(a.selected_output_capacity, LOOM_SCALAR_TYPE_INDEX,
                        &outputs));
  if (!c.has_output)
    TRY(c.Allocate(4, &output));
  TRY(c.program.Call(head, {tokens, outputs},
                     {parameters, input->native, epoch->native, output}));
  *out = output;
  return loomc_ok_status();
}
#undef TRY
std::vector<std::string> Providers() {
  std::vector<std::string> result = {
      "lib/kernels/qwen/output/embedding.loom",
      "lib/kernels/qwen/output/head.loom",
      "lib/kernels/qwen/output/publication.loom",
      "lib/kernels/llama/embedding/embedding_rows.loom"};
  for (const auto &source : QwenSharedSources())
    for (const auto &file : source.provider_files)
      if (std::find(result.begin(), result.end(), file) == result.end())
        result.push_back(file);
  return result;
}
} // namespace
std::vector<Registration> QwenOutputRegistrations() {
  std::vector<Registration> result;
  for (bool head : {false, true}) {
    OperationBuilder builder;
    builder.operation = head ? OpCode::QwenEpochHead : OpCode::Embedding;
    builder.kind = head ? Kind::CommandProgram : Kind::Kernel;
    builder.operands =
        head ? std::vector<OperandContract>{{Port::Input, Access::Read},
                                            {Port::Gamma, Access::Read},
                                            {Port::Weight, Access::Read},
                                            {Port::Epoch, Access::Read}}
             : std::vector<OperandContract>{{Port::TokenIds, Access::Read},
                                            {Port::Weight, Access::Read}};
    builder.supports = head ? HeadSupports : EmbeddingSupports;
    builder.build = head ? HeadBuild : EmbeddingBuild;
    auto symbol =
        head ? "qwen08_epoch_head" : "qwen38_token_embedding_q6k_f16_prefill";
    builder.required_definitions = {symbol};
    Registration r;
    r.id = head ? Id::QwenEpochHead : Id::Embedding;
    r.operation = builder.operation;
    r.kind = builder.kind;
    r.name = head ? "qwen.epoch_head" : "qwen.embedding";
    r.symbol = symbol;
    r.interface_symbol = symbol;
    r.interface_file = "lib/kernels/qwen/output/interface.loom";
    r.provider_files = Providers();
    r.required_config = {"qwen.model.hidden_size",
                         "qwen.model.vocabulary_size",
                         "qwen.model.rms_epsilon",
                         "qwen.model.embedding_row_bytes",
                         "qwen.model.output_weight_bytes",
                         "runner.qwen38.prefill_token_count",
                         "qwen38.greedy_argmax.output_capacity",
                         "ggml.linear_q6k_q8_1_x4.token_capacity",
                         "ggml.linear_q6k_q8_1_x4.output_capacity",
                         "ggml.quantize_q8_1_x4.group_capacity"};
    r.operands = builder.operands;
    r.builder = std::move(builder);
    result.push_back(std::move(r));
  }
  return result;
}
} // namespace hrx::dispatch
