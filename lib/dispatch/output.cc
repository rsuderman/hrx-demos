// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/output.h"
#include "loom/ops/config/ops.h"
#include "loomc/iree.h"
#include <algorithm>
#include <string_view>
namespace hrx::dispatch {
namespace {
using namespace operations;
using namespace loomify_gguf;
loomc_status_t Error(const char *message) {
  return loomc_status_from_iree(
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s", message));
}
bool Fail(std::string *e, const char *message) {
  if (e)
    *e = message;
  return false;
}
bool SupportsHead(const Registration &, const InvocationRequest &r,
                  std::string *e) {
  const auto *a = std::get_if<OutputHeadAttrs>(&r.attributes);
  const auto *x = r.Type(Port::Input), *w = r.Type(Port::Weight),
             *g = r.Type(Port::Gamma);
  return (a && x && w && g && a->width == 3072 &&
          a->vocabulary_size == 128256 && float(a->epsilon) == 1e-5f &&
          a->row_selection == RowSelection::LastActive && x->dimension(0) > 0 &&
          x->dimension(0) <= 512 && w->encoding() == Encoding::Q6_K &&
          g->encoding() == Encoding::Dense) ||
         Fail(e,
              "unsupported fused output geometry, encoding or normalization");
}
bool SupportsPublish(const Registration &r, const InvocationRequest &q,
                     std::string *e) {
  const auto *a = std::get_if<PublishTokenAttrs>(&q.attributes);
  if (!a || a->phase != r.phase || a->history_capacity != 128 ||
      a->terminal_policy != TerminalPolicy::StopOnTokenIds)
    return Fail(
        e, "unsupported native publication phase, history or terminal policy");
  auto ids = a->terminal_ids;
  std::sort(ids.begin(), ids.end());
  return ids == std::vector<int32_t>({128001, 128008, 128009}) ||
         Fail(e, "native publication maps exactly the three primary Llama "
                 "terminal IDs");
}
loomc_status_t BuildPublication(const Registration &r, BuildContext &c,
                                Value *) {
  for (auto p : {Port::Token, Port::Control, Port::History, Port::Progress})
    if (!c.Operand(p) || !c.Operand(p)->parameter.empty())
      return Error(
          "publication requires explicit mutable state buffers and raw token");
  KernelRef kernel;
  auto s = c.module.Kernel(r.symbol, &kernel);
  if (!loomc_status_is_ok(s))
    return s;
  auto i = c.module.Scalar(LOOM_SCALAR_TYPE_INDEX), b = c.module.Buffer();
  s = c.module.CheckKernelInterface(kernel, {}, {i, b, b, b, b});
  if (!loomc_status_is_ok(s))
    return s;
  c.staged = true;
  Value kind;
  s = c.values.Constant(r.phase == Phase::Decode ? 1 : 0,
                        LOOM_SCALAR_TYPE_INDEX, &kind);
  if (!loomc_status_is_ok(s))
    return s;
  return c.program.Invoke(
      kernel, {},
      {kind, c.Operand(Port::Token)->native, c.Operand(Port::Control)->native,
       c.Operand(Port::History)->native, c.Operand(Port::Progress)->native});
}
loomc_status_t BuildHead(const Registration &r, BuildContext &c, Value *out) {
  const auto *control = c.Operand(Port::Control);
  if (!control || !control->parameter.empty())
    return Error("last-active output requires explicit active-count control");
  const auto *x = c.Operand(Port::Input), *g = c.Operand(Port::Gamma),
             *w = c.Operand(Port::Weight);
  if (!x || !g || !w || !x->parameter.empty() ||
      w->parameter != "token_embd.weight")
    return Error(
        "output head supports the canonical tied token_embd.weight only");
  CommandFamilyRef family, expected;
  auto s = c.module.CommandFamily(r.symbol, &family);
  if (!loomc_status_is_ok(s))
    return s;
  s = c.module.CommandFamily(r.interface_symbol, &expected);
  if (!loomc_status_is_ok(s))
    return s;
  s = c.module.CheckCommandFamilyInterface(family, expected);
  if (!loomc_status_is_ok(s))
    return s;
  s = c.program.InspectNative({}, [](const loom_module_t *native,
                                     const std::vector<loom_value_id_t> &) {
    bool width = false, epsilon = false, vocabulary = false;
    for (size_t i = 0; i < native->symbols.count; ++i) {
      const auto &symbol = native->symbols.entries[i];
      if (symbol.name_id == LOOM_STRING_ID_INVALID || !symbol.defining_op ||
          !loom_config_def_isa(symbol.defining_op))
        continue;
      auto name = loom_string_table_get(&native->strings, symbol.name_id);
      auto value = loom_config_def_value(symbol.defining_op);
      std::string_view key(name.data, name.size);
      if (key == "llama.model.hidden_size")
        width = value.kind == LOOM_ATTR_I64 && value.i64 == 3072;
      if (key == "llama.model.rms_epsilon")
        epsilon = value.kind == LOOM_ATTR_F64 && float(value.f64) == 1e-5f;
      if (key == "qwen38.greedy_argmax.output_capacity")
        vocabulary = value.kind == LOOM_ATTR_I64 && value.i64 >= 128256;
    }
    return width && epsilon && vocabulary
               ? iree_ok_status()
               : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "native output configuration disagrees with "
                                  "supported head facts");
  });
  if (!loomc_status_is_ok(s))
    return s;
  c.staged = true;
#define TRY(call)                                                              \
  do {                                                                         \
    auto status = (call);                                                      \
    if (!loomc_status_is_ok(status))                                           \
      return status;                                                           \
  } while (0)
  auto constant = [&](uint64_t n, loom_scalar_type_t t, Value *v) {
    return c.values.Constant(n, t, v);
  };
  Value capacity, begin, origin, hidden, vocab, bytes, q8, logits, partial,
      gamma = g->native, weight, normalized;
  TRY(constant(c.request.Type(Port::Input)->dimension(0),
               LOOM_SCALAR_TYPE_INDEX, &capacity));
  TRY(constant(c.llama_rows ? c.llama_rows->token_begin : 0,
               LOOM_SCALAR_TYPE_INDEX, &begin));
  TRY(constant(c.count_origin, LOOM_SCALAR_TYPE_OFFSET, &origin));
  TRY(constant(3072, LOOM_SCALAR_TYPE_INDEX, &hidden));
  TRY(constant(128256, LOOM_SCALAR_TYPE_INDEX, &vocab));
  TRY(constant(3072ull * 128256 / 256 * 210, LOOM_SCALAR_TYPE_INDEX, &bytes));
  TRY(constant(3072 / 128 * 144, LOOM_SCALAR_TYPE_OFFSET, &q8));
  TRY(constant(128256 * 4, LOOM_SCALAR_TYPE_OFFSET, &logits));
  TRY(constant(((128256 + 4095) / 4096) * 4, LOOM_SCALAR_TYPE_OFFSET,
               &partial));
  TRY(c.values.Parameter(w->native, w->parameter, {}, LOOM_SCALAR_TYPE_I8,
                         {bytes}, &weight));
  if (!g->parameter.empty())
    TRY(c.values.Parameter(g->native, g->parameter, {}, LOOM_SCALAR_TYPE_F32,
                           {hidden}, &gamma));
  TRY(c.Allocate(3072 * 4, &normalized));
  Value token = c.output;
  if (!c.has_output)
    TRY(c.Allocate(4, &token));
  TRY(c.program.Apply(family, {capacity, begin, origin, hidden, vocab, bytes,
                               q8, logits, partial, x->native, control->native,
                               gamma, weight, normalized, token}));
  *out = token;
#undef TRY
  return loomc_ok_status();
}
} // namespace
OperationBuilder OutputBuilder() {
  return {OpCode::OutputHead,
          Kind::CommandFamily,
          {{Port::Input, Access::Read},
           {Port::Gamma, Access::Read},
           {Port::Weight, Access::Read},
           {Port::Control, Access::Read},
           {Port::Output, Access::Write}},
          SupportsHead,
          BuildHead,
          {"llm_output_last_greedy"}};
}
OperationBuilder PublicationBuilder() {
  return {OpCode::PublishToken,
          Kind::Kernel,
          {{Port::Token, Access::Read},
           {Port::Control, Access::ReadWrite},
           {Port::History, Access::ReadWrite},
           {Port::Progress, Access::ReadWrite}},
          SupportsPublish,
          BuildPublication,
          {"llama32_native_token"}};
}
std::vector<Registration> OutputRegistrations() {
  auto file = [](const char *n) {
    return std::string("lib/kernels/llama/") + n;
  };
  std::vector<Registration> entries;
  Registration head{Id::OutputHead,
                    OpCode::OutputHead,
                    Kind::CommandFamily,
                    "llama.output.last_greedy",
                    "llm.output.last_greedy",
                    file("output/output_interface.loom"),
                    {},
                    {},
                    {"llama.model.hidden_size", "llama.model.rms_epsilon",
                     "qwen38.greedy_argmax.output_capacity"}};
  for (auto n :
       {"active_rows.loom", "projection/linear_q6k_q8_1_x4.loom",
        "projection/quantize_q8_1_x4.loom", "output/output.loom",
        "output/output_norm_last.loom", "output/head_count.loom", "rows.loom",
        "config.loom", "output/output_config.loom", "output/greedy_argmax.loom",
        "projection/quantize_q8_1_x4_mmvq_compat.loom",
        "projection/linear_q6k_q8_1_x4_compat.loom",
        "projection/linear_qk_mmvq_compat.loom",
        "projection/linear_qk_common.loom"})
    head.provider_files.push_back(file(n));
  head.builder = OutputBuilder();
  head.operands = head.builder.operands;
  head.interface_symbol = head.symbol;
  entries.push_back(std::move(head));
  for (auto id : {Id::PublishPrefill, Id::PublishDecode}) {
    Registration r{
        id,
        OpCode::PublishToken,
        Kind::Kernel,
        id == Id::PublishPrefill ? "llama.publish.prefill"
                                 : "llama.publish.decode",
        "llama32_finish_isolated",
        file("publication/publication_interface.loom"),
        {file("publication/publication.loom"), file("output/head_count.loom")},
        {},
        {}};
    r.phase = id == Id::PublishPrefill ? Phase::Prefill : Phase::Decode;
    r.builder = PublicationBuilder();
    r.operands = r.builder.operands;
    r.interface_symbol = r.symbol;
    entries.push_back(std::move(r));
  }
  return entries;
}
} // namespace hrx::dispatch
