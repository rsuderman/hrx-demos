// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/invocation.h"
#include "lib/operations/ffn_operations.h"
#include "lib/operations/tensor_operations.h"
#include "loomc/target/amdgpu.h"
#include <algorithm>
#include <cstdio>
#include <memory>
using namespace hrx::dispatch;
using namespace hrx::operations;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto m = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(m.size), m.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto *d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  return false;
}
bool Seed(ModuleBuilder &module, CommandProgramBuilder &program, uint64_t n,
          uint64_t capacity, uint64_t begin, uint64_t width, Value compact,
          bool ids, Value *out) {
  CommandValues values(program);
  Value rows, cap, first, size, columns;
  if (!Ok(values.Constant(n, LOOM_SCALAR_TYPE_INDEX, &rows)) ||
      !Ok(values.Constant(capacity, LOOM_SCALAR_TYPE_INDEX, &cap)) ||
      !Ok(values.Constant(begin, LOOM_SCALAR_TYPE_INDEX, &first)) ||
      !Ok(values.Constant(capacity * width * 4, LOOM_SCALAR_TYPE_OFFSET,
                          &size)) ||
      !Ok(values.Allocate(size, 256, out)))
    return false;
  KernelRef seed;
  if (!Ok(module.Kernel(ids ? "test.seed_active_i32" : "test.seed_active_f32",
                        &seed)))
    return false;
  if (ids)
    return Ok(program.Invoke(seed, {rows}, {rows, cap, first, compact, *out}));
  return Ok(values.Constant(width, LOOM_SCALAR_TYPE_INDEX, &columns)) &&
         Ok(program.Invoke(seed, {rows, columns},
                           {rows, columns, cap, first, compact, *out}));
}
TensorType Weight(uint64_t in, uint64_t out, Encoding encoding) {
  auto type = TensorType::matrix(in, out, ElementType::F32);
  type.setEncoding(encoding).setLayout(Layout::GgmlWeightRows);
  return type;
}
bool Emit(ModuleBuilder &module, const Registry &registry,
          const std::string &name, uint64_t n, uint64_t capacity,
          uint64_t begin, Encoding encoding, const std::string &kind,
          Phase phase, std::vector<RootRef> *roots) {
  const bool ffn = kind == "ffn", embedding = kind == "embedding",
             norm = kind == "norm";
  const uint64_t width = embedding ? 1 : ffn || norm ? 3072 : 256;
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module.DefineProgram(name, {}, ffn ? 7 : 4, &program)))
    return false;
  OperationRegistry definitions;
  std::string error;
  if (ffn) {
    if (!DefineGatedMatmul(&definitions, &error) ||
        !DefineLinearResidual(&definitions, &error))
      return false;
  } else if (!(embedding ? DefineEmbedding(&definitions, &error)
               : norm    ? DefineRmsNorm(&definitions, &error)
                         : DefineLinear(&definitions, &error)))
    return false;
  OperationGraph graph(definitions);
  const auto x = graph.tensor(
      embedding ? TensorType::vector(capacity, ElementType::I32)
                : TensorType::matrix(capacity, width, ElementType::F32),
      &error);
  const auto y = graph.tensor(TensorType::matrix(capacity,
                                                 ffn                 ? 8192
                                                 : embedding || norm ? 3072
                                                                     : 13,
                                                 ElementType::F32),
                              &error);
  NativeLlamaRows rows;
  rows.row_begin = graph.scalar(ScalarType::I64, &error);
  rows.row_count = graph.scalar(ScalarType::I64, &error);
  rows.row_capacity = capacity;
  rows.token_begin = begin;
  rows.phase = phase;
  rows.packet = program->Binding(ffn ? 4 : 2);
  NativeEmissionSession session(registry, graph, module, *program);
  if (!Ok(session.LlamaRows(rows)))
    return false;
  Value seeded;
  if (!Seed(module, *program, n, capacity, begin, width,
            program->Binding(ffn ? 3 : 0), embedding, &seeded) ||
      !Ok(session.Bind(x, seeded)) ||
      !Ok(session.Bind(y, program->Binding(ffn ? 6 : 3))))
    return false;
  auto append = [&](OpCode code, OperationAttrs attrs,
                    std::vector<shapes::Operand> operands) {
    auto operation = graph.operation(code);
    for (const auto &operand : operands)
      operation.bind(operand.port, operand.value);
    operation.bind(Port::RowBegin, rows.row_begin)
        .bind(Port::RowCount, rows.row_count);
    std::visit([&](const auto &a) { operation.attributes(a); }, attrs);
    return operation.append(&error);
  };
  if (ffn) {
    auto gate = graph.tensor(Weight(3072, 8192, Encoding::Q4_K), &error);
    auto up = graph.tensor(Weight(3072, 8192, Encoding::Q4_K), &error);
    auto down = graph.tensor(Weight(8192, 3072, encoding), &error);
    auto residual = graph.tensor(
        TensorType::matrix(capacity, 3072, ElementType::F32), &error);
    if (!Ok(session.Parameter(gate, program->Binding(0), "gate.weight")) ||
        !Ok(session.Parameter(up, program->Binding(1), "up.weight")) ||
        !Ok(session.Parameter(down, program->Binding(2), "down.weight")) ||
        !Ok(session.Bind(residual, program->Binding(5))))
      return false;
    if (!append(OpCode::GatedMatmul, GatedMatmulAttrs{3072, 8192},
                {{Port::Input, x},
                 {Port::GateWeight, gate},
                 {Port::UpWeight, up},
                 {Port::Output, y}}) ||
        !append(OpCode::LinearResidual, LinearResidualAttrs{8192, 3072},
                {{Port::Input, y},
                 {Port::Weight, down},
                 {Port::Residual, residual}}))
      return false;
    if (!Ok(session.EmitExisting(phase == Phase::Decode ? Id::GateUpDecode
                                                        : Id::GateUpPrefill,
                                 graph.invocations()[0])) ||
        !Ok(session.EmitExisting(phase == Phase::Decode ? Id::DownDecode
                                                        : Id::DownPrefill,
                                 graph.invocations()[1])))
      return false;
  } else if (norm) {
    auto gamma =
        graph.tensor(TensorType::vector(3072, ElementType::F32), &error);
    if (!Ok(session.Bind(gamma, program->Binding(1))) ||
        !append(OpCode::RmsNorm, RmsNormAttrs{3072, 1e-5},
                {{Port::Input, x}, {Port::Gamma, gamma}, {Port::Output, y}}) ||
        !Ok(session.EmitExisting(Id::RmsNorm, graph.invocations()[0])))
      return false;
  } else {
    auto weight = graph.tensor(
        Weight(embedding ? 3072 : 256, embedding ? 128256 : 13, encoding),
        &error);
    if (!Ok(session.Parameter(weight, program->Binding(1), "weight")))
      return false;
    if (embedding) {
      if (!append(OpCode::Embedding, EmbeddingAttrs{128256, 3072},
                  {{Port::TokenIds, x},
                   {Port::Weight, weight},
                   {Port::Output, y}}) ||
          !Ok(session.EmitExisting(Id::Embedding, graph.invocations()[0])))
        return false;
    } else if (!append(OpCode::Linear, LinearAttrs{256, 13},
                       {{Port::Input, x},
                        {Port::Weight, weight},
                        {Port::Output, y}}) ||
               !Ok(session.EmitExisting(Id::Projection,
                                        graph.invocations()[0])))
      return false;
  }
  if (!Ok(program->Finish()))
    return false;
  roots->push_back(program->Ref());
  return true;
}

bool FfnWiring(ModuleBuilder &module, const std::string &prefix, bool decode,
               std::vector<RootRef> *roots) {
  const uint64_t n = decode ? 1 : 3, capacity = decode ? 1 : 8,
                 begin = decode ? 0 : 2;
  ProgramRef active, compact;
  if (!Ok(module.Program(prefix + (decode ? ".decode" : ".n3"), &active)) ||
      !Ok(module.Program(prefix + (decode ? ".decode" : ".compact"), &compact)))
    return false;
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module.DefineProgram(
          prefix + (decode ? ".decode.wiring" : ".prefill.wiring"), {}, 9,
          &program)))
    return false;
  CommandValues values(*program);
  Value rows, cap, first, zero, width, ffn_bytes, residual_bytes,
      compact_residual, compact_swiglu;
  if (!Ok(values.Constant(n, LOOM_SCALAR_TYPE_INDEX, &rows)) ||
      !Ok(values.Constant(capacity, LOOM_SCALAR_TYPE_INDEX, &cap)) ||
      !Ok(values.Constant(begin, LOOM_SCALAR_TYPE_INDEX, &first)) ||
      !Ok(values.Constant(0, LOOM_SCALAR_TYPE_INDEX, &zero)) ||
      !Ok(values.Constant(3072, LOOM_SCALAR_TYPE_INDEX, &width)) ||
      !Ok(values.Constant(n * 8192 * 4, LOOM_SCALAR_TYPE_OFFSET, &ffn_bytes)) ||
      !Ok(values.Constant(n * 3072 * 4, LOOM_SCALAR_TYPE_OFFSET,
                          &residual_bytes)) ||
      !Ok(values.Allocate(residual_bytes, 256, &compact_residual)) ||
      !Ok(values.Allocate(ffn_bytes, 256, &compact_swiglu)))
    return false;
  KernelRef seed, compare;
  if (!Ok(module.Kernel("test.seed_active_f32", &seed)) ||
      !Ok(module.Kernel("test.compare_ffn", &compare)) ||
      !Ok(program->Invoke(seed, {rows, width},
                          {rows, width, cap, first, program->Binding(7),
                           program->Binding(5)})) ||
      !Ok(program->Invoke(
          seed, {rows, width},
          {rows, width, rows, zero, program->Binding(7), compact_residual})))
    return false;
  std::vector<Value> bindings;
  for (unsigned i = 0; i < 7; ++i)
    bindings.push_back(program->Binding(i));
  if (!Ok(program->Call(active, {}, bindings)))
    return false;
  bindings[5] = compact_residual;
  bindings[6] = compact_swiglu;
  if (!Ok(program->Call(compact, {}, bindings)) ||
      !Ok(program->Invoke(compare, {rows, cap},
                          {rows, cap, first, program->Binding(5),
                           program->Binding(6), compact_residual,
                           compact_swiglu, program->Binding(7),
                           program->Binding(8)})) ||
      !Ok(program->Finish()))
    return false;
  roots->push_back(program->Ref());
  return true;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 3)
    return 1;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  loomc_target_environment_t *environment = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                 &environment)))
    return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t options{};
  options.next = &target;
  if (!Ok(loomc_context_create(&options, loomc_allocator_system(), &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                 &workspace)))
    return 1;
  auto registry = Registry::Llama3B();
  std::vector<Registration> entries;
  for (auto id : {Id::Embedding, Id::RmsNorm, Id::Projection, Id::GateUpPrefill,
                  Id::GateUpDecode, Id::DownPrefill, Id::DownDecode}) {
    auto selected = registry.Implementations(id);
    entries.insert(entries.end(), selected.begin(), selected.end());
  }
  entries[0].provider_files.push_back(
      "lib/dispatch/testing/testdata/active_row_seed.loom");
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(Load(context, workspace, entries, argv[1], &definitions, &result)) ||
      !Succeeded(result))
    return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  std::vector<RootRef> roots;
  for (const std::string kind : {"projection", "ffn"})
    for (auto encoding : {Encoding::Q4_K, Encoding::Q6_K}) {
      const std::string prefix =
          "active." + kind + (encoding == Encoding::Q4_K ? ".q4" : ".q6");
      for (uint64_t n : {1, 3, 4, 8, 9, 16, 17})
        if (!Emit(*module, registry, prefix + ".n" + std::to_string(n), n,
                  std::max(uint64_t(8), n + 4), 2, encoding, kind,
                  Phase::Prefill, &roots))
          return 1;
      if (!Emit(*module, registry, prefix + ".decode", 1, 1, 0, encoding, kind,
                Phase::Decode, &roots) ||
          !Emit(*module, registry, prefix + ".tail", 3, 5, 2, encoding, kind,
                Phase::Prefill, &roots))
        return 1;
    }
  for (const std::string kind : {"norm", "embedding"}) {
    if (!Emit(*module, registry, "active." + kind, 3, 8, 2, Encoding::Q6_K,
              kind, Phase::Prefill, &roots) ||
        !Emit(*module, registry, "active." + kind + ".decode", 1, 1, 0,
              Encoding::Q6_K, kind, Phase::Decode, &roots) ||
        !Emit(*module, registry, "active." + kind + ".tail", 3, 5, 2,
              Encoding::Q6_K, kind, Phase::Prefill, &roots))
      return 1;
  }
  for (auto encoding : {Encoding::Q4_K, Encoding::Q6_K}) {
    const std::string prefix =
        encoding == Encoding::Q4_K ? "active.ffn.q4" : "active.ffn.q6";
    if (!Emit(*module, registry, prefix + ".compact", 3, 3, 0, encoding, "ffn",
              Phase::Prefill, &roots) ||
        !FfnWiring(*module, prefix, false, &roots) ||
        !FfnWiring(*module, prefix, true, &roots))
      return 1;
  }
  // Independent-reference cases need a compact oracle scratch argument. The
  // actual wrapper only delegates to the production session-generated root.
  for (const std::string encoding : {"q4", "q6"})
    for (unsigned count : {1u, 3u, 9u}) {
      const auto name =
          "active.projection." + encoding + ".n" + std::to_string(count);
      ProgramRef target;
      std::unique_ptr<CommandProgramBuilder> program;
      if (!Ok(module->Program(name, &target)) ||
          !Ok(module->DefineProgram(name + ".reference", {}, 5, &program)) ||
          !Ok(program->Call(target, {},
                            {program->Binding(0), program->Binding(1),
                             program->Binding(2), program->Binding(3)})) ||
          !Ok(program->Finish()))
        return 1;
      roots.push_back(program->Ref());
    }
  loomc_module_t *linked = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify(roots, &linked, &result)) || !Succeeded(result))
    return 1;
  loomc_result_release(result);
  loomc_module_serialize_options_t serialized{};
  serialized.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  bool ok = Ok(loomc_module_serialize_to_path(linked, &serialized,
                                              loomc_make_cstring_view(argv[2]),
                                              loomc_allocator_system()));
  loomc_module_release(linked);
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
  return ok ? 0 : 1;
}
