// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/dispatch/qwen.h"
#include "lib/qwen/testing/model_fixture.h"
#include "tools/loomify-gguf/authoring/commands/command_values.h"
#include "loomc/iree.h"
#include "loomc/target/amdgpu.h"
#include <cstdio>
#include <memory>
using namespace hrx::dispatch;
using namespace hrx::loomify_gguf;
namespace {
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    std::fprintf(stderr, "%.*s\n", int(message.size), message.data);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result)) return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto *d = loomc_result_diagnostic_at(result, i);
      std::fprintf(stderr, "%.*s\n", int(d->message.size), d->message.data);
    }
  return false;
}
bool Projection(ModuleBuilder &module, uint64_t rows, uint64_t width,
                unsigned tag, bool reference, std::vector<RootRef> *roots) {
  const auto suffix = "proj_" + std::to_string(tag) + "_" +
      std::to_string(rows) + "_" + std::to_string(width);
  if (reference) {
    RootRef root;
    if (!Ok(module.Root("reference." + suffix, &root))) return false;
    roots->push_back(root);
    return true;
  }
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module.DefineProgram("actual." + suffix,
                               {}, 4, &program))) return false;
  CommandValues values(*program);
  Value n, input, output;
  if (!Ok(values.Constant(rows, LOOM_SCALAR_TYPE_INDEX, &n)) ||
      !Ok(values.Constant(width, LOOM_SCALAR_TYPE_INDEX, &input)) ||
      !Ok(values.Constant(13, LOOM_SCALAR_TYPE_INDEX, &output))) return false;
  KernelRef init;
  if (!Ok(module.Kernel("test.init." + suffix, &init)) ||
      !Ok(program->Invoke(init, {}, {program->Binding(0), program->Binding(1)})))
    return false;
  KernelRef pack, projection;
  if (!Ok(module.Kernel("ggml_quantize_q8_1_x4_f32", &pack)) ||
      !Ok(module.Kernel("ggml_linear_q" + std::to_string(tag - 8) +
                        "k_q8_1_x4", &projection)) ||
      !Ok(program->Invoke(pack, {n, input},
            {n, input, program->Binding(0), program->Binding(2)})) ||
      !Ok(program->Invoke(projection, {n, input, output},
            {n, input, output, program->Binding(2), program->Binding(1),
             program->Binding(3)}))) return false;
  if (!Ok(program->Finish())) return false;
  roots->push_back(program->Ref());
  return true;
}
bool Ffn(ModuleBuilder &module, uint64_t rows, unsigned layer, bool reference,
         std::vector<RootRef> *roots) {
  const auto suffix = "ffn_" + std::to_string(layer) + "_" + std::to_string(rows);
  if (reference) {
    RootRef root;
    if (!Ok(module.Root("reference." + suffix, &root))) return false;
    roots->push_back(root);
    return true;
  }
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Ok(module.DefineProgram("actual." + suffix,
                               {}, 5, &program))) return false;
  CommandValues values(*program);
  auto index = [&](uint64_t n, Value *v) {
    return Ok(values.Constant(n, LOOM_SCALAR_TYPE_INDEX, v));
  };
  Value l, n, hidden, zero, span, gate_bytes, down_bytes;
  const unsigned gate_tag = 12 + layer % 3, down_tag = 13 + layer % 2;
  const uint64_t gate_block = gate_tag == 12 ? 144 : gate_tag == 13 ? 176 : 210;
  if (!index(layer, &l) || !index(rows, &n) || !index(1024, &hidden) ||
      !index(0, &zero) || !index(1, &span) ||
      !index(1024 / 256 * 3584 * gate_block, &gate_bytes) ||
      !index(3584 / 256 * 1024 * (down_tag == 13 ? 176 : 210), &down_bytes))
    return false;
  Value gate, up, down;
  if (!Ok(values.Parameter(program->Binding(0), "blk.{}.ffn_gate.weight", {l},
                            LOOM_SCALAR_TYPE_I8, {gate_bytes}, &gate)) ||
      !Ok(values.Parameter(program->Binding(0), "blk.{}.ffn_up.weight", {l},
                            LOOM_SCALAR_TYPE_I8, {gate_bytes}, &up)) ||
      !Ok(values.Parameter(program->Binding(0), "blk.{}.ffn_down.weight", {l},
                            LOOM_SCALAR_TYPE_I8, {down_bytes}, &down))) return false;
  KernelRef init;
  if (!Ok(module.Kernel("test.init." + suffix, &init)) ||
      !Ok(program->Invoke(init, {}, {program->Binding(1), gate, up, down,
                                    program->Binding(2)}))) return false;
  if (rows == 1) {
    KernelRef pack;
    ProgramRef gate_up, residual_down;
    Value bytes, packed;
    if (!Ok(values.Constant(1024 / 128 * 144, LOOM_SCALAR_TYPE_OFFSET, &bytes)) ||
        !Ok(values.Allocate(bytes, 256, &packed)) ||
        !Ok(module.Kernel("ggml_quantize_q8_1_x4_f32", &pack)) ||
        !Ok(program->Invoke(pack, {n, hidden},
              {n, hidden, program->Binding(1), packed})) ||
        !Ok(module.Program("qwen38_ffn_gate_up_decode", &gate_up)) ||
        !Ok(module.Program("qwen38_ffn_down_decode", &residual_down)) ||
        !Ok(program->Call(gate_up, {l},
              {program->Binding(0), packed, program->Binding(3)})) ||
        !Ok(program->Call(residual_down, {l},
              {program->Binding(0), program->Binding(3), program->Binding(2)})))
      return false;
  } else {
    ProgramRef gate_up, residual_down;
    if (!Ok(module.Program("qwen38_ffn_gate_up_prefill", &gate_up)) ||
        !Ok(module.Program("qwen38_ffn_down_prefill", &residual_down)) ||
        !Ok(program->Call(gate_up, {n, n, l, zero, span},
              {program->Binding(0), program->Binding(1), program->Binding(4),
               program->Binding(3)})) ||
        !Ok(program->Call(residual_down, {n, n, l, zero, span},
              {program->Binding(0), program->Binding(4), program->Binding(3),
               program->Binding(2)}))) return false;
  }
  if (!Ok(program->Finish())) return false;
  roots->push_back(program->Ref());
  return true;
}
} // namespace
int main(int argc, char **argv) {
  if (argc != 4) return 1;
  loomc_target_environment_t *environment = nullptr;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
  if (!Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                               &environment))) return 1;
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment;
  loomc_context_options_t options{};
  options.next = &target;
  if (!Ok(loomc_context_create(&options, loomc_allocator_system(), &context)) ||
      !Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)))
    return 1;
  Registry registry;
  auto sources = QwenSharedSources();
  if (sources.size() != 13) return 1;
  // Test-only closure entries expose the actual NativeSources to the existing
  // descriptor loader. They can never masquerade as typed operation producers.
  for (const auto &source : sources) {
    Registration entry;
    entry.id = Id::RmsNorm;
    entry.operation = hrx::operations::OpCode::RmsNorm;
    entry.name = "test.qwen.common.sources";
    entry.variant = source.symbol;
    entry.symbol = source.symbol;
    entry.kind = source.kind;
    entry.interface_symbol = source.interface_symbol;
    entry.interface_file = source.interface_file;
    entry.provider_files = source.provider_files;
    entry.provider_files.push_back("lib/dispatch/testing/qwen/common/fixtures.loom");
    entry.required_config = source.required_config;
    entry.builder.operation = entry.operation;
    entry.builder.kind = entry.kind;
    entry.builder.required_definitions = source.required_definitions;
    entry.builder.supports = [](const Registration &, const InvocationRequest &,
                                std::string *) { return false; };
    entry.builder.build = [](const Registration &, BuildContext &, Value *) {
      return loomc_status_from_iree(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                                  "native test source only"));
    };
    std::string error;
    if (!registry.Add(std::move(entry), &error)) {
      std::fprintf(stderr, "%s\n", error.c_str()); return 1;
    }
  }
  auto facts = hrx::qwen::testing::ModelFacts();
  for (auto &layer : facts.layers)
    if (layer.kind == hrx::model::LayerKind::kAttention)
      for (auto &tensor : layer.tensors)
        if (tensor.role == "key") {
          tensor.tensor.storage_type = 14;
          tensor.tensor.byte_size = 1024 / 256 * 512 * 210;
        }
  QwenNativeGeometry geometry;
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Ok(LoadQwenNative(context, workspace, registry, {Id::RmsNorm}, facts,
                         {3, 1, 1, 65, 128, 1}, argv[1], &geometry,
                         &definitions, &result)) || !Succeeded(result)) return 1;
  loomc_result_release(result);
  std::unique_ptr<ModuleBuilder> module;
  if (!Ok(ModuleBuilder::Create(context, workspace, definitions, &module)))
    return 1;
  loomc_module_release(definitions);
  std::vector<RootRef> roots;
  for (uint64_t rows : {1u, 3u}) {
    for (uint64_t width : {256u, 512u})
      for (unsigned tag : {12u, 13u, 14u})
        for (bool reference : {false, true})
          if (!Projection(*module, rows, width, tag, reference, &roots)) return 1;
    for (unsigned layer = 0; layer < 6; ++layer)
      for (bool reference : {false, true})
        if (!Ffn(*module, rows, layer, reference, &roots)) return 1;
  }
  loomc_module_t *linked = nullptr;
  result = nullptr;
  if (!Ok(module->LinkAndVerify(roots, &linked, &result)) || !Succeeded(result))
    return 1;
  loomc_result_release(result);
  bool ok = true;
  for (int i = 0; i < 2; ++i) {
    loomc_module_serialize_options_t serialize{};
    serialize.format = i ? LOOMC_SOURCE_FORMAT_TEXT : LOOMC_SOURCE_FORMAT_BYTECODE;
    ok &= Ok(loomc_module_serialize_to_path(linked, &serialize,
               loomc_make_cstring_view(argv[2 + i]), loomc_allocator_system()));
  }
  loomc_module_release(linked);
  module.reset();
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  loomc_target_environment_release(environment);
  return ok ? 0 : 1;
}
