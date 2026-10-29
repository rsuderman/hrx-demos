// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "tools/loomify-gguf/source_builder.h"
#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "tools/loomify-gguf/authoring/commands/command_values.h"
#include <cstring>
#include <limits>
#include <memory>
namespace hrx::loomify_gguf {
namespace {
// Reusable numerical definition, independent of a GGUF and its shapes. All
// descriptor-specific program structure is built with the host API below.
constexpr char kEmbeddingKernel[] = R"loom(
kernel.def @loomify.embedding.f32(%width: index, %vocab: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%width, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%width: index, %vocab: index, %token: index,
         %table: view<[%vocab]x[%width]xf32>, %output: buffer) {
  %column = kernel.workgroup.id<x> : index
  %origin = index.constant 0 : offset
  %target = buffer.view %output[%origin] : buffer -> view<[%width]xf32>
  %active = index.cmp ult, %column, %width : index
  scf.if %active {
  %valid = index.cmp ult, %token, %vocab : index
  scf.if %valid {
    %value = view.load %table[%token, %column] : view<[%vocab]x[%width]xf32> -> f32
    view.store %value, %target[%column] : f32, view<[%width]xf32>
    scf.yield
  } else {
    %zero = scalar.constant 0.0 : f32
    view.store %zero, %target[%column] : f32, view<[%width]xf32>
    scf.yield
  }
  scf.yield
  }
  kernel.return
}
)loom";
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
bool Check(loomc_status_t status, std::string *error) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    error->assign(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Result(loomc_result_t *result, std::string *error) {
  if (result && loomc_result_succeeded(result))
    return true;
  *error = "Loom construction failed";
  if (result) {
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto *diagnostic = loomc_result_diagnostic_at(result, i);
      error->append("\n");
      error->append(diagnostic->message.data, diagnostic->message.size);
    }
  }
  return false;
}
bool Validate(const hrx::model::ModelDescriptor &model,
              const hrx::model::TensorRecord **tensor, std::string *error) {
  if (model.family != "llama") {
    *error = "embedding generation currently supports the llama architecture";
    return false;
  }
  for (const auto &binding : model.globals) {
    if (binding.role != "token_embedding")
      continue;
    if (*tensor) {
      *error = "ambiguous token_embedding role";
      return false;
    }
    *tensor = &binding.tensor;
  }
  if (!*tensor) {
    *error = "missing token_embedding role";
    return false;
  }
  const auto &t = **tensor;
  const auto width = model.embedding_width;
  const auto vocab = model.vocabulary_size;
  if (!width || !vocab || width > INT64_MAX || vocab > INT64_MAX ||
      width > uint64_t(INT64_MAX) / vocab / sizeof(float) ||
      t.normalized_extents !=
          std::array<int64_t, 4>{int64_t(width), int64_t(vocab), 1, 1}) {
    *error =
        "token_embedding dimensions do not match embedding/vocabulary size";
    return false;
  }
  if (t.storage_type != 0 || t.storage_name != "f32") {
    *error = "token_embedding requires F32 storage; got " + t.storage_name;
    return false;
  }
  if (t.name != "token_embd.weight" ||
      t.byte_size != width * vocab * sizeof(float)) {
    *error = "token_embedding named parameter or storage extent is invalid";
    return false;
  }
  return true;
}
} // namespace
bool BuildEmbeddingProgram(const hrx::model::ModelDescriptor &model,
                           loomc_context_t *context,
                           loomc_workspace_t *workspace,
                           loomc_module_t **out_module, std::string *error) {
  if (out_module)
    *out_module = nullptr;
  if (!error)
    return false;
  error->clear();
  if (!context || !workspace || !out_module) {
    *error = "context, workspace and module output are required";
    return false;
  }
  const hrx::model::TensorRecord *tensor = nullptr;
  if (!Validate(model, &tensor, error))
    return false;
  loomc_source_options_t source_options = {};
  source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
  source_options.identifier = loomc_make_cstring_view("embedding-kernel.loom");
  source_options.contents = {
      reinterpret_cast<const uint8_t *>(kEmbeddingKernel),
      sizeof(kEmbeddingKernel) - 1};
  source_options.storage = LOOMC_SOURCE_STORAGE_BORROWED;
  loomc_source_t *source = nullptr;
  if (!Check(loomc_source_create(&source_options, loomc_allocator_system(),
                                 &source),
             error))
    return false;
  Owner<loomc_source_t, loomc_source_release> source_owner(source);
  loomc_module_t *definitions = nullptr;
  loomc_result_t *result = nullptr;
  if (!Check(loomc_module_deserialize_from_source(
                 context, workspace, source, nullptr, loomc_allocator_system(),
                 &definitions, &result),
             error))
    return false;
  Owner<loomc_module_t, loomc_module_release> definitions_owner(definitions);
  Owner<loomc_result_t, loomc_result_release> parse_result(result);
  if (!Result(result, error))
    return false;
  std::unique_ptr<ModuleBuilder> module;
  if (!Check(ModuleBuilder::Create(context, workspace, definitions, &module),
             error))
    return false;
  KernelRef kernel;
  if (!Check(module->Kernel("loomify.embedding.f32", &kernel), error))
    return false;
  std::unique_ptr<CommandProgramBuilder> program;
  if (!Check(module->DefineProgram("model.embedding", {LOOM_SCALAR_TYPE_INDEX},
                                   2, &program),
             error))
    return false;
  Value width, vocab, table;
  CommandValues values(*program);
  if (!Check(program->IndexConstant(model.embedding_width, &width), error) ||
      !Check(program->IndexConstant(model.vocabulary_size, &vocab), error) ||
      !Check(values.Parameter(program->Binding(0), tensor->name, {},
                              LOOM_SCALAR_TYPE_F32, {vocab, width}, &table),
             error) ||
      !Check(program->Invoke(kernel, {width, vocab},
                             {width, vocab, program->Specialization(0), table,
                              program->Binding(1)}),
             error) ||
      !Check(program->Finish(), error))
    return false;
  result = nullptr;
  loomc_module_t *output = nullptr;
  if (!Check(module->LinkAndVerify({program->Ref()}, &output, &result), error))
    return false;
  Owner<loomc_module_t, loomc_module_release> output_owner(output);
  Owner<loomc_result_t, loomc_result_release> link_result(result);
  if (!Result(result, error))
    return false;
  *out_module = output_owner.release();
  return true;
}
} // namespace hrx::loomify_gguf
