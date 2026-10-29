// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/model-full/llama_step.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf {
namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
::testing::AssertionResult Ok(loomc_status_t status) {
  auto message = loomc_status_message(status);
  std::string text(message.data ? message.data : "", message.size);
  auto code = loomc_status_consume_code(status);
  return code == LOOMC_STATUS_OK ? ::testing::AssertionSuccess()
                                 : ::testing::AssertionFailure() << text;
}
std::string Arguments(const std::vector<std::string> &types) {
  std::string result;
  for (size_t i = 0; i < types.size(); ++i) {
    if (i)
      result += ", ";
    result += "%a" + std::to_string(i) + ": " + types[i];
  }
  return result;
}
// No-op kernel contracts isolate construction/closure from numerical execution.
// The separate real-provider gate is required for target behavior.
std::string Definitions() {
  std::string text;
  auto kernel = [&](const char *name, size_t workloads,
                    const std::vector<std::string> &args) {
    text += "kernel.def @" + std::string(name) + "(";
    for (size_t i = 0; i < workloads; ++i) {
      if (i)
        text += ", ";
      text += "%w" + std::to_string(i) + ": index";
    }
    text += ") { %one = index.constant 1 : index\n"
            "kernel.launch.config workgroups(%one, %one, %one) "
            "workgroup_size(%one, %one, %one) : index\n"
            "} launch(" +
            Arguments(args) + ") { kernel.return }\n";
  };
  kernel("llama32_token_embedding_q6k_f32", 1,
         {"index", "buffer", "buffer", "buffer"});
  kernel("llama32_rmsnorm_f32", 1, {"index", "buffer", "buffer", "buffer"});
  kernel("ggml_quantize_q8_1_x4_qk_compat_f32", 2,
         {"index", "index", "index", "i32", "offset", "buffer", "buffer",
          "buffer"});
  for (const char *name :
       {"ggml_linear_q4k_q8_1_x4_compat", "ggml_linear_q6k_q8_1_x4_compat"})
    kernel(name, 3,
           {"index", "index", "index", "index", "i32", "offset", "buffer",
            "buffer", "buffer", "buffer"});
  kernel("llama32_rotary_query", 1,
         {"index", "offset", "index", "buffer", "buffer", "buffer", "buffer"});
  kernel("llama32_publish_kv", 1,
         {"index", "offset", "index", "offset", "index", "offset", "buffer",
          "buffer", "buffer", "buffer", "buffer", "buffer"});
  kernel("llama32_attention_rows", 1,
         {"index", "offset", "index", "offset", "index", "offset", "buffer",
          "buffer", "buffer", "buffer", "buffer"});
  kernel("llama32_attention_residual", 1,
         {"index", "offset", "index", "buffer", "buffer", "buffer"});
  kernel("llama32_finish_isolated", 0,
         {"index", "buffer", "buffer", "buffer", "buffer"});
  auto family = [&](const char *name, const std::vector<std::string> &args) {
    text +=
        "template.decl @" + std::string(name) + "(" + Arguments(args) + ")\n";
  };
  family("llm.rmsnorm.row", {"index", "f32", "index", "index", "offset", "i1",
                             "f32", "buffer", "buffer", "buffer"});
  family("llm.ffn.gate_up_decode",
         {"index", "index", "i1", "i1", "i1", "i1", "index", "index",
          "view<[%a6]xi8>", "view<[%a6]xi8>", "buffer", "buffer"});
  family("llm.ffn.down_decode",
         {"index", "index", "i1", "i1", "i1", "i1", "index", "index", "index",
          "view<[%a6]xi8>", "buffer", "buffer"});
  family("llm.ffn.gate_up_prefill", {"index",
                                     "index",
                                     "i1",
                                     "i1",
                                     "i1",
                                     "i1",
                                     "index",
                                     "index",
                                     "index",
                                     "index",
                                     "offset",
                                     "i1",
                                     "index",
                                     "index",
                                     "index",
                                     "index",
                                     "index",
                                     "view<[%a14]xi8>",
                                     "view<[%a14]xi8>",
                                     "buffer",
                                     "buffer",
                                     "buffer"});
  family("llm.ffn.down_prefill", {"index",
                                  "index",
                                  "i1",
                                  "i1",
                                  "i1",
                                  "i1",
                                  "index",
                                  "index",
                                  "index",
                                  "index",
                                  "offset",
                                  "i1",
                                  "index",
                                  "index",
                                  "index",
                                  "index",
                                  "view<[%a14]xi8>",
                                  "buffer",
                                  "buffer",
                                  "buffer"});
  family("llm.output.project_greedy",
         {"index", "offset", "offset", "index", "index", "index", "index",
          "index", "index", "i1", "i1", "view<[%a8]xi8>", "buffer", "buffer"});
  return text;
}
model::TensorBinding Tensor(const char *role, std::string name,
                            uint32_t storage, uint64_t width,
                            uint64_t rows = 1) {
  model::TensorBinding binding;
  binding.role = role;
  binding.tensor.name = std::move(name);
  binding.tensor.storage_type = storage;
  binding.tensor.normalized_extents = {int64_t(width), int64_t(rows), 1, 1};
  binding.tensor.byte_size =
      storage == 0 ? width * rows * 4
                   : width * rows / 256 * (storage == 12 ? 144 : 210);
  return binding;
}
model::ModelDescriptor Descriptor() {
  model::ModelDescriptor m;
  m.family = "llama";
  m.embedding_width = 3072;
  m.vocabulary_size = 128256;
  m.key_head_width = 128;
  m.value_head_width = 128;
  m.output_tied_to_embedding = true;
  m.rope.width = 128;
  m.rope.base = 500000;
  m.rms_epsilon = 1e-5;
  m.globals = {Tensor("token_embedding", "token_embd.weight", 14, 3072, 128256),
               Tensor("output", "token_embd.weight", 14, 3072, 128256),
               Tensor("output_norm", "output_norm.weight", 0, 3072),
               Tensor("rope_frequencies", "rope_freqs.weight", 0, 64)};
  for (uint32_t i = 0; i < 28; ++i) {
    model::LayerDescriptor l;
    l.index = i;
    l.feed_forward_width = 8192;
    l.query_heads = 24;
    l.kv_heads = 8;
    auto prefix = "blk." + std::to_string(i) + ".";
    l.tensors = {
        Tensor("input_norm", prefix + "attn_norm.weight", 0, 3072),
        Tensor("post_norm", prefix + "ffn_norm.weight", 0, 3072),
        Tensor("query", prefix + "attn_q.weight", 12, 3072, 3072),
        Tensor("key", prefix + "attn_k.weight", 12, 3072, 1024),
        Tensor("value", prefix + "attn_v.weight", i % 2 ? 12 : 14, 3072, 1024),
        Tensor("attention_output", prefix + "attn_output.weight", 12, 3072,
               3072),
        Tensor("ffn_gate", prefix + "ffn_gate.weight", 12, 3072, 8192),
        Tensor("ffn_up", prefix + "ffn_up.weight", 12, 3072, 8192),
        Tensor("ffn_down", prefix + "ffn_down.weight", i % 2 ? 14 : 12, 8192,
               3072)};
    m.layers.push_back(std::move(l));
  }
  return m;
}
class LlamaStepTest : public ::testing::Test {
protected:
  void SetUp() override {
    loomc_context_t *c = nullptr;
    loomc_workspace_t *w = nullptr;
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &c)));
    context.reset(c);
    ASSERT_TRUE(
        Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w)));
    workspace.reset(w);
    auto text = Definitions();
    loomc_source_options_t options = {};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.identifier = loomc_make_cstring_view("contracts.loom");
    options.contents = loomc_make_byte_span(text.data(), text.size());
    loomc_source_t *s = nullptr;
    ASSERT_TRUE(
        Ok(loomc_source_create(&options, loomc_allocator_system(), &s)));
    Owner<loomc_source_t, loomc_source_release> source(s);
    loomc_module_t *d = nullptr;
    loomc_result_t *r = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        c, w, s, nullptr, loomc_allocator_system(), &d, &r)));
    definitions.reset(d);
    Owner<loomc_result_t, loomc_result_release> result(r);
    if (!loomc_result_succeeded(r))
      for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
        auto *diagnostic = loomc_result_diagnostic_at(r, i);
        ADD_FAILURE() << std::string(diagnostic->message.data,
                                     diagnostic->message.size);
      }
    ASSERT_TRUE(loomc_result_succeeded(r));
    ASSERT_TRUE(Ok(ModuleBuilder::Create(c, w, d, &module)));
    ASSERT_TRUE(Ok(module->Kernel("llama32_rmsnorm_f32", &rms_norm)));
  }
  Owner<loomc_context_t, loomc_context_release> context;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace;
  Owner<loomc_module_t, loomc_module_release> definitions;
  std::unique_ptr<ModuleBuilder> module;
  KernelRef rms_norm;
};
TEST_F(LlamaStepTest,
       ConstructsEveryLayerAndReloadsBytecodeWithoutOldCommands) {
  auto model = Descriptor();
  for (bool authored : {false, true}) {
    SCOPED_TRACE(authored);
    ASSERT_TRUE(Ok(ModuleBuilder::Create(context.get(), workspace.get(),
                                         definitions.get(), &module)));
    if (authored)
      ASSERT_TRUE(
          Ok(DefineLlamaRmsNorm(*module, "authored.rmsnorm", &rms_norm)));
    else
      ASSERT_TRUE(Ok(module->Kernel("llama32_rmsnorm_f32", &rms_norm)));
    ProgramRef prefill, decode;
    ASSERT_TRUE(
        Ok(DefineLlamaStep(model, *module, rms_norm, LlamaStepKind::kPrefill,
                           "authored.prefill", &prefill)));
    ASSERT_TRUE(
        Ok(DefineLlamaStep(model, *module, rms_norm, LlamaStepKind::kDecode,
                           "authored.decode", &decode)));
    loomc_module_t *output = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(module->LinkAndVerify({prefill, decode}, &output, &result)));
    Owner<loomc_module_t, loomc_module_release> artifact(output);
    Owner<loomc_result_t, loomc_result_release> linked(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    for (auto format :
         {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t options = {};
      options.format = format;
      loomc_source_t *source = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          output, &options, loomc_allocator_system(), &source)));
      Owner<loomc_source_t, loomc_source_release> serialized(source);
      if (format == LOOMC_SOURCE_FORMAT_TEXT) {
        auto b = loomc_source_contents(source);
        std::string text(reinterpret_cast<const char *>(b.data), b.data_length);
        EXPECT_EQ(text.find("command.program.launch"), std::string::npos);
        for (uint32_t i = 0; i < 28; ++i)
          EXPECT_NE(text.find("blk." + std::to_string(i) + ".attn_q.weight"),
                    std::string::npos);
        EXPECT_NE(text.find("llm.ffn.gate_up_prefill"), std::string::npos);
        EXPECT_NE(text.find("llm.ffn.gate_up_decode"), std::string::npos);
        const std::string launch = authored
                                       ? "kernel.launch @authored.rmsnorm"
                                       : "kernel.launch @llama32_rmsnorm_f32";
        size_t count = 0;
        for (size_t position = 0;
             (position = text.find(launch, position)) != std::string::npos;
             position += launch.size())
          ++count;
        EXPECT_EQ(count,
                  114u); // two per layer plus output, for prefill and decode.
        if (authored) {
          EXPECT_EQ(text.find("kernel.launch @llama32_rmsnorm_f32"),
                    std::string::npos);
          EXPECT_NE(text.find("template.apply<@llm.rmsnorm.row>"),
                    std::string::npos);
        }
      }
      loomc_module_t *reloaded = nullptr;
      loomc_result_t *reload_result = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context.get(), workspace.get(), source, nullptr,
          loomc_allocator_system(), &reloaded, &reload_result)));
      Owner<loomc_module_t, loomc_module_release> reload(reloaded);
      Owner<loomc_result_t, loomc_result_release> reload_status(reload_result);
      ASSERT_TRUE(loomc_result_succeeded(reload_result));
    }
  }
}
TEST_F(LlamaStepTest, RejectsUnsupportedRoleStorageAndExtentBeforeMutation) {
  for (int mutation = 0; mutation < 4; ++mutation) {
    auto model = Descriptor();
    if (mutation == 0)
      model.embedding_width = 4096;
    if (mutation == 1)
      model.layers[17].tensors[2].tensor.storage_type = 14;
    if (mutation == 2)
      model.globals[0].tensor.byte_size--;
    if (mutation == 3)
      model.layers[0].tensors.push_back(model.layers[0].tensors.front());
    ProgramRef bad;
    auto status = DefineLlamaStep(model, *module, rms_norm,
                                  LlamaStepKind::kPrefill, "bad", &bad);
    EXPECT_EQ(loomc_status_consume_code(status), LOOMC_STATUS_INVALID_ARGUMENT);
  }
  ProgramRef valid;
  EXPECT_TRUE(Ok(DefineLlamaStep(Descriptor(), *module, rms_norm,
                                 LlamaStepKind::kPrefill, "valid", &valid)));
}
TEST_F(LlamaStepTest, RejectsInvalidNormBeforeCreatingRecipeBody) {
  KernelRef wrong_count, wrong_type, foreign;
  ASSERT_TRUE(Ok(module->Kernel("llama32_finish_isolated", &wrong_count)));
  const auto index = module->Scalar(LOOM_SCALAR_TYPE_INDEX);
  const auto buffer = module->Buffer();
  ASSERT_TRUE(Ok(module->DeclareKernel(
      "wrong_type", {index},
      {index, buffer, buffer, module->Scalar(LOOM_SCALAR_TYPE_F32)},
      &wrong_type)));
  std::unique_ptr<ModuleBuilder> other;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context.get(), workspace.get(),
                                       definitions.get(), &other)));
  ASSERT_TRUE(Ok(other->Kernel("llama32_rmsnorm_f32", &foreign)));
  for (auto invalid : {KernelRef{}, wrong_count, wrong_type, foreign}) {
    ProgramRef bad;
    EXPECT_EQ(loomc_status_consume_code(
                  DefineLlamaStep(Descriptor(), *module, invalid,
                                  LlamaStepKind::kPrefill, "same_name", &bad)),
              LOOMC_STATUS_INVALID_ARGUMENT);
  }
  // Reuse the attempted name and finalize the same module: rejected refs left
  // neither a reserved symbol nor an unfinished body behind.
  ProgramRef valid;
  ASSERT_TRUE(
      Ok(DefineLlamaStep(Descriptor(), *module, rms_norm,
                         LlamaStepKind::kPrefill, "same_name", &valid)));
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(module->LinkAndVerify({valid}, &output, &result)));
  Owner<loomc_module_t, loomc_module_release> artifact(output);
  Owner<loomc_result_t, loomc_result_release> linked(result);
  ASSERT_TRUE(loomc_result_succeeded(result));
}
} // namespace
} // namespace hrx::loomify_gguf
