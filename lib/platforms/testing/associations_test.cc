// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <memory>
#include <regex>
#include <string>
namespace {
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto text = loomc_status_message(status);
    ADD_FAILURE() << std::string(text.data, text.size);
  }
  loomc_status_free(status);
  return ok;
}
std::string Escape(const std::string &value) {
  return std::regex_replace(value, std::regex(R"([.^$|()\[\]{}*+?\\])"),
                            R"(\$&)");
}
struct Association {
  const char *kernel;
  const char *target;
  int wave;
};
const Association kAssociations[] = {
    {"ggml_linear_q4k_q8_1_x4", "projection.q4k.canonical.wave32", 32},
    {"ggml_linear_q4k_q8_1_x4_compat", "projection.q4k.compatible.wave32", 32},
    {"ggml_linear_q5k_q8_1_x4", "projection.q5k.wave32", 32},
    {"ggml_linear_q6k_q8_1_x4_compat", "projection.q6k.compatible.wave32", 32},
    {"ggml_linear_q5_0_q8_1_x4", "projection.q5_0.wave32", 32},
    {"ggml_linear_q4_0_q8_1_x4", "projection.q4_0.wave32", 32},
    {"ggml_linear_q4_1_q8_1_x4", "projection.q4_1.wave32", 32},
    {"ggml_linear_iq4_xs_q8_1_x4", "projection.iq4_xs.wave32", 32},
    {"ggml_linear_q8_0_q8_1_x4", "projection.q8_0.wave32", 32},
    {"qwen38_greedy_argmax_f32", "selection.greedy.wave64", 64},
    {"qwen38_greedy_argmax_rows_f32", "selection.greedy.wave64", 64},
    {"qwen38_greedy_argmax_partial_rows_f32", "selection.greedy.wave64", 64},
    {"qwen38_greedy_argmax_reduce_rows_f32", "selection.greedy.wave64", 64},
    {"llama32_rotary_query", "attention.llama_prepare.wave32", 32},
    {"llama32_publish_kv", "attention.llama_prepare.wave32", 32},
    {"llama32_attention_rows", "attention.llama_rows.wave32", 32},
    {"llama32_attention_residual", "attention.llama_rows.wave32", 32},
    {"qwen38_rmsnorm_f32_decode", "normalization.decode.wave32", 32},
    {"qwen38_rmsnorm_f32_prefill_last", "normalization.decode.wave32", 32},
    {"qwen38_output_normalize_spans", "normalization.output_spans.wave32", 32},
    {"qwen38_attention_prefill_f16_wmma", "attention.h256_prefill.wave64", 64},
    {"qwen38_attention_spans_f16_wmma", "attention.h256_spans.wave64", 64},
};
TEST(PlatformAssociations, SelectedSourceAndBytecodeKeepEveryAuthoredBinding) {
  loomc_target_environment_t *e = nullptr;
  ASSERT_TRUE(
      Ok(loomc_target_environment_create_amdgpu(loomc_allocator_system(), &e)));
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment(e, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.target_environment = e;
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.structure_size = sizeof(target);
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *c = nullptr;
  ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(), &c)));
  Owner<loomc_context_t, loomc_context_release> context(c,
                                                        loomc_context_release);
  loomc_workspace_t *w = nullptr;
  ASSERT_TRUE(
      Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w)));
  Owner<loomc_workspace_t, loomc_workspace_release> workspace(
      w, loomc_workspace_release);
  for (const char *filename : {"authored_text.loom", "authored_bc.loombc"}) {
    SCOPED_TRACE(filename);
    const auto path = std::string(std::getenv("TEST_SRCDIR")) + "/" +
                      std::getenv("TEST_WORKSPACE") +
                      "/lib/platforms/testing/" + filename;
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_path(
        c, w, loomc_make_cstring_view(path.c_str()), nullptr,
        loomc_allocator_system(), &m, &r)));
    Owner<loomc_module_t, loomc_module_release> module(m, loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    ASSERT_TRUE(result && loomc_result_succeeded(result.get()));
    loomc_module_serialize_options_t text_options{};
    text_options.format = LOOMC_SOURCE_FORMAT_TEXT;
    loomc_source_t *s = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
        m, &text_options, loomc_allocator_system(), &s)));
    Owner<loomc_source_t, loomc_source_release> source(s, loomc_source_release);
    const auto contents = loomc_source_contents(s);
    const std::string text(reinterpret_cast<const char *>(contents.data),
                           contents.data_length);
    for (const auto &association : kAssociations) {
      SCOPED_TRACE(association.kernel);
      const std::string symbol =
          std::string("llm.platform.") + association.target;
      const std::regex definition(
          "amdgpu\\.target<gfx11-generic>\\s+@" + Escape(symbol) +
          "\\s*\\{[^}]*subgroup_size\\s*=\\s*" +
          std::to_string(association.wave) + "[^}]*\\}");
      EXPECT_TRUE(std::regex_search(text, definition)) << symbol;
      const std::regex kernel("kernel\\.def[^\\n]*@" +
                              Escape(association.kernel) + "\\(");
      std::smatch header;
      ASSERT_TRUE(std::regex_search(text, header, kernel));
      EXPECT_NE(header.str().find("target(@" + symbol + ")"), std::string::npos)
          << header.str();
      EXPECT_EQ(
          std::distance(std::sregex_iterator(text.begin(), text.end(), kernel),
                        std::sregex_iterator()),
          1);
    }
  }
}
} // namespace
