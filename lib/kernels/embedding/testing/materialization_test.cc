// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Exercise admission through actual independent SOURCE/bytecode products.
#include "gtest/gtest.h"
#include "lib/loomify/linking.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/status.h"
#include "loomc/target/amdgpu.h"
#include "loomc/workspace.h"
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
std::string Message(loomc_status_t status) {
  const auto v = loomc_status_message(status);
  std::string text(v.data ? v.data : "", v.size);
  loomc_status_free(status);
  return text;
}
std::string Runfile(const std::string &path) {
  const char *dir = std::getenv("TEST_SRCDIR");
  const char *workspace = std::getenv("TEST_WORKSPACE");
  return std::string(dir ? dir : ".") + "/" +
         (workspace ? workspace : "_main") + "/" + path;
}
struct Input {
  const char *name;
  const char *mode = "gather";
  int64_t rows = 3;
  int64_t hidden = 33;
  int64_t vocabulary = 3;
  int64_t tag = 1;
  int64_t bytes = 198;
  bool expected = true;
};
std::vector<Input> Inputs() {
  return {
      {"dense_odd33"},
      {"dense_odd257", "gather", 3, 257, 3, 1, 1542},
      {"dense_minimum", "gather", 1, 32, 1, 1, 64},
      {"dense_maximum", "gather", 512, 32768, 2147483648LL, 1,
       140737488355328LL},
      {"actual_llama_root", "gather", 3, 3072, 128256, 1, 788004864},
      {"ordinary", "ordinary", 3, 256, 3, 1, 1536},
      {"epoch", "epoch", 513, 256, 3, 1, 1536},
      {"epoch_maximum", "epoch", 2048, 32768, 2147483648LL, 1,
       140737488355328LL},
      {"short_bytes", "gather", 3, 33, 3, 1, 197, false},
      {"long_bytes", "gather", 3, 33, 3, 1, 199, false},
      {"negative_bytes", "gather", 3, 33, 3, 1, -1, false},
      {"unsupported_tag", "gather", 3, 256, 3, 99, 1536, false},
      {"quantized_odd_unchanged", "gather", 3, 33, 3, 8, 102, false},
      {"hidden_below", "gather", 3, 31, 3, 1, 186, false},
      {"hidden_above", "gather", 3, 32769, 3, 1, 196614, false},
      {"hidden_overflow", "gather", 3, 4611686018427387904LL, 3, 1, 0, false},
      {"vocabulary_overflow", "gather", 3, 32768, 9223372036854775807LL, 1, 0,
       false},
      {"vocabulary_above", "gather", 3, 32, 2147483649LL, 1, 137438953536LL,
       false},
      {"vocabulary_zero", "gather", 3, 32, 0, 1, 0, false},
      {"dense_rows_zero", "gather", 0, 32, 3, 1, 192, false},
      {"dense_rows_above", "gather", 513, 32, 3, 1, 192, false},
      {"ordinary_odd_unchanged", "ordinary", 3, 33, 3, 1, 198, false},
      {"ordinary_vocabulary_unchanged", "ordinary", 3, 32, 262145, 1, 16777280,
       false},
      {"ordinary_rows_above", "ordinary", 513, 32, 3, 1, 192, false},
      {"epoch_rows_above", "epoch", 2049, 32, 3, 1, 192, false},
      {"epoch_bad_bytes", "epoch", 3, 256, 3, 1, 1535, false},
      {"ordinary_bad_bytes", "ordinary", 3, 256, 3, 1, 1537, false},
  };
}
using Case = std::tuple<Input, bool>;
class EmbeddingMaterializationTest : public ::testing::TestWithParam<Case> {};
TEST_P(EmbeddingMaterializationTest, ActualGatherContracts) {
  const auto &v = std::get<0>(GetParam());
  const bool bytecode = std::get<1>(GetParam());
  SCOPED_TRACE(v.name);
  loomc_target_environment_t *raw_environment = nullptr;
  auto status = loomc_target_environment_create_amdgpu(loomc_allocator_system(),
                                                       &raw_environment);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment(raw_environment, loomc_target_environment_release);
  loomc_context_target_options_t target{};
  target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target.target_environment = environment.get();
  loomc_context_options_t options{};
  options.next = &target;
  loomc_context_t *raw_context = nullptr;
  status =
      loomc_context_create(&options, loomc_allocator_system(), &raw_context);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_context_t, loomc_context_release> context(raw_context,
                                                        loomc_context_release);
  loomc_workspace_t *raw_workspace = nullptr;
  status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &raw_workspace);
  ASSERT_TRUE(loomc_status_is_ok(status)) << Message(status);
  Owner<loomc_workspace_t, loomc_workspace_release> workspace(
      raw_workspace, loomc_workspace_release);
  const std::string mode(v.mode);
  const bool ordinary = mode == "ordinary";
  const bool dense = mode == "gather";
  std::ostringstream source;
  source
      << "template.decl @llm.embedding." << mode
      << "_f32(%R:index,%H:index,%V:index,%tag:index,%B:index,"
      << (ordinary ? "%origin:offset," : "")
      << (dense ? "" : "%epoch:buffer,%count:buffer,")
      << "%ids:buffer,%weight:view<[%B]xi8>,%output:view<[%R]x[%H]xf32>)\n"
      << "command.def public @probe.root() launch(%epoch:buffer,%count:buffer,%ids:buffer,%weight:buffer,%output:buffer) {\n";
  for (const auto &[name, value] :
       std::vector<std::pair<const char *, int64_t>>{{"R", v.rows},
                                                     {"H", v.hidden},
                                                     {"V", v.vocabulary},
                                                     {"tag", v.tag},
                                                     {"B", v.bytes}})
    source << "  %" << name << " = index.constant " << value << " : index\n";
  source
      << "  %origin = index.constant 0 : offset\n"
      << "  %w = buffer.view %weight[%origin] : buffer -> view<[%B]xi8>\n"
      << "  %o = buffer.view %output[%origin] : buffer -> view<[%R]x[%H]xf32>\n"
      << "  template.apply<@llm.embedding." << mode << "_f32>(%R,%H,%V,%tag,%B,"
      << (ordinary ? "%origin," : "") << (dense ? "" : "%epoch,%count,")
      << "%ids,%w,%o) : (index,index,index,index,index,"
      << (ordinary ? "offset," : "") << (dense ? "" : "buffer,buffer,")
      << "buffer,view<[%B]xi8>,view<[%R]x[%H]xf32>)\n"
      << "  command.return\n}\n";
  const std::string product =
      Runfile("lib/kernels/embedding/testing/contract_provider_" +
              std::string(bytecode ? "bc.loombc" : "text.loom"));
  std::string error;
  loomc_module_t *raw_module = nullptr;
  const bool succeeded = hrx::loomify_gguf::LinkConfiguredProducts(
      context.get(), workspace.get(), source.str(), {product}, {"probe.root"},
      true, &raw_module, &error);
  Owner<loomc_module_t, loomc_module_release> module(raw_module,
                                                     loomc_module_release);
  EXPECT_EQ(succeeded, v.expected) << error;
  if (!v.expected) {
    EXPECT_EQ(raw_module, nullptr);
    EXPECT_FALSE(error.empty());
  }
}
INSTANTIATE_TEST_SUITE_P(ActualOperations, EmbeddingMaterializationTest,
                         ::testing::Combine(::testing::ValuesIn(Inputs()),
                                            ::testing::Bool()));
} // namespace
