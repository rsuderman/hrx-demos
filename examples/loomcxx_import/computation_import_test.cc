// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include <memory>
#include <string>

#include "loomc/module.h"
#include "examples/loomcxx_import/authored_program_data.h"
#include "gtest/gtest.h"

namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;

bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}

bool Succeeded(const loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result) {
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto diagnostic = loomc_result_diagnostic_at(result, i);
      ADD_FAILURE() << std::string(diagnostic->message.data,
                                   diagnostic->message.size);
    }
  }
  return false;
}

TEST(ComputationImport,
     AuthoredHelperAndDirectInvocationSurviveModuleRoundtrip) {
  loomc_context_t *context = nullptr;
  ASSERT_TRUE(
      Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)));
  Owner<loomc_context_t, loomc_context_release> context_owner(context);
  loomc_workspace_t *workspace = nullptr;
  ASSERT_TRUE(Ok(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_owner(workspace);
  const auto *data = authored_program_data_create();
  ASSERT_EQ(authored_program_data_size(), 1u);
  loomc_source_options_t source_options = {};
  source_options.identifier =
      loomc_make_cstring_view("authored_program.loombc");
  source_options.contents = loomc_make_byte_span(data[0].data, data[0].size);
  source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
  loomc_source_t *source = nullptr;
  ASSERT_TRUE(Ok(
      loomc_source_create(&source_options, loomc_allocator_system(), &source)));
  Owner<loomc_source_t, loomc_source_release> source_owner(source);
  loomc_module_t *module = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result)));
  Owner<loomc_module_t, loomc_module_release> module_owner(module);
  Owner<loomc_result_t, loomc_result_release> load_result(result);
  ASSERT_TRUE(Succeeded(result));
  source_owner.reset();
  loomc_source_t *text = nullptr;
  ASSERT_TRUE(Ok(loomc_module_serialize_text_to_source(
      module, nullptr, loomc_allocator_system(), &text)));
  Owner<loomc_source_t, loomc_source_release> text_owner(text);
  auto bytes = loomc_source_contents(text);
  const std::string ir(reinterpret_cast<const char *>(bytes.data),
                       bytes.data_length);
  EXPECT_NE(ir.find("@motif.affine"), std::string::npos);
  EXPECT_NE(ir.find("func.call pure @motif.affine"), std::string::npos);
  EXPECT_NE(ir.find("kernel.launch @authored.affine"), std::string::npos);
  EXPECT_NE(ir.find("@example.authored_affine"), std::string::npos);
  EXPECT_EQ(ir.find("command.program.launch"), std::string::npos);
  EXPECT_EQ(ir.find("kernel.decl"), std::string::npos);
  EXPECT_EQ(ir.find("func.decl"), std::string::npos);
  module_owner.reset();
  module = nullptr;
  result = nullptr;
  ASSERT_TRUE(Ok(loomc_module_deserialize_text_from_source(
      context, workspace, text, nullptr, loomc_allocator_system(), &module,
      &result)));
  Owner<loomc_module_t, loomc_module_release> text_module_owner(module);
  Owner<loomc_result_t, loomc_result_release> text_result(result);
  ASSERT_TRUE(Succeeded(result));
}
} // namespace
