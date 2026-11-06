// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/module.h"
#include "loomc/source.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace hrx::llama {
namespace {
::testing::AssertionResult Ok(loomc_status_t status) {
  const auto message = loomc_status_message(status);
  const std::string text(message.data ? message.data : "", message.size);
  const bool ok = loomc_status_is_ok(status);
  loomc_status_free(status);
  return ok ? ::testing::AssertionSuccess()
            : ::testing::AssertionFailure() << text;
}
std::string Diagnostics(loomc_result_t *result) {
  if (!result)
    return "no native result";
  std::string text;
  for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto *diagnostic = loomc_result_diagnostic_at(result, i);
    text.append(diagnostic->message.data, diagnostic->message.size);
    text += '\n';
  }
  return text;
}

// This is a relocatable-source check: numerical execution and configured strict
// linking are separate gates. In particular no fake FFN/math provider is linked.
TEST(LayerSourceTest, MaintainedSourcesReloadAsTextAndBytecode) {
  loomc_context_t *context_raw = nullptr;
  ASSERT_TRUE(Ok(loomc_context_create(nullptr, loomc_allocator_system(),
                                    &context_raw)));
  std::unique_ptr<loomc_context_t, decltype(&loomc_context_release)> context(
      context_raw, loomc_context_release);
  loomc_workspace_t *workspace_raw = nullptr;
  ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                      &workspace_raw)));
  std::unique_ptr<loomc_workspace_t, decltype(&loomc_workspace_release)> workspace(
      workspace_raw, loomc_workspace_release);
  ASSERT_NE(std::getenv("TEST_SRCDIR"), nullptr);
  ASSERT_NE(std::getenv("TEST_WORKSPACE"), nullptr);
  const std::string root = std::string(std::getenv("TEST_SRCDIR")) + "/" +
                           std::getenv("TEST_WORKSPACE");
  for (const char *file : {"attention.loom", "layer.loom"}) {
    SCOPED_TRACE(file);
    const auto path = root + "/lib/llama/programs/" + file;
    std::ifstream input(path);
    ASSERT_TRUE(input.is_open()) << path;
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    loomc_source_options_t options{};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.identifier = loomc_make_cstring_view(path.c_str());
    options.contents = loomc_make_byte_span(text.data(), text.size());
    loomc_source_t *source_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_source_create(&options, loomc_allocator_system(),
                                     &source_raw)));
    std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)> source(
        source_raw, loomc_source_release);
    loomc_module_t *module_raw = nullptr;
    loomc_result_t *result_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context.get(), workspace.get(), source.get(), nullptr,
        loomc_allocator_system(), &module_raw, &result_raw)));
    std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)> result(
        result_raw, loomc_result_release);
    ASSERT_TRUE(result && loomc_result_succeeded(result.get()))
        << Diagnostics(result.get());
    std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> module(
        module_raw, loomc_module_release);
    for (const auto format : {LOOMC_SOURCE_FORMAT_TEXT,
                              LOOMC_SOURCE_FORMAT_BYTECODE}) {
      loomc_module_serialize_options_t serialization{};
      serialization.format = format;
      loomc_source_t *encoded_raw = nullptr;
      ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
          module.get(), &serialization, loomc_allocator_system(), &encoded_raw)));
      std::unique_ptr<loomc_source_t, decltype(&loomc_source_release)> encoded(
          encoded_raw, loomc_source_release);
      loomc_module_t *decoded_raw = nullptr;
      loomc_result_t *decoded_result_raw = nullptr;
      ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
          context.get(), workspace.get(), encoded.get(), nullptr,
          loomc_allocator_system(), &decoded_raw, &decoded_result_raw)));
      std::unique_ptr<loomc_result_t, decltype(&loomc_result_release)> decoded_result(
          decoded_result_raw, loomc_result_release);
      std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)> decoded(
          decoded_raw, loomc_module_release);
      ASSERT_TRUE(decoded_result && loomc_result_succeeded(decoded_result.get()))
          << Diagnostics(decoded_result.get());
      ASSERT_NE(decoded.get(), nullptr);
    }
  }
}
} // namespace
} // namespace hrx::llama
