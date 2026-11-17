// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Pure generated input/metadata hooks. Actual host acceptance uses the public
// pinned text_request_check process in check_reference.py, not a private
// ChatVM.
#include "lib/loomify/template_probe_runner.h"
#include "iree/base/internal/json.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hrx::loomify_gguf::testing {
namespace {
iree_string_view_t View(std::string_view text) {
  return iree_make_string_view(text.data(), text.size());
}
bool Ok(iree_status_t status) {
  if (iree_status_is_ok(status))
    return true;
  char message[4096] = {};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(message), message, &length);
  iree_status_ignore(status);
  ADD_FAILURE() << message;
  return false;
}
std::string Read(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input) << path;
  return std::string(std::istreambuf_iterator<char>(input), {});
}
std::string Env(const char *name) {
  const char *value = std::getenv(name);
  EXPECT_TRUE(value && *value) << name;
  return value ? value : "";
}
iree_string_view_t Field(iree_string_view_t object, const char *key) {
  iree_string_view_t value{};
  EXPECT_TRUE(Ok(iree_json_try_lookup_object_value(
      object, iree_make_cstring_view(key), &value)));
  return value;
}
std::string String(iree_string_view_t object, const char *key) {
  const auto value = Field(object, key);
  EXPECT_NE(value.data, nullptr) << key;
  std::string text(value.size, '\0');
  iree_host_size_t length = 0;
  EXPECT_TRUE(
      Ok(iree_json_unescape_string(value, text.size(), text.data(), &length)));
  text.resize(length);
  return text;
}
std::string Quote(std::string_view text) {
  constexpr char hex[] = "0123456789abcdef";
  std::string quoted = "\"";
  for (unsigned char c : text) {
    if (c == '\\' || c == '"')
      quoted += '\\';
    if (c < 32) {
      quoted += "\\u00";
      quoted += hex[c >> 4];
      quoted += hex[c & 15];
    } else
      quoted += static_cast<char>(c);
  }
  return quoted + '"';
}
iree_status_t Row(void *data, iree_host_size_t, iree_json_value_type_t type,
                  iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  static_cast<std::vector<iree_string_view_t> *>(data)->push_back(value);
  return iree_ok_status();
}
std::vector<int32_t> Encode(iree_tokenizer_t *tokenizer,
                            std::string_view text) {
  std::vector<int32_t> ids(text.size() + 16);
  iree_host_size_t count = 0;
  EXPECT_TRUE(Ok(iree_tokenizer_encode(
      tokenizer, View(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(ids.data(), nullptr, nullptr,
                                       ids.size()),
      iree_allocator_system(), &count)));
  ids.resize(count);
  return ids;
}
std::string Bytes(const std::vector<int32_t> &ids) {
  std::string result;
  for (uint32_t id : ids)
    for (int byte = 0; byte < 4; ++byte)
      result += static_cast<char>(id >> (8 * byte));
  return result;
}

const char *kInputWrapper = R"(
func.def @test.input_number(%request: buffer, %key: buffer) -> (i64) {
  %raw, %kind, %found = func.call @text.lookup(%request, %key) : (buffer, buffer) -> (buffer, i64, i1)
  func.call @text.output_check(%found) : (i1)
  %value = func.call @jinja.tointeger(%raw, %kind) : (buffer, i64) -> (i64)
  func.return %value : i64
}
func.def public @test.prepared_input(%request: buffer) -> (buffer) {
  %text_key = buffer.constant utf8("text") : buffer
  %text, %found = func.call @text.string(%request, %text_key) : (buffer, buffer) -> (buffer, i1)
  func.call @text.output_check(%found) : (i1)
  %format_key = buffer.constant utf8("format") : buffer
  %boundary_key = buffer.constant utf8("boundary") : buffer
  %capacity_key = buffer.constant utf8("capacity") : buffer
  %format64 = func.call @test.input_number(%request, %format_key) : (buffer, buffer) -> (i64)
  %boundary64 = func.call @test.input_number(%request, %boundary_key) : (buffer, buffer) -> (i64)
  %capacity = func.call @test.input_number(%request, %capacity_key) : (buffer, buffer) -> (i64)
  %format = scalar.trunci %format64 : i64 to i32
  %boundary = scalar.trunci %boundary64 : i64 to i32
  %zero64 = scalar.constant 0 : i64
  %empty = buffer.constant utf8("") : buffer
  %tokens, %count, %media = func.call @prepare_input(%text, %empty, %zero64, %format, %boundary, %capacity) : (buffer, buffer, i64, i32, i32, i64) -> (buffer, i64, buffer)
  %four = scalar.constant 4 : i64
  %bytes = scalar.muli %count, %four : i64
  %live = func.call @text.range(%tokens, %zero64, %bytes) : (buffer, i64, i64) -> (buffer)
  func.return %live : buffer
}
)";
const char *kMetadataWrapper = R"(
func.def public @test.metadata(%unused: buffer) -> (i64, i64) {
  %enabled, %mask, %default = func.call @chat_reasoning() : () -> (i32, i32, i32)
  %enabled64 = scalar.extui %enabled : i32 to i64
  %mask64 = scalar.extui %mask : i32 to i64
  %default64 = scalar.extui %default : i32 to i64
  %two = scalar.constant 2 : i64
  %scaled = scalar.muli %mask64, %two : i64
  %facts = scalar.addi %scaled, %enabled64 : i64
  func.return %facts, %default64 : i64, i64
}
)";

class GeneratedPackageFactsTest : public ::testing::TestWithParam<bool> {};
TEST_P(GeneratedPackageFactsTest, InputModesCapacityAndSourceMetadata) {
  const std::string manifest = Read(Env("LOOMIFY_GENERATED_ADAPTER_MANIFEST"));
  std::vector<iree_string_view_t> entries;
  ASSERT_TRUE(
      Ok(iree_json_enumerate_array_typed(View(manifest), Row, &entries)));
  ASSERT_FALSE(entries.empty());
  const auto root =
      (std::filesystem::path(Env("TEST_SRCDIR")) / Env("TEST_WORKSPACE"))
          .string();
  for (auto entry : entries) {
    SCOPED_TRACE(String(entry, "id"));
    // Always wrap the actual emitted text product; PreparedTemplate then
    // serializes that source in the selected SOURCE/bytecode form.
    const auto source = Read(String(entry, "source_output"));
    const auto json = Read(String(entry, "tokenizer"));
    iree_tokenizer_t *raw = nullptr;
    ASSERT_TRUE(Ok(iree_tokenizer_from_huggingface_json(
        View(json), iree_allocator_system(), &raw)));
    std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> tokenizer(
        raw, iree_tokenizer_free);
    jinja::GeneratedInputRenderer input{
        source + kInputWrapper, "test.prepared_input", {}};
    std::unique_ptr<jinja::PreparedTemplate> prepared;
    std::string error;
    ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
        input, root, Env("TEST_TMPDIR"), GetParam(), &prepared, &error,
        tokenizer.get()))
        << error;
    const auto facts = Field(entry, "input_policy");
    ASSERT_NE(facts.data, nullptr);
    for (int format : {0, 1})
      for (int boundary : {0, 1, 2}) {
        SCOPED_TRACE("format=" + std::to_string(format) +
                     " boundary=" + std::to_string(boundary));
        const auto text = String(facts, format ? "user_input" : "raw_input");
        const auto separator = boundary == 1 ? String(facts, "open_separator")
                               : boundary == 2
                                   ? String(facts, "terminated_separator")
                                   : "";
        const auto expected_text =
            !format     ? separator + text
            : !boundary ? String(facts, "fresh_user_reference")
                        : separator + String(facts, "framed_user_turn") +
                              String(facts, "continued_assistant_suffix");
        const auto ids = Encode(tokenizer.get(), expected_text);
        ASSERT_FALSE(ids.empty());
        auto request = [&](size_t capacity) {
          return "{\"text\":" + Quote(text) +
                 ",\"format\":" + std::to_string(format) +
                 ",\"boundary\":" + std::to_string(boundary) +
                 ",\"capacity\":" + std::to_string(capacity) + "}";
        };
        std::string actual;
        ASSERT_TRUE(prepared->Render(request(ids.size()), &actual, &error))
            << error;
        EXPECT_EQ(actual, Bytes(ids));
        actual = "preserved";
        EXPECT_FALSE(
            prepared->Render(request(ids.size() - 1), &actual, &error));
        EXPECT_EQ(actual, "preserved");
        EXPECT_NE(error.find("token capacity"), std::string::npos) << error;
        ASSERT_TRUE(prepared->Render(request(ids.size()), &actual, &error))
            << error;
        EXPECT_EQ(actual, Bytes(ids));
      }
    jinja::GeneratedInputRenderer metadata{
        source + kMetadataWrapper, "test.metadata", {}};
    ASSERT_TRUE(jinja::PreparedTemplate::Prepare(
        metadata, root, Env("TEST_TMPDIR"), GetParam(), &prepared, &error))
        << error;
    int64_t values = -1, default_effort = -1;
    ASSERT_TRUE(prepared->SelectFormat("{}", &values, &default_effort, &error))
        << error;
    const auto enabled = Field(entry, "supports_reasoning");
    ASSERT_NE(enabled.data, nullptr);
    EXPECT_EQ(values & 1,
              std::string_view(enabled.data, enabled.size) == "true");
    for (auto [name, actual] : std::vector<std::pair<const char *, int64_t>>{
             {"expected_reasoning_efforts", values / 2},
             {"expected_default_effort", default_effort}}) {
      const auto expected = Field(entry, name);
      if (!expected.data)
        continue;
      int64_t number = -1;
      ASSERT_TRUE(Ok(iree_json_parse_int64(expected, &number)));
      EXPECT_EQ(actual, number) << name;
    }
  }
}
INSTANTIATE_TEST_SUITE_P(SourceAndBytecode, GeneratedPackageFactsTest,
                         ::testing::Bool());
} // namespace
} // namespace hrx::loomify_gguf::testing
