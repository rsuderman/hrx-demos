// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
// Actual native chat transport + maintained providers versus independent pinned
// HF/Jinja bytes/tokens. Built in the SDK's allowed external test package; no
// repository target takes private SDK implementation dependencies.
#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "iree/vm/sync.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/json.h"
#include "loom/runtime/program/program.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/tool_xml.h"
#include <cstring>
#include "loom/serve/service/text/input.h"
#include "loomc/target/vm.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
namespace {
iree_string_view_t View(std::string_view s) {
  return iree_make_string_view(s.data(), s.size());
}
std::string Copy(iree_string_view_t s) {
  return s.size ? std::string(s.data, s.size) : std::string{};
}
std::string Diagnostic(iree_status_t status) {
  if (iree_status_is_ok(status)) return {};
  char message[4096] = {};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(message), message, &length);
  message[sizeof(message) - 1] = 0;
  iree_status_ignore(status);
  return message;
}
bool Ok(iree_status_t status) {
  if (iree_status_is_ok(status)) return true;
  ADD_FAILURE() << Diagnostic(status);
  return false;
}
bool Read(const std::string& path, std::string* out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) { ADD_FAILURE() << "cannot read " << path; return false; }
  out->assign(std::istreambuf_iterator<char>(input), {});
  if (input.bad()) { ADD_FAILURE() << "cannot finish reading " << path; return false; }
  return true;
}
std::string Env(const std::string& name) {
  const char* value = std::getenv(name.c_str());
  if (!value || !*value) { ADD_FAILURE() << "required explicit fixture path: " << name; return {}; }
  return value;
}
bool Field(iree_string_view_t object, const char* name, iree_string_view_t* out) {
  return Ok(iree_json_try_lookup_object_value(object, iree_make_cstring_view(name), out));
}
bool Decode(iree_string_view_t raw, std::string* out) {
  out->resize(raw.size);
  iree_host_size_t count = 0;
  if (!Ok(iree_json_unescape_string(raw, out->size(), out->data(), &count))) return false;
  out->resize(count);
  return true;
}
iree_status_t Row(void* data, iree_host_size_t, iree_json_value_type_t type,
                  iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "oracle rows must be objects");
  static_cast<std::vector<iree_string_view_t>*>(data)->push_back(value);
  return iree_ok_status();
}
iree_status_t Token(void* data, iree_host_size_t, iree_string_view_t value) {
  int64_t id = -1;
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &id));
  if (id < 0 || id > std::numeric_limits<int32_t>::max())
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "oracle token ID out of range");
  static_cast<std::vector<int32_t>*>(data)->push_back(static_cast<int32_t>(id));
  return iree_ok_status();
}
// Reference templates omit the HTTP route envelope. Bind that transport-only
// field to the actual loaded policy; it does not change template variables.
bool AddModelRoute(std::string* request, std::string_view route) {
  iree_string_view_t model{};
  if (!Field(View(*request), "model", &model)) return false;
  if (model.data) return true;
  // These fixture route identities are the asserted package aliases, not
  // arbitrary user input needing a new JSON serializer.
  if (route.empty() || route.find_first_of("\"\\") != std::string_view::npos) {
    ADD_FAILURE() << "invalid fixture route"; return false;
  }
  const auto close = request->rfind('}');
  if (close == std::string::npos) return false;
  request->insert(close, ",\"model\":\"" + std::string(route) + "\"");
  return true;
}
// A missing pure-renderer add_generation_prompt variable means no suffix.
// Serve's HTTP default is true, so map that explicit oracle condition to false
// in the existing model_options envelope; do not alter expected reference bytes.
bool TransportRequest(iree_string_view_t row, std::string_view route, std::string* request) {
  iree_string_view_t raw{}, kwargs{}, generation{};
  if (!Field(row, "request", &raw) || !Field(row, "kwargs", &kwargs) ||
      !Field(kwargs, "add_generation_prompt", &generation)) return false;
  *request = Copy(raw);
  if (generation.data) return AddModelRoute(request, route);
  iree_string_view_t options{};
  if (!Field(raw, "model_options", &options)) return false;
  if (!options.data) {
    const auto close = request->rfind('}');
    if (close == std::string::npos) return false;
    request->insert(close, ",\"model_options\":{\"add_generation_prompt\":false}");
  } else {
    // Narrow fixture adapter: preserve existing option bytes and insert only
    // the missing formatter variable. Fixture JSON has unique object keys.
    iree_string_view_t present{};
    if (!Field(options, "add_generation_prompt", &present)) return false;
    if (present.data) { ADD_FAILURE() << "oracle kwargs/request mismatch"; return false; }
    const auto offset = static_cast<size_t>(options.data - raw.data);
    const auto close = request->find_last_not_of(" \t\r\n", offset + options.size - 1);
    if (close == std::string::npos || (*request)[close] != '}') return false;
    const bool empty = Copy(options).find_first_not_of("{} \t\r\n") == std::string::npos;
    request->insert(close, std::string(empty ? "" : ",") + "\"add_generation_prompt\":false");
  }
  return AddModelRoute(request, route);
}
// Inspect current API-owned token storage without altering native IDs.
iree_status_t CopyPreparedTokenIds(iree_vm_buffer_t* storage, size_t count,
                                  size_t capacity, int32_t* destination) {
  if (count > capacity || (count && (!storage ||
      iree_vm_buffer_length(storage) / sizeof(int32_t) < count)))
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "invalid prepared token storage");
  if (count) std::memcpy(destination, iree_vm_buffer_const_data(storage), count * sizeof(int32_t));
  return iree_ok_status();
}
class ChatVM {
public:
  ~ChatVM() {
    if (policy_ready_) loom_serve_text_chat_policy_deinitialize(&policy_);
    loom_runtime_program_free(program_);
    iree_vm_module_release(tools_);
    iree_vm_module_release(input_);
    iree_vm_module_release(json_);
    iree_tokenizer_free(tokenizer_);
    iree_vm_environment_free(environment_);
    loomc_target_environment_release(target_);
  }
  bool Init(const std::string& product, const std::string& tokenizer_path,
            std::string_view expected_name) {
    auto status = loomc_target_environment_create_vm(loomc_allocator_system(), &target_);
    if (!loomc_status_is_ok(status)) {
      auto m = loomc_status_message(status); ADD_FAILURE() << std::string(m.data, m.size);
      loomc_status_free(status); return false;
    }
    loomc_status_free(status);
    std::string tokenizer_json;
    if (!Read(tokenizer_path, &tokenizer_json) ||
        !Ok(iree_tokenizer_from_huggingface_json(View(tokenizer_json), iree_allocator_system(), &tokenizer_)) ||
        !Ok(iree_vm_environment_allocate(iree_allocator_system(), &environment_)) ||
        !Ok(loom_runtime_json_module_create(environment_, iree_allocator_system(), &json_)) ||
        !Ok(loom_runtime_input_module_create(environment_, tokenizer_, iree_allocator_system(), &input_)) ||
        !Ok(loom_serve_text_tool_xml_module_create(environment_, iree_allocator_system(), &tools_))) return false;
    const iree_string_view_t roots[] = {
      IREE_SV("model_name"), IREE_SV("chat_reasoning"), IREE_SV("chat_image_limits"), IREE_SV("chat_begin"),
      IREE_SV("chat_message"), IREE_SV("chat_end"), IREE_SV("parse_tools"),
      IREE_SV("render_tool"), IREE_SV("prepare_input"), IREE_SV("output_next"),
      IREE_SV("complete_text"), IREE_SV("request_defaults"), IREE_SV("request_options")};
    iree_vm_module_t* libraries[] = {json_, input_, tools_};
    loom_runtime_sources_t sources{};
    loom_runtime_compilation_report_t report{};
    auto compiled = loom_runtime_program_allocate(target_, environment_, &sources,
        View(product), IREE_ARRAYSIZE(roots), roots,
        iree_vm_module_span_from_array(libraries), &report, iree_allocator_system(), &program_);
    if (!iree_status_is_ok(compiled)) {
      for (iree_host_size_t i = 0; i < report.diagnostic_count; ++i) {
        const auto& d = report.diagnostics[i];
        std::cerr.write(d.message, std::min<uint64_t>(d.message_length, sizeof(d.message))) << '\n';
      }
      return Ok(compiled);
    }
    iree_vm_buffer_t* options = nullptr;
    if (!Ok(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span("{}", 2), 1, iree_allocator_system(), &options))) return false;
    auto bound = loom_serve_text_chat_policy_initialize(environment_, program_, options, &policy_);
    iree_vm_buffer_release(options);
    if (!Ok(bound)) return false;
    policy_ready_ = true;
    EXPECT_EQ(Copy(policy_.name), expected_name);
    // Both current host flag values retain the provider's complete-response ABI.
    for (bool truncated : {false, true}) {
      std::string calls;
      if (!Ok(ParseTools("[]", "", &calls, truncated))) return false;
      EXPECT_EQ(calls, "[]");
    }
    return true;
  }
  bool SupportsReasoning() const { return policy_.reasoning.support == LOOM_SERVE_REASONING_SUPPORTED; }
  std::string_view name() const { return {policy_.name.data, policy_.name.size}; }
  iree_status_t RequestOptions(const std::string& body) {
    // Invoke the bound policy entry directly. Calling Render would also run
    // chat_begin, whose independent date lookup could mask lost validation.
    std::string request = body;
    if (!AddModelRoute(&request, name()))
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid fixture route envelope");
    iree_vm_buffer_t* input = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(request.data(), request.size()), 1,
        iree_allocator_system(), &input));
    iree_vm_variant_t args[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&policy_.types, policy_.defaults),
      iree_vm_buffer_variant_from_ptr_move(&policy_.types, &input),
      iree_vm_variant_from_i64(123)};
    iree_vm_variant_t result = {};
    auto status = iree_vm_invoke(policy_.invocation, policy_.request_options,
        iree_vm_variant_span_from_array(args), iree_vm_variant_span_from_ptr(&result, 1));
    iree_vm_buffer_t* options = nullptr;
    if (iree_status_is_ok(status))
      status = iree_vm_buffer_ptr_from_variant_borrowed(&policy_.types, result, &options);
    if (iree_status_is_ok(status) && !options)
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null request options");
    iree_vm_variant_reset(&result);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(args));
    return status;
  }
  iree_status_t Render(const std::string& body, loom_serve_text_chat_t* chat) {
    // Admission fixtures use the same real route envelope as reference rows.
    body_ = body;
    if (!AddModelRoute(&body_, name()))
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid fixture route envelope");
    return loom_serve_text_chat_initialize(&policy_, policy_.name, View(body_),
        123, 128, iree_allocator_system(), chat);
  }
  iree_status_t ParseTools(std::string_view schemas, std::string_view response,
                          std::string* out, bool truncated = false) {
    const std::string_view inputs[] = {schemas, response};
    iree_vm_variant_t args[3] = {}, result[1] = {};
    args[2] = iree_vm_variant_from_i64(truncated);
    auto status = iree_ok_status();
    for (size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      iree_vm_buffer_t* buffer = nullptr;
      status = iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
          iree_make_const_byte_span(inputs[i].data(), inputs[i].size()), 1,
          iree_allocator_system(), &buffer);
      if (iree_status_is_ok(status))
        args[i] = iree_vm_buffer_variant_from_ptr_move(&policy_.types, &buffer);
    }
    if (iree_status_is_ok(status))
      status = iree_vm_invoke(policy_.invocation, policy_.parse_tools,
          iree_vm_variant_span_from_array(args), iree_vm_variant_span_from_array(result));
    iree_vm_buffer_t* buffer = nullptr;
    if (iree_status_is_ok(status))
      status = iree_vm_buffer_ptr_from_variant_borrowed(&policy_.types, result[0], &buffer);
    if (iree_status_is_ok(status)) {
      if (!buffer) status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null parsed tools");
      else out->assign(static_cast<const char*>(iree_vm_buffer_const_data(buffer)),
                       iree_vm_buffer_length(buffer));
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(result));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(args));
    return status;
  }
  bool Encode(std::string_view text, std::vector<int32_t>* tokens) {
    tokens->resize(text.size() + 16);
    iree_host_size_t count = 0;
    if (!Ok(iree_tokenizer_encode(tokenizer_, View(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(tokens->data(), nullptr, nullptr, tokens->size()),
        iree_allocator_system(), &count))) return false;
    tokens->resize(count);
    return true;
  }
  bool Prepare(const loom_serve_text_chat_t& chat, size_t capacity,
               std::vector<int32_t>* tokens) {
    tokens->resize(capacity);
    iree_host_size_t count = 0, span_count = 0;
    loom_serve_text_media_span_t* spans = nullptr;
    iree_vm_buffer_t* token_storage = nullptr;
    auto status = loom_serve_text_chat_prepare_input(&chat, 0,
        LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
        capacity, 0, iree_allocator_system(), &token_storage, &count, &spans, &span_count);
    iree_allocator_free(iree_allocator_system(), spans);
    if (iree_status_is_ok(status)) status = CopyPreparedTokenIds(token_storage, count, capacity, tokens->data());
    iree_vm_buffer_release(token_storage);
    if (!Ok(status)) return false;
    EXPECT_EQ(span_count, 0u);
    tokens->resize(count);
    return true;
  }
private:
  loomc_target_environment_t* target_ = nullptr;
  iree_vm_environment_t* environment_ = nullptr;
  iree_tokenizer_t* tokenizer_ = nullptr;
  iree_vm_module_t* json_ = nullptr;
  iree_vm_module_t* input_ = nullptr;
  iree_vm_module_t* tools_ = nullptr;
  loom_runtime_program_t* program_ = nullptr;
  loom_serve_text_chat_policy_t policy_{};
  bool policy_ready_ = false;
  std::string body_; // Native chat borrows request bytes through deinitialize.
};
std::string ToolRequest(std::string_view escaped_arguments, std::string_view schemas = {}) {
  constexpr std::string_view default_schemas = R"([{"type":"function","function":{"name":"fetch","parameters":{"type":"object","properties":{"n":{"type":"number"},"nested":{"type":"object","properties":{"x":{"type":"number"}}},"text":{"type":"string"}}}}}])";
  if (schemas.empty()) schemas = default_schemas;
  return std::string(R"({"messages":[{"role":"user","content":"fetch"},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"fetch","arguments":")") +
      std::string(escaped_arguments) + R"("}}]}],"tools":)" + std::string(schemas) + "}";
}
void CheckAdmission(ChatVM& vm, std::string_view revision) {
  auto rejects = [&](std::string_view label, const std::string& request) {
    SCOPED_TRACE(std::string(label));
    // These are transport-valid JSON/JSONC inputs. Admission failures must
    // exercise the selected source contract, not a malformed outer fixture.
    auto cursor = View(request);
    iree_string_view_t parsed{};
    ASSERT_TRUE(Ok(iree_json_consume_value(&cursor, &parsed)));
    ASSERT_EQ(cursor.size, 0u);
    loom_serve_text_chat_t chat{};
    auto status = vm.Render(request, &chat);
    const bool accepted = iree_status_is_ok(status);
    const auto diagnostic = Diagnostic(status);
    EXPECT_FALSE(accepted) << "request unexpectedly admitted";
    if (accepted) loom_serve_text_chat_deinitialize(&chat);
  };
  rejects("duplicate literal argument keys", ToolRequest(R"({\"n\":1,\"n\":2})"));
  rejects("duplicate escaped-equivalent argument keys", ToolRequest(R"({\"n\":1,\"\\u006e\":2})"));
  rejects("nested duplicate argument keys", ToolRequest(R"({\"nested\":{\"x\":1,\"x\":2}})"));
  rejects("JSONC argument comment", ToolRequest(R"({\"n\":1/*comment*/})"));
  rejects("argument trailing comma", ToolRequest(R"({\"n\":1,})"));
  rejects("JSONC schema comment", ToolRequest(R"({})",
      R"([{"type":"function","function":{"name":"fetch","parameters":{"type":/*comment*/"object","properties":{}}}}])"));
  rejects("schema trailing comma", ToolRequest(R"({})",
      R"([{"type":"function","function":{"name":"fetch","parameters":{"type":"object","properties":{}}},}])"));
  rejects("unknown model option", R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"typo":true}})");
  rejects("wrong thinking boolean type", R"({"messages":[{"role":"user","content":"hello"}],"enable_thinking":"true"})");
  if (revision == "7f0e5290") {
    rejects("unsupported 0.8B effort", R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"reasoning_effort":"low"}})");
  } else {
    rejects("wrong preserve boolean type", R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"preserve_thinking":"false"}})");
    rejects("wrong effort type", R"({"messages":[{"role":"user","content":"hello"}],"model_options":{"reasoning_effort":true}})");
  }
  std::string plain_prompt;
  for (bool escaped : {false, true}) {
    const std::string body = escaped
        ? R"({"messages":[{"role":"u\u0073er","content":"Hello"}]})"
        : R"({"messages":[{"role":"user","content":"Hello"}]})";
    loom_serve_text_chat_t role_chat{};
    ASSERT_TRUE(Ok(vm.Render(body, &role_chat)));
    const auto prompt = Copy(iree_string_builder_view(&role_chat.prompt));
    if (escaped) EXPECT_EQ(prompt, plain_prompt);
    else plain_prompt = prompt;
    loom_serve_text_chat_deinitialize(&role_chat);
  }
  // Strings containing JSONC-looking punctuation remain ordinary string data.
  const auto request = ToolRequest(R"({\"text\":\"literal //, \\\",} café\"})");
  loom_serve_text_chat_t chat{};
  ASSERT_TRUE(Ok(vm.Render(request, &chat)));
  const auto prompt = Copy(iree_string_builder_view(&chat.prompt));
  EXPECT_NE(prompt.find(R"(literal //, ",} café)"), std::string::npos);
  loom_serve_text_chat_deinitialize(&chat);
}

// A raw caller supplies its own rendered bytes. Initialize ordinary native chat
// ownership first, then replace only its prompt buffer for this input fixture.
bool RawChat(ChatVM& vm, std::string_view text, loom_serve_text_chat_t* chat) {
  if (!Ok(vm.Render(R"({"messages":[{"role":"user","content":"Hello"}]})", chat))) return false;
  iree_string_builder_reset(&chat->prompt);
  if (!Ok(iree_string_builder_append_string(&chat->prompt, View(text)))) {
    loom_serve_text_chat_deinitialize(chat);
    return false;
  }
  chat->max_tokens = 1;
  return true;
}
iree_status_t PrepareMode(const loom_serve_text_chat_t& chat,
    loom_serve_text_chat_input_format_t format, loom_serve_text_chat_boundary_t boundary,
    size_t capacity, int32_t* tokens, iree_host_size_t* count) {
  loom_serve_text_media_span_t* spans = nullptr;
  iree_host_size_t span_count = 0;
  iree_vm_buffer_t* token_storage = nullptr;
  auto status = loom_serve_text_chat_prepare_input(&chat, 0, format, boundary,
      capacity, 0, iree_allocator_system(), &token_storage, count, &spans, &span_count);
  if (iree_status_is_ok(status)) status = CopyPreparedTokenIds(token_storage, *count, capacity, tokens);
  iree_vm_buffer_release(token_storage);
  iree_allocator_free(iree_allocator_system(), spans);
  EXPECT_EQ(span_count, 0u);
  return status;
}
void CheckInputPolicy(ChatVM& vm, std::string_view revision,
                      const std::string& one_user_reference) {
  const bool llama = revision == "5816fce1";
  const bool preserve = revision == "5da44855" || revision == "cd8e9439";
  auto fresh_reference = one_user_reference;
  if (preserve) fresh_reference.replace(fresh_reference.find("Hello"), 5, u8" \u2003Hello\u00a0 ");
  const std::string assistant = llama
      ? "<|start_header_id|>assistant<|end_header_id|>\n\n"
      : "<|im_start|>assistant\n";
  const auto generation = one_user_reference.rfind(assistant);
  ASSERT_NE(generation, std::string::npos);
  const auto suffix = one_user_reference.substr(generation);
  const std::string user_turn = llama
      ? "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
      : preserve ? u8"<|im_start|>user\n \u2003Hello\u00a0 <|im_end|>\n" : "<|im_start|>user\nHello<|im_end|>\n";
  const std::string raw = llama ? u8"<|begin_of_text|> \u2003Hello\u00a0 "
                                : u8"<|im_start|>assistant\n \u2003Hello\u00a0 ";
  auto separator = [&](loom_serve_text_chat_boundary_t boundary) {
    if (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN)
      return std::string(llama ? "<|eot_id|>" : "<|im_end|>\n");
    if (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED)
      return std::string(llama ? "" : "\n");
    return std::string{};
  };
  for (const auto format : {LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, LOOM_SERVE_TEXT_CHAT_INPUT_USER}) {
    for (const auto boundary : {LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
                               LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,
                               LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED}) {
      SCOPED_TRACE("input format=" + std::to_string(format) + " boundary=" + std::to_string(boundary));
      const bool user = format == LOOM_SERVE_TEXT_CHAT_INPUT_USER;
      const std::string text = user ? u8" \u2003Hello\u00a0 " : raw;
      const auto expected_text = user
          ? (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH
              ? fresh_reference : separator(boundary) + user_turn + suffix)
          : separator(boundary) + raw;
      std::vector<int32_t> expected;
      ASSERT_TRUE(vm.Encode(expected_text, &expected)); ASSERT_FALSE(expected.empty());
      loom_serve_text_chat_t chat{};
      ASSERT_TRUE(RawChat(vm, text, &chat));
      struct Cleanup { loom_serve_text_chat_t* chat; ~Cleanup() { loom_serve_text_chat_deinitialize(chat); } } cleanup{&chat};
      std::vector<int32_t> actual(expected.size(), -991);
      iree_host_size_t count = 0;
      ASSERT_TRUE(Ok(PrepareMode(chat, format, boundary, expected.size(), actual.data(), &count)));
      EXPECT_EQ(count, expected.size()); EXPECT_EQ(actual, expected);
      if (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH) {
        // Capacity+1 probing must reject truncation and leave output untouched.
        std::fill(actual.begin(), actual.end(), -991); count = 99;
        auto status = PrepareMode(chat, format, boundary, expected.size() - 1, actual.data(), &count);
        const bool accepted = iree_status_is_ok(status);
        Diagnostic(status);
        EXPECT_FALSE(accepted); EXPECT_EQ(count, 0u);
        EXPECT_EQ(actual, std::vector<int32_t>(expected.size(), -991));
      }
    }
  }
  for (const auto boundary : {LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,
                             LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED}) {
    SCOPED_TRACE("retained boundary=" + std::to_string(boundary));
    const std::string history = "HISTORY|", tail = "NEXT";
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(RawChat(vm, history + tail, &chat));
    struct Cleanup { loom_serve_text_chat_t* chat; ~Cleanup() { loom_serve_text_chat_deinitialize(chat); } } cleanup{&chat};
    loom_serve_text_chat_completion_t completion{};
    ASSERT_TRUE(Ok(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(history.data(), history.size()), 1,
        iree_allocator_system(), &completion.storage)));
    completion.transcript = iree_make_string_view(
        static_cast<const char*>(iree_vm_buffer_const_data(completion.storage)), history.size());
    completion.reusable = true;
    struct HistoryCleanup { loom_serve_text_chat_completion_t* value;
      ~HistoryCleanup() { loom_serve_text_chat_completion_deinitialize(value); } } history_cleanup{&completion};
    // OPEN preserves a deliberately unrelated pending token; TERMINATED uses
    // this known fixture's canonical EOS. Neither comes from transcript text.
    const int32_t anchor = boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN ? 7 : (llama ? 128009 : 248046);
    const loom_serve_text_input_prefix_t prefix{&completion, 3, anchor, boundary};
    std::vector<int32_t> expected;
    ASSERT_TRUE(vm.Encode(separator(boundary) + tail, &expected));
    expected.insert(expected.begin(), anchor);
    const auto capacity = prefix.position + expected.size();
    std::vector<int32_t> scratch(capacity, -991);
    loom_serve_text_input_t input{};
    struct InputCleanup { loom_serve_text_input_t* value;
      ~InputCleanup() { loom_serve_text_input_deinitialize(value, iree_allocator_system()); } } input_cleanup{&input};
    ASSERT_TRUE(Ok(loom_serve_text_input_prepare(&chat, prefix, true, capacity,
        &input, iree_allocator_system())));
    ASSERT_EQ(input.count, expected.size()); ASSERT_NE(input.tokens, nullptr);
    EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count), expected);
    EXPECT_EQ(input.tokens[0], anchor);
    EXPECT_EQ(input.history, completion.storage);
    EXPECT_EQ(input.retained_count, prefix.position);
  }
  std::cout << "INPUT_POLICY_COVERAGE " << revision
            << " modes=6 exact_capacity=6 overflow=2 actual_retained_boundaries=2\n";
}

void CheckPublicToolParser(ChatVM& vm, std::string_view revision) {
  const std::string schemas = R"([{"type":"function","function":{"name":"inspect","parameters":{"type":"object","properties":{"flag":{"type":"boolean"},"word":{"type":"string"},"nested":{"type":"object","properties":{"label":{"type":"string"},"z":{"type":"array","items":{"type":"boolean"}}},"required":["label","z"]}},"required":["flag","word","nested"]}}}])";
  auto xml = [](std::string_view flag) {
    return std::string("<tool_call>\n<function=inspect>\n<parameter=flag>\n") +
        std::string(flag) + "\n</parameter>\n<parameter=word>\nTrue\n</parameter>\n"
        "<parameter=nested>\n{\"label\":\"True\",\"z\":[true,false]}\n</parameter>\n"
        "</function>\n</tool_call>";
  };
  for (bool value : {true, false}) {
    const char* spelling = revision == "7f0e5290" ? (value ? "True" : "False")
                                                : (value ? "true" : "false");
    std::string parsed;
    ASSERT_TRUE(Ok(vm.ParseTools(schemas, xml(spelling), &parsed)));
    std::vector<iree_string_view_t> calls;
    ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(parsed), Row, &calls)));
    ASSERT_EQ(calls.size(), 1u);
    iree_string_view_t function{}, name{}, raw_arguments{};
    ASSERT_TRUE(Field(calls[0], "function", &function));
    ASSERT_TRUE(Field(function, "name", &name)); EXPECT_EQ(Copy(name), "inspect");
    ASSERT_TRUE(Field(function, "arguments", &raw_arguments));
    std::string arguments; ASSERT_TRUE(Decode(raw_arguments, &arguments));
    EXPECT_EQ(arguments, std::string("{\"flag\":") + (value ? "true" : "false") +
        ",\"word\":\"True\",\"nested\":{\"label\":\"True\",\"z\":[true,false]}}");
  }
  if (revision == "c3cf9e34") {
    // The native XML codec uses schema types as hints, not validation. A
    // non-JSON Boolean spelling remains a string; callers own schema and
    // permission validation before executing a generated tool call.
    std::string parsed;
    ASSERT_TRUE(Ok(vm.ParseTools(schemas, xml("True"), &parsed)));
    std::vector<iree_string_view_t> calls;
    ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(parsed), Row, &calls)));
    ASSERT_EQ(calls.size(), 1u);
    iree_string_view_t function{}, raw_arguments{};
    ASSERT_TRUE(Field(calls[0], "function", &function));
    ASSERT_TRUE(Field(function, "arguments", &raw_arguments));
    std::string arguments; ASSERT_TRUE(Decode(raw_arguments, &arguments));
    EXPECT_EQ(arguments,
        "{\"flag\":\"True\",\"word\":\"True\",\"nested\":{\"label\":\"True\",\"z\":[true,false]}}");
  }
  std::cout << "PUBLIC_TOOL_PARSER_COVERAGE " << revision
            << " typed_booleans=2 string_True_preserved nested_JSON_preserved\n";
}
void CheckCanonicalReplay(ChatVM& vm, std::string_view revision,
    const std::string& schema_request, const std::string& history_request,
    const std::string& history_reference, const std::vector<int32_t>& history_ids) {
  const std::string next_tool = "<|im_start|>user\n<tool_response>";
  const auto boundary = history_reference.find(next_tool);
  ASSERT_NE(boundary, std::string::npos);
  const auto expected = history_reference.substr(0, boundary);
  const std::string assistant = "<|im_start|>assistant\n<think>\n";
  const auto start = expected.rfind(assistant);
  const auto end = expected.rfind("<|im_end|>\n");
  ASSERT_NE(start, std::string::npos); ASSERT_NE(end, std::string::npos);
  ASSERT_GT(end, start + assistant.size());
  // Independent HF assistant-history bytes supply plan/Working/two tool calls.
  // Enable thinking for the initial 0.8 request so its generated suffix can carry
  // that reasoning. This modified initial prompt is not a new HF oracle row;
  // the final assistant-history rendering is independent of enable_thinking.
  const auto response = expected.substr(start + assistant.size(), end - start - assistant.size());
  auto initial_request = schema_request;
  if (revision == "7f0e5290")
    initial_request.insert(initial_request.rfind('}'), ",\"enable_thinking\":true");
  loom_serve_text_chat_completion_t completion{};
  struct CompletionCleanup { loom_serve_text_chat_completion_t* value;
    ~CompletionCleanup() { loom_serve_text_chat_completion_deinitialize(value); } } completion_cleanup{&completion};
  {
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(Ok(vm.Render(initial_request, &chat)));
    struct Cleanup { loom_serve_text_chat_t* chat;
      ~Cleanup() { loom_serve_text_chat_deinitialize(chat); } } cleanup{&chat};
    loom_serve_text_tool_calls_t calls{};
    loom_serve_text_tool_calls_initialize(iree_allocator_system(), &calls);
    struct CallsCleanup { loom_serve_text_tool_calls_t* value;
      ~CallsCleanup() { loom_serve_text_tool_calls_deinitialize(value); } } calls_cleanup{&calls};
    loom_serve_text_chat_scratch_t scratch{};
    loom_serve_text_chat_scratch_initialize(iree_allocator_system(), &scratch);
    struct ScratchCleanup { loom_serve_text_chat_scratch_t* value;
      ~ScratchCleanup() { loom_serve_text_chat_scratch_deinitialize(value); } } scratch_cleanup{&scratch};
    ASSERT_TRUE(Ok(loom_serve_text_chat_complete(&chat, View(response), false, 123, &calls,
        &scratch, iree_allocator_system(), &completion)));
    ASSERT_EQ(calls.count, 2u);
    for (size_t i = 0; i < calls.count; ++i) {
      const auto call = loom_serve_text_tool_calls_at(&calls, i);
      EXPECT_FALSE(Copy(call.id).empty()); EXPECT_EQ(Copy(call.name), "fetch");
      EXPECT_EQ(Copy(call.arguments), i == 0 ? "{\"path\":\"a\"}" : "{\"path\":\"b\"}");
    }
    EXPECT_NE(Copy(loom_serve_text_tool_calls_at(&calls, 0).id),
              Copy(loom_serve_text_tool_calls_at(&calls, 1).id));
    EXPECT_EQ(Copy(completion.transcript), expected);
    EXPECT_FALSE(completion.reusable);
  }
  // Completion owns its transcript after the original request and all source
  // call scratch retire. The actual host must replay even for a matching byte
  // prefix when the source refuses cached-state reuse.
  loom_serve_text_chat_t fresh{};
  ASSERT_TRUE(Ok(vm.Render(history_request, &fresh)));
  struct Cleanup { loom_serve_text_chat_t* chat;
    ~Cleanup() { loom_serve_text_chat_deinitialize(chat); } } cleanup{&fresh};
  EXPECT_EQ(Copy(iree_string_builder_view(&fresh.prompt)), history_reference);
  EXPECT_TRUE(loom_serve_text_chat_completion_is_prefix(&completion, &fresh));
  fresh.max_tokens = 1;
  const loom_serve_text_input_prefix_t prefix{&completion, 3, 7, LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN};
  std::vector<int32_t> scratch(history_ids.size(), -991);
  loom_serve_text_input_t input{};
  struct InputCleanup { loom_serve_text_input_t* value;
    ~InputCleanup() { loom_serve_text_input_deinitialize(value, iree_allocator_system()); } } input_cleanup{&input};
  ASSERT_TRUE(Ok(loom_serve_text_input_prepare(&fresh, prefix, false, scratch.size(),
      &input, iree_allocator_system())));
  EXPECT_EQ(input.history, nullptr); EXPECT_EQ(input.retained_count, 0u);
  ASSERT_EQ(input.count, history_ids.size());
  EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count), history_ids);
  std::cout << "CANONICAL_REPLAY_COVERAGE " << revision
            << " reasoning_content_two_tools_HF_prefix actual_host_full_replay\n";
}

// The only deferred spelling is the explicitly identified n floating argument.
// Keep the entire surrounding Unicode/tool/nested payload exact in these rows.
bool MaskFloat(std::string* prompt, std::string* spelling, double* value) {
  const std::string open = "<parameter=n>\n", close = "\n</parameter>";
  auto begin = prompt->find(open);
  if (begin == std::string::npos || prompt->find(open, begin + open.size()) != std::string::npos) return false;
  begin += open.size();
  const auto end = prompt->find(close, begin);
  if (end == std::string::npos) return false;
  *spelling = prompt->substr(begin, end - begin);
  if (!Ok(iree_json_parse_double(View(*spelling), value)) || !std::isfinite(*value)) return false;
  prompt->replace(begin, end - begin, "[deferred floating spelling]");
  return true;
}
bool IsFloatCase(std::string_view name) {
  return name == "tool-arguments-types" || name == "tool-arguments-float-decimal" ||
      name == "tool-arguments-float-exponent" || name == "tool-arguments-negative-zero" ||
      name == "tool-arguments-small-exponent";
}
struct Revision { const char* id; const char* key; bool bytecode; };
class RenderTest : public ::testing::TestWithParam<Revision> {};
TEST_P(RenderTest, DirectRequestOptionsPreservesDateValidationBeforeChatBegin) {
  const auto revision = GetParam();
  const auto product = Env(std::string("LOOMIFY_RENDER_") + revision.key + (revision.bytecode ? "_BC" : "_TEXT"));
  const auto tokenizer = Env(std::string("LOOMIFY_RENDER_TOKENIZER_") + revision.key);
  ASSERT_FALSE(product.empty()); ASSERT_FALSE(tokenizer.empty());
  const bool llama = std::string_view(revision.id) == "5816fce1";
  ChatVM vm; ASSERT_TRUE(vm.Init(product, tokenizer,
      llama ? "loomify-llama" : "loomify-qwen35"));
  // Both are well-formed requests. Only the policy-specific date value differs;
  // no chat_begin/rendering call can supply the expected Llama failure here.
  const std::string valid = R"({"messages":[{"role":"user","content":"Hello"}],"date_string":"26 Jul 2024"})";
  const std::string invalid = R"({"messages":[{"role":"user","content":"Hello"}],"date_string":26})";
  ASSERT_TRUE(Ok(vm.RequestOptions(valid)));
  auto status = vm.RequestOptions(invalid);
  const bool accepted = iree_status_is_ok(status);
  const auto diagnostic = Diagnostic(status);
  EXPECT_EQ(accepted, !llama) << diagnostic;
  // A rejected call must not prevent subsequent valid use of the same binding.
  EXPECT_TRUE(Ok(vm.RequestOptions(valid)));
}
TEST_P(RenderTest, ActualTransportMatchesPinnedReference) {
  const auto revision = GetParam();
  const auto product = Env(std::string("LOOMIFY_RENDER_") + revision.key + (revision.bytecode ? "_BC" : "_TEXT"));
  const auto tokenizer = Env(std::string("LOOMIFY_RENDER_TOKENIZER_") + revision.key);
  const auto reference = Env("LOOMIFY_RENDER_REFERENCE");
  ASSERT_FALSE(product.empty()); ASSERT_FALSE(tokenizer.empty()); ASSERT_FALSE(reference.empty());
  ChatVM vm; ASSERT_TRUE(vm.Init(product, tokenizer,
      std::string_view(revision.id) == "5816fce1" ? "loomify-llama" : "loomify-qwen35"));
  std::string data; ASSERT_TRUE(Read(reference, &data));
  std::vector<iree_string_view_t> rows;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(data), Row, &rows)));
  size_t count = 0, exact = 0, negative = 0, deferred = 0;
  std::string one_user_reference, schema_request, history_request, history_reference;
  std::vector<int32_t> history_ids;
  for (const auto row : rows) {
    iree_string_view_t raw_revision{}, raw_name{};
    ASSERT_TRUE(Field(row, "revision", &raw_revision));
    if (Copy(raw_revision) != revision.id) continue;
    ASSERT_TRUE(Field(row, "case", &raw_name));
    const auto name = Copy(raw_name);
    SCOPED_TRACE(std::string(revision.id) + "/" + name + (revision.bytecode ? "/bc" : "/text"));
    ++count;
    std::string request; ASSERT_TRUE(TransportRequest(row, vm.name(), &request));
    iree_string_view_t error{}; ASSERT_TRUE(Field(row, "error", &error));
    loom_serve_text_chat_t chat{};
    auto status = vm.Render(request, &chat);
    if (error.data) {
      ++negative;
      const bool accepted = iree_status_is_ok(status);
      const auto diagnostic = Diagnostic(status);
      EXPECT_FALSE(accepted) << "reference rejected request but native transport accepted it";
      if (accepted) loom_serve_text_chat_deinitialize(&chat);
      continue;
    }
    ASSERT_TRUE(Ok(status));
    struct Cleanup {
      loom_serve_text_chat_t* chat;
      ~Cleanup() { loom_serve_text_chat_deinitialize(chat); }
    } cleanup{&chat};
    const auto actual = Copy(iree_string_builder_view(&chat.prompt));
    iree_string_view_t text{}, ids{};
    ASSERT_TRUE(Field(row, "text", &text)); ASSERT_TRUE(Field(row, "token_ids", &ids));
    std::string expected; ASSERT_TRUE(Decode(text, &expected));
    if (name == "one-user") one_user_reference = expected;
    ASSERT_TRUE(ids.data) << "oracle must include complete token vectors";
    std::vector<int32_t> expected_ids, actual_ids, prepared_ids;
    ASSERT_TRUE(Ok(iree_json_enumerate_array(ids, Token, &expected_ids)));
    if (name == "tool-schema") schema_request = request;
    if (name == "tool-multiple-results") {
      history_request = request; history_reference = expected; history_ids = expected_ids;
    }
    ASSERT_TRUE(vm.Encode(actual, &actual_ids));
    ASSERT_TRUE(vm.Prepare(chat, actual.size() + 16, &prepared_ids));
    EXPECT_EQ(prepared_ids, actual_ids) << "actual source input adapter differs from native flag0 encoding";
    if (IsFloatCase(name)) {
      ++deferred;
      auto masked_actual = actual, masked_expected = expected;
      std::string actual_spelling, reference_spelling; double a = 0, e = 0;
      ASSERT_TRUE(MaskFloat(&masked_actual, &actual_spelling, &a));
      ASSERT_TRUE(MaskFloat(&masked_expected, &reference_spelling, &e));
      EXPECT_EQ(masked_actual, masked_expected) << "only floating parameter spelling is deferred";
      EXPECT_EQ(a, e); if (e == 0) EXPECT_EQ(std::signbit(a), std::signbit(e));
      std::cout << "DEFERRED_FLOAT_SPELLING " << revision.id << '/' << name
                << " reference=" << reference_spelling << " native=" << actual_spelling << '\n';
      if (actual == expected) EXPECT_EQ(actual_ids, expected_ids);
    } else {
      ++exact;
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(actual_ids, expected_ids);
    }
  }
  ASSERT_FALSE(one_user_reference.empty());
  CheckInputPolicy(vm, revision.id, one_user_reference);
  if (std::string_view(revision.id) != "5816fce1") {
    CheckAdmission(vm, revision.id);
    CheckPublicToolParser(vm, revision.id);
    ASSERT_FALSE(schema_request.empty()); ASSERT_FALSE(history_request.empty());
    CheckCanonicalReplay(vm, revision.id, schema_request, history_request, history_reference, history_ids);
  }
  EXPECT_EQ(count, std::string_view(revision.id) == "7f0e5290" ? 34u :
                   std::string_view(revision.id) == "c3cf9e34" ? 41u : 7u);
  EXPECT_GT(exact, 0u);
  std::cout << "RENDER_COVERAGE " << revision.id << (revision.bytecode ? " bc" : " text")
            << " rows=" << count << " exact=" << exact << " negative=" << negative
            << " deferred_float=" << deferred << '\n';
}
class Qwen3RenderTest : public ::testing::TestWithParam<Revision> {};
TEST_P(Qwen3RenderTest, ActualTransportPreservesGgufTemplateBytesAndTokens) {
  const auto revision = GetParam();
  const bool bc = revision.bytecode;
  const bool non_thinking = std::string_view(revision.id) == "cd8e9439";
  ChatVM vm;
  ASSERT_TRUE(vm.Init(Env(std::string("LOOMIFY_RENDER_") + revision.key + (bc ? "_BC" : "_TEXT")),
      Env(std::string("LOOMIFY_RENDER_TOKENIZER_") + revision.key),
      non_thinking ? "loomify-qwen2" : "loomify-qwen3"));
  EXPECT_EQ(vm.SupportsReasoning(), !non_thinking);
  std::string data;
  ASSERT_TRUE(Read(Env("LOOMIFY_RENDER_REFERENCE"), &data));
  std::vector<iree_string_view_t> rows;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(data), Row, &rows)));
  size_t count = 0, negative = 0, admission_negative = 0;
  std::string one_user_reference;
  for (const auto row : rows) {
    iree_string_view_t row_revision{}, name{};
    ASSERT_TRUE(Field(row, "revision", &row_revision));
    if (Copy(row_revision) != revision.id) continue;
    ASSERT_TRUE(Field(row, "case", &name));
    SCOPED_TRACE(Copy(name));
    std::cout << "QWEN3_CASE_BEGIN format=" << (bc ? "bc" : "text") << " case=" << Copy(name) << std::endl;
    ++count;
    std::string request;
    ASSERT_TRUE(TransportRequest(row, vm.name(), &request));
    loom_serve_text_chat_t chat{};
    iree_string_view_t error{};
    ASSERT_TRUE(Field(row, "error", &error));
    iree_string_view_t admission_error{};
    ASSERT_TRUE(Field(row, "native_admission_error", &admission_error));
    auto render_status = vm.Render(request, &chat);
    if (admission_error.data) {
      // These remain valid template-domain oracle rows. Native request
      // admission is a separate contract, exercised by an explicit negative;
      // supported counterparts below still require exact bytes and token IDs.
      ++admission_negative;
      std::string expected_diagnostic;
      ASSERT_TRUE(Decode(admission_error, &expected_diagnostic));
      const bool accepted = iree_status_is_ok(render_status);
      const auto diagnostic = Diagnostic(render_status);
      EXPECT_FALSE(accepted);
      EXPECT_NE(diagnostic.find(expected_diagnostic), std::string::npos);
      if (accepted) loom_serve_text_chat_deinitialize(&chat);
      const bool matches = !accepted &&
          diagnostic.find(expected_diagnostic) != std::string::npos;
      std::cout << "QWEN3_CASE_RESULT format=" << (bc ? "bc" : "text")
                << " case=" << Copy(name) << " status="
                << (matches ? "PASS_HOST_REJECTION" : "FAIL_HOST_REJECTION")
                << std::endl;
      continue;
    }
    if (error.data) {
      ++negative;
      const bool accepted = iree_status_is_ok(render_status);
      const auto diagnostic = Diagnostic(render_status);
      EXPECT_FALSE(accepted);
      if (accepted) loom_serve_text_chat_deinitialize(&chat);
      std::cout << "QWEN3_CASE_RESULT format=" << (bc ? "bc" : "text") << " case=" << Copy(name)
                << " status=" << (accepted ? "FAIL_TEMPLATE_REJECTION" : "PASS_TEMPLATE_REJECTION") << std::endl;
      continue;
    }
    if (!Ok(render_status)) {
      // Keep the failure, then execute the remaining independent oracle rows.
      std::cout << "QWEN3_CASE_RESULT format=" << (bc ? "bc" : "text") << " case=" << Copy(name)
                << " status=FAIL_HOST_ADMISSION" << std::endl;
      continue;
    }
    struct Cleanup { loom_serve_text_chat_t* c; ~Cleanup() { loom_serve_text_chat_deinitialize(c); } } cleanup{&chat};
    const auto actual = Copy(iree_string_builder_view(&chat.prompt));
    iree_string_view_t text{}, ids{};
    ASSERT_TRUE(Field(row, "text", &text));
    ASSERT_TRUE(Field(row, "token_ids", &ids));
    std::string expected;
    ASSERT_TRUE(Decode(text, &expected));
    if (Copy(name) == (non_thinking ? "qwen25-one-user" : "qwen3-one-user")) one_user_reference = expected;
    std::vector<int32_t> expected_ids, actual_ids, prepared_ids;
    ASSERT_TRUE(Ok(iree_json_enumerate_array(ids, Token, &expected_ids)));
    ASSERT_TRUE(vm.Encode(actual, &actual_ids));
    ASSERT_TRUE(vm.Prepare(chat, actual.size() + 16, &prepared_ids));
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(actual_ids, expected_ids);
    EXPECT_EQ(prepared_ids, expected_ids);
    std::cout << "QWEN3_CASE_RESULT format=" << (bc ? "bc" : "text") << " case=" << Copy(name)
              << " status=" << ((actual == expected && actual_ids == expected_ids && prepared_ids == expected_ids) ? "PASS" : "FAIL_PARITY")
              << " bytes_equal=" << (actual == expected) << " native_tokens_equal=" << (actual_ids == expected_ids)
              << " prepared_tokens_equal=" << (prepared_ids == expected_ids) << std::endl;
  }
  ASSERT_FALSE(one_user_reference.empty());
  CheckInputPolicy(vm, revision.id, one_user_reference);
  const std::string schemas = R"([{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}])";
  const std::string call = R"(<tool_call>{"name":"f","arguments":{"text":"é </tool_call>","nested":{"ok":true},"n":1}}</tool_call>)";
  std::string parsed;
  ASSERT_TRUE(Ok(vm.ParseTools(schemas, call + "\n" + call, &parsed)));
  std::vector<iree_string_view_t> calls;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(parsed), Row, &calls)));
  ASSERT_EQ(calls.size(), 2u);
  for (auto item : calls) {
    iree_string_view_t function{}, name{}, raw_args{};
    ASSERT_TRUE(Field(item, "function", &function));
    ASSERT_TRUE(Field(function, "name", &name)); EXPECT_EQ(Copy(name), "f");
    ASSERT_TRUE(Field(function, "arguments", &raw_args));
    std::string args;
    ASSERT_TRUE(Decode(raw_args, &args));
    EXPECT_EQ(args, R"({"text":"é </tool_call>","nested":{"ok":true},"n":1})");
  }
  ASSERT_TRUE(Ok(vm.ParseTools(schemas, "<tool_call>{\"name\":\"f\"", &parsed)));
  EXPECT_EQ(parsed, "[]"); // Incomplete generation is not a complete call.
  for (const auto& bad : {
      R"(<tool_call>{"name":"other","arguments":{}}</tool_call>)",
      R"(<tool_call>{"name":"f","name":"f","arguments":{}}</tool_call>)",
      R"(<tool_call>{"name":"f","arguments":[],}</tool_call>)",
      R"(<tool_call>{"name":"f","arguments":[]}</tool_call>)"}) {
    parsed = "unchanged";
    auto status = vm.ParseTools(schemas, bad, &parsed);
    const bool accepted = iree_status_is_ok(status);
    const auto diagnostic = Diagnostic(status);
    EXPECT_FALSE(accepted) << bad;
    EXPECT_EQ(parsed, "unchanged");
  }
  EXPECT_EQ(count, non_thinking ? 28u : 40u);
  EXPECT_EQ(negative, non_thinking ? 2u : 1u);
  EXPECT_EQ(admission_negative, non_thinking ? 4u : 5u);
  std::cout << "QWEN3_GGUF_RENDER_COVERAGE " << (bc ? "bc" : "text") << " oracle_rows_seen=" << count << " expected_template_errors=" << negative << " expected_host_errors=" << admission_negative << " float_parity=deferred" << std::endl;
}
INSTANTIATE_TEST_SUITE_P(Qwen3SourceAndBytecode, Qwen3RenderTest, ::testing::Values(
    Revision{"5da44855", "5DA44855", false}, Revision{"5da44855", "5DA44855", true},
    Revision{"cd8e9439", "CD8E9439", false}, Revision{"cd8e9439", "CD8E9439", true}));

INSTANTIATE_TEST_SUITE_P(KnownRevisions, RenderTest, ::testing::Values(
    Revision{"7f0e5290", "7F0E5290", false}, Revision{"7f0e5290", "7F0E5290", true},
    Revision{"c3cf9e34", "C3CF9E34", false}, Revision{"c3cf9e34", "C3CF9E34", true},
    Revision{"5816fce1", "5816FCE1", false}, Revision{"5816fce1", "5816FCE1", true}));
// Gemma uses the same actual native ChatVM harness as the qualified protocols.
// This suite consumes independent HF rows, not product-computed expected text.
void CheckGemmaInput(ChatVM& vm, const std::string& fresh_user) {
  const std::string raw=u8" \u2003Hello\u00a0 ";
  for(auto format:{LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,LOOM_SERVE_TEXT_CHAT_INPUT_USER}) {
    for(auto boundary:{LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED}) {
      SCOPED_TRACE("Gemma input="+std::to_string(format)+" boundary="+std::to_string(boundary));
      const bool user=format==LOOM_SERVE_TEXT_CHAT_INPUT_USER;
      const std::string separator=boundary==LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN ? "<end_of_turn>\n" : boundary==LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED ? "\n" : "";
      const std::string expected_text=user ? (boundary==LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH ? fresh_user : separator+"<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n") : separator+raw;
      std::vector<int32_t> expected;ASSERT_TRUE(vm.Encode(expected_text,&expected));
      loom_serve_text_chat_t chat{};ASSERT_TRUE(RawChat(vm,raw,&chat));
      std::vector<int32_t> actual(expected.size(),-991);iree_host_size_t count=0;
      ASSERT_TRUE(Ok(PrepareMode(chat,format,boundary,expected.size(),actual.data(),&count)));
      EXPECT_EQ(count,expected.size());EXPECT_EQ(actual,expected);
      if(boundary==LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH){
        std::fill(actual.begin(),actual.end(),-991);count=99;
        auto status=PrepareMode(chat,format,boundary,expected.size()-1,actual.data(),&count);
        EXPECT_FALSE(iree_status_is_ok(status));Diagnostic(status);EXPECT_EQ(count,0u);
        EXPECT_EQ(actual,std::vector<int32_t>(expected.size(),-991));
      }
      loom_serve_text_chat_deinitialize(&chat);
    }
  }
  // Direct raw-input retained-boundary proof. Production structured completion
  // below refuses reuse; this fixture does not claim a structured cache hit.
  std::vector<int32_t> eot;ASSERT_TRUE(vm.Encode("<end_of_turn>",&eot));ASSERT_EQ(eot.size(),1u);
  for(auto boundary:{LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED}){
    loom_serve_text_chat_t chat{};ASSERT_TRUE(RawChat(vm,"HISTORY|NEXT",&chat));
    loom_serve_text_chat_completion_t completion{};
    ASSERT_TRUE(Ok(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,iree_make_const_byte_span("HISTORY|",8),1,iree_allocator_system(),&completion.storage)));
    completion.transcript=iree_make_string_view(static_cast<const char*>(iree_vm_buffer_const_data(completion.storage)),8);completion.reusable=true;
    const int32_t anchor=boundary==LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN ? 7 : eot[0];
    const std::string sep=boundary==LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN ? "<end_of_turn>\n" : "\n";
    std::vector<int32_t> expected;ASSERT_TRUE(vm.Encode(sep+"NEXT",&expected));expected.insert(expected.begin(),anchor);
    const loom_serve_text_input_prefix_t prefix{&completion,3,anchor,boundary};
    std::vector<int32_t> scratch(expected.size()+3,-991);loom_serve_text_input_t input{};
    ASSERT_TRUE(Ok(loom_serve_text_input_prepare(&chat,prefix,true,scratch.size(),&input,iree_allocator_system())));
    EXPECT_EQ(input.retained_count,3u);EXPECT_EQ(input.count,expected.size());
    EXPECT_EQ(std::vector<int32_t>(input.tokens,input.tokens+input.count),expected);
    loom_serve_text_input_deinitialize(&input,iree_allocator_system());loom_serve_text_chat_completion_deinitialize(&completion);loom_serve_text_chat_deinitialize(&chat);
  }
}
void CheckGemmaOutputReplay(ChatVM& vm,const std::string& replay_request,const std::string& replay_text,const std::vector<int32_t>& replay_ids){
  const std::string visible=u8"  spaced café 🙂  ";
  std::vector<int32_t> terminal; ASSERT_TRUE(vm.Encode("<end_of_turn>", &terminal));
  ASSERT_EQ(terminal, std::vector<int32_t>({106}));
  // The canonical106 case is the host boundary shared by both independently
  // forced raw-stop publisher witnesses. This test does not implement or prove
  // the raw1/raw106 mapping; it verifies actual SDK completion and fresh followup
  // at their common published boundary, alongside the ordinary open boundary.
  for (int32_t last_token : {9, terminal[0]}) {
  SCOPED_TRACE("Gemma canonical followup last_token=" + std::to_string(last_token));
  loom_serve_text_chat_completion_t completion{};
  {
    loom_serve_text_chat_t chat{};ASSERT_TRUE(Ok(vm.Render(R"({"messages":[{"role":"user","content":"Hello"}]})",&chat)));
    for(size_t end=0;end<=visible.size();++end){
      const auto prefix=View(std::string_view(visible).substr(0,end));
      if(!iree_unicode_utf8_validate(prefix))continue;
      loom_serve_text_chat_output_t span{};
      ASSERT_TRUE(Ok(loom_serve_text_chat_output_next(&chat,prefix,LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING,iree_allocator_system(),&span)));
      if(span.advanced)EXPECT_EQ(span.kind,LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT);
    }
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)),visible);
    loom_serve_text_tool_calls_t calls{};loom_serve_text_tool_calls_initialize(iree_allocator_system(),&calls);
    loom_serve_text_chat_scratch_t scratch{};loom_serve_text_chat_scratch_initialize(iree_allocator_system(),&scratch);
    ASSERT_TRUE(Ok(loom_serve_text_chat_complete(&chat,View(visible),false,last_token,&calls,&scratch,iree_allocator_system(),&completion)));
    EXPECT_EQ(calls.count,0u);EXPECT_FALSE(completion.reusable);
    EXPECT_EQ(Copy(iree_string_builder_view(&chat.output.content)),visible);
    EXPECT_EQ(Copy(completion.transcript),std::string("<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n")+u8"spaced café 🙂<end_of_turn>\n");
    loom_serve_text_chat_scratch_deinitialize(&scratch);loom_serve_text_tool_calls_deinitialize(&calls);loom_serve_text_chat_deinitialize(&chat);
  }
  loom_serve_text_chat_t fresh{};ASSERT_TRUE(Ok(vm.Render(replay_request,&fresh)));
  EXPECT_EQ(Copy(iree_string_builder_view(&fresh.prompt)),replay_text);
  fresh.max_tokens=1;
  const bool stopped = last_token == terminal[0];
  const loom_serve_text_input_prefix_t prefix{&completion,3,
      stopped ? terminal[0] : 7,
      stopped ? LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED : LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN};
  std::vector<int32_t> scratch(replay_ids.size(),-991);loom_serve_text_input_t input{};
  ASSERT_TRUE(Ok(loom_serve_text_input_prepare(&fresh,prefix,false,scratch.size(),&input,iree_allocator_system())));
  EXPECT_EQ(input.history,nullptr);EXPECT_EQ(input.retained_count,0u);EXPECT_EQ(input.count,replay_ids.size());
  EXPECT_EQ(std::vector<int32_t>(input.tokens,input.tokens+input.count),replay_ids);
  loom_serve_text_input_deinitialize(&input,iree_allocator_system());loom_serve_text_chat_deinitialize(&fresh);loom_serve_text_chat_completion_deinitialize(&completion);
  }
}
// An added tokenizer media spelling in a plain string is not a media request.
// Binding accepts the JSON, but the actual encoded OOV ID must not reach gather.
void CheckGemmaTokenDomain(ChatVM& vm) {
  std::vector<int32_t> encoded;
  ASSERT_TRUE(vm.Encode("<image_soft_token>", &encoded));
  ASSERT_EQ(encoded, std::vector<int32_t>({262144}));
  for (auto format : {LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, LOOM_SERVE_TEXT_CHAT_INPUT_USER}) {
    loom_serve_text_chat_t chat{};
    ASSERT_TRUE(RawChat(vm, "<image_soft_token>", &chat));
    std::vector<int32_t> scratch(128, -991); iree_host_size_t count = 99;
    auto status = PrepareMode(chat, format, LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
                             scratch.size(), scratch.data(), &count);
    EXPECT_FALSE(iree_status_is_ok(status)); Diagnostic(status);
    EXPECT_EQ(count, 0u); EXPECT_EQ(scratch, std::vector<int32_t>(128, -991));
    loom_serve_text_chat_deinitialize(&chat);
  }
  loom_serve_text_chat_t chat{};
  ASSERT_TRUE(Ok(vm.Render(R"({"messages":[{"role":"user","content":"<image_soft_token>"}]})", &chat)));
  std::vector<int32_t> scratch(128, -991); iree_host_size_t count = 99;
  auto status = PrepareMode(chat, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
      LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, scratch.size(), scratch.data(), &count);
  EXPECT_FALSE(iree_status_is_ok(status)); Diagnostic(status);
  EXPECT_EQ(count, 0u); EXPECT_EQ(scratch, std::vector<int32_t>(128, -991));
  loom_serve_text_chat_deinitialize(&chat);
}
class GemmaRenderTest : public ::testing::TestWithParam<bool> {};
TEST_P(GemmaRenderTest, ActualTransportInputOutputAndReplayMatchesHF){
  const bool bc=GetParam();ChatVM vm;
  ASSERT_TRUE(vm.Init(Env(bc?"LOOMIFY_RENDER_GEMMA_BC":"LOOMIFY_RENDER_GEMMA_TEXT"),Env("LOOMIFY_RENDER_TOKENIZER_GEMMA"),"loomify-gemma3"));
  CheckGemmaTokenDomain(vm);
  std::string data;ASSERT_TRUE(Read(Env("LOOMIFY_RENDER_GEMMA_REFERENCE"),&data));
  std::vector<iree_string_view_t> rows;ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(data),Row,&rows)));
  size_t exact=0,negative=0;std::string one_user,replay_request,replay_text;std::vector<int32_t> replay_ids;
  for(auto row:rows){
    iree_string_view_t name{},error{},kwargs{},generation{};ASSERT_TRUE(Field(row,"case",&name));ASSERT_TRUE(Field(row,"error",&error));ASSERT_TRUE(Field(row,"kwargs",&kwargs));ASSERT_TRUE(Field(kwargs,"add_generation_prompt",&generation));
    const bool gen=Copy(generation)=="true";SCOPED_TRACE(Copy(name)+(gen?"/generation":"/no-generation"));
    std::string request;ASSERT_TRUE(TransportRequest(row,vm.name(),&request));
    loom_serve_text_chat_t chat{};auto status=vm.Render(request,&chat);
    if(error.data){++negative;const bool accepted=iree_status_is_ok(status);Diagnostic(status);EXPECT_FALSE(accepted);if(accepted)loom_serve_text_chat_deinitialize(&chat);continue;}
    ASSERT_TRUE(Ok(status));++exact;
    iree_string_view_t text{},ids{};ASSERT_TRUE(Field(row,"text",&text));ASSERT_TRUE(Field(row,"token_ids",&ids));
    std::string expected;ASSERT_TRUE(Decode(text,&expected));std::vector<int32_t> expected_ids,actual_ids,prepared;
    ASSERT_TRUE(Ok(iree_json_enumerate_array(ids,Token,&expected_ids)));
    const auto actual=Copy(iree_string_builder_view(&chat.prompt));EXPECT_EQ(actual,expected);
    ASSERT_TRUE(vm.Encode(actual,&actual_ids));EXPECT_EQ(actual_ids,expected_ids);
    ASSERT_TRUE(vm.Prepare(chat,expected_ids.size(),&prepared));EXPECT_EQ(prepared,expected_ids);
    if(Copy(name)=="one-user"&&gen)one_user=expected;
    if(Copy(name)=="replay"&&gen){replay_request=request;replay_text=expected;replay_ids=expected_ids;}
    loom_serve_text_chat_deinitialize(&chat);
  }
  EXPECT_EQ(exact,24u);EXPECT_EQ(negative,8u);ASSERT_FALSE(one_user.empty());ASSERT_FALSE(replay_request.empty());
  CheckGemmaInput(vm,one_user);CheckGemmaOutputReplay(vm,replay_request,replay_text,replay_ids);
  for(const char* request:{
    R"({"messages":[{"role":"user","content":"U"}],"tools":[{"type":"function","function":{"name":"fetch","parameters":{"type":"object","properties":{}}}}]})",
    R"({"messages":[{"role":"user","content":"U"}],"enable_thinking":true})",
    R"({"messages":[{"role":"user","content":"U"}],"reasoning_effort":"low"})",
    R"({"messages":[{"role":"user","content":"U"}],"reasoning_effort":null})",
    R"({"messages":[{"role":"user","content":"U"},{"role":"tool","content":"T"}]})",
    R"({"messages":[{"role":"user","content":"U"}],"model_options":{"unknown":true}})"
  }){
    loom_serve_text_chat_t chat{};auto status=vm.Render(request,&chat);const bool accepted=iree_status_is_ok(status);Diagnostic(status);EXPECT_FALSE(accepted);if(accepted)loom_serve_text_chat_deinitialize(&chat);
  }
  loom_serve_text_chat_t escaped{};ASSERT_TRUE(Ok(vm.Render(R"({"messages":[{"role":"u\u0073er","content":"Hello"}]})",&escaped)));
  EXPECT_EQ(Copy(iree_string_builder_view(&escaped.prompt)),one_user);loom_serve_text_chat_deinitialize(&escaped);
  std::cout<<"GEMMA_COVERAGE "<<(bc?"bc":"source")<<" exact24 source_errors8 native_rejections6 input_modes6 overflow2 raw_retained2 outputUTF8 canonicalReplay2 oovRejected3\n";
}
INSTANTIATE_TEST_SUITE_P(GemmaSourceAndBytecode,GemmaRenderTest,::testing::Values(false,true));
} // namespace
