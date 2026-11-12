// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/text/testing/render_harness.h"

namespace hrx::loomify_gguf::testing {
iree_string_view_t View(std::string_view s) {
  return iree_make_string_view(s.data(), s.size());
}
std::string Copy(iree_string_view_t s) {
  return s.size ? std::string(s.data, s.size) : std::string{};
}
std::string Diagnostic(iree_status_t status) {
  if (iree_status_is_ok(status))
    return {};
  char message[4096] = {};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(message), message, &length);
  message[sizeof(message) - 1] = 0;
  iree_status_ignore(status);
  return message;
}
bool Ok(iree_status_t status) {
  if (iree_status_is_ok(status))
    return true;
  ADD_FAILURE() << Diagnostic(status);
  return false;
}
bool Read(const std::string &path, std::string *out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    ADD_FAILURE() << "cannot read " << path;
    return false;
  }
  out->assign(std::istreambuf_iterator<char>(input), {});
  if (input.bad()) {
    ADD_FAILURE() << "cannot finish reading " << path;
    return false;
  }
  return true;
}
std::string Env(const std::string &name) {
  const char *value = std::getenv(name.c_str());
  if (!value || !*value) {
    ADD_FAILURE() << "required explicit fixture path: " << name;
    return {};
  }
  return value;
}
bool Field(iree_string_view_t object, const char *name,
           iree_string_view_t *out) {
  return Ok(iree_json_try_lookup_object_value(
      object, iree_make_cstring_view(name), out));
}
bool Decode(iree_string_view_t raw, std::string *out) {
  out->resize(raw.size);
  iree_host_size_t count = 0;
  if (!Ok(iree_json_unescape_string(raw, out->size(), out->data(), &count)))
    return false;
  out->resize(count);
  return true;
}
iree_status_t Row(void *data, iree_host_size_t, iree_json_value_type_t type,
                  iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "oracle rows must be objects");
  static_cast<std::vector<iree_string_view_t> *>(data)->push_back(value);
  return iree_ok_status();
}
iree_status_t Token(void *data, iree_host_size_t, iree_string_view_t value) {
  int64_t id = -1;
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &id));
  if (id < 0 || id > std::numeric_limits<int32_t>::max())
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "oracle token ID out of range");
  static_cast<std::vector<int32_t> *>(data)->push_back(
      static_cast<int32_t>(id));
  return iree_ok_status();
}
// Reference templates omit the HTTP route envelope. Bind that transport-only
// field to the actual loaded policy; it does not change template variables.
bool AddModelRoute(std::string *request, std::string_view route) {
  iree_string_view_t model{};
  if (!Field(View(*request), "model", &model))
    return false;
  if (model.data)
    return true;
  // These fixture route identities are the asserted package aliases, not
  // arbitrary user input needing a new JSON serializer.
  if (route.empty() || route.find_first_of("\"\\") != std::string_view::npos) {
    ADD_FAILURE() << "invalid fixture route";
    return false;
  }
  const auto close = request->rfind('}');
  if (close == std::string::npos)
    return false;
  request->insert(close, ",\"model\":\"" + std::string(route) + "\"");
  return true;
}
// A missing pure-renderer add_generation_prompt variable means no suffix.
// Serve's HTTP default is true, so map that explicit oracle condition to false
// in the existing model_options envelope; do not alter expected reference
// bytes.
bool TransportRequest(iree_string_view_t row, std::string_view route,
                      std::string *request) {
  iree_string_view_t raw{}, kwargs{}, generation{};
  if (!Field(row, "request", &raw) || !Field(row, "kwargs", &kwargs) ||
      !Field(kwargs, "add_generation_prompt", &generation))
    return false;
  *request = Copy(raw);
  if (generation.data)
    return AddModelRoute(request, route);
  iree_string_view_t options{};
  if (!Field(raw, "model_options", &options))
    return false;
  if (!options.data) {
    const auto close = request->rfind('}');
    if (close == std::string::npos)
      return false;
    request->insert(close,
                    ",\"model_options\":{\"add_generation_prompt\":false}");
  } else {
    // Narrow fixture adapter: preserve existing option bytes and insert only
    // the missing formatter variable. Fixture JSON has unique object keys.
    iree_string_view_t present{};
    if (!Field(options, "add_generation_prompt", &present))
      return false;
    if (present.data) {
      ADD_FAILURE() << "oracle kwargs/request mismatch";
      return false;
    }
    const auto offset = static_cast<size_t>(options.data - raw.data);
    const auto close =
        request->find_last_not_of(" \t\r\n", offset + options.size - 1);
    if (close == std::string::npos || (*request)[close] != '}')
      return false;
    const bool empty =
        Copy(options).find_first_not_of("{} \t\r\n") == std::string::npos;
    request->insert(close, std::string(empty ? "" : ",") +
                               "\"add_generation_prompt\":false");
  }
  return AddModelRoute(request, route);
}
// Inspect current API-owned token storage without altering native IDs.
iree_status_t CopyPreparedTokenIds(iree_vm_buffer_t *storage, size_t count,
                                   size_t capacity, int32_t *destination) {
  if (count > capacity ||
      (count &&
       (!storage || iree_vm_buffer_length(storage) / sizeof(int32_t) < count)))
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid prepared token storage");
  if (count)
    std::memcpy(destination, iree_vm_buffer_const_data(storage),
                count * sizeof(int32_t));
  return iree_ok_status();
}
ChatVM::~ChatVM() {
  if (policy_ready_)
    loom_serve_text_chat_policy_deinitialize(&policy_);
  loom_runtime_program_free(program_);
  iree_vm_module_release(tools_);
  iree_vm_module_release(input_);
  iree_vm_module_release(json_);
  iree_tokenizer_free(tokenizer_);
  iree_vm_environment_free(environment_);
  loomc_target_environment_release(target_);
}

bool ChatVM::Init(const std::string &product, const std::string &tokenizer_path,
                  std::string_view expected_name) {
  auto status =
      loomc_target_environment_create_vm(loomc_allocator_system(), &target_);
  if (!loomc_status_is_ok(status)) {
    auto m = loomc_status_message(status);
    ADD_FAILURE() << std::string(m.data, m.size);
    loomc_status_free(status);
    return false;
  }
  loomc_status_free(status);
  std::string tokenizer_json;
  if (!Read(tokenizer_path, &tokenizer_json) ||
      !Ok(iree_tokenizer_from_huggingface_json(
          View(tokenizer_json), iree_allocator_system(), &tokenizer_)) ||
      !Ok(iree_vm_environment_allocate(iree_allocator_system(),
                                       &environment_)) ||
      !Ok(loom_runtime_json_module_create(environment_, iree_allocator_system(),
                                          &json_)) ||
      !Ok(loom_runtime_input_module_create(environment_, tokenizer_,
                                           iree_allocator_system(), &input_)) ||
      !Ok(loom_serve_text_tool_xml_module_create(
          environment_, iree_allocator_system(), &tools_)))
    return false;
  const iree_string_view_t roots[] = {
      IREE_SV("model_name"),        IREE_SV("chat_reasoning"),
      IREE_SV("chat_image_limits"), IREE_SV("chat_begin"),
      IREE_SV("chat_message"),      IREE_SV("chat_end"),
      IREE_SV("parse_tools"),       IREE_SV("render_tool"),
      IREE_SV("prepare_input"),     IREE_SV("output_next"),
      IREE_SV("complete_text"),     IREE_SV("request_defaults"),
      IREE_SV("request_options")};
  iree_vm_module_t *libraries[] = {json_, input_, tools_};
  loom_runtime_sources_t sources{};
  loom_runtime_compilation_report_t report{};
  auto compiled = loom_runtime_program_allocate(
      target_, environment_, &sources, View(product), IREE_ARRAYSIZE(roots),
      roots, iree_vm_module_span_from_array(libraries), &report,
      iree_allocator_system(), &program_);
  if (!iree_status_is_ok(compiled)) {
    for (iree_host_size_t i = 0; i < report.diagnostic_count; ++i) {
      const auto &d = report.diagnostics[i];
      std::cerr.write(d.message,
                      std::min<uint64_t>(d.message_length, sizeof(d.message)))
          << '\n';
    }
    return Ok(compiled);
  }
  iree_vm_buffer_t *options = nullptr;
  if (!Ok(iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                               iree_make_const_byte_span("{}", 2), 1,
                               iree_allocator_system(), &options)))
    return false;
  auto bound = loom_serve_text_chat_policy_initialize(environment_, program_,
                                                      options, &policy_);
  iree_vm_buffer_release(options);
  if (!Ok(bound))
    return false;
  policy_ready_ = true;
  EXPECT_EQ(Copy(policy_.name), expected_name);
  // Both current host flag values retain the provider's complete-response ABI.
  for (bool truncated : {false, true}) {
    std::string calls;
    if (!Ok(ParseTools("[]", "", &calls, truncated)))
      return false;
    EXPECT_EQ(calls, "[]");
  }
  return true;
}

bool ChatVM::SupportsReasoning() const {
  return policy_.reasoning.support == LOOM_SERVE_REASONING_SUPPORTED;
}

std::string_view ChatVM::name() const {
  return {policy_.name.data, policy_.name.size};
}

iree_status_t ChatVM::RequestOptions(const std::string &body) {
  // Invoke the bound policy entry directly. Calling Render would also run
  // chat_begin, whose independent date lookup could mask lost validation.
  std::string request = body;
  if (!AddModelRoute(&request, name()))
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid fixture route envelope");
  iree_vm_buffer_t *input = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_clone(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_const_byte_span(request.data(), request.size()), 1,
      iree_allocator_system(), &input));
  iree_vm_variant_t args[] = {
      iree_vm_buffer_variant_from_ptr_borrowed(&policy_.types,
                                               policy_.defaults),
      iree_vm_buffer_variant_from_ptr_move(&policy_.types, &input),
      iree_vm_variant_from_i64(123)};
  iree_vm_variant_t result = {};
  auto status = iree_vm_invoke(policy_.invocation, policy_.request_options,
                               iree_vm_variant_span_from_array(args),
                               iree_vm_variant_span_from_ptr(&result, 1));
  iree_vm_buffer_t *options = nullptr;
  if (iree_status_is_ok(status))
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy_.types, result,
                                                      &options);
  if (iree_status_is_ok(status) && !options)
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null request options");
  iree_vm_variant_reset(&result);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(args));
  return status;
}

iree_status_t ChatVM::Render(const std::string &body,
                             loom_serve_text_chat_t *chat) {
  // Admission fixtures use the same real route envelope as reference rows.
  body_ = body;
  if (!AddModelRoute(&body_, name()))
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid fixture route envelope");
  return loom_serve_text_chat_initialize(&policy_, policy_.name, View(body_),
                                         123, 128, iree_allocator_system(),
                                         chat);
}

iree_status_t ChatVM::ParseTools(std::string_view schemas,
                                 std::string_view response, std::string *out,
                                 bool truncated) {
  const std::string_view inputs[] = {schemas, response};
  iree_vm_variant_t args[3] = {}, result[1] = {};
  args[2] = iree_vm_variant_from_i64(truncated);
  auto status = iree_ok_status();
  for (size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    iree_vm_buffer_t *buffer = nullptr;
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(inputs[i].data(), inputs[i].size()), 1,
        iree_allocator_system(), &buffer);
    if (iree_status_is_ok(status))
      args[i] = iree_vm_buffer_variant_from_ptr_move(&policy_.types, &buffer);
  }
  if (iree_status_is_ok(status))
    status = iree_vm_invoke(policy_.invocation, policy_.parse_tools,
                            iree_vm_variant_span_from_array(args),
                            iree_vm_variant_span_from_array(result));
  iree_vm_buffer_t *buffer = nullptr;
  if (iree_status_is_ok(status))
    status = iree_vm_buffer_ptr_from_variant_borrowed(&policy_.types, result[0],
                                                      &buffer);
  if (iree_status_is_ok(status)) {
    if (!buffer)
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null parsed tools");
    else
      out->assign(static_cast<const char *>(iree_vm_buffer_const_data(buffer)),
                  iree_vm_buffer_length(buffer));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(result));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(args));
  return status;
}

bool ChatVM::Encode(std::string_view text, std::vector<int32_t> *tokens) {
  tokens->resize(text.size() + 16);
  iree_host_size_t count = 0;
  if (!Ok(iree_tokenizer_encode(
          tokenizer_, View(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
          iree_tokenizer_make_token_output(tokens->data(), nullptr, nullptr,
                                           tokens->size()),
          iree_allocator_system(), &count)))
    return false;
  tokens->resize(count);
  return true;
}

bool ChatVM::Prepare(const loom_serve_text_chat_t &chat, size_t capacity,
                     std::vector<int32_t> *tokens) {
  tokens->resize(capacity);
  iree_host_size_t count = 0, span_count = 0;
  loom_serve_text_media_span_t *spans = nullptr;
  iree_vm_buffer_t *token_storage = nullptr;
  auto status = loom_serve_text_chat_prepare_input(
      &chat, 0, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
      LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, capacity, 0, iree_allocator_system(),
      &token_storage, &count, &spans, &span_count);
  iree_allocator_free(iree_allocator_system(), spans);
  if (iree_status_is_ok(status))
    status =
        CopyPreparedTokenIds(token_storage, count, capacity, tokens->data());
  iree_vm_buffer_release(token_storage);
  if (!Ok(status))
    return false;
  EXPECT_EQ(span_count, 0u);
  tokens->resize(count);
  return true;
}

// A raw caller supplies its own rendered bytes. Initialize ordinary native chat
// ownership first, then replace only its prompt buffer for this input fixture.
bool RawChat(ChatVM &vm, std::string_view text, loom_serve_text_chat_t *chat) {
  if (!Ok(vm.Render(R"({"messages":[{"role":"user","content":"Hello"}]})",
                    chat)))
    return false;
  iree_string_builder_reset(&chat->prompt);
  if (!Ok(iree_string_builder_append_string(&chat->prompt, View(text)))) {
    loom_serve_text_chat_deinitialize(chat);
    return false;
  }
  chat->max_tokens = 1;
  return true;
}
iree_status_t PrepareMode(const loom_serve_text_chat_t &chat,
                          loom_serve_text_chat_input_format_t format,
                          loom_serve_text_chat_boundary_t boundary,
                          size_t capacity, int32_t *tokens,
                          iree_host_size_t *count) {
  loom_serve_text_media_span_t *spans = nullptr;
  iree_host_size_t span_count = 0;
  iree_vm_buffer_t *token_storage = nullptr;
  auto status = loom_serve_text_chat_prepare_input(
      &chat, 0, format, boundary, capacity, 0, iree_allocator_system(),
      &token_storage, count, &spans, &span_count);
  if (iree_status_is_ok(status))
    status = CopyPreparedTokenIds(token_storage, *count, capacity, tokens);
  iree_vm_buffer_release(token_storage);
  iree_allocator_free(iree_allocator_system(), spans);
  EXPECT_EQ(span_count, 0u);
  return status;
}
void CheckInputPolicy(ChatVM &vm, const InputPolicyExpectations &expectations) {
  const auto generation =
      expectations.fresh_user_reference.rfind(expectations.assistant_header);
  ASSERT_NE(generation, std::string::npos);
  const auto suffix = expectations.fresh_user_reference.substr(generation);
  auto separator = [&](loom_serve_text_chat_boundary_t boundary) {
    if (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN)
      return expectations.open_separator;
    if (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED)
      return expectations.terminated_separator;
    return std::string{};
  };
  for (const auto format :
       {LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, LOOM_SERVE_TEXT_CHAT_INPUT_USER}) {
    for (const auto boundary : {LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
                                LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN,
                                LOOM_SERVE_TEXT_CHAT_BOUNDARY_TERMINATED}) {
      SCOPED_TRACE("input format=" + std::to_string(format) +
                   " boundary=" + std::to_string(boundary));
      const bool user = format == LOOM_SERVE_TEXT_CHAT_INPUT_USER;
      const std::string text =
          user ? expectations.user_input : expectations.raw_input;
      const auto expected_text =
          user ? (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH
                      ? expectations.fresh_user_reference
                      : separator(boundary) + expectations.framed_user_turn +
                            suffix)
               : separator(boundary) + expectations.raw_input;
      std::vector<int32_t> expected;
      ASSERT_TRUE(vm.Encode(expected_text, &expected));
      ASSERT_FALSE(expected.empty());
      loom_serve_text_chat_t chat{};
      ASSERT_TRUE(RawChat(vm, text, &chat));
      struct Cleanup {
        loom_serve_text_chat_t *chat;
        ~Cleanup() { loom_serve_text_chat_deinitialize(chat); }
      } cleanup{&chat};
      std::vector<int32_t> actual(expected.size(), -991);
      iree_host_size_t count = 0;
      ASSERT_TRUE(Ok(PrepareMode(chat, format, boundary, expected.size(),
                                 actual.data(), &count)));
      EXPECT_EQ(count, expected.size());
      EXPECT_EQ(actual, expected);
      if (boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH) {
        // Capacity+1 probing must reject truncation and leave output untouched.
        std::fill(actual.begin(), actual.end(), -991);
        count = 99;
        auto status = PrepareMode(chat, format, boundary, expected.size() - 1,
                                  actual.data(), &count);
        const bool accepted = iree_status_is_ok(status);
        Diagnostic(status);
        EXPECT_FALSE(accepted);
        EXPECT_EQ(count, 0u);
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
    struct Cleanup {
      loom_serve_text_chat_t *chat;
      ~Cleanup() { loom_serve_text_chat_deinitialize(chat); }
    } cleanup{&chat};
    loom_serve_text_chat_completion_t completion{};
    ASSERT_TRUE(Ok(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(history.data(), history.size()), 1,
        iree_allocator_system(), &completion.storage)));
    completion.transcript = iree_make_string_view(
        static_cast<const char *>(
            iree_vm_buffer_const_data(completion.storage)),
        history.size());
    completion.reusable = true;
    struct HistoryCleanup {
      loom_serve_text_chat_completion_t *value;
      ~HistoryCleanup() { loom_serve_text_chat_completion_deinitialize(value); }
    } history_cleanup{&completion};
    // OPEN preserves a deliberately unrelated pending token; TERMINATED uses
    // this known fixture's canonical EOS. Neither comes from transcript text.
    const int32_t anchor = boundary == LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN
                               ? 7
                               : expectations.terminal_anchor;
    const loom_serve_text_input_prefix_t prefix{&completion, 3, anchor,
                                                boundary};
    std::vector<int32_t> expected;
    ASSERT_TRUE(vm.Encode(separator(boundary) + tail, &expected));
    expected.insert(expected.begin(), anchor);
    const auto capacity = prefix.position + expected.size();
    std::vector<int32_t> scratch(capacity, -991);
    loom_serve_text_input_t input{};
    struct InputCleanup {
      loom_serve_text_input_t *value;
      ~InputCleanup() {
        loom_serve_text_input_deinitialize(value, iree_allocator_system());
      }
    } input_cleanup{&input};
    ASSERT_TRUE(Ok(loom_serve_text_input_prepare(
        &chat, prefix, true, capacity, &input, iree_allocator_system())));
    ASSERT_EQ(input.count, expected.size());
    ASSERT_NE(input.tokens, nullptr);
    EXPECT_EQ(std::vector<int32_t>(input.tokens, input.tokens + input.count),
              expected);
    EXPECT_EQ(input.tokens[0], anchor);
    EXPECT_EQ(input.history, completion.storage);
    EXPECT_EQ(input.retained_count, prefix.position);
  }
  std::cout
      << "INPUT_POLICY_COVERAGE " << vm.name()
      << " modes=6 exact_capacity=6 overflow=2 actual_retained_boundaries=2\n";
}

void CheckRenderRevision(const TextRevisionCase &test_case, bool bc,
                         const std::string &reference_path) {
  SCOPED_TRACE(test_case.id);
  ChatVM vm;
  ASSERT_TRUE(
      vm.Init(Env(std::string("LOOMIFY_RENDER_") + test_case.key +
                  (bc ? "_BC" : "_TEXT")),
              Env(std::string("LOOMIFY_RENDER_TOKENIZER_") + test_case.key),
              test_case.package_name));
  EXPECT_EQ(vm.SupportsReasoning(), test_case.supports_reasoning);
  std::string data;
  ASSERT_TRUE(Read(reference_path, &data));
  std::vector<iree_string_view_t> rows;
  ASSERT_TRUE(Ok(iree_json_enumerate_array_typed(View(data), Row, &rows)));
  size_t count = 0, negative = 0, admission_negative = 0;
  RenderReferences references;
  for (const auto row : rows) {
    iree_string_view_t row_revision{}, name{};
    ASSERT_TRUE(Field(row, "revision", &row_revision));
    ASSERT_EQ(Copy(row_revision), test_case.id);
    ASSERT_TRUE(Field(row, "case", &name));
    SCOPED_TRACE(Copy(name));
    std::cout << "RENDER_CASE_BEGIN format=" << (bc ? "bc" : "text")
              << " case=" << Copy(name) << std::endl;
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
      if (accepted)
        loom_serve_text_chat_deinitialize(&chat);
      const bool matches = !accepted && diagnostic.find(expected_diagnostic) !=
                                            std::string::npos;
      std::cout << "RENDER_CASE_RESULT format=" << (bc ? "bc" : "text")
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
      if (accepted)
        loom_serve_text_chat_deinitialize(&chat);
      std::cout << "RENDER_CASE_RESULT format=" << (bc ? "bc" : "text")
                << " case=" << Copy(name) << " status="
                << (accepted ? "FAIL_TEMPLATE_REJECTION"
                             : "PASS_TEMPLATE_REJECTION")
                << std::endl;
      continue;
    }
    if (!Ok(render_status)) {
      // Keep the failure, then execute the remaining independent oracle rows.
      std::cout << "RENDER_CASE_RESULT format=" << (bc ? "bc" : "text")
                << " case=" << Copy(name) << " status=FAIL_HOST_ADMISSION"
                << std::endl;
      continue;
    }
    struct Cleanup {
      loom_serve_text_chat_t *c;
      ~Cleanup() { loom_serve_text_chat_deinitialize(c); }
    } cleanup{&chat};
    const auto actual = Copy(iree_string_builder_view(&chat.prompt));
    iree_string_view_t text{}, ids{};
    ASSERT_TRUE(Field(row, "text", &text));
    ASSERT_TRUE(Field(row, "token_ids", &ids));
    std::string expected;
    ASSERT_TRUE(Decode(text, &expected));
    if (Copy(name) == test_case.representative_case)
      references.one_user = expected;
    ASSERT_TRUE(ids.data) << "oracle must include complete token vectors";
    std::vector<int32_t> expected_ids, actual_ids, prepared_ids;
    ASSERT_TRUE(Ok(iree_json_enumerate_array(ids, Token, &expected_ids)));
    ASSERT_TRUE(vm.Encode(actual, &actual_ids));
    ASSERT_TRUE(vm.Prepare(chat, actual.size() + 16, &prepared_ids));
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(actual_ids, expected_ids);
    EXPECT_EQ(prepared_ids, expected_ids);
    std::cout << "RENDER_CASE_RESULT format=" << (bc ? "bc" : "text")
              << " case=" << Copy(name) << " status="
              << ((actual == expected && actual_ids == expected_ids &&
                   prepared_ids == expected_ids)
                      ? "PASS"
                      : "FAIL_PARITY")
              << " bytes_equal=" << (actual == expected)
              << " native_tokens_equal=" << (actual_ids == expected_ids)
              << " prepared_tokens_equal=" << (prepared_ids == expected_ids)
              << std::endl;
  }
  EXPECT_EQ(count, test_case.expected_rows);
  EXPECT_EQ(negative, test_case.expected_template_errors);
  EXPECT_EQ(admission_negative, test_case.expected_host_rejections);
  ASSERT_FALSE(references.one_user.empty());
  ASSERT_NE(test_case.check_policy, nullptr);
  test_case.check_policy(vm, references);
  std::cout << "RENDER_COVERAGE " << test_case.id << (bc ? " bc" : " text")
            << " rows=" << count << " template_errors=" << negative
            << " host_errors=" << admission_negative << '\n';
}
} // namespace hrx::loomify_gguf::testing
