// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/text/testing/selected_policy.h"
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/json.h"
#include "iree/base/internal/json.h"
#include "loomc/loomc.h"
#include "loomc/link.h"
#include "loomc/target/vm.h"
#include "gtest/gtest.h"
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
bool Ok(loomc_status_t s) {
  bool ok = loomc_status_is_ok(s);
  if (!ok) {
    auto m = loomc_status_message(s);
    ADD_FAILURE() << std::string(m.data, m.size);
  }
  loomc_status_free(s);
  return ok;
}
bool NativeOk(iree_status_t s) {
  if (iree_status_is_ok(s))
    return true;
  char text[4096];
  iree_host_size_t n = 0;
  iree_status_format(s, sizeof(text), text, &n);
  ADD_FAILURE() << text;
  iree_status_ignore(s);
  return false;
}
bool Result(loomc_result_t *r) {
  if (r && loomc_result_succeeded(r))
    return true;
  if (r)
    for (size_t i = 0; i < loomc_result_diagnostic_count(r); ++i) {
      auto d = loomc_result_diagnostic_at(r, i);
      ADD_FAILURE() << std::string(d->message.data, d->message.size);
    }
  return false;
}
enum class Profile {
  InlineHistory,
  ExplicitReasoning,
  Hermes,
  HermesNonThinking
};
enum class Startup { Default, Enabled, Disabled, NoGeneration };
class VM {
public:
  ~VM() {
    iree_vm_process_release(process);
    if (invocation)
      iree_vm_invocation_deinitialize(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(bytecode);
    iree_vm_module_release(input_module);
    iree_vm_module_release(json_module);
    iree_vm_environment_free(runtime);
    loomc_allocator_free(loomc_allocator_system(),
                         const_cast<uint8_t *>(image.data));
  }
  bool Init(Profile selected, bool bc) {
    loomc_target_environment_t *e = nullptr;
    if (!Ok(loomc_target_environment_create_vm(loomc_allocator_system(), &e)))
      return false;
    env.reset(e);
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = e;
    loomc_context_options_t opts{};
    opts.next = &target;
    loomc_context_t *c = nullptr;
    if (!Ok(loomc_context_create(&opts, loomc_allocator_system(), &c)))
      return false;
    context.reset(c);
    loomc_workspace_t *w = nullptr;
    if (!Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w)))
      return false;
    workspace.reset(w);
    const char *template_path = nullptr;
    switch (selected) {
    case Profile::InlineHistory:
      template_path = "lib/text/qwen35/templates/7f0e5290.jinja";
      break;
    case Profile::ExplicitReasoning:
      template_path = "lib/text/qwen35/templates/c3cf9e34.jinja";
      break;
    case Profile::Hermes:
      template_path = "lib/text/qwen3/templates/5da44855.jinja";
      break;
    case Profile::HermesNonThinking:
      template_path = "lib/text/qwen25/templates/cd8e9439.jinja";
      break;
    }
    loomc_module_t *m = nullptr;
    std::string error;
    if (!hrx::loomify_gguf::testing::LinkSelectedText(
            c, w, template_path, {}, "<|im_end|>",
            {"output_next", "complete_text", "render_tool", "chat_begin"}, &m,
            &error)) {
      ADD_FAILURE() << error;
      return false;
    }
    Owner<loomc_module_t, loomc_module_release> module(m, loomc_module_release);
    // Exercise the same fully configured composition in each serialized form.
    loomc_module_serialize_options_t serialization{};
    serialization.format =
        bc ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
    loomc_source_t *bytes = nullptr;
    if (!Ok(loomc_module_serialize_to_source(m, &serialization,
                                             loomc_allocator_system(), &bytes)))
      return false;
    Owner<loomc_source_t, loomc_source_release> stored(bytes,
                                                       loomc_source_release);
    loomc_module_t *restored = nullptr;
    loomc_result_t *rr = nullptr;
    auto status = loomc_module_deserialize_from_source(
        c, w, bytes, nullptr, loomc_allocator_system(), &restored, &rr);
    Owner<loomc_result_t, loomc_result_release> result(rr,
                                                       loomc_result_release);
    module.reset(restored);
    if (!Ok(status) || !Result(rr))
      return false;
    loomc_compiler_t *compiler_raw = nullptr;
    if (!Ok(loomc_compiler_create(c, nullptr, loomc_allocator_system(),
                                  &compiler_raw)))
      return false;
    Owner<loomc_compiler_t, loomc_compiler_release> compiler(
        compiler_raw, loomc_compiler_release);
    loomc_target_profile_t *p = nullptr;
    if (!Ok(loomc_target_profile_select(e, loomc_make_cstring_view("vm:core"),
                                        loomc_allocator_system(), &p)))
      return false;
    Owner<loomc_target_profile_t, loomc_target_profile_release> profile(
        p, loomc_target_profile_release);
    const loomc_string_view_t roots[] = {
        loomc_make_cstring_view("output_next"),
        loomc_make_cstring_view("complete_text"),
        loomc_make_cstring_view("render_tool"),
        loomc_make_cstring_view("chat_begin")};
    loomc_emit_options_t emit{};
    emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit.structure_size = sizeof(emit);
    emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
    loomc_compile_artifact_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
    options.structure_size = sizeof(options);
    options.roots = roots;
    options.root_count = 4;
    options.target_profile = p;
    options.emit_options = &emit;
    loomc_result_t *compiled_raw = nullptr;
    status =
        loomc_compile_artifact(compiler_raw, w, nullptr, module.get(), &options,
                               loomc_allocator_system(), &compiled_raw);
    Owner<loomc_result_t, loomc_result_release> compiled(compiled_raw,
                                                         loomc_result_release);
    if (!Ok(status) || !Result(compiled_raw))
      return false;
    const loomc_artifact_t *artifact = nullptr;
    for (size_t i = 0; i < loomc_result_artifact_count(compiled_raw); ++i) {
      auto a = loomc_result_artifact_at(compiled_raw, i);
      if (a->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE)
        artifact = a;
    }
    if (!artifact || !Ok(loomc_byte_sequence_clone(
                         artifact->contents, loomc_allocator_system(), &image)))
      return false;
    if (!NativeOk(
            iree_vm_environment_allocate(iree_allocator_system(), &runtime)))
      return false;
    if (!NativeOk(iree_vm_ref_types_resolve(
            iree_vm_environment_lookup_ref_type_table(runtime, IREE_SV("vm")),
            &types)))
      return false;
    if (!NativeOk(iree_vm_bytecode_module_create(
            runtime, IREE_SV("compiled"),
            {iree_make_const_byte_span(image.data, image.data_length),
             iree_allocator_null()},
            iree_allocator_system(), &bytecode)))
      return false;
    if (!NativeOk(loom_runtime_json_module_create(
            runtime, iree_allocator_system(), &json_module)) ||
        !NativeOk(loom_runtime_input_module_create(
            runtime, nullptr, iree_allocator_system(), &input_module)))
      return false;
    iree_vm_module_t *libraries[] = {json_module, input_module};
    if (!NativeOk(iree_vm_program_create(
            {bytecode, iree_vm_module_span_from_array(libraries)},
            iree_allocator_system(), &program)))
      return false;
    if (!NativeOk(iree_vm_invocation_initialize(
            iree_make_byte_span(storage.data(), storage.size()), &invocation)))
      return false;
    return NativeOk(iree_vm_process_create(program, invocation,
                                           iree_vm_variant_span_empty(),
                                           iree_allocator_system(), &process));
  }
  iree_vm_variant_t Buffer(const std::string &text) {
    iree_vm_buffer_t *b = nullptr;
    EXPECT_TRUE(NativeOk(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(text.data(), text.size()), 8,
        iree_allocator_system(), &b)));
    return iree_vm_buffer_variant_from_ptr_move(&types, &b);
  }
  bool Call(const std::string &root, std::vector<iree_vm_variant_t> &args,
            std::vector<iree_vm_variant_t> &out) {
    iree_vm_function_t fn = iree_vm_function_null();
    bool found = false;
    for (uint32_t i = 0; i < bytecode->descriptor->counts.export_count; ++i) {
      iree_vm_module_export_declaration_t entry{};
      bytecode->vtable->query_export(bytecode, i, &entry);
      std::string name(entry.export_name.data, entry.export_name.size);
      if (name == root || name.rfind(root + "$", 0) == 0 ||
          name.rfind(root + ".", 0) == 0) {
        found = NativeOk(iree_vm_process_lookup_function(
            process, IREE_SV("compiled"), entry.export_name, &fn));
        break;
      }
    }
    if (!found) {
      ADD_FAILURE() << "missing export " << root;
      return false;
    }
    bool ok = NativeOk(iree_vm_invoke(
        invocation, fn, {args.data(), args.size()}, {out.data(), out.size()}));
    iree_vm_variant_span_reset({args.data(), args.size()});
    return ok;
  }
  int64_t Initial(bool thinking, bool generation) {
    return Start(!generation ? Startup::NoGeneration
                 : thinking  ? Startup::Enabled
                             : Startup::Disabled);
  }
  std::string Request(Startup startup, bool generation) {
    std::string result =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"U\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}}}}}],"
        "\"add_generation_prompt\":";
    result += generation ? "true" : "false";
    if (startup != Startup::Default)
      result += std::string(",\"enable_thinking\":") +
                (startup == Startup::Enabled ? "true" : "false");
    return result + '}';
  }
  int64_t Begin(const std::string &request, std::string *prompt) {
    std::vector<iree_vm_variant_t> args{Buffer(request), Buffer("")}, out(2);
    int64_t state = -1;
    if (Call("chat_begin", args, out)) {
      void *pointer = nullptr;
      EXPECT_TRUE(NativeOk(
          iree_vm_ptr_from_variant_borrowed(out[0], types.buffer, &pointer)));
      if (pointer) {
        auto *buffer = static_cast<iree_vm_buffer_t *>(pointer);
        prompt->assign(
            static_cast<const char *>(iree_vm_buffer_const_data(buffer)),
            iree_vm_buffer_length(buffer));
      }
      EXPECT_TRUE(NativeOk(iree_vm_i64_from_variant(out[1], &state)));
    }
    iree_vm_variant_span_reset({out.data(), out.size()});
    return state;
  }
  int64_t Start(Startup startup) {
    request_ = Request(startup, startup != Startup::NoGeneration);
    return Begin(request_, &prompt_);
  }
  std::array<int64_t, 4> Next(const std::string &text, int64_t cursor,
                              int64_t state, int32_t phase) {
    std::vector<iree_vm_variant_t> args = {Buffer(text),
                                           iree_vm_variant_from_i64(cursor),
                                           iree_vm_variant_from_i64(state),
                                           iree_vm_variant_from_i32(phase)},
                                   out(4);
    std::array<int64_t, 4> words{};
    if (Call("output_next", args, out))
      for (size_t i = 0; i < 4; ++i)
        EXPECT_TRUE(NativeOk(iree_vm_i64_from_variant(out[i], &words[i])));
    iree_vm_variant_span_reset({out.data(), out.size()});
    return words;
  }
  std::string FormatTools(const std::string &calls) {
    std::vector<iree_string_view_t> records;
    const auto view = [](std::string_view value) {
      return iree_make_string_view(value.data(), value.size());
    };
    auto status = iree_json_enumerate_array_typed(
        view(calls),
        [](void *data, iree_host_size_t, iree_json_value_type_t kind,
           iree_string_view_t value) {
          if (kind != IREE_JSON_VALUE_TYPE_OBJECT)
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
          static_cast<std::vector<iree_string_view_t> *>(data)->push_back(
              value);
          return iree_ok_status();
        },
        &records);
    if (!NativeOk(status))
      return {};
    auto string = [](iree_string_view_t object, const char *key) {
      iree_string_view_t raw{};
      auto status = iree_json_try_lookup_object_value(
          object, iree_make_cstring_view(key), &raw);
      EXPECT_TRUE(NativeOk(status));
      EXPECT_NE(raw.data, nullptr);
      std::string decoded(raw.size, '\0');
      iree_host_size_t size = 0;
      EXPECT_TRUE(NativeOk(iree_json_unescape_string(raw, decoded.size(),
                                                     decoded.data(), &size)));
      decoded.resize(size);
      return decoded;
    };
    std::string formatted;
    for (size_t i = 0; i < records.size(); ++i) {
      iree_string_view_t function{};
      if (!NativeOk(iree_json_try_lookup_object_value(
              records[i], IREE_SV("function"), &function)))
        return {};
      std::vector<iree_vm_variant_t> arguments{
          Buffer(string(function, "name")),
          Buffer(string(function, "arguments")), iree_vm_variant_from_i64(i)},
          results(2);
      if (Call("render_tool", arguments, results)) {
        void *pointer = nullptr;
        EXPECT_TRUE(NativeOk(iree_vm_ptr_from_variant_borrowed(
            results[0], types.buffer, &pointer)));
        if (pointer) {
          const auto *buffer = static_cast<iree_vm_buffer_t *>(pointer);
          formatted.append(
              static_cast<const char *>(iree_vm_buffer_const_data(buffer)),
              iree_vm_buffer_length(buffer));
        }
      }
      iree_vm_variant_span_reset({results.data(), results.size()});
    }
    return formatted;
  }
  std::pair<std::string, int32_t> Complete(const std::string &content,
                                           const std::string &thought,
                                           const std::string &tools,
                                           Startup startup) {
    const int64_t initial = Start(startup);
    EXPECT_GE(Begin(Request(startup, false), &base_prompt_), 0);
    const auto formatted = FormatTools(tools);
    std::vector<iree_vm_variant_t> args = {Buffer(prompt_),
                                           Buffer("unused raw response"),
                                           Buffer(content),
                                           Buffer(thought),
                                           Buffer(formatted),
                                           iree_vm_variant_from_i64(initial),
                                           iree_vm_variant_from_i64(1)},
                                   out(2);
    std::pair<std::string, int32_t> result;
    if (Call("complete_text", args, out)) {
      void *ptr = nullptr;
      EXPECT_TRUE(NativeOk(
          iree_vm_ptr_from_variant_borrowed(out[0], types.buffer, &ptr)));
      if (ptr) {
        auto b = static_cast<iree_vm_buffer_t *>(ptr);
        result.first.assign(
            static_cast<const char *>(iree_vm_buffer_const_data(b)),
            iree_vm_buffer_length(b));
      }
      EXPECT_TRUE(NativeOk(iree_vm_i32_from_variant(out[1], &result.second)));
    }
    iree_vm_variant_span_reset({out.data(), out.size()});
    return result;
  }
  const std::string &BasePrompt() const { return base_prompt_; }

private:
  std::string request_, prompt_, base_prompt_;
  Owner<loomc_target_environment_t, loomc_target_environment_release> env{
      nullptr, loomc_target_environment_release};
  Owner<loomc_context_t, loomc_context_release> context{nullptr,
                                                        loomc_context_release};
  Owner<loomc_workspace_t, loomc_workspace_release> workspace{
      nullptr, loomc_workspace_release};
  loomc_byte_span_t image{};
  iree_vm_environment_t *runtime = nullptr;
  iree_vm_module_t *bytecode = nullptr;
  iree_vm_module_t *json_module = nullptr;
  iree_vm_module_t *input_module = nullptr;
  iree_vm_program_t *program = nullptr;
  iree_vm_process_t *process = nullptr;
  iree_vm_invocation_t *invocation = nullptr;
  iree_vm_ref_types_t types{};
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage{};
};
struct Fixture {
  std::string response, reasoning, content, tail;
  Startup startup;
};
void Check(VM &vm, const Fixture &f, size_t split, bool bytewise) {
  int64_t cursor = 0, state = vm.Start(f.startup);
  ASSERT_GE(state, 0);
  std::string reasoning, content;
  auto drain = [&](size_t size, int phase) {
    std::string text = f.response.substr(0, size);
    for (size_t guard = 0; guard < text.size() + 10; ++guard) {
      auto span = vm.Next(text, cursor, state, phase);
      ASSERT_GE(span[0], 0);
      ASSERT_LE(span[0], 2);
      ASSERT_EQ(span[1], cursor);
      ASSERT_GE(span[2], cursor);
      ASSERT_LE(span[2], int64_t(size));
      if (span[2] == cursor) {
        EXPECT_EQ(span[3], state);
        return;
      }
      auto bytes = text.substr(span[1], span[2] - span[1]);
      // Every published UTF-8 span starts and ends at a complete code point.
      if (span[0] && !bytes.empty()) {
        EXPECT_NE(static_cast<unsigned char>(bytes.front()) & 0xc0, 0x80);
        if (size_t(span[2]) < size)
          EXPECT_NE(static_cast<unsigned char>(text[span[2]]) & 0xc0, 0x80);
      }
      if (span[0] == 1)
        content += bytes;
      if (span[0] == 2)
        reasoning += bytes;
      cursor = span[2];
      state = span[3];
    }
    FAIL() << "parser did not settle";
  };
  if (bytewise)
    for (size_t i = 0; i <= f.response.size(); ++i)
      drain(i, 0);
  else {
    drain(split, 0);
    drain(f.response.size(), 0);
  }
  drain(f.response.size(), 1);
  EXPECT_EQ(reasoning, f.reasoning);
  EXPECT_EQ(content, f.content);
  EXPECT_EQ(f.response.substr(cursor), f.tail);
}
TEST(QwenOutput, ModelGeneratedOpenerAtEverySplitAndBytecode) {
  for (bool bc : {false, true}) {
    VM vm;
    ASSERT_TRUE(vm.Init(Profile::Hermes, bc));
    EXPECT_GE(vm.Initial(true, true), 0);
    EXPECT_GE(vm.Initial(false, true), 0);
    EXPECT_GE(vm.Initial(true, false), 0);
    // The source's enabled/default generation suffix is neutral. Allowing
    // thinking does not itself open a reasoning frame in the response.
    const std::string suffix = "<|im_start|>assistant\n";
    for (auto startup : {Startup::Default, Startup::Enabled}) {
      std::string prompt;
      ASSERT_GE(vm.Begin(vm.Request(startup, true), &prompt), 0);
      ASSERT_GE(prompt.size(), suffix.size());
      EXPECT_EQ(prompt.substr(prompt.size() - suffix.size()), suffix);
    }
    const std::vector<Fixture> fixtures = {
        {"<think>\nreason 🦙</think>\n\nanswer", "reason 🦙", "answer", "",
         Startup::Enabled},
        {"<think>reason</think>answer", "reason", "answer", "",
         Startup::Enabled},
        {"<think>\nreason</think>answer", "reason", "answer", "",
         Startup::Default},
        {"reason</think>answer", "", "reason</think>answer", "",
         Startup::Enabled},
        {"reason</think>answer", "", "reason</think>answer", "",
         Startup::Default},
        {"plain 🦙", "", "plain 🦙", "", Startup::Enabled},
        {"plain 🦙", "", "plain 🦙", "", Startup::Default},
        {"<think>\n<tool_call>{}", "", "", "<tool_call>{}", Startup::Enabled},
        // A partial opener is buffered while streaming, then flushed as
        // content at finalization when it never establishes a frame.
        {"<thi", "", "<thi", "", Startup::Enabled},
        {"<thi", "", "<thi", "", Startup::Default},
        {"plain", "", "plain", "", Startup::Disabled},
        {"", "", "", "", Startup::Enabled},
    };
    for (const auto &f : fixtures) {
      for (size_t split = 0; split <= f.response.size(); ++split)
        Check(vm, f, split, false);
      Check(vm, f, 0, true);
    }
  }
}

TEST(QwenOutput, ModelGeneratedOpeningPreservesCanonicalWhitespace) {
  for (bool bc : {false, true}) {
    VM vm;
    ASSERT_TRUE(vm.Init(Profile::Hermes, bc));
    const std::string calls =
        R"([{"id":"call_f","type":"function","function":{"name":"f","arguments":"{\"x\":\"v\"}"}}])";
    auto result =
        vm.Complete(" answer ", "\n thought \n", calls, Startup::Enabled);
    EXPECT_EQ(
        result.first,
        vm.BasePrompt() +
            "<|im_start|>assistant\n<think>\n thought \n</think>\n\n answer \n"
            "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\":\"v\"}}\n</tool_call><|im_end|>\n");
    EXPECT_EQ(result.second, 0);
    result = vm.Complete(" answer ", "", "[]", Startup::Disabled);
    EXPECT_EQ(
        result.first,
        vm.BasePrompt() +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n answer <|im_end|>\n");
    EXPECT_EQ(result.second, 0);
    result = vm.Complete(" answer ", " thought ", "[]", Startup::NoGeneration);
    EXPECT_EQ(
        result.first,
        vm.BasePrompt() +
            "<|im_start|>assistant\n<think>\n thought \n</think>\n\n answer <|im_end|>\n");
    EXPECT_EQ(result.second, 0);
  }
}

TEST(QwenOutput, NeverThinkingKeepsLiteralMarkersAndNoPrefixCompletionPlain) {
  for (bool bc : {false, true}) {
    VM vm;
    ASSERT_TRUE(vm.Init(Profile::HermesNonThinking, bc));
    const Fixture fixtures[] = {
        {u8"  café 🦙 <think>literal</think> \n", "",
         u8"  café 🦙 <think>literal</think> \n", "", Startup::Disabled},
        {u8"answer 🦙<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
         "", u8"answer 🦙",
         "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>",
         Startup::Disabled},
    };
    for (const auto &fixture : fixtures) {
      for (size_t split = 0; split <= fixture.response.size(); ++split)
        Check(vm, fixture, split, false);
      Check(vm, fixture, 0, true);
    }
    const auto result =
        vm.Complete(" answer ", "ignored thought", "[]", Startup::NoGeneration);
    EXPECT_EQ(result.first,
              vm.BasePrompt() + "<|im_start|>assistant\n answer <|im_end|>\n");
    EXPECT_EQ(result.second, 0);
  }
}
TEST(QwenOutput, EverySplitAndBytewiseGrowthFromSourceAndBytecode) {
  for (bool bc : {false, true})
    for (bool thinking : {false, true}) {
      VM vm;
      ASSERT_TRUE(vm.Init(
          thinking ? Profile::ExplicitReasoning : Profile::InlineHistory, bc));
      EXPECT_GE(vm.Initial(false, true), 0);
      EXPECT_GE(vm.Initial(true, true), 0);
      EXPECT_GE(vm.Initial(false, false), 0);
      EXPECT_GE(vm.Initial(true, false), 0);
      std::vector<Fixture> fixtures = {
          {"reason</think>\n\nanswer", "reason", "answer", "",
           Startup::Enabled},
          {"reason</think>answer</think>literal", "reason",
           "answer</think>literal", "", Startup::Enabled},
          {"reason</think>\nanswer", "reason", "answer", "", Startup::Enabled},
          {"reason</think>\n\n\nanswer", "reason", "\nanswer", "",
           Startup::Enabled},
          {"reason</think>", "reason", "", "", Startup::Enabled},
          {"reason</thi", "reason</thi", "", "", Startup::Enabled},
          {"reason 🦙 café 中", "reason 🦙 café 中", "", "", Startup::Enabled},
          {"answer 🦙 café 中", "", "answer 🦙 café 中", "", Startup::Disabled},
          {"answer<tool_cal", "", "answer<tool_cal", "", Startup::Disabled},
          {"answer\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>", "",
           "answer\n\n", "<tool_call>\n<function=f>\n</function>\n</tool_call>",
           Startup::Disabled},
          {"reason<tool_call>broken", "reason", "", "<tool_call>broken",
           Startup::Enabled},
          {"<tool_call>broken", "", "", "<tool_call>broken", Startup::Enabled},
          {"<tool_call>broken", "", "", "<tool_call>broken", Startup::Disabled},
          {"reason</think>\n\nanswer<tool_call>x", "reason", "answer",
           "<tool_call>x", Startup::Enabled},
          {"", "", "", "", Startup::Default},
          {"raw bytes", thinking ? "raw bytes" : "",
           thinking ? "" : "raw bytes", "", Startup::Default},
          {"explicit false", "", "explicit false", "", Startup::Disabled},
          {"no prefix </think>", "", "no prefix </think>", "",
           Startup::NoGeneration},
      };
      const std::string tool_marker = "<tool_call>", think_marker = "</think>";
      for (size_t n = 1; n < tool_marker.size(); ++n) {
        auto partial = tool_marker.substr(0, n);
        fixtures.push_back({"answer" + partial, "", "answer" + partial, "",
                            Startup::Disabled});
        fixtures.push_back(
            {"reason" + partial, "reason" + partial, "", "", Startup::Enabled});
      }
      for (size_t n = 1; n < think_marker.size(); ++n) {
        auto partial = think_marker.substr(0, n);
        fixtures.push_back(
            {"reason" + partial, "reason" + partial, "", "", Startup::Enabled});
      }
      for (const auto &f : fixtures) {
        SCOPED_TRACE(::testing::Message()
                     << "bc=" << bc << " default=" << thinking
                     << " text=" << f.response);
        for (size_t split = 0; split <= f.response.size(); ++split)
          Check(vm, f, split, false);
        Check(vm, f, 0, true);
      }
    }
}
TEST(QwenOutput, CanonicalAssistantAndConservativeReplayFromSourceAndBytecode) {
  for (bool bc : {false, true})
    for (bool thinking : {false, true}) {
      VM vm;
      ASSERT_TRUE(vm.Init(
          thinking ? Profile::ExplicitReasoning : Profile::InlineHistory, bc));
      const std::string header = "<|im_start|>assistant\n<think>\n";
      const std::string calls =
          R"([{"id":"call_f","type":"function","function":{"name":"f","arguments":"{\"x\":\"v\"}"}}])";
      const std::string xml =
          "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n</tool_call>";
      auto result = vm.Complete("\u3000 answer \u00a0", "\u2003 thought \u202f",
                                calls, Startup::Enabled);
      EXPECT_EQ(result.first, vm.BasePrompt() + header +
                                  "thought\n</think>\n\nanswer\n\n" + xml +
                                  "<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      result = vm.Complete(" answer ", "", "[]", Startup::Disabled);
      EXPECT_EQ(result.first,
                vm.BasePrompt() + header + "\n</think>\n\nanswer<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      result =
          vm.Complete(" answer ", " thought ", "[]", Startup::NoGeneration);
      EXPECT_EQ(result.first, vm.BasePrompt() + header +
                                  "thought\n</think>\n\nanswer<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      result = vm.Complete("", "", calls, Startup::NoGeneration);
      EXPECT_EQ(result.first, vm.BasePrompt() + header + "\n</think>\n\n" +
                                  xml + "<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      result = vm.Complete("", " truncated thought ", "[]", Startup::Enabled);
      EXPECT_EQ(result.first,
                vm.BasePrompt() + header +
                    "truncated thought\n</think>\n\n<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      // DEFAULT mode uses the declared revision default in reconstruction too.
      result = vm.Complete(" answer ", thinking ? " thought " : "", "[]",
                           Startup::Default);
      EXPECT_EQ(result.first, vm.BasePrompt() + header +
                                  (thinking ? "thought" : "") +
                                  "\n</think>\n\nanswer<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
    }
}
} // namespace
