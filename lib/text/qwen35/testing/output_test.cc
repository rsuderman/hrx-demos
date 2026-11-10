// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
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
class VM {
public:
  ~VM() {
    iree_vm_process_release(process);
    if (invocation)
      iree_vm_invocation_deinitialize(invocation);
    iree_vm_program_release(program);
    iree_vm_module_release(bytecode);
    iree_vm_environment_free(runtime);
    loomc_allocator_free(loomc_allocator_system(),
                         const_cast<uint8_t *>(image.data));
  }
  bool Init(bool thinking, bool bc) {
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
    std::filesystem::path path =
        std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".";
    path /=
        std::getenv("TEST_WORKSPACE") ? std::getenv("TEST_WORKSPACE") : "_main";
    path /= "lib/text/protocols/chatml_thinking/output.loom";
    std::ifstream f(path);
    if (!f) {
      ADD_FAILURE() << path;
      return false;
    }
    std::string text{std::istreambuf_iterator<char>(f), {}};
    const auto strings_path = path.parent_path().parent_path().parent_path() / "common/strings.loom";
    std::ifstream strings_file(strings_path);
    if (!strings_file) { ADD_FAILURE() << strings_path; return false; }
    const std::string strings{std::istreambuf_iterator<char>(strings_file), {}};
    // These are independent fixture values for the selected revision default.
    // The production output implementation is read from its BUILD-owned file.
    auto decl = text.find("func.decl @chatml.default_thinking() -> (i1)");
    text.replace(
        decl,
        std::string("func.decl @chatml.default_thinking() -> (i1)").size(),
        std::string("func.def @chatml.default_thinking() -> (i1) { %v = "
                    "scalar.constant ") +
            (thinking ? "true" : "false") + " : i1 func.return %v : i1 }");
    // Parse the output provider and its real shared utilities separately.
    // Concatenating a declaration and definition into one source unit is invalid;
    // native MERGE resolves that interface without any fixture helper shim.
    auto parse = [&](const std::string& contents, const char* identifier) {
      loomc_source_options_t so{};
      so.format = LOOMC_SOURCE_FORMAT_TEXT;
      so.identifier = loomc_make_cstring_view(identifier);
      so.contents = {reinterpret_cast<const uint8_t*>(contents.data()), contents.size()};
      so.storage = LOOMC_SOURCE_STORAGE_BORROWED;
      loomc_source_t* raw = nullptr;
      if (!Ok(loomc_source_create(&so, loomc_allocator_system(), &raw)))
        return Owner<loomc_module_t, loomc_module_release>(nullptr, loomc_module_release);
      Owner<loomc_source_t, loomc_source_release> source(raw, loomc_source_release);
      loomc_module_t* parsed_module = nullptr;
      loomc_result_t* result = nullptr;
      const auto status = loomc_module_deserialize_from_source(
          c, w, raw, nullptr, loomc_allocator_system(), &parsed_module, &result);
      Owner<loomc_module_t, loomc_module_release> module(parsed_module, loomc_module_release);
      Owner<loomc_result_t, loomc_result_release> parsed(result, loomc_result_release);
      if (!Ok(status) || !Result(result)) module.reset();
      return module;
    };
    auto output = parse(text, "output-fixture");
    auto utilities = parse(strings, "text-utilities");
    if (!output || !utilities) return false;
    loomc_linker_t* linker_raw = nullptr;
    if (!Ok(loomc_linker_create(c, nullptr, loomc_allocator_system(), &linker_raw))) return false;
    Owner<loomc_linker_t, loomc_linker_release> linker(linker_raw, loomc_linker_release);
    const loomc_link_module_provider_t providers[] = {
        {output.get(), loomc_make_cstring_view("output"), LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {utilities.get(), loomc_make_cstring_view("utilities"), LOOMC_LINK_PROVIDER_ROLE_INPUT}};
    loomc_link_options_t link{};
    link.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    link.structure_size = sizeof(link);
    link.mode = LOOMC_LINK_MODE_MERGE;
    link.module_providers = providers;
    link.module_provider_count = 2;
    loomc_module_t* m = nullptr;
    loomc_result_t* r = nullptr;
    auto status = loomc_link_module(linker_raw, w, &link, &m, &r);
    Owner<loomc_module_t, loomc_module_release> module(m, loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> merged(r, loomc_result_release);
    if (!Ok(status) || !Result(r)) return false;
    if (bc) {
      loomc_module_serialize_options_t serialization{};
      serialization.format = LOOMC_SOURCE_FORMAT_BYTECODE;
      loomc_source_t *bytes = nullptr;
      if (!Ok(loomc_module_serialize_to_source(
              m, &serialization, loomc_allocator_system(), &bytes)))
        return false;
      Owner<loomc_source_t, loomc_source_release> stored(bytes,
                                                         loomc_source_release);
      loomc_module_t *restored = nullptr;
      loomc_result_t *rr = nullptr;
      status = loomc_module_deserialize_from_source(
          c, w, bytes, nullptr, loomc_allocator_system(), &restored, &rr);
      Owner<loomc_result_t, loomc_result_release> result(rr,
                                                         loomc_result_release);
      module.reset(restored);
      if (!Ok(status) || !Result(rr))
        return false;
    }
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
        loomc_make_cstring_view("chatml.output_initial")};
    loomc_emit_options_t emit{};
    emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit.structure_size = sizeof(emit);
    emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
    loomc_compile_artifact_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
    options.structure_size = sizeof(options);
    options.roots = roots;
    options.root_count = 3;
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
    if (!NativeOk(
            iree_vm_program_create({bytecode, iree_vm_module_span_empty()},
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
    std::vector<iree_vm_variant_t> args = {iree_vm_variant_from_i32(thinking),
                                           iree_vm_variant_from_i32(
                                               generation)},
                                   out(1);
    int64_t state = -1;
    if (Call("chatml.output_initial", args, out))
      EXPECT_TRUE(NativeOk(iree_vm_i64_from_variant(out[0], &state)));
    iree_vm_variant_span_reset({out.data(), out.size()});
    return state;
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
  std::pair<std::string, int32_t> Complete(const std::string &prompt,
                                           const std::string &content,
                                           const std::string &thought,
                                           const std::string &tools,
                                           int64_t initial) {
    std::vector<iree_vm_variant_t> args = {Buffer(prompt),
                                           Buffer("unused raw response"),
                                           Buffer(content),
                                           Buffer(thought),
                                           Buffer(tools),
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

private:
  Owner<loomc_target_environment_t, loomc_target_environment_release> env{
      nullptr, loomc_target_environment_release};
  Owner<loomc_context_t, loomc_context_release> context{nullptr,
                                                        loomc_context_release};
  Owner<loomc_workspace_t, loomc_workspace_release> workspace{
      nullptr, loomc_workspace_release};
  loomc_byte_span_t image{};
  iree_vm_environment_t *runtime = nullptr;
  iree_vm_module_t *bytecode = nullptr;
  iree_vm_program_t *program = nullptr;
  iree_vm_process_t *process = nullptr;
  iree_vm_invocation_t *invocation = nullptr;
  iree_vm_ref_types_t types{};
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage{};
};
struct Fixture {
  std::string response, reasoning, content, tail;
  int64_t state;
};
void Check(VM &vm, const Fixture &f, size_t split, bool bytewise) {
  int64_t cursor = 0, state = f.state;
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
TEST(QwenOutput, EverySplitAndBytewiseGrowthFromSourceAndBytecode) {
  for (bool bc : {false, true})
    for (bool thinking : {false, true}) {
      VM vm;
      ASSERT_TRUE(vm.Init(thinking, bc));
      EXPECT_EQ(vm.Initial(false, true), 1);
      EXPECT_EQ(vm.Initial(true, true), 2);
      EXPECT_EQ(vm.Initial(false, false), 17);
      EXPECT_EQ(vm.Initial(true, false), 17);
      std::vector<Fixture> fixtures = {
          {"reason</think>\n\nanswer", "reason", "answer", "", 2},
          {"reason</think>answer</think>literal", "reason",
           "answer</think>literal", "", 2},
          {"reason</think>\nanswer", "reason", "answer", "", 2},
          {"reason</think>\n\n\nanswer", "reason", "\nanswer", "", 2},
          {"reason</think>", "reason", "", "", 2},
          {"reason</thi", "reason</thi", "", "", 2},
          {"reason 🦙 café 中", "reason 🦙 café 中", "", "", 2},
          {"answer 🦙 café 中", "", "answer 🦙 café 中", "", 1},
          {"answer<tool_cal", "", "answer<tool_cal", "", 1},
          {"answer\n\n<tool_call>\n<function=f>\n</function>\n</tool_call>", "",
           "answer\n\n", "<tool_call>\n<function=f>\n</function>\n</tool_call>",
           1},
          {"reason<tool_call>broken", "reason", "", "<tool_call>broken", 2},
          {"<tool_call>broken", "", "", "<tool_call>broken", 2},
          {"<tool_call>broken", "", "", "<tool_call>broken", 1},
          {"reason</think>\n\nanswer<tool_call>x", "reason", "answer",
           "<tool_call>x", 2},
          {"", "", "", "", 0},
          {"raw bytes", thinking ? "raw bytes" : "",
           thinking ? "" : "raw bytes", "", 0},
          {"explicit false", "", "explicit false", "", 1},
          {"no prefix </think>", "", "no prefix </think>", "", 17},
      };
      const std::string tool_marker = "<tool_call>", think_marker = "</think>";
      for (size_t n = 1; n < tool_marker.size(); ++n) {
        auto partial = tool_marker.substr(0, n);
        fixtures.push_back({"answer" + partial, "", "answer" + partial, "", 1});
        fixtures.push_back({"reason" + partial, "reason" + partial, "", "", 2});
      }
      for (size_t n = 1; n < think_marker.size(); ++n) {
        auto partial = think_marker.substr(0, n);
        fixtures.push_back({"reason" + partial, "reason" + partial, "", "", 2});
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
      ASSERT_TRUE(vm.Init(thinking, bc));
      const std::string prefix = "history<|im_start|>assistant\n<think>\n";
      auto result =
          vm.Complete(prefix, "\u3000 answer \u00a0", "\u2003 thought \u202f",
                      "<tool_call>x</tool_call>", 2);
      EXPECT_EQ(result.first,
                prefix + "thought\n</think>\n\nanswer\n\n<tool_call>x</"
                         "tool_call><|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      const std::string disabled = prefix + "\n</think>\n\n";
      result = vm.Complete(disabled, " answer ", "", "", 1);
      EXPECT_EQ(result.first, disabled + "answer<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      result = vm.Complete("history", " answer ", " thought ", "", 17);
      EXPECT_EQ(result.first,
                prefix + "thought\n</think>\n\nanswer<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      result = vm.Complete("history", "", "", "<tool_call>x</tool_call>", 17);
      EXPECT_EQ(result.first,
                prefix + "\n</think>\n\n<tool_call>x</tool_call><|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      result = vm.Complete(prefix, "", " truncated thought ", "", 2);
      EXPECT_EQ(result.first,
                prefix + "truncated thought\n</think>\n\n<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
      // DEFAULT mode uses the declared revision default in reconstruction too.
      result = vm.Complete(thinking ? prefix : disabled, " answer ",
                           thinking ? " thought " : "", "", 0);
      EXPECT_EQ(result.first,
                thinking ? prefix + "thought\n</think>\n\nanswer<|im_end|>\n"
                         : disabled + "answer<|im_end|>\n");
      EXPECT_EQ(result.second, 0);
    }
}
} // namespace
