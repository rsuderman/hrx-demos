// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/runtime/program/json.h"
#include "loomc/loomc.h"
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
    iree_vm_module_release(json);
    iree_vm_environment_free(runtime);
    loomc_allocator_free(loomc_allocator_system(),
                         const_cast<uint8_t *>(image.data));
  }
  bool Init(bool bc) {
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
    path /= "lib/text/common/json.loom";
    // External SDK contract gates place this test in an admitted Serve package.
    // Keep source selection explicit without changing the production module.
    if (const char *external = std::getenv("TEXT_JSON_SOURCE_PATH"))
      path = external;
    std::ifstream f(path);
    if (!f) {
      ADD_FAILURE() << path;
      return false;
    }
    std::string text{std::istreambuf_iterator<char>(f), {}};
    loomc_source_options_t so{};
    so.format = LOOMC_SOURCE_FORMAT_TEXT;
    so.identifier = loomc_make_cstring_view("json-fixture");
    so.contents = {reinterpret_cast<const uint8_t *>(text.data()), text.size()};
    so.storage = LOOMC_SOURCE_STORAGE_BORROWED;
    loomc_source_t *raw = nullptr;
    if (!Ok(loomc_source_create(&so, loomc_allocator_system(), &raw)))
      return false;
    Owner<loomc_source_t, loomc_source_release> source(raw,
                                                       loomc_source_release);
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    auto status = loomc_module_deserialize_from_source(
        c, w, raw, nullptr, loomc_allocator_system(), &m, &r);
    Owner<loomc_module_t, loomc_module_release> module(m, loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> parsed(r, loomc_result_release);
    if (!Ok(status) || !Result(r))
      return false;
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
        loomc_make_cstring_view("text.json_dump")};
    loomc_emit_options_t emit{};
    emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit.structure_size = sizeof(emit);
    emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
    loomc_compile_artifact_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
    options.structure_size = sizeof(options);
    options.roots = roots;
    options.root_count = 1;
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
            runtime, iree_allocator_system(), &json)))
      return false;
    if (!NativeOk(iree_vm_program_create(
            {bytecode, iree_vm_module_span_from_ptr(&json, 1)},
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
  std::string Dump(const std::string &text) {
    std::vector<iree_vm_variant_t> args = {Buffer(text)}, out(1);
    std::string result;
    if (Call("text.json_dump", args, out)) {
      void *ptr = nullptr;
      EXPECT_TRUE(NativeOk(
          iree_vm_ptr_from_variant_borrowed(out[0], types.buffer, &ptr)));
      if (ptr) {
        auto b = static_cast<iree_vm_buffer_t *>(ptr);
        result.assign(static_cast<const char *>(iree_vm_buffer_const_data(b)),
                      iree_vm_buffer_length(b));
      }
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
  iree_vm_module_t *json = nullptr;
  iree_vm_program_t *program = nullptr;
  iree_vm_process_t *process = nullptr;
  iree_vm_invocation_t *invocation = nullptr;
  iree_vm_ref_types_t types{};
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage{};
};
// Anchored independently against HF4.57.1's actual chat-template tojson
// override: json.dumps(ensure_ascii=False, indent=None, separators=None,
// sort_keys=False). Inputs obey the caller's strict unique-key JSON contract.
TEST(QwenJson, ExactNonFloatingBytesFromSourceAndBytecodeWithNativeJson) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"null", "null"},
      {" true ", "true"},
      {" false ", "false"},
      {"-0", "0"},
      {"0", "0"},
      {"-1", "-1"},
      {"999999999999999999999999999999999999999",
       "999999999999999999999999999999999999999"},
      {"-999999999999999999999999999999999999999",
       "-999999999999999999999999999999999999999"},
      {R"( { "z" : [1,true,null, -0], "a": {"empty":{}, "array":[]} } )",
       R"({"z": [1, true, null, 0], "a": {"empty": {}, "array": []}})"},
      {R"(["", "with , and : inside", "  significant  ","-0", -0])",
       R"(["", "with , and : inside", "  significant  ", "-0", 0])"},
      {R"({"\u007a": "\u0061", "\u0022": "\u005c", "\/": "\/"})",
       R"({"z": "a", "\"": "\\", "/": "/"})"},
      {R"("\u003c\u003e\u0026\u0027/")", R"("<>&'/")"},
      {R"("caf\u00e9 \u4e2d \ud83e\udd99")", "\"café 中 🦙\""},
      {"\"café 中 🦙\"", "\"café 中 🦙\""},
      {R"("\u2028\u2029\u00a0")", "\"\u2028\u2029\u00a0\""},
      {R"("\u0000\u0001\u0007\u0008\u0009\u000a\u000b\u000c\u000d\u000e\u001f")",
       R"("\u0000\u0001\u0007\b\t\n\u000b\f\r\u000e\u001f")"},
      {R"("quote:\" slash:\\ solidus:\/ backspace:\b tab:\t newline:\n formfeed:\f return:\r")",
       R"("quote:\" slash:\\ solidus:/ backspace:\b tab:\t newline:\n formfeed:\f return:\r")"},
      {R"({"b": 2, "a": 1, "nested": [{"second":false,"first":true}]})",
       R"({"b": 2, "a": 1, "nested": [{"second": false, "first": true}]})"},
      {R"({"\u000a": "x", "\u003cscript\u003e": "\u0026"})",
       R"({"\n": "x", "<script>": "&"})"},
  };
  for (bool bc : {false, true}) {
    VM vm;
    ASSERT_TRUE(vm.Init(bc));
    for (const auto &test : cases) {
      SCOPED_TRACE(::testing::Message()
                   << "bc=" << bc << " input=" << test.first);
      EXPECT_EQ(vm.Dump(test.first), test.second);
      EXPECT_EQ(vm.Dump(test.second), test.second);
    }
    // Exercise expansion/storage with all ASCII controls, not only the
    // representative short-escape anchors above.
    const std::string hex = "0123456789abcdef";
    std::string input = "\"", expected = "\"";
    for (unsigned i = 0; i < 32; ++i) {
      std::string escaped = "\\u00";
      escaped += hex[i / 16];
      escaped += hex[i % 16];
      input += escaped;
      switch (i) {
      case 8:
        expected += "\\b";
        break;
      case 9:
        expected += "\\t";
        break;
      case 10:
        expected += "\\n";
        break;
      case 12:
        expected += "\\f";
        break;
      case 13:
        expected += "\\r";
        break;
      default:
        expected += escaped;
      }
    }
    input += '"';
    expected += '"';
    EXPECT_EQ(vm.Dump(input), expected);
  }
}
// This is the explicitly deferred numeric behavior, not a claim of Python
// float formatting parity: e.g. HF's 1e0 becomes1.0, while our spelling stays.
TEST(QwenJson, DeferredFloatLexemesRemainUnchangedFromSourceAndBytecode) {
  for (bool bc : {false, true}) {
    VM vm;
    ASSERT_TRUE(vm.Init(bc));
    EXPECT_EQ(vm.Dump("[1e0,1.0,-0.0,1e-0,-0e0,1E+03,-1.25e-12,0,-0]"),
              "[1e0, 1.0, -0.0, 1e-0, -0e0, 1E+03, -1.25e-12, 0, 0]");
  }
}
} // namespace
