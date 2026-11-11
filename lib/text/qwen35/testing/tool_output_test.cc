// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "iree/vm/buffer.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/runtime/program/json.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/service/text/tool_xml.h"
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
    iree_vm_module_release(tools);
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
    auto read = [](const char *env, const char *leaf) {
      std::filesystem::path path =
          std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".";
      path /= std::getenv("TEST_WORKSPACE") ? std::getenv("TEST_WORKSPACE")
                                            : "_main";
      path /= "lib/text";
      path /= leaf;
      if (const char *external = std::getenv(env))
        path = external;
      std::ifstream f(path);
      if (!f)
        ADD_FAILURE() << path;
      return std::string(std::istreambuf_iterator<char>(f), {});
    };
    unsigned source_id = 0;
    auto parse = [&](const std::string &text) {
      loomc_source_options_t so{};
      so.format = LOOMC_SOURCE_FORMAT_TEXT;
      const auto identity =
          "tool-output-fixture-" + std::to_string(source_id++);
      so.identifier = loomc_make_cstring_view(identity.c_str());
      so.contents = {reinterpret_cast<const uint8_t *>(text.data()),
                     text.size()};
      so.storage = LOOMC_SOURCE_STORAGE_BORROWED;
      loomc_source_t *raw = nullptr;
      if (!Ok(loomc_source_create(&so, loomc_allocator_system(), &raw)))
        return Owner<loomc_module_t, loomc_module_release>(
            nullptr, loomc_module_release);
      Owner<loomc_source_t, loomc_source_release> source(raw,
                                                         loomc_source_release);
      loomc_module_t *m = nullptr;
      loomc_result_t *r = nullptr;
      auto status = loomc_module_deserialize_from_source(
          c, w, raw, nullptr, loomc_allocator_system(), &m, &r);
      Owner<loomc_result_t, loomc_result_release> result(r,
                                                         loomc_result_release);
      Owner<loomc_module_t, loomc_module_release> module(m,
                                                         loomc_module_release);
      if (!Ok(status) || !Result(r))
        module.reset();
      return module;
    };
    auto common = parse(read("TEXT_STRINGS_SOURCE_PATH", "common/strings.loom"));
    auto adapter =
        parse(read("TEXT_TOOL_OUTPUT_SOURCE_PATH", "protocols/chatml_thinking/tool_output.loom"));
    auto fixture = parse(R"(
      func.decl @chatml.normalize_tool_output(%schemas: buffer, %response: buffer) -> (buffer)
      func.decl import("tools") @parse_xml(%schemas: buffer, %response: buffer, %truncated: i64) -> (buffer)
      func.def public @tool_fixture(%schemas: buffer, %response: buffer, %enabled: i32) -> (buffer) {
        %zero = scalar.constant 0 : i32
        %normalize = scalar.cmpi ne, %enabled, %zero : i32
        %text = scf.if %normalize -> (buffer) {
          %value = func.call @chatml.normalize_tool_output(%schemas, %response) : (buffer, buffer) -> (buffer)
          scf.yield %value : buffer
        } else { scf.yield %response : buffer }
        %complete = scalar.constant 0 : i64
        %calls = func.call @parse_xml(%schemas, %text, %complete) : (buffer, buffer, i64) -> (buffer)
        func.return %calls : buffer
      }
    )");
    if (!common || !adapter || !fixture)
      return false;
    loomc_link_module_provider_t providers[] = {
        {common.get(), loomc_make_cstring_view("common"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {adapter.get(), loomc_make_cstring_view("adapter"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {fixture.get(), loomc_make_cstring_view("fixture"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT}};
    loomc_linker_t *l = nullptr;
    if (!Ok(loomc_linker_create(c, nullptr, loomc_allocator_system(), &l)))
      return false;
    Owner<loomc_linker_t, loomc_linker_release> linker(l, loomc_linker_release);
    loomc_link_options_t link{};
    link.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    link.structure_size = sizeof(link);
    link.mode = LOOMC_LINK_MODE_MERGE;
    link.module_providers = providers;
    link.module_provider_count = 3;
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    auto status = loomc_link_module(l, w, &link, &m, &r);
    Owner<loomc_result_t, loomc_result_release> linked(r, loomc_result_release);
    Owner<loomc_module_t, loomc_module_release> module(m, loomc_module_release);
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
        loomc_make_cstring_view("chatml.normalize_tool_output"),
        loomc_make_cstring_view("tool_fixture")};
    loomc_emit_options_t emit{};
    emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit.structure_size = sizeof(emit);
    emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
    loomc_compile_artifact_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
    options.structure_size = sizeof(options);
    options.roots = roots;
    options.root_count = 2;
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
    if (!NativeOk(loom_serve_text_tool_xml_module_create(
            runtime, iree_allocator_system(), &tools)))
      return false;
    iree_vm_module_t *libraries[] = {json, tools};
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
            std::vector<iree_vm_variant_t> &out, bool expect_ok = true) {
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
    auto status = iree_vm_invoke(invocation, fn, {args.data(), args.size()},
                                 {out.data(), out.size()});
    bool ok = iree_status_is_ok(status);
    if (expect_ok)
      NativeOk(status);
    else
      iree_status_ignore(status);
    iree_vm_variant_span_reset({args.data(), args.size()});
    return ok;
  }
  std::pair<bool, std::string> Run(const std::string &schemas,
                                   const std::string &response,
                                   bool parse = false, bool enabled = true,
                                   bool expect_ok = true) {
    std::vector<iree_vm_variant_t> args = {Buffer(schemas), Buffer(response)},
                                   out(1);
    if (parse)
      args.push_back(iree_vm_variant_from_i32(enabled));
    bool ok = Call(parse ? "tool_fixture" : "chatml.normalize_tool_output",
                   args, out, expect_ok);
    std::string result;
    if (ok) {
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
    return {ok, result};
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
  iree_vm_module_t *tools = nullptr;
  iree_vm_program_t *program = nullptr;
  iree_vm_process_t *process = nullptr;
  iree_vm_invocation_t *invocation = nullptr;
  iree_vm_ref_types_t types{};
  alignas(iree_max_align_t) std::array<uint8_t, 32768> storage{};
};
const std::string kSchemas = R"([
 {"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"b":{"type":"boolean"},"c":{"type":"boolean"},"s":{"type":"string"},"obj":{"type":"object"},"arr":{"type":"array"}},"required":["b"]}}},
 {"type":"function","function":{"name":"g","parameters":{"type":"object","properties":{"b":{"type":"string"}},"required":["b"]}}}
])";
std::string CallText(const std::string &name, const std::string &parameters) {
  return "<tool_call>\n<function=" + name + ">\n" + parameters +
         "</function>\n</tool_call>";
}
std::string Parameter(const std::string &name, const std::string &value) {
  return "<parameter=" + name + ">\n" + value + "\n</parameter>\n";
}
TEST(QwenToolOutput,
     BooleanNormalizationAndActualNativeRecordsFromSourceAndBytecode) {
  for (bool bc : {false, true}) {
    VM vm;
    ASSERT_TRUE(vm.Init(bc));
    auto upper = CallText("f", Parameter("b", "True"));
    auto lower = CallText("f", Parameter("b", "true"));
    EXPECT_EQ(vm.Run(kSchemas, upper).second, lower);
    EXPECT_EQ(vm.Run(kSchemas, lower).second, lower);
    EXPECT_EQ(
        vm.Run(kSchemas, upper, true).second,
        R"([{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}}])");
    // The selected27B branch bypasses normalization: lowercase JSON succeeds,
    // Python Boolean spelling is rejected by the actual native codec.
    EXPECT_EQ(
        vm.Run(kSchemas, lower, true, false).second,
        R"([{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}}])");
    EXPECT_FALSE(vm.Run(kSchemas, upper, true, false, false).first);
    upper = CallText("f", Parameter("b", "False") + Parameter("c", "True"));
    lower = CallText("f", Parameter("b", "false") + Parameter("c", "true"));
    EXPECT_EQ(vm.Run(kSchemas, upper).second, lower);
    EXPECT_EQ(
        vm.Run(kSchemas, upper, true).second,
        R"([{"type":"function","function":{"name":"f","arguments":"{\"b\":false,\"c\":true}"}}])");
    const std::string framed =
        " \r\n<tool_call> \t<function=f>\r\n<parameter=b>\r\n \tTrue "
        "\r\n</parameter>\r\n</function> \r\n</tool_call>\t ";
    const std::string expected =
        " \r\n<tool_call> \t<function=f>\r\n<parameter=b>\r\n \ttrue "
        "\r\n</parameter>\r\n</function> \r\n</tool_call>\t ";
    EXPECT_EQ(vm.Run(kSchemas, framed).second, expected);
    EXPECT_EQ(
        vm.Run(kSchemas, framed, true).second,
        R"([{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}}])");
    const std::string mixed = CallText(
        "f", Parameter("b", "True") + Parameter("s", "True") +
                 Parameter("obj", R"({"nested":true,"word":"False"})") +
                 Parameter("arr", R"([true,"True"])"));
    const std::string normalized = CallText(
        "f", Parameter("b", "true") + Parameter("s", "True") +
                 Parameter("obj", R"({"nested":true,"word":"False"})") +
                 Parameter("arr", R"([true,"True"])"));
    EXPECT_EQ(vm.Run(kSchemas, mixed).second, normalized);
    EXPECT_EQ(
        vm.Run(kSchemas, mixed, true).second,
        R"([{"type":"function","function":{"name":"f","arguments":"{\"b\":true,\"s\":\"True\",\"obj\":{\"nested\":true,\"word\":\"False\"},\"arr\":[true,\"True\"]}"}}])");
    const auto string_call = CallText("g", Parameter("b", "True"));
    EXPECT_EQ(vm.Run(kSchemas, string_call).second, string_call);
    EXPECT_EQ(
        vm.Run(kSchemas, string_call, true).second,
        R"([{"type":"function","function":{"name":"g","arguments":"{\"b\":\"True\"}"}}])");
    const auto opaque = CallText("f", Parameter("b", "True") +
                                          Parameter("s", "<parameter=b>True"));
    EXPECT_EQ(
        vm.Run(kSchemas, opaque, true).second,
        R"([{"type":"function","function":{"name":"f","arguments":"{\"b\":true,\"s\":\"<parameter=b>True\"}"}}])");
    const auto multi =
        CallText("f", Parameter("b", "True")) + "\n" + string_call;
    EXPECT_EQ(
        vm.Run(kSchemas, multi, true).second,
        R"([{"type":"function","function":{"name":"f","arguments":"{\"b\":true}"}},{"type":"function","function":{"name":"g","arguments":"{\"b\":\"True\"}"}}])");
  }
}
TEST(QwenToolOutput, UnknownAndMalformedStayNativeValidationAuthority) {
  for (bool bc : {false, true}) {
    VM vm;
    ASSERT_TRUE(vm.Init(bc));
    const std::vector<std::string> unchanged = {
        CallText("unknown", Parameter("b", "True")),
        CallText("f", Parameter("unknown", "True")),
        CallText("f", Parameter("b", "TRUE")),
        CallText("f", Parameter("b", "\"True\"")),
        CallText("f",
                 Parameter("b", "true") + Parameter("obj", R"({"b":True})")),
        "<tool_call><function=f><parameter=b>True</parameter></function>",
        "<tool_call><function=f><parameter=b>True</"
        "parameter><parameter=c>False</function></tool_call>",
        "<tool_call><function=f><parameter=b>True</parameter></function></"
        "tool_call>trailing",
    };
    for (const auto &text : unchanged) {
      SCOPED_TRACE(text);
      EXPECT_EQ(vm.Run(kSchemas, text).second, text);
      EXPECT_FALSE(vm.Run(kSchemas, text, true, true, false).first);
    }
    // Duplicate/required checks remain the actual codec's responsibility.
    EXPECT_FALSE(
        vm.Run(kSchemas,
               CallText("f", Parameter("b", "True") + Parameter("b", "False")),
               true, true, false)
            .first);
    EXPECT_FALSE(vm.Run(kSchemas, CallText("f", Parameter("s", "True")), true,
                        true, false)
                     .first);
    for (const std::string &bad_name :
         {std::string("bad key"), std::string("café"), std::string(129, 'x')}) {
      const std::string schemas =
          "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":"
          "{\"type\":\"object\",\"properties\":{\"" +
          bad_name + "\":{\"type\":\"boolean\"}}}}}]";
      const auto invalid_name = CallText("f", Parameter(bad_name, "True"));
      EXPECT_EQ(vm.Run(schemas, invalid_name).second, invalid_name);
      EXPECT_FALSE(vm.Run(schemas, invalid_name, true, true, false).first);
    }
    const auto text = CallText("f", Parameter("b", "True"));
    EXPECT_EQ(vm.Run("null", text).second,
              text); // explicitly outside admitted schemas
    EXPECT_EQ(vm.Run("[]", text).second, text);
    EXPECT_FALSE(vm.Run("[]", text, true, true, false).first);
  }
}
} // namespace
