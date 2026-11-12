// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "loomc/loomc.h"
#include "loomc/target/vm.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
namespace {
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
bool Ok(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << std::string(message.data, message.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto *d = loomc_result_diagnostic_at(result, i);
      ADD_FAILURE() << std::string(d->code.data, d->code.size) << ": "
                    << std::string(d->message.data, d->message.size);
    }
  return false;
}
TEST(Qwen3Composition, BothChatRootsCompileTogetherFromSourceAndBytecode) {
  for (const char *format : {"text.loom", "bc.loombc"}) {
    SCOPED_TRACE(format);
    const char *source_dir = std::getenv("TEST_SRCDIR");
    const char *workspace_name = std::getenv("TEST_WORKSPACE");
    ASSERT_TRUE(source_dir && workspace_name);
    const auto path = std::filesystem::path(source_dir) / workspace_name /
                      "lib/text/qwen3" / (std::string("composition_") + format);
    loomc_target_environment_t *target_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_target_environment_create_vm(loomc_allocator_system(),
                                                      &target_raw)));
    Owner<loomc_target_environment_t, loomc_target_environment_release> target(
        target_raw, loomc_target_environment_release);
    loomc_context_target_options_t target_options{};
    target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.structure_size = sizeof(target_options);
    target_options.target_environment = target.get();
    loomc_context_options_t options{};
    options.next = &target_options;
    loomc_context_t *context_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(),
                                        &context_raw)));
    Owner<loomc_context_t, loomc_context_release> context(
        context_raw, loomc_context_release);
    loomc_workspace_t *workspace_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                          &workspace_raw)));
    Owner<loomc_workspace_t, loomc_workspace_release> workspace(
        workspace_raw, loomc_workspace_release);
    loomc_module_t *module_raw = nullptr;
    loomc_result_t *result_raw = nullptr;
    auto status = loomc_module_deserialize_from_path(
        context.get(), workspace.get(),
        loomc_make_cstring_view(path.string().c_str()), nullptr,
        loomc_allocator_system(), &module_raw, &result_raw);
    Owner<loomc_module_t, loomc_module_release> module(module_raw,
                                                       loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> result(result_raw,
                                                       loomc_result_release);
    ASSERT_TRUE(Ok(status));
    ASSERT_TRUE(Succeeded(result.get()));
    // Match configured package assembly: public source data resolves the
    // selected product declarations before compiling the two public roots.
    const loomc_string_view_t link_roots[] = {
        loomc_make_cstring_view("chat_begin"),
        loomc_make_cstring_view("chat_message"),
        loomc_make_cstring_view("loomify.text.tools_prefix"),
        loomc_make_cstring_view("loomify.text.tools_suffix"),
        loomc_make_cstring_view("loomify.text.effort_xhigh"),
        loomc_make_cstring_view("loomify.text.effort_low"),
        loomc_make_cstring_view("loomify.text.default_effort"),
        loomc_make_cstring_view("loomify.text.default_effort_instruction"),
        loomc_make_cstring_view("loomify.text.model_name")};
    loomc_linker_t *linker_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_linker_create(context.get(), nullptr,
                                       loomc_allocator_system(), &linker_raw)));
    Owner<loomc_linker_t, loomc_linker_release> linker(linker_raw,
                                                       loomc_linker_release);
    const loomc_link_module_provider_t provider{
        module.get(), loomc_make_cstring_view("composition"),
        LOOMC_LINK_PROVIDER_ROLE_INPUT};
    loomc_link_options_t link{};
    link.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    link.structure_size = sizeof(link);
    link.mode = LOOMC_LINK_MODE_LINK;
    link.module_providers = &provider;
    link.module_provider_count = 1;
    link.root_symbols = link_roots;
    link.root_symbol_count = sizeof(link_roots) / sizeof(link_roots[0]);
    loomc_module_t *linked_raw = nullptr;
    result_raw = nullptr;
    status = loomc_link_module(linker.get(), workspace.get(), &link,
                               &linked_raw, &result_raw);
    result.reset(result_raw);
    Owner<loomc_module_t, loomc_module_release> linked(linked_raw,
                                                       loomc_module_release);
    ASSERT_TRUE(Ok(status));
    ASSERT_TRUE(Succeeded(result.get()));
    module = std::move(linked);
    loomc_compiler_t *compiler_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_compiler_create(
        context.get(), nullptr, loomc_allocator_system(), &compiler_raw)));
    Owner<loomc_compiler_t, loomc_compiler_release> compiler(
        compiler_raw, loomc_compiler_release);
    loomc_target_profile_t *profile_raw = nullptr;
    ASSERT_TRUE(Ok(loomc_target_profile_select(
        target.get(), loomc_make_cstring_view("vm:core"),
        loomc_allocator_system(), &profile_raw)));
    Owner<loomc_target_profile_t, loomc_target_profile_release> profile(
        profile_raw, loomc_target_profile_release);
    // Each root alone compiled before the fix. The real regression requires
    // both callers of the shared JSON formatter in one VM compilation.
    const loomc_string_view_t roots[] = {
        loomc_make_cstring_view("chat_begin"),
        loomc_make_cstring_view("chat_message")};
    loomc_emit_options_t emit{};
    emit.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit.structure_size = sizeof(emit);
    emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM);
    loomc_compile_artifact_options_t compile{};
    compile.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
    compile.structure_size = sizeof(compile);
    compile.roots = roots;
    compile.root_count = 2;
    compile.target_profile = profile.get();
    compile.emit_options = &emit;
    result_raw = nullptr;
    status = loomc_compile_artifact(compiler.get(), workspace.get(), nullptr,
                                    module.get(), &compile,
                                    loomc_allocator_system(), &result_raw);
    result.reset(result_raw);
    ASSERT_TRUE(Ok(status));
    ASSERT_TRUE(Succeeded(result.get()));
  }
}
} // namespace
