// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/registry/emission/consumer_emitter.h"

#include <cstring>
#include <memory>
#include <string>

#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/pass.h"
#include "loomc/source.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf {
namespace {

template <typename T, void (*Release)(T *)> struct Releaser {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Releaser<T, Release>>;
using Context = Owner<loomc_context_t, loomc_context_release>;
using Workspace = Owner<loomc_workspace_t, loomc_workspace_release>;
using Module = Owner<loomc_module_t, loomc_module_release>;
using Result = Owner<loomc_result_t, loomc_result_release>;
using Source = Owner<loomc_source_t, loomc_source_release>;
using Index = Owner<loomc_link_index_t, loomc_link_index_release>;
using Builder =
    Owner<loomc_link_index_builder_t, loomc_link_index_builder_free>;
using PassProgram = Owner<loomc_pass_program_t, loomc_pass_program_release>;

bool CheckStatus(loomc_status_t status) {
  const bool ok = loomc_status_is_ok(status);
  if (!ok) {
    const auto message = loomc_status_message(status);
    ADD_FAILURE() << loomc_status_code_string(loomc_status_code(status)) << ": "
                  << std::string(message.data ? message.data : "",
                                 message.size);
  }
  loomc_status_free(status);
  return ok;
}

std::string Diagnostics(const loomc_result_t *result) {
  std::string text;
  if (!result)
    return text;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const auto *diagnostic = loomc_result_diagnostic_at(result, i);
    text.append(diagnostic->message.data, diagnostic->message.size);
    text.push_back('\n');
  }
  return text;
}

constexpr char kProvider[] = R"loom(
template.decl @ggml.qk.scale_min(%weight: buffer, %base: offset, %group: index) -> (f32, f32)
template.def<@ggml.qk.scale_min> device @test_scale_min(%weight: buffer, %base: offset, %group: index) -> (f32, f32) {
  %scale = scalar.constant 2.0 : f32
  %minimum = scalar.constant 3.0 : f32
  template.return %scale, %minimum : f32, f32
}
func.def public @unrelated_provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)loom";

constexpr char kWrongProvider[] = R"loom(
template.decl @ggml.qk.scale_min(%weight: buffer, %base: offset, %group: index) -> (i32, i32)
template.def<@ggml.qk.scale_min> device @wrong_scale_min(%weight: buffer, %base: offset, %group: index) -> (i32, i32) {
  %value = scalar.constant 2 : i32
  template.return %value, %value : i32, i32
}
)loom";

class ConsumerEmitterTest : public ::testing::Test {
protected:
  void SetUp() override {
    loomc_context_t *context = nullptr;
    ASSERT_TRUE(CheckStatus(
        loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    context_.reset(context);
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(CheckStatus(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    workspace_.reset(workspace);
  }

  Module Parse(const char *text) {
    loomc_source_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
    options.structure_size = sizeof(options);
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier = loomc_make_cstring_view("emitter_test.loom");
    options.contents = loomc_make_byte_span(text, std::strlen(text));
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t *source = nullptr;
    if (!CheckStatus(
            loomc_source_create(&options, loomc_allocator_system(), &source)))
      return {};
    Source source_owner(source);
    return Deserialize(source, LOOMC_SOURCE_FORMAT_TEXT);
  }

  Module Deserialize(loomc_source_t *source, loomc_source_format_t format) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    const auto status =
        format == LOOMC_SOURCE_FORMAT_BYTECODE
            ? loomc_module_deserialize_bytecode_from_source(
                  context_.get(), workspace_.get(), source, nullptr,
                  loomc_allocator_system(), &module, &result)
            : loomc_module_deserialize_text_from_source(
                  context_.get(), workspace_.get(), source, nullptr,
                  loomc_allocator_system(), &module, &result);
    Result result_owner(result);
    CheckStatus(status);
    EXPECT_NE(result, nullptr);
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    return Module(module);
  }

  Module Emit() {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    CheckStatus(
        EmitScaleMinProbe(context_.get(), workspace_.get(), &module, &result));
    Result result_owner(result);
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    return Module(module);
  }

  Index MakeIndex(const char *text) {
    Module provider = Parse(text);
    if (!provider)
      return {};
    loomc_link_index_builder_t *builder = nullptr;
    if (!CheckStatus(loomc_link_index_builder_allocate(
            context_.get(), nullptr, loomc_allocator_system(), &builder)))
      return {};
    Builder builder_owner(builder);
    loomc_link_index_source_options_t options = {};
    options.provider_name = loomc_make_cstring_view("test.provider");
    options.role = LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
    if (!CheckStatus(loomc_link_index_builder_add_module(
            builder, provider.get(), &options, nullptr)))
      return {};
    // Snapshot must no longer depend on its source module even before freeze.
    provider.reset();
    loomc_link_index_t *index = nullptr;
    loomc_result_t *result = nullptr;
    CheckStatus(loomc_link_index_builder_finish(builder, &index, &result));
    Result result_owner(result);
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    return Index(index);
  }

  Module Link(loomc_link_index_t *index, const loomc_module_t *input) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    CheckStatus(LinkConsumer(
        context_.get(), workspace_.get(), index, input,
        loomc_make_cstring_view("registry_scale_min_probe"), &module, &result));
    Result result_owner(result);
    EXPECT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
    return Module(module);
  }

  Source Serialize(const loomc_module_t *module, loomc_source_format_t format) {
    loomc_module_serialize_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
    options.structure_size = sizeof(options);
    options.format = format;
    options.text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_GENERIC;
    loomc_source_t *source = nullptr;
    CheckStatus(loomc_module_serialize_to_source(
        module, &options, loomc_allocator_system(), &source));
    return Source(source);
  }

  std::string Text(const loomc_module_t *module) {
    Source source = Serialize(module, LOOMC_SOURCE_FORMAT_TEXT);
    if (!source)
      return {};
    const auto bytes = loomc_source_contents(source.get());
    return std::string(reinterpret_cast<const char *>(bytes.data),
                       bytes.data_length);
  }

  void Verify(loomc_module_t *module) {
    loomc_module_verify_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS;
    options.structure_size = sizeof(options);
    options.scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(CheckStatus(loomc_module_verify(
        module, &options, loomc_allocator_system(), &result)));
    Result result_owner(result);
    ASSERT_TRUE(result && loomc_result_succeeded(result))
        << Diagnostics(result);
  }

  Result SelectTemplates(loomc_module_t *module) {
    loomc_pass_program_t *program = nullptr;
    loomc_result_t *prepared = nullptr;
    CheckStatus(loomc_pass_program_create_from_pipeline_text(
        context_.get(), loomc_make_cstring_view("select-templates{mode=final}"),
        nullptr, loomc_allocator_system(), &program, &prepared));
    PassProgram program_owner(program);
    Result prepared_owner(prepared);
    EXPECT_TRUE(prepared && loomc_result_succeeded(prepared))
        << Diagnostics(prepared);
    if (!program)
      return {};
    loomc_result_t *result = nullptr;
    CheckStatus(loomc_pass_program_run(program, workspace_.get(), module,
                                       nullptr, loomc_allocator_system(),
                                       &result));
    return Result(result);
  }

  void ExpectProbe(const loomc_module_t *module) {
    const std::string text = Text(module);
    EXPECT_NE(text.find("registry_scale_min_probe"), std::string::npos);
    const auto first_store = text.find("view.store");
    ASSERT_NE(first_store, std::string::npos) << text;
    const auto second_store = text.find("view.store", first_store + 1);
    ASSERT_NE(second_store, std::string::npos) << text;
    EXPECT_EQ(text.find("view.store", second_store + 1), std::string::npos);
    EXPECT_NE(text.find("kernel.return"), std::string::npos);
    loomc_module_function_t function = {};
    ASSERT_TRUE(loomc_module_try_lookup_function(
        module, loomc_make_cstring_view("registry_scale_min_probe"),
        &function));
    EXPECT_EQ(function.kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
  }

  Context context_;
  Workspace workspace_;
};

TEST_F(ConsumerEmitterTest, EmitsNewApplicationAndBothOutputStores) {
  Module probe = Emit();
  ASSERT_NE(probe, nullptr);
  ExpectProbe(probe.get());
  EXPECT_NE(Text(probe.get()).find("template.apply<@ggml.qk.scale_min>"),
            std::string::npos);
  Verify(probe.get());
}

TEST_F(ConsumerEmitterTest, LinksSnapshotAndReloadsAfterOriginalOwnersRelease) {
  Index index = MakeIndex(kProvider);
  ASSERT_NE(index, nullptr);
  Module probe = Emit();
  ASSERT_NE(probe, nullptr);
  Module linked = Link(index.get(), probe.get());
  ASSERT_NE(linked, nullptr);
  Verify(linked.get());
  ExpectProbe(linked.get());
  EXPECT_NE(Text(linked.get()).find("test_scale_min"), std::string::npos);
  EXPECT_EQ(Text(linked.get()).find("unrelated_provider"), std::string::npos);
  Source text = Serialize(linked.get(), LOOMC_SOURCE_FORMAT_TEXT);
  Source bytecode = Serialize(linked.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  ASSERT_NE(text, nullptr);
  ASSERT_NE(bytecode, nullptr);
  linked.reset();
  probe.reset();
  index.reset();
  workspace_.reset();
  context_.reset();
  SetUp();
  Module from_text = Deserialize(text.get(), LOOMC_SOURCE_FORMAT_TEXT);
  Module from_bytecode =
      Deserialize(bytecode.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  ASSERT_NE(from_text, nullptr);
  ASSERT_NE(from_bytecode, nullptr);
  for (auto *module : {from_text.get(), from_bytecode.get()}) {
    Verify(module);
    ExpectProbe(module);
    EXPECT_NE(Text(module).find("test_scale_min"), std::string::npos);
  }
}

TEST_F(ConsumerEmitterTest,
       MissingAndIncompatibleProvidersFailThenValidLinkWorks) {
  Module probe = Emit();
  ASSERT_NE(probe, nullptr);
  for (const char *provider : {"", kWrongProvider}) {
    SCOPED_TRACE(provider[0] ? "incompatible provider" : "missing provider");
    Index index = MakeIndex(provider);
    ASSERT_NE(index, nullptr);
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(CheckStatus(
        LinkConsumer(context_.get(), workspace_.get(), index.get(), probe.get(),
                     loomc_make_cstring_view("registry_scale_min_probe"),
                     &module, &result)));
    Module module_owner(module);
    Result result_owner(result);
    ASSERT_NE(result, nullptr);
    if (loomc_result_succeeded(result)) {
      // Linking intentionally retains open template applications. Final native
      // template selection owns the missing-provider diagnostic at this pin.
      ASSERT_NE(module, nullptr);
      Result selected = SelectTemplates(module);
      ASSERT_NE(selected, nullptr);
      EXPECT_FALSE(loomc_result_succeeded(selected.get()));
      EXPECT_FALSE(Diagnostics(selected.get()).empty());
    } else {
      EXPECT_EQ(module, nullptr);
      EXPECT_FALSE(Diagnostics(result).empty());
    }
  }
  Index valid = MakeIndex(kProvider);
  ASSERT_NE(valid, nullptr);
  Module linked = Link(valid.get(), probe.get());
  ASSERT_NE(linked, nullptr);
  Result selected = SelectTemplates(linked.get());
  ASSERT_NE(selected, nullptr);
  EXPECT_TRUE(loomc_result_succeeded(selected.get()))
      << Diagnostics(selected.get());
  Verify(linked.get());
}

TEST_F(ConsumerEmitterTest, RepeatedEmissionsLeaveLibraryInventoryUnchanged) {
  Index index = MakeIndex(kProvider);
  ASSERT_NE(index, nullptr);
  const auto symbols = loomc_link_index_symbol_count(index.get());
  const auto providers = loomc_link_index_provider_count(index.get());
  Module first_probe = Emit();
  Module second_probe = Emit();
  ASSERT_NE(first_probe, nullptr);
  ASSERT_NE(second_probe, nullptr);
  Module first = Link(index.get(), first_probe.get());
  Module second = Link(index.get(), second_probe.get());
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first.get(), second.get());
  Verify(first.get());
  Verify(second.get());
  EXPECT_EQ(Text(first.get()), Text(second.get()));
  EXPECT_EQ(loomc_link_index_symbol_count(index.get()), symbols);
  EXPECT_EQ(loomc_link_index_provider_count(index.get()), providers);
}

TEST_F(ConsumerEmitterTest, InvalidArgumentsClearOutputs) {
  loomc_module_t *module = reinterpret_cast<loomc_module_t *>(1);
  loomc_result_t *result = reinterpret_cast<loomc_result_t *>(1);
  EXPECT_EQ(loomc_status_consume_code(
                EmitScaleMinProbe(nullptr, workspace_.get(), &module, &result)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(loomc_status_consume_code(
                LinkConsumer(context_.get(), workspace_.get(), nullptr, nullptr,
                             loomc_make_cstring_view(""), &module, &result)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(result, nullptr);
}

} // namespace
} // namespace hrx::loomify_gguf
