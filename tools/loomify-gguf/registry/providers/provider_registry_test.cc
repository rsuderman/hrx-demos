// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/registry/providers/provider_registry.h"

#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "gtest/gtest.h"
#include "loomc/link.h"
#include "loomc/module_config.h"

namespace hrx::loomify_gguf {
namespace {

template <typename T, void (*Release)(T*)>
struct Deleter {
  void operator()(T* value) const { Release(value); }
};
template <typename T, void (*Release)(T*)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
using Context = Owner<loomc_context_t, loomc_context_release>;
using Workspace = Owner<loomc_workspace_t, loomc_workspace_release>;
using Source = Owner<loomc_source_t, loomc_source_release>;
using Module = Owner<loomc_module_t, loomc_module_release>;
using Result = Owner<loomc_result_t, loomc_result_release>;
using Linker = Owner<loomc_linker_t, loomc_linker_release>;

::testing::AssertionResult IsOk(loomc_status_t status) {
  const auto code = loomc_status_code(status);
  const auto message = loomc_status_message(status);
  std::string copied(message.size ? std::string(message.data, message.size) : "");
  loomc_status_free(status);
  if (code == LOOMC_STATUS_OK) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << loomc_status_code_string(code) << ": "
                                       << copied;
}

::testing::AssertionResult Succeeded(loomc_result_t* result) {
  if (result && loomc_result_succeeded(result)) {
    return ::testing::AssertionSuccess();
  }
  std::string messages;
  if (result) {
    for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto* diagnostic = loomc_result_diagnostic_at(result, i);
      messages.append(diagnostic->message.data, diagnostic->message.size);
      messages += '\n';
    }
  }
  return ::testing::AssertionFailure() << messages;
}

constexpr char kFunction[] = R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
func.def @hidden(%x: i32) -> (i32) {
  func.return %x : i32
}
)";

constexpr char kKernelAndMotif[] = R"(
kernel.def export("external_dispatch") @dispatch_body() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
template.decl @motif.identity(%x: i32) -> (i32)
template.def<@motif.identity> @identity_implementation(%x: i32) -> (i32) {
  template.return %x : i32
}
)";

class ProviderRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_context_t* context = nullptr;
    ASSERT_TRUE(IsOk(loomc_context_create(nullptr, loomc_allocator_system(),
                                         &context)));
    context_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    ASSERT_TRUE(IsOk(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                           &workspace)));
    workspace_.reset(workspace);
    ASSERT_TRUE(IsOk(ProviderRegistryBuilder::Create(context_.get(), &builder_)));
  }

  Module Parse(const char* text, loomc_context_t* context = nullptr) {
    loomc_source_options_t options = {};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier = loomc_make_cstring_view("registry_test.loom");
    options.contents = loomc_make_byte_span(text, std::strlen(text));
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    loomc_source_t* source = nullptr;
    EXPECT_TRUE(IsOk(loomc_source_create(&options, loomc_allocator_system(),
                                        &source)));
    Source source_owner(source);
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    EXPECT_TRUE(IsOk(loomc_module_deserialize_from_source(
        context ? context : context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &module, &result)));
    Result result_owner(result);
    EXPECT_TRUE(Succeeded(result));
    return Module(module);
  }

  std::unique_ptr<FrozenProviders> Freeze() {
    std::unique_ptr<FrozenProviders> frozen;
    loomc_result_t* result = nullptr;
    EXPECT_TRUE(IsOk(builder_->Freeze(&frozen, &result)));
    Result result_owner(result);
    EXPECT_TRUE(Succeeded(result));
    return frozen;
  }

  std::string Text(const loomc_module_t* module) {
    loomc_source_t* source = nullptr;
    loomc_module_serialize_options_t options = {};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    EXPECT_TRUE(IsOk(loomc_module_serialize_text_to_source(
        module, &options, loomc_allocator_system(), &source)));
    Source owner(source);
    if (!source) return {};
    const auto bytes = loomc_source_contents(source);
    return std::string(reinterpret_cast<const char*>(bytes.data),
                       bytes.data_length);
  }

  Context context_;
  Workspace workspace_;
  std::unique_ptr<ProviderRegistryBuilder> builder_;
};

TEST_F(ProviderRegistryTest, CopiesKindsExportsAndTemplateMetadata) {
  auto ordinary = Parse(kFunction);
  auto special = Parse(kKernelAndMotif);
  ASSERT_NE(ordinary, nullptr);
  ASSERT_NE(special, nullptr);
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("ordinary"),
                                       ordinary.get())));
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("special"),
                                       special.get())));
  ordinary.reset();
  special.reset();
  workspace_.reset();
  auto frozen = Freeze();
  ASSERT_NE(frozen, nullptr);
  builder_.reset();
  EXPECT_EQ(frozen->context(), context_.get());
  EXPECT_NE(frozen->native_index(), nullptr);
  loomc_link_index_provider_t provider = {};
  ASSERT_TRUE(loomc_link_index_provider_at(frozen->native_index(), 0, &provider));
  EXPECT_EQ(provider.role, LOOMC_LINK_PROVIDER_ROLE_LIBRARY);
  auto entry = frozen->FindCandidates(loomc_make_cstring_view("@entry"));
  ASSERT_EQ(entry.size(), 1u);
  EXPECT_EQ(entry[0].provider_name, "ordinary");
  EXPECT_EQ(entry[0].function_kind, LOOMC_MODULE_FUNCTION_KIND_FUNCTION);
  auto kernel = frozen->FindCandidates(loomc_make_cstring_view("dispatch_body"));
  ASSERT_EQ(kernel.size(), 1u);
  EXPECT_EQ(kernel[0].function_kind, LOOMC_MODULE_FUNCTION_KIND_KERNEL);
  ASSERT_TRUE(kernel[0].export_symbol.has_value());
  EXPECT_EQ(*kernel[0].export_symbol, "external_dispatch");
  EXPECT_TRUE(frozen->FindCandidates(loomc_make_cstring_view("hidden")).empty());
  EXPECT_TRUE(frozen->FindCandidates(loomc_make_cstring_view("missing")).empty());
  auto symbols = frozen->EnumerateSymbols();
  bool found_private = false, found_family = false, found_implementation = false;
  for (const auto& symbol : symbols) {
    if (symbol.name == "hidden") {
      found_private = true;
      EXPECT_EQ(symbol.identity, LOOMC_LINK_SYMBOL_IDENTITY_PRIVATE);
    }
    if (symbol.name == "motif.identity") {
      found_family = true;
      EXPECT_EQ(symbol.kind, LOOMC_LINK_SYMBOL_KIND_UNKNOWN);
      EXPECT_EQ(symbol.function_kind, LOOMC_MODULE_FUNCTION_KIND_UNKNOWN);
    }
    if (symbol.name == "identity_implementation") {
      found_implementation = true;
      EXPECT_EQ(symbol.kind, LOOMC_LINK_SYMBOL_KIND_FUNCTION_TEMPLATE);
    }
  }
  EXPECT_TRUE(found_private);
  EXPECT_TRUE(found_family);
  EXPECT_TRUE(found_implementation);
  frozen.reset();
  EXPECT_EQ(entry[0].name, "entry");
  EXPECT_EQ(*kernel[0].export_symbol, "external_dispatch");
}

TEST_F(ProviderRegistryTest, FailedAddsDoNotConsumeLabelsOrOrdinals) {
  auto module = Parse(kFunction);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(loomc_status_consume_code(
                builder_->AddModule(loomc_make_cstring_view(""), module.get())),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(
                builder_->AddModule(loomc_make_cstring_view("valid"), nullptr)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  loomc_context_t* foreign = nullptr;
  ASSERT_TRUE(IsOk(loomc_context_create(nullptr, loomc_allocator_system(),
                                       &foreign)));
  Context foreign_owner(foreign);
  auto foreign_module = Parse(kFunction, foreign);
  ASSERT_NE(foreign_module, nullptr);
  EXPECT_EQ(loomc_status_consume_code(builder_->AddModule(
                loomc_make_cstring_view("valid"), foreign_module.get())),
            LOOMC_STATUS_INVALID_ARGUMENT);
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("valid"),
                                       module.get())));
  EXPECT_EQ(loomc_status_consume_code(builder_->AddModule(
                loomc_make_cstring_view("valid"), module.get())),
            LOOMC_STATUS_ALREADY_EXISTS);
  auto frozen = Freeze();
  ASSERT_NE(frozen, nullptr);
  EXPECT_EQ(loomc_link_index_provider_count(frozen->native_index()), 1u);
  auto candidates = frozen->FindCandidates(loomc_make_cstring_view("entry"));
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates[0].provider_ordinal, 0u);
}

TEST_F(ProviderRegistryTest, RejectsRepeatedFreezeAndAddsAfterFreeze) {
  auto module = Parse(kFunction);
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("provider"),
                                       module.get())));
  auto frozen = Freeze();
  ASSERT_NE(frozen, nullptr);
  EXPECT_EQ(loomc_status_consume_code(builder_->AddModule(
                loomc_make_cstring_view("later"), module.get())),
            LOOMC_STATUS_FAILED_PRECONDITION);
  std::unique_ptr<FrozenProviders> second;
  loomc_result_t* result = nullptr;
  EXPECT_EQ(loomc_status_consume_code(builder_->Freeze(&second, &result)),
            LOOMC_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(second, nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST_F(ProviderRegistryTest, SnapshotSurvivesMutationAndLinksRepeatedly) {
  auto module = Parse(R"(
config.decl @answer : index
func.def public @entry() -> (index) {
  %answer = config.get @answer : index
  func.return %answer : index
}
)");
  ASSERT_NE(module, nullptr);
  const auto before = Text(module.get());
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("snapshot"),
                                       module.get())));
  loomc_config_binding_t binding = {};
  binding.key = loomc_make_cstring_view("answer");
  binding.value = loomc_make_cstring_view("7");
  loomc_module_config_options_t config = {};
  config.config.bindings = &binding;
  config.config.binding_count = 1;
  loomc_result_t* result = nullptr;
  ASSERT_TRUE(IsOk(loomc_module_apply_config(module.get(), workspace_.get(),
                                            &config, loomc_allocator_system(),
                                            &result)));
  Result config_result(result);
  ASSERT_TRUE(Succeeded(result));
  EXPECT_NE(Text(module.get()), before);
  module.reset();
  loomc_workspace_trim(workspace_.get());
  workspace_.reset();
  auto frozen = Freeze();
  ASSERT_NE(frozen, nullptr);
  builder_.reset();
  const auto inventory = frozen->EnumerateSymbols();
  context_.reset();
  loomc_linker_t* linker = nullptr;
  ASSERT_TRUE(IsOk(loomc_linker_create(frozen->context(), nullptr,
                                      loomc_allocator_system(), &linker)));
  Linker linker_owner(linker);
  for (int attempt = 0; attempt < 2; ++attempt) {
    loomc_workspace_t* scratch = nullptr;
    ASSERT_TRUE(IsOk(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                           &scratch)));
    Workspace scratch_owner(scratch);
    auto root = loomc_make_cstring_view("entry");
    loomc_link_options_t options = {};
    options.link_index = frozen->native_index();
    options.mode = LOOMC_LINK_MODE_LINK;
    options.root_symbols = &root;
    options.root_symbol_count = 1;
    loomc_module_t* linked = nullptr;
    result = nullptr;
    ASSERT_TRUE(IsOk(loomc_link_module(linker, scratch, &options, &linked, &result)));
    Result link_result(result);
    Module linked_owner(linked);
    ASSERT_TRUE(Succeeded(result));
    ASSERT_NE(linked, nullptr);
    EXPECT_NE(Text(linked).find("config.get @answer"), std::string::npos);
    EXPECT_EQ(frozen->EnumerateSymbols().size(), inventory.size());
  }
}

TEST_F(ProviderRegistryTest, DoesNotChooseFirstDuplicateConcreteDefinition) {
  auto first = Parse(kFunction);
  auto second = Parse(kFunction);
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("first"),
                                       first.get())));
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("second"),
                                       second.get())));
  auto frozen = Freeze();
  ASSERT_NE(frozen, nullptr);
  const auto candidates = frozen->FindCandidates(loomc_make_cstring_view("entry"));
  ASSERT_EQ(candidates.size(), 2u);
  EXPECT_EQ(candidates[0].provider_name, "first");
  EXPECT_EQ(candidates[1].provider_name, "second");
  loomc_linker_t* linker = nullptr;
  ASSERT_TRUE(IsOk(loomc_linker_create(context_.get(), nullptr,
                                      loomc_allocator_system(), &linker)));
  Linker owner(linker);
  auto root = loomc_make_cstring_view("entry");
  loomc_link_options_t options = {};
  options.link_index = frozen->native_index();
  options.mode = LOOMC_LINK_MODE_LINK;
  options.root_symbols = &root;
  options.root_symbol_count = 1;
  loomc_module_t* output = nullptr;
  loomc_result_t* result = nullptr;
  ASSERT_TRUE(IsOk(loomc_link_module(linker, workspace_.get(), &options, &output,
                                    &result)));
  Module output_owner(output);
  Result result_owner(result);
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  EXPECT_GT(loomc_result_diagnostic_count(result), 0u);
  EXPECT_EQ(output, nullptr);
}

TEST_F(ProviderRegistryTest, KeepsLegalTemplateAlternativesAndOwnersSeparate) {
  const char* implementations[] = {R"(
template.decl @choice(%x: i32) -> (i32)
template.def<@choice> priority(10) @fast(%x: i32) -> (i32) {
  template.return %x : i32
}
)", R"(
template.decl @choice(%x: i32) -> (i32)
template.def<@choice> priority(1) @fallback(%x: i32) -> (i32) {
  template.return %x : i32
}
)"};
  for (int i = 0; i < 2; ++i) {
    auto module = Parse(implementations[i]);
    ASSERT_NE(module, nullptr);
    ASSERT_TRUE(IsOk(builder_->AddModule(
        loomc_make_cstring_view(i ? "fallback" : "fast"), module.get())));
  }
  auto frozen = Freeze();
  ASSERT_NE(frozen, nullptr);
  int definitions = 0;
  for (const auto& symbol : frozen->EnumerateSymbols()) {
    if (symbol.kind == LOOMC_LINK_SYMBOL_KIND_FUNCTION_TEMPLATE) ++definitions;
  }
  EXPECT_EQ(definitions, 2);
  ASSERT_TRUE(IsOk(ProviderRegistryBuilder::Create(context_.get(), &builder_)));
  auto other = Parse(kFunction);
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("other"),
                                       other.get())));
  auto second = Freeze();
  ASSERT_NE(second, nullptr);
  EXPECT_NE(frozen->native_index(), second->native_index());
  EXPECT_TRUE(frozen->FindCandidates(loomc_make_cstring_view("entry")).empty());
  EXPECT_EQ(second->FindCandidates(loomc_make_cstring_view("entry")).size(), 1u);
  static_assert(!std::is_copy_constructible<FrozenProviders>::value, "move-only");
  FrozenProviders moved(std::move(*second));
  EXPECT_EQ(second->native_index(), nullptr);
  EXPECT_EQ(moved.FindCandidates(loomc_make_cstring_view("entry")).size(), 1u);
}

TEST_F(ProviderRegistryTest, InvalidOutputsDoNotConsumeBuilder) {
  std::unique_ptr<ProviderRegistryBuilder> invalid;
  EXPECT_EQ(loomc_status_consume_code(ProviderRegistryBuilder::Create(nullptr,
                                                                     &invalid)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(invalid, nullptr);
  EXPECT_EQ(loomc_status_consume_code(builder_->Freeze(nullptr, nullptr)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  auto module = Parse(kFunction);
  ASSERT_TRUE(IsOk(builder_->AddModule(loomc_make_cstring_view("valid"),
                                       module.get())));
  EXPECT_NE(Freeze(), nullptr);
}

}  // namespace
}  // namespace hrx::loomify_gguf
