// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "lib/serving/testing/ordinary_prepare_harness.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/token_policy.h"
#include "lib/loomify/testing/prepare_vm.h"
#include "loomc/link.h"
#include "loomc/source.h"
#include "loomc/target/vm.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace hrx::loomify_gguf::testing {
namespace {
using Module = std::unique_ptr<loomc_module_t, decltype(&loomc_module_release)>;

::testing::AssertionResult Ok(loomc_status_t status) {
  if (loomc_status_is_ok(status)) return ::testing::AssertionSuccess();
  auto message = loomc_status_message(status);
  std::string text(message.data, message.size);
  loomc_status_free(status);
  return ::testing::AssertionFailure() << text;
}
std::string Diagnostics(loomc_result_t* result) {
  std::string text;
  if (result) for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
    const auto* diagnostic = loomc_result_diagnostic_at(result, i);
    text.append(diagnostic->message.data, diagnostic->message.size);
    text += '\n';
  }
  return text;
}
template <typename T>
std::vector<T> Words(const std::vector<uint8_t>& bytes) {
  EXPECT_EQ(bytes.size() % sizeof(T), 0u);
  std::vector<T> words(bytes.size() / sizeof(T));
  if (!bytes.empty()) std::memcpy(words.data(), bytes.data(), words.size() * sizeof(T));
  return words;
}
class PreparationContext {
 public:
  explicit PreparationContext(const OrdinaryPreparationCase& test_case)
      : case_(test_case) {}
  void Init() {
    ASSERT_TRUE(Ok(loomc_target_environment_create_vm(loomc_allocator_system(), &environment_)));
    loomc_context_target_options_t target{};
    target.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target.target_environment = environment_;
    loomc_context_options_t options{};
    options.next = &target;
    ASSERT_TRUE(Ok(loomc_context_create(&options, loomc_allocator_system(), &context_)));
    ASSERT_TRUE(Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace_)));
    const char* srcdir = std::getenv("TEST_SRCDIR");
    const char* workspace = std::getenv("TEST_WORKSPACE");
    ASSERT_NE(srcdir, nullptr);
    ASSERT_NE(workspace, nullptr);
    root_ = std::filesystem::path(srcdir) / workspace;
  }
  ~PreparationContext() {
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
    loomc_target_environment_release(environment_);
  }
  Module Parse(const std::string& text) {
    loomc_source_options_t options{};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    const auto identifier = "ordinary-configuration-" + std::to_string(next_source_++) + ".loom";
    options.identifier = loomc_make_cstring_view(identifier.c_str());
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.contents = loomc_make_byte_span(text.data(), text.size());
    loomc_source_t* source = nullptr;
    if (!Ok(loomc_source_create(&options, loomc_allocator_system(), &source))) return {nullptr, loomc_module_release};
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(context_, workspace_, source, nullptr,
        loomc_allocator_system(), &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    if (result) loomc_result_release(result);
    loomc_source_release(source);
    return {module, loomc_module_release};
  }
  Module Load(const std::filesystem::path& path) {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_path(context_, workspace_,
        loomc_make_cstring_view(path.c_str()), nullptr, loomc_allocator_system(), &module, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    if (result) loomc_result_release(result);
    return {module, loomc_module_release};
  }
  Module Build(int64_t tokens, int64_t context, bool bytecode,
               bool missing_history = false) {
    ConfigurationFacts facts;
    std::string error;
    auto model = case_.model();
    EXPECT_TRUE(DeriveConfiguration(model, "fixture.gguf",
        {uint64_t(tokens), uint64_t(context)}, &facts, &error)) << error;
    if (missing_history) {
      const auto begin = facts.source.find("config.def @loomify.serving.history_capacity");
      EXPECT_NE(begin, std::string::npos);
      if (begin != std::string::npos) facts.source.erase(begin, facts.source.find('\n', begin) - begin + 1);
    }
    auto config = Parse(facts.source + EmitTerminalMarkerSource(case_.terminal_marker));
    auto prepare = Load(root_ / (bytecode ? case_.prepare_bytecode : case_.prepare_source));
    if (!config || !prepare) return {nullptr, loomc_module_release};
    auto common = Load(root_ / "lib/serving/ordinary_prepare.loom");
    std::vector<loomc_link_module_provider_t> providers = {
      {config.get(), loomc_make_cstring_view("configuration"), LOOMC_LINK_PROVIDER_ROLE_INPUT},
      {prepare.get(), loomc_make_cstring_view("preparation"), LOOMC_LINK_PROVIDER_ROLE_INPUT},
    };
    if (!bytecode) providers.push_back({common.get(), loomc_make_cstring_view("ordinary preparation"), LOOMC_LINK_PROVIDER_ROLE_INPUT});
    const auto root = loomc_make_cstring_view("prepare");
    loomc_link_options_t link_options{};
    link_options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    link_options.structure_size = sizeof(link_options);
    link_options.mode = LOOMC_LINK_MODE_LINK;
    link_options.module_providers = providers.data();
    link_options.module_provider_count = providers.size();
    link_options.root_symbols = &root;
    link_options.root_symbol_count = 1;
    link_options.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    loomc_linker_t* linker = nullptr;
    EXPECT_TRUE(Ok(loomc_linker_create(context_, nullptr, loomc_allocator_system(), &linker)));
    if (!linker) return {nullptr, loomc_module_release};
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    const auto status = loomc_link_module(linker, workspace_, &link_options, &module, &result);
    const bool succeeded = loomc_status_is_ok(status) && result && loomc_result_succeeded(result);
    if (missing_history) {
      EXPECT_FALSE(succeeded);
      EXPECT_EQ(module, nullptr);
      EXPECT_FALSE(Diagnostics(result).empty());
    } else {
      EXPECT_TRUE(succeeded) << Diagnostics(result);
    }
    loomc_status_free(status);
    if (result) loomc_result_release(result);
    loomc_linker_release(linker);
    return {module, loomc_module_release};
  }
  Module RoundTrip(const loomc_module_t* module, loomc_source_format_t format) {
    loomc_module_serialize_options_t options{};
    options.format = format;
    loomc_source_t* source = nullptr;
    EXPECT_TRUE(Ok(loomc_module_serialize_to_source(module, &options, loomc_allocator_system(), &source)));
    if (!source) return {nullptr, loomc_module_release};
    loomc_module_t* reloaded = nullptr;
    loomc_result_t* result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(context_, workspace_, source, nullptr,
        loomc_allocator_system(), &reloaded, &result)));
    EXPECT_TRUE(result && loomc_result_succeeded(result)) << Diagnostics(result);
    if (format == LOOMC_SOURCE_FORMAT_TEXT) {
      const auto bytes = loomc_source_contents(source);
      const std::string text(reinterpret_cast<const char*>(bytes.data), bytes.data_length);
      for (const char* scope : {"input", "json", "prepare"})
        EXPECT_NE(text.find(std::string("import(\"") + scope + "\")"), std::string::npos);
    }
    if (result) loomc_result_release(result);
    loomc_source_release(source);
    return {reloaded, loomc_module_release};
  }
  void CheckRecords(const PreparationRequest& request, const PreparationObservation& observed) {
    const auto hidden = case_.hidden, layers = case_.layers, kv = case_.kv;
    const auto epoch = case_.epoch_root, marker = case_.terminal_marker;
    // Independently stated native record contract. Never read production config
    // values or call a production geometry builder to compute expected records.
    const int64_t t = request.prefill, c = request.context;
    const int64_t p = ((c + 63) / 64) * 64;
    const int64_t h = std::max<int64_t>(t, 129), b = 768 + 4*h;
    const int64_t residual = 4*t*hidden, arena = b + 4*layers*p*kv;
    const int64_t sizes[] = {residual, arena, 2064, 2048+4*((c+63)/64), 4*h, 64, 0, 0, 0, 0, 0};
    std::vector<int64_t> allocations;
    for (size_t i = 0; i < 11; ++i) {
      allocations.push_back(sizes[i]); allocations.push_back(256);
      allocations.push_back(i == 0 ? residual : i == 1 ? b : 0);
    }
    EXPECT_EQ(Words<int64_t>(observed.buffers[2]), allocations);
    EXPECT_EQ(Words<int64_t>(observed.buffers[3]), (std::vector<int64_t>{0,12,512,0,512,4*h,512+4*h,32}));
    std::vector<int64_t> origins(64, 0);
    origins[0] = origins[1] = 256; origins[2] = b;
    EXPECT_EQ(Words<int64_t>(observed.buffers[4]), origins);
    EXPECT_TRUE(observed.buffers[5].empty());
    EXPECT_EQ(Words<int64_t>(observed.buffers[6]), (std::vector<int64_t>{64,2048,0,64,256,b,4,1,1,b,2*layers,2*p*kv,128*kv}));
    EXPECT_EQ(std::string(observed.buffers[1].begin(), observed.buffers[1].end()), marker);
    int64_t maximum = 0;
    for (const auto& shape : request.shapes) maximum = std::max(maximum, shape[0]);
    EXPECT_EQ(observed.effective_prefill, std::min(t, maximum));
    EXPECT_EQ(observed.shared, int32_t(2 + request.shapes.size()));
    ASSERT_EQ(observed.stages.size(), 2 + request.shapes.size());
    EXPECT_EQ(observed.stages[0].root, epoch);
    EXPECT_EQ(observed.stages[0].tag, t);
    EXPECT_EQ(observed.stages[1].root, epoch);
    EXPECT_EQ(observed.stages[1].tag, 1);
    for (size_t i = 0; i < request.shapes.size(); ++i) {
      EXPECT_EQ(observed.stages[2+i].root, epoch);
      EXPECT_EQ(observed.stages[2+i].tag, request.shapes[i][0]);
    }
    EXPECT_NE(observed.stages[0].id, observed.stages[1].id);
    EXPECT_EQ(Words<int32_t>(observed.buffers[0]), (std::vector<int32_t>{observed.stages[0].id, observed.stages[1].id}));
    ASSERT_EQ(observed.parameters.size(), observed.stages.size());
    for (size_t i = 0; i < observed.parameters.size(); ++i) {
      const auto& parameter = observed.parameters[i];
      EXPECT_EQ(parameter.stage, observed.stages[i].id);
      EXPECT_EQ(parameter.binding, 0); EXPECT_EQ(parameter.source, 0);
      EXPECT_TRUE(parameter.directory.empty());
      EXPECT_EQ(parameter.path, request.weights); EXPECT_EQ(parameter.policy, "weights.loom");
    }
    ASSERT_EQ(observed.requirements.size(), 2 + request.shapes.size());
    for (auto condition : observed.requirements) EXPECT_EQ(condition, 1);
  }
  const OrdinaryPreparationCase& case_;
  loomc_target_environment_t* environment_ = nullptr;
  loomc_context_t* context_ = nullptr;
  loomc_workspace_t* workspace_ = nullptr;
  std::filesystem::path root_;
  uint64_t next_source_ = 0;
};

}  // namespace

void CheckPreparationRecords(const OrdinaryPreparationCase& test_case,
                             const std::vector<Capacity>& capacities) {
  SCOPED_TRACE(test_case.name);
  PreparationContext context(test_case);
  ASSERT_NO_FATAL_FAILURE(context.Init());
  for (const auto& capacity : capacities) for (bool bytecode : {false,true}) {
    SCOPED_TRACE(::testing::Message() << "T=" << capacity[0] << " C=" << capacity[1] << " bytecode=" << bytecode);
    auto linked = context.Build(capacity[0], capacity[1], bytecode);
    ASSERT_TRUE(linked);
    auto reloaded = context.RoundTrip(linked.get(), bytecode ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT);
    ASSERT_TRUE(reloaded);
    PreparationRequest request;
    request.prefill = capacity[0]; request.context = capacity[1];
    request.pool = (capacity[1]+63)/64*64;
    request.shapes = {{1,1},{capacity[0],1}};
    request.weights = "external/immutable fixture.gguf";
    PreparationObservation observed;
    std::string error;
    ASSERT_TRUE(testing::RunPreparation(context.environment_, context.context_, context.workspace_, reloaded.get(), request, &observed, &error)) << error;
    context.CheckRecords(request, observed);
  }
}

void CheckEffectivePrefill(const OrdinaryPreparationCase& test_case) {
  SCOPED_TRACE(test_case.name);
  PreparationContext context(test_case);
  ASSERT_NO_FATAL_FAILURE(context.Init());
  auto linked = context.Build(128,512,true);
  ASSERT_TRUE(linked);
  PreparationRequest request;
  request.shapes = {{1,1},{4,1}};
  PreparationObservation observed;
  std::string error;
  ASSERT_TRUE(testing::RunPreparation(context.environment_, context.context_, context.workspace_, linked.get(), request, &observed, &error)) << error;
  context.CheckRecords(request, observed);
}

void CheckSingleRowStageOrdinals(const OrdinaryPreparationCase& test_case) {
  SCOPED_TRACE(test_case.name);
  PreparationContext context(test_case);
  ASSERT_NO_FATAL_FAILURE(context.Init());
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode);
    auto linked = context.Build(8, 512, bytecode);
    ASSERT_TRUE(linked);
    PreparationRequest request;
    request.prefill = 8;
    request.shapes = {{1,1},{2,1},{2,2},{4,1},{4,2},{4,4},
                      {8,1},{8,2},{8,4},{8,8}};
    PreparationObservation observed;
    std::string error;
    ASSERT_TRUE(testing::RunPreparation(context.environment_, context.context_, context.workspace_,
        linked.get(), request, &observed, &error)) << error;
    // Native packing prefers the span-1 sibling for the admitted one row.
    // Every advertised slot still registers the same token-specialized root.
    context.CheckRecords(request, observed);
  }
}

void CheckRejectedPreparationRequests(const OrdinaryPreparationCase& test_case) {
  SCOPED_TRACE(test_case.name);
  PreparationContext context(test_case);
  ASSERT_NO_FATAL_FAILURE(context.Init());
  for (int variant = 0; variant < 12; ++variant) {
    SCOPED_TRACE(variant);
    auto linked = context.Build(128,512,variant % 2 != 0);
    ASSERT_TRUE(linked);
    PreparationRequest request;
    switch (variant) {
      case 0: request.rows = 2; break;
      case 1: request.pins = 1; break;
      case 2: request.draft_depth = 1; break;
      case 3: request.options_type = 1; break;
      case 4: request.options_fields = 1; break;
      case 5: request.prefill = 127; break;
      case 6: request.context = 511; break;
      case 7: request.pool = 576; break;
      case 8: request.shapes = {{1,2}}; break;
      case 9: request.shapes = {{0,1}}; break;
      case 10: request.shapes = {{129,1}}; break;
      case 11: request.shapes.clear(); break;
    }
    PreparationObservation observed;
    std::string error;
    EXPECT_FALSE(testing::RunPreparation(context.environment_, context.context_, context.workspace_, linked.get(), request, &observed, &error));
    EXPECT_FALSE(error.empty());
    ASSERT_FALSE(observed.requirements.empty());
    EXPECT_EQ(observed.requirements.back(), 0);
  }
}

void CheckMissingPreparationConfig(const OrdinaryPreparationCase& test_case) {
  SCOPED_TRACE(test_case.name);
  PreparationContext context(test_case);
  ASSERT_NO_FATAL_FAILURE(context.Init());
  for (bool bytecode : {false,true}) EXPECT_FALSE(context.Build(128,512,bytecode,true));
}

}  // namespace hrx::loomify_gguf::testing
