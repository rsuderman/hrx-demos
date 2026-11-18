// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT
#include "loomc/target/vm.h"
#include "lib/loomify/configuration.h"
#include "lib/loomify/token_policy.h"
#include "lib/loomify/testing/configuration_fixture.h"
#include "lib/loomify/testing/alternating_qwen_fixture.h"
#include "lib/loomify/testing/prepare_vm.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>

namespace hrx::loomify_gguf::testing {
namespace {
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, decltype(Release)>;
using Module = Owner<loomc_module_t, loomc_module_release>;
bool Ok(loomc_status_t status) {
  bool ok = loomc_status_is_ok(status);
  if (!ok) {
    auto m = loomc_status_message(status);
    ADD_FAILURE() << std::string(m.data, m.size);
  }
  loomc_status_free(status);
  return ok;
}
bool Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return true;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto *d = loomc_result_diagnostic_at(result, i);
      ADD_FAILURE() << std::string(d->message.data, d->message.size);
    }
  return false;
}
std::string Path(const char *leaf) {
  std::filesystem::path root =
      std::getenv("TEST_SRCDIR") ? std::getenv("TEST_SRCDIR") : ".";
  root /=
      std::getenv("TEST_WORKSPACE") ? std::getenv("TEST_WORKSPACE") : "_main";
  return (root / "lib/architectures/qwen35" / leaf).string();
}
std::string Text(const char *leaf) {
  std::ifstream f(Path(leaf));
  return std::string(std::istreambuf_iterator<char>(f), {});
}
std::vector<int64_t> Words64(const std::vector<uint8_t> &bytes) {
  EXPECT_EQ(bytes.size() % 8, 0u);
  std::vector<int64_t> out;
  for (size_t i = 0; i + 8 <= bytes.size(); i += 8) {
    uint64_t v = 0;
    for (unsigned j = 0; j < 8; ++j)
      v |= uint64_t(bytes[i + j]) << (j * 8);
    out.push_back(static_cast<int64_t>(v));
  }
  return out;
}
std::vector<int32_t> Words32(const std::vector<uint8_t> &bytes) {
  EXPECT_EQ(bytes.size() % 4, 0u);
  std::vector<int32_t> out;
  for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
    uint32_t v = 0;
    for (unsigned j = 0; j < 4; ++j)
      v |= uint32_t(bytes[i + j]) << (j * 8);
    out.push_back(static_cast<int32_t>(v));
  }
  return out;
}
class QwenPrepareTest : public ::testing::Test {
protected:
  void SetUp() override {
    loomc_target_environment_t *e = nullptr;
    ASSERT_TRUE(
        Ok(loomc_target_environment_create_vm(loomc_allocator_system(), &e)));
    environment.reset(e);
    loomc_context_target_options_t t{};
    t.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    t.target_environment = e;
    loomc_context_options_t o{};
    o.next = &t;
    loomc_context_t *c = nullptr;
    ASSERT_TRUE(Ok(loomc_context_create(&o, loomc_allocator_system(), &c)));
    context.reset(c);
    loomc_workspace_t *w = nullptr;
    ASSERT_TRUE(
        Ok(loomc_workspace_create(nullptr, loomc_allocator_system(), &w)));
    workspace.reset(w);
  }
  Module Parse(const std::string &text) {
    loomc_source_options_t o{};
    o.format = LOOMC_SOURCE_FORMAT_TEXT;
    const auto identity =
        "qwen-preparation-test-" + std::to_string(source_id++);
    o.identifier = loomc_make_cstring_view(identity.c_str());
    o.contents = {reinterpret_cast<const uint8_t *>(text.data()), text.size()};
    o.storage = LOOMC_SOURCE_STORAGE_BORROWED;
    loomc_source_t *s = nullptr;
    if (!Ok(loomc_source_create(&o, loomc_allocator_system(), &s)))
      return {nullptr, loomc_module_release};
    Owner<loomc_source_t, loomc_source_release> source(s, loomc_source_release);
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    auto status = loomc_module_deserialize_from_source(
        context.get(), workspace.get(), s, nullptr, loomc_allocator_system(),
        &m, &r);
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    Module module(m, loomc_module_release);
    if (!Ok(status) || !Succeeded(r))
      return {nullptr, loomc_module_release};
    return module;
  }
  Module Product(bool bytecode) {
    if (!bytecode)
      return Parse(Text("prepare.loom"));
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    auto path = Path("prepare_library.loombc");
    auto status = loomc_module_deserialize_from_path(
        context.get(), workspace.get(), loomc_make_cstring_view(path.c_str()),
        nullptr, loomc_allocator_system(), &m, &r);
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    Module module(m, loomc_module_release);
    if (!Ok(status) || !Succeeded(r))
      return {nullptr, loomc_module_release};
    return module;
  }
  Module Link(uint64_t tokens, uint64_t capacity, bool bytecode,
              bool config = true, bool large = false,
              const model::ModelDescriptor *descriptor = nullptr) {
    ConfigurationFacts facts;
    std::string error;
    if (!DeriveConfiguration(descriptor ? *descriptor : QwenModelFacts(large),
                             "fixture.gguf", {tokens, capacity}, &facts,
                             &error)) {
      ADD_FAILURE() << error;
      return {nullptr, loomc_module_release};
    }
    auto cfg = Parse((config ? facts.source : "") +
                     EmitTerminalMarkerSource("<|im_end|>"));
    auto prepare = Product(bytecode);
    if (!cfg || !prepare)
      return {nullptr, loomc_module_release};
    loomc_link_module_provider_t providers[] = {
        {cfg.get(), loomc_make_cstring_view("config"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT},
        {prepare.get(), loomc_make_cstring_view("prepare"),
         LOOMC_LINK_PROVIDER_ROLE_INPUT}};
    loomc_linker_t *l = nullptr;
    if (!Ok(loomc_linker_create(context.get(), nullptr,
                                loomc_allocator_system(), &l)))
      return {nullptr, loomc_module_release};
    Owner<loomc_linker_t, loomc_linker_release> linker(l, loomc_linker_release);
    const auto root = loomc_make_cstring_view("prepare");
    loomc_link_options_t o{};
    o.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    o.structure_size = sizeof(o);
    o.mode = LOOMC_LINK_MODE_LINK;
    o.module_providers = providers;
    o.module_provider_count = 2;
    o.root_symbols = &root;
    o.root_symbol_count = 1;
    o.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    loomc_module_t *m = nullptr;
    loomc_result_t *r = nullptr;
    auto status = loomc_link_module(l, workspace.get(), &o, &m, &r);
    Owner<loomc_result_t, loomc_result_release> result(r, loomc_result_release);
    Module module(m, loomc_module_release);
    if (!Ok(status))
      return {nullptr, loomc_module_release};
    if (!config) {
      EXPECT_FALSE(r && loomc_result_succeeded(r));
      return {nullptr, loomc_module_release};
    }
    if (!Succeeded(r))
      return {nullptr, loomc_module_release};
    // Prove linked native imports survive both source and reloaded bytecode.
    loomc_module_serialize_options_t serialization{};
    serialization.format =
        bytecode ? LOOMC_SOURCE_FORMAT_BYTECODE : LOOMC_SOURCE_FORMAT_TEXT;
    loomc_source_t *s = nullptr;
    if (!Ok(loomc_module_serialize_to_source(m, &serialization,
                                             loomc_allocator_system(), &s)))
      return {nullptr, loomc_module_release};
    Owner<loomc_source_t, loomc_source_release> source(s, loomc_source_release);
    loomc_module_t *restored = nullptr;
    loomc_result_t *rr = nullptr;
    status = loomc_module_deserialize_from_source(
        context.get(), workspace.get(), s, nullptr, loomc_allocator_system(),
        &restored, &rr);
    Owner<loomc_result_t, loomc_result_release> reload_result(
        rr, loomc_result_release);
    Module reloaded(restored, loomc_module_release);
    if (!Ok(status) || !Succeeded(rr))
      return {nullptr, loomc_module_release};
    return reloaded;
  }
  void Records(const PreparationRequest &request,
               const PreparationObservation &out, bool large = false,
               bool alternating = false) {
    const int64_t P = ((request.context + 63) / 64) * 64;
    // Independent current-profile contract, not generated/emitter-derived
    // geometry.
    const int64_t S = alternating ? 4 * 4 * (18432 + 262144)
                      : large     ? 48 * 4 * (30720 + 786432)
                                  : 18 * 4 * (18432 + 262144);
    const int64_t B = 256 + S + 2048 + 256;
    const int64_t residual = large ? 10485760 : 2097152;
    const int64_t planes = alternating ? 8 : large ? 32 : 12;
    const int64_t plane_token_bytes = large ? 2048 : 1024;
    std::vector<int64_t> allocations;
    const std::array<int64_t, 11> sizes = {
        residual, B + planes * P * plane_token_bytes,
        4192,     2048 + 4 * ((request.context + 63) / 64),
        4096,     64,
        0,        0,
        0,        0,
        0};
    for (size_t i = 0; i < 11; ++i) {
      allocations.push_back(sizes[i]);
      allocations.push_back(256);
      allocations.push_back(i == 0 ? residual : i == 1 ? B : 0);
    }
    EXPECT_EQ(Words64(out.buffers[2]), allocations);
    EXPECT_EQ(Words64(out.buffers[3]),
              (std::vector<int64_t>{0, 12, 256 + S, 0, 256 + S, 2048,
                                    256 + S + 2048, 32}));
    std::vector<int64_t> origins(64, 0);
    origins[0] = origins[1] = 256;
    origins[2] = B;
    EXPECT_EQ(Words64(out.buffers[4]), origins);
    EXPECT_EQ(Words64(out.buffers[5]), std::vector<int64_t>(64, 0));
    EXPECT_EQ(
        Words64(out.buffers[6]),
        (std::vector<int64_t>{64, 2048, 0, 384, 256, B, S, 1, 1, B, planes,
                              P * plane_token_bytes, 64 * plane_token_bytes}));
    const int64_t max_tokens =
        std::max_element(request.shapes.begin(), request.shapes.end(),
                         [](auto a, auto b) { return a[0] < b[0]; })
            ->at(0);
    EXPECT_EQ(out.effective_prefill, max_tokens);
    EXPECT_EQ(out.shared, int32_t(request.shapes.size() + 2));
    EXPECT_EQ(
        Words32(out.buffers[0]),
        (std::vector<int32_t>{101, 102, int32_t(request.shapes.size()), 0}));
    EXPECT_EQ(std::string(out.buffers[1].begin(), out.buffers[1].end()),
              "<|im_end|>");
    ASSERT_EQ(out.stages.size(), request.shapes.size() + 2);
    ASSERT_EQ(out.parameters.size(), out.stages.size());
    std::vector<int64_t> tags = {request.prefill, 1};
    for (auto shape : request.shapes)
      tags.push_back(shape[0]);
    for (size_t i = 0; i < tags.size(); ++i) {
      const bool eligible = !large && !alternating && request.prefill == 128 &&
                            request.context == 512;
      const bool decode = eligible && i != 0 && tags[i] == 1;
      EXPECT_EQ(out.stages[i].root, decode ? "generated.qwen_decode_epoch"
                                           : "generated.qwen_epoch");
      EXPECT_EQ(out.stages[i].tag, tags[i]);
      EXPECT_EQ(out.stages[i].id, 101 + i);
      const auto &p = out.parameters[i];
      EXPECT_EQ(p.stage, out.stages[i].id);
      EXPECT_EQ(p.binding, 0);
      EXPECT_EQ(p.source, 0);
      EXPECT_TRUE(p.directory.empty());
      EXPECT_EQ(p.path, request.weights);
      EXPECT_EQ(p.policy, "weights.loom");
    }
    EXPECT_EQ(out.requirements,
              std::vector<int32_t>(request.shapes.size() + 2, 1));
  }
  unsigned source_id = 0;
  Owner<loomc_target_environment_t, loomc_target_environment_release>
      environment{nullptr, loomc_target_environment_release};
  Owner<loomc_context_t, loomc_context_release> context{nullptr,
                                                        loomc_context_release};
  Owner<loomc_workspace_t, loomc_workspace_release> workspace{
      nullptr, loomc_workspace_release};
};
TEST_F(QwenPrepareTest, CompleteRecordsAndRegistrationsFromSourceAndBytecode) {
  for (bool bc : {false, true})
    for (uint64_t T : {1, 128, 129, 512}) {
      for (uint64_t C : {std::max<uint64_t>(T, 64), std::max<uint64_t>(T, 65),
                         uint64_t(513), uint64_t(262144)}) {
        SCOPED_TRACE(::testing::Message()
                     << "T=" << T << " C=" << C << " BC=" << bc);
        auto module = Link(T, C, bc);
        ASSERT_TRUE(module);
        PreparationRequest request;
        request.prefill = T;
        request.context = C;
        request.pool = (C + 63) / 64 * 64;
        request.shapes = {{1, 1}, {int64_t(T), 1}};
        request.weights = "external immutable fixture.gguf";
        PreparationObservation observed;
        std::string error;
        ASSERT_TRUE(RunPreparation(environment.get(), context.get(),
                                   workspace.get(), module.get(), request,
                                   &observed, &error))
            << error;
        Records(request, observed);
      }
    }
}
TEST_F(QwenPrepareTest, Qwen27AllRecordsAcrossCapacityAndPoolBoundaries) {
  for (bool bc : {false, true})
    for (uint64_t T : {1, 128, 257, 512})
      for (uint64_t C : {std::max<uint64_t>(T, 65), uint64_t(513)}) {
        SCOPED_TRACE(::testing::Message()
                     << "27B T=" << T << " C=" << C << " BC=" << bc);
        auto module = Link(T, C, bc, true, true);
        ASSERT_TRUE(module);
        PreparationRequest request;
        request.prefill = T;
        request.context = C;
        request.pool = (C + 63) / 64 * 64;
        request.shapes = {{1, 1}, {int64_t(T), 1}};
        PreparationObservation out;
        std::string error;
        ASSERT_TRUE(RunPreparation(environment.get(), context.get(),
                                   workspace.get(), module.get(), request, &out,
                                   &error))
            << error;
        Records(request, out, true);
      }
}
TEST_F(QwenPrepareTest, EffectivePrefillUsesRequestedShapeMaximum) {
  for (bool bytecode : {false, true}) {
    auto module = Link(128, 512, bytecode);
    ASSERT_TRUE(module);
    PreparationRequest request;
    request.shapes = {{17, 1}, {1, 1}};
    PreparationObservation observed;
    std::string error;
    ASSERT_TRUE(RunPreparation(environment.get(), context.get(),
                               workspace.get(), module.get(), request,
                               &observed, &error))
        << error;
    EXPECT_EQ(observed.effective_prefill, 17);
    Records(request, observed);
  }
}
TEST_F(QwenPrepareTest, NativeSingleRowShapeInventoryPreservesStageOrdinals) {
  for (bool bytecode : {false, true}) {
    SCOPED_TRACE(bytecode);
    auto module = Link(8, 512, bytecode);
    ASSERT_TRUE(module);
    PreparationRequest request;
    request.prefill = 8;
    request.shapes = {{1, 1}, {2, 1}, {2, 2}, {4, 1}, {4, 2},
                      {4, 4}, {8, 1}, {8, 2}, {8, 4}, {8, 8}};
    PreparationObservation observed;
    std::string error;
    ASSERT_TRUE(RunPreparation(environment.get(), context.get(),
                               workspace.get(), module.get(), request,
                               &observed, &error))
        << error;
    Records(request, observed);
  }
}
TEST_F(QwenPrepareTest, QualifiedDecodeAndOneTokenShapesKeepExactStageOrder) {
  for (bool bytecode : {false, true}) {
    auto module = Link(128, 512, bytecode);
    ASSERT_TRUE(module);
    PreparationRequest request;
    request.shapes = {{1, 1}, {2, 1},  {2, 2},   {4, 1},  {4, 2},
                      {4, 4}, {17, 1}, {128, 1}, {128, 2}};
    PreparationObservation observed;
    std::string error;
    ASSERT_TRUE(RunPreparation(environment.get(), context.get(),
                               workspace.get(), module.get(), request,
                               &observed, &error))
        << error;
    Records(request, observed);
    EXPECT_EQ(observed.stages[0].root, "generated.qwen_epoch");
    EXPECT_EQ(observed.stages[1].root, "generated.qwen_decode_epoch");
    EXPECT_EQ(observed.stages[2].root, "generated.qwen_decode_epoch");
    for (size_t i = 3; i < observed.stages.size(); ++i)
      EXPECT_EQ(observed.stages[i].root, "generated.qwen_epoch");
  }
}
TEST_F(QwenPrepareTest, IndependentEightLayerGgufControlsNativeStorageRecords) {
  model::ModelDescriptor descriptor;
  std::string error;
  ASSERT_TRUE(LoadAlternatingQwenFixture(&descriptor, nullptr, &error))
      << error;
  for (bool bytecode : {false, true}) {
    auto module = Link(3, 65, bytecode, true, false, &descriptor);
    ASSERT_TRUE(module);
    PreparationRequest request;
    request.prefill = 3;
    request.context = 65;
    request.pool = 128;
    request.shapes = {{1, 1}, {3, 1}};
    request.weights = "independent sparse fixture.gguf";
    PreparationObservation observed;
    ASSERT_TRUE(RunPreparation(environment.get(), context.get(),
                               workspace.get(), module.get(), request,
                               &observed, &error))
        << error;
    Records(request, observed, false, true);
  }
}
TEST_F(QwenPrepareTest, MissingLiveConfigurationFailsNativeLink) {
  EXPECT_FALSE(Link(128, 512, false, false));
  EXPECT_FALSE(Link(128, 512, true, false));
}
TEST_F(QwenPrepareTest,
       NativeRequirementCallbacksObserveAndRejectUnsupportedInputs) {
  for (bool bc : {false, true})
    for (int bad = 0; bad < 11; ++bad) {
      SCOPED_TRACE(::testing::Message() << "bad=" << bad << " BC=" << bc);
      auto module = Link(128, 512, bc);
      ASSERT_TRUE(module);
      PreparationRequest request;
      switch (bad) {
      case 0:
        request.rows = 2;
        break;
      case 1:
        request.pins = 1;
        break;
      case 2:
        request.draft_depth = 1;
        break;
      case 3:
        request.options_type = 1;
        break;
      case 4:
        request.options_fields = 1;
        break;
      case 5:
        request.prefill = 127;
        break;
      case 6:
        request.context = 511;
        break;
      case 7:
        request.pool = 448;
        break;
      case 8:
        request.shapes = {{129, 1}};
        break;
      case 9:
        request.shapes = {{1, 2}};
        break;
      case 10:
        request.shapes.clear();
        break;
      }
      PreparationObservation observed;
      std::string error;
      EXPECT_FALSE(RunPreparation(environment.get(), context.get(),
                                  workspace.get(), module.get(), request,
                                  &observed, &error));
      EXPECT_NE(error.find("prepare requirement rejected"), std::string::npos)
          << error;
      ASSERT_FALSE(observed.requirements.empty());
      EXPECT_EQ(observed.requirements.back(), 0);
      if (bad != 8 && bad != 9) {
        EXPECT_TRUE(observed.stages.empty());
        EXPECT_TRUE(observed.parameters.empty());
      }
    }
}
} // namespace
} // namespace hrx::loomify_gguf::testing
