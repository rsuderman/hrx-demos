// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace hrx::loomify_gguf::benchmark {
namespace {
loomc_cmd_buffer_ref_t Ref(uint32_t root, uint64_t offset = 0,
                           uint64_t bytes = UINT64_MAX) {
  return {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, root, offset, bytes};
}
uint32_t Word(const PreparedRoot &root, uint64_t absolute) {
  const uint64_t offset = absolute - root.original_byte_offset;
  uint32_t value = 0;
  for (unsigned lane = 0; lane < 4; ++lane)
    value |= static_cast<uint32_t>(root.bytes[offset + lane]) << (lane * 8);
  return value;
}
Invocation Make(bool publish, uint64_t head_size = 128, bool gated = false) {
  Invocation invocation;
  invocation.kernel_symbol =
      publish ? "llm.decoder.publish_kv_h" + std::to_string(head_size)
      : gated ? "llm.attention.gated_window_rows_h" + std::to_string(head_size)
              : "llm.decoder.attention_rows_h128";
  invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
  invocation.command.payload.dispatch.workgroup_count[0] = publish ? 2 : 4;
  invocation.command.payload.dispatch.workgroup_count[1] = 8;
  invocation.command.payload.dispatch.workgroup_count[2] = 1;
  const size_t n = publish ? 7 : gated ? 10 : 9;
  const size_t offsets = publish ? 4 : gated ? 7 : 6;
  invocation.arguments.resize(n + (gated ? 6 : 5));
  invocation.command.argument_count = invocation.arguments.size();
  for (size_t i = 0; i < n; ++i)
    invocation.arguments[i].kind = i >= offsets ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                                   : !publish && i == 5
                                       ? LOOMC_CMD_ARGUMENT_KIND_B32
                                       : LOOMC_CMD_ARGUMENT_KIND_INDEX;
  const std::vector<uint64_t> scalars =
      publish ? std::vector<uint64_t>{1, 3, 128, 192, 16, 12, 8}
      : gated
          ? std::vector<uint64_t>{2, 1, 3, 128, 192, 0x3d800000, 64, 16, 12, 8}
          : std::vector<uint64_t>{2, 1, 3, 128, 192, 0x3d800000, 16, 12, 8};
  for (size_t i = 0; i < n; ++i)
    invocation.arguments[i].value.scalar_bits = scalars[i];
  for (size_t i = n; i < invocation.arguments.size(); ++i) {
    invocation.arguments[i].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[i].value.buffer =
        Ref(static_cast<uint32_t>(i - n), 64 * (i - n + 1));
  }
  return invocation;
}
Plan Profile() {
  Plan plan;
  plan.workload.token_capacity = 8;
  plan.workload.active_tokens = 5;
  plan.workload.history_tokens = 65;
  plan.workload.context_capacity = 128;
  plan.workload.seed = 7;
  return plan;
}
constexpr uint64_t kBudget = 16 * 1024 * 1024;

TEST(AttentionInputs, PublisherUsesShiftedEpochAndNonidentityPages) {
  const auto invocation = Make(true);
  auto plan = Profile();
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 2}));
  EXPECT_EQ(prepared.source_argument_types.size(), 12);
  EXPECT_EQ(prepared.source_argument_types[4], "offset");
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  EXPECT_TRUE(prepared.resets.empty());
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  EXPECT_TRUE(prepared.weight_sources.empty());
  const auto epoch = invocation.arguments[7].value.buffer;
  const auto map = invocation.arguments[8].value.buffer;
  const auto *epoch_root = FindInputRoot(prepared.roots, epoch);
  const auto *map_root = FindInputRoot(prepared.roots, map);
  ASSERT_TRUE(epoch_root && map_root);
  EXPECT_EQ(Word(*epoch_root, epoch.byte_offset + 12), 5);
  EXPECT_EQ(Word(*epoch_root, epoch.byte_offset + 16), 65);
  EXPECT_EQ(Word(*map_root, map.byte_offset + 8), 1);
  EXPECT_EQ(Word(*map_root, map.byte_offset + 12), 2);
  const auto cache = invocation.arguments[11].value.buffer;
  const auto *cache_root = FindInputRoot(prepared.roots, cache);
  ASSERT_TRUE(cache_root);
  const uint64_t plane = 192 * 2 * 128 * 2;
  EXPECT_EQ(cache_root->bytes.size(), 16 + 4 * plane);
  const size_t selected_begin = 16 + 2 * plane;
  EXPECT_TRUE(std::any_of(cache_root->bytes.begin() + selected_begin,
                          cache_root->bytes.end(),
                          [](uint8_t byte) { return byte != 0; }));
  EXPECT_TRUE(std::all_of(cache_root->bytes.begin(),
                          cache_root->bytes.begin() + selected_begin,
                          [](uint8_t byte) { return byte == 0; }));
}

TEST(AttentionInputs, CausalAndGatedAttentionUseExactHeadGeometry) {
  const auto plan = Profile();
  for (const auto &invocation :
       {Make(false), Make(false, 128, true), Make(false, 256, true)}) {
    PreparedCase prepared;
    std::string error;
    ASSERT_EQ(
        PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
        PreparationStatus::kReady)
        << error;
    EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 4, 2}));
    EXPECT_EQ(prepared.source_argument_types[5], "f32");
    EXPECT_TRUE(prepared.resets.empty());
    EXPECT_EQ(prepared.reset_policy, "none_overwrite_only");
    EXPECT_EQ(prepared.buffer_extents.back().byte_length,
              8 * 4 *
                  (invocation.kernel_symbol.find("h256") != std::string::npos
                       ? 256
                       : 128) *
                  4);
  }
}

TEST(AttentionInputs, DenseCacheDoesNotReadOrAllocateMapOrigin) {
  auto invocation = Make(false);
  invocation.arguments[4].value.scalar_bits = 0;
  invocation.arguments[8].value.scalar_bits = UINT64_MAX;
  const auto map = invocation.arguments[10].value.buffer;
  invocation.arguments[10].value.buffer.byte_length = 4;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  const auto *root = FindInputRoot(prepared.roots, map);
  ASSERT_TRUE(root);
  EXPECT_EQ(root->bytes.size(), 4);
  EXPECT_TRUE(root->initialized_ranges.empty());
}

TEST(AttentionInputs, ReadOnlyAliasesAgreeAndWriteAliasesStayUnsupported) {
  auto invocation = Make(false, 128, true);
  invocation.arguments[13].value.buffer = invocation.arguments[12].value.buffer;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  invocation.arguments[15].value.buffer = invocation.arguments[12].value.buffer;
  prepared.setup_recipe_id = "preserved";
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  EXPECT_NE(error.find("disjoint read/write"), std::string::npos);
}

TEST(AttentionInputs, RejectsBadGeometryScaleRangesAndBudgetAtomically) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  auto invocation = Make(false);
  --invocation.command.argument_count;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  invocation = Make(false);
  invocation.arguments[5].value.scalar_bits |= uint64_t{1} << 32;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  invocation = Make(false);
  invocation.arguments[0].value.scalar_bits = 3;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[5].value.scalar_bits = 0x7fc00000;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[6].value.scalar_bits = UINT64_MAX;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[8].value.scalar_bits = 9;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  invocation.arguments[9].value.buffer.byte_length = 16;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Make(false);
  EXPECT_EQ(PrepareAttentionCase(Profile(), invocation, 1, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

TEST(AttentionInputs, InsufficientLivePagePoolAndUnknownOperationAreExplicit) {
  auto invocation = Make(true, 256);
  invocation.arguments[3].value.scalar_bits = 64;
  PreparedCase prepared;
  std::string error;
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kUnsupported);
  invocation = Make(true, 256);
  invocation.kernel_symbol = "unrecognized.attention";
  error = "prior provider diagnostic";
  prepared.setup_recipe_id = "preserved";
  EXPECT_EQ(
      PrepareAttentionCase(Profile(), invocation, kBudget, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "prior provider diagnostic");
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

class CacheSpanInputs : public ::testing::Test {
protected:
  void SetUp() override {
    const char *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(temporary && *temporary);
    source = std::filesystem::path(temporary) / "attention-recipe-weights.bin";
    std::ofstream file(source, std::ios::binary);
    for (unsigned i = 0; i < 320; ++i) {
      const float value = i < 256 ? 2.0f : 1.25f;
      uint32_t bits;
      std::memcpy(&bits, &value, 4);
      for (unsigned byte = 0; byte < 4; ++byte)
        file.put(static_cast<char>(bits >> (byte * 8)));
    }
    file.close();
    ASSERT_TRUE(file);
  }
  void TearDown() override { std::filesystem::remove(source); }
  Plan Model(const std::string &name) {
    Plan plan = Profile();
    plan.source_asset = source.string();
    const bool llama = name.find("llama32_") == 0;
    const bool gemma = name.find("gemma3_") == 0;
    plan.model.key_head_width = plan.model.value_head_width = llama ? 128 : 256;
    plan.model.embedding_width = 4 * plan.model.key_head_width;
    model::LayerDescriptor layer;
    layer.query_heads = 4;
    layer.kv_heads = gemma ? 1 : 2;
    plan.model.layers = {layer, layer};
    model::TensorRecord gamma;
    gamma.name = "norm.weight";
    gamma.byte_size = 1024;
    gamma.rank = 1;
    gamma.normalized_extents = {256, 1, 1, 1};
    auto factors = gamma;
    factors.name = "rope_freqs.weight";
    factors.byte_size = 256;
    factors.absolute_offset = 1024;
    factors.normalized_extents = {64, 1, 1, 1};
    plan.model.globals = {{"query_norm", gamma}, {"rope_frequencies", factors}};
    plan.programs.resize(1);
    plan.programs[0].parameters = {{gamma.name, 0, 64, 1024, 4},
                                   {factors.name, 0, 2048, 256, 4}};
    return plan;
  }
  Invocation Selected(const std::string &name) {
    const bool llama = name.find("llama32_") == 0;
    const bool gemma = name.find("gemma3_") == 0;
    const bool publication = name.find("publish_kv") != std::string::npos ||
                             name.find("kv_prepare") != std::string::npos;
    const bool query = name == "qwen38_attention_query_prepare_spans";
    const bool wmma = name == "qwen38_attention_spans_f16_wmma";
    std::vector<std::string> types;
    if (llama)
      types = {"index", "offset", "index", "offset", "index", "offset"};
    else if (gemma)
      types = {"index", "offset", "offset", "offset"};
    if (llama || gemma)
      types.insert(types.end(), llama && publication ? 6 : 5, "buffer");
    else if (query)
      types = {"buffer", "f32", "buffer", "buffer", "buffer", "buffer"};
    else if (publication)
      types = {"buffer", "buffer", "offset", "f32",
               "buffer", "buffer", "buffer", "buffer"};
    else
      types = {"buffer", "buffer", "offset", "buffer",
               "buffer", "buffer", "buffer"};
    Invocation invocation;
    invocation.kernel_symbol = name;
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = types.size();
    invocation.command.payload.dispatch.workgroup_count[0] =
        wmma          ? 1
        : publication ? (gemma ? 1 : 2)
                      : 4;
    invocation.command.payload.dispatch.workgroup_count[1] = wmma ? 4 : 8;
    invocation.command.payload.dispatch.workgroup_count[2] =
        llama || gemma ? 1 : 4;
    invocation.arguments.resize(types.size());
    for (size_t i = 0; i < types.size(); ++i) {
      auto &argument = invocation.arguments[i];
      argument.kind = types[i] == "index"    ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                      : types[i] == "offset" ? LOOMC_CMD_ARGUMENT_KIND_OFFSET
                      : types[i] == "f32"    ? LOOMC_CMD_ARGUMENT_KIND_B32
                                             : LOOMC_CMD_ARGUMENT_KIND_BUFFER;
      if (types[i] == "buffer")
        argument.value.buffer = Ref(i + 1, 64);
      else if (types[i] == "f32")
        argument.value.scalar_bits = 0x35800000;
    }
    if (llama || gemma) {
      invocation.arguments[0].value.scalar_bits = 1;
      invocation.arguments[1].value.scalar_bits = 16;
      invocation.arguments[llama ? 3 : 2].value.scalar_bits = 12;
      invocation.arguments[llama ? 5 : 3].value.scalar_bits = 8;
    } else if (!query)
      invocation.arguments[2].value.scalar_bits = 4096;
    if (query || (!llama && !gemma && publication) || (llama && publication)) {
      const size_t parameter = query ? 3 : llama ? 8 : 6;
      invocation.arguments[parameter].value.buffer = {
          LOOMC_CMD_BUFFER_ROLE_FIXED, 0, llama ? uint64_t{2048} : uint64_t{64},
          llama ? uint64_t{256} : uint64_t{1024}};
    }
    return invocation;
  }
  std::filesystem::path source;
};

TEST_F(CacheSpanInputs, AllSevenSelectedWrapperAbisPrepare) {
  for (const std::string name :
       {"llama32_publish_kv", "llama32_attention_rows", "gemma3_publish_kv",
        "gemma3_attention_rows", "qwen38_attention_query_prepare_spans",
        "qwen38_attention_kv_prepare_spans",
        "qwen38_attention_spans_f16_wmma"}) {
    SCOPED_TRACE(name);
    PreparedCase prepared;
    std::string error;
    const auto invocation = Selected(name);
    ASSERT_EQ(PrepareAttentionCase(Model(name), invocation, kBudget, &prepared,
                                   &error),
              PreparationStatus::kReady)
        << error;
    EXPECT_EQ(prepared.source_argument_types.size(),
              invocation.arguments.size());
    EXPECT_EQ(prepared.measurement_subject, "kernel");
    EXPECT_TRUE(prepared.resets.empty());
    EXPECT_EQ(prepared.arithmetic_check, "unchecked");
    EXPECT_EQ(prepared.weight_sources.size(),
              name == "llama32_publish_kv" ||
                      name == "qwen38_attention_query_prepare_spans" ||
                      name == "qwen38_attention_kv_prepare_spans"
                  ? 1
                  : 0);
  }
}

TEST_F(CacheSpanInputs, QwenPacketsKeepInactiveSpansAndChannelProbabilityGate) {
  const std::string name = "qwen38_attention_spans_f16_wmma";
  auto invocation = Selected(name);
  auto plan = Model(name);
  plan.workload.context_capacity = 253;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 4}));
  const auto epoch = invocation.arguments[0].value.buffer;
  const auto rows = invocation.arguments[1].value.buffer;
  const auto gate = invocation.arguments[4].value.buffer;
  ASSERT_TRUE(FindInputRoot(prepared.roots, epoch));
  EXPECT_EQ(Word(*FindInputRoot(prepared.roots, epoch), epoch.byte_offset + 4),
            1);
  EXPECT_EQ(Word(*FindInputRoot(prepared.roots, epoch), epoch.byte_offset + 16),
            65);
  EXPECT_EQ(Word(*FindInputRoot(prepared.roots, epoch), epoch.byte_offset + 32),
            0);
  EXPECT_EQ(Word(*FindInputRoot(prepared.roots, rows), rows.byte_offset + 2048),
            3);
  const auto cache = invocation.arguments[5].value.buffer;
  const auto *cache_root = FindInputRoot(prepared.roots, cache);
  ASSERT_TRUE(cache_root);
  ASSERT_EQ(cache_root->bytes.size(), 4096 + 2 * 256 * 2 * 256 * 2);
  EXPECT_EQ(cache_root->bytes[4095], 0);
  EXPECT_NE(cache_root->bytes[4097], 0);
  const auto *gate_root = FindInputRoot(prepared.roots, gate);
  ASSERT_TRUE(gate_root);
  EXPECT_EQ(gate_root->bytes.size(), 8 * 4 * 256 * 4);
  for (uint64_t offset = 0; offset < gate_root->bytes.size(); offset += 4)
    EXPECT_EQ(Word(*gate_root, gate.byte_offset + offset), 0x3f000000);
}

TEST_F(CacheSpanInputs,
       LegacyDecodeAndShiftedTokenBeginRetainOriginalControls) {
  const std::string name = "llama32_attention_rows";
  auto invocation = Selected(name);
  auto plan = Model(name);
  plan.workload.active_tokens = 1;
  invocation.arguments[2].value.scalar_bits = 1;
  invocation.arguments[4].value.scalar_bits = 3;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  const auto control = invocation.arguments[6].value.buffer;
  const auto *root = FindInputRoot(prepared.roots, control);
  ASSERT_TRUE(root);
  EXPECT_EQ(Word(*root, control.byte_offset + 12), 65);
  EXPECT_EQ(Word(*root, control.byte_offset + 16), 0);
  EXPECT_EQ(prepared.buffer_extents.back().byte_length, 8 * 4 * 128 * 4);
  plan.workload.active_tokens = 2;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kFailure);
}

TEST_F(CacheSpanInputs, FreshHistoryAndExactGammaAreSupported) {
  const std::string name = "qwen38_attention_query_prepare_spans";
  auto invocation = Selected(name);
  auto plan = Model(name);
  plan.workload.history_tokens = 0;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  const auto gamma = invocation.arguments[3].value.buffer;
  EXPECT_EQ(Word(*FindInputRoot(prepared.roots, gamma), gamma.byte_offset),
            0x40000000);
  EXPECT_EQ(prepared.weight_sources[0].key, "norm.weight");
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 1024);
}

TEST_F(CacheSpanInputs, ReadWriteAndWmmaInputAliasesStayExplicitlyUnsupported) {
  const std::string name = "qwen38_attention_spans_f16_wmma";
  auto invocation = Selected(name);
  auto plan = Model(name);
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  invocation.arguments[6].value.buffer = invocation.arguments[3].value.buffer;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  invocation = Selected(name);
  invocation.arguments[4].value.buffer = invocation.arguments[3].value.buffer;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kUnsupported);
}

TEST_F(CacheSpanInputs, ConflictingSemanticReadAliasesAreRecipeUnsupported) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  for (const std::string name :
       {"llama32_attention_rows", "qwen38_attention_query_prepare_spans"}) {
    auto invocation = Selected(name);
    auto plan = Model(name);
    const size_t epoch = name == "llama32_attention_rows" ? 6 : 0;
    const size_t projection = name == "llama32_attention_rows" ? 8 : 2;
    invocation.arguments[projection].value.buffer =
        invocation.arguments[epoch].value.buffer;
    EXPECT_EQ(
        PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
        PreparationStatus::kUnsupported)
        << name << ": " << error;
    EXPECT_EQ(prepared.setup_recipe_id, "preserved");
    EXPECT_NE(error.find("fixture policies overlap"), std::string::npos);
  }
  const std::string name = "qwen38_attention_kv_prepare_spans";
  auto invocation = Selected(name);
  auto plan = Model(name);
  invocation.arguments[1].value.buffer = invocation.arguments[0].value.buffer;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("packets disagree"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  invocation = Selected(name);
  invocation.arguments[4].value.buffer = invocation.arguments[6].value.buffer;
  invocation.arguments[4].value.buffer.byte_length = UINT64_MAX;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kUnsupported);
}

TEST_F(CacheSpanInputs,
       ConsistentF32ReadAliasesRetainFinalControlAndInputBytes) {
  const std::string name = "qwen38_attention_kv_prepare_spans";
  auto invocation = Selected(name);
  auto plan = Model(name);
  invocation.arguments[5].value.buffer = invocation.arguments[4].value.buffer;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  const auto epoch = invocation.arguments[0].value.buffer;
  const auto *root = FindInputRoot(prepared.roots, epoch);
  ASSERT_TRUE(root);
  EXPECT_EQ(Word(*root, epoch.byte_offset), 5);
  EXPECT_EQ(Word(*root, epoch.byte_offset + 4), 1);
  EXPECT_EQ(Word(*root, epoch.byte_offset + 12), 5);
  EXPECT_EQ(Word(*root, epoch.byte_offset + 16), 65);
  const auto input = invocation.arguments[4].value.buffer;
  const auto *input_root = FindInputRoot(prepared.roots, input);
  ASSERT_TRUE(input_root);
  // Absolute word ordinal16 + seed7 gives exactly positive8/1024.
  EXPECT_EQ(Word(*input_root, input.byte_offset), 0x3c000000);
}

TEST_F(CacheSpanInputs, MalformedMetadataWinsOverTinyInputBudget) {
  const std::string name = "qwen38_attention_kv_prepare_spans";
  auto invocation = Selected(name);
  auto plan = Model(name);
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  invocation.arguments[3].value.scalar_bits = 0x7f800000;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, 1, &prepared, &error),
            PreparationStatus::kFailure);
  invocation = Selected(name);
  invocation.arguments[0].value.buffer.byte_length = 4;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, 1, &prepared, &error),
            PreparationStatus::kFailure);
  invocation = Selected(name);
  plan.model.globals[0].tensor.rank = 2;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, 1, &prepared, &error),
            PreparationStatus::kFailure);
  const std::string llama = "llama32_publish_kv";
  invocation = Selected(llama);
  plan = Model(llama);
  plan.model.globals[1].tensor.storage_type = 1;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, 1, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

TEST_F(CacheSpanInputs, InvalidAbiBoundsWeightsAndBudgetsPreserveOutput) {
  const std::string name = "qwen38_attention_kv_prepare_spans";
  auto invocation = Selected(name);
  auto plan = Model(name);
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  invocation.command.argument_count = 0;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kFailure);
  invocation = Selected(name);
  invocation.arguments[3].value.scalar_bits |= uint64_t{1} << 32;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kFailure);
  invocation = Selected(name);
  invocation.arguments[2].value.scalar_bits = UINT64_MAX - 1;
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kFailure);
  invocation = Selected(name);
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, 1, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  plan.programs[0].parameters.clear();
  EXPECT_EQ(PrepareAttentionCase(plan, invocation, kBudget, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}

} // namespace
} // namespace hrx::loomify_gguf::benchmark
