// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
loomc_cmd_argument_t Index(uint64_t value) {
  loomc_cmd_argument_t arg{};
  arg.kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
  arg.value.scalar_bits = value;
  return arg;
}
loomc_cmd_argument_t Offset(uint64_t value) {
  auto arg = Index(value);
  arg.kind = LOOMC_CMD_ARGUMENT_KIND_OFFSET;
  return arg;
}
loomc_cmd_argument_t
Buffer(uint32_t root, uint64_t offset, uint64_t length,
       loomc_cmd_buffer_role_t role = LOOMC_CMD_BUFFER_ROLE_REBINDABLE) {
  loomc_cmd_argument_t arg{};
  arg.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
  arg.value.buffer = {role, root, offset, length};
  return arg;
}
Invocation Invoke(const std::string &name,
                  std::initializer_list<loomc_cmd_argument_t> args, uint32_t x,
                  uint32_t y) {
  Invocation invocation;
  invocation.kernel_symbol = name;
  invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
  invocation.command.argument_count = args.size();
  invocation.command.payload.dispatch.workgroup_count[0] = x;
  invocation.command.payload.dispatch.workgroup_count[1] = y;
  invocation.command.payload.dispatch.workgroup_count[2] = 1;
  invocation.arguments = args;
  return invocation;
}
uint32_t ReadWord(const PreparedCase &prepared, loomc_cmd_buffer_ref_t binding,
                  uint64_t origin = 0) {
  const auto *root = FindInputRoot(prepared.roots, binding);
  EXPECT_NE(root, nullptr);
  if (!root)
    return 0;
  const auto begin = binding.byte_offset + origin - root->original_byte_offset;
  EXPECT_LE(begin + 4, root->bytes.size());
  if (begin + 4 > root->bytes.size())
    return 0;
  uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i)
    value |= uint32_t(root->bytes[begin + i]) << (i * 8);
  return value;
}
class ElementwiseInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    plan.workload.token_capacity = 8;
    plan.workload.active_tokens = 3;
    plan.workload.history_tokens = 5;
    plan.workload.context_capacity = 512;
    plan.workload.seed = 7;
    plan.model.embedding_width = 1024;
    plan.model.key_head_width = 128;
    plan.model.layers.resize(1);
    plan.model.layers[0].query_heads = 8;
    plan.model.ssm = model::SsmDescriptor{};
    plan.model.ssm->inner_width = 1024;
  }
  void Ready(const Invocation &invocation, PreparedCase *prepared) {
    ASSERT_EQ(
        PrepareElementwiseCase(plan, invocation, 1 << 20, prepared, &error),
        PreparationStatus::kReady)
        << error;
    EXPECT_EQ(prepared->arithmetic_check, "unchecked");
    size_t ordinal = 0;
    for (const auto &argument : invocation.arguments)
      if (argument.kind == LOOMC_CMD_ARGUMENT_KIND_BUFFER) {
        ASSERT_LT(ordinal, prepared->buffer_extents.size());
        EXPECT_EQ(prepared->buffer_extents[ordinal].binding.root_index,
                  argument.value.buffer.root_index);
        EXPECT_EQ(prepared->buffer_extents[ordinal].binding.byte_offset,
                  argument.value.buffer.byte_offset);
        ++ordinal;
      }
    EXPECT_EQ(ordinal, prepared->buffer_extents.size());
  }
  Plan plan;
  std::string error;
};

TEST_F(ElementwiseInputsTest, DecoderResidualRestoresOnlyActiveReadWriteRows) {
  const auto invocation =
      Invoke("llm.decoder.residual",
             {Index(1024), Buffer(0, 16, 12), Buffer(0, 256, 32768),
              Buffer(0, 40000, 32768)},
             1, 8);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 1024}));
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[1].value.buffer), 3);
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[1].value.buffer, 4), 0);
  ASSERT_EQ(prepared.resets.size(), 1);
  const auto &reset = prepared.resets[0];
  EXPECT_EQ(reset.target.byte_offset, 40000);
  EXPECT_EQ(reset.target.byte_length, 3 * 1024 * 4);
  const auto *root = FindInputRoot(prepared.roots, reset.target);
  ASSERT_NE(root, nullptr);
  const auto begin = reset.target.byte_offset - root->original_byte_offset;
  EXPECT_TRUE(std::equal(reset.seed.bytes.begin(), reset.seed.bytes.end(),
                         root->bytes.begin() + begin));
  EXPECT_EQ(prepared.measurement_subject, "reset_plus_kernel");
  EXPECT_EQ(prepared.reset_policy, "copy_seed_each_invocation");
}

TEST_F(ElementwiseInputsTest, ExactOutputAndBothInputAliasesProduceOneSeed) {
  const auto same = Buffer(0, 64, 32768);
  const auto invocation = Invoke("qwen38_residual_add_f32_prefill",
                                 {Index(8), same, same, same}, 1, 3);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  ASSERT_EQ(prepared.roots.size(), 1);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_offset, 64);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 3 * 1024 * 4);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3}));
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 8);
}

TEST_F(ElementwiseInputsTest, ResetSeedBudgetIncludesMergedAliases) {
  const auto same = Buffer(0, 64, 32768);
  const auto invocation = Invoke("qwen38_residual_add_f32_prefill",
                                 {Index(8), same, same, same}, 1, 3);
  PreparedCase prepared;
  prepared.setup_recipe_id = "retained";
  // The single shared root fits, but its active read-write seed does not.
  EXPECT_EQ(PrepareElementwiseCase(plan, invocation, 32768, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "retained");
  EXPECT_TRUE(prepared.roots.empty());
  EXPECT_TRUE(prepared.resets.empty());
  // Both input aliases need one merged seed, not two copies.
  ASSERT_EQ(PrepareElementwiseCase(plan, invocation, 32768 + 3 * 1024 * 4,
                                   &prepared, &error),
            PreparationStatus::kReady)
      << error;
  ASSERT_EQ(prepared.roots.size(), 1);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.roots[0].bytes.size(), 32768);
  EXPECT_EQ(prepared.resets[0].seed.bytes.size(), 3 * 1024 * 4);
}

TEST_F(ElementwiseInputsTest, ReadOnlyPartialAliasesGetConsistentFiniteBytes) {
  const auto invocation = Invoke("llm_ffn_swiglu_f32_prefill",
                                 {Index(256), Index(8), Buffer(0, 64, 8192),
                                  Buffer(0, 80, 8192), Buffer(1, 0, 8192)},
                                 1, 3);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  EXPECT_EQ(prepared.roots.size(), 2);
  EXPECT_TRUE(prepared.resets.empty());
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3, 256}));
  const auto *root =
      FindInputRoot(prepared.roots, invocation.arguments[2].value.buffer);
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(root, FindInputRoot(prepared.roots,
                                invocation.arguments[3].value.buffer));
  EXPECT_EQ(root->initialized_ranges.size(), 2);
}

TEST_F(ElementwiseInputsTest,
       InPlaceGateCapturesLaunchedRowsNotPaddedCapacity) {
  const auto gate = Buffer(0, 32, 8192);
  const auto invocation =
      Invoke("llm_ffn_swiglu_f32_prefill",
             {Index(256), Index(8), gate, Buffer(1, 0, 8192), gate}, 1, 3);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_offset, 32);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 3 * 256 * 4);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 8 * 256 * 4);
}

TEST_F(ElementwiseInputsTest, ActiveFfnPreservesRowAndCountOrigins) {
  const auto residual = Buffer(1, 128, 32768);
  const auto invocation =
      Invoke("llm_ffn_residual_add_f32_prefill_active",
             {Index(1024), Index(8), Index(2), Offset(16), Buffer(0, 0, 20),
              residual, Buffer(2, 0, 32768), residual},
             1, 8);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[4].value.buffer, 16), 3);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_offset, 128 + 2 * 1024 * 4);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 3 * 1024 * 4);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 1024}));
  EXPECT_EQ(prepared.source_argument_types[3], "offset");
  EXPECT_EQ(invocation.arguments[3].value.scalar_bits, 16);
}

TEST_F(ElementwiseInputsTest,
       ActiveFfnResidualTailResetsOnlySelectedLaunchedRows) {
  plan.model.embedding_width = 4096;
  const uint64_t width = 1536;
  const auto residual = Buffer(1, 128, 8 * width * 4);
  const auto invocation =
      Invoke("llm_ffn_residual_add_f32_prefill_active",
             {Index(width), Index(8), Index(2), Offset(16), Buffer(0, 0, 20),
              residual, Buffer(2, 64, 8 * width * 4), residual},
             2, 4);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{4, width}));
  EXPECT_EQ(prepared.source_argument_types,
            (std::vector<std::string>{"index", "index", "index", "offset",
                                      "buffer", "buffer", "buffer", "buffer"}));
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[4].value.buffer, 16), 3);
  ASSERT_EQ(prepared.buffer_extents.size(), 4);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 20);
  for (size_t i = 1; i < prepared.buffer_extents.size(); ++i)
    EXPECT_EQ(prepared.buffer_extents[i].byte_length, 8 * width * 4);
  ASSERT_EQ(prepared.resets.size(), 1);
  const auto &reset = prepared.resets[0];
  EXPECT_EQ(reset.target.byte_offset, 128 + 2 * width * 4);
  // Count3 extends beyond launch4 at origin2: only rows2 and3 execute.
  EXPECT_EQ(reset.target.byte_length, 2 * width * 4);
  const auto *root = FindInputRoot(prepared.roots, reset.target);
  ASSERT_NE(root, nullptr);
  EXPECT_TRUE(std::equal(reset.seed.bytes.begin(), reset.seed.bytes.end(),
                         root->bytes.begin() + 2 * width * 4));
  EXPECT_EQ(prepared.reset_policy, "copy_seed_each_invocation");
}

TEST_F(ElementwiseInputsTest, GatingVariantsKeepOperationWorkloadsAndPackets) {
  std::vector<Invocation> invocations = {
      Invoke("llm.projection.active_swiglu",
             {Index(8), Index(256), Buffer(0, 0, 12), Buffer(1, 0, 8192),
              Buffer(2, 0, 8192), Buffer(3, 0, 8192)},
             1, 8),
      Invoke("gemma3_geglu",
             {Index(256), Buffer(0, 0, 4), Buffer(1, 0, 8192),
              Buffer(2, 0, 8192), Buffer(3, 0, 8192)},
             1, 8),
      Invoke("llm_ffn_swiglu_f32_prefill_active",
             {Index(256), Index(8), Index(2), Offset(16), Buffer(0, 0, 20),
              Buffer(1, 0, 8192), Buffer(2, 0, 8192), Buffer(3, 0, 8192)},
             1, 8)};
  for (const auto &invocation : invocations) {
    SCOPED_TRACE(invocation.kernel_symbol);
    PreparedCase prepared;
    Ready(invocation, &prepared);
    EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 256}));
    EXPECT_TRUE(prepared.resets.empty());
  }
}

TEST_F(ElementwiseInputsTest,
       DecodeAndOrdinaryResidualVariantsSharePreparation) {
  plan.workload.active_tokens = 1;
  const auto residual_input = Buffer(0, 0, 4096);
  for (const auto &invocation :
       {Invoke("qwen38_residual_add_f32_decode",
               {residual_input, Buffer(1, 0, 4096), residual_input}, 1, 1),
        Invoke(
            "llm_ffn_residual_add_f32_decode",
            {Index(1024), residual_input, Buffer(1, 0, 4096), residual_input},
            1, 1)}) {
    PreparedCase prepared;
    Ready(invocation, &prepared);
    ASSERT_EQ(prepared.resets.size(), 1);
    EXPECT_EQ(prepared.resets[0].target.byte_length, 4096);
  }
  PreparedCase gate;
  Ready(Invoke("llm_ffn_swiglu_f32_decode",
               {Index(256), Buffer(0, 0, 1024), Buffer(1, 0, 1024),
                Buffer(2, 0, 1024)},
               1, 1),
        &gate);
  EXPECT_EQ(gate.source_workloads, (std::vector<uint64_t>{256}));
  PreparedCase residual;
  Ready(Invoke("llm_ffn_residual_add_f32_prefill",
               {Index(1024), Index(8), Buffer(0, 0, 32768), Buffer(1, 0, 32768),
                Buffer(2, 0, 32768)},
               1, 8),
        &residual);
  EXPECT_EQ(residual.source_workloads, (std::vector<uint64_t>{8, 1024}));
}

TEST_F(ElementwiseInputsTest, FfnResidualTailsUseLogicalWidthAndLaunchedRows) {
  // Neither the model width nor the rounded grid describes the row stride.
  plan.model.embedding_width = 4096;
  plan.workload.active_tokens = 1;
  for (uint64_t width : {1024, 1536, 2560}) {
    for (bool prefill : {false, true}) {
      SCOPED_TRACE(::testing::Message() << width << "/prefill=" << prefill);
      const uint64_t capacity = prefill ? 8 : 1;
      const uint64_t launched = prefill ? 3 : 1;
      const uint64_t bytes = capacity * width * 4;
      const auto lhs = Buffer(0, 64, bytes);
      const auto rhs = Buffer(1, 128, bytes);
      const auto output = prefill ? rhs : lhs;
      const uint32_t groups = (width + 1023) / 1024;
      const auto invocation =
          prefill ? Invoke("llm_ffn_residual_add_f32_prefill",
                           {Index(width), Index(capacity), lhs, rhs, output},
                           groups, launched)
                  : Invoke("llm_ffn_residual_add_f32_decode",
                           {Index(width), lhs, rhs, output}, groups, launched);
      PreparedCase prepared;
      Ready(invocation, &prepared);
      EXPECT_EQ(prepared.source_workloads,
                prefill ? (std::vector<uint64_t>{launched, width})
                        : (std::vector<uint64_t>{width}));
      EXPECT_EQ(prepared.source_argument_types,
                prefill ? (std::vector<std::string>{"index", "index", "buffer",
                                                    "buffer", "buffer"})
                        : (std::vector<std::string>{"index", "buffer", "buffer",
                                                    "buffer"}));
      ASSERT_EQ(prepared.buffer_extents.size(), 3);
      for (const auto &extent : prepared.buffer_extents)
        EXPECT_EQ(extent.byte_length, bytes);
      ASSERT_EQ(prepared.resets.size(), 1);
      const auto &reset = prepared.resets.front();
      EXPECT_EQ(reset.target.root_index, output.value.buffer.root_index);
      EXPECT_EQ(reset.target.byte_offset, output.value.buffer.byte_offset);
      EXPECT_EQ(reset.target.byte_length, launched * width * 4);
      EXPECT_EQ(reset.seed.bytes.size(), launched * width * 4);
      const auto *root = FindInputRoot(prepared.roots, reset.target);
      ASSERT_NE(root, nullptr);
      // A partial tile does not allocate/reset a padded 2048/3072-wide row.
      EXPECT_EQ(root->bytes.size(), bytes);
      EXPECT_TRUE(std::equal(reset.seed.bytes.begin(), reset.seed.bytes.end(),
                             root->bytes.begin()));
      EXPECT_NE(ReadWord(prepared, reset.target) & 0x7F800000u, 0x7F800000u);
      EXPECT_EQ(prepared.reset_policy, "copy_seed_each_invocation");
    }
  }
}

TEST_F(ElementwiseInputsTest,
       FfnResidualDecodeStageKeepsCapacity512ButResetsOneRow) {
  plan.workload.token_capacity = plan.workload.active_tokens = 1;
  const uint64_t bytes = 512 * 1024 * 4;
  const auto residual = Buffer(0, 0, bytes);
  const auto invocation = Invoke(
      "llm_ffn_residual_add_f32_prefill",
      {Index(1024), Index(512), residual, Buffer(6, 4194560, bytes), residual},
      1, 1);
  PreparedCase prepared;
  ASSERT_EQ(
      PrepareElementwiseCase(plan, invocation, 8 << 20, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{1, 1024}));
  ASSERT_EQ(prepared.buffer_extents.size(), 3);
  for (const auto &extent : prepared.buffer_extents)
    EXPECT_EQ(extent.byte_length, bytes);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 1024 * 4);
  EXPECT_EQ(prepared.resets[0].seed.bytes.size(), 1024 * 4);
}

TEST_F(ElementwiseInputsTest, FfnResidualRejectsStaleAbiAndMismatchedTailGrid) {
  const auto lhs = Buffer(0, 0, 49152);
  const auto rhs = Buffer(1, 0, 49152);
  const auto out = Buffer(2, 0, 49152);
  for (const auto &invocation :
       {Invoke("llm_ffn_residual_add_f32_decode", {lhs, rhs, out}, 1, 1),
        Invoke("llm_ffn_residual_add_f32_prefill", {Index(8), lhs, rhs, out}, 1,
               1),
        Invoke("llm_ffn_residual_add_f32_prefill",
               {Index(1536), Index(8), lhs, rhs, out}, 1, 1),
        Invoke("llm_ffn_residual_add_f32_prefill",
               {Index(1536), Index(8), lhs, rhs, out}, 3, 1),
        Invoke("llm_ffn_residual_add_f32_prefill",
               {Index(1537), Index(8), lhs, rhs, out}, 2, 1),
        Invoke("llm_ffn_residual_add_f32_prefill",
               {Index(1536), Index(0), lhs, rhs, out}, 2, 1),
        Invoke("llm_ffn_residual_add_f32_prefill",
               {Index(1536), Index(2), lhs, rhs, out}, 2, 3),
        Invoke("llm_ffn_residual_add_f32_decode", {Index(1536), lhs, rhs, out},
               2, 2),
        Invoke(
            "llm_ffn_residual_add_f32_prefill_active",
            {Index(8), Index(2), Offset(16), Buffer(3, 0, 20), lhs, rhs, out},
            1, 8),
        Invoke("llm_ffn_residual_add_f32_prefill_active",
               {Index(1536), Index(8), Index(2), Offset(16), Buffer(3, 0, 20),
                lhs, rhs, out},
               1, 8)}) {
    PreparedCase prepared;
    prepared.setup_recipe_id = "retained";
    EXPECT_EQ(
        PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
        PreparationStatus::kFailure);
    EXPECT_EQ(prepared.setup_recipe_id, "retained");
  }
}

TEST_F(ElementwiseInputsTest, NonTileWidthResidualUsesModelGeometry) {
  plan.model.embedding_width = 1152;
  PreparedCase prepared;
  Ready(Invoke("gemma3_residual",
               {Buffer(0, 0, 4), Buffer(1, 0, 36864), Buffer(2, 0, 36864)}, 2,
               8),
        &prepared);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 3 * 1152 * 4);
}

TEST_F(ElementwiseInputsTest,
       AttentionResidualSupportsPrefillAndDecodeControl) {
  const auto count = Buffer(0, 0, 28);
  const auto residual = Buffer(1, 0, 32768);
  PreparedCase prefill;
  const auto invocation = Invoke(
      "llama32_attention_residual",
      {Index(0), Offset(16), Index(2), count, Buffer(2, 0, 32768), residual}, 1,
      8);
  Ready(invocation, &prefill);
  EXPECT_EQ(ReadWord(prefill, count.value.buffer, 16), 3);
  EXPECT_EQ(ReadWord(prefill, count.value.buffer, 20), 5);
  ASSERT_EQ(prefill.resets.size(), 1);
  EXPECT_EQ(prefill.resets[0].target.byte_offset, 2 * 1024 * 4);
  plan.workload.active_tokens = 1;
  PreparedCase decode;
  Ready(Invoke("llama32_attention_residual",
               {Index(1), Offset(16), Index(0), count, Buffer(2, 0, 4096),
                Buffer(1, 0, 4096)},
               1, 1),
        &decode);
  EXPECT_EQ(ReadWord(decode, count.value.buffer, 16), 5);
}

TEST_F(ElementwiseInputsTest, ClearsUseValidNonzeroOverwriteFixtures) {
  for (const auto &invocation :
       {Invoke("qwen08_gdn_clear_raw", {Index(8), Buffer(0, 0, 32768)}, 32, 1),
        Invoke("qwen38_attention_clear_attended",
               {Index(8), Buffer(0, 0, 32768)}, 3, 1)}) {
    PreparedCase prepared;
    Ready(invocation, &prepared);
    EXPECT_TRUE(prepared.resets.empty());
    EXPECT_NE(ReadWord(prepared, invocation.arguments[1].value.buffer), 0);
  }
}

TEST_F(ElementwiseInputsTest, UnsupportedAndUnsafeCasesPreserveOutput) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "retained";
  error = "prior reason";
  auto invocation = Invoke("unknown.operation", {}, 1, 1);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "prior reason");
  invocation = Invoke(
      "qwen38_residual_add_f32_prefill",
      {Index(8), Buffer(0, 0, 32768), Buffer(1, 0, 32768), Buffer(0, 4, 32768)},
      1, 8);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("partial"), std::string::npos);
  invocation.arguments[3] = Buffer(2, 0, 32768);
  EXPECT_EQ(PrepareElementwiseCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  invocation.arguments[0].kind = LOOMC_CMD_ARGUMENT_KIND_B32;
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "retained");
}

TEST_F(ElementwiseInputsTest, F32PrefixPreservesCapacityAndLogicalRows) {
  plan.workload.active_tokens = 2;
  const auto invocation = Invoke("llm.projection.prepare_f32_prefix",
                                 {Index(8), Index(256), Buffer(0, 16, 12),
                                  Buffer(1, 64, 8192), Buffer(1, 9000, 8192)},
                                 3, 1);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3, 256}));
  EXPECT_EQ(prepared.buffer_extents[1].byte_length, 8192);
  EXPECT_EQ(prepared.buffer_extents[2].byte_length, 8192);
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[2].value.buffer), 2);
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[2].value.buffer, 8), 0);
  EXPECT_TRUE(prepared.resets.empty());
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  EXPECT_NE(ReadWord(prepared, invocation.arguments[4].value.buffer), 0);
}

TEST_F(ElementwiseInputsTest, F32PrefixInPlaceIncludesOnlyActiveRowsInBudget) {
  plan.workload.active_tokens = 2;
  const auto same = Buffer(1, 64, 8192);
  const auto invocation =
      Invoke("llm.projection.prepare_f32_prefix",
             {Index(8), Index(256), Buffer(0, 16, 12), same, same}, 3, 1);
  PreparedCase prepared;
  prepared.setup_recipe_id = "retained";
  EXPECT_EQ(PrepareElementwiseCase(plan, invocation, 8204, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "retained");
  ASSERT_EQ(
      PrepareElementwiseCase(plan, invocation, 8204 + 2048, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_offset, 64);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 2048);
  EXPECT_EQ(prepared.measurement_subject, "reset_plus_kernel");
}

TEST_F(ElementwiseInputsTest, F32SplitUsesUnequalDisjointViewsAndCount) {
  plan.workload.active_tokens = 2;
  const auto invocation = Invoke(
      "llm.decoder.split_qkv",
      {Index(512), Index(256), Buffer(0, 16, 12), Buffer(1, 64, 12288),
       Buffer(1, 13000, 6144), Buffer(1, 20000, 3072), Buffer(1, 24000, 3072)},
      3, 1);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3, 512, 256}));
  ASSERT_EQ(prepared.buffer_extents.size(), 5);
  EXPECT_EQ(prepared.buffer_extents[1].byte_length, 12288);
  EXPECT_EQ(prepared.buffer_extents[2].byte_length, 6144);
  EXPECT_EQ(prepared.buffer_extents[3].byte_length, 3072);
  EXPECT_EQ(prepared.buffer_extents[4].byte_length, 3072);
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[2].value.buffer), 2);
  EXPECT_TRUE(prepared.resets.empty());
  EXPECT_EQ(prepared.roots.size(), 2);
}

TEST_F(ElementwiseInputsTest, F32RowsRejectUnsafeAliasesAndMalformedGeometry) {
  auto invocation =
      Invoke("llm.decoder.split_qkv",
             {Index(512), Index(256), Buffer(0, 16, 12), Buffer(1, 64, 12288),
              Buffer(2, 0, 6144), Buffer(2, 0, 3072), Buffer(3, 0, 3072)},
             3, 1);
  PreparedCase prepared;
  prepared.setup_recipe_id = "retained";
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("disjoint"), std::string::npos);
  invocation.arguments[5] = Buffer(4, 0, 3072);
  EXPECT_EQ(PrepareElementwiseCase(plan, invocation, 16, &prepared, &error),
            PreparationStatus::kUnsupported);
  invocation.arguments[1] = Index(257);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  invocation.arguments[1] = Index(256);
  invocation.command.payload.dispatch.workgroup_count[1] = 2;
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  invocation.command.payload.dispatch.workgroup_count[1] = 1;
  invocation.arguments[0].kind = LOOMC_CMD_ARGUMENT_KIND_B32;
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  invocation = Invoke("llm.projection.prepare_f32_prefix",
                      {Index(8), Index(256), Buffer(0, 16, 12),
                       Buffer(1, 64, 8192), Buffer(1, 68, 8192)},
                      3, 1);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kUnsupported);
  invocation.arguments[3].value.buffer.byte_length = 4;
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "retained");
}

TEST_F(ElementwiseInputsTest, DenseF32UsesExactRectangularWeight) {
  const char *temporary = std::getenv("TEST_TMPDIR");
  ASSERT_NE(temporary, nullptr);
  const auto path = std::filesystem::path(temporary) / "dense-f32.gguf";
  plan.source_asset = path.string();
  plan.workload.active_tokens = 2;
  model::TensorBinding weight;
  weight.tensor.name = "gate.weight";
  weight.tensor.storage_type = 0;
  weight.tensor.rank = 2;
  weight.tensor.normalized_extents = {7, 3, 1, 1};
  weight.tensor.byte_size = 84;
  weight.tensor.absolute_offset = 8;
  plan.model.globals.push_back(weight);
  plan.programs.resize(1);
  plan.programs[0].parameters.push_back({"gate.weight", 7, 1000, 84, 4});
  const auto write = [&](uint32_t bits) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write("metadata", 8);
    for (unsigned i = 0; i < 21; ++i)
      for (unsigned lane = 0; lane < 4; ++lane)
        file.put(static_cast<char>(bits >> (lane * 8)));
  };
  write(0x3E800000);
  auto invocation = Invoke(
      "llm.projection.f32_rows_kernel",
      {Index(7), Index(3), Buffer(0, 16, 12), Buffer(1, 64, 84),
       Buffer(7, 1000, 84, LOOMC_CMD_BUFFER_ROLE_FIXED), Buffer(1, 200, 36)},
      3, 1);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3}));
  EXPECT_EQ(prepared.buffer_extents[1].byte_length, 84);
  EXPECT_EQ(prepared.buffer_extents[3].byte_length, 36);
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "gate.weight");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 8);
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 84);
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[4].value.buffer),
            0x3E800000);
  EXPECT_TRUE(prepared.resets.empty());
  invocation.arguments[5] = Buffer(1, 64, 36);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kUnsupported);
  invocation.arguments[5] = Buffer(1, 200, 36);
  plan.model.globals[0].tensor.storage_type = 1;
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  plan.model.globals[0].tensor.storage_type = 0;
  plan.model.globals[0].tensor.rank = 1;
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  plan.model.globals[0].tensor.rank = 2;
  plan.model.globals[0].tensor.normalized_extents[0] = 6;
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  plan.model.globals[0].tensor.normalized_extents[0] = 7;
  write(0x7F800000);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
  std::error_code ec;
  std::filesystem::resize_file(path, 8, ec);
  ASSERT_FALSE(ec);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("truncated"), std::string::npos);
  std::filesystem::remove(path, ec);
  EXPECT_FALSE(ec);
}

TEST_F(ElementwiseInputsTest, BiasUsesActualWeightsAndRestoresProjection) {
  const char *temporary = std::getenv("TEST_TMPDIR");
  ASSERT_NE(temporary, nullptr);
  const auto path = std::filesystem::path(temporary) / "elementwise-bias.gguf";
  plan.source_asset = path.string();
  model::TensorBinding weight;
  weight.tensor.name = "bias";
  weight.tensor.storage_type = 0;
  weight.tensor.rank = 1;
  weight.tensor.normalized_extents = {256, 1, 1, 1};
  weight.tensor.byte_size = 1024;
  weight.tensor.absolute_offset = 8;
  plan.model.globals.push_back(weight);
  plan.programs.resize(1);
  plan.programs[0].parameters.push_back({"bias", 7, 1000, 1024, 4});
  const auto write = [&](uint32_t bits) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write("metadata", 8);
    for (unsigned i = 0; i < 256; ++i)
      for (unsigned lane = 0; lane < 4; ++lane)
        file.put(static_cast<char>(bits >> (lane * 8)));
  };
  write(0x3E800000); // Exactly 0.25 F32.
  const auto invocation = Invoke(
      "llm.decoder.add_bias_active",
      {Index(256), Buffer(0, 0, 12),
       Buffer(7, 1000, 1024, LOOMC_CMD_BUFFER_ROLE_FIXED), Buffer(1, 64, 8192)},
      1, 8);
  PreparedCase prepared;
  Ready(invocation, &prepared);
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "bias");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 8);
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 1024);
  EXPECT_EQ(prepared.representative_weight_spans.size(), 1);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 3 * 256 * 4);
  EXPECT_EQ(ReadWord(prepared, invocation.arguments[2].value.buffer),
            0x3E800000);
  write(0x7F800000);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
  std::error_code ec;
  std::filesystem::resize_file(path, 8, ec);
  ASSERT_FALSE(ec);
  EXPECT_EQ(
      PrepareElementwiseCase(plan, invocation, 1 << 20, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("truncated"), std::string::npos);
  std::filesystem::remove(path, ec);
  EXPECT_FALSE(ec);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
