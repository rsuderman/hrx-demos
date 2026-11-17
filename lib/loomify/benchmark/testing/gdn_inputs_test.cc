// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/stateful_inputs.h"

#include "lib/loomify/benchmark/qwen_span_inputs.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace hrx::loomify_gguf::benchmark {
namespace {
uint64_t Word(const std::vector<uint8_t> &bytes, size_t offset,
              unsigned width) {
  uint64_t result = 0;
  for (unsigned i = 0; i < width; ++i)
    result |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
  return result;
}
float Float(const std::vector<uint8_t> &bytes, size_t offset) {
  const uint32_t bits = static_cast<uint32_t>(Word(bytes, offset, 4));
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
TEST(QwenSpanInputs, ExactContinuedSingleSpanAndNonidentityPageMap) {
  Workload workload;
  workload.token_capacity = 8;
  workload.active_tokens = 3;
  workload.history_tokens = 62;
  workload.context_capacity = 253;
  QwenSpanControls prepared;
  std::string error;
  ASSERT_TRUE(BuildQwenSpanControls(workload, true, 8192, &prepared, &error))
      << error;
  ASSERT_EQ(prepared.epoch.size(), 1548);
  EXPECT_EQ(Word(prepared.epoch, 0, 4), 3);
  EXPECT_EQ(Word(prepared.epoch, 4, 4), 1);
  EXPECT_EQ(Word(prepared.epoch, 8, 4), 0);
  EXPECT_EQ(Word(prepared.epoch, 12, 4), 3);
  EXPECT_EQ(Word(prepared.epoch, 16, 4), 62);
  EXPECT_EQ(Word(prepared.epoch, 20, 4), 0);
  EXPECT_EQ(Word(prepared.epoch, 24, 4), 0);
  EXPECT_EQ(Word(prepared.epoch, 28, 4), UINT32_MAX);
  ASSERT_EQ(prepared.rows.size(), 2048 + 16);
  for (size_t offset : {0, 8, 16})
    EXPECT_EQ(Word(prepared.rows, offset, 8), 0);
  EXPECT_EQ(Word(prepared.rows, 24, 8), 62);
  EXPECT_EQ(prepared.page_map_origin, 2048);
  EXPECT_EQ(prepared.physical_capacity, 256);
  EXPECT_EQ(Word(prepared.rows, 2048, 4), 3);
  EXPECT_EQ(Word(prepared.rows, 2052, 4), 2);
  EXPECT_EQ(Word(prepared.rows, 2056, 4), 1);
  EXPECT_EQ(Word(prepared.rows, 2060, 4), 0);
}
TEST(QwenSpanInputs, RejectsBadWorkloadAndBudgetAtomically) {
  Workload workload;
  workload.token_capacity = 8;
  workload.active_tokens = 5;
  workload.history_tokens = 17;
  QwenSpanControls prepared;
  std::string error;
  ASSERT_TRUE(BuildQwenSpanControls(workload, false, 3596, &prepared, &error));
  EXPECT_EQ(prepared.rows.size(), 2048);
  EXPECT_EQ(prepared.page_map_origin, 0);
  EXPECT_EQ(prepared.physical_capacity, 0);
  const auto saved = prepared.epoch;
  EXPECT_FALSE(BuildQwenSpanControls(workload, false, 3595, &prepared, &error));
  EXPECT_EQ(prepared.epoch, saved);
  workload.history_tokens = workload.context_capacity - 4;
  EXPECT_FALSE(BuildQwenSpanControls(workload, false, 8192, &prepared, &error));
  workload.history_tokens = 17;
  workload.active_tokens = 0;
  EXPECT_FALSE(BuildQwenSpanControls(workload, false, 8192, &prepared, &error));
}

class GdnInputsTest : public ::testing::Test {
protected:
  static constexpr uint64_t kQkv = 6144;
  static constexpr uint64_t kHeads = 16;
  static constexpr uint64_t kState = 128 * 2048;
  void SetUp() override {
    const char *tmp = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(tmp && *tmp);
    source = std::filesystem::path(tmp) / "gdn-input-weights.bin";
    std::ofstream file(source, std::ios::binary);
    const auto write = [&](float value) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      for (unsigned i = 0; i < 4; ++i)
        file.put(static_cast<char>(bits >> (8 * i)));
    };
    for (unsigned i = 0; i < 8; ++i)
      write(0);
    for (uint64_t i = 0; i < 4 * kQkv; ++i)
      write(-0.25f + static_cast<float>(i % 7) / 64);
    for (uint64_t i = 0; i < kHeads; ++i)
      write(-0.5f - static_cast<float>(i) / 64);
    for (uint64_t i = 0; i < kHeads; ++i)
      write(static_cast<float>(i) / 64);
    file.close();
    ASSERT_TRUE(file);
    plan.source_asset = source.string();
    plan.model.rms_epsilon = 1.0e-6;
    plan.model.ssm =
        model::SsmDescriptor{4, 2048, 128, 16, 16, 3 * kQkv, kState};
    plan.workload.token_capacity = 8;
    plan.workload.active_tokens = 3;
    plan.workload.history_tokens = 17;
    plan.workload.context_capacity = 128;
    model::LayerDescriptor layer;
    layer.kind = model::LayerKind::kGatedDeltaNet;
    model::TensorRecord conv;
    conv.name = "blk.0.ssm_conv1d.weight";
    conv.rank = 2;
    conv.normalized_extents = {4, kQkv, 1, 1};
    conv.byte_size = 4 * kQkv * 4;
    conv.absolute_offset = 32;
    auto a = conv;
    a.name = "blk.0.ssm_a";
    a.rank = 1;
    a.normalized_extents = {kHeads, 1, 1, 1};
    a.byte_size = kHeads * 4;
    a.absolute_offset += conv.byte_size;
    auto bias = a;
    bias.name = "blk.0.ssm_dt.bias";
    bias.absolute_offset += a.byte_size;
    layer.tensors = {{"convolution", conv}, {"a", a}, {"bias", bias}};
    plan.model.layers.push_back(layer);
    plan.programs.resize(1);
    plan.programs[0].parameters = {{conv.name, 0, 1024, conv.byte_size, 4},
                                   {a.name, 0, 64, a.byte_size, 4},
                                   {bias.name, 0, 256, bias.byte_size, 4}};
    convolution = InvocationFor(true);
    recurrent = InvocationFor(false);
  }
  Invocation InvocationFor(bool conv) {
    Invocation invocation;
    invocation.kernel_symbol =
        conv ? "qwen38_gdn_conv_spans" : "qwen38_gdn_recurrent_spans";
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.payload.dispatch.workgroup_count[0] = 16;
    invocation.command.payload.dispatch.workgroup_count[1] = conv ? 1 : 32;
    invocation.command.payload.dispatch.workgroup_count[2] = 1;
    invocation.arguments.resize(conv ? 9 : 11);
    invocation.command.argument_count = invocation.arguments.size();
    for (size_t i = 0; i < invocation.arguments.size(); ++i) {
      invocation.arguments[i].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
      invocation.arguments[i].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE,
                                              static_cast<uint32_t>(i + 1), 64,
                                              UINT64_MAX};
    }
    invocation.arguments[2].kind = LOOMC_CMD_ARGUMENT_KIND_OFFSET;
    invocation.arguments[2].value.scalar_bits = 4096;
    if (conv) {
      invocation.arguments[3].kind = LOOMC_CMD_ARGUMENT_KIND_B32;
      const float epsilon = 1.0e-6f;
      uint32_t bits;
      std::memcpy(&bits, &epsilon, sizeof(bits));
      invocation.arguments[3].value.scalar_bits = bits;
      invocation.arguments[5].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 0,
                                              1024, 4 * kQkv * 4};
    } else {
      invocation.arguments[7].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 0,
                                              64, 64};
      invocation.arguments[8].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 0,
                                              256, 64};
    }
    return invocation;
  }
  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove(source, ignored);
  }
  Plan plan;
  Invocation convolution, recurrent;
  std::filesystem::path source;
};
TEST_F(GdnInputsTest, ConvUsesActualTapsContinuedStateAndPartialSpan) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(
      PrepareGdnCase(plan, convolution, 8 * 1024 * 1024, &prepared, &error),
      PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{1}));
  EXPECT_EQ(prepared.source_argument_types[2], "offset");
  EXPECT_EQ(prepared.source_argument_types[3], "f32");
  EXPECT_EQ(prepared.measurement_subject, "reset_plus_kernel");
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_offset, 64 + 4096);
  EXPECT_EQ(prepared.resets[0].target.byte_length, 3 * kQkv * 4);
  EXPECT_TRUE(std::any_of(prepared.resets[0].seed.bytes.begin(),
                          prepared.resets[0].seed.bytes.end(),
                          [](uint8_t value) { return value != 0; }));
  const auto &weight_ref = convolution.arguments[5].value.buffer;
  const auto *weight = FindInputRoot(prepared.roots, weight_ref);
  ASSERT_TRUE(weight);
  EXPECT_FLOAT_EQ(Float(weight->bytes,
                        weight_ref.byte_offset - weight->original_byte_offset),
                  -0.25f);
  const auto &epoch_ref = convolution.arguments[0].value.buffer;
  const auto *epoch = FindInputRoot(prepared.roots, epoch_ref);
  ASSERT_TRUE(epoch);
  const auto begin = epoch_ref.byte_offset - epoch->original_byte_offset;
  EXPECT_EQ(Word(epoch->bytes, begin + 12, 4), 3);
  EXPECT_EQ(Word(epoch->bytes, begin + 16, 4), 17);
}
TEST_F(GdnInputsTest,
       RecurrentUsesRealCoefficientsAndMatchingSyntheticQkNorms) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  ASSERT_EQ(prepared.representative_weight_spans.size(), 2);
  ASSERT_EQ(prepared.weight_sources.size(), 2);
  EXPECT_EQ(prepared.weight_sources[0].key, "blk.0.ssm_a");
  EXPECT_EQ(prepared.weight_sources[1].key, "blk.0.ssm_dt.bias");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 32 + 4 * kQkv * 4);
  EXPECT_EQ(prepared.weight_sources[1].source_offset, 32 + 4 * kQkv * 4 + 64);
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_EQ(prepared.resets[0].target.byte_length, kState * 4);
  const auto &a_ref = recurrent.arguments[7].value.buffer;
  const auto *a = FindInputRoot(prepared.roots, a_ref);
  ASSERT_TRUE(a);
  EXPECT_FLOAT_EQ(Float(a->bytes, a_ref.byte_offset - a->original_byte_offset),
                  -0.5f);
  const auto &qkv_ref = recurrent.arguments[3].value.buffer;
  const auto &norm_ref = recurrent.arguments[4].value.buffer;
  const auto *qkv = FindInputRoot(prepared.roots, qkv_ref);
  const auto *norm = FindInputRoot(prepared.roots, norm_ref);
  ASSERT_TRUE(qkv && norm);
  double squares = 0;
  const auto qkv_begin = qkv_ref.byte_offset - qkv->original_byte_offset;
  for (size_t i = 0; i < 128; ++i) {
    const auto value = Float(qkv->bytes, qkv_begin + i * 4);
    squares += static_cast<double>(value) * value;
  }
  const auto inverse =
      Float(norm->bytes, norm_ref.byte_offset - norm->original_byte_offset);
  EXPECT_GT(inverse, 0);
  EXPECT_NEAR(squares * inverse * inverse, 1.0, 1.0e-6);
  plan.model.rms_epsilon = 8;
  ASSERT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  norm = FindInputRoot(prepared.roots, norm_ref);
  ASSERT_TRUE(norm);
  EXPECT_FLOAT_EQ(
      Float(norm->bytes, norm_ref.byte_offset - norm->original_byte_offset),
      0.125f);
}
TEST_F(GdnInputsTest, FreshStateIsZeroWithoutReplacingContinuedCoverage) {
  plan.workload.history_tokens = 0;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  ASSERT_EQ(prepared.resets.size(), 1);
  EXPECT_TRUE(std::all_of(prepared.resets[0].seed.bytes.begin(),
                          prepared.resets[0].seed.bytes.end(),
                          [](uint8_t value) { return value == 0; }));
  const auto *qkv =
      FindInputRoot(prepared.roots, recurrent.arguments[3].value.buffer);
  ASSERT_TRUE(qkv);
  EXPECT_TRUE(std::any_of(qkv->bytes.begin(), qkv->bytes.end(),
                          [](uint8_t value) { return value != 0; }));
}
TEST_F(GdnInputsTest, BudgetIncludesIndependentResetSeedStorage) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  uint64_t root_bytes = 0;
  for (const auto &root : prepared.roots)
    root_bytes += root.bytes.size();
  ASSERT_EQ(prepared.resets.size(), 1);
  const auto seed_bytes = prepared.resets[0].seed.bytes.size();
  ASSERT_GT(seed_bytes, 0);
  const auto saved_recipe = prepared.setup_recipe_id;
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, root_bytes, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, saved_recipe);
  EXPECT_NE(error.find("budget"), std::string::npos);
  ASSERT_EQ(PrepareGdnCase(plan, recurrent, root_bytes + seed_bytes, &prepared,
                           &error),
            PreparationStatus::kReady)
      << error;
}
TEST_F(GdnInputsTest, RejectsBadAbiGeometryAliasesAndBudget) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 1024, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  --recurrent.command.argument_count;
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  ++recurrent.command.argument_count;
  recurrent.arguments[2].kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  recurrent.arguments[2].kind = LOOMC_CMD_ARGUMENT_KIND_OFFSET;
  recurrent.arguments[10].value.buffer = recurrent.arguments[3].value.buffer;
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  plan.model.ssm->convolution_history_elements = 1;
  EXPECT_EQ(
      PrepareGdnCase(plan, convolution, 8 * 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
}
TEST_F(GdnInputsTest, UnknownFamilyPreservesPriorDiagnosticAndOutput) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error = "prior provider diagnostic";
  recurrent.kernel_symbol = "unclaimed.kernel";
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 1024, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "prior provider diagnostic");
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 1024, nullptr, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "prior provider diagnostic");
}
TEST_F(GdnInputsTest, InvalidMetadataIsNotMaskedBySmallBudget) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  auto invalid = recurrent;
  invalid.arguments[3].value.buffer.byte_length = 4;
  EXPECT_EQ(PrepareGdnCase(plan, invalid, 1, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("native view"), std::string::npos);
  auto &bindings = plan.model.layers[0].tensors;
  const auto a =
      std::find_if(bindings.begin(), bindings.end(),
                   [](const auto &binding) { return binding.role == "a"; });
  ASSERT_NE(a, bindings.end());
  a->tensor.rank = 2;
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 1, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("tensor shape"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}
TEST_F(GdnInputsTest, RejectsTruncatedAndNonfiniteWeightsWithoutFallback) {
  PreparedCase prepared;
  std::string error;
  {
    std::fstream file(source, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(32 + 4 * kQkv * 4);
    const char infinity[] = {0, 0, static_cast<char>(0x80), 0x7f};
    file.write(infinity, sizeof(infinity));
  }
  EXPECT_EQ(PrepareGdnCase(plan, recurrent, 8 * 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
  std::filesystem::resize_file(source, 64);
  EXPECT_EQ(
      PrepareGdnCase(plan, convolution, 8 * 1024 * 1024, &prepared, &error),
      PreparationStatus::kFailure);
  EXPECT_NE(error.find("truncated"), std::string::npos);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
