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
using Ref = loomc_cmd_buffer_ref_t;
uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}
uint32_t Word(const PreparedRoot &root, uint64_t absolute) {
  uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i)
    value |= uint32_t(root.bytes[absolute - root.original_byte_offset + i])
             << (8 * i);
  return value;
}
class QkInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *tmp = std::getenv("TEST_TMPDIR");
    ASSERT_TRUE(tmp && *tmp);
    source = std::filesystem::path(tmp) / "qk-input-weights.bin";
    std::ofstream file(source, std::ios::binary);
    file.write("0123456789abcdef", 16);
    plan.source_asset = source.string();
    plan.programs.resize(1);
    plan.model.layers.resize(1);
    plan.model.layers[0].query_heads = 2;
    plan.model.layers[0].kv_heads = 1;
    uint64_t source_offset = 16;
    const auto weight = [&](const std::string &key, uint64_t elements,
                            uint64_t native_offset, float value) {
      model::TensorRecord tensor;
      tensor.name = key;
      tensor.rank = 1;
      tensor.storage_type = 0;
      tensor.normalized_extents = {static_cast<int64_t>(elements), 1, 1, 1};
      tensor.absolute_offset = source_offset;
      tensor.byte_size = elements * 4;
      plan.model.layers[0].tensors.push_back({key, tensor});
      plan.programs[0].parameters.push_back(
          {key, 0, native_offset, elements * 4, 4});
      for (uint64_t i = 0; i < elements; ++i) {
        const auto bits = Bits(value);
        for (unsigned j = 0; j < 4; ++j)
          file.put(static_cast<char>(bits >> (8 * j)));
      }
      source_offset += elements * 4;
    };
    weight("gamma128", 128, 64, 0.75f);
    weight("gamma256", 256, 1024, 0.5f);
    weight("bias256", 256, 4096, -0.25f);
    weight("factors", 64, 8192, 2.0f);
    file.close();
    ASSERT_TRUE(file);
    plan.workload.token_capacity = 8;
    plan.workload.active_tokens = 3;
    plan.workload.history_tokens = 17;
    plan.workload.context_capacity = 128;
  }
  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove(source, ignored);
  }
  Invocation InvocationFor(const std::string &name) {
    Invocation inv;
    inv.kernel_symbol = name;
    inv.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    inv.command.payload.dispatch.workgroup_count[0] = 2;
    inv.command.payload.dispatch.workgroup_count[1] = 8;
    inv.command.payload.dispatch.workgroup_count[2] = 1;
    const auto scalar = [&](loomc_cmd_argument_kind_t kind, uint64_t value) {
      loomc_cmd_argument_t arg{};
      arg.kind = kind;
      arg.value.scalar_bits = value;
      inv.arguments.push_back(arg);
    };
    const bool gemma = name == "gemma3_prepare_qk",
               llama = name == "llama32_rotary_query";
    const bool bias = name == "llm.decoder.prepare_qk_bias_h128",
               raw = name == "llm.attention.prepare_qk_raw_h256",
               factored = name == "llm.attention.prepare_qk_factored_h128";
    if (gemma) {
      plan.model.key_head_width = 256;
      scalar(LOOMC_CMD_ARGUMENT_KIND_INDEX, 0);
      scalar(LOOMC_CMD_ARGUMENT_KIND_OFFSET, 12);
    } else if (llama) {
      plan.model.key_head_width = 128;
      scalar(LOOMC_CMD_ARGUMENT_KIND_INDEX, 0);
      scalar(LOOMC_CMD_ARGUMENT_KIND_OFFSET, 12);
      scalar(LOOMC_CMD_ARGUMENT_KIND_INDEX, 2);
    } else {
      scalar(LOOMC_CMD_ARGUMENT_KIND_INDEX, 128);
      if (raw)
        scalar(LOOMC_CMD_ARGUMENT_KIND_INDEX, 128);
      else if (!bias)
        scalar(LOOMC_CMD_ARGUMENT_KIND_B32, Bits(1.0e-6f));
      scalar(LOOMC_CMD_ARGUMENT_KIND_B32, Bits(-0.01f));
      if (factored)
        scalar(LOOMC_CMD_ARGUMENT_KIND_B8, 1);
      scalar(LOOMC_CMD_ARGUMENT_KIND_OFFSET, 12);
    }
    const size_t begin = inv.arguments.size();
    const size_t buffers = factored ? 5 : raw ? 3 : 4;
    for (size_t i = 0; i < buffers; ++i) {
      loomc_cmd_argument_t arg{};
      arg.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
      arg.value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE,
                          static_cast<uint32_t>(i + 1), 64, UINT64_MAX};
      inv.arguments.push_back(arg);
    }
    if (!raw && !llama)
      inv.arguments[begin + 2].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 0,
                                               gemma  ? 1024u
                                               : bias ? 4096u
                                                      : 64u,
                                               gemma || bias ? 1024u : 512u};
    if (factored || llama)
      inv.arguments[llama ? 4 : begin + 3].value.buffer = {
          LOOMC_CMD_BUFFER_ROLE_FIXED, 0, 8192, 256};
    inv.command.argument_count = inv.arguments.size();
    return inv;
  }
  Plan plan;
  std::filesystem::path source;
};
const char *const kNames[] = {"llm.decoder.prepare_qk_h128",
                              "llm.decoder.prepare_qk_bias_h128",
                              "llm.attention.prepare_qk_factored_h128",
                              "llm.attention.prepare_qk_raw_h256",
                              "gemma3_prepare_qk",
                              "llama32_rotary_query"};
TEST_F(QkInputsTest, AllSixContractsUseActualWeightsAndShiftedControls) {
  for (const auto *name : kNames) {
    SCOPED_TRACE(name);
    auto inv = InvocationFor(name);
    PreparedCase prepared;
    std::string error;
    ASSERT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
              PreparationStatus::kReady)
        << error;
    EXPECT_EQ(prepared.arithmetic_check, "unchecked");
    EXPECT_TRUE(prepared.resets.empty());
    const bool llama = inv.kernel_symbol == "llama32_rotary_query";
    EXPECT_EQ(prepared.source_workloads, llama ? std::vector<uint64_t>({8})
                                               : std::vector<uint64_t>({8, 2}));
    const size_t epoch_arg =
        llama                                      ? 3
        : inv.kernel_symbol == "gemma3_prepare_qk" ? 2
        : inv.kernel_symbol == "llm.attention.prepare_qk_factored_h128"
            ? 5
            : inv.arguments.size() - 4 +
                  (inv.kernel_symbol == "llm.attention.prepare_qk_raw_h256"
                       ? 1
                       : 0);
    const auto ref = inv.arguments[epoch_arg].value.buffer;
    const auto *epoch = FindInputRoot(prepared.roots, ref);
    ASSERT_TRUE(epoch);
    EXPECT_EQ(Word(*epoch, ref.byte_offset + 12), 3);
    EXPECT_EQ(Word(*epoch, ref.byte_offset + 16), 17);
    const auto out_ref = inv.arguments.back().value.buffer;
    const auto *output = FindInputRoot(prepared.roots, out_ref);
    ASSERT_TRUE(output);
    EXPECT_EQ(output->bytes.front(),
              0xA5); // untouched/inactive rows have deterministic bytes
    if (inv.kernel_symbol == "llm.attention.prepare_qk_raw_h256")
      EXPECT_TRUE(prepared.weight_sources.empty());
    else
      EXPECT_FALSE(prepared.weight_sources.empty());
    for (const auto &weight : prepared.weight_sources)
      EXPECT_GE(weight.source_offset, 16);
  }
}
TEST_F(QkInputsTest, FalseFactorsDoNotNeedParameterOrFactorReads) {
  auto inv = InvocationFor("llm.attention.prepare_qk_factored_h128");
  inv.arguments[3].value.scalar_bits = 0;
  inv.arguments[8].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 30, 0, 4};
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "gamma128");
  EXPECT_EQ(prepared.source_argument_types[3], "i1");
  // An unread factor may alias an initialized packet; zero semantic length
  // must not impose an input-alias restriction or change the packet bytes.
  inv.arguments[8].value.buffer = inv.arguments[5].value.buffer;
  ASSERT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  const auto epoch = inv.arguments[5].value.buffer;
  const auto *root = FindInputRoot(prepared.roots, epoch);
  ASSERT_TRUE(root);
  EXPECT_EQ(Word(*root, epoch.byte_offset + 12), 3);
  EXPECT_EQ(Word(*root, epoch.byte_offset + 16), 17);
}
TEST_F(QkInputsTest, FreshAndDecodeControlsPreserveTokenBegin) {
  auto inv = InvocationFor("llama32_rotary_query");
  inv.arguments[0].value.scalar_bits = 1;
  plan.workload.active_tokens = 1;
  plan.workload.history_tokens = 0;
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  const auto ref = inv.arguments[3].value.buffer;
  const auto *epoch = FindInputRoot(prepared.roots, ref);
  ASSERT_TRUE(epoch);
  EXPECT_EQ(Word(*epoch, ref.byte_offset + 12), 0);
  EXPECT_EQ(inv.arguments[2].value.scalar_bits, 2);
  plan.workload.active_tokens = 2;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  plan.workload.active_tokens = 1;
  inv.arguments[2].value.scalar_bits = 512;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
}
TEST_F(QkInputsTest, RejectsExactAbiAndScalarViolations) {
  PreparedCase prepared;
  std::string error;
  auto inv = InvocationFor("llm.attention.prepare_qk_factored_h128");
  inv.arguments[3].value.scalar_bits = 2;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  inv.arguments[3].value.scalar_bits = 1;
  inv.arguments[1].value.scalar_bits = uint64_t{1} << 32;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  inv = InvocationFor("llm.attention.prepare_qk_raw_h256");
  inv.arguments[1].value.scalar_bits = 3;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  inv = InvocationFor("llm.decoder.prepare_qk_h128");
  --inv.command.argument_count;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
}
TEST_F(QkInputsTest, BudgetAndUnknownFamilyPreserveOutputAndDiagnostic) {
  auto inv = InvocationFor("llm.decoder.prepare_qk_h128");
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  inv.kernel_symbol = "unclaimed";
  error = "prior";
  EXPECT_EQ(PrepareQkCase(plan, inv, 1, nullptr, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "prior");
}
TEST_F(QkInputsTest, AliasesAndTruncatedViewsAreExplicitlyRejected) {
  auto inv = InvocationFor("llm.decoder.prepare_qk_h128");
  PreparedCase prepared;
  std::string error;
  inv.arguments.back().value.buffer = inv.arguments[5].value.buffer;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kUnsupported);
  inv = InvocationFor("llm.decoder.prepare_qk_h128");
  inv.arguments[5].value.buffer.byte_length = 4;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1, &prepared, &error),
            PreparationStatus::kFailure);
}
TEST_F(QkInputsTest, DistinctInputAliasesCannotOverwriteTheControlPacket) {
  auto inv = InvocationFor("llm.decoder.prepare_qk_h128");
  inv.arguments[5].value.buffer = inv.arguments[4].value.buffer;
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  EXPECT_NE(error.find("semantic argument spans"), std::string::npos);
}
TEST_F(QkInputsTest, InvalidFactorMetadataIsNotMaskedByTheBudget) {
  auto inv = InvocationFor("llama32_rotary_query");
  auto &bindings = plan.model.layers[0].tensors;
  const auto factors =
      std::find_if(bindings.begin(), bindings.end(), [](const auto &binding) {
        return binding.role == "factors";
      });
  ASSERT_NE(factors, bindings.end());
  factors->tensor.rank = 2;
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
  EXPECT_NE(error.find("rank-one F32"), std::string::npos);
}
TEST_F(QkInputsTest, RejectsNonfiniteRealWeightsAndInvalidDivisors) {
  auto inv = InvocationFor("llama32_rotary_query");
  PreparedCase prepared;
  std::string error;
  const auto *tensor = FindBenchmarkTensor(plan, "factors");
  ASSERT_TRUE(tensor);
  {
    std::fstream file(source, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(tensor->absolute_offset);
    const auto bits = Bits(0.5f);
    for (unsigned i = 0; i < 4; ++i)
      file.put(static_cast<char>(bits >> (8 * i)));
  }
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
  {
    std::fstream file(source, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(tensor->absolute_offset);
    const uint32_t bits = 0x7f800000;
    for (unsigned i = 0; i < 4; ++i)
      file.put(static_cast<char>(bits >> (8 * i)));
  }
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kFailure);
}
TEST_F(QkInputsTest, UnitFactorsRequireSelectedProducerAndAreNotWeights) {
  auto inv = InvocationFor("llama32_rotary_query");
  inv.command_ordinal = 1;
  inv.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 30, 32,
                                   256};
  PreparedCase prepared;
  std::string error;
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kUnsupported);
  Invocation producer;
  producer.kernel_symbol = "llama32_unit_rope_factors";
  producer.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
  producer.command.argument_count = 1;
  for (auto &count : producer.command.payload.dispatch.workgroup_count)
    count = 1;
  producer.arguments = {inv.arguments[4]};
  plan.invocations = {producer, inv};
  plan.programs[0].commands = {producer.command, inv.command};
  ASSERT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_TRUE(prepared.weight_sources.empty());
  EXPECT_TRUE(prepared.representative_weight_spans.empty());
  const auto ref = inv.arguments[4].value.buffer;
  const auto *root = FindInputRoot(prepared.roots, ref);
  ASSERT_TRUE(root);
  for (unsigned i = 0; i < 64; ++i)
    EXPECT_EQ(Word(*root, ref.byte_offset + i * 4), Bits(1.0f));
  plan.invocations[0].kernel_symbol = "unproven.writer";
  EXPECT_EQ(PrepareQkCase(plan, inv, 1024 * 1024, &prepared, &error),
            PreparationStatus::kUnsupported);
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
