// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
class BasicInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary, nullptr);
    path = std::filesystem::path(temporary) /
           ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directories(path));
    plan.source_asset = (path / "weights.gguf").string();
    plan.model.embedding_width = 4;
    plan.model.vocabulary_size = 33;
    plan.model.rms_epsilon = 0.00001f;
    plan.workload.active_tokens = 2;
    plan.workload.history_tokens = 7;
    plan.workload.seed = 19;
    plan.programs.resize(1);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  static loomc_cmd_argument_t Index(uint64_t value) {
    loomc_cmd_argument_t argument{};
    argument.kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
    argument.value.scalar_bits = value;
    return argument;
  }
  static loomc_cmd_argument_t Offset(uint64_t value) {
    auto argument = Index(value);
    argument.kind = LOOMC_CMD_ARGUMENT_KIND_OFFSET;
    return argument;
  }
  static loomc_cmd_argument_t Epsilon(uint32_t bits = 0x3727c5ac) {
    auto argument = Index(bits);
    argument.kind = LOOMC_CMD_ARGUMENT_KIND_B32;
    return argument;
  }
  static loomc_cmd_argument_t Buffer(unsigned root, uint64_t length,
                                     bool fixed = false, uint64_t origin = 0) {
    loomc_cmd_argument_t argument{};
    argument.kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    argument.value.buffer = {fixed ? LOOMC_CMD_BUFFER_ROLE_FIXED
                                   : LOOMC_CMD_BUFFER_ROLE_REBINDABLE,
                             root, origin, length};
    return argument;
  }
  void Call(std::string name, std::vector<loomc_cmd_argument_t> arguments,
            unsigned x = 1, unsigned y = 1) {
    invocation = {};
    invocation.kernel_symbol = std::move(name);
    invocation.arguments = std::move(arguments);
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = invocation.arguments.size();
    invocation.command.payload.dispatch.workgroup_count[0] = x;
    invocation.command.payload.dispatch.workgroup_count[1] = y;
    invocation.command.payload.dispatch.workgroup_count[2] = 1;
  }
  void Tensor(uint64_t width, uint64_t rows, uint32_t tag,
              const std::vector<uint8_t> &bytes) {
    model::TensorBinding binding;
    binding.tensor.name = "test.weight";
    binding.tensor.storage_type = tag;
    binding.tensor.rank = rows ? 2 : 1;
    binding.tensor.normalized_extents = {static_cast<int64_t>(width),
                                         static_cast<int64_t>(rows ? rows : 1),
                                         1, 1};
    binding.tensor.byte_size = bytes.size();
    binding.tensor.absolute_offset = 8;
    plan.model.globals = {binding};
    plan.programs[0].parameters = {
        {binding.tensor.name, 7, 16, bytes.size(), 4}};
    std::ofstream stream(plan.source_asset, std::ios::binary | std::ios::trunc);
    stream.write("metadata", 8);
    stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  }
  void Gamma(uint64_t width = 4) {
    std::vector<uint8_t> bytes(width * 4);
    for (uint64_t i = 0; i < width; ++i) {
      bytes[i * 4 + 2] = 0x80;
      bytes[i * 4 + 3] = 0x3f;
    }
    Tensor(width, 0, 0, bytes);
  }
  uint32_t Word(const PreparedCase &prepared, size_t argument,
                uint64_t offset = 0) {
    const auto binding = invocation.arguments[argument].value.buffer;
    const auto *root = FindInputRoot(prepared.roots, binding);
    if (!root) {
      ADD_FAILURE() << "missing prepared input root";
      return UINT32_MAX;
    }
    const uint64_t begin =
        binding.byte_offset - root->original_byte_offset + offset;
    if (begin + 4 > root->bytes.size()) {
      ADD_FAILURE() << "missing prepared input word";
      return UINT32_MAX;
    }
    uint32_t value = 0;
    for (unsigned lane = 0; lane < 4; ++lane)
      value |= uint32_t(root->bytes[begin + lane]) << (lane * 8);
    return value;
  }
  PreparationStatus Prepare(uint64_t budget = 1 << 20) {
    return PrepareBasicCase(plan, invocation, budget, &prepared, &error);
  }
  std::filesystem::path path;
  Plan plan;
  Invocation invocation;
  PreparedCase prepared;
  std::string error;
};

TEST_F(BasicInputsTest, ActiveRmsUsesRuntimeWidthAndRealGamma) {
  Gamma();
  Call("llm.decoder.rmsnorm_active",
       {Index(8), Index(4), Epsilon(), Buffer(0, 12), Buffer(1, 128),
        Buffer(7, 16, true, 16), Buffer(2, 128)},
       1, 3);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3, 4}));
  EXPECT_EQ(prepared.buffer_extents[1].byte_length, 128);
  EXPECT_EQ(Word(prepared, 3), 2);
  EXPECT_EQ(Word(prepared, 5), 0x3f800000u);
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 8);
}

TEST_F(BasicInputsTest, LastNormRetainsBeginControlOriginAndCompactOutput) {
  Gamma();
  Call("llm.decoder.output_norm_last",
       {Index(8), Index(4), Index(3), Offset(20), Epsilon(), Buffer(0, 32),
        Buffer(1, 128), Buffer(7, 16, true, 16), Buffer(2, 16)});
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{8, 4}));
  EXPECT_EQ(prepared.buffer_extents.back().byte_length, 16);
  EXPECT_EQ(Word(prepared, 5, 20), 2);
  EXPECT_EQ(invocation.arguments[2].value.scalar_bits, 3);
  invocation.arguments[2].value.scalar_bits = 7;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
}

TEST_F(BasicInputsTest, LegacyNormsUseOneParameterizedRecipe) {
  Gamma();
  for (const char *name :
       {"llama32_rmsnorm_active_f32", "gemma3_rmsnorm_active_f32"}) {
    Call(name,
         {Index(8), Index(1), Offset(4), Buffer(0, 16), Buffer(1, 128),
          Buffer(7, 16, true, 16), Buffer(2, 128)},
         1, 8);
    ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
    EXPECT_EQ(Word(prepared, 3, 4), 2);
    EXPECT_EQ(prepared.weight_sources.at(0).byte_length, 16);
  }
}

TEST_F(BasicInputsTest, InvalidEpsilonAbiBudgetAndGammaPreserveOutput) {
  Gamma();
  Call("llm.decoder.rmsnorm_active",
       {Index(8), Index(4), Epsilon(0x7fc00000), Buffer(0, 12), Buffer(1, 128),
        Buffer(7, 16, true, 16), Buffer(2, 128)},
       1, 3);
  prepared.setup_recipe_id = "untouched";
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
  invocation.arguments[2] = Epsilon();
  EXPECT_EQ(Prepare(1), PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
  invocation.arguments[0].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  invocation.arguments[0] = Index(8);
  plan.programs[0].parameters.clear();
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
}

TEST_F(BasicInputsTest, GdnNormGateUsesHeadGammaAndSeparateNonzeroGate) {
  Gamma(128);
  plan.model.ssm.emplace();
  plan.model.ssm->state_width = 128;
  plan.model.ssm->value_heads = 2;
  plan.model.ssm->inner_width = 256;
  Call("qwen38_gdn_norm_gate_prefill",
       {Index(4), Epsilon(), Buffer(0, 4096), Buffer(7, 512, true, 16),
        Buffer(1, 4096), Buffer(2, 4096)},
       2, 3);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.weight_sources.at(0).byte_length, 512);
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3}));
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 4096);
  EXPECT_NE(Word(prepared, 2), Word(prepared, 4));
  plan.model.ssm->state_width = 64;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
}

TEST_F(BasicInputsTest, ArgmaxReducerUsesValidTailTileWinnerIds) {
  plan.model.vocabulary_size = 4097;
  Call("qwen38_greedy_argmax_reduce_rows_f32",
       {Index(2), Index(4097), Buffer(0, 16), Buffer(1, 16), Buffer(2, 8)}, 2);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{2, 4097}));
  EXPECT_EQ(Word(prepared, 3), 17);
  EXPECT_EQ(Word(prepared, 3, 4), 4096);
  EXPECT_EQ(Word(prepared, 3, 12), 4096);
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  Call("qwen38_greedy_argmax_partial_rows_f32",
       {Index(2), Index(4097), Buffer(0, 32776), Buffer(1, 16), Buffer(2, 16)},
       2, 2);
  EXPECT_EQ(Prepare(), PreparationStatus::kReady) << error;
  invocation.command.payload.dispatch.workgroup_count[0] = 1;
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
}

TEST_F(BasicInputsTest, SnapshotInitializesActualPacketOrigin) {
  Call("llm.epoch.snapshot_count",
       {Index(0), Offset(20), Buffer(0, 32), Buffer(1, 12)});
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_TRUE(prepared.source_workloads.empty());
  EXPECT_EQ(Word(prepared, 2, 20), 2);
  EXPECT_EQ(Word(prepared, 2, 24), 0);
  invocation.arguments[1] = Offset(21);
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
}

TEST_F(BasicInputsTest, OutputSpansSelectsALiveNonzeroInputRow) {
  Gamma();
  Call("qwen38_output_normalize_spans",
       {Buffer(0, 1548), Epsilon(), Buffer(1, 8192), Buffer(7, 16, true, 16),
        Buffer(2, 48)},
       3);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3}));
  EXPECT_EQ(Word(prepared, 0, 8), 1);
  EXPECT_EQ(Word(prepared, 0, 1292), 1);
  EXPECT_EQ(prepared.buffer_extents.back().byte_length, 48);
}

TEST_F(BasicInputsTest, PublicationUsesSourceTerminalAndLiveNontrivialSlot) {
  plan.publication_tokens =
      PublicationTokenFacts{{{0, "stop"}, {9, "eos"}}, 1, "GGUF"};
  Call("qwen3_publish_epoch_id",
       {Buffer(0, 2064), Buffer(1, 16), Buffer(2, 64)});
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(Word(prepared, 0, 8), 1);
  EXPECT_EQ(Word(prepared, 0, 28), 3);
  EXPECT_EQ(Word(prepared, 0, 2060), 9);
  EXPECT_EQ(Word(prepared, 1), 1);
  plan.publication_tokens.reset();
  EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
  Call("qwen08_publish_selected_ids",
       {Buffer(0, 32), Buffer(1, 4), Buffer(2, 64)});
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{1}));
  EXPECT_EQ(Word(prepared, 1), 32);
}

TEST_F(BasicInputsTest, EmbeddingCopiesFullQuantizedTableAndValidHighTokenIds) {
  plan.model.embedding_width = 256;
  plan.model.vocabulary_size = 3;
  for (uint32_t tag : {12u, 13u, 14u}) {
    const uint64_t block_bytes = tag == 12 ? 144 : tag == 13 ? 176 : 210;
    std::vector<uint8_t> table(3 * block_bytes, 0x11);
    for (unsigned row = 0; row < 3; ++row) {
      const uint64_t scale = row * block_bytes + (tag == 14 ? 208 : 0);
      table[scale] = 0;
      table[scale + 1] = 0x3c;
      if (tag != 14) {
        table[scale + 2] = 0;
        table[scale + 3] = 0x38;
      }
    }
    Tensor(256, 3, tag, table);
    Call("llm.embedding.ordinary_f32_kernel",
         {Index(256), Index(3), Index(tag), Offset(20), Buffer(0, 40),
          Buffer(1, 12), Buffer(2, 8), Buffer(7, table.size(), true, 16),
          Buffer(3, 2048)},
         2);
    ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
    EXPECT_EQ(prepared.weight_sources.at(0).byte_length, table.size());
    const auto *weight =
        FindInputRoot(prepared.roots, invocation.arguments[7].value.buffer);
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->bytes, table);
    EXPECT_EQ(Word(prepared, 6), 1);
    EXPECT_EQ(Word(prepared, 6, 4), 2);
    EXPECT_EQ(Word(prepared, 4, 20), 2);
    EXPECT_EQ(Word(prepared, 4, 24), 7);
    EXPECT_EQ(Word(prepared, 5), 2);
    EXPECT_EQ(prepared.arithmetic_check, "unchecked");
    EXPECT_EQ(Prepare(table.size() - 1), PreparationStatus::kUnsupported);
    EXPECT_NE(error.find("budget"), std::string::npos);
    // The full table fits this budget, but its controls, IDs and output do not.
    EXPECT_EQ(Prepare(table.size()), PreparationStatus::kUnsupported);
    EXPECT_NE(error.find("budget"), std::string::npos);
    EXPECT_EQ(prepared.weight_sources.at(0).key, "test.weight");
    EXPECT_EQ(prepared.weight_sources.at(0).source_offset, 8);
    const auto declared_length =
        invocation.arguments[7].value.buffer.byte_length;
    invocation.arguments[7].value.buffer.byte_length = table.size() - 1;
    EXPECT_EQ(Prepare(1), PreparationStatus::kFailure);
    invocation.arguments[7].value.buffer.byte_length = declared_length;
    table[(tag == 14 ? 208 : 0) + 1] = 0x7c;
    Tensor(256, 3, tag, table);
    EXPECT_EQ(Prepare(), PreparationStatus::kFailure);
    EXPECT_NE(error.find("nonfinite"), std::string::npos);
  }
}

TEST_F(BasicInputsTest, EmbeddingAdaptersRetainTheirNativeIdAndControlDomains) {
  plan.model.embedding_width = 256;
  plan.model.vocabulary_size = 3;
  for (bool q6 : {false, true}) {
    const uint64_t block_bytes = q6 ? 210 : 144;
    Tensor(256, 3, q6 ? 14 : 12, std::vector<uint8_t>(3 * block_bytes));
    Call(q6 ? "qwen38_token_embedding_q6k_f16_prefill"
            : "qwen38_token_embedding_q4k_f32_prefill",
         {Buffer(0, 2048), Buffer(7, 3 * block_bytes, true, 16),
          Buffer(1, 2048)},
         2);
    ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
    EXPECT_EQ(prepared.buffer_extents[0].byte_length, 2048);
    EXPECT_EQ(Word(prepared, 0, 511 * 4), 2);
    EXPECT_EQ(prepared.buffer_extents.back().byte_length, 2048);
  }
  Tensor(256, 3, 12, std::vector<uint8_t>(432));
  Call("llm.token_embedding_epoch_f32",
       {Index(256), Index(3), Index(12), Buffer(0, 32), Buffer(1, 12),
        Buffer(2, 8), Buffer(7, 432, true, 16), Buffer(3, 2048)},
       2);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(Word(prepared, 3, 12), 2);
  EXPECT_EQ(Word(prepared, 3, 20), 0);
  plan.model.embedding_width = 32;
  Tensor(32, 3, 8, std::vector<uint8_t>(102));
  Call("gemma3_token_embedding_epoch_q8_f32",
       {Buffer(0, 32), Buffer(1, 12), Buffer(2, 8), Buffer(7, 102, true, 16),
        Buffer(3, 256)},
       2);
  ASSERT_EQ(Prepare(), PreparationStatus::kReady) << error;
  EXPECT_EQ(Word(prepared, 0, 12), 2);
  EXPECT_EQ(prepared.weight_sources.at(0).byte_length, 102);
}

TEST_F(BasicInputsTest, UnknownKernelIsExplicitlyUnsupported) {
  Call("not.a.basic.operation", {});
  error = "earlier family exclusion";
  EXPECT_EQ(Prepare(), PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "earlier family exclusion");
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
