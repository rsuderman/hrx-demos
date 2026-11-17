// Copyright 2026 AMD ROCm Software
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
uint16_t HalfBits(const std::vector<uint8_t> &bytes, size_t offset) {
  return bytes[offset] | (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}
float DecodeHalf(uint16_t bits) {
  const unsigned exponent = (bits >> 10) & 31;
  const unsigned fraction = bits & 1023;
  const float magnitude =
      exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
               : std::ldexp(float(fraction), -24);
  return bits & 0x8000 ? -magnitude : magnitude;
}
class ProjectionInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary, nullptr);
    directory = std::filesystem::path(temporary) /
                ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    plan.source_asset = (directory / "weights.gguf").string();
    plan.workload.seed = 5;
    plan.model.family = "unrelated-model-name";
    model::TensorBinding binding;
    binding.tensor.name = "selected.matrix.weight";
    binding.tensor.storage_type = 12;
    binding.tensor.rank = 2;
    binding.tensor.normalized_extents = {256, 3, 1, 1};
    binding.tensor.byte_size = 432;
    binding.tensor.absolute_offset = 8;
    model::LayerDescriptor layer;
    layer.tensors.push_back(binding);
    plan.model.layers.push_back(layer);
    Program program;
    program.parameters.push_back({binding.tensor.name, 3, 1000, 432, 4});
    plan.programs.push_back(program);
    invocation.kernel_symbol = "ggml_linear_q4k_q8_1_x4";
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = 6;
    invocation.command.payload.dispatch.workgroup_count[0] = 1;
    invocation.command.payload.dispatch.workgroup_count[1] = 1;
    invocation.command.payload.dispatch.workgroup_count[2] = 1;
    invocation.arguments.resize(6);
    for (size_t i = 0; i < 6; ++i)
      invocation.arguments[i].kind = i < 3 ? LOOMC_CMD_ARGUMENT_KIND_INDEX
                                           : LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[0].value.scalar_bits = 4;
    invocation.arguments[1].value.scalar_bits = 256;
    invocation.arguments[2].value.scalar_bits = 3;
    invocation.arguments[3].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            64, 1152};
    invocation.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 3,
                                            1000, 432};
    invocation.arguments[5].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            1408, 48};
    weights.assign(432, 0x33);
    for (size_t block = 0; block < weights.size(); block += 144) {
      weights[block] = 0;
      weights[block + 1] = 0x3C; // F16 d=1.
      weights[block + 2] = 0;
      weights[block + 3] = 0x38; // F16 dmin=.5.
    }
    WriteWeights();
  }
  void Encoding(const char *symbol, uint32_t type, uint64_t block_values,
                uint64_t block_bytes, uint64_t scale_offset, uint64_t scales,
                uint64_t width = 256) {
    invocation.kernel_symbol = symbol;
    invocation.arguments[1].value.scalar_bits = width;
    invocation.arguments[3].value.buffer.byte_length = 4 * (width / 128) * 144;
    auto &tensor = plan.model.layers[0].tensors[0].tensor;
    tensor.storage_type = type;
    tensor.normalized_extents[0] = width;
    tensor.byte_size = 3 * (width / block_values) * block_bytes;
    plan.programs[0].parameters[0].byte_length = tensor.byte_size;
    invocation.arguments[4].value.buffer.byte_length = tensor.byte_size;
    weights.assign(tensor.byte_size, 0x33);
    for (size_t block = 0; block < weights.size(); block += block_bytes)
      for (size_t scale = 0; scale < scales; ++scale) {
        weights[block + scale_offset + scale * 2] = 0;
        weights[block + scale_offset + scale * 2 + 1] = 0x3C;
      }
    WriteWeights();
  }
  void Controls(size_t dimensions, uint64_t bits = UINT32_MAX,
                uint64_t begin = 1, uint64_t origin = 8) {
    decltype(invocation.arguments) controls(4);
    controls[0].kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
    controls[0].value.scalar_bits = begin;
    controls[1].kind = LOOMC_CMD_ARGUMENT_KIND_B32;
    controls[1].value.scalar_bits = bits;
    controls[2].kind = LOOMC_CMD_ARGUMENT_KIND_OFFSET;
    controls[2].value.scalar_bits = origin;
    controls[3].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    controls[3].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 2, 128, 12};
    invocation.arguments.insert(invocation.arguments.begin() + dimensions,
                                controls.begin(), controls.end());
    invocation.command.argument_count = invocation.arguments.size();
    plan.workload.active_tokens = 2;
  }
  void Quantizer() {
    invocation.kernel_symbol = "ggml_quantize_q8_1_x4_f32";
    invocation.arguments.resize(4);
    invocation.arguments[2].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[2].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            64, 4096};
    invocation.arguments[3].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[3].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            5000, 1152};
    invocation.command.argument_count = 4;
    // Three launched rows of two groups each; scalar capacity remains four.
    invocation.command.payload.dispatch.workgroup_count[0] = 6;
  }
  void Pair() {
    auto second_tensor = plan.model.layers[0].tensors[0];
    second_tensor.tensor.name = "selected.up.weight";
    second_tensor.tensor.absolute_offset = 8 + 432;
    plan.model.layers[0].tensors.push_back(second_tensor);
    plan.programs[0].parameters.push_back(
        {second_tensor.tensor.name, 3, 2000, 432, 4});
    auto up = weights;
    for (size_t block = 0; block < up.size(); block += 144)
      up[block + 16] = 0x44; // Different finite payload, same encoding/scales.
    weights.insert(weights.end(), up.begin(), up.end());
    WriteWeights();
    invocation.invocation_id = "p0/c0";
    invocation.command.begins_barrier_wave = true;
    auto second = invocation;
    second.invocation_id = "p0/c1";
    second.command_ordinal = 1;
    second.arguments[4].value.buffer.byte_offset = 2000;
    second.arguments[5].value.buffer.byte_offset = 1600;
    plan.invocations = {invocation, second};
    plan.programs[0].commands = {invocation.command, second.command};
    plan.programs[0].layout.command_count = 2;
    plan.implementations.resize(1);
    plan.implementations[0].implementation_id = "retained-q4-implementation";
    plan.implementations[0].roots.resize(1);
  }
  void Gated() {
    Pair();
    invocation.kernel_symbol = "llm.projection.q4k_swiglu";
    auto up = invocation.arguments[4];
    up.value.buffer.byte_offset = 2000;
    invocation.arguments.insert(invocation.arguments.begin() + 5, up);
    invocation.command.argument_count = invocation.arguments.size();
  }
  void WriteWeights() {
    std::ofstream file(plan.source_asset, std::ios::binary | std::ios::trunc);
    file.write("metadata", 8);
    file.write(reinterpret_cast<const char *>(weights.data()), weights.size());
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
  }
  std::filesystem::path directory;
  Plan plan;
  Invocation invocation;
  std::vector<uint8_t> weights;
};

TEST_F(ProjectionInputsTest, GatedProjectionKeepsBothExactWeightSources) {
  Gated();
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PreparationStatus::kReady,
            PrepareProjectionCase(plan, invocation, 10000, &prepared, &error))
      << error;
  ASSERT_EQ(2u, prepared.weight_sources.size());
  EXPECT_EQ("selected.matrix.weight", prepared.weight_sources[0].key);
  EXPECT_EQ("selected.up.weight", prepared.weight_sources[1].key);
  EXPECT_EQ(8u, prepared.weight_sources[0].source_offset);
  EXPECT_EQ(440u, prepared.weight_sources[1].source_offset);
  ASSERT_EQ(4u, prepared.buffer_extents.size());
  EXPECT_EQ((std::vector<uint64_t>{1, 256, 3}), prepared.source_workloads);
  const auto *root =
      FindInputRoot(prepared.roots, invocation.arguments[4].value.buffer);
  ASSERT_NE(root, nullptr);
  const auto begin = root->original_byte_offset;
  EXPECT_EQ(0x33, root->bytes[1000 - begin + 16]);
  EXPECT_EQ(0x44, root->bytes[2000 - begin + 16]);
}

TEST_F(ProjectionInputsTest, GatedProjectionValidatesTheSecondMatrix) {
  Gated();
  PreparedCase prepared;
  std::string error;
  plan.model.layers[0].tensors[1].tensor.storage_type = 14;
  EXPECT_EQ(PreparationStatus::kFailure,
            PrepareProjectionCase(plan, invocation, 10000, &prepared, &error));
  plan.model.layers[0].tensors[1].tensor.storage_type = 12;
  EXPECT_EQ(PreparationStatus::kUnsupported,
            PrepareProjectionCase(plan, invocation, 1, &prepared, &error));
  weights[432 + 1] = 0x7C;
  WriteWeights();
  EXPECT_EQ(PreparationStatus::kFailure,
            PrepareProjectionCase(plan, invocation, 10000, &prepared, &error));
}

TEST_F(ProjectionInputsTest, UsesExactTensorAndPreservesAliasesAndCapacity) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{1, 256, 3}));
  EXPECT_EQ(prepared.source_argument_types,
            (std::vector<std::string>{"index", "index", "index", "buffer",
                                      "buffer", "buffer"}));
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 4);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 1152);
  EXPECT_EQ(prepared.buffer_extents[1].byte_length, 432);
  EXPECT_EQ(prepared.buffer_extents[2].byte_length, 48);
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "selected.matrix.weight");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 8);
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 432);
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  ASSERT_EQ(prepared.representative_weight_spans.size(), 1);
  EXPECT_EQ(prepared.representative_weight_spans[0].binding.byte_offset, 1000);
  const auto *packed =
      FindInputRoot(prepared.roots, invocation.arguments[3].value.buffer);
  EXPECT_EQ(packed, FindInputRoot(prepared.roots,
                                  invocation.arguments[5].value.buffer));
  ASSERT_NE(packed, nullptr);
  EXPECT_EQ(packed->original_byte_offset, 64);
  EXPECT_EQ(packed->bytes.size(), 1392);
  const auto *weight =
      FindInputRoot(prepared.roots, invocation.arguments[4].value.buffer);
  ASSERT_NE(weight, nullptr);
  EXPECT_EQ(weight->bytes, weights);
}

TEST_F(ProjectionInputsTest, PackedBlocksHaveExactScaleSumAndSignedPayload) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  const auto &bytes =
      FindInputRoot(prepared.roots, invocation.arguments[3].value.buffer)
          ->bytes;
  bool distinct = false;
  for (size_t group = 0; group < 8; ++group)
    for (size_t block = 0; block < 4; ++block) {
      const size_t base = group * 144;
      EXPECT_EQ(HalfBits(bytes, base + block * 4), 0x2000);
      int sum = 0;
      for (size_t element = 0; element < 32; ++element) {
        const auto byte = bytes[base + 16 + block * 32 + element];
        const int value = byte < 128 ? byte : int(byte) - 256;
        sum += value;
        if (element == 0)
          EXPECT_EQ(value, 127);
        else if (element == 1)
          EXPECT_EQ(value, -127);
        else
          EXPECT_LE(std::abs(value), 3);
      }
      EXPECT_FLOAT_EQ(DecodeHalf(HalfBits(bytes, base + block * 4 + 2)),
                      static_cast<float>(sum) / 128.0f);
      distinct |= bytes[base + 18] != bytes[18];
    }
  EXPECT_TRUE(distinct);
}

TEST_F(ProjectionInputsTest, SeedChangesValidFixtureButNotNativeArguments) {
  PreparedCase a, b;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &a, &error),
            PreparationStatus::kReady)
      << error;
  ++plan.workload.seed;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 4096, &b, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_NE(
      FindInputRoot(a.roots, invocation.arguments[3].value.buffer)->bytes,
      FindInputRoot(b.roots, invocation.arguments[3].value.buffer)->bytes);
  EXPECT_EQ(a.source_workloads, b.source_workloads);
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 4);
}

TEST_F(ProjectionInputsTest, FullRuntimeCapacityRemainsSeparateFromGridRows) {
  invocation.arguments[0].value.scalar_bits = 512;
  invocation.arguments[3].value.buffer.byte_length = 147456;
  invocation.arguments[5].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 1,
                                          0, 6144};
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 200000, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads.front(), 1);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 147456);
  EXPECT_EQ(prepared.buffer_extents[2].byte_length, 6144);
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 512);
}

TEST_F(ProjectionInputsTest, RejectsMalformedOrOverlappingBindingsAtomically) {
  PreparedCase prepared;
  prepared.setup_recipe_id = "untouched";
  std::string error;
  auto changed = invocation;
  changed.kernel_symbol = "unclaimed.projection";
  error = "not claimed";
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 4096, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_EQ(error, "not claimed");
  changed = invocation;
  changed.arguments[2].kind = LOOMC_CMD_ARGUMENT_KIND_B64;
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  changed = invocation;
  changed.arguments[5].value.buffer = changed.arguments[3].value.buffer;
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("disjoint"), std::string::npos);
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 1800, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
}

TEST_F(ProjectionInputsTest, BudgetExclusionPrecedesIoButNotRangeValidation) {
  ASSERT_TRUE(std::filesystem::remove(plan.source_asset));
  PreparedCase prepared;
  prepared.setup_recipe_id = "untouched";
  std::string error;
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 1800, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  auto changed = invocation;
  changed.arguments[3].value.buffer.byte_length = 1;
  EXPECT_EQ(PrepareProjectionCase(plan, changed, 1, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
}

TEST_F(ProjectionInputsTest, TensorEncodingShapeRangeAndScaleMustBeExact) {
  PreparedCase prepared;
  std::string error;
  auto changed = plan;
  changed.model.layers[0].tensors[0].tensor.storage_type = 14;
  EXPECT_EQ(PrepareProjectionCase(changed, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  changed = plan;
  changed.model.layers[0].tensors[0].tensor.normalized_extents[1] = 4;
  EXPECT_EQ(PrepareProjectionCase(changed, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  changed = plan;
  changed.programs[0].parameters[0].byte_length += 144;
  EXPECT_EQ(PrepareProjectionCase(changed, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  weights[1] = 0x7C;
  WriteWeights();
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
}

TEST_F(ProjectionInputsTest, TruncatedWeightFileDoesNotProducePartialCase) {
  weights.resize(100);
  WriteWeights();
  PreparedCase prepared;
  prepared.setup_recipe_id = "untouched";
  std::string error;
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 4096, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(prepared.setup_recipe_id, "untouched");
}
TEST_F(ProjectionInputsTest,
       EveryQuantizedEncodingUsesItsExactBlocksAndScaleOffsets) {
  struct EncodingCase {
    const char *name;
    uint32_t type;
    uint64_t values, bytes, offset, scales;
  };
  const EncodingCase encodings[] = {
      {"ggml_linear_q4k_q8_1_x4", 12, 256, 144, 0, 2},
      {"ggml_linear_q5k_q8_1_x4", 13, 256, 176, 0, 2},
      {"ggml_linear_q6k_q8_1_x4", 14, 256, 210, 208, 1},
      {"ggml_linear_q8_0_q8_1_x4", 8, 32, 34, 0, 1},
      {"ggml_linear_q5_0_q8_1_x4", 6, 32, 22, 0, 1}};
  for (const auto &encoding : encodings) {
    SCOPED_TRACE(encoding.name);
    Encoding(encoding.name, encoding.type, encoding.values, encoding.bytes,
             encoding.offset, encoding.scales);
    PreparedCase prepared;
    std::string error;
    ASSERT_EQ(PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
              PreparationStatus::kReady)
        << error;
    ASSERT_EQ(prepared.weight_sources.size(), 1);
    EXPECT_EQ(prepared.weight_sources[0].byte_length, weights.size());
    const auto *root =
        FindInputRoot(prepared.roots, invocation.arguments[4].value.buffer);
    ASSERT_NE(root, nullptr);
    EXPECT_TRUE(
        std::equal(weights.begin(), weights.end(), root->bytes.begin()));
    auto wrong_type = plan;
    ++wrong_type.model.layers[0].tensors[0].tensor.storage_type;
    EXPECT_EQ(
        PrepareProjectionCase(wrong_type, invocation, 8192, &prepared, &error),
        PreparationStatus::kFailure);
    weights[encoding.offset + 1] = 0x7C;
    WriteWeights();
    EXPECT_EQ(PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
              PreparationStatus::kFailure);
    EXPECT_NE(error.find("nonfinite"), std::string::npos);
  }
}
TEST_F(ProjectionInputsTest,
       Block32AllowsWidth128ButQ6RetainsItsActualMaximum) {
  Encoding("ggml_linear_q5_0_q8_1_x4", 6, 32, 22, 0, 1, 128);
  PreparedCase prepared;
  std::string error;
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  Encoding("ggml_linear_q6k_q8_1_x4", 14, 256, 210, 208, 1);
  Encoding("ggml_linear_q6k_q8_1_x4", 14, 256, 210, 208, 1, 17408);
  invocation.arguments[5].value.buffer.root_index = 1;
  ASSERT_EQ(PrepareProjectionCase(plan, invocation, 200000, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  invocation.arguments[1].value.scalar_bits = 17664;
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
            PreparationStatus::kFailure);
}
TEST_F(ProjectionInputsTest,
       CompatibilityProjectionsPreserveShiftedDynamicAndFixedI32) {
  for (bool q6 : {false, true}) {
    SCOPED_TRACE(q6);
    Encoding(q6 ? "ggml_linear_q6k_q8_1_x4_compat"
                : "ggml_linear_q4k_q8_1_x4_compat",
             q6 ? 14 : 12, 256, q6 ? 210 : 144, q6 ? 208 : 0, q6 ? 1 : 2);
    Controls(3);
    invocation.command.payload.dispatch.workgroup_count[1] = 3;
    for (const uint64_t bits :
         {uint64_t{UINT32_MAX}, uint64_t{2}, uint64_t{0}}) {
      invocation.arguments[4].value.scalar_bits = bits;
      PreparedCase prepared;
      std::string error;
      ASSERT_EQ(
          PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
          PreparationStatus::kReady)
          << error;
      EXPECT_EQ(prepared.source_argument_types[4], "i32");
      EXPECT_EQ(invocation.arguments[4].value.scalar_bits, bits);
      const auto *root =
          FindInputRoot(prepared.roots, invocation.arguments[6].value.buffer);
      ASSERT_NE(root, nullptr);
      EXPECT_EQ(root->bytes[bits == UINT32_MAX ? 8 : 0],
                bits == UINT32_MAX ? 2 : 0);
    }
    invocation.arguments[4].value.scalar_bits = UINT64_MAX;
    PreparedCase untouched;
    untouched.setup_recipe_id = "untouched";
    std::string error;
    EXPECT_EQ(PrepareProjectionCase(plan, invocation, 8192, &untouched, &error),
              PreparationStatus::kFailure);
    EXPECT_EQ(untouched.setup_recipe_id, "untouched");
    // Restore canonical arguments for the next table iteration.
    invocation.arguments.erase(invocation.arguments.begin() + 3,
                               invocation.arguments.begin() + 7);
    invocation.command.argument_count = 6;
    invocation.command.payload.dispatch.workgroup_count[1] = 1;
  }
}
TEST_F(ProjectionInputsTest,
       ActivationQuantizersUseFiniteF32AndOriginalCapacity) {
  Quantizer();
  for (bool compat : {false, true}) {
    if (compat) {
      invocation.kernel_symbol = "ggml_quantize_q8_1_x4_qk_compat_f32";
      Controls(2);
    }
    PreparedCase prepared;
    std::string error;
    ASSERT_EQ(PrepareProjectionCase(plan, invocation, 10000, &prepared, &error),
              PreparationStatus::kReady)
        << error;
    EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3, 256}));
    EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 4);
    EXPECT_TRUE(prepared.weight_sources.empty());
    const auto input = invocation.arguments[compat ? 6 : 2].value.buffer;
    const auto output = invocation.arguments.back().value.buffer;
    const auto *in = FindInputRoot(prepared.roots, input);
    const auto *out = FindInputRoot(prepared.roots, output);
    ASSERT_NE(in, nullptr);
    ASSERT_NE(out, nullptr);
    const uint64_t begin = input.byte_offset - in->original_byte_offset;
    for (size_t i = 0; i < 4 * 256; ++i) {
      float value;
      std::memcpy(&value, in->bytes.data() + begin + i * 4, 4);
      EXPECT_TRUE(std::isfinite(value));
    }
    const uint64_t out_begin = output.byte_offset - out->original_byte_offset;
    EXPECT_TRUE(std::all_of(out->bytes.begin() + out_begin,
                            out->bytes.begin() + out_begin + 1152,
                            [](uint8_t x) { return x == 0xA5; }));
    auto bad = invocation;
    --bad.command.argument_count;
    EXPECT_EQ(PrepareProjectionCase(plan, bad, 10000, &prepared, &error),
              PreparationStatus::kFailure);
    if (compat) {
      bad = invocation;
      bad.arguments[4].value.scalar_bits = 9;
      EXPECT_EQ(PrepareProjectionCase(plan, bad, 10000, &prepared, &error),
                PreparationStatus::kFailure);
    }
    bad = invocation;
    bad.command.payload.dispatch.workgroup_count[0] = 5;
    EXPECT_EQ(PrepareProjectionCase(plan, bad, 10000, &prepared, &error),
              PreparationStatus::kFailure);
    EXPECT_EQ(PrepareProjectionCase(plan, invocation, 1, &prepared, &error),
              PreparationStatus::kUnsupported);
  }
}
TEST_F(ProjectionInputsTest,
       FixedCountDummyMayAliasAnyLiveArgumentWithoutWritingIt) {
  const auto original = invocation;
  for (bool q6 : {false, true}) {
    invocation = original;
    Encoding(q6 ? "ggml_linear_q6k_q8_1_x4_compat"
                : "ggml_linear_q4k_q8_1_x4_compat",
             q6 ? 14 : 12, 256, q6 ? 210 : 144, q6 ? 208 : 0, q6 ? 1 : 2);
    Controls(3, 1, 0, 0);
    const auto unused = invocation.arguments[6].value.buffer;
    for (const size_t aliased : {size_t{7}, size_t{8}, size_t{9}}) {
      SCOPED_TRACE(aliased);
      invocation.arguments[6].value.buffer =
          invocation.arguments[aliased].value.buffer;
      PreparedCase prepared;
      std::string error;
      ASSERT_EQ(
          PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
          PreparationStatus::kReady)
          << error;
      const auto input = invocation.arguments[7].value.buffer;
      const auto *packed = FindInputRoot(prepared.roots, input);
      ASSERT_TRUE(packed);
      EXPECT_EQ(HalfBits(packed->bytes,
                         input.byte_offset - packed->original_byte_offset),
                0x2000);
      const auto weight = invocation.arguments[8].value.buffer;
      const auto *root = FindInputRoot(prepared.roots, weight);
      ASSERT_TRUE(root);
      EXPECT_TRUE(std::equal(weights.begin(), weights.end(),
                             root->bytes.begin() + weight.byte_offset -
                                 root->original_byte_offset));
      const auto output = invocation.arguments[9].value.buffer;
      const auto *out = FindInputRoot(prepared.roots, output);
      ASSERT_TRUE(out);
      EXPECT_EQ(out->bytes[output.byte_offset - out->original_byte_offset],
                0xA5);
      // A dynamic count actually reads the view and cannot share these values.
      invocation.arguments[4].value.scalar_bits = UINT32_MAX;
      EXPECT_EQ(
          PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
          PreparationStatus::kFailure);
      invocation.arguments[4].value.scalar_bits = 1;
    }
    invocation.arguments[6].value.buffer = unused;
  }
  invocation = original;
  Quantizer();
  invocation.kernel_symbol = "ggml_quantize_q8_1_x4_qk_compat_f32";
  Controls(2, 1, 0, 0);
  for (size_t aliased : {size_t{6}, size_t{7}}) {
    invocation.arguments[5].value.buffer =
        invocation.arguments[aliased].value.buffer;
    PreparedCase prepared;
    std::string error;
    ASSERT_EQ(PrepareProjectionCase(plan, invocation, 10000, &prepared, &error),
              PreparationStatus::kReady)
        << error;
    const auto output = invocation.arguments[7].value.buffer;
    const auto *root = FindInputRoot(prepared.roots, output);
    ASSERT_TRUE(root);
    EXPECT_EQ(root->bytes[output.byte_offset - root->original_byte_offset],
              0xA5);
  }
}
TEST_F(ProjectionInputsTest,
       ActualQwen3FinalProjectionFixedDummyReachesBudgetBeforeIo) {
  // Captured p0/c508: capacity1 K1024 C151936, fixed_count1, count view
  // exactly aliases packed input. Preserve those native placements; no large
  // weight fixture is needed to prove the budget classification before IO.
  Encoding("ggml_linear_q6k_q8_1_x4_compat", 14, 256, 210, 208, 1, 1024);
  invocation.arguments[0].value.scalar_bits = 1;
  invocation.arguments[2].value.scalar_bits = 151936;
  invocation.arguments[3].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 6,
                                          4608, 1152};
  invocation.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 0, 0,
                                          127626240};
  invocation.arguments[5].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 6,
                                          5888, 607744};
  auto &tensor = plan.model.layers[0].tensors[0].tensor;
  tensor.normalized_extents[1] = 151936;
  tensor.byte_size = 127626240;
  plan.programs[0].parameters[0] = {tensor.name, 0, 0, 127626240, 4};
  Controls(3, 1, 0, 0);
  invocation.arguments[6].value.buffer = invocation.arguments[7].value.buffer;
  ASSERT_TRUE(std::filesystem::remove(plan.source_asset));
  PreparedCase prepared;
  prepared.setup_recipe_id = "preserved";
  std::string error;
  EXPECT_EQ(PrepareProjectionCase(plan, invocation, 8192, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  EXPECT_EQ(prepared.setup_recipe_id, "preserved");
}
TEST_F(ProjectionInputsTest,
       PairMergesRootsOnceAndPreservesOrderedExactWeights) {
  Pair();
  PreparedSequenceCase prepared;
  std::string error;
  ASSERT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.invocation_indices, (std::array<size_t, 2>{0, 1}));
  EXPECT_EQ(prepared.inputs.measurement_subject, "sequence");
  EXPECT_EQ(prepared.inputs.reset_policy, "none_overwrite_only");
  EXPECT_EQ(prepared.inputs.arithmetic_check, "unchecked");
  EXPECT_TRUE(prepared.inputs.resets.empty());
  EXPECT_EQ(prepared.inputs.buffer_extents.size(), 6);
  ASSERT_EQ(prepared.inputs.weight_sources.size(), 2);
  EXPECT_EQ(prepared.inputs.weight_sources[0].key, "selected.matrix.weight");
  EXPECT_EQ(prepared.inputs.weight_sources[1].key, "selected.up.weight");
  EXPECT_EQ(prepared.inputs.representative_weight_spans.size(), 2);
  ASSERT_EQ(prepared.inputs.roots.size(), 2);
  const auto packed = plan.invocations[0].arguments[3].value.buffer;
  const auto *root = FindInputRoot(prepared.inputs.roots, packed);
  ASSERT_TRUE(root);
  EXPECT_EQ(root->bytes.size(),
            1584); // Shared input and both disjoint outputs.
  EXPECT_EQ(root->initialized_ranges.size(), 3); // Packed input written once.
  for (size_t member = 0; member < 2; ++member) {
    const auto output = plan.invocations[member].arguments[5].value.buffer;
    EXPECT_EQ(root->bytes[output.byte_offset - root->original_byte_offset],
              0xA5);
    const auto weight = plan.invocations[member].arguments[4].value.buffer;
    const auto *fixed = FindInputRoot(prepared.inputs.roots, weight);
    ASSERT_TRUE(fixed);
    EXPECT_EQ(fixed->bytes.size(),
              1432); // Native fixed parameter gap retained.
    EXPECT_TRUE(std::equal(weights.begin() + member * 432,
                           weights.begin() + (member + 1) * 432,
                           fixed->bytes.begin() + weight.byte_offset -
                               fixed->original_byte_offset));
  }
}
TEST_F(ProjectionInputsTest, PairRejectsNonadjacentProgramsAndUnrecordedWaves) {
  Pair();
  const auto original = plan;
  PreparedSequenceCase prepared;
  prepared.inputs.setup_recipe_id = "preserved";
  std::string error;
  plan.invocations[1].command_ordinal = 2;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.programs.push_back(plan.programs[0]);
  plan.invocations[1].program_ordinal = 1;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.invocations[1].command.begins_barrier_wave = false;
  plan.programs[0].commands[1].begins_barrier_wave = false;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  auto hidden = plan.programs[0].commands[0];
  hidden.kind = LOOMC_CMD_COMMAND_KIND_BARRIER_EXECUTION;
  plan.programs[0].commands.insert(plan.programs[0].commands.begin() + 1,
                                   hidden);
  plan.invocations[1].command_ordinal = 2;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.programs[0].commands[1].kind = LOOMC_CMD_COMMAND_KIND_FILL;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(prepared.inputs.setup_recipe_id, "preserved");
}
TEST_F(ProjectionInputsTest,
       PairRejectsDifferentRetainedGeometryAndSharedInput) {
  Pair();
  const auto original = plan;
  PreparedSequenceCase prepared;
  std::string error;
  plan.invocations[1].arguments[0].value.scalar_bits = 3;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.invocations[1].command.payload.dispatch.workgroup_count[1] = 2;
  plan.programs[0].commands[1] = plan.invocations[1].command;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.invocations[1].implementation_index = 1;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.invocations[1].request_root_ordinal = 1;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.invocations[1].arguments[3].value.buffer.byte_offset += 16;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
}
TEST_F(ProjectionInputsTest, PairOutputsMustBeIndependentOfEveryReadAndWrite) {
  Pair();
  PreparedSequenceCase prepared;
  std::string error;
  const auto original = plan;
  plan.invocations[1].arguments[5].value.buffer =
      plan.invocations[0].arguments[5].value.buffer;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
  plan = original;
  plan.invocations[1].arguments[5].value.buffer =
      plan.invocations[0].arguments[4].value.buffer;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kUnsupported);
}
TEST_F(ProjectionInputsTest, PairBudgetsMergedHullIncludingGapBeforeIo) {
  Pair();
  ASSERT_TRUE(std::filesystem::remove(plan.source_asset));
  PreparedSequenceCase prepared;
  prepared.inputs.setup_recipe_id = "preserved";
  std::string error;
  // Payload sum without the native fixed gap would fit this cap; merged hull
  // must include that gap and reject before trying the deliberately absent IO.
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 2500, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
  plan.model.layers[0].tensors[1].tensor.storage_type = 14;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 1, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_EQ(prepared.inputs.setup_recipe_id, "preserved");
}
TEST_F(ProjectionInputsTest,
       PairRejectsConflictingInitializedReadonlyWeightBytes) {
  Pair();
  // Each exact parameter remains a valid matrix, but their native read views
  // overlap while GGUF bytes disagree. Shared initialized-range checking must
  // reject this realization atomically rather than silently overwrite either.
  plan.programs[0].parameters[1].byte_offset = 1100;
  plan.invocations[1].arguments[4].value.buffer.byte_offset = 1100;
  PreparedSequenceCase prepared;
  prepared.inputs.setup_recipe_id = "preserved";
  std::string error;
  EXPECT_EQ(PrepareProjectionPair(plan, 0, 1, 3500, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("conflicting initialization"), std::string::npos);
  EXPECT_EQ(prepared.inputs.setup_recipe_id, "preserved");
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
