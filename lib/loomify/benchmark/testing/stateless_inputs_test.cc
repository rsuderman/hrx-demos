// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: MIT
#include "lib/loomify/benchmark/bundle.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

#include "gtest/gtest.h"

namespace hrx::loomify_gguf::benchmark {
namespace {
class StatelessInputsTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *directory = std::getenv("TEST_TMPDIR");
    ASSERT_NE(directory, nullptr);
    path = std::filesystem::path(directory) /
           ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directories(path));
    plan.source_asset = (path / "weights.gguf").string();
    plan.model.embedding_width = 4;
    plan.workload.seed = 7;
    model::TensorBinding gamma;
    gamma.tensor.name = "blk.0.attn_norm.weight";
    gamma.tensor.storage_type = 0;
    gamma.tensor.byte_size = 16;
    gamma.tensor.absolute_offset = 8;
    plan.model.globals.push_back(gamma);
    Program program;
    program.parameters.push_back({gamma.tensor.name, 3, 1000, 16, 4});
    plan.programs.push_back(program);
    invocation.kernel_symbol = "qwen38_rmsnorm_f32_prefill";
    invocation.command.kind = LOOMC_CMD_COMMAND_KIND_DISPATCH_DIRECT;
    invocation.command.argument_count = 5;
    invocation.command.payload.dispatch.workgroup_count[0] = 1;
    invocation.command.payload.dispatch.workgroup_count[1] = 3;
    invocation.command.payload.dispatch.workgroup_count[2] = 1;
    invocation.arguments.resize(5);
    invocation.arguments[0].kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
    invocation.arguments[0].value.scalar_bits = 8;
    invocation.arguments[1].kind = LOOMC_CMD_ARGUMENT_KIND_B32;
    const float epsilon = 0.00001f;
    uint32_t bits;
    std::memcpy(&bits, &epsilon, sizeof(bits));
    invocation.arguments[1].value.scalar_bits = bits;
    for (unsigned i = 2; i < 5; ++i)
      invocation.arguments[i].kind = LOOMC_CMD_ARGUMENT_KIND_BUFFER;
    invocation.arguments[2].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            64, 128};
    invocation.arguments[3].value.buffer = {LOOMC_CMD_BUFFER_ROLE_FIXED, 3,
                                            1000, 16};
    invocation.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 0,
                                            256, 128};
    WriteGamma({0x3F800000, 0x40000000, 0x3F000000, 0xBF800000});
  }
  void WriteGamma(const std::vector<uint32_t> &values) {
    std::ofstream file(plan.source_asset, std::ios::binary | std::ios::trunc);
    file.write("metadata", 8);
    for (uint32_t value : values)
      for (unsigned lane = 0; lane < 4; ++lane)
        file.put(static_cast<char>(value >> (lane * 8)));
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  std::filesystem::path path;
  Plan plan;
  Invocation invocation;
};

TEST_F(StatelessInputsTest,
       UsesActualParameterPlacementAndPreservesRootAliases) {
  PreparedCase prepared;
  std::string error;
  ASSERT_EQ(PrepareStatelessCase(plan, invocation, 512, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.setup_recipe_id, "rms_f32_prefill/v1");
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{3}));
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 8);
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 128);
  EXPECT_EQ(prepared.arithmetic_check, "unchecked");
  EXPECT_EQ(prepared.measurement_subject, "kernel");
  ASSERT_EQ(prepared.weight_sources.size(), 1);
  EXPECT_EQ(prepared.weight_sources[0].key, "blk.0.attn_norm.weight");
  EXPECT_EQ(prepared.weight_sources[0].source_offset, 8);
  EXPECT_EQ(prepared.weight_sources[0].byte_length, 16);
  ASSERT_EQ(prepared.roots.size(), 2);
  const auto *activations =
      FindInputRoot(prepared.roots, invocation.arguments[2].value.buffer);
  ASSERT_NE(activations, nullptr);
  EXPECT_EQ(activations->original_byte_offset, 64);
  EXPECT_EQ(activations->bytes.size(), 320);
  EXPECT_EQ(activations, FindInputRoot(prepared.roots,
                                       invocation.arguments[4].value.buffer));
  const auto *gamma =
      FindInputRoot(prepared.roots, invocation.arguments[3].value.buffer);
  ASSERT_NE(gamma, nullptr);
  EXPECT_EQ(gamma->original_byte_offset, 1000);
  EXPECT_EQ(gamma->bytes,
            (std::vector<uint8_t>{0, 0, 0x80, 0x3F, 0, 0, 0, 0x40, 0, 0, 0,
                                  0x3F, 0, 0, 0x80, 0xBF}));
}

TEST_F(StatelessInputsTest, UnsupportedOrMismatchedRecipesFailExplicitly) {
  PreparedCase prepared;
  std::string error;
  auto changed = invocation;
  changed.kernel_symbol = "unknown.operation";
  EXPECT_EQ(PrepareStatelessCase(plan, changed, 512, &prepared, &error),
            PreparationStatus::kUnsupported);
  changed = invocation;
  changed.arguments[1].kind = LOOMC_CMD_ARGUMENT_KIND_INDEX;
  EXPECT_EQ(PrepareStatelessCase(plan, changed, 512, &prepared, &error),
            PreparationStatus::kFailure);
  changed = invocation;
  changed.arguments[4].value.buffer = changed.arguments[2].value.buffer;
  EXPECT_EQ(PrepareStatelessCase(plan, changed, 512, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("disjoint"), std::string::npos);
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 335, &prepared, &error),
            PreparationStatus::kUnsupported);
  EXPECT_NE(error.find("budget"), std::string::npos);
}

TEST_F(StatelessInputsTest, NonfiniteOrMissingGammaIsNotDeclaredSafe) {
  PreparedCase prepared;
  std::string error;
  WriteGamma({0x7F800000, 0, 0, 0});
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 512, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("nonfinite"), std::string::npos);
  plan.programs[0].parameters[0].key = "other.weight";
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 512, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("exact GGUF F32"), std::string::npos);
}

TEST_F(StatelessInputsTest, LogicalGridRowsDoNotReplaceRuntimeCapacity) {
  PreparedCase prepared;
  std::string error;
  invocation.arguments[0].value.scalar_bits = 512;
  invocation.command.payload.dispatch.workgroup_count[1] = 1;
  invocation.arguments[2].value.buffer.byte_length = 8192;
  invocation.arguments[4].value.buffer = {LOOMC_CMD_BUFFER_ROLE_REBINDABLE, 1,
                                          0, 8192};
  ASSERT_EQ(PrepareStatelessCase(plan, invocation, 20000, &prepared, &error),
            PreparationStatus::kReady)
      << error;
  EXPECT_EQ(prepared.source_workloads, (std::vector<uint64_t>{1}));
  EXPECT_EQ(prepared.buffer_extents[0].byte_length, 8192);
  EXPECT_EQ(invocation.arguments[0].value.scalar_bits, 512);
  invocation.command.payload.dispatch.workgroup_count[1] = 513;
  EXPECT_EQ(PrepareStatelessCase(plan, invocation, 20000, &prepared, &error),
            PreparationStatus::kFailure);
  EXPECT_NE(error.find("grid/capacity"), std::string::npos);
}
// Exercise scalar emission through a genuine retained native request rather
// than testing a second copy of literal-formatting logic.
class BenchmarkScalarEmissionTest : public ::testing::Test {
protected:
  template <class T, void (*Release)(T *)>
  using Owner = std::unique_ptr<T, decltype(Release)>;
  static bool Ok(loomc_status_t status) {
    const bool ok = loomc_status_is_ok(status);
    if (!ok) {
      const auto text = loomc_status_message(status);
      ADD_FAILURE() << std::string(text.data, text.size);
    }
    loomc_status_free(status);
    return ok;
  }
  static bool Succeeded(loomc_result_t *result) {
    if (result && loomc_result_succeeded(result))
      return true;
    for (size_t i = 0; result && i < loomc_result_diagnostic_count(result);
         ++i) {
      const auto *diagnostic = loomc_result_diagnostic_at(result, i);
      ADD_FAILURE() << std::string(diagnostic->message.data,
                                   diagnostic->message.size);
    }
    return false;
  }
  void SetUp() override {
    const char *temporary = std::getenv("TEST_TMPDIR");
    ASSERT_NE(temporary, nullptr);
    path = std::filesystem::path(temporary) /
           ::testing::UnitTest::GetInstance()->current_test_info()->name();
    ASSERT_TRUE(std::filesystem::create_directories(path));
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
  }
  void TearDown() override {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  bool MakeCase(int32_t count, bool flag, Plan *plan, PreparedCase *prepared) {
    std::ostringstream text;
    text << R"loom(
kernel.def @benchmark.scalar_probe() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%count: i32, %enabled: i1, %output: buffer) {
  kernel.return
}
command.def public @benchmark.scalar_program() launch(%output: buffer) {
  %count = scalar.constant )loom"
         << count << R"loom( : i32
  %enabled = scalar.constant )loom"
         << (flag ? "true" : "false") << R"loom( : i1
  kernel.launch @benchmark.scalar_probe(%count, %enabled, %output) : (i32, i1, buffer)
  command.return
}
)loom";
    const auto contents = text.str();
    loomc_source_options_t source_options{};
    source_options.format = LOOMC_SOURCE_FORMAT_TEXT;
    source_options.storage = LOOMC_SOURCE_STORAGE_COPY;
    source_options.identifier =
        loomc_make_cstring_view("benchmark-scalar-probe.loom");
    source_options.contents =
        loomc_make_byte_span(contents.data(), contents.size());
    loomc_source_t *raw_source = nullptr;
    if (!Ok(loomc_source_create(&source_options, loomc_allocator_system(),
                                &raw_source)))
      return false;
    Owner<loomc_source_t, loomc_source_release> source(raw_source,
                                                       loomc_source_release);
    loomc_module_t *raw_module = nullptr;
    loomc_result_t *raw_result = nullptr;
    const auto parse = loomc_module_deserialize_from_source(
        context, workspace, source.get(), nullptr, loomc_allocator_system(),
        &raw_module, &raw_result);
    Owner<loomc_module_t, loomc_module_release> module(raw_module,
                                                       loomc_module_release);
    Owner<loomc_result_t, loomc_result_release> parsed(raw_result,
                                                       loomc_result_release);
    if (!Ok(parse) || !Succeeded(parsed.get()))
      return false;
    std::vector<std::shared_ptr<loomc_request_t>> requests;
    const auto root = loomc_make_cstring_view("benchmark.scalar_program");
    loomc_cmd_program_module_options_t options{};
    options.type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS;
    options.structure_size = sizeof(options);
    options.root_symbols = &root;
    options.root_symbol_count = 1;
    options.request_sink = {
        [](void *data, loomc_request_t *request) {
          static_cast<std::vector<std::shared_ptr<loomc_request_t>> *>(data)
              ->emplace_back(request, loomc_request_release);
          return loomc_ok_status();
        },
        &requests};
    loomc_product_t *raw_product = nullptr;
    raw_result = nullptr;
    const auto build = loomc_cmd_program_product_build_module(
        workspace, module.get(), &options, loomc_allocator_system(),
        &raw_product, &raw_result);
    Owner<loomc_product_t, loomc_product_release> product(
        raw_product, loomc_product_release);
    Owner<loomc_result_t, loomc_result_release> built(raw_result,
                                                      loomc_result_release);
    if (!Ok(build) || !Succeeded(built.get()))
      return false;
    if (requests.size() != 1) {
      ADD_FAILURE() << "expected one retained probe request";
      return false;
    }
    Implementation implementation;
    implementation.implementation_id =
        "scalar-probe-" + std::to_string(count) + (flag ? "-true" : "-false");
    implementation.request = requests[0];
    for (size_t i = 0; i < loomc_request_root_count(requests[0].get()); ++i) {
      loomc_request_root_t item{};
      if (!loomc_request_root_at(requests[0].get(), i, &item))
        return false;
      implementation.roots.push_back(item);
    }
    Invocation invocation;
    invocation.invocation_id = "p0/c0";
    invocation.kernel_symbol = "benchmark.scalar_probe";
    if (!loomc_cmd_program_product_command_at(product.get(), 0, 0,
                                              &invocation.command))
      return false;
    for (uint32_t i = 0; i < invocation.command.argument_count; ++i) {
      loomc_cmd_argument_t argument{};
      if (!loomc_cmd_program_product_argument_at(product.get(), 0, 0, i,
                                                 &argument))
        return false;
      invocation.arguments.push_back(argument);
    }
    loomc_request_binding_t binding{};
    if (!loomc_request_binding_at(requests[0].get(), 0, &binding))
      return false;
    invocation.request_root_ordinal = binding.root_ordinal;
    plan->workload_id = "scalar-emission-probe";
    plan->implementations = {std::move(implementation)};
    plan->invocations = {invocation};
    prepared->setup_recipe_id = "scalar-emission-probe/v1";
    prepared->source_argument_types = {"i32", "i1", "buffer"};
    if (invocation.arguments.size() != 3)
      return false;
    prepared->buffer_extents = {{invocation.arguments[2].value.buffer, 4, 4}};
    std::string error;
    return AllocateInputRoots(prepared->buffer_extents, 4, &prepared->roots,
                              &error);
  }
  std::filesystem::path path;
  loomc_context_t *context = nullptr;
  loomc_workspace_t *workspace = nullptr;
};

TEST_F(BenchmarkScalarEmissionTest, PreservesSignedI32AndBooleanCarriers) {
  for (const int32_t count : {INT32_MIN, -1, 0, 17}) {
    for (const bool flag : {false, true}) {
      SCOPED_TRACE(std::to_string(count) + (flag ? " true" : " false"));
      Plan plan;
      PreparedCase prepared;
      ASSERT_TRUE(MakeCase(count, flag, &plan, &prepared));
      const auto &arguments = plan.invocations[0].arguments;
      ASSERT_EQ(arguments.size(), 3);
      EXPECT_EQ(arguments[0].kind, LOOMC_CMD_ARGUMENT_KIND_B32);
      EXPECT_EQ(arguments[0].value.scalar_bits, static_cast<uint32_t>(count));
      EXPECT_EQ(arguments[1].kind, LOOMC_CMD_ARGUMENT_KIND_B8);
      EXPECT_EQ(arguments[1].value.scalar_bits, flag ? 1u : 0u);
      const auto directory =
          path / (std::to_string(count) + (flag ? "-true" : "-false"));
      ASSERT_TRUE(std::filesystem::create_directory(directory));
      std::vector<EmittedCase> emitted;
      std::string error;
      ASSERT_TRUE(EmitPreparedBenchmarkBundle(plan, {{0, std::move(prepared)}},
                                              directory, context, workspace,
                                              &emitted, &error))
          << error;
      ASSERT_EQ(emitted.size(), 1);
      ASSERT_EQ(emitted[0].timed_dispatches.size(), 1);
      std::ifstream file(directory / "benchmark.loom");
      const std::string text((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
      EXPECT_NE(
          text.find("check.literal value(" + std::to_string(count) + ") : i32"),
          std::string::npos);
      EXPECT_NE(text.find(std::string("check.literal value(") +
                          (flag ? "true" : "false") + ") : i1"),
                std::string::npos);
    }
  }
}

TEST_F(BenchmarkScalarEmissionTest, RejectsInvalidCarrierBitsAndSourceType) {
  Plan plan;
  PreparedCase prepared;
  ASSERT_TRUE(MakeCase(-1, true, &plan, &prepared));
  auto reject = [&](Plan candidate, PreparedCase inputs, const char *name,
                    const char *message) {
    const auto directory = path / name;
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    std::vector<EmittedCase> emitted;
    std::string error;
    EXPECT_FALSE(EmitPreparedBenchmarkBundle(
        candidate, {{0, std::move(inputs)}}, directory, context, workspace,
        &emitted, &error));
    EXPECT_NE(error.find(message), std::string::npos) << error;
    EXPECT_FALSE(std::filesystem::exists(directory / "benchmark.loombc"));
  };
  auto changed = plan;
  changed.invocations[0].arguments[0].value.scalar_bits |= UINT64_C(1) << 32;
  reject(changed, prepared, "upper-bits", "upper bits");
  changed = plan;
  changed.invocations[0].arguments[1].value.scalar_bits = 2;
  reject(changed, prepared, "nonboolean", "zero or one");
  changed = plan;
  changed.invocations[0].arguments[0].value.scalar_bits = 0;
  auto changed_inputs = prepared;
  changed_inputs.source_argument_types[0] = "f32";
  reject(changed, changed_inputs, "wrong-source-type", "source runtime ABI");
}
} // namespace
} // namespace hrx::loomify_gguf::benchmark
