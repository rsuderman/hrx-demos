// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/commands/command_values.h"

#include <cstring>
#include <memory>
#include <string>

#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loomc/interop.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf {
namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
using Module = Owner<loomc_module_t, loomc_module_release>;
using Result = Owner<loomc_result_t, loomc_result_release>;
using Source = Owner<loomc_source_t, loomc_source_release>;

::testing::AssertionResult Ok(loomc_status_t status) {
  auto code = loomc_status_code(status);
  auto message = loomc_status_message(status);
  std::string copy(message.data ? message.data : "", message.size);
  loomc_status_free(status);
  if (code == LOOMC_STATUS_OK)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << copy;
}
::testing::AssertionResult Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return ::testing::AssertionSuccess();
  std::string diagnostics;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto d = loomc_result_diagnostic_at(result, i);
      diagnostics.append(d->message.data, d->message.size);
      diagnostics += '\n';
    }
  return ::testing::AssertionFailure() << diagnostics;
}

constexpr char kDefinitions[] = R"(
config.def @width = 16 : index
config.def @floating = 1.5 : f32
kernel.def @sink(%count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%count: index, %weights: buffer, %scratch: buffer, %output: buffer) {
  kernel.return
}
)";

class CommandValuesTest : public ::testing::Test {
protected:
  void SetUp() override {
    loomc_context_t *context = nullptr;
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    context_.reset(context);
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    workspace_.reset(workspace);
    loomc_source_options_t options = {};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier = loomc_make_cstring_view("command_values_test.loom");
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.contents =
        loomc_make_byte_span(kDefinitions, std::strlen(kDefinitions));
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(
        Ok(loomc_source_create(&options, loomc_allocator_system(), &source)));
    Source source_owner(source);
    loomc_module_t *definitions = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &definitions, &result)));
    Module definitions_owner(definitions);
    Result result_owner(result);
    ASSERT_TRUE(Succeeded(result));
    ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                         definitions, &module_)));
    ASSERT_TRUE(Ok(module_->Kernel("sink", &sink_)));
    ASSERT_TRUE(Ok(module_->DefineProgram("authored", {LOOM_SCALAR_TYPE_INDEX},
                                          2, &program_)));
  }
  Module Finish() {
    auto root = program_->Ref();
    EXPECT_TRUE(Ok(program_->Finish()));
    loomc_module_t *output = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(module_->LinkAndVerify({root}, &output, &result)));
    Result result_owner(result);
    EXPECT_TRUE(Succeeded(result));
    return Module(output);
  }
  const loom_module_t *Native(loomc_module_t *module) {
    loomc_module_interop_view_t view = {};
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_get_interop_view(
        module, loomc_allocator_system(), &view, &result)));
    Result result_owner(result);
    EXPECT_TRUE(Succeeded(result));
    return view.module;
  }
  const loom_op_t *Find(const loom_module_t *module, const char *name) {
    for (size_t i = 0; i < module->symbols.count; ++i) {
      const auto &symbol = module->symbols.entries[i];
      auto actual = loom_string_table_get(&module->strings, symbol.name_id);
      if (actual.size == std::strlen(name) &&
          std::memcmp(actual.data, name, actual.size) == 0)
        return symbol.defining_op;
    }
    return nullptr;
  }
  std::vector<const loom_op_t *> Operations(const loom_module_t *module) {
    const auto *def = Find(module, "authored");
    const auto *body =
        loom_region_const_entry_block(loom_command_program_def_body(def));
    const loom_op_t *op = nullptr;
    const loom_op_t *serial = nullptr;
    loom_block_for_each_op(body, op) if (loom_command_serial_isa(op)) serial =
        op;
    std::vector<const loom_op_t *> result;
    auto *block =
        loom_region_const_entry_block(loom_command_serial_body(serial));
    loom_block_for_each_op(block, op) result.push_back(op);
    return result;
  }
  Owner<loomc_context_t, loomc_context_release> context_;
  Owner<loomc_workspace_t, loomc_workspace_release> workspace_;
  std::unique_ptr<ModuleBuilder> module_;
  std::unique_ptr<CommandProgramBuilder> program_;
  KernelRef sink_;
};

TEST_F(CommandValuesTest, ConstructsDependentNamedViewAndConfigSizedScratch) {
  CommandValues values(*program_);
  Value width, stride, bytes, scratch, weight;
  ASSERT_TRUE(Ok(values.Config("width", &width)));
  ASSERT_TRUE(Ok(values.Constant(4, LOOM_SCALAR_TYPE_OFFSET, &stride)));
  ASSERT_TRUE(Ok(values.Scale(width, stride, &bytes)));
  ASSERT_TRUE(Ok(values.Allocate(bytes, 256, &scratch)));
  ASSERT_TRUE(
      Ok(values.Parameter(program_->Binding(0), "blk.{}.projection.weight",
                          {program_->Specialization(0)}, LOOM_SCALAR_TYPE_F32,
                          {width, width}, &weight)));
  ASSERT_TRUE(Ok(program_->Invoke(
      sink_, {width}, {width, weight, scratch, program_->Binding(1)})));
  auto module = Finish();
  ASSERT_NE(module, nullptr);
  auto *native = Native(module.get());
  const loom_op_t *parameter = nullptr;
  for (auto *op : Operations(native))
    if (loom_command_parameter_isa(op))
      parameter = op;
  ASSERT_NE(parameter, nullptr);
  auto key = loom_string_table_get(&native->strings,
                                   loom_command_parameter_pattern(parameter));
  EXPECT_EQ(std::string(key.data, key.size), "blk.{}.projection.weight");
  auto type =
      loom_module_value_type(native, loom_command_parameter_result(parameter));
  EXPECT_EQ(loom_type_rank(type), 2);
  EXPECT_EQ(loom_type_element_type(type), LOOM_SCALAR_TYPE_F32);
  EXPECT_TRUE(loom_dim_is_dynamic(type.dims[0]));
  EXPECT_EQ(type.dims[0], type.dims[1]);

  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t options = {};
    options.format = format;
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
        module.get(), &options, loomc_allocator_system(), &source)));
    Source source_owner(source);
    loomc_module_t *loaded = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &loaded, &result)));
    Module loaded_owner(loaded);
    Result result_owner(result);
    ASSERT_TRUE(Succeeded(result));
    ASSERT_NE(Find(Native(loaded), "authored"), nullptr);
  }
}

TEST_F(CommandValuesTest, ComputesSelectionAndAllowsScopedScheduling) {
  CommandValues values(*program_);
  Value one, width, width32, mask, bit, selected, zero, condition, extent,
      bytes, stride;
  ASSERT_TRUE(Ok(values.Constant(1, LOOM_SCALAR_TYPE_I32, &one)));
  ASSERT_TRUE(Ok(values.Config("width", &width)));
  ASSERT_TRUE(Ok(values.Cast(program_->Specialization(0), LOOM_SCALAR_TYPE_I32,
                             &width32)));
  ASSERT_TRUE(Ok(values.Constant(3, LOOM_SCALAR_TYPE_I32, &mask)));
  ASSERT_TRUE(Ok(values.ShiftLeft(one, width32, &bit)));
  ASSERT_TRUE(Ok(values.BitAnd(mask, bit, &selected)));
  ASSERT_TRUE(Ok(values.Constant(0, LOOM_SCALAR_TYPE_I32, &zero)));
  ASSERT_TRUE(Ok(values.NotEqual(selected, zero, &condition)));
  ASSERT_TRUE(Ok(
      values.Select(condition, width, program_->Specialization(0), &extent)));
  ASSERT_TRUE(Ok(values.Constant(4, LOOM_SCALAR_TYPE_OFFSET, &stride)));
  ASSERT_TRUE(Ok(values.Scale(width, stride, &bytes)));
  Value escaped;
  ASSERT_TRUE(Ok(program_->If(condition, [&](CommandProgramBuilder &body) {
    CommandValues nested(body);
    return nested.Constant(1, LOOM_SCALAR_TYPE_INDEX, &escaped);
  })));
  EXPECT_FALSE(Ok(values.Add(escaped, width, &extent)));
  ASSERT_TRUE(Ok(program_->Concurrent([&](CommandProgramBuilder &body) {
    return body.Serial([&](CommandProgramBuilder &inner) {
      return inner.Invoke(sink_, {width},
                          {width, program_->Binding(0), program_->Binding(1),
                           program_->Binding(1)});
    });
  })));
  ASSERT_NE(Finish(), nullptr);
}

TEST_F(CommandValuesTest, PreflightErrorsDoNotMutateOrPoisonTheProgram) {
  CommandValues values(*program_);
  Value extent, result;
  ASSERT_TRUE(Ok(values.Constant(4, LOOM_SCALAR_TYPE_INDEX, &extent)));
  EXPECT_FALSE(Ok(values.Constant(-1, LOOM_SCALAR_TYPE_OFFSET, &result)));
  EXPECT_FALSE(Ok(values.Constant(2, LOOM_SCALAR_TYPE_I1, &result)));
  EXPECT_FALSE(Ok(values.Parameter(program_->Binding(0), "bad.{0}", {extent},
                                   LOOM_SCALAR_TYPE_I32, {extent}, &result)));
  EXPECT_FALSE(Ok(values.Parameter(program_->Binding(0), "missing.{}", {},
                                   LOOM_SCALAR_TYPE_I32, {extent}, &result)));
  EXPECT_FALSE(Ok(values.Allocate(extent, 3, &result)));
  EXPECT_FALSE(Ok(values.Cast(extent, LOOM_SCALAR_TYPE_F32, &result)));
  EXPECT_FALSE(Ok(values.Parameter(program_->Binding(0), "too.many.dims", {},
                                   LOOM_SCALAR_TYPE_I32,
                                   {extent, extent, extent}, &result)));
  Value sum;
  ASSERT_TRUE(Ok(values.Add(extent, extent, &sum)));
  Value product, quotient;
  ASSERT_TRUE(Ok(values.Multiply(sum, extent, &product)));
  ASSERT_TRUE(Ok(values.Divide(product, extent, &quotient)));
  ASSERT_NE(Finish(), nullptr);
}

TEST_F(CommandValuesTest, TypedPreflightErrorsLeaveBothDefinitionsUsable) {
  CommandValues values(*program_);
  std::unique_ptr<CommandProgramBuilder> sibling;
  ASSERT_TRUE(Ok(module_->DefineProgram("sibling", {}, 2, &sibling)));
  Value index, offset, word, floating, invalid;
  ASSERT_TRUE(Ok(values.Constant(4, LOOM_SCALAR_TYPE_INDEX, &index)));
  ASSERT_TRUE(Ok(values.Constant(4, LOOM_SCALAR_TYPE_OFFSET, &offset)));
  ASSERT_TRUE(Ok(values.Constant(4, LOOM_SCALAR_TYPE_I32, &word)));
  ASSERT_TRUE(Ok(values.Config("floating", &floating)));
  size_t before = 0;
  ASSERT_TRUE(
      Ok(program_->InspectNative({}, [&](const auto *native, const auto &) {
        before = Operations(native).size();
        return iree_ok_status();
      })));
  EXPECT_FALSE(Ok(values.Config("missing", &invalid)));
  EXPECT_FALSE(Ok(values.Config("sink", &invalid)));
  EXPECT_FALSE(Ok(values.Cast(floating, LOOM_SCALAR_TYPE_INDEX, &invalid)));
  EXPECT_FALSE(Ok(values.Cast(floating, LOOM_SCALAR_TYPE_OFFSET, &invalid)));
  EXPECT_FALSE(
      Ok(values.Cast(program_->Binding(0), LOOM_SCALAR_TYPE_INDEX, &invalid)));
  EXPECT_FALSE(Ok(values.Add(index, offset, &invalid)));
  EXPECT_FALSE(Ok(values.Multiply(offset, offset, &invalid)));
  EXPECT_FALSE(Ok(values.Divide(offset, offset, &invalid)));
  EXPECT_FALSE(Ok(values.Scale(offset, index, &invalid)));
  EXPECT_FALSE(Ok(values.ShiftLeft(word, index, &invalid)));
  EXPECT_FALSE(Ok(values.BitAnd(word, index, &invalid)));
  EXPECT_FALSE(Ok(values.NotEqual(word, index, &invalid)));
  EXPECT_FALSE(Ok(values.Equal(index, offset, &invalid)));
  EXPECT_FALSE(Ok(values.Select(index, word, word, &invalid)));
  EXPECT_FALSE(Ok(values.Allocate(index, 16, &invalid)));
  EXPECT_FALSE(Ok(values.Parameter(word, "weights", {}, LOOM_SCALAR_TYPE_I32,
                                   {index}, &invalid)));
  EXPECT_FALSE(Ok(values.Parameter(program_->Binding(0), "weights.{}", {word},
                                   LOOM_SCALAR_TYPE_I32, {index}, &invalid)));
  EXPECT_FALSE(Ok(values.Parameter(program_->Binding(0), "weights", {},
                                   LOOM_SCALAR_TYPE_I32, {offset}, &invalid)));
  ASSERT_TRUE(
      Ok(program_->InspectNative({}, [&](const auto *native, const auto &) {
        EXPECT_EQ(Operations(native).size(), before);
        return iree_ok_status();
      })));
  Value scratch, weights;
  ASSERT_TRUE(Ok(values.Allocate(offset, 16, &scratch)));
  ASSERT_TRUE(Ok(values.Parameter(program_->Binding(0), "weights", {},
                                  LOOM_SCALAR_TYPE_I32, {index}, &weights)));
  ASSERT_TRUE(Ok(program_->Invoke(
      sink_, {index}, {index, weights, scratch, program_->Binding(1)})));
  Value sibling_count;
  ASSERT_TRUE(Ok(CommandValues(*sibling).Constant(1, LOOM_SCALAR_TYPE_INDEX,
                                                  &sibling_count)));
  ASSERT_TRUE(Ok(sibling->Invoke(sink_, {sibling_count},
                                 {sibling_count, sibling->Binding(0),
                                  sibling->Binding(1), sibling->Binding(1)})));
  ASSERT_TRUE(Ok(sibling->Finish()));
  ASSERT_NE(Finish(), nullptr);
}
} // namespace
} // namespace hrx::loomify_gguf
