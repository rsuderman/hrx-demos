// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/command/command_program_builder.h"
#include "tools/loomify-gguf/authoring/pipeline/pipeline_builder.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/ops/view/ops.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf {
namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *p) const { Release(p); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
using Context = Owner<loomc_context_t, loomc_context_release>;
using Workspace = Owner<loomc_workspace_t, loomc_workspace_release>;
using Module = Owner<loomc_module_t, loomc_module_release>;
using Source = Owner<loomc_source_t, loomc_source_release>;
using Result = Owner<loomc_result_t, loomc_result_release>;

::testing::AssertionResult Ok(loomc_status_t status) {
  const auto code = loomc_status_code(status);
  const auto message = loomc_status_message(status);
  const std::string copy(message.data ? message.data : "", message.size);
  loomc_status_free(status);
  if (code == LOOMC_STATUS_OK)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << copy;
}
::testing::AssertionResult NativeOk(iree_status_t status) {
  return Ok(loomc_status_from_iree(status));
}
::testing::AssertionResult Succeeded(loomc_result_t *result) {
  if (result && loomc_result_succeeded(result))
    return ::testing::AssertionSuccess();
  std::string message;
  if (result)
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      const auto *diagnostic = loomc_result_diagnostic_at(result, i);
      message.append(diagnostic->message.data, diagnostic->message.size);
      message += '\n';
    }
  return ::testing::AssertionFailure() << message;
}

iree_status_t ChannelType(loom_module_t *module, uint64_t width,
                          loom_type_t *out) {
  loom_type_id_t id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_type_id(
      module,
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, width, 0),
      &id));
  return loom_channel_type_make(module, id, out);
}

// The source shape follows Loom's pipeline roundtrip fixture: explicit pool,
// fixed storage, FIFO channel, independently placed strands and composition.
iree_status_t BuildResident(loom_module_t *module, loom_builder_t *top) {
  loom_type_t channel;
  IREE_RETURN_IF_ERROR(ChannelType(module, loom_dim_pack_static(4), &channel));
  PipelineInterface consumer;
  consumer.name = "consume";
  consumer.run = {channel};
  PipelineRef child;
  IREE_RETURN_IF_ERROR(PipelineBuilder::Define(
      module, top, consumer,
      [](PipelineBody &body) {
        return body.Strand({{0, 3}, {1, 1}, {1, 1}}, [](PipelineBody &strand) {
          ChannelAccess access;
          IREE_RETURN_IF_ERROR(strand.Acquire(strand.Binding(0), &access));
          return strand.Release(access.ownership);
        });
      },
      &child));
  PipelineInterface resident;
  resident.name = "resident";
  resident.kernel_scope = true;
  resident.is_public = true;
  resident.target = "array";
  PipelineRef parent;
  return PipelineBuilder::Define(
      module, top, resident,
      [&](PipelineBody &body) {
        loom_value_id_t pool, bytes, storage, offset, slots, capacity, fifo;
        IREE_RETURN_IF_ERROR(body.MemoryPool(
            LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, {0, 2}, &pool));
        IREE_RETURN_IF_ERROR(body.OffsetConstant(32, &bytes));
        IREE_RETURN_IF_ERROR(body.Allocate(
            pool, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 16, bytes, &storage));
        IREE_RETURN_IF_ERROR(body.OffsetConstant(0, &offset));
        IREE_RETURN_IF_ERROR(
            body.View(storage, offset,
                      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                          loom_dim_pack_static(2),
                                          loom_dim_pack_static(4), 0),
                      &slots));
        IREE_RETURN_IF_ERROR(body.IndexConstant(2, &capacity));
        IREE_RETURN_IF_ERROR(body.BindFifo(
            slots, capacity,
            loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                                loom_dim_pack_static(4), 0),
            &fifo));
        IREE_RETURN_IF_ERROR(
            body.Strand({{0, 2}, {1, 1}, {1, 1}}, [&](PipelineBody &strand) {
              ChannelAccess access;
              IREE_RETURN_IF_ERROR(strand.Reserve(fifo, &access));
              loom_op_t *constant = nullptr;
              IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
                  strand.Native(), loom_attr_i64(7),
                  loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN,
                  &constant));
              for (int64_t i = 0; i < 4; ++i) {
                loom_op_t *store = nullptr;
                IREE_RETURN_IF_ERROR(loom_view_store_build(
                    strand.Native(), 0, 0,
                    loom_scalar_constant_result(constant), access.payload,
                    nullptr, 0, &i, 1, 0, 0, LOOM_LOCATION_UNKNOWN, &store));
              }
              return strand.Publish(access.ownership);
            }));
        return body.Compose(child, {}, {fifo});
      },
      &parent);
}

class PipelineBuilderTest : public ::testing::Test {
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
    options.identifier = loomc_make_cstring_view("pipeline_seed.loom");
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    constexpr char text[] = "target.decl @array\n";
    options.contents = loomc_make_byte_span(text, sizeof(text) - 1);
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(
        Ok(loomc_source_create(&options, loomc_allocator_system(), &source)));
    Source source_owner(source);
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &module, &result)));
    Result result_owner(result);
    ASSERT_TRUE(Succeeded(result));
    module_.reset(module);
    native_ = loomc_module_get_mutable_interop_view(module).module;
    ASSERT_NE(native_, nullptr);
    loom_builder_initialize(native_, &native_->arena,
                            loom_module_block(native_), &top_);
  }
  void Verify() {
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_verify(module_.get(), nullptr,
                                       loomc_allocator_system(), &result)));
    Result owner(result);
    ASSERT_TRUE(Succeeded(result));
  }
  Module Parse(const char *text) {
    loomc_source_options_t options = {};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    const auto identifier =
        "pipeline_contract_" + std::to_string(parse_count_++) + ".loom";
    options.identifier = loomc_make_cstring_view(identifier.c_str());
    options.storage = LOOMC_SOURCE_STORAGE_COPY;
    options.contents = loomc_make_byte_span(text, std::strlen(text));
    loomc_source_t *source = nullptr;
    EXPECT_TRUE(
        Ok(loomc_source_create(&options, loomc_allocator_system(), &source)));
    Source source_owner(source);
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_deserialize_from_source(
        context_.get(), workspace_.get(), source, nullptr,
        loomc_allocator_system(), &module, &result)));
    Result result_owner(result);
    EXPECT_TRUE(Succeeded(result));
    return Module(module);
  }
  const loom_op_t *Symbol(const char *name) {
    for (size_t i = 0; i < native_->symbols.count; ++i) {
      const auto &symbol = native_->symbols.entries[i];
      auto text = loom_string_table_get(&native_->strings, symbol.name_id);
      if (iree_string_view_equal(text, iree_make_cstring_view(name)))
        return symbol.defining_op;
    }
    return nullptr;
  }
  Context context_;
  Workspace workspace_;
  Module module_;
  loom_module_t *native_ = nullptr;
  loom_builder_t top_ = {};
  size_t parse_count_ = 0;
};

TEST_F(PipelineBuilderTest,
       ConstructsStorageStrandsChannelOwnershipAndComposition) {
  ASSERT_TRUE(NativeOk(BuildResident(native_, &top_)));
  Verify();
  auto *parent = Symbol("resident");
  auto *child = Symbol("consume");
  ASSERT_NE(parent, nullptr);
  ASSERT_NE(child, nullptr);
  ASSERT_TRUE(loom_pipeline_def_isa(parent));
  EXPECT_EQ(loom_pipeline_def_scope(parent), LOOM_PIPELINE_SCOPE_KERNEL);
  const auto *body =
      loom_region_const_entry_block(loom_pipeline_def_body(parent));
  const loom_op_t *op = nullptr;
  const loom_op_t *compose = nullptr;
  const loom_op_t *bind = nullptr;
  const loom_op_t *producer = nullptr;
  loom_block_for_each_op(body, op) {
    if (loom_pipeline_compose_isa(op))
      compose = op;
    if (loom_channel_bind_isa(op))
      bind = op;
    if (loom_pipeline_strand_isa(op))
      producer = op;
  }
  ASSERT_NE(compose, nullptr);
  ASSERT_NE(bind, nullptr);
  ASSERT_NE(producer, nullptr);
  auto bindings = loom_pipeline_compose_bindings(compose);
  ASSERT_EQ(bindings.count, 1u);
  EXPECT_EQ(bindings.values[0], loom_channel_bind_result(bind));
  EXPECT_EQ(loom_pipeline_compose_callee(compose).symbol_id,
            loom_pipeline_def_callee(child).symbol_id);
  const auto *producer_body =
      loom_region_const_entry_block(loom_pipeline_strand_body(producer));
  loom_value_id_t write = LOOM_VALUE_ID_INVALID;
  loom_block_for_each_op(producer_body, op) {
    if (loom_channel_reserve_isa(op))
      write = loom_channel_reserve_write(op);
    if (loom_channel_publish_isa(op)) {
      EXPECT_EQ(loom_channel_publish_write(op), write);
    }
  }
  EXPECT_NE(write, LOOM_VALUE_ID_INVALID);
}

TEST_F(PipelineBuilderTest, RoundtripsBothFormatsInFreshContext) {
  ASSERT_TRUE(NativeOk(BuildResident(native_, &top_)));
  Verify();
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t options = {};
    options.format = format;
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
        module_.get(), &options, loomc_allocator_system(), &source)));
    Source source_owner(source);
    if (const auto *directory = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR")) {
      const auto bytes = loomc_source_contents(source);
      std::ofstream artifact(std::string(directory) +
                                 (format == LOOMC_SOURCE_FORMAT_TEXT
                                      ? "/resident.loom"
                                      : "/resident.loombc"),
                             std::ios::binary);
      artifact.write(reinterpret_cast<const char *>(bytes.data),
                     bytes.data_length);
      ASSERT_TRUE(artifact.good());
    }
    loomc_context_t *fresh = nullptr;
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &fresh)));
    Context fresh_owner(fresh);
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    Workspace workspace_owner(workspace);
    loomc_module_t *reloaded = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        fresh, workspace, source, nullptr, loomc_allocator_system(), &reloaded,
        &result)));
    Module reloaded_owner(reloaded);
    Result result_owner(result);
    EXPECT_TRUE(Succeeded(result));
  }
}

TEST_F(PipelineBuilderTest, ComposesDependentChannelTypesBySpecialization) {
  PipelineInterface consumer;
  consumer.name = "dependent_consumer";
  consumer.specializations = {loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)};
  consumer.dependent_run = [](loom_module_t *module,
                              const std::vector<loom_value_id_t> &args,
                              std::vector<loom_type_t> *out) {
    loom_type_t channel;
    IREE_RETURN_IF_ERROR(
        ChannelType(module, loom_dim_pack_dynamic(args[0]), &channel));
    *out = {channel};
    return iree_ok_status();
  };
  PipelineRef child;
  ASSERT_TRUE(NativeOk(PipelineBuilder::Define(
      native_, &top_, consumer,
      [](PipelineBody &body) {
        return body.Strand({{0, 2}, {1, 1}, {1, 1}}, [](PipelineBody &strand) {
          ChannelAccess access;
          IREE_RETURN_IF_ERROR(strand.Acquire(strand.Binding(0), &access));
          return strand.Release(access.ownership);
        });
      },
      &child)));
  PipelineInterface parent = consumer;
  parent.name = "dependent_parent";
  parent.is_public = true;
  PipelineRef outer;
  ASSERT_TRUE(NativeOk(PipelineBuilder::Define(
      native_, &top_, parent,
      [&](PipelineBody &body) {
        return body.Compose(child, {body.Specialization(0)}, {body.Binding(0)});
      },
      &outer)));
  Verify();
}

TEST_F(PipelineBuilderTest,
       RejectsMalformedDomainsForeignReferencesAndNestedStrands) {
  PipelineInterface interface;
  interface.name = "invalid_attempts";
  PipelineRef ref;
  ASSERT_TRUE(NativeOk(PipelineBuilder::Define(
      native_, &top_, interface,
      [&](PipelineBody &body) {
        auto status = body.Strand({{0, 2}, {1}, {1, 1}}, [](PipelineBody &) {
          return iree_ok_status();
        });
        EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
        iree_status_ignore(status);
        status = body.Compose({}, {}, {});
        EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
        iree_status_ignore(status);
        return body.Strand({{0, 2}, {1, 1}, {1, 1}}, [](PipelineBody &strand) {
          auto status =
              strand.Strand({{0, 2}, {1, 1}, {1, 1}},
                            [](PipelineBody &) { return iree_ok_status(); });
          EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
          iree_status_ignore(status);
          return iree_ok_status();
        });
      },
      &ref)));
  Verify();
}

TEST_F(PipelineBuilderTest, ModuleOwnerEmitsPipelineRootWithoutCommandWrapper) {
  std::unique_ptr<ModuleBuilder> owner;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                       module_.get(), &owner)));
  ASSERT_TRUE(Ok(owner->BuildNative(BuildResident)));
  KernelPipelineRef pipeline;
  auto generic_status = owner->KernelPipeline("consume", &pipeline);
  EXPECT_EQ(loomc_status_code(generic_status), LOOMC_STATUS_INVALID_ARGUMENT);
  loomc_status_free(generic_status);
  ASSERT_TRUE(Ok(owner->KernelPipeline("resident", &pipeline)));
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(owner->LinkAndVerify({pipeline}, &output, &result)));
  Module output_owner(output);
  Result result_owner(result);
  ASSERT_TRUE(Succeeded(result));
  ASSERT_NE(output, nullptr);
}

TEST_F(PipelineBuilderTest, CommandDirectlyInvokesAuthoredKernelPipeline) {
  std::unique_ptr<ModuleBuilder> owner;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                       module_.get(), &owner)));
  ASSERT_TRUE(Ok(owner->BuildNative(BuildResident)));
  KernelPipelineRef pipeline;
  ASSERT_TRUE(Ok(owner->KernelPipeline("resident", &pipeline)));
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(owner->DefineProgram("invoke_resident", {}, 0, &program)));
  ASSERT_TRUE(Ok(program->Invoke(pipeline, {}, {})));
  const auto root = program->Ref();
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(owner->LinkAndVerify({root}, &output, &result)));
  Module output_owner(output);
  Result result_owner(result);
  ASSERT_TRUE(Succeeded(result));
  loomc_module_serialize_options_t options = {};
  options.format = LOOMC_SOURCE_FORMAT_TEXT;
  loomc_source_t *source = nullptr;
  ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
      output, &options, loomc_allocator_system(), &source)));
  Source source_owner(source);
  const auto text = loomc_source_contents(source);
  const std::string ir(reinterpret_cast<const char *>(text.data),
                       text.data_length);
  EXPECT_NE(ir.find("kernel.launch @resident"), std::string::npos);
  EXPECT_NE(ir.find("pipeline.def<kernel>"), std::string::npos);
  EXPECT_EQ(ir.find("command.program.launch"), std::string::npos);
}

TEST_F(PipelineBuilderTest, CommandLaunchSubstitutesDependentRunViewType) {
  std::unique_ptr<ModuleBuilder> owner;
  ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                       module_.get(), &owner)));
  ASSERT_TRUE(
      Ok(owner->BuildNative([](loom_module_t *module, loom_builder_t *top) {
        PipelineInterface interface;
        interface.name = "dependent_entry";
        interface.kernel_scope = true;
        interface.target = "array";
        interface.specializations = {loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)};
        interface.dependent_run = [](loom_module_t *,
                                     const std::vector<loom_value_id_t> &args,
                                     std::vector<loom_type_t> *out) {
          *out = {loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                      loom_dim_pack_dynamic(args[0]), 0)};
          return iree_ok_status();
        };
        PipelineRef ref;
        return PipelineBuilder::Define(
            module, top, interface,
            [](PipelineBody &) { return iree_ok_status(); }, &ref);
      })));
  KernelPipelineRef pipeline;
  ASSERT_TRUE(Ok(owner->KernelPipeline("dependent_entry", &pipeline)));
  std::unique_ptr<CommandProgramBuilder> program;
  ASSERT_TRUE(Ok(owner->DefineProgram("dependent_driver",
                                      {LOOM_SCALAR_TYPE_INDEX}, 1, &program)));
  const auto width = program->Specialization(0);
  std::vector<Value> views;
  ASSERT_TRUE(Ok(program->BuildNative(
      {width, program->Binding(0)},
      [](loom_module_t *, loom_builder_t *builder,
         const std::vector<loom_value_id_t> &inputs,
         std::vector<loom_value_id_t> *out) {
        loom_op_t *offset = nullptr;
        IREE_RETURN_IF_ERROR(
            loom_index_constant_build(builder, loom_attr_i64(0),
                                      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                      LOOM_LOCATION_UNKNOWN, &offset));
        loom_op_t *view = nullptr;
        IREE_RETURN_IF_ERROR(loom_buffer_view_build(
            builder, inputs[1], loom_index_constant_result(offset),
            loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                loom_dim_pack_dynamic(inputs[0]), 0),
            LOOM_LOCATION_UNKNOWN, &view));
        *out = {loom_buffer_view_result(view)};
        return iree_ok_status();
      },
      &views)));
  ASSERT_EQ(views.size(), 1u);
  ASSERT_TRUE(Ok(program->Invoke(pipeline, {width}, {views[0]})));
  const auto root = program->Ref();
  ASSERT_TRUE(Ok(program->Finish()));
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  ASSERT_TRUE(Ok(owner->LinkAndVerify({root}, &output, &result)));
  Module output_owner(output);
  Result result_owner(result);
  ASSERT_TRUE(Succeeded(result));
}

TEST_F(PipelineBuilderTest,
       PipelineDeclarationRequiresExplicitCompatibleProvider) {
  auto declarations = Parse(R"(
target.decl @array
pipeline.decl<kernel> target(@array) @provided(%width: index)
  run(%input: view<[%width]xi32>)
)");
  auto provider = Parse(R"(
target.decl @array
pipeline.def<kernel> public target(@array) @provided(%width: index)
  run(%input: view<[%width]xi32>) {
}
)");
  ASSERT_NE(declarations, nullptr);
  ASSERT_NE(provider, nullptr);
  for (bool supply_provider : {false, true}) {
    std::unique_ptr<ModuleBuilder> owner;
    ASSERT_TRUE(Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                         declarations.get(), &owner)));
    if (supply_provider)
      ASSERT_TRUE(Ok(owner->AddProvider(provider.get())));
    KernelPipelineRef pipeline;
    ASSERT_TRUE(Ok(owner->KernelPipeline("provided", &pipeline)));
    std::unique_ptr<CommandProgramBuilder> program;
    ASSERT_TRUE(Ok(owner->DefineProgram(
        "external_pipeline_driver", {LOOM_SCALAR_TYPE_INDEX}, 1, &program)));
    const auto width = program->Specialization(0);
    std::vector<Value> views;
    ASSERT_TRUE(Ok(program->BuildNative(
        {width, program->Binding(0)},
        [](loom_module_t *, loom_builder_t *builder,
           const std::vector<loom_value_id_t> &inputs,
           std::vector<loom_value_id_t> *out) {
          loom_op_t *offset = nullptr;
          IREE_RETURN_IF_ERROR(loom_index_constant_build(
              builder, loom_attr_i64(0),
              loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN,
              &offset));
          loom_op_t *view = nullptr;
          IREE_RETURN_IF_ERROR(loom_buffer_view_build(
              builder, inputs[1], loom_index_constant_result(offset),
              loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I32,
                                  loom_dim_pack_dynamic(inputs[0]), 0),
              LOOM_LOCATION_UNKNOWN, &view));
          *out = {loom_buffer_view_result(view)};
          return iree_ok_status();
        },
        &views)));
    ASSERT_TRUE(Ok(program->Invoke(pipeline, {width}, {views[0]})));
    const auto root = program->Ref();
    ASSERT_TRUE(Ok(program->Finish()));
    loomc_module_t *output = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(owner->LinkAndVerify({root}, &output, &result)));
    Module output_owner(output);
    Result result_owner(result);
    if (supply_provider) {
      ASSERT_TRUE(Succeeded(result));
      EXPECT_NE(output, nullptr);
    } else {
      ASSERT_NE(result, nullptr);
      EXPECT_FALSE(loomc_result_succeeded(result));
      EXPECT_EQ(output, nullptr);
    }
  }
}

} // namespace
} // namespace hrx::loomify_gguf
