// Copyright 2026 AMD ROCm™ Software
// SPDX-License-Identifier: MIT

#include "tools/loomify-gguf/authoring/command/command_program_builder.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "loom/ops/command/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loomc/interop.h"
#include "loomc/link.h"
#include "gtest/gtest.h"

namespace hrx::loomify_gguf {
namespace {
template <typename T, void (*Release)(T *)> struct Deleter {
  void operator()(T *value) const { Release(value); }
};
template <typename T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, Deleter<T, Release>>;
using Context = Owner<loomc_context_t, loomc_context_release>;
using Workspace = Owner<loomc_workspace_t, loomc_workspace_release>;
using Module = Owner<loomc_module_t, loomc_module_release>;
using Source = Owner<loomc_source_t, loomc_source_release>;
using Result = Owner<loomc_result_t, loomc_result_release>;

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
  std::string message;
  if (result) {
    for (size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
      auto diagnostic = loomc_result_diagnostic_at(result, i);
      message.append(diagnostic->message.data, diagnostic->message.size);
      message += '\n';
    }
  }
  return ::testing::AssertionFailure() << message;
}

constexpr char kDefinitions[] = R"(
kernel.def @first(%count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%count: index, %input: buffer, %output: buffer) {
  kernel.return
}
kernel.def @second(%count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%count: index, %input: buffer, %output: buffer) {
  kernel.return
}
kernel.decl @external(%count: index) launch(%count: index, %input: buffer, %output: buffer)
func.def @ordinary() {
  func.return
}
kernel.entry.decl @resolved(%count: index, %input: buffer, %output: buffer)
)";

class CommandBuilderTest : public ::testing::Test {
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
    definitions_ = Parse(kDefinitions);
    ASSERT_NE(definitions_, nullptr);
  }
  Module Parse(const char *text) {
    loomc_source_options_t options = {};
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    auto name = "fixture_" + std::to_string(source_count_++) + ".loom";
    options.identifier = loomc_make_cstring_view(name.c_str());
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
  std::unique_ptr<ModuleBuilder> Builder(const loomc_module_t *seed = nullptr) {
    std::unique_ptr<ModuleBuilder> out;
    EXPECT_TRUE(
        Ok(ModuleBuilder::Create(context_.get(), workspace_.get(),
                                 seed ? seed : definitions_.get(), &out)));
    return out;
  }
  std::unique_ptr<CommandProgramBuilder>
  Program(ModuleBuilder &owner, const char *name = "authored") {
    std::unique_ptr<CommandProgramBuilder> out;
    EXPECT_TRUE(
        Ok(owner.DefineProgram(name, {LOOM_SCALAR_TYPE_INDEX}, 3, &out)));
    return out;
  }
  Module Finish(ModuleBuilder &owner, std::vector<RootRef> roots) {
    loomc_module_t *module = nullptr;
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(owner.LinkAndVerify(roots, &module, &result)));
    Result result_owner(result);
    EXPECT_TRUE(Succeeded(result));
    return Module(module);
  }
  const loom_module_t *Native(loomc_module_t *module) {
    loomc_module_interop_view_t view = {};
    loomc_result_t *result = nullptr;
    EXPECT_TRUE(Ok(loomc_module_get_interop_view(
        module, loomc_allocator_system(), &view, &result)));
    Result owner(result);
    EXPECT_TRUE(Succeeded(result));
    return view.module;
  }
  const loom_op_t *Symbol(const loom_module_t *module, const char *name) {
    for (size_t i = 0; i < module->symbols.count; ++i) {
      auto s = module->symbols.entries[i];
      auto n = loom_string_table_get(&module->strings, s.name_id);
      if (n.size == std::strlen(name) && std::memcmp(n.data, name, n.size) == 0)
        return s.defining_op;
    }
    return nullptr;
  }
  void Pair(ModuleBuilder &owner, CommandProgramBuilder &body) {
    KernelRef first, second;
    ASSERT_TRUE(Ok(owner.Kernel("first", &first)));
    ASSERT_TRUE(Ok(owner.Kernel("second", &second)));
    auto n = body.Specialization(0);
    ASSERT_TRUE(
        Ok(body.Invoke(first, {n}, {n, body.Binding(0), body.Binding(1)})));
    ASSERT_TRUE(
        Ok(body.Invoke(second, {n}, {n, body.Binding(1), body.Binding(2)})));
  }
  std::vector<const loom_op_t *> Calls(const loom_op_t *program) {
    std::vector<const loom_op_t *> calls;
    const loom_op_t *op;
    loom_block_for_each_op(
        loom_region_const_entry_block(loom_command_program_def_body(program)),
        op) {
      if (!loom_command_serial_isa(op))
        continue;
      const loom_op_t *call;
      loom_block_for_each_op(
          loom_region_const_entry_block(loom_command_serial_body(op)), call) {
        if (loom_kernel_launch_isa(call) ||
            loom_command_program_launch_isa(call))
          calls.push_back(call);
      }
    }
    return calls;
  }
  Context context_;
  Workspace workspace_;
  Module definitions_;
  size_t source_count_ = 0;
};

TEST_F(CommandBuilderTest, ModuleReferencesReuseKernelsAcrossDefinitions) {
  auto owner = Builder();
  KernelRef first;
  ASSERT_TRUE(Ok(owner->Kernel("first", &first)));
  auto a = Program(*owner, "a");
  auto b = Program(*owner, "b");
  for (auto *body : {a.get(), b.get()}) {
    auto n = body->Specialization(0);
    ASSERT_TRUE(
        Ok(body->Invoke(first, {n}, {n, body->Binding(0), body->Binding(2)})));
    ASSERT_TRUE(Ok(body->Finish()));
  }
  EXPECT_EQ(Symbol(Native(definitions_.get()), "a"), nullptr);
  definitions_.reset();
  auto module = Finish(*owner, {a->Ref(), b->Ref()});
  ASSERT_NE(module, nullptr);
  auto native = Native(module.get());
  for (auto name : {"a", "b"}) {
    auto calls = Calls(Symbol(native, name));
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(
        native->symbols.entries[loom_kernel_launch_callee(calls[0]).symbol_id]
            .defining_op,
        Symbol(native, "first"));
  }
}

TEST_F(CommandBuilderTest, PreservesOrderWiringAndBothSerializationFormats) {
  auto owner = Builder();
  auto body = Program(*owner);
  Pair(*owner, *body);
  ASSERT_TRUE(Ok(body->Finish()));
  auto module = Finish(*owner, {body->Ref()});
  ASSERT_NE(module, nullptr);
  for (auto format : {LOOMC_SOURCE_FORMAT_TEXT, LOOMC_SOURCE_FORMAT_BYTECODE}) {
    loomc_module_serialize_options_t options = {};
    options.format = format;
    loomc_source_t *source = nullptr;
    ASSERT_TRUE(Ok(loomc_module_serialize_to_source(
        module.get(), &options, loomc_allocator_system(), &source)));
    Source source_owner(source);
    loomc_context_t *context = nullptr;
    loomc_workspace_t *workspace = nullptr;
    ASSERT_TRUE(
        Ok(loomc_context_create(nullptr, loomc_allocator_system(), &context)));
    Context context_owner(context);
    ASSERT_TRUE(Ok(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace)));
    Workspace workspace_owner(workspace);
    loomc_module_t *reloaded = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(loomc_module_deserialize_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(),
        &reloaded, &result)));
    Module reloaded_owner(reloaded);
    Result result_owner(result);
    ASSERT_TRUE(Succeeded(result));
    auto native = Native(reloaded);
    auto calls = Calls(Symbol(native, "authored"));
    ASSERT_EQ(calls.size(), 2u);
    EXPECT_EQ(
        native->symbols.entries[loom_kernel_launch_callee(calls[0]).symbol_id]
            .defining_op,
        Symbol(native, "first"));
    EXPECT_EQ(
        native->symbols.entries[loom_kernel_launch_callee(calls[1]).symbol_id]
            .defining_op,
        Symbol(native, "second"));
    EXPECT_EQ(loom_kernel_launch_arguments(calls[0]).values[2],
              loom_kernel_launch_arguments(calls[1]).values[1]);
  }
}

TEST_F(CommandBuilderTest, PreflightErrorsRecoverAndNeverAppendInvalidCalls) {
  auto owner = Builder(), other = Builder();
  auto body = Program(*owner), sibling = Program(*owner, "sibling"),
       foreign_body = Program(*other);
  KernelRef kernel, foreign;
  ASSERT_TRUE(Ok(owner->Kernel("first", &kernel)));
  ASSERT_TRUE(Ok(other->Kernel("first", &foreign)));
  auto n = body->Specialization(0);
  auto args = std::vector<Value>{n, body->Binding(0), body->Binding(2)};
  EXPECT_EQ(loomc_status_consume_code(body->Invoke(foreign, {n}, args)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(body->Invoke(kernel, {}, args)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(
      loomc_status_consume_code(body->Invoke(kernel, {body->Binding(0)}, args)),
      LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(
                body->Invoke(kernel, {sibling->Specialization(0)}, args)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(
                body->Invoke(kernel, {foreign_body->Specialization(0)}, args)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(
                body->Invoke(kernel, {body->Specialization(99)}, args)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(body->Invoke(kernel, {n}, {n})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  ASSERT_TRUE(Ok(body->Invoke(kernel, {n}, args)));
  ASSERT_TRUE(Ok(body->Invoke(kernel, {n}, args)));
  ASSERT_TRUE(Ok(body->Finish()));
  ASSERT_TRUE(Ok(sibling->Finish()));
  auto module = Finish(*owner, {body->Ref()});
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(Calls(Symbol(Native(module.get()), "authored")).size(), 2u);
}

TEST_F(CommandBuilderTest,
       DirectDeclaredKernelUsesExplicitProviderAndStrictClosure) {
  for (int scenario = 0; scenario < 4; ++scenario) {
    auto empty = Parse("");
    auto owner = Builder(empty.get());
    KernelRef external;
    auto index = owner->Scalar(LOOM_SCALAR_TYPE_INDEX);
    auto buffer = owner->Buffer();
    ASSERT_TRUE(Ok(owner->DeclareKernel("external", {index},
                                        {index, buffer, buffer}, &external)));
    auto body = Program(*owner);
    auto n = body->Specialization(0);
    ASSERT_TRUE(Ok(
        body->Invoke(external, {n}, {n, body->Binding(0), body->Binding(2)})));
    ASSERT_TRUE(Ok(body->Finish()));
    auto provider = Parse(scenario == 2 ? R"(
kernel.def @external(%count: i32) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%count: index, %input: buffer, %output: buffer) { kernel.return }
)"
                                        : R"(
kernel.def @external(%count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%count: index, %input: buffer, %output: buffer) { kernel.return }
)");
    ASSERT_NE(provider, nullptr);
    if (scenario) {
      ASSERT_TRUE(Ok(owner->AddProvider(provider.get())));
    }
    if (scenario == 3) {
      ASSERT_TRUE(Ok(owner->AddProvider(provider.get())));
    }
    provider.reset(); // Owner snapshotted it; no borrowed lifetime remains.
    loomc_module_t *linked = nullptr;
    loomc_result_t *result = nullptr;
    ASSERT_TRUE(Ok(owner->LinkAndVerify({body->Ref()}, &linked, &result)));
    Module module(linked);
    Result result_owner(result);
    ASSERT_NE(result, nullptr);
    if (scenario == 1) {
      ASSERT_TRUE(Succeeded(result));
      ASSERT_NE(module, nullptr);
      EXPECT_TRUE(
          loom_kernel_def_isa(Symbol(Native(module.get()), "external")));
    } else {
      EXPECT_FALSE(loomc_result_succeeded(result));
      EXPECT_GT(loomc_result_diagnostic_count(result), 0u);
      EXPECT_EQ(linked, nullptr);
    }
  }
}

TEST_F(CommandBuilderTest, UnfinishedDefinitionsBlockFinalizationRecoverably) {
  auto owner = Builder();
  auto body = Program(*owner);
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  EXPECT_EQ(loomc_status_consume_code(
                owner->LinkAndVerify({body->Ref()}, &output, &result)),
            LOOMC_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(result, nullptr);
  Pair(*owner, *body);
  ASSERT_TRUE(Ok(body->Finish()));
  EXPECT_EQ(loomc_status_consume_code(body->Finish()),
            LOOMC_STATUS_FAILED_PRECONDITION);
  EXPECT_NE(Finish(*owner, {body->Ref()}), nullptr);
}

TEST_F(CommandBuilderTest,
       NativeMutationFailureInvalidatesAllBodiesAndPreventsOutput) {
  auto owner = Builder();
  auto a = Program(*owner, "a"), b = Program(*owner, "b");
  auto root = a->Ref();
  auto status = owner->BuildNative([](loom_module_t *module, loom_builder_t *) {
    loom_string_id_t name;
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, IREE_SV("partial"), &name));
    loom_symbol_id_t symbol;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name, &symbol));
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected native construction failure");
  });
  EXPECT_EQ(loomc_status_consume_code(status), LOOMC_STATUS_RESOURCE_EXHAUSTED);
  Value value;
  EXPECT_EQ(loomc_status_consume_code(a->IndexConstant(7, &value)),
            LOOMC_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(loomc_status_consume_code(b->Finish()),
            LOOMC_STATUS_FAILED_PRECONDITION);
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  EXPECT_EQ(
      loomc_status_consume_code(owner->LinkAndVerify({root}, &output, &result)),
      LOOMC_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(result, nullptr);
}

TEST_F(CommandBuilderTest,
       OwnerDestructionAndFinalizationInvalidateOutstandingBodies) {
  auto owner = Builder();
  auto body = Program(*owner);
  KernelRef stale;
  ASSERT_TRUE(Ok(owner->Kernel("first", &stale)));
  owner.reset();
  EXPECT_EQ(loomc_status_consume_code(body->Finish()),
            LOOMC_STATUS_FAILED_PRECONDITION);
  auto next = Builder();
  auto current = Program(*next);
  auto n = current->Specialization(0);
  EXPECT_EQ(loomc_status_consume_code(current->Invoke(
                stale, {n}, {n, current->Binding(0), current->Binding(2)})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  Pair(*next, *current);
  ASSERT_TRUE(Ok(current->Finish()));
  auto root = current->Ref();
  ASSERT_NE(Finish(*next, {root}), nullptr);
  KernelRef k;
  EXPECT_EQ(loomc_status_consume_code(next->Kernel("first", &k)),
            LOOMC_STATUS_FAILED_PRECONDITION);
  EXPECT_EQ(loomc_status_consume_code(current->IndexConstant(1, &n)),
            LOOMC_STATUS_FAILED_PRECONDITION);
}

TEST_F(CommandBuilderTest,
       ReusesChildReferencesWithoutCloningAndRejectsSelfRecursion) {
  auto owner = Builder();
  auto child = Program(*owner, "child");
  Pair(*owner, *child);
  ASSERT_TRUE(Ok(child->Finish()));
  auto parent = Program(*owner, "parent");
  auto n = parent->Specialization(0);
  EXPECT_EQ(loomc_status_consume_code(parent->Call(
                parent->Ref(), {n},
                {parent->Binding(0), parent->Binding(1), parent->Binding(2)})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  ASSERT_TRUE(Ok(parent->Call(
      child->Ref(), {n},
      {parent->Binding(0), parent->Binding(1), parent->Binding(2)})));
  ASSERT_TRUE(Ok(parent->Finish()));
  auto module = Finish(*owner, {parent->Ref()});
  ASSERT_NE(module, nullptr);
  auto native = Native(module.get());
  auto calls = Calls(Symbol(native, "parent"));
  ASSERT_EQ(calls.size(), 1u);
  EXPECT_TRUE(loom_command_program_launch_isa(calls[0]));
  EXPECT_EQ(native->symbols
                .entries[loom_command_program_launch_callee(calls[0]).symbol_id]
                .defining_op,
            Symbol(native, "child"));
}

TEST_F(CommandBuilderTest,
       RejectsNamesKindsTypesAndForeignRootsBeforeMutation) {
  auto owner = Builder(), other = Builder();
  KernelRef kernel;
  ProgramRef program;
  EXPECT_EQ(loomc_status_consume_code(owner->Kernel("ordinary", &kernel)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(owner->Kernel("resolved", &kernel)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(owner->Kernel("absent", &kernel)),
            LOOMC_STATUS_NOT_FOUND);
  EXPECT_EQ(loomc_status_consume_code(owner->Program("first", &program)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  std::unique_ptr<CommandProgramBuilder> body;
  EXPECT_EQ(
      loomc_status_consume_code(owner->DefineProgram("first", {}, 0, &body)),
      LOOMC_STATUS_ALREADY_EXISTS);
  EXPECT_EQ(loomc_status_consume_code(owner->DefineProgram("", {}, 0, &body)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(
                owner->DefineProgram("x", {LOOM_SCALAR_TYPE_NONE}, 0, &body)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(owner->DefineProgram(
                "x", {}, std::numeric_limits<size_t>::max(), &body)),
            LOOMC_STATUS_RESOURCE_EXHAUSTED);
  EXPECT_EQ(
      loomc_status_consume_code(owner->DeclareKernel(
          "external2", {other->Scalar(LOOM_SCALAR_TYPE_INDEX)}, {}, &kernel)),
      LOOMC_STATUS_INVALID_ARGUMENT);
  auto own_body = Program(*owner);
  auto foreign_body = Program(*other);
  ASSERT_TRUE(Ok(own_body->Finish()));
  loomc_module_t *output = nullptr;
  loomc_result_t *result = nullptr;
  EXPECT_EQ(loomc_status_consume_code(
                owner->LinkAndVerify({foreign_body->Ref()}, &output, &result)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_NE(Finish(*owner, {own_body->Ref()}), nullptr);
}

TEST_F(CommandBuilderTest,
       ScopedRegionsAllowCapturesButRejectEscapedNestedValues) {
  auto owner = Builder();
  auto body = Program(*owner);
  KernelRef first;
  ASSERT_TRUE(Ok(owner->Kernel("first", &first)));
  auto outer = body->Specialization(0);
  Value escaped;
  ASSERT_TRUE(Ok(body->Concurrent([&](CommandProgramBuilder &nested) {
    auto status = nested.IndexConstant(7, &escaped);
    if (!loomc_status_is_ok(status))
      return status;
    return nested.Invoke(first, {outer},
                         {outer, nested.Binding(0), nested.Binding(2)});
  })));
  EXPECT_EQ(loomc_status_consume_code(body->Invoke(
                first, {escaped}, {outer, body->Binding(0), body->Binding(2)})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  ASSERT_TRUE(Ok(body->Finish()));
  EXPECT_NE(Finish(*owner, {body->Ref()}), nullptr);
}

TEST_F(CommandBuilderTest,
       DynamicDiagnosticsOwnTheirMessageUntilStatusRelease) {
  auto owner = Builder();
  KernelRef missing;
  const std::string name(256, 'x');
  const std::string expected = "symbol not found: " + name;
  auto status = owner->Kernel(name, &missing);
  ASSERT_EQ(loomc_status_code(status), LOOMC_STATUS_NOT_FOUND);
  // The helper's temporary concatenation has been destroyed. Reuse its heap
  // size class before reading the diagnostic; retaining a borrowed c_str is
  // invalid even though a short immediate diagnostic read may appear to work.
  std::vector<std::string> pressure(512, std::string(expected.size(), '!'));
  auto message = loomc_status_message(status);
  EXPECT_EQ(std::string(message.data, message.size), expected);
  EXPECT_EQ(pressure.back().front(), '!');
  loomc_status_free(status);
}

TEST_F(CommandBuilderTest, NativeInspectionFailuresRemainRecoverable) {
  auto owner = Builder();
  auto body = Program(*owner);
  auto sibling = Program(*owner, "sibling");
  bool inspected = false;
  auto expect_index = [&](const loom_module_t *module,
                          const std::vector<loom_value_id_t> &operands) {
    inspected = true;
    if (!loom_type_equal(loom_module_value_type(module, operands[0]),
                         loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "index operand required");
    }
    return iree_ok_status();
  };
  EXPECT_EQ(loomc_status_consume_code(
                body->InspectNative({body->Binding(0)}, expect_index)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_TRUE(inspected);
  inspected = false;
  EXPECT_EQ(loomc_status_consume_code(body->InspectNative(
                {sibling->Specialization(0)}, expect_index)),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_FALSE(inspected);
  Value escaped;
  ASSERT_TRUE(Ok(body->Serial([&](CommandProgramBuilder &nested) {
    return nested.IndexConstant(7, &escaped);
  })));
  EXPECT_EQ(
      loomc_status_consume_code(body->InspectNative({escaped}, expect_index)),
      LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_FALSE(inspected);
  ASSERT_TRUE(Ok(body->InspectNative({body->Specialization(0)}, expect_index)));
  EXPECT_TRUE(inspected);
  Pair(*owner, *body);
  ASSERT_TRUE(Ok(body->Finish()));
  ASSERT_TRUE(Ok(sibling->Finish()));
  inspected = false;
  EXPECT_EQ(
      loomc_status_consume_code(body->InspectNative({escaped}, expect_index)),
      LOOMC_STATUS_FAILED_PRECONDITION);
  EXPECT_FALSE(inspected);
  auto output = Finish(*owner, {body->Ref()});
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(Calls(Symbol(Native(output.get()), "authored")).size(), 2u);
}


TEST_F(CommandBuilderTest, ProgramInterfaceChecksDefinitionAndDeclarationGroups) {
  auto declared = Parse("command.program.decl @declared(%n: index) launch(%a: buffer, %b: buffer)\n");
  ASSERT_NE(declared, nullptr);
  auto owner = Builder(declared.get()), foreign = Builder(declared.get());
  ProgramRef declaration, other;
  ASSERT_TRUE(Ok(owner->Program("declared", &declaration)));
  ASSERT_TRUE(Ok(foreign->Program("declared", &other)));
  const auto index = owner->Scalar(LOOM_SCALAR_TYPE_INDEX), buffer = owner->Buffer();
  ASSERT_TRUE(Ok(owner->CheckProgramInterface(declaration, {index}, {buffer, buffer})));
  EXPECT_EQ(loomc_status_consume_code(owner->CheckProgramInterface(other, {index}, {buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(owner->CheckProgramInterface(declaration, {}, {buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(owner->CheckProgramInterface(declaration, {index}, {buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(owner->CheckProgramInterface(declaration, {buffer}, {buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(loomc_status_consume_code(owner->CheckProgramInterface(declaration, {foreign->Scalar(LOOM_SCALAR_TYPE_INDEX)}, {buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  auto body = Program(*owner);
  ASSERT_TRUE(Ok(body->Finish()));
  ASSERT_TRUE(Ok(owner->CheckProgramInterface(body->Ref(), {index}, {buffer, buffer, buffer})));
  EXPECT_EQ(loomc_status_consume_code(owner->CheckProgramInterface(body->Ref(), {index}, {buffer, buffer})),
            LOOMC_STATUS_INVALID_ARGUMENT);
  auto module = Finish(*owner, {body->Ref()});
  ASSERT_NE(module, nullptr);
}
} // namespace
} // namespace hrx::loomify_gguf
