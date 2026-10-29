# Host C++ module and command authoring

`ModuleBuilder` owns one mutable Loom module and stable typed references to its
kernels, command programs and kernel-scoped persistent pipelines. It clones its
input definitions once. Each command body belongs to this owner, so kernels can
be passed directly to multiple command programs without cloning each definition
or creating a child-program wrapper for each invocation.

This is an ordinary host C++ API over exact-version native Loom operations. Its
input definitions can originate from source, bytecode or loomcxx. It does not add
a language, compiler frontend or runtime, and does not execute computation during
authoring. Build against the same HRX revision/configuration as LoomC. All mutation
and workspace access require external synchronization.

## Author a command program

The following omits status handling for readability. Check every status and
`loomc_result_succeeded(result)`; `emit_example.cc` is a complete caller.

```cpp
using namespace hrx::loomify_gguf;
std::unique_ptr<ModuleBuilder> module;
ModuleBuilder::Create(context, workspace, definitions, &module);
KernelRef add, scale;
module->Kernel("add_one", &add);
module->Kernel("times_two", &scale);
std::unique_ptr<CommandProgramBuilder> program;
module->DefineProgram("example", {LOOM_SCALAR_TYPE_INDEX}, 3, &program);
auto n = program->Specialization(0);
program->Invoke(add, {n}, {n, program->Binding(0), program->Binding(1)});
program->Invoke(scale, {n}, {n, program->Binding(1), program->Binding(2)});
program->Finish();
loomc_module_t* output = nullptr;
loomc_result_t* result = nullptr;
module->LinkAndVerify({program->Ref()}, &output, &result);
```

`Kernel` resolves an imported definition once at the module boundary. `Invoke`
accepts its typed reference directly. Its operand lists are kernel workloads and
device arguments. Program specializations and issue-time buffer bindings are
separate groups. Native substitution-aware type checks handle dependent argument
types. Calls remain distinct and append to a serial completion schedule.

`Call(ProgramRef, specializations, buffers)` optionally reuses a child command
program. `Invoke(KernelPipelineRef, specializations, run_arguments)` emits the
existing kernel-scoped pipeline launch boundary. `Serial`, `Concurrent` and `If`
provide scoped region callbacks; outer values may be captured, nested values
cannot escape. `BuildNative` is a low-level synchronous extension seam for native
operation builders, with scope checks and terminal mutation-error handling.

## Explicit dependencies and complete artifacts

`DeclareKernel` authors an explicit interface using owner-scoped `Scalar` and
`Buffer` types. `AddProvider` snapshots an explicit definition module. Finalization
selects strict Loom LINK with the requested roots, resolves reachable exact
source declarations, verifies and returns independently owned output. It reports
missing, incompatible or ambiguous providers through normal Loom diagnostics.
There is no implicit provider filesystem search or custom linker. Runtime imports
and external weight/state resources remain explicit runtime contracts.

Typed `RootRef` accepts program, kernel or kernel-pipeline references, so a pipeline
artifact does not require a dummy command. `Root(name)` resolves other existing
entry kinds, such as function exports. Strict source closure is distinct from
later specialization, target compilation and execution qualification.

## Ownership and failure

The original definitions stay unchanged and may be released after creation.
Provider snapshots may likewise outlive their caller's modules. References belong
to one module, while values additionally belong to a command/lexical scope.
`Finish` seals one command body. Unfinished definitions block module finalization.

Validation errors before mutation are recoverable. A native mutation or region
callback failure terminally fails the entire module owner and prevents output.
There is no rollback. Finalization seals the owner even if linking returns a failed
result. Finalization or owner destruction invalidates outstanding body builders;
stale references cannot be used in a later owner. The caller releases returned
modules and results through the normal LoomC APIs.

## Build, run and inspect

```sh
bazel test //tools/loomify-gguf/authoring/command:command_program_builder_test
bazel test //tools/loomify-gguf/authoring/command:command_execution_test \
  --test_env=LD_LIBRARY_PATH --test_env=IREE_HAL_AMDGPU_LIBHSA_PATH

output="${TMPDIR:-/tmp}/loomify-command.loombc"
bazel run //tools/loomify-gguf/authoring/command:emit_example -- \
  tools/loomify-gguf/authoring/command/testdata/kernels.loom "$output"
bazel run @hrx_system//loom/src/loom/tools/loom-format -- \
  "$output" --from=bc --to=text
```

GPU execution requires AMD hardware and a configured ROCr runtime. The example
constructs two command programs passing kernels directly; independent test-only
oracles check `2 * (input + 1)` and `3 * (input + 1)`. Seven active elements sit
between guards, with intermediate/output buffers and input preservation checked.
Host tests inspect invocation identity and wiring, scope/lifecycle errors, source
closure and text/binary reload in fresh contexts. These establish a bounded
construction capability, not complete GGUF model generation.
