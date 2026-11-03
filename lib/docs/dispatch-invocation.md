# Output-producing dispatch invocation

`lib/dispatch:invocation` joins the existing dispatch registry, result inference,
`OperationGraph`, `ModuleBuilder`, `CommandProgramBuilder`, and `CommandValues`.
It creates logical outputs and supplies native output arguments internally. Normal
Loom allocation and native family selection remain authoritative.

Load the selected registrations together with `dispatch::Load(context, workspace,
entries, source_root, ...)` before creating a `ModuleBuilder`. This overload
validates each registration and deduplicates its interface/provider files into one
native MERGE closure, followed by strict LINK verification of every selected root.
`AddProvider` alone does not make declarations available for authoring lookup.

Declare the graph's operation definitions and external tensor values normally.
Declare the native program's external buffer bindings normally. Then bind their
identities once at the module boundary:

```cpp
InvocationProgram author(registry, graph, module, program);
// Check each returned status, as with the underlying native builders.
author.Bind(residual, program.Binding(0));
author.Bind(gamma, program.Binding(1));
author.Parameter(gate_weight, program.Binding(2), "blk.0.ffn_gate.weight");
author.Parameter(up_weight, program.Binding(2), "blk.0.ffn_up.weight");
author.Parameter(down_weight, program.Binding(2), "blk.0.ffn_down.weight");
ValueId normalized, activation, result;
author.Invoke(Id::RmsNorm, RmsNormAttrs{3072},
              {{Port::Input, residual}, {Port::Gamma, gamma}}, &normalized);
author.Invoke(Id::GateUpDecode, GatedMatmulAttrs{3072, 8192},
              {{Port::Input, normalized}, {Port::GateWeight, gate_weight},
               {Port::UpWeight, up_weight}}, &activation);
author.Invoke(Id::DownDecode, LinearResidualAttrs{8192, 3072},
              {{Port::Input, activation}, {Port::Weight, down_weight},
               {Port::Residual, residual}}, &result);
```

Every result is an existing typed graph `ValueId`; inspect its type through
`graph.value(result)->tensorType()`. Norm creates a distinct `[rows,3072]` F32
value. Gate/up creates `[rows,8192]` F32. Down is explicitly ReadWrite and returns
exactly `residual`. Graph invocation inspection and native emitted calls use these
same identities. `Native` exposes the exact scoped output buffer for inspection
or explicit assembly, and rejects handles that have escaped their native body.
No output or scratch buffers are required from the author at an invocation.

The supported provider profile is Llama 3B, F32 activations/gamma, Q4_K gate/up,
Q4_K or Q6_K down, QkCompat packing, and norm epsilon `1e-5`. Native-fixed
semantics cannot be changed by operation attributes. Epsilon admission compares
at the provider's F32 precision, accepting the GGUF header's promoted F32 value
without rewriting the original logical attribute; incompatible values still fail. Named immutable FFN weights
are materialized as dependent byte views internally; their size SSA values are
also the native family's corresponding shape arguments. Decode gate/up packs its
F32 input internally before applying the existing public native family.

This first mapping covers a full static row domain, with `1 <= rows <= 512` for
prefill and exactly one row for decode. For prefill, call `Control` with the
existing consumer control buffer before invocation. The consumer must supply an
i32 active count at byte offset zero equal to the graph's static row count;
row origin is zero. Norm, gate/up, and down therefore operate on the same complete
row domain. This is an existing valid-input precondition, not a generated runtime
check. Explicit graph `RowBegin`/`RowCount` operands fail before mutation. General
active-view/capacity mapping belongs to subsequent common and attention binders;
this API does not claim full dynamic-capacity model support.

`CommandFamilyRef` and `CommandProgramBuilder::Apply` check module ownership,
body scope, argument count, and native dependent argument types before emitting
a void host-family application. Missing/foreign/incompatible registrations and
logical shapes fail preflight. Before staging, norm and decode packing use the
existing static kernel interface check. Selected command families are compared
against their known public native declaration, including argument counts, scalar
and buffer types, and remapped dependent shape relationships. These knowable
registration mismatches leave the same invocation object and native owner usable.
Value-dependent application checks still occur through native Invoke/Apply once
actual shape SSA exists. After native staging begins, any failure marks the
existing native owner terminal: discard the owner and graph together. A failed
invocation leaves the caller's result handle unchanged. Successful outputs and
native parameter SSA never transfer to another body or module.

The native tests emit norm and composed norm/whole-FFN modules for prefill/decode
and both down formats, inspect actual output argument identities, strict LINK,
serialize `.loom`/`.loombc`, and compile every concrete gfx1100 kernel request to
nonempty artifacts. This proves native generation and compilation; GPU numerical
qualification and complete-model consumer integration are separate checkpoints.

## Emitting an immutable existing graph

`NativeEmissionSession(registry, const_graph, module, program)` shares the same
native admission, allocation, parameter-view and dispatch staging as mutable
`InvocationProgram`. It exposes `graph()` for ownership checks and the existing
`Bind`, `Parameter`, `Control` and `Native` methods. The caller retains the graph
and module/body for the session's lifetime.

`EmitExisting(id, graph.invocations()[index])` emits that exact graph-owned
invocation. Fresh native output storage is associated with the existing logical
output; ReadWrite returns the same bound residual storage. It adds no graph
values or operations, clones no graph and casts away no constness. Emit in forward
append order; copied/foreign operations, replay and inferred output/type
disagreement fail before staging. An unbound Write output gets generated native
storage. A compatible existing scoped buffer mapping is reused, whether generated
by an earlier Write or explicitly bound at the module boundary. This preserves
repeated mutable writes and real external outputs without requiring caller
bookkeeping for temporaries. Named immutable parameter mappings cannot be outputs. Native handles retain the
original builder's body scope. Mutable `Invoke` delegates native staging here,
then adds its newly inferred logical result and invocation exactly once.

The first owner-resolved FFN proof contains explicit logical row controls. Use
`SpecializeOneRow(owner.inputs().row_begin, owner.inputs().row_count)` once before
emission to assert that these exact owned I64 controls mean begin zero/count one
for the entire proof entrypoint. `one_row_specialized()` reports that explicit
setting. Only full capacity one is admitted under this setting; capacity one
alone and unresolved runtime scalars do not establish it. Mismatched controls,
other capacities and late/repeated specialization fail. The original graph still
retains the control operands for inspection and shape verification; native staging
omits only the explicitly specialized pair. Prefill continues to require the
existing control buffer with its i32 count field at byte zero fixed to one by the
caller. This is a named compile-time proof contract, not a general Loom Serve
entrypoint or a runtime bounds check.

Native tests exercise const owner-resolved FFN references in both phases and both
down formats, preserve every original graph/value identity, inspect actual native
output/alias buffers, and strictly link/serialize/target-compile the generated
module. Named norm gamma and compiled parameter-root binding remain a separate
parameter checkpoint; this extraction's const fixture uses an explicit gamma
boundary. No numerical or dynamic active-view qualification is claimed.

## Binding owner parameters

`BindRegionParameters(owner, region_view, parameter_root, session, error)` in
`lib/dispatch/parameters.h` resolves used parameter ValueIds through the owner's
descriptor associations. It registers canonical GGUF keys once per logical
identity, validates supported encoded extents/byte spans before registration, and
rejects a session for another graph. The owner and module/body remain retained;
failed registration requires discarding the session. Call before region emission.

Named normalization gamma is staged as an F32 vector; Q4_K/Q6_K whole-FFN weights
remain encoded I8 byte views. Both mutable authoring and const existing-operation
emission share this behavior. Output and scratch creation stay internal.

GGUF absolute source offsets, provider tensor-relative offsets and compiled root
placements are separate coordinates. A consumer prepares each compiled execution
root from its own fresh reflected requirements. The pinned native reflection
reader is currently internal to HRX's allowed consumers; the downstream native
parameter test proves declarations, actual norm argument wiring and FFN artifact
compilation. The task-owned external allowed-package fixture provides reflected
byte/provider evidence without exporting a supported downstream reflection API.

## Operation-owned implementations

Pass an optional final variant name to `Invoke` or `EmitExisting` to select an
explicit registered implementation. With no name, the registry tries implementations
in registration order and uses the first whose operation-owned support check accepts
the request. The same input/configuration interface is used for all variants.

Register the numerical implementation and its `OperationBuilder`; the shared session
handles logical bindings/ownership and delegates native arguments, output allocation
and private scratch to that builder. Load required operation IDs with the registry
before defining the command body so all registered alternatives are available.
See [dispatch registration](dispatch-registry.md) for the callback and linking contract.
