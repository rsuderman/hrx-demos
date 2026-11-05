# Known numerical dispatches

`//lib/dispatch:registry` maps known semantic dispatches to independently
owned files in `lib/kernels/llama`. `Registry::Llama3B()` provides the initial
profile. `Find(Id)` or `Find(name)` inspects the first registered implementation.
`Resolve` selects a supporting named implementation in registration order. `OpCode`, `Port` and `Access` are the existing
operation graph types. Native declarations define the callable ABI, including
dependent weight views.

| Registration | Operation | Public native callable | Kind |
| --- | --- | --- | --- |
| `llama.rmsnorm` | RmsNorm | `llama32_rmsnorm_f32` | Kernel |
| `llama.ffn.gate_up_prefill` | GatedMatmul | `llm.ffn.gate_up_prefill` | Command template family |
| `llama.ffn.gate_up_decode` | GatedMatmul | `llm.ffn.gate_up_decode` | Command template family |
| `llama.ffn.down_prefill` | LinearResidual | `llm.ffn.down_prefill` | Command template family |
| `llama.ffn.down_decode` | LinearResidual | `llm.ffn.down_decode` | Command template family |
| `llama.embedding` | Embedding | selected Q6 token embedding | Kernel |
| `llama.projection` | Linear | Q4_K/Q6_K compatibility linear | Kernel |
| `llama.attention_output` | LinearResidual | compatibility linear + guarded attention residual | Kernel composition |
| typed Rope / RotatePublishKV / CausalAttention | attention operations | typed dense attention providers | Kernel |
| output head | OutputHead | `llm.output.last_greedy` | Command template family |
| prefill/decode publication | PublishToken | `llama32_finish_isolated` | Kernel |


Norm reads input and gamma and writes a fresh F32 output. Gate/up reads input and
two weight matrices and writes a fresh F32 activation. Down reads activation and
weights and updates the residual in place. Shape inference and logical output
creation use the existing shape and invocation APIs. Each registration supplies an
operation builder that allocates its native output/scratch and emits the call.
Norm's input, gamma and output must be disjoint. The down residual may alias its
output as declared by ReadWrite. FFN temporaries and packing remain inside the
whole family bodies. Decode gate/up consumes Q8_1X4 packed input; prefill gate/up
packs its F32 input itself. This native distinction is the invocation adapter's
responsibility, not a change to the logical GatedMatmul operation.

## Loading and construction

`Load(context, workspace, registry, {Id::RmsNorm, ...}, source_root, &module,
&result)` collects all registered implementations for the requested operation IDs
and deduplicates their files through the normal native loader. Model code does not
enumerate internal variants. Explicit registration/file-list overloads remain useful
for package-level inspection. It checks the actual declaration kind, merges compatible declarations and
definitions, checks required config definitions, and performs strict native LINK
on the public symbol. On success it returns the merged authoring module, retaining
the original public family declarations for subsequent applications and overrides.
The module owns its source snapshots. `ModuleBuilder::Create` can use it directly;
`AddProvider` alone would not import declarations into the authoring module.

The caller supplies a context with AMDGPU support for target-bearing sources,
a workspace, and a source root such as the Bazel runfiles workspace. Release both
returned module and result. Native diagnosed failure can return OK API status with
a failed result, so check both. `Add` rejects duplicate `(id, variant)` keys,
conflicting operation names/contracts, missing builders and incompatible operands.
It checks the registered operation builder rather than a fixed Llama catalog. Native loading,
merge, link, and concrete specialization reject missing files/symbols, ABI
mismatches and incompatible family applications. An unspecialized template LINK
is not a proof that every possible future application has a viable provider.

## Named implementations and builders

`Registration::variant` names an implementation (default `"default"`). Entries
sharing a semantic ID/name retain identical logical operands/results/effects but
may reference different linked symbols and builders. An `OperationBuilder` holds
plain support/build callbacks and its semantic contract; no native owner-bound
handles are kept in the registry. Provider files supply configuration defaults;
operation attributes retain their existing C++ defaults. Explicit caller attributes
are validated, never silently replaced with a different mathematical policy.

A support callback examines a type-checked `InvocationRequest` without mutation.
Empty variant selection chooses the first supporting implementation; an explicit
variant must itself support the request and never silently falls back. Existing
native command-template family selection can still route internal implementations.
Additional vector/WMMA implementations fit this interface without model changes.

The build callback receives scoped native bindings, the existing inferred result,
and an optional existing output/ReadWrite alias through `BuildContext`. It allocates
fresh output and scratch using `Allocate`, resolves checked native refs and emits
ordinary calls through `CommandProgramBuilder`. A callback must set `staged` before
its first native mutation (`Allocate` does so automatically). Late failures dispose
the native owner, preserving the existing invocation failure contract. Immutable
`EmitExisting` and mutable `Invoke` share these exact operation builders.

`registered_norm_execution_test` executes the registered norm producer, then consumes
its internally allocated output through the existing residual-add kernel for an
observable independent numerical comparison. The variant invocation test uses the
existing C++-authored RMSNorm definition to exercise real default/fallback/override
calls and operation-ID loading, not synthetic numerical implementations.

## Admitted first profile

The initial registration profile is Llama 3.2 3B: hidden width 3072, FFN width 8192,
F32 activations/gamma, 1..512 prefill rows, and one decode row. Gate/up weights are
Q4_K; down weights are Q4_K or Q6_K. Use QkCompat packing, generic projections,
and the existing F32 arithmetic/reduction policy. `NormBuilder`/`FfnBuilder` own
support checks and native argument construction for these actual providers. The graph/shape layer validates
logical element types, tensor layouts and weight dimensions.

The four family declarations preserve their original operands and dependent ABI.
Restricted providers have native applicability predicates fixing geometry,
generic=true, compat=true, Q4 gate flags, and the correct Q4/Q6 down flags and byte
sizes. Unsupported flag combinations have no matching owned implementation. The
selected predicates are compile-time admission, not runtime bounds enforcement.
Prefill control/count/row-origin operands retain the existing serving contract.
Active rows must be valid; inactive output rows are not newly initialized by a
Write-only operation.

`config.loom` supplies this explicit profile's width/epsilon and leaf compile
capacities: 512 rows, Q4 output capacity 8192, Q6 output capacity 128256, and 32768
quantization groups. FFN widths and weight sizes are family operands; the registry
does not silently infer them from byte counts or redefine the ABI in C++.

## Owned source and build closure

Files in `lib/kernels/llama` are grouped into shallow operation folders:
`attention/`, `embedding/`, `ffn/`, `norm/`, `output/`, `projection/`
and `publication/`. Each operation keeps its interfaces, implementation,
configuration and private helpers together. Projection owns the linear and
activation-packing helpers reused by FFN and output. Attention owns rotary and
KV cache helpers; token publication has its own folder. The root `rows.loom`
and `config.loom` remain shared across operations.

One root BUILD package retains the existing library target names; exported source
labels include their operation path, such as
`//lib/kernels/llama:embedding/embedding.loom`. Registration provider lists use
the same paths and explicitly include shared root and projection dependencies.

The curated seed is the repository source at `9045b4b`:

- `norm/norm.loom`: the single `llama32_rmsnorm_f32` kernel from Llama 3B bindings.
- `rows.loom`: only shared RMSNorm, SwiGLU and residual row implementations from
  Qwen's `layer_decode.loom`; these are device dependencies, not command entries.
- `ffn/ffn.loom`: whole public family bodies from `llm/common/ffn.loom`. Non-admitted
  canonical, Q5, channel8 and Qwen WMMA branches are explicitly removed in source.
  The retained Q4/Q6 QkCompat arithmetic and public family selection are unchanged.
- `projection/linear_*` and `projection/quantize_*`: selected required native functions, device motifs,
  kernels, targets and declarations from `llm/motifs/ggml`. Canonical helper
  functions used by compatibility kernels remain dependencies; their presence
  does not admit a canonical-packing dispatch.
- `config.loom`: minimal selected Llama config plus explicit compile capacities.

Each file records its seed path for attribution. These are maintained source
files, not generated files or runtime imports of the old tree. The package's
`norm_interface`, `ffn_interface`, `norm`, and `ffn` Loom library targets refer
only to owned files and normal native tools. `norm_gfx1100` produces an actual
AMDGPU kernel binary. There is no production dependency on `loom-pipeline` or on
the separate Loomcxx branch.

`//lib/dispatch/testing:registry_test` proves load/print/strict-link behavior,
profile rejection, incompatible kernel and dependent-family interfaces, explicit
closure rejection, and public-family override selection. Static command fixtures
apply all four complete FFN families, including Q4 and Q6 down; the native command
transaction publishes their concrete gfx1100 kernel requests and the normal
artifact compiler emits them. The override fixture is a selection witness only.
These checks prove construction and compilation, not numerical accuracy, GPU
execution, complete-model generation, or serving behavior. Numerical qualification
of the owned bodies remains separate work.


## Packet rows and checked attention regions

Load registered IDs together with packet attention using `LoadLlamaNative` and
one `AttentionNativeConfig`. Configure the session's `LlamaRows` before emitting:
original packet, phase, physical capacity, static row origin and exact logical
row controls. The session derives a private count snapshot each invocation.
Reserve `count_control` for the head and `packet_positions` for attention;
callers cannot supply buffer bindings for those derived identities. Publication
retains the distinct original packet. Norm, embedding, projection and whole FFN
consume active rows without requiring initialized inactive scratch. The existing
selected-FFN wrapper now uses `FfnRegion::Emit` with a configured session.

For packet attention, pass the actual `LlamaStructure` and recorded single-layer
Attention selection to `NativeEmissionSession::EmitAttentionRegion`. It validates
owner, packet-derived positions, row controls, vector layer slot, actual loaded
capacity/profile, role-resolved factors and private intermediates before staging.
It consumes the existing Rope/RotatePublishKV/CausalAttention subrange and owns
native scratch for its existing attended output. Distinct logical K/V planes map
internally onto the arena; callers do not fake them with duplicate arena bindings.
Set `LlamaBuildOptions::attention_policy` to `LlamaPacketMixed`; the separate typed
dense route retains `F32Accumulate` admission. Emit `Id::AttentionOutput` for the
following LinearResidual: it selects Q4/Q6 projection, allocates fresh projection
scratch, adds it to the original residual and returns that residual alias.

Pre-staging admission failure leaves session order/bindings unchanged. A native
failure after staging is terminal and requires discarding the mutable owner.
Unsupported controlled custom norm/embedding variants take this terminal path
because count staging may already have occurred; they never silently substitute
a built-in active provider. Static custom variants retain normal registry routing.

Useful test entrypoints under `//lib/dispatch/testing` are `invocation_test`,
`attention_region_test`, `active_rows_execution_test`,
`attention_regions_{dense,pooled}_execution_test`,
`attention_output_execution_test`, `head_rows_execution_test`,
`head_publication_{loom,loombc}_execution_test` and
`joined_adapters_{loom,loombc}_execution_test`. The head-row fixture is private-count
snapshot stress; the head-publication fixtures execute the actual registered
full head before publication. Joined one/two-layer fixtures execute the session's
attention and whole-FFN providers, replay changed packets, compare compact/padded
execution and check inactive residual/cache preservation. They establish adapter
composition; complete model/pipeline generation and real-model fidelity are
separate deliveries.
