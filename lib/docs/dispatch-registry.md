# Known numerical dispatches

`//lib/dispatch:registry` maps five known semantic dispatches to independently
owned files in `lib/kernels/llama`. `Registry::Llama3B()` provides the initial
profile. `Find(Id)` or `Find(name)` returns the registration; there is no discovery
or automatic provider selector. `OpCode`, `Port` and `Access` are the existing
operation graph types. Native declarations define the callable ABI, including
dependent weight views.

| Registration | Operation | Public native callable | Kind |
| --- | --- | --- | --- |
| `llama.rmsnorm` | RmsNorm | `llama32_rmsnorm_f32` | Kernel |
| `llama.ffn.gate_up_prefill` | GatedMatmul | `llm.ffn.gate_up_prefill` | Command template family |
| `llama.ffn.gate_up_decode` | GatedMatmul | `llm.ffn.gate_up_decode` | Command template family |
| `llama.ffn.down_prefill` | LinearResidual | `llm.ffn.down_prefill` | Command template family |
| `llama.ffn.down_decode` | LinearResidual | `llm.ffn.down_decode` | Command template family |

Norm reads input and gamma and writes a fresh F32 output. Gate/up reads input and
two weight matrices and writes a fresh F32 activation. Down reads activation and
weights and updates the residual in place. Shape inference and logical output
creation belong to their own APIs; this registry does not allocate outputs.
Norm's input, gamma and output must be disjoint. The down residual may alias its
output as declared by ReadWrite. FFN temporaries and packing remain inside the
whole family bodies. Decode gate/up consumes Q8_1X4 packed input; prefill gate/up
packs its F32 input itself. This native distinction is the invocation adapter's
responsibility, not a change to the logical GatedMatmul operation.

## Loading and construction

`Load(context, workspace, registration, source_root, &module, &result)` reads the
explicit interface and provider file list through the normal native module
loader. It checks the actual declaration kind, merges compatible declarations and
definitions, checks required config definitions, and performs strict native LINK
on the public symbol. On success it returns the merged authoring module, retaining
the original public family declarations for subsequent applications and overrides.
The module owns its source snapshots. `ModuleBuilder::Create` can use it directly;
`AddProvider` alone would not import declarations into the authoring module.

The caller supplies a context with AMDGPU support for target-bearing sources,
a workspace, and a source root such as the Bazel runfiles workspace. Release both
returned module and result. Native diagnosed failure can return OK API status with
a failed result, so check both. `Add` rejects duplicate identities/names and
incompatible semantic contracts or incomplete known file lists. Native loading,
merge, link, and concrete specialization reject missing files/symbols, ABI
mismatches and incompatible family applications. An unspecialized template LINK
is not a proof that every possible future application has a viable provider.

## Admitted first profile

The initial registration profile is Llama 3.2 3B: hidden width 3072, FFN width 8192,
F32 activations/gamma, 1..512 prefill rows, and one decode row. Gate/up weights are
Q4_K; down weights are Q4_K or Q6_K. Use QkCompat packing, generic projections,
and the existing F32 arithmetic/reduction policy. `CheckProfile` validates these
geometry, encoding, phase and packing choices. The graph/shape layer validates
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

The curated seed is the repository source at `9045b4b`:

- `norm.loom`: the single `llama32_rmsnorm_f32` kernel from Llama 3B bindings.
- `rows.loom`: only shared RMSNorm, SwiGLU and residual row implementations from
  Qwen's `layer_decode.loom`; these are device dependencies, not command entries.
- `ffn.loom`: whole public family bodies from `llm/common/ffn.loom`. Non-admitted
  canonical, Q5, channel8 and Qwen WMMA branches are explicitly removed in source.
  The retained Q4/Q6 QkCompat arithmetic and public family selection are unchanged.
- `linear_*` and `quantize_*`: selected required native functions, device motifs,
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
