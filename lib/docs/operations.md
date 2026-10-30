# Typed operation authoring

The reusable library lives in `lib/operations`. Its standalone demonstration is
in `examples/operations`, and its native tests are in `lib/operations/testing`.
User-facing command-line tools belong under `tools/`.

This C++17 library declares operation contracts, typed graph values and connected
invocations. Each successful append adds exactly one node. The FFN helpers
register illustrative F32/Q8_0 authoring contracts; they do not select kernels,
allocate tensors or establish model execution support.

Register contracts before constructing a graph:

```cpp
hrx::operations::OperationRegistry registry;
std::string error;
if (!hrx::operations::DefineGatedMatmul(&registry, &error))
  return false;
if (!hrx::operations::DefineLinearResidual(&registry, &error))
  return false;
hrx::operations::OperationGraph graph(registry);
```

The helpers use the public `define(...).port(...).attributes<T>().verify(...)`
builder. Clients can register their own contracts through that same interface.
An enum member alone does not register a definition. `GatedMatmul` reads input,
gate and up weights and writes output. `LinearResidual` reads input and weight,
and updates its `Residual` port with `Access::ReadWrite`.

Use enum-named ports and ordinary attribute assignments:

```cpp
using namespace hrx::operations;
GatedMatmulAttrs attrs;
attrs.input_width = hidden_width;
attrs.ffn_width = ffn_width;
attrs.activation = Activation::SiLU;
if (!graph.operation(OpCode::GatedMatmul)
         .bind(Port::Input, normalized)
         .bind(Port::GateWeight, gate_weight)
         .bind(Port::UpWeight, up_weight)
         .bind(Port::Output, hidden)
         .attributes(attrs)
         .append(&error))
  return false;
```

See [the complete checked example](../../examples/operations/connected_ffn.cc) for seven typed value
declarations and RMSNorm, gated matmul and residual projection connected through
shared handles. RMSNorm reads the same residual updated by the final call. Activations use dense
F32 RowMajor matrices `[rows, width]`. Weights use logical F32 elements,
`Encoding::Q8_0` and `Layout::GgmlWeightRows` with `[input, output]` axes; this
metadata does not describe original GGUF rank or physical byte sizes. The
verifiers accept positive dimensions, including small non-model sizes, without
inventing native block-size or alignment restrictions.

`GatedMatmul` checks `[T,H]`, two weights `[H,F]`, and output `[T,F]`, with SiLU
on the gate multiplied by the up projection. Read operands may share a compatible
handle; output must differ from every read operand. `LinearResidual` checks
input `[T,F]`, weight `[F,H]` and residual `[T,H]`; all three handles must differ.
Different handles do not prove physical buffers are disjoint.

A graph owns a snapshot of completed definitions, copied declarations and
attributes, and append-ordered invocations. Registering later does not change an
existing graph. Value handles retain owner identity across vector growth but do
not keep graph data alive. Inspection references are borrowed until owner
mutation or destruction. Keep handles or copy records across mutations.

Builders are move-only; terminal `finish` and `append` attempts consume them,
including failed attempts. Expected validation failures publish no partial
record and preserve earlier values/calls. Fallible calls require an error output,
clear it on success and return false (or an invalid value handle) on failure.
Owners must outlive unfinished builders. Allocation exceptions are not converted
to validation errors.

The graph records ordered mutable values, not SSA versions. Sharing a handle
connects calls or expresses an update; it does not infer allocation,
initialization, scheduling, fusion, conversions or native callables.

Host validation:

```sh
bazel test //lib/operations/testing:all
bazel run //examples/operations:connected_ffn
```

The FFN example prints `rms_norm`, `gated_matmul`, then `linear_residual`.
Neither the examples nor the tests need a GGUF, compiler, GPU or server. Library dependencies are the C++ standard
library; tests additionally use GoogleTest.

Additional operation groups:

| Header | Registration helpers |
| --- | --- |
| `tensor_operations.h` | DefineEmbedding, DefineRmsNorm, DefineQuantize, DefineLinear |
| `row_operations.h` | DefineCopyRows, DefineArgMax |
| `attention_operations.h` | DefineRope, DefineRotatePublishKV, DefineCausalAttention |
| `control_operations.h` | DefinePublishToken |

Author concrete positive rank-one through rank-three dense shapes by named calls:

```cpp
auto positions_type = TensorType::vector(token_capacity, ElementType::I32);
auto query_type = TensorType::vector(token_capacity, ElementType::F32);
query_type.addDimension(query_heads).addDimension(key_width);
```

GGML weights and packed activations remain rank-two. Logical decoded element
precision, storage encoding and axis layout are separate facts. The tensor group
admits dense F32/F16 or known GGML quantized weights; RMSNorm uses F32 gamma,
input and output, with finite positive epsilon. Quantize writes Q8_1X4 logical
F32 rows and requires width divisible by 128. Linear accepts dense F32/F16 or
Q8_1X4 input and writes F32, with explicit F32Accumulate policy. These are
metadata contracts, not numerical implementation qualification.

`port(...)` declares a required port; `optionalPort(...)` declares an optional
port and `PortDefinition::required()` exposes the distinction. Present optional
values receive the same owner/kind/duplicate checks. Rotary FactorMode::None
rejects a Factors operand; PerPair requires a dense F32 vector of half the even,
positive rotary width. Rotary base/scaling must be finite and positive; scaling
None requires factor 1. All query/key/value tensors are logical F32 head-shaped
values. Separate F16 KeyCache/ValueCache handles expose capacity and geometry;
publication uses ReadWrite and attention reads those same handles. Attention
requires Q divisible by KV and checks Q*value_width arithmetic. This does not
specify a native cache layout or allocate/cache-version resources.

CopyRows handles contiguous Range or one-row LastActive output and records its
destination as ReadWrite. It accepts dense matrices of matching width/type with
distinct source/destination handles. Range destination capacity can differ from
source capacity; runtime row counts must fit both. ArgMax reduces width-axis 1
with LowestIndex tie policy from F32 matrices to I32 token vectors.

Positions are I32 vectors and row controls are I64 scalar bindings. These APIs
validate metadata, not runtime IDs, positions, row begin/count, finite padding,
or causal bounds. A later execution binding must check their actual contents.
No implicit conversions, scheduling or primitive decomposition are added.

PublishToken consumes I32[1], with ReadWrite Control I32[3], History
I32[history_capacity+1] and Progress I32[8]. History capacity is positive and
checked before addition. Phase is Prefill or Decode. TerminalPolicy::None uses
an empty list; StopOnTokenIds uses nonempty nonnegative I32 IDs, copied into the
invocation with the rest of its attrs. The graph does not execute bookkeeping.

Additional standalone examples (each prints only its direct operation sequence):

```sh
bazel run //examples/operations:tensor_chain
bazel run //examples/operations:head_selection
bazel run //examples/operations:attention_graph
bazel run //examples/operations:token_publication
```
