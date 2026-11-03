# Dispatch result shapes

`//lib/dispatch/shapes:result` provides `hrx::dispatch::shapes::InferResult`.
It accepts an existing `OperationGraph`, an `OpCode`, existing operation attributes,
and input `{Port, ValueId}` bindings. Callers supply no fresh output tensor.
The returned `Result` contains a `TensorType`, result port, access mode and optional
source alias identity.

| Operation | Result | Identity and access |
| --- | --- | --- |
| RmsNorm | F32 `[rows,width]`, matching input | Fresh `Output`, Write |
| Linear | F32 `[rows,output_width]` | Fresh `Output`, Write |
| GatedMatmul | F32 `[rows,ffn_width]` | Fresh `Output`, Write |
| LinearResidual | F32 `[rows,output_width]` | Exact supplied `Residual`, ReadWrite |

Widths come from operation attributes and must agree with weight axes and input
shapes. Model construction derives these attributes and graph input dimensions
from `ModelDescriptor` and existing build options. The row dimension remains the
input capacity in both prefill and decode; optional I64 RowBegin/RowCount bindings
describe active rows and do not shrink the result type. Existing serving runtime
preconditions determine their valid values. ReadWrite preserves inactive residual
rows; a fresh Write output only defines its active rows.

Inference validates a candidate using a temporary instance of the existing
OperationGraph with the built-in operation definitions. Repeated input identities
remain repeated in this check, so native logical distinctness rules still apply.
The source graph's definitions need not have been populated. No source values or
operations are added or modified. On failure the previous result remains intact.
The returned alias is a borrowed logical identity: retain its source graph owner.
Fresh results have an invalid alias and are created later by Invocation.

Existing type verifiers check shapes, representations, attributes and row-control
ports. The dispatch shape layer limits weight encodings to Dense F32/F16 where
Linear already allows them, Q4_K, Q6_K and existing Q8_0. Whole FFN verifiers still
require quantized weights. Existing Linear activation formats remain accepted.
Positive tensor dimensions, logical element counts and decoded byte counts must
fit signed 64-bit native indices. Quantized TensorTypes use decoded logical F32
axes; this byte check does not compute encoded GGUF storage sizes or packing.

This package depends only on existing operation metadata and verifiers. It does
not read GGUF payloads, select providers, allocate native buffers, generate calls
or enforce runtime bounds. Invocation consumes its output shape and alias facts
and supplies actual native output arguments.
