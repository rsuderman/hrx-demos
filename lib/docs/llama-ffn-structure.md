# Normalized Llama FFN structure

`//lib/llama:ffn_structure` supplies the internal production helper
`hrx::llama::detail::AppendNormalizedFfn`. It appends three existing typed operations
to a caller-owned graph:

```text
residual → RmsNorm → GatedMatmul(SiLU) → LinearResidual → residual
```

The last operation updates the same residual `ValueId`. There is no implicit SSA
version, kernel selection, activation packing, numerical execution or new graph
representation. This is the first logical-structure checkpoint; complete Llama
model construction is not implemented by this target.

The caller registers `DefineRmsNorm`, `DefineGatedMatmul` and
`DefineLinearResidual`, creates an `OperationGraph` and declares a dense F32
`[rows, embedding_width]` residual. With a `model::ModelDescriptor` named `facts`:

```cpp
hrx::llama::detail::NormalizedFfnRegion region;
if (!hrx::llama::detail::AppendNormalizedFfn(
        &graph, facts, /*layer_slot=*/0, residual, &region, &error)) {
  // Discard the private graph; do not publish a partially constructed model.
}
```

`layer_slot` indexes the descriptor's layer vector, not its semantic `index` field.
The helper requires exactly one `post_norm`, `ffn_gate`, `ffn_up` and `ffn_down`
binding. `post_norm` is the descriptor role for `blk.N.ffn_norm.weight`.
Normalization uses dense F32 weights. Gate/up/down accept numeric GGUF tags
8/Q8_0, 12/Q4_K and 14/Q6_K, with logical F32 `[input, output]` GGML axes.
`storage_name` is diagnostic only. Other storage and incompatible axes are rejected.

The resulting region records the operation range, boundary residual, internal
activation IDs and parameter associations. Each association maps a graph value to
layer/tensor vector indices in the existing descriptor; tensor names, encoded byte
sizes and file offsets stay in that descriptor. The helper does not infer physical
bytes from logical tensor types or require a native provider's fixed geometry.

The future complete model builder calls this same helper while retaining immutable
descriptor and graph ownership. It must publish its completed owner only after all
required regions succeed. The helper publishes its region only on success, but
native graph append does not provide multi-operation rollback: discard the private
graph after a failed append. An explicitly scoped test fixture demonstrates that
ownership/publication rule; there is no public partial-model factory.

Run `//lib/llama/testing:ffn_structure_test` and
`//lib/operations/testing:ffn_operations_test` with Bazel. These tests prove logical
topology, descriptor association, encoding/shape admission and failure behavior.
They do not qualify provider arithmetic, dispatch binding or complete model output.
