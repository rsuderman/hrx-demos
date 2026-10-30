# Examples

Runnable demonstrations of library APIs belong here. Reusable implementation
code lives under `lib/`; user-facing command-line tools live under `tools/`.

The examples declare operation contracts and build connected invocation graphs:

| Example | Operations |
| --- | --- |
| [Connected FFN](../../examples/operations/connected_ffn.cc) | RMSNorm, gated matmul, residual projection |
| [Tensor chain](../../examples/operations/tensor_chain.cc) | Embedding, RMSNorm, quantization, linear |
| [Head selection](../../examples/operations/head_selection.cc) | Row selection, RMSNorm, linear, argmax |
| [Attention](../../examples/operations/attention_graph.cc) | RoPE, KV publication, causal attention |
| [Token publication](../../examples/operations/token_publication.cc) | Argmax, token/control publication |

Run an example by its target name:

```sh
bazel run //examples/operations:connected_ffn
bazel run //examples/operations:attention_graph
```

Run its library tests with `bazel test //lib/operations/testing:all`.
