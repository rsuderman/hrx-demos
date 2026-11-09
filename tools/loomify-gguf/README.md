# loomify-gguf

`loomify-gguf` reads a supported GGUF descriptor, generates data-only
configuration and native-links maintained Loom architecture and kernel modules.
It exports a numerical module through the public C++ builder or a complete
package for existing Loom Serve. The qualified full-model profiles are
Llama3.2 3B Q4_K_M and Qwen3.5 0.8B UD-Q5_K_XL; weights and tokenizer stay external.

```sh
bazel build //tools/loomify-gguf:loomify-gguf
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --stage=model --gguf=/path/to/model.gguf --tokenizer=/path/to/tokenizer.json \
  --output-dir=/path/to/new-package --prefill-capacity=128 \
  --context-capacity=512 --format=bc
```

The [central export guide](../../lib/docs/gguf-export.md) documents the public
`gguf_model_builder.h` API, declared library products, supported model bounds,
source/bytecode serialization and package validation. Maintained architecture
programs own invocation order and preparation records; C++ owns facts, native
linking and safe publication.

The default `--stage=embedding` remains an independent F32 token-embedding
example. It links the maintained `embedding.loom` command and operation-owned
kernel with generated shape facts. `model.embedding(token:index)` takes the
immutable named-parameter root and an output buffer, and writes zeros for an
out-of-vocabulary token. Use `--output=/path/to/embedding.loombc` or
`--format=loom` for text. Its numeric/error/guard tests live in
`//tools/loomify-gguf/testing:{framework_test,embedding_execution_test}`.

Generated artifacts belong outside the checkout. This developer tool is not
included in the serving wheel. Architecture/tensor reports remain available in
[loom-model-inspect](../loom-model-inspect/README.md); imported-kernel examples and
numerical test responsibilities are described in [library documentation](../../lib/docs/README.md).
