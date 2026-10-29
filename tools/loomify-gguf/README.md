# loomify-gguf

`loomify-gguf` constructs a Loom module from a recognized GGUF model descriptor.
The current bounded stage is **F32 Llama token embedding lookup**. It emits the
complete module for that stage, including its numerical kernel and command body.
It does not yet generate the remaining model stages.

```sh
bazel run //:loomify-gguf -- --gguf=/path/to/model.gguf \
  --output=/path/to/embedding.loombc --stage=embedding
bazel run //:loomify-gguf -- --gguf=/path/to/model.gguf \
  --output=/path/to/embedding.loom --format=loom
```

The shared [model loader](../model/README.md) validates the checkpoint structure.
The generation recipe finds the `token_embedding` role, checks F32 storage,
embedding/vocabulary dimensions and byte extent, and constructs the program with
the host [module and command builders](authoring/command/README.md). An explicit
named parameter requirement binds `token_embd.weight`; file offsets are not baked
into the command. Unsupported selected storage is diagnosed rather than silently
converted. Other tensors may use storage types the embedding stage does not read.

The emitted root is:

```loom
command.program.def public @model.embedding(%token: index)
  launch(%parameters: buffer, %output: buffer)
```

`%token` is a planning specialization. `%parameters` is the normal immutable
parameter root, populated by Loom's named parameter binding path; `%output` has
room for `embedding_width` F32 values. Valid tokens select their embedding row.
Out-of-vocabulary token indices write a zero row. The logical table shape is
`[vocabulary_size, embedding_width]`, matching GGUF's linear `[width, vocab]`
storage. Weights remain external and must stay available for materialization.

The module is verified and strictly linked before serialization. `.loom` and
`.loombc` contain the same structured source program; target compilation and
execution happen through Loom's existing path. Checkpoint recognition and module
closure do not establish full-model behavior.

For checks:

```sh
bazel test //tools/loomify-gguf/testing:framework_test
bazel test //tools/loomify-gguf/authoring/model:embedding_execution_test
```

The GPU fixture writes a complete tiny synthetic GGUF outside the checkout,
invokes the actual CLI to create bytecode, and reads the named table through the
pinned IREE GGUF index. Those bytes populate the immutable parameter root. An
independent formula oracle checks distinct token rows, invalid-token behavior,
input preservation and output guards. This qualifies the bounded generated stage;
it does not exercise a complete serving-session weight-gather lifecycle.

Generated artifacts belong outside the checkout. This developer target is not
included in the serving wheel. Architecture/tensor reports remain available in
[loom-model-inspect](../loom-model-inspect/README.md).

A separate [full-model C++ authoring example](authoring/model-full/README.md)
constructs the supported Llama 3.2 3B Q4_K_M single-token prefill/decode recipe.
Its emitter can demonstrate caller-selected native RMSNorm kernel authoring with
`--authored-rmsnorm`. This is a separate developer tool; the embedding CLI above
continues to generate only its documented bounded stage. Full-model trajectory
checks and remaining tensor/logit qualification are distinguished in that example.
