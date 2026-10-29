# GGUF embedding fixture

These are test-only synthetic checkpoint and execution fixtures for the bounded
embedding generation recipe. `embedding_fixture` writes a structurally complete
one-layer F32 Llama GGUF with unrelated tensors and distinct embedding rows. The
selected table follows a distractor tensor at a nonzero aligned data offset.

`emit_embedding_fixture` reads `token_embd.weight` through IREE's named GGUF index
and uses those bytes to initialize the packed immutable root in a generated Loom
check scenario. The actual `loomify-gguf` CLI constructs the subject bytecode from
the GGUF descriptor. A separate mathematical oracle computes the known row values
without reading the table or calling the subject. Tests cover three token rows,
an invalid token, unchanged parameter owners and guarded outputs.

This proves named source-byte selection followed by generated-command numerical
consumption. It does not replace a full runtime provider upload/serving gate or
full-model checks. Synthetic weights and generated modules are build outputs,
never committed model assets.
