# Loomify host tests

`framework_test` exercises the GGUF-driven embedding authoring library. It parses
real synthetic GGUF headers into the shared descriptor, checks named payload
identity, inspects actual generated command/kernel references and dimensions,
and reloads both text and bytecode in fresh module handles. Negative cases cover
missing/ambiguous roles, unsupported storage, inconsistent extents/dimensions and
unsupported architectures.

The actual CLI and generated command execute through
`//tools/loomify-gguf/authoring/model:embedding_execution_test`; its independent
oracle and guarded parameter/output owners test the numerical stage.
