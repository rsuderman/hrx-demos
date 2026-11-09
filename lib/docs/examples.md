# Examples

Runnable demonstrations of library APIs belong here. Reusable implementation
code lives under `lib/`; user-facing command-line tools live under `tools/`.

## Imported Loomcxx computation

`examples/loomcxx_import/testdata/affine.cc` owns its numerical helper and kernel.
The example imports it with the existing pinned source-provider/header callbacks,
then native-links `examples/loomcxx_import/affine.loom`, a maintained two-buffer
command wrapper. It writes ordinary closed Loom bytecode; no C++ command builder
is involved. Run `bazel run --@hrx_system//loom/config/import:enable=cxx
//examples/loomcxx_import:emit_example -- INPUT.cc OUTPUT.loombc`.
The import roundtrip and GPU execution tests retain signed-input, guard and input
immutability checks against their independent scalar oracle.

The F32 GGUF embedding CLI similarly links maintained `embedding.loom` with the
operation-owned `lib/kernels/embedding/f32.loom` and two generated shape facts.
Its `model.embedding(token:index)` command retains two buffer bindings, named
external weights and zero output for an out-of-vocabulary index. Tiny GGUF fixture
and independent row/guard/alias checks live beside `framework_test.cc` in
`tools/loomify-gguf/testing`; these fixtures do not qualify complete model math.
