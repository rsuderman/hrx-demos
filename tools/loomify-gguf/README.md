# loomify-gguf

`loomify-gguf` is the C++ developer foundation for generating Loom model
programs from GGUF checkpoints. With no arguments it prints an empty loomcxx
function to stdout:

```sh
bazel run //:loomify-gguf > /tmp/model.cxx
bazel run //:loomify-gguf -- --help
```

Inspect a local model without loading its tensor payloads:

```sh
bazel run //:loomify-gguf -- --gguf=/path/to/model.gguf
```

The report includes the GGUF version, optional name, architecture, counts,
alignment/data offset, metadata values and absolute tensor byte ranges. Arrays
are summarized by type and length, not expanded. Reports contain at most 64
metadata entries and 32 tensors, and text fields are bounded to 160 bytes.
Tensor shapes have four extents, with absent dimensions padded to one by ggml.
Storage type names describe the file; they do not qualify execution support.

Inspection accepts little-endian GGUF v2/v3 and requires a nonempty
`general.architecture` string. It validates catalog structure and every tensor
span against the file size. It bounds header reads to 64 MiB and catalog counts
to 4,096 metadata entries and 16,384 tensors. Files exceeding those inspection
limits are rejected. The pinned maintained reader owns GGUF parsing; see
[the dependency and patch notes](../../third_party/ggml/README.md).
No report is written until validation succeeds. Failures write diagnostics to
stderr and exit nonzero. Inspection does not require a GPU or tokenizer.

The `:source_builder` library owns source construction independently of CLI
and file handling. Both targets use the pinned HRX shared C++ build macros; the
CLI uses HRX flag handling. This source-only tool does not need the C++ importer
or a GPU to run.

The output has a stable Loom symbol, `model.pipeline`, and imports as an ordinary
empty function. It is groundwork for model construction. Architecture-specific
validation, numerical providers, and inference are future stages; inspecting
a GGUF does not construct an executable model.

To import the output into bytecode using the optional pinned HRX frontend:

```sh
bazel run --@hrx_system//loom/config/import:enable=cxx \
  @hrx_system//loom/src/loom/tools/loom-import-cxx -- \
  --root=model_pipeline --to=bc --output=/tmp/model.loombc /tmp/model.cxx
```

Generated source and bytecode should be written outside the checkout during
development. The tool is a developer target and is not added to the serving
wheel.
