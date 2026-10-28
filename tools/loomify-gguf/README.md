# loomify-gguf

`loomify-gguf` is the C++ developer foundation for generating Loom model
programs from GGUF checkpoints. With no arguments it prints an empty loomcxx
function to stdout:

```sh
bazel run //:loomify-gguf > /tmp/model.cxx
bazel run //:loomify-gguf -- --help
```

Load a local model's architecture and print every tensor, including unbound
extras, without loading its tensor payloads:

```sh
bazel run //:loomify-gguf -- --gguf=/path/to/model.gguf
```

Loading accepts little-endian GGUF v2/v3 and requires a nonempty
`general.architecture` string. It validates catalog structure and every tensor
span against the file size. It bounds header reads to 64 MiB and catalog counts
to 4,096 metadata entries and 16,384 tensors. Files exceeding those inspection
limits are rejected. The pinned maintained reader owns GGUF parsing; see
[the dependency and patch notes](../../third_party/ggml/README.md).
No report is written until validation succeeds. Failures write diagnostics to
stderr and exit nonzero. Loading does not require a GPU or tokenizer.

This mode supports dense `llama` and `qwen35`, independently of model size or
storage datatype. It validates required typed metadata, per-layer dimensions,
logical tensor shapes, tied output, and ordered attention/recurrent identities.
The report retains exact tensor names using lossless byte escaping and prints
numeric ggml storage tags, readable names, normalized extents and absolute ranges.
Qwen SSM history/state sizes are logical element counts per layer/sequence;
they do not specify device allocation or runtime batching.

The owned `GgufCatalog` in `gguf_catalog.h` retains all admitted tensors and all
metadata keys/scalars. Source GGUF tags distinguish scalar widths. Numeric and
boolean architecture arrays retain typed values; string arrays and tokenizer
arrays retain type/count with `elements_omitted=true`. Tensor extents follow
GGML order, padded to four dimensions; original rank is unavailable. The
bounded `GgufInfo` presentation derives from this same parser path.
`LoadModelDescriptor` produces owned global/per-layer role bindings with full
storage facts; neither result borrows the parser or input file. Readers and
loaders publish output only on success. Library file callers must disable stdio
buffering before any I/O to prevent payload prefetch, as the CLI does.

The initial descriptor rejects MoE, MTP, fused **attention** QKV layouts,
optional biases/scales, unequal key/value widths, and unsupported RoPE schemes
explicitly. Qwen GDN's fused QKV is a supported recurrent role. Linear/none RoPE
scaling, base/width/sections/original context, and optional Llama
`rope_freqs.weight` are retained. Other architectures remain available through
the catalog inspection library; the CLI reports unsupported architectures as
errors. Structural admission does not establish numerical or kernel
compatibility, choose capacities, construct a pipeline, or read tensor payloads.

The `:source_builder` library owns source construction independently of CLI
and file handling. Both targets use the pinned HRX shared C++ build macros; the
CLI uses HRX flag handling. This source-only tool does not need the C++ importer
or a GPU to run.

The no-argument source output has a stable Loom symbol, `model.pipeline`, and
imports as an ordinary empty function. It is groundwork for model construction.
Numerical providers and inference remain future stages; loading a GGUF
architecture does not construct an executable model.

To import the output into bytecode using the optional pinned HRX frontend:

```sh
bazel run --@hrx_system//loom/config/import:enable=cxx \
  @hrx_system//loom/src/loom/tools/loom-import-cxx -- \
  --root=model_pipeline --to=bc --output=/tmp/model.loombc /tmp/model.cxx
```

Generated source and bytecode should be written outside the checkout during
development. The tool is a developer target and is not added to the serving
wheel.
