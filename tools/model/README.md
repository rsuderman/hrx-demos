# Model loading

These native C++ libraries own GGUF catalog facts and logical architecture
validation. They perform no reporting, source generation, compilation or GPU
execution. Human presentation lives in [loom-model-inspect](../loom-model-inspect/README.md).

- `:gguf_catalog` owns complete tensor names, original ranks, normalized four-dimensional extents,
  numeric/readable storage types, checked absolute byte ranges and typed metadata.
- `:model_descriptor` binds dense Llama/Qwen3.5 global/per-layer roles, geometry,
  RoPE/SSM facts, attention/recurrent identities and logical tied output weights.

`ReadGgufCatalog` borrows an unbuffered seekable `FILE*`. Callers must disable
stdio buffering before any I/O to prevent payload prefetch. The owned bounded parser reads catalog records without tensor allocation or
data loading; it has no llama.cpp/ggml build dependency. Encoded storage sizing
supports F32, F16, Q8_0, Q4_K and Q6_K and rejects other tensor encodings. Admission is limited to little-endian GGUF
v2/v3, a 64 MiB header, 4,096 metadata entries and 16,384 tensors.

Catalog records retain exact names and scalar source types. Numeric/boolean
architecture arrays retain values; string arrays and tokenizer arrays retain
explicit type/count summaries. Returned catalogs and descriptors own their data,
retain no parser/file handles and leave output unchanged on failure. Retained
array elements have a combined 64 MiB storage budget. Tensor rows must be
divisible by their encoding block size; offsets and extents are checked before
publishing absolute file spans.

Current descriptor limitations include MoE/MTP, fused attention QKV, optional
biases/scales, unequal K/V head widths and unsupported RoPE schemes. Qwen GDN
QKV remains a supported recurrent role. Successful structural loading does not
establish provider, datatype or runtime compatibility.

```sh
bazel test //tools/model/testing:all
```
