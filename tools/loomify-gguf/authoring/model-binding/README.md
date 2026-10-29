# GGUF parameter binding contract

The model catalog identifies a tensor by its canonical GGUF key, GGML storage
type, extents and absolute file range. Existing Loom command programs bind
immutable data using `command.parameter` against an explicit parameter buffer.
The result is a logical typed view, with an exact byte footprint after planning.
The command compiler packs those named requirements into aligned fixed roots.
Its reflected offsets are relative to those roots, rather than the GGUF file.

Loom Serve's existing parameter input path indexes declared checkpoint files
through IREE's parameter-file parser. For GGUF, the resulting index maps the key
to `tensor_data_offset + tensor_relative_offset` and its encoded byte size.
Parameter-provider gathers take offsets relative to that indexed tensor and copy
to the command's reflected fixed-root placement. A generated program should
preserve canonical tensor keys and use this existing path. It should not replace
named parameters with absolute GGUF offsets.

Storage encoding is part of the numerical implementation contract. The GGUF
runtime index does not expose dtype/shape metadata for its entries: the catalog
must qualify those facts before construction. An unchanged weight policy preserves
canonical bytes. A transformation requires an explicit existing preparation policy
whose result length matches the reflected final tensor size.

The runtime parameter bank owns immutable fixed roots; its indexed sources and
preparation plan retain file handles and keys. Preparation and activation must
complete before execution. Recordings retain their views; consumers and borrowing
residencies release them before bank retirement completes. Merely matching layout
does not establish checkpoint identity.

The focused host test checks agreement between the actual catalog and runtime
GGUF parser. Distinct named F32 payloads sit at nonzero offsets, and key order
varies between fixtures. It checks tensor-relative reads, missing keys, retained
file-handle ownership and truncated ranges. The catalog's pinned GGML reader
requires canonical packed tensor ordering, so fixtures must obey that restriction.

Run with the repository's Bazel version:

```sh
bazel test //tools/loomify-gguf/authoring/model-binding:gguf_parameter_binding_test
```

This host test proves the source-index contract. A generated-program numerical
test must separately exercise command reflection, fixed-root placement and actual
kernel consumption of those bytes. It does not qualify a model or target backend.
