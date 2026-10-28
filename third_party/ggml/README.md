# GGUF metadata reader

`loomify-gguf` uses the ggml-base/GGUF code from llama.cpp commit
`dace2471abb49c118f95e6be09b4b8de2c1118b2`, under the upstream MIT license
(`LICENSE` in the pinned source repository). The Bazel repository keeps the
upstream source and license intact. No llama inference runtime or device
backend is built.

The reader runs with `no_alloc=true` and `ctx=nullptr`. Its callback is bounded
to the first 64 MiB, with an additional fixed-header/count admission check.
Tensor data is neither allocated nor read. The adapter separately verifies
each tensor's range against the actual file size.

`metadata_validation.patch` adds three narrow malformed-input checks before
upstream assertion/division sites: the alignment metadata type, zero tensor
rank, and zero dimensions. It does not introduce a second parser or change
tensor execution support. Keep this patch visible when updating the pin.
