# Qwen model authoring

`lib/qwen/structure.h` retains the existing owned `ModelDescriptor` and one typed
`OperationGraph`. `BuildQwenStructure` creates the complete sequence:

```
Embedding
for each descriptor layer:
    QwenGdnLayer or QwenAttentionLayer
QwenEpochHead
```

Each complete layer includes input norm, its mixer and output residual, post norm,
and the whole gated FFN/down residual. The author adds none of these again. The
original descriptor layer index selects canonical named weights; independent
recurrent/attention ordinals select persistent state planes. Only the logical
factory contains the architecture loop. Native emission traverses its records.

Parameters retain source asset/name identity and descriptor coordinates. Q4_K,
Q5_K and Q6_K matrices retain encoded source byte ranges and decoded logical axes.
The F32 convolution tensor has GGML source extents `[taps, channels]`; its unchanged
bytes are described as row-major logical `[channels, taps]`. No payload transpose
occurs. Tied embedding/output weights share a graph value and native requirement.

GDN histories and recurrent matrices, and attention K/V caches, remain distinct
owned tensor identities. Their physical mappings belong to the native epoch
session, using the existing resident origins/map tables. Attention graph capacity
C is logical; physical pool capacity P determines native strides. A raw arena does
not prove a tensor's shape or state ordinal.

`QwenEpochHead` reads the actual selected-row epoch packet. The full packet is
I32[1048]; the head's readable prefix through the selection table is387 words.
Public output is I32[configured selected-output capacity], which is one in the
first serving profile. Only indices below runtime output_count are defined;
count zero writes no output ID. Padded normalization/projection rows are private
native scratch. Existing Llama OutputHead has a different packet and result
contract and is not used for this head.

`ExtractQwenConfiguration` borrows an unbuffered seekable file and composes the
production GGUF reader and descriptor loader. The reader admits GGML Q5_K tag13,
block256/176 encoded bytes. The returned descriptor survives the file and catalog.
Both extraction and complete graph construction publish outputs only on success.

The first numerical profile is ordinary text Qwen3.5 0.8B, one resident row/span,
with explicit deployment T and C. Structural fixture geometry/routing variations
do not qualify additional numerical implementations. Native registrations own
geometry/encoding support, internal scratch and fixed numerical choices.
