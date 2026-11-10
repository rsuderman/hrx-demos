# Library documentation

Loomify reads supported GGUF model facts and links maintained Loom architecture
and numerical modules. C++ owns configuration, native linking and export; Loom
owns model sequencing, temporary storage and preparation records.

- [GGUF export](gguf-export.md): the public module/package builder, CLI and supported profiles.
- [Component coverage](coverage.md): available numerical programs, kernels, text/token policies and generation pieces, their limits and test entrypoints. Start here for model onboarding; update affected entries with coverage changes in the same source change.
- [Examples](examples.md): actual Loomcxx imports and the independent F32 embedding example.

`lib/architectures/{llama,qwen35,gemma3}` contains maintained model, preparation and
execution programs. `lib/text/protocols` owns independently selected chat framing,
`lib/text/common` owns shared text helpers and `lib/serving` owns the shared greedy
request policy. Complete Qwen attention/GDN layers retain their internal FFNs.
`lib/programs/ffn` and `lib/kernels` own reusable numerical sequences and kernels;
model configuration selects existing qualified encoding/arithmetic policies.

Tests live with their implementation. Architecture tests cover complete command
roots, parameter identities, preparation records and packet/state behavior.
Kernel tests retain independent arithmetic/precision/guard expectations, active
rows and continuation. `lib/loomify/testing` covers the public builder,
header-only GGUF facts, relocation and safe package publication. Prompt/decode
acceptance uses existing Loom Serve; it does not establish independent numerical
parity for every model.
