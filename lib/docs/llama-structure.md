# Llama structure ownership

`//lib/llama:structure` defines `LlamaStructure`, the immutable owner of a copied or
moved `ModelDescriptor` and the existing nonmovable `OperationGraph`. Consumers
retain `shared_ptr<const LlamaStructure>` so descriptor coordinates and graph
values remain valid. `BuildLlamaStructure` is a reserved complete-model signature;
this checkpoint does not implement full model construction or publish partial
models through that API.

Internal `detail::StructureConstruction` supports the component recipe. It checks
Llama dense geometry, phase, positive concrete row/cache/history capacities and a
logical source-asset key before construction. Runtime row offsets/counts and
absolute positions are separate typed inputs. The declared cache planes are
separate mutable F16 values per layer; physical cache maps, initialization and
runtime scalar bounds remain preparation/binding/execution obligations. Tests
explicitly retain incomplete fixtures through a test-only peer.

Parameter associations retain global or layer/tensor vector coordinates into the
owned descriptor. A logical source asset plus tensor name identifies a parameter;
repeated records must agree on normalized extents, numeric storage, absolute offset
and byte size. Diagnostic storage names do not affect identity. Consistent tied
role records and repeated FFN records share one graph value. Matrix admission uses
one checked Q4_K/Q6_K/Q8_0 mapping, with dense F32 vectors for normalization and
factors. No physical allocation or native-provider capability is inferred.

The recipe records known embedding, attention, FFN, layer and output operation
ranges. Public selection accepts these records or contiguous consecutive layers;
absent blocks and overflowing/out-of-range layer selections fail without changing
caller output. Region outside ports derive from ordered read-before-write,
downstream reads before write-only replacements, and explicitly declared final result/state
exports. Matching active-row domains can define an internal temporary; a different
span cannot hide an outside read. A write-only update invalidates prior coverage;
ReadWrite preserves inactive definitions. Matching active producers cover CopyRows
Range and LastActive source reads, and LastActive row zero fully defines a one-row
destination for the head. Per-operation access domains remain on the graph;
a region's outside interface describes value identities and access modes, not a
replacement range language or runtime initialization proof.

Run `//lib/llama/testing:structure_test` and `//lib/llama/testing:ffn_structure_test`.
These native descriptor fixtures verify ownership, aliases, spans and region cuts.
Embedding, attention and output helpers, complete-model publication, numerical
binding and lifecycle qualification remain subsequent work.
