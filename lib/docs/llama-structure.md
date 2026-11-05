# Llama structure ownership

`//lib/llama:structure` defines `LlamaStructure`, the immutable owner of a copied or
moved `ModelDescriptor` and the existing nonmovable `OperationGraph`. Consumers
retain `shared_ptr<const LlamaStructure>` so descriptor coordinates and graph
values remain valid. `BuildLlamaStructure` assembles the complete model through component helpers
and publishes its owner only after every append succeeds.

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
Complete factory and native root tests live in `//lib/llama/testing:model_structure_test`
and `//lib/llama/testing:program_test`. The latter accepts `LOOMIFY_TEST_GGUF`
for the immutable known-model join; it does not download weights.


## Complete configured commands

`BuildLlamaStructure` now assembles embedding, each descriptor vector slot's
normalized attention and normalized FFN, one fused last-active output head and
publication. Layer descriptor indices select tensor names, while vector slots
select cache planes. Output head owns final normalization, so no separate final
norm is added. Distinct logical token input and publication history identities
share their physical binding only during native lowering.

`model::ExtractLlamaConfiguration` borrows an unbuffered seekable `FILE*` and
returns the existing owned `ModelDescriptor`. It does not close the file, and
unsupported Llama Q/K normalization tensors reject before role information is
lost. Existing Qwen extraction is unchanged.

`DefineLlamaProgram` in `program.h` lowers the complete graph into a caller-owned
preloaded module. Load its registry closure with `LlamaProgramDispatches()` and
`LoadLlamaNative`, using matching layer/cache/pool configuration. Define prefill
and decode roots with separate calls to the same factory/lowerer. Public roots
have no specializations; row start and control origins are zero. Prefill capacity
is 1..512 with runtime active count 1..capacity; decode capacity is one. The
packet-derived count snapshot stays private and publication uses the original
packet. Existing registered adapters own variant selection.

The seven bindings are named parameters, residual F32 rows, original I32 control,
cache map, shared F16 KV arena, token/history I32 storage, and I32 progress.
`LlamaProgram::buffers` reports ordered byte minima; named parameter requirements
use normal reflection rather than raw GGUF offsets. `pool_capacity` is physical
token capacity, not a block count; zero selects dense storage. `cache_base` and
`map_origin` offset their respective arenas and are included in byte minima.
The result retains the logical owner and module-scoped program reference; no
command-body values escape. Failure leaves output unchanged. After a native
staging failure, discard the mutable module according to existing authoring
rules. Pipeline assembly and serving ownership remain with the exporter.
