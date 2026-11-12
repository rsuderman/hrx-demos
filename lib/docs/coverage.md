# Loomify component index

Start here when onboarding a model or extending an existing profile. This is the
maintained map of the owned pieces on this branch: where to call them, which
contracts constrain reuse, and where their tests live. The implementation,
BUILD closure and tests remain authoritative. This is documentation, not a
second operation registry or a generated-code format.

**Available** below means implemented in the linked source. It does not mean
every admitted dimension, arithmetic policy or device has independent numerical
qualification. Tests show concrete coverage, not a universal guarantee. Check
the caller's packing, precision, row/origin and state requirements before reuse.
Other branches and downloaded reference packages are not current coverage.

For onboarding, start with the target pipeline's required pieces and look them
up below. Record only candidate-specific reuse, extension, copied behavior and
missing pieces. Inspect uncertain matches at their owning source; different
dimensions or weight tags alone do not require a new model-specific operation.
Keep the architecture's layer loop and operation order in maintained Loom.

## Platform ownership

The maintained generation recipe is `amdgpu:gfx11-generic`. Its concrete target
requirements live in `lib/platforms/amdgpu/gfx11/targets.loom`; shared kernels
reference abstract schedule targets and need the selected provider when linked.
The Qwen H256/F16 wave64 implementation lives in that platform's
`attention_h256_schedule.loom`, built as the separate
`attention_h256_implementation` product. Qwen35 declares an H256 schedule
requirement; the existing platform mapping loads its concrete body alongside
the shared numerical product. Its family adapter still supplies model
configuration. Llama, Gemma and Qwen3 do not load this schedule.

Qwen3's scaled-F16 projection implementation is a separate selected native
product under `lib/platforms/amdgpu/gfx11/projection`: Q4_K store, Q6_K store
and fused Q4_K gate/up/SwiGLU. It owns the base64 matrix tiles, wave32 fragment
layout and launch geometry. Shared projection operations own dimensions,
preparation and encoding admission. The finite architecture requirement and
platform mapping load this product through the same numerical composition
helper used by the public builder; it is loaded for every Qwen3 capacity and
unused operations are removed by native materialization. Shared architecture
products declare platform-owned operations and leave their concrete bodies in
the selected product: compatible concrete definitions can follow native
precedence, so this is not an override or arbitrary-provider conflict verifier.
Qwen35 uses this same selection path for its H256 schedule; explicit raw-source
component fixtures include that schedule separately. Its shared production
numerical product retains the declaration and wrapper, with the implementation
provided only by the selected platform product.

The public builder and CLI select the recipe during generation; Serve selects
its actual device later. Component fixtures explicitly include the maintained
target provider. Model/configuration, codecs, arithmetic policy and other shared
numerical helpers are not duplicated per platform. These source boundaries enable
later providers; they do not qualify an unimplemented GPU family or backend.

## Shared numerical pieces

Use the owning operation/interface and its declared dependencies; do not import
an entire architecture merely to reach a helper. Some shared helpers retain
historical family names or paths. Their actual signature and body determine
reuse, not the name. `R` means read, `W` write and `RW` read/write; the effects
below summarize roles and do not grant unspecified aliasing or access regions.

| Piece / entrypoint | Source and build owner | Current contract and operand effects | Existing proof entrypoint |
|---|---|---|---|
| Quantized projection: `llm.projection.pack_input`, `llm.projection.project_packed` | [Composition/interface](../programs/projection/projection.loom), [BUILD](../programs/projection/BUILD.bazel) | Q4_K, Q5_0, Q5_K, Q6_K, Q8_0 selected by actual weight tag. R input/weight/count; W packed activations/output. Canonical and compatible policies have different active-row/packing semantics. See format bounds below. | [Native materialization](../programs/projection/testing/materialization_test.cc), [SOURCE/BC operation checks](../programs/projection/testing/operations_checks.loom), [BUILD](../programs/projection/testing/BUILD.bazel). |
| Matrix projection: `llm.projection.matrix`; fused gate/up: `llm.projection.gated_matrix` | [Shared operations](../programs/projection/matrix.loom), [input preparation](../kernels/projection/scaled_f16.loom), [half-FMA weight staging](../kernels/projection/f16_staging.loom), [selected gfx11 product](../platforms/amdgpu/gfx11/projection/BUILD.bazel) | Q4_K/Q6_K raw weights for matrix projection; fused gate/up currently Q4_K. Power-of-two-scaled F16 activations, half-rounded weight scale/bias followed by one half FMA for code*scale+bias, and F32 matrix accumulation/output. This differs from full F32 dequantization followed by one final half conversion. Static capacity T<=16 uses maintained canonical packed projection; T>=17 uses base64 WMMA with Tp=ceil(T/64)*64. Runtime active N only controls preparation/consumption: N1 decode with T128 still computes padded WMMA work. Input R only on active prefix; preparation W defines all Tp rows/scales, including exact zero padding; matrix W produces Tp output rows. No input/output alias. | [Independent scalar golden generator](../programs/projection/testing/matrix/fixtures.py), [component BUILD](../programs/projection/testing/matrix/BUILD.bazel): Q4/Q6 dyadic exact cases, actual Q/V/O/down dimensions and fused gate/up; bitwise preparation/padding witnesses across T17..512 and N1..511. [Explicit real-weight qualification helper](../programs/projection/testing/matrix/real_assets.py) requires immutable GGUF/index/tools; availability is separate from a particular model's full fidelity. |
| Shared matrix SwiGLU FFN: `llm.ffn.matrix_swiglu` | [Composition](../programs/ffn/matrix_swiglu.loom), [BUILD](../programs/ffn/BUILD.bazel), thin [Qwen3](../programs/qwen3/ffn.loom)/[Qwen2](../programs/qwen2/ffn.loom) adapters | RMS -> existing fused Q4_K gate/up/SwiGLU -> Q4_K/Q6_K down projection -> active residual update. R gamma/weights/count; RW residual; owns local intermediate buffers. Reuses existing matrix arithmetic policies and inactive-row handling; no model-wide FFN variant registry. | Existing Qwen3 layer/matrix checks and [Qwen2 configured-layer fixtures](../architectures/qwen2/BUILD.bazel). |
| QKV bias adapters: `llm.decoder.prepare_qk_bias_h128`, `llm.decoder.add_bias_active` | [Implementation](../kernels/decoder/qkv_bias.loom), [BUILD](../kernels/decoder/BUILD.bazel) | First: add F32 bias then full H128 NeoX RoPE, without head normalization; R projection/bias/epoch, W disjoint rotated output, inactive rows zero. Second: R bias/count, RW projected active values in place, inactive rows preserved; width256..32768 multiple256. Rows1..512; Q/K head count1..128. | [Independent bias/rotary fixtures](../kernels/decoder/testing/qkv_bias_fixtures.py), [GPU checks](../kernels/decoder/testing/BUILD.bazel). |
| Q4_K projection bodies | [q4k.loom](../kernels/projection/q4k.loom), [canonical](../kernels/projection/q4k_canonical.loom), [compatible](../kernels/projection/q4k_compat.loom) | Stored GGUF enum12, 256-value/144-byte blocks. Use through projection composition with matching activation packing and arithmetic policy. | Projection operation checks above; [FFN capability checks](../programs/ffn/testing/BUILD.bazel). |
| Q5_0 projection body | [q5_0.loom](../kernels/projection/q5_0.loom), [shared block32 codec](../kernels/projection/block32.loom), [shared launch/control](../kernels/projection/block32_launch.loom) | Enum6, 32-value/22-byte blocks; canonical packed-activation route. Distinct from Q5_K. | [Gemma operation checks](../programs/projection/testing/gemma3_checks.loom) and native materialization negatives. |
| Q5_K projection bodies | [q5k.loom](../kernels/projection/q5k.loom) | Enum13, 256-value/176-byte blocks; canonical operation route. This is not Q5_0. | Projection operation checks; [Qwen common checks](../kernels/qwen/testing/common/checks.loom). |
| Q6_K projection bodies | [q6k.loom](../kernels/projection/q6k.loom), [canonical](../kernels/projection/q6k_canonical.loom), [compatible](../kernels/projection/q6k_compat.loom) | Enum14, 256-value/210-byte blocks. Match reduction/packing policy, not just the storage tag. | Projection operation checks and FFN capability checks. |
| Q8_0 projection bodies | [q8_0.loom](../kernels/projection/q8_0.loom), [shared launch/control](../kernels/projection/block32_launch.loom) | Enum8, 32-value/34-byte blocks; consumes the operation's packed Q8_1_x4 activations, not arbitrary Q8 buffers. | Projection operation checks; Qwen common/output fixtures. |
| Activation packing and encoded-word helpers | [Quantization sources/BUILD](../kernels/quantization/BUILD.bazel), [common packed-word helper](../kernels/quantization/q8_1_x4.loom), [canonical](../kernels/quantization/q8_1_x4_canonical.loom), [MMVQ policy](../kernels/quantization/q8_1_x4_mmvq_compat.loom) | F32 input to Q8_1_x4 groups; native/MMQ/canonical arithmetic choices are part of the contract. R activation/count; W packed data and scale metadata. Width multiple128 in the public projection wrapper. | Projection checks and [compatible/canonical FFN fixtures](../programs/ffn/testing/BUILD.bazel). |
| Weighted RMS row: `llm.rmsnorm.row` | [rmsnorm_row.loom](../kernels/normalization/rmsnorm_row.loom) | Hidden width1..32768; explicit epsilon, row and output offset; F32 input/weight, F32 or F16 output. R input/gamma; W separate output row; source asserts no alias. It does not add1 to stored gamma. | [Llama norm checks](../kernels/llama/testing/registered_norm_checks.loom), Qwen common/FFN fixtures. |
| Residual row: `llm.residual_add_f32.row` | [residual_row.loom](../kernels/elementwise/residual_row.loom) | Width1024..32768, multiple4; rows1..512. R lhs/rhs; W selected output row. A caller aliasing output to an input is RW at that caller; verify its actual alias contract. Width1152 and input/output alias cases have focused coverage. | Llama norm checks and Qwen common fixtures; [elementwise launch wrappers](../kernels/elementwise/ffn_launches.loom). |
| SwiGLU row: `llm.swiglu_f32.row` | [swiglu_row.loom](../kernels/elementwise/swiglu_row.loom), [launch wrappers](../kernels/elementwise/ffn_launches.loom) | SiLU(gate)*up; FFN width256..32768, multiple256; rows1..512. R gate/up; W output row. Not GELU gating. | Llama norm checks, FFN capability/component fixtures. |
| GeGLU row: `llm.geglu_tanh_f32.row` | [geglu_row.loom](../kernels/elementwise/geglu_row.loom) | Tanh-GELU(gate)*up, explicit approximate-transcendental permission; width256..32768 multiple256. R gate/up; W output row. | [Row checks](../kernels/elementwise/testing/gemma3_checks.loom), independent Python F64 goldens with explicit F32 stage boundaries; unchanged fixed budgets. |
| Q8_0 scaled embedding gather | [q8_0_rows.loom](../kernels/embedding/q8_0_rows.loom) | Signed block32 decode and F32 scale/output, without an F16 intermediate. R tokens/table; W rows. The primitive requires caller-admitted IDs and does not guard vocabulary bounds. Gemma text admission rejects encoded out-of-vocabulary IDs before numerical execution. | [SOURCE/BC gather checks](../kernels/embedding/testing/q8_0_checks.loom), including signed endpoints, nonzero origins and guards. |
| F32 embedding gather | [f32.loom](../kernels/embedding/f32.loom) | R token/table; W one output row. Invalid token produces zero in this kernel; do not assume every embedding provider shares that policy. | [F32 checks](../kernels/embedding/testing/f32_checks.loom), [examples](examples.md). |
| Q4_K / Q6_K embedding rows | [q4k_rows.loom](../kernels/embedding/q4k_rows.loom), [q6k_rows.loom](../kernels/embedding/q6k_rows.loom), [BUILD](../kernels/embedding/BUILD.bazel) | Dequantize selected stored rows to F32. R weight/token; W output row. Row bytes/block divisibility and valid token admission belong to the caller/format contract. | [Q4_K checks](../kernels/embedding/testing/q4k_checks.loom), [Llama embedding checks](../kernels/llama/testing/embedding_checks.loom), Qwen output fixtures. |
| Online softmax and causal GQA row: `llm_attention_online_softmax_update`, `llm.attention.row_f32` | [core.loom](../kernels/attention/core.loom), [BUILD](../kernels/attention/BUILD.bazel) | Shared attention arithmetic and logical-to-physical cache traversal. R query/KV/map; W attention output. Caller owns head geometry, causal position, cache population and scale. `row_f32` preserves global attention; `row_window_f32` adds a causal logical window (0 means global), with full paged KV storage rather than ring allocation. | [Llama attention checks](../kernels/llama/testing/attention_sequence_checks.loom), [Qwen attention fixtures](../architectures/qwen35/testing/run_attention_partial_validation.py), [window checks](../kernels/attention/testing/gemma3_window_checks.loom) at 511/512 and page63/64 with nonidentity maps. |
| NeoX RoPE and Q/K RMS values: `llm.attention.rope_neox_value`, `llm.attention.head_rms_value` | [rotary.loom](../kernels/attention/rotary.loom), [normalization.loom](../kernels/attention/normalization.loom) | Configurable head/rotary geometry, frequency step, epsilon and position. R head/norm weights; returns scalar value. Helpers are reusable candidates; full Qwen attention carries additional sequence/layout policy. | Qwen attention fixtures and complete attention/layer checks. |
| KV position mapping: `qwen38_kv_position` | [kv_position.loom](../kernels/attention/kv_position.loom) | R map; returns physical pool position for a logical position. Mapping geometry/layout is explicit. This helper is not an allocator, eviction policy or sliding-window implementation. | Llama/Qwen attention and state fixtures. |
| Config-free ordinary decoder rows | [Rows and H128 variants](../kernels/decoder/rows.loom), [BUILD](../kernels/decoder/BUILD.bazel) | Active/last RMS: R count/input/gamma, W separate output; residual: R source/count, RW residual; Q6_K embedding: R immutable table/IDs/count/epoch, W compact F32 rows honoring descriptor B. `prepare_qk_h128` reads learned gamma/projections/epoch and writes Q/K; `publish_kv_h128` reads K/V/epoch/map and writes selected F16 cache rows; `attention_rows_h128` reads causal query/cache/map/epoch and writes attended rows. H128 is an explicit fixed-128-thread operation variant required by current native reduction/geometry lowering, not a checkpoint hidden-size constant. RMS/QK/attention define zero inactive output rows for full-T consumers; residual preserves inactive rows; cache writes only active logical positions. | [SOURCE/BC row, embedding and attention fixtures](../kernels/decoder/testing/BUILD.bazel), [independent scalar expectations](../kernels/decoder/testing/reference.py), [attention reference](../kernels/decoder/testing/attention_reference.py). Concrete GQA16:8/H128 geometry; broader admitted head/row bounds are not universal qualification. |
| Active-row predicate: `llm.active_row` | [active_rows.loom](../kernels/epoch/active_rows.loom), [BUILD](../kernels/epoch/BUILD.bazel) | Returns whether row>=begin and row-begin<count for a positive count. Nonnegative fixed_count bypasses the count buffer; negative fixed_count reads one i32 at the supplied byte origin. It does not impose count<=capacity. Used by Llama/Gemma wrappers; the ordinary decoder's different packet predicate stays separate. R dynamic count only; no payload reads or writes. | [SOURCE/BC predicate checks](../kernels/epoch/testing/BUILD.bazel) cover fixed/dynamic counts and row boundaries; existing model/operation fixtures retain device caller checks. |
| Matrix output extent: `llm.projection.matrix_output_rows` | [matrix.loom](../programs/projection/matrix.loom) | Pure inline helper for capacity T1..512: T rows when T<=16, otherwise ceil(T/64)*64. Shared matrix/gated-matrix preparation and Qwen3 attention/FFN scratch allocation use the same extent. This is a storage-capacity rule, not runtime active-row admission. | [SOURCE/BC output-row checks](../programs/projection/testing/matrix/BUILD.bazel) cover 1,16,17,63,64,65,511,512; existing matrix execution/preparation checks remain. |
| Greedy selection | [greedy_argmax.loom](../kernels/selection/greedy_argmax.loom), [BUILD](../kernels/selection/BUILD.bazel) | Row and tiled reductions, lowest-ID tie break; configured vocabulary capacity256..262144. R logits; W selected IDs/partials. Selection is separate from stop mapping/publication. Current hardware targets/profiles are declared in source/BUILD. | [Llama output/head checks](../kernels/llama/testing/output_checks.loom), [Qwen output checks](../kernels/qwen/testing/output). |

### Projection bounds are not whole-model bounds

The public [projection composition](../programs/projection/projection.loom)
currently admits launch/row capacity1..2048 and output width1..262144, subject
to generated capacity definitions and the selected kernel's contract:

| Stored weight format | Input width contract | Available policy route |
|---|---|---|
| Q4_K | 256..32768, multiple256 | Canonical and compatible |
| Q5_0 | 128..32768, multiple128 | Canonical |
| Q5_K | 256..32768, multiple256 | Canonical |
| Q6_K | 256..17408, multiple256 | Canonical and compatible |
| Q8_0 | 128..32768, multiple128 | Canonical |

These bounds admit candidates for testing, not universal arithmetic/device
qualification. Correct byte counts, input packing, count/origin behavior and
config capacities still apply. Model/FFN wrappers may impose narrower bounds.

## Composed programs and architecture-owned pieces

| Piece | Source/build entrypoint | Reuse boundary and existing tests |
|---|---|---|
| Shared complete FFN | [ffn.loom](../programs/ffn/ffn.loom), [interface](../programs/ffn/interface.loom), [BUILD](../programs/ffn/BUILD.bazel) | Pre-RMS, gate/up, SwiGLU, down and residual composition. Hidden1024..32768 multiple1024; FFN256..32768 multiple256; rows1..512, decode one row. Format/profile and gate/up identity constraints are in the implementation. It is not an arbitrary-activation/post-norm block. [Native/capability/real-FFN fixtures](../programs/ffn/testing/BUILD.bazel). |
| Last-active greedy head: `llm.output.last_greedy` | [Composition](../programs/output/last_greedy.loom), [BUILD](../programs/output/BUILD.bazel) | Calls the supplied `llm.output.norm_last` hook, compatible input packing and Q6_K projection, then partial/reduced argmax. R count/input/gamma/weight; W normalized scratch/private selected token. Llama and Qwen3 supply their own norm hooks and caller-owned scratch. This is not a head for every encoding or normalization. [Llama head](../programs/llama/output.loom), [Qwen3 head](../programs/qwen3/output.loom), [Llama head checks](../kernels/llama/testing/output_checks.loom). |
| Llama blocks | [attention/layer/output/FFN adapter](../programs/llama/BUILD.bazel), [provider closure](../kernels/llama/BUILD.bazel) | Maintained Llama ordering, native cache/packet bindings and numerical policies. The selected embedding tag admits Q4_K/Q6_K; the Q6_K last-row head binds tied or independent output from descriptor facts. Adjacent H128 RoPE uses the actual frequency tensor when present and unit divisors for ordinary RoPE otherwise. Layer format lookups preserve mixed Q4_K/Q6_K projections. No checkpoint-expanded sequence or separate branded numerical architecture. [Layer/model/preparation fixtures](../architectures/llama/BUILD.bazel), [configuration cases](../loomify/testing/configuration_test.cc). Availability and bounded execution are separate from independent full-model fidelity. |
| Qwen attention, GDN, FFN and output blocks | [program closure](../programs/qwen35/BUILD.bazel), [shared providers](../kernels/qwen/BUILD.bazel), [GDN kernels](../kernels/qwen/gdn/BUILD.bazel), [attention kernels](../kernels/qwen/attention/BUILD.bazel) | Qwen35 hybrid sequence and state semantics. Complete attention/GDN blocks retain internal FFNs. Recurrent/convolution state and updates are not generic KV attention. [Component fixtures](../kernels/qwen/testing), [model fixtures](../architectures/qwen35/testing/BUILD.bazel). |
| Gemma3 blocks | [Programs](../programs/gemma3/BUILD.bazel), [kernels](../kernels/gemma3/BUILD.bazel), [architecture](../architectures/gemma3/BUILD.bazel) | Gemma pre/post plain RMS, Q/K head RMS, full NeoX RoPE, local/global attention, GeGLU and post-norm residual sequence. Canonical all-T packing receives explicitly zeroed inactive RMS/GeGLU rows. [Poisoned padding checks](../kernels/gemma3/testing/padding_checks.loom) and direct whole-epoch window/state checks exercise this composition. |
| Qwen3 ordinary blocks | [Architecture](../architectures/qwen3/BUILD.bazel), [compact layer sequence](../programs/qwen3/layer.loom) | Pre-RMS, independent Q/K/V projections, weighted Q/K head RMS, full NeoX rotary, ungated causal GQA, attention output/residual, SwiGLU FFN and tied Q6_K head. Query width may differ from hidden width. Shared scalar and projection operations provide math; selected platform products provide WMMA schedules. The initial Q4_K_M fixture uses Q4_K/Q6_K matrix tags and H128. [Descriptor/configuration tests](../loomify/testing/qwen3_configuration_test.cc) check extracted facts/tensor bindings; full-model correctness is separate. |
| Qwen2 ordinary blocks | [Architecture](../architectures/qwen2/BUILD.bazel), [layer sequence](../programs/qwen2/layer.loom) | Qwen2.5 pre-RMS, Q/K/V projections plus F32 biases, full NeoX rotary with no Q/K head norm, ordinary causal GQA, output/residual and shared matrix SwiGLU FFN, tied Q6_K last-row head. Actual 1.5B Q4_K_M geometry E1536/Q1536/KV256/H128/F8960/28 layers. Per-layer V/down Q4_K/Q6_K tags come from GGUF; no unrolled checkpoint copy. | [Header/configuration/range tests](../loomify/testing/qwen2_configuration_test.cc), SOURCE/BC layer fixtures at mixed-format boundaries1->2 and4->5. Whole-model fidelity is separate. |
| Shared ordinary Qwen decoder | [Attention composition](../programs/decoder/attention.loom), [fact helpers](../loomify/architectures/ordinary_decoder.cc) | Qwen2 and Qwen3 adapters share `programs/decoder/attention.loom` (`llm.decoder.matrix_attention`), preserving seven scratch allocations, per-layer Q4/Q6 projections, KV publication, causal attention and residual order. Each selected product supplies one typed `llm.decoder.prepare_qkv`: Qwen2 bias-before-RoPE plus active V bias; Qwen3 learned QK RMS before RoPE. `loomify/architectures/ordinary_decoder.cc` shares matrix bindings and numerical/cache/preparation facts while family admission and learned tensors stay local. Focused Qwen2/Qwen3 configuration tests cover fixed `qwen38` capacity ownership, all mixed-layer tags/encoded extents, parameter order/ranges, nine neutral preparation facts, and distinct hidden-width bounds. |
| Embedding/output wrappers | [Llama Q6 embedding](../kernels/llama/embedding/embedding.loom), [Q4 epoch adapter](../kernels/llama/embedding/q4k_epoch.loom), [Llama head](../programs/llama/output.loom), [Qwen embedding](../kernels/qwen/output/embedding.loom), [Qwen head](../programs/qwen35/output.loom) | Own row selection, parameter layout and scratch around shared numerical operations. Q4 epoch gather reuses the shared Q4_K F32 row decoder, reads IDs at descriptor input_begin, writes compact active rows and preserves inactive rows. It does not introduce an F16 rounding stage. Tied weights are parameter aliases, not a reason to duplicate a table. Check architecture/configuration admission before borrowing a wrapper. |
| Selected-ID publication | [Llama](../kernels/llama/publication/publication.loom), [Qwen](../kernels/qwen/output/publication.loom) | R epoch/private selected IDs; W public output. Matching direct epoch contract permits one logical result at physical slot0..15, no-output preserves output. Llama's source-EOS policy supplies three extracted stop IDs: existing header EOGs or plain ChatML endoftext/im_end/endofturn, with canonical im_end distinct from declared endofturn EOS. The same kernel maps all three without changing the private raw winner. [Gemma publication](../kernels/gemma3/publication.loom) and Qwen3 use shared [two-stop mapping](../kernels/publication/two_stops.loom): a pure `i32` token/two-stop-ID/terminal operation. It maps either supplied source stop to the canonical terminal, preserving private raw IDs and nonstops. GGUF EOS and vocabulary-derived policy stop identities remain distinct; no numeric token IDs are embedded in the common helper. [Forced publication checks](../kernels/gemma3/testing/publication_checks.loom) cover both stops, nonstops and no-output; [Llama relocated-ID checks](../kernels/llama/testing/token_policy_checks.loom) cover all three numerical/host stop routes. Not a general sampler or multi-output implementation. [Epoch checks](../architectures/llama/testing/epoch_checks.loom), head/output component fixtures. |
| Model / epoch / preparation / execution | [Llama](../architectures/llama/BUILD.bazel), [Qwen35](../architectures/qwen35/BUILD.bazel), [Gemma3](../architectures/gemma3/BUILD.bazel), [Qwen3](../architectures/qwen3/BUILD.bazel) | Each public epoch owns private selected-ID scratch, calls private model computation, then publishes. Model owns layers/head/state effects; preparation owns allocations/stages; Llama/Gemma/Qwen3/Qwen2 preparation adapters retain only root/diagnostic strings; [ordinary preparation](../serving/ordinary_prepare.loom) reads the nine generated `loomify.serving.*` geometry facts directly, and [ordinary host control](../serving/ordinary_control.loom) remains shared; Qwen35 retains its own host policy. Shared [snapshot count](../kernels/epoch/count.loom) owns the 12-byte model count snapshot. Reuse this split; family layouts remain explicit. [Host contract test](../serving/testing/epoch_control_test.cc). |

Ordinary preparation tests share the neutral
[VM/link/record harness](../serving/testing/ordinary_prepare_harness.h).
Independent geometry, roots, markers and capacity cases belong to each
architecture: [Llama](../architectures/llama/testing/prepare_test.cc),
[Gemma3](../architectures/gemma3/testing/prepare_test.cc),
[Qwen3](../architectures/qwen3/testing/prepare_test.cc) and
[Qwen2](../architectures/qwen2/testing/prepare_test.cc).
Each owns its strict `:prepare_test` target and direct descriptor-fixture
dependency; [Qwen35](../architectures/qwen35/testing/prepare_test.cc) keeps
its distinct hybrid-storage suite and the existing preparation VM support.


Ordinary preparation reads the generated `loomify.serving.*` facts:
`hidden_size`, `layer_count`, `kv_size`, `history_capacity`, `token_capacity`,
`cache_capacity`, `pool_capacity`, `cache_base` and `map_origin`. The first seven
are index values; the last two are offsets. The common configuration writer
serializes already-admitted family geometry. The architecture adapter supplies
only `ordinary.epoch_root` and `ordinary.unsupported`; token policy
supplies the terminal marker. These feed the existing single-request pooled
allocation/stage records, not an arbitrary state layout. The concrete epoch and
publication commands remain architecture-owned.

## Text, tokenizer and serving policy

Select text behavior independently of numerical architecture. Similar marker
spellings are not proof of equivalent role, whitespace, tool or thinking rules.

| Piece | Owner | Available behavior and boundary |
|---|---|---|
| Byte comparison and assembly: `text.equal`, `text.at`, `text.range`, `text.join8`, `text.append` | [strings.loom](../text/common/strings.loom), [common BUILD](../text/common/BUILD.bazel) | Equal compares whole buffers; at checks a marker at a byte offset; range copies a caller-admitted span; join8 concatenates eight buffers; append copies into caller-owned output and returns its end offset. These manipulate bytes, not token IDs. Used by the maintained renderers and output parsers below. |
| Whitespace: `text.trim`, `text.trim_lf` | [strings.loom](../text/common/strings.loom) | Trim strips the implementation's whitespace set at both ends; trim_lf removes only LF at selected ends, preserving spaces/tabs. Reuse only where the target template requires that behavior. [Renderer reference cases](../text/testing/render_reference.py) and [Qwen3 cases](../text/qwen3/testing/render_cases.json) exercise protocol whitespace; these are not standalone proofs of every helper input. |
| JSON access/serialization: `text.lookup`, `text.string`, `text.json_quote`, `text.json_dump` | [String access](../text/common/strings.loom), [JSON helpers](../text/common/json.loom) | Lookup returns raw member bytes/type/presence; string decodes a present string member. Quote escapes a string; dump formats a JSON document using the maintained policy. No Python-compatible floating-point serialization claim. [JSON tests](../text/qwen35/testing/json_test.cc), [tool output tests](../text/qwen35/testing/tool_output_test.cc). |
| Shared model identity: `model_name`, `chat_image_limits` | [identity.loom](../text/common/identity.loom) | Common hooks consume linked model-name data and declare no image support. Reuse the selected text product/common closure rather than copying identity code into each architecture. |
| Llama header protocol | [Implementation](../text/protocols/llama_headers/input.loom), [BUILD](../text/protocols/llama_headers/BUILD.bazel), [reference fixture](../text/llama/templates/5816fce1.jinja) | Known header/date framing, bounded plain-text policy and plain output/continuation hooks. The fallback date and two system prose literals are captured from the GGUF template; the generated preamble is consumed by both chat start and fresh-user input preparation. No general Gemma/ChatML formatter. Some output helpers trim content; verify semantics before extracting them. |
| ChatML thinking protocol | [Common sources and selected policies](../text/protocols/chatml_thinking/BUILD.bazel), [inline-history template](../text/qwen35/templates/7f0e5290.jinja), [explicit-reasoning template](../text/qwen35/templates/c3cf9e34.jinja) | Shared admission/input/output/tool mechanics with `inline_history` versus `explicit_reasoning` products. [Renderer tests](../text/qwen35/testing/revision_cases.cc), [stream parser](../text/qwen35/testing/output_test.cc), [tool parser](../text/qwen35/testing/tool_output_test.cc), [independent references](../text/testing/render_reference.py). |
| Qwen3 ChatML/Hermes policy | [Selected Hermes product](../text/protocols/chatml_thinking/BUILD.bazel), [known template skeleton](../text/qwen3/templates/5da44855.jinja) | Exact supported GGUF skeleton selection, with two known prose slots captured from actual GGUF bytes. Reuses shared ChatML/string/strict-mapping mechanics; preserves whitespace, later system messages, histories without a real user, first-tool framing and string tool arguments verbatim. Default thinking renders the generation header only; optional model-generated think opener is handled incrementally. Hermes JSON tool calls differ from Qwen35 XML calls. Qwen3 history bound is 268435455; existing providers retain their own bounds. No arbitrary Jinja, media or exact floating-point tool JSON parity. [Renderer/parser tests](../text/qwen35/BUILD.bazel), [independent Qwen3 cases](../text/qwen3/testing/render_cases.json), [template selection tests](../loomify/testing/chat_policy_test.cc) and token-policy tests below are the proof entrypoints; implementation/fixture availability does not imply full SDK or Serve qualification. |
| Qwen2.5 nonthinking ChatML/Hermes policy | [Selected product](../text/protocols/chatml_thinking/BUILD.bazel), [known template](../text/qwen25/templates/cd8e9439.jinja) | Reuses ChatML framing, Hermes output parsing and native gpt2/qwen2 tokenization. Captures default-system prose from actual GGUF; supports full-history and fresh-user input modes, never starts reasoning, preserves template string-to-JSON tool-argument behavior and rejects empty history. Source-derived `im_end`/`endoftext` stops share two-stop publication; declared BOS/PAD does not request BOS insertion. | Selected [renderer/reference cases](../text/qwen25/BUILD.bazel), shared parser and token-policy tests. No arbitrary Jinja or exact floating-point tool JSON parity. |
| Plain ChatML policy | [Selected plain product](../text/protocols/chatml_thinking/BUILD.bazel), [exact template58c1a1f0](../text/plain_chatml/templates/58c1a1f0.jinja), [local cases/oracle adapter](../text/plain_chatml/BUILD.bazel) | Exact known no-slot revision with gpt2/dbrx tokenizer; no BOS insertion, default system, thinking or tool interpretation. Reuses shared ChatML framing; plain output preserves bytes and literal markup. Ordinary nonempty system/user/assistant string histories are the admitted host domain; unsupported tools/reasoning/media reject without changing old policies. Raw template missing-generation defaults false, while Serve's existing default supplies true; explicit false omits the assistant header. Source-declared endofturn EOS and canonical im_end remain distinct. [Matcher cases](../loomify/testing/chat_policy_test.cc), [token binding cases](../loomify/testing/token_policy_test.cc) and local SOURCE/BC renderer/output suites define proof entrypoints; actual execution qualification is separate. |
| Gemma turns protocol | [Implementation](../text/protocols/gemma_turns/BUILD.bazel), [template](../text/gemma/templates/7de1c58e.jinja) | BOS/turn framing, plain UTF8 output preserving visible whitespace, canonical future-history replay with reuse disabled. No tools, reasoning or media requests. [Independent HF reference driver](../text/gemma/testing/reference_adapter.py) and native SDK SOURCE/BC tests cover 24 positive/8 template-error cases, domain rejection and output/replay. |
| Known-template recognition / data capture | [chat_policy.h](../loomify/chat_policy.h), [selector](../loomify/chat_policy.cc), shared [matching/capture](../loomify/text/matching.cc), protocol-local [Llama](../loomify/text/llama_headers.cc), [ChatML](../loomify/text/chatml_thinking.cc), [Gemma](../loomify/text/gemma_turns.cc) | Seven maintained semantic skeletons; selected GGUF bytes supply supported literal/default slots. The five ChatML entries select revision-local capture/emission callbacks and explicit ordered marker metadata; shared selection retains candidate-only publication and exact unknown-structure rejection. [Focused matcher tests](../loomify/testing/chat_policy_test.cc) also cover all ChatML marker/token requirements, Hermes generated data/terminal ownership and source-derived effort defaults. Llama captures the fallback date, knowledge-cutoff prefix and date label into generated data; its [renderer](../text/protocols/llama_headers/input.loom) consumes the assembled system preamble at chat start and fresh-user input preparation. Existing structural admission and slot limits remain unchanged. Explicit named-template selection. Unknown structural variants reject. [Selection/mutation tests](../loomify/testing/chat_policy_test.cc). Arbitrary Jinja compilation is not available. |
| GGUF text/token facts | [text_metadata.h](../../tools/model/text_metadata.h), [implementation](../../tools/model/text_metadata.cc) | Independent template issues, roles/IDs, insertion flags and selective vocabulary/merge spans; no full tokenizer reconstruction. [Tests](../../tools/model/testing/text_metadata_test.cc). |
| Native tokenizer association / consumed markers | [token_policy.h](../loomify/token_policy.h), [implementation](../loomify/token_policy.cc) | Uses the existing native tokenizer JSON parser. Explicit selected-text requirements admit the supported Llama/Qwen3/Qwen2/Qwen35/Gemma tokenizer combinations and consumed special identities; marker agreement is not full ordinary-vocabulary/tokenization parity. Renderer-owned BOS insertion and optional numerical publication facts are checked independently. Declared EOS and canonical terminal remain separate; Gemma source vocabulary bounds reject encoded added-media IDs before numerical execution. Qwen3 and Qwen2 use the native gpt2/qwen2 association, no BOS insertion, declared `im_end` EOS and a vocabulary-derived `endoftext` second stop, both published as canonical `im_end`. Native one-shot/incremental decode fixtures exercise byte fallback and special-token flags. [Tests](../loomify/testing/token_policy_test.cc), [real-asset tests](../loomify/testing/token_policy_real_assets_test.cc). |
| Greedy request options | [greedy.loom](../serving/greedy.loom), [request_check.loom](../serving/request_check.loom) | Shared current request/default validation. Model heads perform argmax; this does not implement temperature/top-k/top-p sampling or RNG. |
| Host interface and resource products | [control_interface.loom](../serving/control_interface.loom), [product inventory](../serving/BUILD.bazel) | Existing hook contract and declared products for native linking; source presence alone does not register a new architecture/text provider. [Assembly tests](../serving/testing/assembly_contract_test.cc). |

Native renderer support is maintained in
[the neutral test harness](../text/testing/render_harness.h).
Local [Qwen35](../text/qwen35/testing/revision_cases.cc),
[Qwen3](../text/qwen3/testing/revision_cases.cc),
[Qwen2](../text/qwen25/testing/revision_cases.cc),
[Llama](../text/llama/testing/revision_cases.cc) and
[Gemma](../text/gemma/testing/revision_cases.cc) suites own their independent
expectations and SOURCE/BC cases. Qwen2 and Qwen3 reuse
[Hermes protocol properties](../text/protocols/chatml_thinking/testing/hermes_policy.h).
The [independent pinned reference runner](../text/testing/render_reference.py)
requires explicit local adapter, cases, immutable template/tokenizer and output
paths; original-GGUF template input uses the explicit `--source-override` flag.
The [maintained external consumer template](../text/testing/consumer/BUILD.bazel)
selects these test-only source sets. Its actual integration entrypoint remains
`//internal_consumer:existing_render_test` with the recorded SDK-private visibility
exception, isolated from strict production/product/preparation builds.
Existing output, tool-output and JSON tests retain their paths and coverage.

Token-to-byte normalization/decoding belongs to the existing native tokenizer;
protocol output parsers classify content/reasoning/tool streams. A new tokenizer
JSON may use existing implementation yet fail our current narrow policy gate.
Test the parser and actual encoding/decoding before declaring an engine missing.
Declared EOS, template end-of-turn, PAD, BOS and publication mapping remain
distinct. Multiple-stop behavior is not established simply by adding marker data.

For text onboarding, follow the actual GGUF template through
`SelectChatPolicy`: its maintained revision entry identifies the reference
fixture, tokenizer requirements and selected text product. Matching captures
admitted literal/default slots into linked configuration data; the selected
protocol supplies rendering and output parsing. Start from the protocol entries
above, then inspect this selection code for an uncertain match. Similar marker
strings do not establish equivalent role, whitespace, tool or reasoning rules.

## Extraction, configuration and packaging

| Piece | Owner | Current coverage |
|---|---|---|
| Header/tensor catalog | [gguf_catalog.h](../../tools/model/gguf_catalog.h), [implementation](../../tools/model/gguf_catalog.cc) | F32/F16/Q8_0/Q5_0/Q4_K/Q5_K/Q6_K encoded size/range interpretation. [Tests](../../tools/model/testing/gguf_catalog_test.cc). |
| Logical model descriptor | [model_descriptor.h](../../tools/model/model_descriptor.h), [shared reader](../../tools/model/model_descriptor.cc), [family loaders](../../tools/model/architectures), [tensor-name helper](../../tools/model/tensor_names.h) | Llama, Qwen3, Qwen2, Qwen35 and Gemma3 role/shape/layer/state mapping; family loaders own metadata/layout differences, while loading and configuration share role-to-checkpoint-name spelling; logical admission is separate from numerical/provider support. Gemma effective local-window/global RoPE metadata is required and retained; other family restrictions remain explicit. [Tests](../../tools/model/testing/model_descriptor_test.cc). |
| Generated configuration/bindings | [configuration.h](../loomify/configuration.h), [common producer](../loomify/configuration_internal.h), [family mappers](../loomify/architectures) | Numeric facts, named tensor bindings and pure layer data lookups for supported families. The common producer retains constructed-descriptor name/shape/encoding/range checks. Ordinary Llama/Gemma3/Qwen3/Qwen2 preparation uses one shared typed fact writer; numerical configuration symbols remain family-owned. Token/prefill capacities remain<=512; logical context capacity is separate. Gemma tests use context1024 packages and context576 direct window/state witnesses; family restrictions remain explicit. [Tests](../loomify/testing/configuration_test.cc). |
| Architecture selection | [architectures.h](../loomify/architectures.h), [finite entries](../loomify/architectures.cc), [resource BUILD inventory](../serving/BUILD.bazel) | Each entry selects explicit implementation/preparation/execution products, its data-only configuration mapper, source-only publication policy and existing context limit (Llama131072; others262144). Qwen3 and Qwen2 require the selected scaled-F16 projection product; Qwen35 requires H256 prefill. Geometry/encoding admission stays in the mapper and native providers. `architecture_selection` owns these policies; configuration/token-policy facades depend on it, while generic native linking depends only on `architecture_types`. |
| Source publication facts | [publication.h](../loomify/publication.h), [collector](../loomify/publication.cc), [family policies](../loomify/architectures) | One vocabulary scan validates unique source stop identities, int32 IDs and declared EOS agreement, then emits family-prefixed configuration transactionally. Gemma EOS differs from its turn terminal; Qwen2/Qwen3 share ChatML collection with separate prefixes; Qwen35 succeeds without FILE/vocabulary/publication facts. No native tokenizer dependency. Existing [token-policy tests](../loomify/testing/token_policy_test.cc) cover this separately from JSON binding. |
| Builder / native link / materialization | [Public API](../loomify/gguf_model_builder.h), [linking](../loomify/linking.cc), [maintained passes](../loomify/materialize.loom) | Open -> Configure -> BuildModule/WriteModule or WritePackage. Configuration can be inspected before product loading; numerical-only generation is independent of chat/tokenizer admission. [Builder tests](../loomify/testing/public_builder_test.cc). |
| Package writer / converter | [package.cc](../loomify/package.cc), [CLI](../../tools/loomify-gguf/main.cc), [guide](gguf-export.md) | Five-file source/BC Serve package, external GGUF/tokenizer references, safe new-directory publication. Use the existing path for a new family. [Package tests](../loomify/testing/package_test.cc). |

A new family adds its loader and scoped tensor-name exceptions beside
`tools/model/architectures/<family>.cc/.h`, then one entry in
[the model-reader list](../../tools/model/architectures.cc). Common parsing,
role spelling, tensor binding and output rollback remain shared; the reader has
no dependency on Loomify or Serve. Add its configuration mapper/publication
policy in `lib/loomify/architectures/<family>.cc/.h` and one entry in
[the generation selector](../loomify/architectures.cc). Declare sources and
numerical products explicitly in the relevant BUILD files and resource inventory;
these two finite entries do not implement a pipeline or widen admission. Extend
focused descriptor/configuration/publication tests and architecture-local cases.
Neither layer uses runtime registration or provider discovery.

Model-level package generation exists for the supported Llama, Qwen35, Gemma3
and Qwen3 profiles described in the [export guide](gguf-export.md), with the
component and model proof entrypoints linked above.
Independent full-model fidelity remains a separate qualification; successful
linking or a plausible response must not upgrade a component's numerical claim.

Qwen3 uses an ordinary Q/K-head-RMS, full NeoX rotary, ungated GQA and
SwiGLU sequence with independent query and residual widths (the initial fixture
has E1024, Q2048, KV1024, H128 and F3072). Its compact architecture calls shared
projection/scalar operations rather than copying the downloaded unrolled model.
Its matched tied Q6_K last-row head uses the shared
[greedy composition](../programs/output/last_greedy.loom) and explicit norm hook;
that head retains its existing compatible packing policy, distinct from the new
canonical short-row matrix route. Common ordinary preparation is maintained in
[ordinary_prepare.loom](../serving/ordinary_prepare.loom), with family adapters
supplying facts. Hardware-specific matrix scheduling is not GGUF configuration.

## Qualification boundaries

The Gemma components, source-driven configuration, package construction and native
text/host contracts have concrete tests. Direct whole-epoch tests compare active
outputs and full state across SOURCE/BC, offsets, dependent decode and reset;
excluded local-prefix invariance and included local/global-prefix sensitivity
exercise window ownership. Their synthetic history is not an independent
full-model accuracy oracle. Standalone numerical checks retain independent
expected values and fixed budgets. Live model semantic qualification remains a
separate gate from linking, bitwise ownership and prompt plausibility.

Arbitrary Jinja compilation, tokenizer reconstruction from GGUF alone, media,
sampling and general reasoning/tool support are not implied. A reusable operation
still needs its own actual geometry, precision and device qualification. Keep
candidate-model inventories and speculative implementation plans outside this
index; promote entries only with owned source, a real caller and relevant proof.

## Keeping the index current

1. **Before scoping:** map the candidate's requirements to these entries. Read
   only the relevant source/contracts/tests and changed pieces; do not repeat a
   whole-repository inventory merely because a new checkpoint is selected.
2. **With a coverage change:** update the affected entry in the same change as
   a new/removed provider, encoding, public template, dimension/active-row/alias
   contract, numerical policy, text revision, token/stop behavior or resource
   registration. Adjust proof links and known gaps. New values inside an already
   documented supported contract need no artificial catalog churn.
3. **During review:** compare the implementation delta with this index. Check
   format/geometry, read/write/state semantics, target/packing policy and actual
   tests. A newly admitted bound without numerical evidence must say so; never
   replace a limit with an unqualified “generic” claim.
4. **At integration:** land the index update with the corresponding source and
   validation. Keep machine-specific logs/hashes, Beads status and investigation
   notes in the external workspace. The committed document describes its own
   source tree, so it does not need an ever-changing checked-commit field.

Do not maintain duplicate capability tables in onboarding plans or Beads. Link
the stable component sections here, recording only candidate-specific gaps,
decisions and additional required evidence there. Source/BUILD/test renames must
update these links along with their ordinary callers.
