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
| Quantized projection: `llm.projection.pack_input`, `llm.projection.project_packed` | [Composition/interface](../programs/projection/projection.loom), [BUILD](../programs/projection/BUILD.bazel) | Q4_K, Q5_0, Q5_K, Q6_K, Q8_0 selected by actual weight tag. R input/weight/count; W packed activations/output. Canonical and compatible policies have different active-row/packing semantics. See format bounds below. | [Native materialization](../programs/projection/testing/materialization_test.cc), [build-generated independent operation checks](../programs/projection/testing/fixtures.py), [BUILD](../programs/projection/testing/BUILD.bazel). |
| Matrix projection: `llm.projection.matrix`; shared QKV: `llm.projection.matrix_qkv`; gate/up: `llm.projection.gated_matrix` | [Shared operations](../programs/projection/matrix.loom), [input preparation](../kernels/projection/scaled_f16.loom), [half-FMA weight staging](../kernels/projection/f16_staging.loom), [selected gfx11 product](../platforms/amdgpu/gfx11/projection/BUILD.bazel) | Q4_K/Q6_K raw weights for matrix projection; fused gate/up currently Q4_K. Power-of-two-scaled F16 activations, half-rounded weight scale/bias followed by one half FMA for code*scale+bias, and F32 matrix accumulation/output. This differs from full F32 dequantization followed by one final half conversion. Static capacity T<=16 uses maintained canonical packed projection; T>=17 uses base64 WMMA with Tp=ceil(T/64)*64. Q/K/V share one prepared input while retaining independent tags/widths; short gate/up shares one prefix copy and canonical pack. Ordinary decode uses its separate T1 stage. With a larger prefill capacity, runtime active N controls preparation/consumption. Count-guarded providers skip wholly inactive 64-row tiles; partial-tile output stores mask inactive rows to exact positive zero while preserving active-row arithmetic. Exact zero for every inactive/padded row is the accepted contract, including cases where the previous unguarded half-weight staging produced inactive NaNs. Numerical and performance measurements for this change remain pending. Input R only on active prefix; preparation W defines all Tp rows/scales, including exact zero padding; matrix W produces Tp output rows. No input/output alias. | [Independent scalar golden generator](../programs/projection/testing/matrix/fixtures.py), [component BUILD](../programs/projection/testing/matrix/BUILD.bazel): Q4/Q6 dyadic exact cases, actual Q/V/O/down dimensions and fused gate/up; bitwise preparation/padding witnesses across T17..512 and N1..511. [Shared preparation fixtures](../programs/projection/testing/matrix/shared_preparation_fixtures.py) add exact nonzero mixed-format QKV, short gate/up and packed-byte checks with independent anchors and matched bundle benchmark declarations. [Explicit real-weight qualification helper](../programs/projection/testing/matrix/real_assets.py) requires immutable GGUF/index/tools; availability is separate from a particular model's full fidelity. |
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
| F32 epoch embedding: `llm.embedding.epoch_f32` | [epoch_f32.loom](../kernels/embedding/epoch_f32.loom), [source closure](../kernels/embedding/BUILD.bazel) | One command-level operation admits format12/Q4_K,13/Q5_K or14/Q6_K and full-block geometry, then issues the private neutral kernel with one256-thread group per candidate row and unchanged F32 row decoder. Admission precedes device launch so numerical-only/full-module export retains no unresolved format decisions. R epoch/count/IDs/table, W compact active output rows; IDs begin at descriptor input_begin while output starts row0, inactive rows preserve prior bytes. The operation owns count/address/encoding logic; its shared table contract admits rows1..2048, full256-value blocks, positive signed-I32 vocabulary identities and exact encoded footprint with checked signed-offset products. It no longer truncates valid IDs at128000 or rows at512. Consumed Llama Q4/Q6 epoch entrypoints are thin adapters. The tag-selecting [ordinary gather](../kernels/embedding/ordinary_f32.loom) shares the table/decode helpers but zeroes inactive or invalid-position rows and remains a different contract. Independent [build-generated epoch/ordinary cases](../kernels/embedding/testing/epoch_contract_fixtures.py) include high IDs, rows513, shifted positions, poisoned unused IDs and zero/negative/oversized counts. | [Q4 independent SOURCE/BC fixtures](../kernels/llama/testing/plain_embedding_checks.loom), [Q6 independent epoch fixtures](../kernels/llama/testing/embedding_checks.loom) exercise generic and compatibility entrypoints with untouched input/weight/canary expectations; [model composition](../architectures/llama/testing/configured_model_test.cc) checks selected geometry/tag and unsupported encoding. |
| Online softmax and causal GQA row: `llm_attention_online_softmax_update`, `llm.attention.row_f32` | [core.loom](../kernels/attention/core.loom), [BUILD](../kernels/attention/BUILD.bazel) | Shared attention arithmetic and logical-to-physical cache traversal. R query/KV/map; W attention output. Caller owns head geometry, causal position, cache population and scale. `row_f32` preserves global attention; `row_window_f32` adds a causal logical window (0 means global), with full paged KV storage rather than ring allocation. | [Llama attention checks](../kernels/llama/testing/attention_sequence_checks.loom), [Qwen attention fixtures](../architectures/qwen35/testing/run_attention_partial_validation.py), [window checks](../kernels/attention/testing/gemma3_window_checks.loom) at 511/512 and page63/64 with nonidentity maps. |
| NeoX RoPE and Q/K RMS values: `llm.attention.rope_neox_value`, `llm.attention.head_rms_value` | [rotary.loom](../kernels/attention/rotary.loom), [normalization.loom](../kernels/attention/normalization.loom) | Configurable head/rotary geometry, frequency step, epsilon and position. R head/norm weights; returns scalar value. Helpers are reusable candidates; full Qwen attention carries additional sequence/layout policy. | Qwen attention fixtures and complete attention/layer checks. |
| Raw H256 Q/K preparation: `llm.attention.prepare_qk_raw_h256` | [raw_rows.loom](../kernels/attention/raw_rows.loom), [BUILD](../kernels/attention/BUILD.bazel) | Rows1..512, heads1..128, positive even rotary width<=256, logical capacity<=262144. Applies full/partial NeoX rotary directly to projections with the supplied frequency step; no gamma, bias or factor tensor. R epoch/projection; W disjoint F32 rows, zero inactive rows, preserve the nonrotary tail. | Independent [SOURCE/BC checks](../kernels/attention/testing/raw/BUILD.bazel) include H256 query/key geometry, positions0/63/64/511/512/1023 and inactive rows; admitted geometry is broader than these witnesses. |
| Fused QKV row split: `llm.decoder.split_qkv` | [split_qkv.loom](../kernels/decoder/split_qkv.loom), [decoder BUILD](../kernels/decoder/BUILD.bazel) | Token-major fused width Q+2KV to compact disjoint F32 Q/K/V rows. Q and KV widths256..32768, multiple256; rows1..512. R count/active fused rows; W outputs, zero inactive rows. No synthetic checkpoint aliases or weight slicing. | Guarded independent [SOURCE/BC split cases](../kernels/decoder/testing/split/BUILD.bazel) cover count0/1/partial rows and Q/K/V boundaries. |
| H128/H256 KV publication and gated window attention | [publisher](../kernels/decoder/rows.loom), [gated wrappers](../kernels/attention/factored_rows.loom) | Static H128/H256 entrypoints share bodies while retaining fixed launch geometry. Publication R K/V/epoch/map, W only selected F16 cache positions. Gated attention R Q/F32 per-head gate/epoch/map/cache, W attended F32 rows with inactive zeros. Window0 is full causal history; positive windows restrict reads over full paged storage. | Independent [H256 state/publication cases](../kernels/attention/testing/spark_state/BUILD.bazel) cover local/global causality, head gating, page/window boundaries and packet guards; existing [H128 LFM3 cases](../kernels/attention/testing/lfm3/BUILD.bazel) preserve factor-aware behavior. |
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

The complete FFN admits hidden widths1024..32768 in multiples256, intersected
with each selected projection's bounds. Compatible/profile0 and canonical/profile1
packing recipes remain distinct. The [capability fixtures](../programs/ffn/testing/BUILD.bazel)
include E1536/F4608 with compatible Q4/Q6 down and E2560/F9216 with canonical
Q4/Q6 down, including partial and empty counts. Canonical FFN arithmetic defines
all statically launched rows of private scratch regardless of active count;
compatible FFN preserves inactive rows. Active-count guards on persistent state
and KV writes remain separate from private scratch semantics.
Shared residual launchers carry the exact
logical width separately from their ceiling1024-channel grid; [tail fixtures](../kernels/elementwise/testing/BUILD.bazel)
exercise widths1024/1536/2560/3072, multiple rows, input aliases and guards.

| Piece | Source/build entrypoint | Reuse boundary and existing tests |
|---|---|---|
| Shared complete FFN | [ffn.loom](../programs/ffn/ffn.loom), [interface](../programs/ffn/interface.loom), [BUILD](../programs/ffn/BUILD.bazel) | Pre-RMS, gate/up, SwiGLU, down and residual composition. Hidden1024..32768 multiple256; FFN256..32768 multiple256; rows1..512, decode one row. Format/profile and gate/up identity constraints are in the implementation. It is not an arbitrary-activation/post-norm block. [Native/capability/real-FFN fixtures](../programs/ffn/testing/BUILD.bazel). |
| Canonical mixed-format SwiGLU: `llm.ffn.canonical_swiglu` | [canonical_swiglu.loom](../programs/ffn/canonical_swiglu.loom), [BUILD](../programs/ffn/BUILD.bazel) | Gate/up/down select independent actual tags through canonical row projection; owns gate/up/activation scratch. R normalized input/count/weights; W down output. Callers define inactive input rows and own RMS/postnorm/residual placement; LFM3 and Spark retain distinct layer ordering. Rows1..512, FFN256..32768 multiple256, subject to each projection's encoding/input bound above. [Independent mixed-format SOURCE/BC cases](../programs/ffn/testing/spark/BUILD.bazel). |
| Last-active greedy head: `llm.output.last_greedy` | [Composition](../programs/output/last_greedy.loom), [BUILD](../programs/output/BUILD.bazel) | Calls the supplied `llm.output.norm_last` hook, explicit packing policy and weight tag, then partial/reduced argmax. Existing Llama/Qwen2/Qwen3 callers retain compatible Q6_K; LFM3 uses canonical Q5_K; Spark supplies generated output tags and tied/untied bindings with canonical policy. R count/input/gamma/weight; W normalized scratch/private selected token. Llama and Qwen3 supply their own norm hooks and caller-owned scratch. This is not a head for every encoding or normalization. [Llama head](../programs/llama/output.loom), [Qwen3 head](../programs/qwen3/output.loom), [Llama head checks](../kernels/llama/testing/output_checks.loom). |
| Llama blocks | [attention/layer/output/FFN adapter](../programs/llama/BUILD.bazel), [provider closure](../kernels/llama/BUILD.bazel) | Maintained Llama ordering, native cache/packet bindings and numerical policies. The selected embedding tag admits Q4_K/Q6_K; the Q6_K last-row head binds tied or independent output from descriptor facts. Adjacent H128 RoPE uses the actual frequency tensor when present and unit divisors for ordinary RoPE otherwise. Layer format lookups preserve mixed Q4_K/Q6_K projections. No checkpoint-expanded sequence or separate branded numerical architecture. [Layer/model/preparation fixtures](../architectures/llama/BUILD.bazel), [configuration cases](../loomify/testing/configuration_test.cc). Availability and bounded execution are separate from independent full-model fidelity. |
| Qwen attention, GDN, FFN and output blocks | [program closure](../programs/qwen35/BUILD.bazel), [shared providers](../kernels/qwen/BUILD.bazel), [GDN kernels](../kernels/qwen/gdn/BUILD.bazel), [attention kernels](../kernels/qwen/attention/BUILD.bazel) | Qwen35 hybrid sequence and state semantics. Complete attention/GDN blocks retain internal FFNs. Recurrent/convolution state and updates are not generic KV attention. [Component fixtures](../kernels/qwen/testing), [model fixtures](../architectures/qwen35/testing/BUILD.bazel). |
| Gemma3 blocks | [Programs](../programs/gemma3/BUILD.bazel), [kernels](../kernels/gemma3/BUILD.bazel), [architecture](../architectures/gemma3/BUILD.bazel) | Gemma pre/post plain RMS, Q/K head RMS, full NeoX RoPE, local/global attention, GeGLU and post-norm residual sequence. Canonical all-T packing receives explicitly zeroed inactive RMS/GeGLU rows. [Poisoned padding checks](../kernels/gemma3/testing/padding_checks.loom) and direct whole-epoch window/state checks exercise this composition. |
| Qwen3 ordinary blocks | [Architecture](../architectures/qwen3/BUILD.bazel), [compact layer sequence](../programs/qwen3/layer.loom) | Pre-RMS, independent Q/K/V projections, weighted Q/K head RMS, full NeoX rotary, ungated causal GQA, attention output/residual, SwiGLU FFN and tied Q6_K head. Query width may differ from hidden width. Shared scalar and projection operations provide math; selected platform products provide WMMA schedules. The initial Q4_K_M fixture uses Q4_K/Q6_K matrix tags and H128. [Descriptor/configuration tests](../loomify/testing/qwen3_configuration_test.cc) check extracted facts/tensor bindings; full-model correctness is separate. |
| Qwen2 ordinary blocks | [Architecture](../architectures/qwen2/BUILD.bazel), [layer sequence](../programs/qwen2/layer.loom) | Qwen2.5 pre-RMS, Q/K/V projections plus F32 biases, full NeoX rotary with no Q/K head norm, ordinary causal GQA, output/residual and shared matrix SwiGLU FFN, tied Q6_K last-row head. Actual 1.5B Q4_K_M geometry E1536/Q1536/KV256/H128/F8960/28 layers. Per-layer V/down Q4_K/Q6_K tags come from GGUF; no unrolled checkpoint copy. | [Header/configuration/range tests](../loomify/testing/qwen2_configuration_test.cc), SOURCE/BC layer fixtures at mixed-format boundaries1->2 and4->5. Whole-model fidelity is separate. |
| Spark ordinary blocks | [Architecture](../architectures/spark2_5/BUILD.bazel), [compact layer](../programs/spark2_5/layer.loom), [mapper](../loomify/architectures/spark2_5.cc) | Pre-RMS, fused quantized QKV plus F32 per-head gate, raw H256 full/partial NeoX, causal local/global GQA, sigmoid head gating, output/residual, pre-RMS canonical mixed-format SwiGLU and final RMS/greedy head. No Q/K norm or postnorm. Derived dimensions/tags/ranges and layer routing drive composition; generated output facts select tied or independent weights and tag. Actual E2560/Q4096/KV1024/F10240/36-layer checkpoint is one witness. [Descriptor/configuration cases](../loomify/testing/spark2_5_configuration_test.cc) and [SOURCE/BC materialization](../architectures/spark2_5/testing/materialization_test.cc) cover variations, including untied tag12 output. |
| Shared ordinary Qwen decoder | [Attention composition](../programs/decoder/attention.loom), [fact helpers](../loomify/architectures/ordinary_decoder.cc) | Qwen2 and Qwen3 adapters share `programs/decoder/attention.loom` (`llm.decoder.matrix_attention`), preserving seven scratch allocations, per-layer Q4/Q6 projections, KV publication, causal attention and residual order. Each selected product supplies one typed `llm.decoder.prepare_qkv`: Qwen2 bias-before-RoPE plus active V bias; Qwen3 learned QK RMS before RoPE. `loomify/architectures/ordinary_decoder.cc` shares matrix bindings and numerical/cache/preparation facts while family admission and learned tensors stay local. Focused Qwen2/Qwen3 configuration tests cover fixed `qwen38` capacity ownership, all mixed-layer tags/encoded extents, parameter order/ranges, ten neutral preparation facts, and distinct hidden-width bounds. |
| Embedding/output wrappers | [Llama Q6 embedding](../kernels/llama/embedding/embedding.loom), [Q4 epoch adapter](../kernels/llama/embedding/q4k_epoch.loom), [Llama head](../programs/llama/output.loom), [Qwen embedding](../kernels/qwen/output/embedding.loom), [Qwen head](../programs/qwen35/output.loom) | Own row selection, parameter layout and scratch around shared numerical operations. Llama Q4/Q6 epoch adapters reuse the common config-free F32 epoch operation, which owns encoding selection, input_begin, count and compact active-row addressing. The model has one embedding invocation; adapters preserve consumed entrypoint signatures. No F16 rounding stage is introduced. Tied weights are parameter aliases, not a reason to duplicate a table. Check architecture/configuration admission before borrowing a wrapper. |
| Selected-ID publication | [Llama](../kernels/llama/publication/publication.loom), [Qwen](../kernels/qwen/output/publication.loom) | R epoch/private selected IDs; W public output. Matching direct epoch contract permits one logical result at physical slot0..15, no-output preserves output. Llama's source-EOS policy supplies three extracted stop IDs: existing header EOGs or plain ChatML endoftext/im_end/endofturn, with canonical im_end distinct from declared endofturn EOS. The same kernel maps all three without changing the private raw winner. [Gemma publication](../kernels/gemma3/publication.loom) and Qwen3 use shared [two-stop mapping](../kernels/publication/two_stops.loom): a pure `i32` token/two-stop-ID/terminal operation. It maps either supplied source stop to the canonical terminal, preserving private raw IDs and nonstops. GGUF EOS and vocabulary-derived policy stop identities remain distinct; no numeric token IDs are embedded in the common helper. [Forced publication checks](../kernels/gemma3/testing/publication_checks.loom) cover both stops, nonstops and no-output; [Llama relocated-ID checks](../kernels/llama/testing/token_policy_checks.loom) cover all three numerical/host stop routes. Not a general sampler or multi-output implementation. [Epoch checks](../architectures/llama/testing/epoch_checks.loom), head/output component fixtures. |
| Model / epoch / preparation / execution | [Llama](../architectures/llama/BUILD.bazel), [Qwen35](../architectures/qwen35/BUILD.bazel), [Gemma3](../architectures/gemma3/BUILD.bazel), [Qwen3](../architectures/qwen3/BUILD.bazel) | Each public epoch owns private selected-ID scratch, calls private model computation, then publishes. Model owns layers/head/state effects; preparation owns allocations/stages; Llama/Gemma/Qwen3/Qwen2 preparation adapters retain only root/diagnostic strings; [ordinary preparation](../serving/ordinary_prepare.loom) reads the ten generated `loomify.serving.*` geometry facts directly, and [ordinary host control](../serving/ordinary_control.loom) remains shared; Qwen35 retains its own host policy. Shared [snapshot count](../kernels/epoch/count.loom) owns the 12-byte model count snapshot. Reuse this split; family layouts remain explicit. [Host contract test](../serving/testing/epoch_control_test.cc). |

Ordinary preparation tests share the neutral
[VM/link/record harness](../serving/testing/ordinary_prepare_harness.h).
Independent geometry, roots, markers and capacity cases belong to each
architecture: [Llama](../architectures/llama/testing/prepare_test.cc),
[Gemma3](../architectures/gemma3/testing/prepare_test.cc),
[Qwen3](../architectures/qwen3/testing/prepare_test.cc) and
[Qwen2](../architectures/qwen2/testing/prepare_test.cc) and
[Spark](../architectures/spark2_5/testing/prepare_test.cc).
Each owns its strict `:prepare_test` target and direct descriptor-fixture
dependency; [Qwen35](../architectures/qwen35/testing/prepare_test.cc) keeps
its distinct hybrid-storage suite and the existing preparation VM support.


Ordinary preparation reads the generated `loomify.serving.*` facts:
`hidden_size`, `layer_count`, `kv_size`, `history_capacity`, `token_capacity`,
`cache_capacity`, `pool_capacity`, `cache_base`, `map_origin` and
`trailing_state_bytes`. Cache base and map origin are offsets; the other eight
are index values. Spark uses all attention layers and zero trailing state;
LFM3 appends convolution history after its attention KV planes. The common configuration writer
serializes already-admitted family geometry. The architecture adapter supplies
only `ordinary.epoch_root` and `ordinary.unsupported`; token policy
supplies the terminal marker. These feed the existing single-request pooled
allocation/stage records, not an arbitrary state layout. The concrete epoch and
publication commands remain architecture-owned.

## Text, tokenizer and serving policy

The selected GGUF template owns fresh prompt structure, literal text and defaults.
The shared compiler and analysis generate request adaptation, prompt rendering,
output/tool parsing, completion and continuation controls from that source. Native
token binding validates its consumed globals and analyzed terminal.
Numerical architecture does not select prompt formatting.

| Piece | Owner | Available behavior and boundary |
|---|---|---|
| Byte, string and JSON operations | [strings.loom](../text/common/strings.loom), [json.loom](../text/common/json.loom), [jinja.loom](../text/common/jinja.loom) | Shared byte comparison/assembly, selected trimming, JSON lookup/quoting and compiler value operations. Preserve whitespace according to the source template. JSON helpers do not promise Python-compatible floating-point spelling. |
| GGUF input compilation | [Compiler](../loomify/text/jinja/compiler.h), [package composition](../loomify/chat_policy.cc) | Compile actual selected source, including supported literal/structural changes. Unsupported language features fail with source diagnostics before publication. Reasoning metadata is qualified from the actual source, without fixture fallback. Generated text belongs to control, independently of weight preparation. |
| Generated request and input preparation | [Request emitter](../loomify/text/jinja/request_emitter.h), [environment](../text/generated_input/environment.loom), [request](../text/generated_input/request.loom), [preparation](../text/generated_input/prepare.loom) | Adapt admitted native requests from the analyzed source, preserve omitted options, and evaluate the whole prompt at chat begin. Fresh USER uses the same evaluator; continued USER uses the source-qualified continuation renderer or its precise unavailable diagnostic. Shared member/set/project operations remain neutral. |
| Generated response and host controls | [Format analysis](../loomify/text/jinja/format_analysis.h), [output emitter](../loomify/text/jinja/output_emitter.h), [output helpers](../text/output/BUILD.bazel), [host](../text/generated_input/host.loom) | Source-derived content/reasoning/tool parsing, startup state, completion replay and finite reasoning metadata. Plain, JSON-call and tagged-call formats reuse shared mechanics. Supported sources include all nine maintained templates; supported mutations compile through the same path. No handwritten per-template protocol products or fingerprint selection. |
| GGUF source selection and control preparation | [Selection](../loomify/chat_policy.h), [implementation](../loomify/chat_policy.cc) | Select owned scalar/named GGUF source with explicit duplicate/default diagnostics, then `PrepareGeneratedText` compiles and validates actual source-derived controls. Numerical architecture does not select text behavior. |
| GGUF token facts and native association | [Metadata](../../tools/model/text_metadata.h), [binding](../loomify/token_policy.cc) | Validate consumed role IDs, actual BOS/terminal pieces, atomic markers and the matching native tokenizer JSON. EOS and canonical terminal remain distinct; numerical publication is checked independently. Gemma rejects encoded IDs outside its source vocabulary. This is not full tokenizer reconstruction or universal ordinary-vocabulary parity. |
| Identity and greedy request policy | [Identity](../text/common/identity.loom), [greedy](../serving/greedy.loom), [request checks](../serving/request_check.loom) | Shared model-name data, no image support, and existing greedy option validation. Model heads use argmax; no new sampling/RNG is implied. |
| Native host contract and declared products | [Interface](../serving/control_interface.loom), [resource inventory](../serving/BUILD.bazel), [assembly tests](../serving/testing/assembly_contract_test.cc) | Existing public hooks and BUILD-owned products for linking. Component and public-package tests remain distinct; internal environment probes do not add serving exports. |

Original Jinja fixtures are independent test inputs, not production renderer assets.
The [public native checker driver](../text/testing/check_reference.py) checks actual
published controls in SOURCE and bytecode against independent prompt/full-token-ID
expectations. Direct compiler tests retain source mutations. Component tests additionally
probe the internal environment. Response/parser/completion, option discovery and
RAW/USER input tests retain their own authored assertions. Source compilation does
not expand native endpoint admission or claim exact float-tool serialization parity.

Actual native text-policy checks use the pinned runtime's public test executable
`//loom/src/loom/serve/testing:text_request_check`, driven by
[check_reference.py](../text/testing/check_reference.py). It owns the real native
modules and reports prompt/full tokens, streaming output, tool calls and
completion. Each invocation prepares one policy/request; it does not expose
RAW/USER or persistent checkpoint/session tests. Focused source preparation tests
and actual model lifecycle checks own those distinct boundaries. No private SDK
visibility exception or copied native module is part of this testing contract.
Actual-source XML and Hermes parser ledgers live in
[generated output tests](../loomify/testing/output_emitter_test.cc).
The [independent pinned reference runner](../text/testing/render_reference.py)
requires explicit local adapter, cases, immutable template/tokenizer and output
paths; original-GGUF template input uses the explicit `--source-override` flag.
The [generated component consumer](../text/testing/generated_adapter_consumer.BUILD.bazel)
uses public compiler/runtime modules for source preparation checks. Compiler,
input preparation and native serving acceptance are separate responsibilities;
passing an internal wrapper does not prove the runtime can bind the exported
hooks. Existing output and JSON checks retain their independent expectations.

Token-to-byte normalization/decoding belongs to the existing native tokenizer;
generated output parsers classify content/reasoning/tool streams. A new tokenizer
JSON may use an existing implementation yet fail consumed-role or terminal checks.
Test the parser and actual encoding/decoding before declaring an engine missing.
Declared EOS, template end-of-turn, PAD, BOS and publication mapping remain
distinct. Multiple-stop behavior is not established simply by adding marker data.

For text onboarding, select the actual GGUF source with `SelectChatTemplateSource`
and prepare its controls with `PrepareGeneratedText` and the matching native tokenizer.
Reuse shared analysis/emission for supported structures. Add a compiler feature for
an unsupported source construct rather than handwriting a prompt renderer or adding
a profile. Validate actual published controls against independent original Jinja and
full token IDs. Similar marker strings alone do not prove compatibility.

## Extraction, configuration and packaging

| Piece | Owner | Current coverage |
|---|---|---|
| Header/tensor catalog | [gguf_catalog.h](../../tools/model/gguf_catalog.h), [implementation](../../tools/model/gguf_catalog.cc) | F32/F16/Q8_0/Q5_0/Q4_K/Q5_K/Q6_K encoded size/range interpretation. [Tests](../../tools/model/testing/gguf_catalog_test.cc). |
| Logical model descriptor | [model_descriptor.h](../../tools/model/model_descriptor.h), [shared reader](../../tools/model/model_descriptor.cc), [family loaders](../../tools/model/architectures), [tensor-name helper](../../tools/model/tensor_names.h) | Llama, Qwen3, Qwen2, Qwen35, Gemma3, LFM3 and Spark role/shape/layer/state mapping; family loaders own metadata/layout differences, while loading and configuration share role-to-checkpoint-name spelling; logical admission is separate from numerical/provider support. Gemma effective local-window/global RoPE metadata is required and retained; Spark retains fused QKV and per-layer effective rotary widths, with zero inheriting the model width; other family restrictions remain explicit. [Tests](../../tools/model/testing/model_descriptor_test.cc). |
| Generated configuration/bindings | [configuration.h](../loomify/configuration.h), [common producer](../loomify/configuration_internal.h), [family mappers](../loomify/architectures) | Numeric facts, named tensor bindings and pure layer data lookups for supported families. The common producer retains constructed-descriptor name/shape/encoding/range checks. Checked canonical projection and ordinary embedding facts own format-specific bounds and exact bytes; transactional binding retains role/name/rank/extents/ranges and rejects overlap before publication. Ordinary Llama/Gemma3/Qwen3/Qwen2/LFM3/Spark preparation uses one shared typed fact writer; numerical configuration symbols remain family-owned. Token/prefill capacities remain<=512; logical context capacity is separate. Gemma tests use context1024 packages and context576 direct window/state witnesses; family restrictions remain explicit. [Tests](../loomify/testing/configuration_test.cc). |
| Architecture selection | [architectures.h](../loomify/architectures.h), [finite assembly entries](../architectures/assemblies.bzl), [resource BUILD inventory](../serving/BUILD.bazel) | Each entry selects explicit implementation/preparation/execution products, its data-only configuration mapper, fixed numerical publication-slot layout and existing context limit (Llama131072; others262144). Qwen3 and Qwen2 require the selected scaled-F16 projection product; Qwen35 requires H256 prefill. Geometry/encoding admission stays in the mapper and native providers. `architecture_selection` owns these policies; configuration/token-policy facades depend on it, while generic native linking depends only on `architecture_types`. |
| Source publication facts | [publication.h](../loomify/publication.h), [source profiles and slot emitter](../loomify/publication.cc) | Numerical architecture selects only fixed publication-slot layout/config prefix. Central known source profiles resolve declared EOS and unique requested pieces through one bounded GGUF vocabulary scan; separate layout emission preserves exact ordered config slots and commits owned facts transactionally. Llama3 and plain ChatML share the three-slot kernel but have distinct source profiles/canonical terminal indices; Qwen2/Qwen3 use text+turn slots, Gemma uses role-bound EOS+turn slots, Spark uses one declared-EOS slot without fabricated secondary stops, and Qwen35 is host-managed with no FILE/vocabulary requirement. Unspecified/invalid layouts reject. No recognized chat-template, external/native tokenizer or full vocabulary retention is required for numerical-only generation; package binding independently verifies every raw identity and canonical terminal. [Token-policy tests](../loomify/testing/token_policy_test.cc) cover layouts, source errors, duplicate vocabulary, canonical EOS distinction and output preservation; [public builder tests](../loomify/testing/public_builder_test.cc) retain sparse numerical-only and unsupported-template construction. |
| Builder / native link / materialization | [Public API](../loomify/gguf_model_builder.h), [linking](../loomify/linking.cc), [maintained passes](../loomify/materialize.loom) | Open -> Configure -> BuildModule/WriteModule or WritePackage. Configuration can be inspected before product loading; numerical-only generation is independent of chat/tokenizer admission. [Builder tests](../loomify/testing/public_builder_test.cc). |
| Package writer / converter | [package.cc](../loomify/package.cc), [CLI](../../tools/loomify-gguf/main.cc), [guide](gguf-export.md) | Five-file source/BC Serve package, external GGUF/tokenizer references, safe new-directory publication. Use the existing path for a new family. [Package tests](../loomify/testing/package_test.cc). |

A new family adds its authored logical reader and scoped tensor-name exceptions
in `tools/model/architectures/<family>.cc`, its data mapper in
`lib/loomify/architectures/<family>.cc`, its maintained Loom sequence and focused
cases, then one [finite assembly entry](../architectures/assemblies.bzl).
[Build-owned generation](../../build_tools/model_assembly.bzl) emits separate
logical/native declarations and selectors, resource product triples and eligible
ordinary prepare/control/epoch forwarding shells. Logical extraction retains no
native-serving dependency; generated shells do not generate operation order.
The ordinary six-family ABI preserves Llama's existing
`llama32_publish_epoch_id`; Qwen35 remains an explicit custom assembly.
Existing operation-owned source closures and actual product outputs remain
explicit. Extend publication/text association only for actual semantic changes;
there is no runtime registration or provider discovery. [SOURCE/BC test helpers](../../build_tools/loom_testing.bzl)
retain provider/check boundaries and independent cases, while projection fixture
expansions live only in build outputs.

Model-level package generation exists for the supported Llama, Qwen35, Gemma3,
LFM3, Spark and Qwen3 profiles. The [export guide](gguf-export.md) describes the
shared export flow; component and model proof entrypoints are linked above.
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

### Numerical coverage extensions

The Llama composition keeps hidden, query and KV widths independent. Its
[mapper](../loomify/architectures/llama.cc) binds E-to-Q projections and Q-to-E
output projections; the attention program allocates and addresses query rows
using Q. MiniCPM5's ordinary Llama GGUF is a witness at E1536/Q2048/KV256,
F4608 and vocabulary130560, without brand-specific scaling. The
[model fixtures](../architectures/llama/testing/configured_model_test.cc) and
[geometry fixtures](../kernels/llama/testing/BUILD.bazel) cover these dimensions,
compatible FFN tails and high-ID gather/greedy selection.
For ordinary Llama GGUFs declaring `</s>` as their single EOS, numerical
publication resolves that source identity and aliases the existing three stop
slots to it. No token ID is fixed, and this does not admit a chat template or
tokenizer policy. [Publication and builder tests](../loomify/testing/BUILD.bazel)
cover relocated IDs, invalid source metadata and numerical-only configuration.

Qwen35 retains its H256/GDN sequence while admitting hidden widths in
multiples256, subject to each projection's bounds. Its E2560/F9216/32-layer
[model witness](../architectures/qwen35/testing/model_test.cc) checks mixed
V/down tags and compact state ordinals; the
[block emitter](../architectures/qwen35/testing/emit_block_validation.cc) exposes
isolated GDN and attention roots with their actual per-layer weights.
GDN first-advance reads the caller-prepared clean anchor and writes the selected
destination; subsequent advances read that destination. Position0 alone does
not clear an aliased anchor. Reset tests keep the destination dirty while
providing a separate clean anchor and require fresh-equivalent logical state.

The [three-tap convolution](../kernels/convolution/shortconv3.loom) curates the
existing LFM2 pipeline's B/C/X compute and history commit. It reads B/C/X,
weights and selected history, writes the output, then updates only two history
rows at the supplied byte origin. Fresh position0 ignores old history; inactive
or invalid packets never commit history. [SOURCE/BC fixtures](../kernels/convolution/testing/BUILD.bazel)
exercise split chunks, reset, malformed packets and guards. New
[H64 row wrappers](../kernels/decoder/rows.loom) reuse head RMS, NeoX rotary,
F16 native64-page publication and plain causal attention. Their
[fixtures](../kernels/decoder/testing/BUILD.bazel) use Q32/KV8, boundary positions,
page63/64 and invalid-packet no-mutation checks.

The [LFM2 descriptor](../../tools/model/architectures/lfm2.cc),
[mapper](../loomify/architectures/lfm2.cc) and
[model](../architectures/lfm2/model.loom) assemble these pieces with ordinary
embedding, canonical projection/FFN and a tied Q6 head. The final stored gamma
comes from `token_embd_norm.weight`, after all layers. Attention and convolution
use separate compact state ordinals. A family packet guard rejects invalid whole
chunks before either history or KV writes. [Configuration tests](../loomify/testing/lfm2_configuration_test.cc)
cover actual role/shape facts, and the [layer runner](../architectures/lfm2/testing/run_layer_validation.py)
uses reflected parameter bindings and runtime-generated sparse weights for
independent mixer/FFN/state and gamma-sensitive head checks. LFM2 and LFM2.5
share a numerical recipe; their actual GGUF weights remain independent inputs.

These are numerical construction and staged execution interfaces. They do not
enable new chat-template policies or establish independent whole-model fidelity.

### Spark configuration and bounded execution

The [Spark reader](../../tools/model/architectures/spark2_5.cc) and
[mapper](../loomify/architectures/spark2_5.cc) derive dimensions, fused QKV,
per-role tags/ranges, local/global routing and effective rotary geometry from
GGUF. The pure typed layer lookup returns window, rotary width and F32 log-step
for each layer, allowing independent supported effective profiles in constructed
descriptors; GGUF extraction still follows its maintained metadata schema. Admission intersects consumed operations: hidden1024..32768 multiple256,
H256, layers1..64, divisible Q/KV head counts1..128, FFN256..32768 multiple256,
vocabulary256..262144 and each actual projection's format-specific input bound.
In particular a Q6_K head/QKV/down operand imposes its own input<=17408;
checkpoint equality is not an admission rule. Constructed descriptor tests cover
shape/tag/routing variation and transactional failures; they do not prove every
admitted shape on hardware.

The training context1048576 remains a source fact. Generated context is bounded
separately at262144 and prefill capacity1..512. Actual generated SOURCE/bytecode
Serve checks use context1024/prefill128 and cover prompt, dependent decode,
chunked prefill, reset/replay, a second turn, thinking and clean shutdown. Local
window512 restricts reads over full F16 paged KV planes for all36 layers; it does
not introduce ring eviction. Larger execution contexts, universal admitted
geometry and independent full-model numerical fidelity are not established by
these checks. Final source review and scoped validation remain separate from
the availability of these components.

### LFM3 reusable pieces and composition

The LFM3 reader and [data-only mapper](../loomify/architectures/lfm3.cc)
derive layer kinds, compact attention/convolution ordinals, actual tensor tags
and ranges, rotary/window facts and history extent. The [model](../architectures/lfm3/model.loom)
and [layer](../programs/lfm3/layer.loom) retain the maintained operation order.
The initial mapper admits hidden widths1024..8192 in multiples1024, H128,
FFN widths256..32768 in multiples256, seven-tap short convolution, and rows1..512;
individual operations retain their own format/geometry contracts, including Q6_K down-input<=17408.
It uses canonical quantized projection separately for mixed gate/up/down tags,
without requiring another matrix or fused mixed-format provider.
See [descriptor/configuration tests](../loomify/testing/lfm3_configuration_test.cc).

| Reusable piece | Owner and contract | Proof entrypoint |
|---|---|---|
| Q5_K row and ordinary embedding | [Q5 row](../kernels/embedding/q5k_rows.loom) shares the [Q5 codec](../kernels/projection/q5k_codec.loom) with projection. [Ordinary gather](../kernels/embedding/ordinary_f32.loom) selects actual tags12/13/14 and owns dispatch/footprint checks. It reads active IDs/table, writes F32 rows and zeroes inactive or invalid-position rows. Generic epoch gather retains inactive-preservation semantics. | [SOURCE/BC gather checks](../kernels/embedding/testing/q5k_checks.loom), direct row/epoch/ordinary packet cases. |
| Canonical row projection and F32 gate projection | [Rows](../programs/projection/rows.loom) reuse pack/project with actual tags and own packed scratch; callers define inactive inputs. [F32 kernel](../kernels/projection/f32_rows.loom) reads active rows and row-major weights, writes disjoint outputs with inactive zeros; input/output widths1..32768 and rows1..512. | [F32 checks](../programs/projection/testing/f32_rows_checks.loom) include2048→16 and inactive/input guards. |
| GDN F32 or prepacked gate routing | The shared [row projection operation](../programs/projection/rows.loom) selects the existing F32 kernel from actual tag0 and original normalized F32 activations; other supported tags retain the existing packed operation. [GDN assembly](../programs/qwen35/gdn.loom) selects alpha and beta independently, without an architecture/model flag. F32 weights are row-major W[N,K], exact4*K*Nbytes, T<=C<=512 and K,N1..32768; no repacking or new dot arithmetic. | [Configuration/lookup tests](../loomify/testing/configuration_test.cc) cover0/0,0/13,13/0 and malformed F32 extents. [Model tests](../architectures/qwen35/testing/model_test.cc) inspect actual kernel inputs and recurrence outputs after SOURCE/BC materialization; external real-header tests cover both Ornith9B variants. This numerical capability does not imply those models' exact chat templates or complete Serve packages are admitted. |
| Explicit-encoding last-active greedy head | [Shared head](../programs/output/last_greedy.loom) takes encoding and packing policy. It owns normalized, packed, logit and reduction scratch derived from hidden/vocabulary dimensions; callers supply residual/gamma/weight/output and their authored norm hook. Llama/Qwen2/Qwen3 retain policy0/tag14; LFM3/Spark select canonical policy1 and actual output tag. Gemma uses the same sequence through its stored-gamma norm hook. Numerical argmax remains separate from source stop publication. | [Q5 head checks](../programs/output/testing/q5_head_checks.loom), existing head/output checks. |
| Seven-tap short convolution | [Operation](../kernels/convolution/shortconv.loom) reads B/C/X projections and channel-major F32 weights, writes output and updates only six selected history rows. Fresh absolute position0 never reads old history; append carries samples across chunks. It is not GDN convolution or SiLU preparation. Width256..32768 multiple256, rows1..512. | [Arithmetic/history/guard checks](../kernels/convolution/testing/checks.loom) and [programs](../kernels/convolution/testing/programs.loom). |
| Factor-aware H128 Q/K preparation and gated window attention | [Row wrappers](../kernels/attention/factored_rows.loom) reuse head RMS, NeoX rotary and causal row/window attention. Global frequency divisors are a bound F32 tensor; local rows use factor1 and their own rotary base. R query/norms/factors/gate/map/cache; W rotated rows/attention and selected KV positions through publication. | [Focused SOURCE/BC checks](../kernels/attention/testing/lfm3/checks.loom). |
| One/two-stop epoch publication | [Shared templates](../kernels/publication/epoch_two_stops.loom) use one packet-guard/slot-write body with selected one-stop or two-stop mapping. Qwen2/Qwen3/LFM3 retain two stops; Spark supplies a single source EOS mapped to the packet terminal, preserving nonstops and private winners. | Existing publication/epoch fixtures and generated model package tests. |

[Ordinary preparation](../serving/ordinary_prepare.loom) now takes ten generated
facts: the existing nine plus `trailing_state_bytes`, zero for previous ordinary
models. LFM3 passes compact attention count for KV planes and appends derived
history bytes. Total model layers remain a separate numerical fact. Existing
generated packages disable session/prefix caches and checkpoint pins; no claim is
made that appended private history supports nonzero checkpoint restoration.
Fresh/reset and short-chunk commits must fully define logical history.
[Preparation tests](../architectures/lfm3/testing/prepare_test.cc) reuse the
existing cold-record harness and preserve zero-extra-state records.

The [Compact ChatML source](../text/compact_chatml/templates/f45005c5.jinja)
drives generated compact reasoning, XML tools, request adaptation and continuation,
independently of architecture. Input traversal, source BOS insertion and individual
tool-result framing come from the compiled GGUF template.
[Revision cases](../text/compact_chatml/testing/render_cases.json)
and the public native checker compare original-template bytes and
full token IDs. Rich dynamic tool extensions outside the admitted request domain
are rejected rather than silently ignored. Original GGUF facts supply IDs and
source literals; no numerical family branch selects formatting.

The Gemma components, source-driven configuration, package construction and native
text/host contracts have concrete tests. Direct whole-epoch tests compare active
outputs and full state across SOURCE/BC, offsets, dependent decode and reset;
excluded local-prefix invariance and included local/global-prefix sensitivity
exercise window ownership. Their synthetic history is not an independent
full-model accuracy oracle. Standalone numerical checks retain independent
expected values and fixed budgets. Live model semantic qualification remains a
separate gate from linking, bitwise ownership and prompt plausibility.

### Jinja input compiler proof

The standalone [chat-template compiler](../loomify/text/jinja/compiler.h) parses
supported source into a transient AST and emits ordinary Loom control flow.
Its generated `jinja.evaluate(environment)` accepts an explicit JSON object;
missing names remain distinct from null. [Shared value helpers](../text/common/jinja.loom)
operate on strings, JSON values and sequences rather than interpreting templates.
The compiler has no template-revision lookup or model-family dependency.
`ReadTemplateStringDefault` reuses the same parser for a bounded, unshadowed
external option's consistent literal default/d declaration. It preserves rendering
semantics and reports unsupported or ambiguous facts without inventing a value.
Effort-aware package composition can use this source fact for native discovery;
it does not inject missing prompt options or interpret a template at runtime.
Supported operations include checked signed integer addition/remainder, dynamic
item lookup, namespace fields carried through branches and loops, defaults,
Unicode string operations, ordered key/items iteration and reverse slices.
Items views are supported as immediate loop iterables, rather than general Python
view values. Owning list concatenation, list conversion and join support text
accumulation without mutating inputs. Scoped generation blocks produce text,
preserving namespace effects while keeping ordinary assignments local.
Statically named nonrecursive macros become Loom functions with explicit lexical
captures and namespace field results. Namespace identities are resolved by the
compiler; they are not runtime JSON objects. Container equality, unsafe mixed
integer/float comparisons and compound mapping keys have explicit capability
boundaries. Tuple data cannot cross macro or namespace-storage boundaries or
participate in addition; supported constant tuple arguments remain available.
Exact floating-point JSON spelling remains separate from the
lexical-preserving serializer.

The [native proof consumer](../loomify/text/jinja/testing/consumer.BUILD.bazel)
executes generated SOURCE and bytecode against the independent
[Jinja reference](../loomify/text/jinja/testing/reference.py) and
[cases](../loomify/text/jinja/testing/cases.json). Complete-template coverage includes
Llama, plain ChatML, Qwen2.5, Qwen3, Qwen3.5, Gemma, Spark and LFM3, together with focused
[arithmetic/item](../loomify/text/jinja/testing/c1_cases.json) and
[namespace/string](../loomify/text/jinja/testing/c2_cases.json) witnesses, plus
[macro/traversal](../loomify/text/jinja/testing/c3_cases.json) and
[full-template mutation](../loomify/text/jinja/testing/c3_profile_mutations.json)
cases. Deferred float comparisons preserve numeric value, signed zero and exact
surrounding bytes; the independent reference also checks native token IDs against
HF tokenization of the actual output. Original Jinja bytes and IDs remain distinct.
Compiler tests include branches beyond existing Serve admission. A finite supplied
date keeps evaluation independent of the system clock.
Focused [list/generation cases](../loomify/text/jinja/testing/c4_cases.json) cover
ownership, macro-local state and generation scope. LFM3's richer tool inputs are
compiler-language witnesses; they do not enable richer serving request behavior.

The compiler entry point is independent of serving request admission. Package
integration preserves selected output/token policies and RAW/USER convenience
framing; supported compiler expressions do not expand tool or media serving admission.

The shared [request adaptation](../text/generated_input/request.loom),
[environment projection](../text/generated_input/environment.loom) and
[host hooks](../text/generated_input/host.loom) compose with emitted controls.
Fresh USER invokes the same evaluator; continued USER consumes source-qualified
construction. [Component tests](../text/testing/generated_adapter_test.cc)
exercise source preparation, while the public native checker and model lifecycle
checks exercise serving and continuation with the pinned callback contract.
The package builder compiles the actual selected GGUF source through this one path
for Llama headers, plain ChatML, Qwen2.5 Hermes, Qwen3 Hermes with thinking, both
Qwen3.5 revisions, Gemma turns, LFM compact ChatML and Spark GLM. The old protocol
implementations, fingerprint table and adapter merge are removed. Generated text
does not enter numerical weight preparation. Export targets the existing pinned
runtime; it does not require new callbacks or a companion HRX change. Callback
representability is narrower than compiler-language support: unavailable native
context produces a source diagnostic rather than an inferred model profile.
Completion retains the executed history prefix and serializes the new turn;
the next full Jinja prompt and existing native prefix checks decide reuse.
Actual installed-wheel and live Serve qualification remain release gates.

[Template analysis](../loomify/template_analysis.h) constructs owned selectors,
source-derived reasoning metadata and the real continued-USER product. Templates
without an external effort option need no native discovery execution. Effort
metadata uses the existing ordered input construction only until its first active
source-valid context, then checks the finite source domain; an invalid first active
context fails rather than searching for a replacement. USER unavailability remains
a precise optional diagnostic and does not disable structured chat.
The [test-only qualifier](../loomify/testing/template_qualification.h) retains
independent full-assistant recipes, projection/selector counterexamples, startup
and tool-capture checks, and route/shape coverage. Ordinary export does not invoke
that suite or require synthetic coverage. Context construction, normalization and
pattern lowering are shared implementations rather than copied test algorithms.

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
