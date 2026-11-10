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

## Shared numerical pieces

Use the owning operation/interface and its declared dependencies; do not import
an entire architecture merely to reach a helper. Some shared helpers retain
historical family names or paths. Their actual signature and body determine
reuse, not the name. `R` means read, `W` write and `RW` read/write; the effects
below summarize roles and do not grant unspecified aliasing or access regions.

| Piece / entrypoint | Source and build owner | Current contract and operand effects | Existing proof entrypoint |
|---|---|---|---|
| Quantized projection: `llm.projection.pack_input`, `llm.projection.project_packed` | [Composition/interface](../programs/projection/projection.loom), [BUILD](../programs/projection/BUILD.bazel) | Q4_K, Q5_0, Q5_K, Q6_K, Q8_0 selected by actual weight tag. R input/weight/count; W packed activations/output. Canonical and compatible policies have different active-row/packing semantics. See format bounds below. | [Native materialization](../programs/projection/testing/materialization_test.cc), [SOURCE/BC operation checks](../programs/projection/testing/operations_checks.loom), [BUILD](../programs/projection/testing/BUILD.bazel). |
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
| Llama blocks | [attention/layer/output/FFN adapter](../programs/llama/BUILD.bazel), [provider closure](../kernels/llama/BUILD.bazel) | Maintained Llama ordering, native cache/packet bindings and numerical policies. Reuse shared math underneath; do not treat the whole block as a generic decoder. [Layer/model/preparation fixtures](../architectures/llama/BUILD.bazel). |
| Qwen attention, GDN, FFN and output blocks | [program closure](../programs/qwen35/BUILD.bazel), [shared providers](../kernels/qwen/BUILD.bazel), [GDN kernels](../kernels/qwen/gdn/BUILD.bazel), [attention kernels](../kernels/qwen/attention/BUILD.bazel) | Qwen35 hybrid sequence and state semantics. Complete attention/GDN blocks retain internal FFNs. Recurrent/convolution state and updates are not generic KV attention. [Component fixtures](../kernels/qwen/testing), [model fixtures](../architectures/qwen35/testing/BUILD.bazel). |
| Gemma3 blocks | [Programs](../programs/gemma3/BUILD.bazel), [kernels](../kernels/gemma3/BUILD.bazel), [architecture](../architectures/gemma3/BUILD.bazel) | Gemma pre/post plain RMS, Q/K head RMS, full NeoX RoPE, local/global attention, GeGLU and post-norm residual sequence. Canonical all-T packing receives explicitly zeroed inactive RMS/GeGLU rows. [Poisoned padding checks](../kernels/gemma3/testing/padding_checks.loom) and direct whole-epoch window/state checks exercise this composition. |
| Embedding/output wrappers | [Llama embedding](../kernels/llama/embedding/embedding.loom), [Llama head](../programs/llama/output.loom), [Qwen embedding](../kernels/qwen/output/embedding.loom), [Qwen head](../programs/qwen35/output.loom) | Own row selection, parameter layout and scratch around shared numerical operations. Tied weights are parameter aliases, not a reason to duplicate a table. Check architecture/configuration admission before borrowing a wrapper. |
| Selected-ID publication | [Llama](../kernels/llama/publication/publication.loom), [Qwen](../kernels/qwen/output/publication.loom) | R epoch/private selected IDs; W public output. Matching direct epoch contract permits one logical result at physical slot0..15, no-output preserves output. Llama retains its EOG mapping; [Gemma publication](../kernels/gemma3/publication.loom) maps source EOS/EOT to the canonical terminal, preserving private raw IDs and nonstops. [Forced publication checks](../kernels/gemma3/testing/publication_checks.loom) cover both stops, nonstops and no-output. Not a general sampler or multi-output implementation. [Epoch checks](../architectures/llama/testing/epoch_checks.loom), head/output component fixtures. |
| Model / epoch / preparation / execution | [Llama](../architectures/llama/BUILD.bazel), [Qwen35](../architectures/qwen35/BUILD.bazel), [Gemma3](../architectures/gemma3/BUILD.bazel) | Each public epoch owns private selected-ID scratch, calls private model computation, then publishes. Model owns layers/head/state effects; preparation owns allocations/stages; Llama/Gemma thin control adapters share [ordinary host control](../serving/ordinary_control.loom); Qwen retains its own host policy. Shared [snapshot count](../kernels/epoch/count.loom) owns the 12-byte model count snapshot. Reuse this split; family layouts remain explicit. [Host contract test](../serving/testing/epoch_control_test.cc). |

## Text, tokenizer and serving policy

Select text behavior independently of numerical architecture. Similar marker
spellings are not proof of equivalent role, whitespace, tool or thinking rules.

| Piece | Owner | Available behavior and boundary |
|---|---|---|
| Shared strings / JSON / identity | [Common text sources](../text/common/BUILD.bazel) | Existing pure text helpers. Reuse concrete helpers rather than copying an entire protocol. Do not move model-specific prose into common code. |
| Llama header protocol | [Implementation](../text/protocols/llama_headers/input.loom), [BUILD](../text/protocols/llama_headers/BUILD.bazel), [reference fixture](../text/llama/templates/5816fce1.jinja) | Known header/date framing, bounded plain-text policy and plain output/continuation hooks. No general Gemma/ChatML formatter. Some output helpers trim content; verify semantics before extracting them. |
| ChatML thinking protocol | [Common sources and two revisions](../text/protocols/chatml_thinking/BUILD.bazel) | Shared admission/input/output/tool mechanics with inline-history versus explicit-reasoning variants. [Renderer tests](../text/qwen35/testing/render_test.cc), [stream parser](../text/qwen35/testing/output_test.cc), [tool parser](../text/qwen35/testing/tool_output_test.cc), [independent references](../text/qwen35/testing/render_reference.py). |
| Gemma turns protocol | [Implementation](../text/protocols/gemma_turns/BUILD.bazel), [template](../text/gemma/templates/7de1c58e.jinja) | BOS/turn framing, plain UTF8 output preserving visible whitespace, canonical future-history replay with reuse disabled. No tools, reasoning or media requests. [Independent HF reference driver](../text/gemma/testing/render_reference.py) and native SDK SOURCE/BC tests cover 24 positive/8 template-error cases, domain rejection and output/replay. |
| Known-template recognition / data capture | [chat_policy.h](../loomify/chat_policy.h), [implementation](../loomify/chat_policy.cc) | Four maintained semantic skeletons; selected GGUF bytes supply supported literal/default slots. Explicit named-template selection. Unknown structural variants reject. [Selection/mutation tests](../loomify/testing/chat_policy_test.cc). Arbitrary Jinja compilation is not available. |
| GGUF text/token facts | [text_metadata.h](../../tools/model/text_metadata.h), [implementation](../../tools/model/text_metadata.cc) | Independent template issues, roles/IDs, insertion flags and selective vocabulary/merge spans; no full tokenizer reconstruction. [Tests](../../tools/model/testing/text_metadata_test.cc). |
| Native tokenizer association / consumed markers | [token_policy.h](../loomify/token_policy.h), [implementation](../loomify/token_policy.cc) | Uses the existing native tokenizer JSON parser. Explicit selected-text requirements admit the supported Llama/Qwen35/Gemma tokenizer combinations and consumed special identities; marker agreement is not full ordinary-vocabulary/tokenization parity. Renderer-owned BOS insertion and optional numerical publication facts are checked independently. Declared EOS and canonical terminal remain separate; Gemma source vocabulary bounds reject encoded added-media IDs before numerical execution. Native one-shot/incremental decode fixtures exercise byte fallback and special-token flags. [Tests](../loomify/testing/token_policy_test.cc), [real-asset tests](../loomify/testing/token_policy_real_assets_test.cc). |
| Greedy request options | [greedy.loom](../serving/greedy.loom), [request_check.loom](../serving/request_check.loom) | Shared current request/default validation. Model heads perform argmax; this does not implement temperature/top-k/top-p sampling or RNG. |
| Host interface and resource products | [control_interface.loom](../serving/control_interface.loom), [product inventory](../serving/BUILD.bazel) | Existing hook contract and declared products for native linking; source presence alone does not register a new architecture/text provider. [Assembly tests](../serving/testing/assembly_contract_test.cc). |

Token-to-byte normalization/decoding belongs to the existing native tokenizer;
protocol output parsers classify content/reasoning/tool streams. A new tokenizer
JSON may use existing implementation yet fail our current narrow policy gate.
Test the parser and actual encoding/decoding before declaring an engine missing.
Declared EOS, template end-of-turn, PAD, BOS and publication mapping remain
distinct. Multiple-stop behavior is not established simply by adding marker data.

## Extraction, configuration and packaging

| Piece | Owner | Current coverage |
|---|---|---|
| Header/tensor catalog | [gguf_catalog.h](../../tools/model/gguf_catalog.h), [implementation](../../tools/model/gguf_catalog.cc) | F32/F16/Q8_0/Q5_0/Q4_K/Q5_K/Q6_K encoded size/range interpretation. [Tests](../../tools/model/testing/gguf_catalog_test.cc). |
| Logical model descriptor | [model_descriptor.h](../../tools/model/model_descriptor.h), [implementation](../../tools/model/model_descriptor.cc) | Llama, Qwen35 and Gemma3 role/shape/layer/state mapping; logical admission is separate from numerical/provider support. Gemma effective local-window/global RoPE metadata is required and retained; other family restrictions remain explicit. [Tests](../../tools/model/testing/model_descriptor_test.cc). |
| Generated configuration/bindings | [configuration.h](../loomify/configuration.h), [common producer](../loomify/configuration_internal.h), [family mappers](../loomify/architectures) | Numeric facts, named tensor bindings and pure layer data lookups for supported families. Token/prefill capacities remain<=512; logical context capacity is separate. Gemma tests use context1024 packages and context576 direct window/state witnesses; family restrictions remain explicit. [Tests](../loomify/testing/configuration_test.cc). |
| Architecture product selection | [architectures.cc](../loomify/architectures.cc), [resource BUILD inventory](../serving/BUILD.bazel) | Finite Llama/Qwen35/Gemma3 implementation/preparation/execution products. Extending this table does not itself implement a model. |
| Builder / native link / materialization | [Public API](../loomify/gguf_model_builder.h), [linking](../loomify/linking.cc), [maintained passes](../loomify/materialize.loom) | Open -> Configure -> BuildModule/WriteModule or WritePackage. Configuration can be inspected before product loading; numerical-only generation is independent of chat/tokenizer admission. [Builder tests](../loomify/testing/public_builder_test.cc). |
| Package writer / converter | [package.cc](../loomify/package.cc), [CLI](../../tools/loomify-gguf/main.cc), [guide](gguf-export.md) | Five-file source/BC Serve package, external GGUF/tokenizer references, safe new-directory publication. Use the existing path for a new family. [Package tests](../loomify/testing/package_test.cc). |

Model-level package generation and bounded prompt/decode behavior exist for the
supported Llama3.2 3B and Qwen35 0.8B/27B profiles described in the export guide.
Independent full-model fidelity remains a separate qualification; successful
linking or a plausible response must not upgrade a component's numerical claim.

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
