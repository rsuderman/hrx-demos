# GGUF model export

`loomify-gguf` constructs supported [model architectures](coverage.md) by deriving data-only configuration and parameter bindings from the
GGUF descriptor, selecting a config-free numerical provider closure, and linking
maintained Loom programs. It writes the model document and source providers
consumed by existing Loom Serve. GGUF weights and matching tokenizer JSON remain
external. C++ handles admission, linking and packaging; the numerical sequence
is authored in Loom.

Build the converter with the repository's Bazel version, then export to a new
directory:

```sh
bazel build //tools/loomify-gguf:loomify-gguf
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --gguf=/path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --tokenizer=/path/to/tokenizer.json --output-dir=/path/to/exported-llama \
  --context-capacity=512 --prefill-capacity=128 --format=bc
loom-serve --device=amdgpu --pool_backing=fixed --model=/path/to/exported-llama/model.json \
  --listen=127.0.0.1:18081
```

`--format=loom` writes textual numerical source instead of bytecode. Both formats
serialize the same configuration-linked module. The package contains `model.json`, `model.loombc` (or
`model.loom`), `prepare.loom`, `control.loom` and `weights.loom`. Package-owned
references are relative, so this directory can move; external asset references
are absolute and still need their original files. The GGUF reference retains the
supplied path's format suffix, including a `.gguf` alias to an extensionless blob;
publication checks that alias still resolves to the source opened for inspection.
The converter never copies checkpoint bytes or converts the tokenizer. It refuses an existing output
directory and publishes `model.json` only after provider serialization succeeds.

Text controls target the HRX revision pinned in `MODULE.bazel`. The original
Jinja renders each complete request. Output scanning uses the source-selected
state; tool parsing and formatting use the existing native callback inputs.
Compiler support alone does not imply serving support: export reports a source
diagnostic when a callback would require request or call context the runtime
does not supply. Equivalent output routes share a parser; model names and
template fingerprints do not select behavior.

`complete_text` retains the rendered history prefix and serializes the completed
assistant turn from source-derived transformations and actual formatted calls.
It does not retroactively rerender previous messages. The next request still
renders its complete history through Jinja. Existing native prefix checks decide
reuse: an ordinary mismatch prepares fresh input; an explicitly required
checkpoint mismatch returns the runtime's existing incompatibility error.
Generated controls qualify their bounded plain-output reuse candidate separately
from the existing authored wheel catalog's completion policy.

Serve packages require `context_capacity` to be a multiple of64 and use
`--pool_backing=fixed` with the current Loom Serve pin. Package export rejects
an unaligned context before writing the destination; it never rounds the logical
context silently. The model document omits `pool_capacity`: fixed backing derives
that physical capacity from the context. `rows=1` and `max_sessions=1` retain
the fixed one-resident-request ceiling; resident growth is unsupported.
Numerical-only module generation and
configuration inspection continue to support arbitrary admitted context sizes.

For a local export/Serve smoke check with already-built tools:

```sh
python3 build_tools/loomify-smoke.py \
  --converter bazel-bin/tools/loomify-gguf/loomify-gguf \
  --serve /path/to/venv/bin/loom-serve \
  --gguf /path/to/model.gguf --tokenizer /path/to/tokenizer.json
```

Alternatively, replace the two asset arguments with `--recipe path/to/recipe.json`
to use its pinned files from the configured Hugging Face cache. Missing files
fail without downloading; explicit asset arguments override recipe lookup.
The script checks both formats, valid generation/accounting, matching greedy
responses and shutdown, without a fixed expected answer or token count. It reports
whether dependent decode was exercised. `--prompt`, `--max-tokens` and `--timeout`
are configurable. Temporary packages/state are deleted; use `--report result.json`
to keep the JSON summary. `--platform` selects the converter's generation recipe
independently of the server's runtime `--device`; both requested values are
recorded in the summary. It uses the caller's existing runtime/library setup.
See [GPU implementation contracts](hardware-support.md) for device bring-up.

The bounded Llama serving profile is ordinary greedy text with one active request,
no draft/checkpoint/session/prefix cache, maximum prefill chunk T in1..512 and
context C between T and131072. It uses Serve's existing packed-epoch path with
one span. Physical KV pool capacity rounds C up to64 tokens. The native prepare
program checks that requested capacities match the generated program. The text
control policy retains the existing Llama chat framing and end-of-generation
mapping. Isolated Serve step execution is explicitly rejected. Its sole numerical
request root is `generated.llama_epoch`, with the same seven-buffer request role
as `generated.qwen_epoch`. Both retain reserved native stage slots0/1 with T/1
tags, followed by shape stages at2+shape. Llama stages register its family epoch;
the bounded experimental Qwen one-token route is described below. Existing cold
allocation, view/origin and cache geometry are retained.

Each `lib/architectures/<family>/epoch.loom` owns the public numerical epoch root,
a16-byte private selected-ID buffer, its private model invocation and final
publication to the request output. Each `model.loom` owns the private embedding,
repeated-layer and head sequence, writing only private selected IDs. The Qwen
head retains all four private projection/argmax rows; only result0 is published.
Llama and Gemma share host encode/issue/progress hooks in
`lib/serving/ordinary_control.loom`; their `control_core.loom` files supply the
family terminal policy. Qwen keeps its own host control.
`lib/programs/llama` contains its complete layer, attention,
FFN adapter and normalized vocabulary-selection programs. `lib/programs/ffn`
contains the shared FFN sequence; each profile selects its numerical packing
and normalization providers. Generated configuration supplies tensor bindings,
layer extents/quantization and capacities, without generating command order.
The epoch passes the existing ordinary packet directly to its model, without a
legacy control/history bridge, input token copy or private progress allocation.
One-token continuation uses the same root. For active input interval[B,B+N),
Llama writes compact activations0..N-1 and uses the descriptor's absolute position;
Qwen retains its packed activation placement. Llama snapshots descriptor.length
once in model-owned count scratch for all layers/FFNs. Public output_count0
preserves all16 output words; output_count1 writes only descriptor0's physical
slot0..15, independently of selected_rows[0]'s logical activation-row selection.
Canonical host encoding remains one resident row/span with compact output slot0;
unsupported batching, media, proposal and verification modes are rejected. The model build
returns its owned numerical module and configuration source. Maintained preparation
reads the same configuration to construct native cold allocation, row-view, origin
and KV geometry records. The exporter links that preparation module and writes its
native text serialization; no C++ geometry object or template rendering is involved. Named immutable parameters
preserve canonical GGUF keys; compiler-reflected packed root offsets are distinct
from file offsets.

The same converter can write a numerical module without a tokenizer, or inspect
configuration without loading maintained Loom products or creating a compiler context:

```sh
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --gguf=/path/to/model.gguf --output=/path/to/model.loombc
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --gguf=/path/to/model.gguf --configuration-output=/path/to/new-inspection
```

Inspection writes `numeric.loom`: the exact numerical configuration, including
publication token IDs and pure per-layer lookup functions. Adding `--tokenizer`
prepares package data and writes supplemental `serve-data.loom`, which contains no
second numerical configuration. Control joins the full generated input data/code;
numerical preparation uses only the numerical configuration and bound terminal-token
facts. Inspection is
output-only, and does not certify native materialization
or inference. It can accompany either export. Successful requested inspection remains
available if later native export fails; partial failed inspection is cleaned up.
Inspection and native destinations must be distinct, and existing paths are refused.
`--output` and `--output-dir` conflict; numerical module export rejects tokenizer/chat
flags. `--chat-template` requires a tokenizer. Serialization `--format` affects the
module only; inspection is always text and stays outside the five-file package.

The public C++ interface is `lib/loomify/gguf_model_builder.h`:

```cpp
std::unique_ptr<GgufModelBuilder> builder;
if (!GgufModelBuilder::Open(gguf_path, &builder, &error)) return false;
GgufBuildOptions build;
build.library_root = library_root;  // explicit declared products/runfiles
build.prefill_capacity = 128;
build.context_capacity = 512;
GgufPackageOptions package;
package.output_directory = new_directory;
package.tokenizer = tokenizer_path;
std::unique_ptr<GgufConfiguration> configured;
if (!builder->Configure(build, &package, &configured, &error)) return false;
// These strings are precisely what the native construction consumes.
const auto& numerical_source = configured->configuration_source();
const auto& supplemental_source = configured->package_data_source();
return configured->WritePackage(context, workspace, &error);
```

For numerical-only construction, pass `nullptr` instead of package options to
`Configure`, then call `configured->BuildModule` and `BuiltGgufModule::WriteModule`
for a new text/bytecode file. Numerical configuration needs no tokenizer or supported
chat template. Configure performs no Loom product reads or linking; package
configuration compiles the selected Jinja source and uses the existing native tokenizer parser and bounded GGUF marker
scan. Empty resource/output roots are permitted for inspection. The result owns
copied facts, generated strings, options and resource root, and survives builder or
option destruction. Failed configuration preserves the caller's previous result.

The caller supplies native context/workspace synchronously when constructing the
module or package. The owned module retains its native dependencies. Both native
paths consume the prepared sources directly; they do not regenerate numerical
configuration. The library is `//lib/loomify:gguf_model_builder`, independent of the
CLI target and flags. Its finite host mappers live in `lib/loomify/architectures`;
model command order remains authored in Loom, while `tools/model` stays the shared
fact-reader library also used by the inspector.

The CLI selects its conventional runfiles-root string for native exports, and
loads only the selected BUILD-declared products after successful optional inspection. It
does not search the checkout or current directory. The library accepts an explicit
resource root without performing discovery. Private generated resource paths and
runfiles share one BUILD inventory. Native normalization uses the maintained embedded
`lib/loomify/materialize.loom` asset through the existing native pass API.

`//lib/loomify:package_test` is the explicit real-asset source/bytecode,
relocation, missing-provider and safe-publication gate. It requires
`LOOMIFY_TEST_GGUF`, `LOOMIFY_TEST_TOKENIZER`, `LOOMIFY_TEST_QWEN_GGUF` and
`LOOMIFY_TEST_QWEN_TOKENIZER`. The maintained preparation tests are
`//lib/architectures/llama:prepare_test` and
`//lib/architectures/qwen35:prepare_test`; they invoke actual preparation through
observing native VM imports and inspect records, registrations and rejected
requests. Existing Serve is the end-to-end consumer for prompt/decode/replay and
real registration failure checks. Generation/loading alone does not establish
numerical parity or resolve independent model-fidelity work.

The converter also accepts the known Qwen3.5 0.8B UD-Q5_K_XL checkpoint.
It selects the architecture from the owned GGUF descriptor, not the filename:

```sh
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --gguf=/path/to/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/qwen/tokenizer.json --output-dir=/path/to/exported-qwen \
  --context-capacity=512 --prefill-capacity=128 --format=bc
loom-serve --device=amdgpu --pool_backing=fixed --model=/path/to/exported-qwen/model.json \
  --listen=127.0.0.1:18081
```

Qwen numerical orchestration is maintained privately in
`lib/architectures/qwen35/model.loom`, behind the public root in `epoch.loom`.
Complete GDN/attention layer bundles retain their internal FFNs and qualified
numerical policies; the exporter selects and links the maintained product.

The generated Qwen model name is `loomify-qwen35`. It uses ordinary greedy text,
one active row/span, no MTP/vision or retained/prefix caches, T in1..512 and C from
T through262144. Its `generated.qwen_epoch` entry invokes the private model's
embedding, all configured hybrid layer bundles and selected-row normalized
vocabulary head, then publishes the selected private ID to the requested slot. It directly accepts the existing
epoch buffers. An experimental 0.8B T128/C512 configuration selects
`generated.qwen_decode_epoch` for stage1 and each one-token shape stage; other
configurations retain the original entry. Both roots share the same model body,
while the decode root specializes its static row count to one. Physical buffer
capacity, persistent GDN/KV state and stage order remain unchanged; unconsumed
residual scratch rows need not be rewritten. This experimental route has no
qualified performance claim. The owned prepare
program reports the complete convolution/recurrent GDN slot and paged KV geometry
so native request reset clears fresh state and subsequent chunks retain it. The
Qwen control policy combines generated source-derived controls with their matching
native terminal marker. The source supplies omitted prompt defaults.
For nonthinking greedy
chat, pass `enable_thinking:false` and `temperature:0`.

Source/bytecode export, relocation and failure tests share the package gate above.
Qwen preparation retains its distinct native state layout and registration policy.
End-to-end Serve acceptance and independent numerical fidelity remain distinct checks.


The retained layer state gate lives in
`lib/architectures/llama/testing/run_layer_validation.py`. It executes the actual
maintained layer wrappers and checks finite/nontrivial active output, exact
inactive residuals, reflected weight ranges, permuted cache-page crossings,
other-plane/prefix/suffix guards, continuation and reset. It is a state/ABI test;
independent arithmetic expectations remain in the operation/kernel suites.
Old-versus-new bitwise equality is temporary external migration evidence and is
not maintained as an independent numerical oracle. Real-weight FFN wrappers
still select actual Llama Q6-down layers0/1 and Q4-down layer3, and Qwen layers0/1.
The epoch packet tests now link maintained commands and the real relocated
unpack/scatter helpers, retaining their exact packet/token/cache expectations.

Package configuration compiles the actual selected GGUF chat template into prompt,
request, output/tool, completion and input controls. Native linking assembles these
controls with architecture execution hooks, shared greedy request policy and neutral
text helpers against the public hook contract. Native token binding validates consumed
GGUF globals and the analyzed terminal; numerical publication IDs are checked
separately. Missing providers, unresolved requested definitions, incompatible hook
signatures or mismatched tokenizers fail before publication. Numerical-only `BuildModule`
remains independent of those package checks.

`GgufPackageOptions::template_name` / `--chat-template=NAME` selects an explicitly named
template from the GGUF when needed. Supported literal and structural changes use the
same analysis/emission path as the original sources; unsupported constructs fail with
source diagnostics before publication. No response-profile option or fingerprint
catalog is required. Source-qualified reasoning metadata and USER continuation are
constructed with the generated controls; omitted request options remain absent.
Generation does not run the development synthetic conversation/assistant coverage
suite. Only templates requiring finite effort metadata execute source discovery,
stopping at the first active context and validating its exact source domain.
Other templates construct controls and USER products without native discovery.
The development qualifier retains the full differential source/selector/assistant
checks separately; missing synthetic coverage is not an export admission rule.
The model manifest stays unchanged. Runtime guard activation, a compatible published
runtime/pin/wheel and live Serve acceptance remain separate from this development
export path. Qwen packages use the honest family
name `loomify-qwen35`; control `model_name` is emitted from that same value.
Llama retains its explicitly selected greedy text baseline. Qwen float tool arguments
remain functional through the existing JSON representation; exact Hugging Face float
spelling is deferred, while other declared rendering/token/output checks remain.

## Gemma3 text profile

The Gemma3 profile derives its 26-layer layout, effective local window and global
RoPE base from GGUF metadata. It uses plain stored-gamma RMS normalization,
GeGLU, tied Q8_0 embedding/output weights, and mixed Q5_0/Q4_K/Q6_K projections.
Local layers use a 512-token causal window; global layers retain full logical
context. KV storage remains fully paged, without ring eviction. Prefill capacity
is at most512 and is distinct from logical context capacity; context1024 permits
bounded prompts crossing the local-window boundary.

The selected turns template owns BOS and turn framing. Output preserves visible
whitespace; future structured conversations rerender canonical history and replay
it without claiming a cached prefix. Tools, reasoning options and media requests
are rejected. The matching tokenizer JSON remains an external input. Its extra
added tokens may load, but encoded IDs outside the GGUF vocabulary are rejected
before numerical execution. Declared EOS and end-of-turn remain separate source
facts: the numerical publisher maps either raw stop to the canonical end-of-turn
ID, while other special IDs remain nonstops.

SOURCE/BC component and direct epoch tests cover numerical anchors, inactive
scratch initialization, offsets, state ownership, dependent decode and local/global
window sensitivity. These bounded checks do not establish independent full-model
numerical fidelity or qualify every admitted context size.

## Generation platform

`--platform=amdgpu:gfx11-generic` selects the maintained generation recipe and is
also the default. The C++ equivalent is `GgufBuildOptions::platform`; prepared
configurations and built modules expose `platform()`. Configuration inspection
records the selection without loading numerical products. Unknown recipes fail
before native generation or output publication.

The chosen BUILD-produced target resource is a mandatory numerical LINK input.
Exported SOURCE and bytecode retain the concrete wave32/wave64 requirements; a
missing selected resource fails rather than falling back to abstract targets.
Native LINK itself permits abstract `target.decl` inputs, so it does not certify
arbitrary modified provider blobs. Use the version-matched maintained resources.

This option does not choose a physical device. Serve still uses `--device` and
specializes kernels for its actual HAL profile, subject to the exported target
requirements. Exact gfx1100/gfx1151 compatibility tests do not add another
architecture. Other hardware-family recipes require implemented, qualified
providers; changing a target name alone does not supply them. GGUF extraction,
model configuration, text selection and numerical policy remain shared.
