# GGUF model export

`loomify-gguf --stage=model` constructs the supported Llama3.2 3B Q4_K_M
architecture by deriving data-only configuration and parameter bindings from the
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
  --stage=model --gguf=/path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --tokenizer=/path/to/tokenizer.json --output-dir=/path/to/exported-llama \
  --context-capacity=512 --prefill-capacity=128 --format=bc
loom-serve --device=amdgpu --model=/path/to/exported-llama/model.json \
  --listen=127.0.0.1:18081
```

`--format=loom` writes textual numerical source instead of bytecode. Both formats
serialize the same configuration-linked module. The package contains `model.json`, `model.loombc` (or
`model.loom`), `prepare.loom`, `control.loom` and `weights.loom`. Package-owned
references are relative, so this directory can move; external asset references
are absolute and still need their original files. The converter never copies
checkpoint bytes or converts the tokenizer. It refuses an existing output
directory and publishes `model.json` only after provider serialization succeeds.

The bounded Llama serving profile is ordinary greedy text with one active request,
no draft/checkpoint/session/prefix cache, maximum prefill chunk T in1..512 and
context C between T and131072. It uses Serve's existing packed-epoch path with
one span. Physical KV pool capacity rounds C up to64 tokens. The native prepare
program checks that requested capacities match the generated program. The text
control policy retains the existing Llama chat framing and end-of-generation
mapping. Isolated Serve step execution is explicitly unsupported by this package;
the generated numerical prefill/decode roots remain usable through their direct
seven-buffer interface with the exported offsets and requirements.

The maintained programs in `lib/llama/programs` own embedding, repeated decoder
layers, normalized vocabulary selection and publication. `lib/programs/ffn`
contains their shared FFN sequence; each profile selects its numerical packing
and normalization providers. Generated configuration supplies tensor bindings,
layer extents/quantization and capacities, without generating command order.
The maintained epoch translates one-span packets, invokes prefill at the maximum
configured row capacity, and scatters the selected token. Active count masks
unused rows; one-token continuation follows the same path. The model build
returns its owned module together with the admitted storage layout. Preparation
copies that layout's cold allocation, row-view, origin and KV geometry records;
it does not independently recalculate model extents. Named immutable parameters
preserve canonical GGUF keys; compiler-reflected packed root offsets are distinct
from file offsets.

The converter still supports its independent F32 embedding demonstration:

```sh
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --stage=embedding --gguf=/path/to/f32-model.gguf --output=embedding.loombc
```

Host/provider/package and CLI failure tests are in
`//tools/loomify-gguf/serving:serving_test` and `:cli_test`. The packet replay
`:epoch_execution_test` covers consecutive spans, inactive storage, selected and
unselected results, EOS and capacities below/above129 history words on native
AMDGPU. `:prepare_execution_test` evaluates production preparation and compares every cold
record to the admitted layout. `:model_export_test` is an explicit manual gate using
`--test_env=LOOMIFY_TEST_GGUF=/path/to/the/supported.gguf` and
`--test_env=LOOMIFY_TEST_TOKENIZER=/path/to/tokenizer.json`; it constructs the full
model, writes and relocates both package formats, reloads all providers and plans
all three command roots from the moved package.
The manual `:cli_model_test` also requires `LOOMIFY_TEST_TOKENIZER` and exports
both formats outside the checkout before relocating the generated package.
Existing Serve is the end-to-end consumer. Successful generation/loading alone
does not establish numerical parity or resolve independent model-fidelity work.

The same model mode also accepts the known Qwen3.5 0.8B UD-Q5_K_XL checkpoint.
It selects the architecture from the owned GGUF descriptor, not the filename:

```sh
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --stage=model --gguf=/path/to/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/qwen/tokenizer.json --output-dir=/path/to/exported-qwen \
  --context-capacity=512 --prefill-capacity=128 --format=bc
loom-serve --device=amdgpu --model=/path/to/exported-qwen/model.json \
  --listen=127.0.0.1:18081
```

Qwen full-model export retains its existing C++ recipe; the shared configuration-
linked Qwen FFN proof does not convert its complete hybrid architecture.

The generated Qwen model name is `loomify-qwen-08b`. It uses ordinary greedy text,
one active row/span, no MTP/vision or retained/prefix caches, T in1..512 and C from
T through262144. Its sole `generated.qwen_epoch` entry is built by the descriptor
recipe in `lib/qwen/program.h`: embedding, all24 registered hybrid layer bundles,
then selected-row normalized vocabulary head. It directly accepts the existing
epoch buffers; a one-token continuation uses the same entry. The owned prepare
program reports the complete convolution/recurrent GDN slot and paged KV geometry
so native request reset clears fresh state and subsequent chunks retain it. The
Qwen control policy keeps Qwen framing/EOS behavior and requires ordinary inference;
it does not use Llama's terminal mapping or an isolated numerical buffer shim.
For the ordinary chat profile, pass `enable_thinking:false` and `temperature:0`.

Qwen companion tests are in `//tools/loomify-gguf/serving/qwen:package_test`,
`:prepare_execution_test` and `:packet_execution_test`. The VM component tests run
production preparation/packet bodies with valid-input test callbacks for server
imports; they inspect all cold geometry records and replay increasing/decreasing
active token counts while checking the whole token-envelope zero fill. The manual
`:model_export_test` and `:cli_model_test` require explicit
`LOOMIFY_TEST_QWEN_GGUF` and `LOOMIFY_TEST_QWEN_TOKENIZER` fixtures. They exercise
complete generated source/bytecode packages, command products and relocation, plus
actual CLI conversion outside the checkout with owned provider runfiles. End-to-end
Serve acceptance and independent numerical fidelity remain distinct checks.
