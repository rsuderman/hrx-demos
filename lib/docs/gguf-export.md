# GGUF model export

`loomify-gguf --stage=model` constructs the supported Llama3.2 3B Q4_K_M
architecture from the GGUF descriptor, invokes Loomify's registered numerical
implementations, and writes the model document and source providers consumed by
existing Loom Serve. GGUF weights and matching tokenizer JSON remain external.

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
use the same model recipe. The package contains `model.json`, `model.loombc` (or
`model.loom`), `prepare.loom`, `control.loom` and `weights.loom`. Package-owned
references are relative, so this directory can move; external asset references
are absolute and still need their original files. The converter never copies
checkpoint bytes or converts the tokenizer. It refuses an existing output
directory and publishes `model.json` only after provider serialization succeeds.

The bounded serving profile is ordinary greedy text with one active request,
no draft/checkpoint/session/prefix cache, maximum prefill chunk T in1..512 and
context C between T and131072. It uses Serve's existing packed-epoch path with
one span. Physical KV pool capacity rounds C up to64 tokens. The native prepare
program checks that requested capacities match the generated program. The text
control policy retains the existing Llama chat framing and end-of-generation
mapping. Isolated Serve step execution is explicitly unsupported by this package;
the generated numerical prefill/decode roots remain usable through their direct
seven-buffer interface with the exported offsets and requirements.

The recipe in `lib/llama/program.h` remains responsible for the numerical layer
sequence. The serving wrapper translates one-span packets, calls that prefill
program at its maximum configured row capacity, and scatters the selected token.
Active count masks unused rows; a one-token continuation follows the same path.
The prepare program retains the host-owned page map at byte2048 in origins and
uses the private arena extent as the generated cache base. Named immutable
parameters preserve canonical GGUF keys; compiler-reflected packed root offsets
are distinct from file offsets.

The converter still supports its independent F32 embedding demonstration:

```sh
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --stage=embedding --gguf=/path/to/f32-model.gguf --output=embedding.loombc
```

Host/provider/package and CLI failure tests are in
`//tools/loomify-gguf/serving:serving_test` and `:cli_test`. The packet replay
`:epoch_execution_test` covers consecutive spans, inactive storage, selected and
unselected results, EOS and capacities below/above129 history words on native
AMDGPU. `:model_export_test` is an explicit manual gate using
`--test_env=LOOMIFY_TEST_GGUF=/path/to/the/supported.gguf`; it constructs the full
model, reloads both serialization formats and plans all three command roots.
Existing Serve is the end-to-end consumer. Successful generation/loading alone
does not establish numerical parity or resolve independent model-fidelity work.
