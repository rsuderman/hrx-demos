# GGUF model export

`loomify-gguf --stage=model` constructs the supported Llama3.2 3B Q4_K_M
and Qwen3.5 0.8B UD-Q5_K_XL architectures by deriving data-only configuration and parameter bindings from the
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

The maintained programs in `lib/architectures/llama` own embedding, repeated decoder
layers, normalized vocabulary selection and publication. `lib/programs/ffn`
contains their shared FFN sequence; each profile selects its numerical packing
and normalization providers. Generated configuration supplies tensor bindings,
layer extents/quantization and capacities, without generating command order.
The maintained epoch translates one-span packets, invokes prefill at the maximum
configured row capacity, and scatters the selected token. Active count masks
unused rows; one-token continuation follows the same path. The model build
returns its owned numerical module and configuration source. Maintained preparation
reads the same configuration to construct native cold allocation, row-view, origin
and KV geometry records. The exporter links that preparation module and writes its
native text serialization; no C++ geometry object or template rendering is involved. Named immutable parameters
preserve canonical GGUF keys; compiler-reflected packed root offsets are distinct
from file offsets.

The converter still supports its independent F32 embedding demonstration:

```sh
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --stage=embedding --gguf=/path/to/f32-model.gguf --output=embedding.loombc
```

The public C++ interface is `tools/loomify-gguf/gguf_model_builder.h`:

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
return builder->WritePackage(context, workspace, build, package, &error);
```

Call `BuildModule` instead to obtain a move-only `BuiltGgufModule`, then use
`WriteModule` for a new text/bytecode file. Numerical-only builds need no Serve
companions or tokenizer. The caller supplies its configured native context and
workspace; the owned module retains its native dependencies. Package writing calls
the same numerical construction routine and links preparation from its exact saved
configuration. The CLI reads declared bytecode products and companions from its
runfiles; it does not search the current checkout for model sources.

`//tools/loomify-gguf:package_test` is the explicit real-asset source/bytecode,
relocation, missing-provider and safe-publication gate. It requires
`LOOMIFY_TEST_GGUF`, `LOOMIFY_TEST_TOKENIZER`, `LOOMIFY_TEST_QWEN_GGUF` and
`LOOMIFY_TEST_QWEN_TOKENIZER`. The maintained preparation tests are
`//lib/architectures/llama:prepare_test` and
`//lib/architectures/qwen35:prepare_test`; they invoke actual preparation through
observing native VM imports and inspect records, registrations and rejected
requests. Existing Serve is the end-to-end consumer for prompt/decode/replay and
real registration failure checks. Generation/loading alone does not establish
numerical parity or resolve independent model-fidelity work.

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

Qwen orchestration is maintained in `lib/architectures/qwen35/model.loom`.
Complete GDN/attention layer bundles retain their internal FFNs and qualified
numerical policies; the exporter selects and links the maintained product.

The generated Qwen model name is `loomify-qwen-08b`. It uses ordinary greedy text,
one active row/span, no MTP/vision or retained/prefix caches, T in1..512 and C from
T through262144. Its sole `generated.qwen_epoch` entry invokes
embedding, all24 configured hybrid layer bundles,
then selected-row normalized vocabulary head. It directly accepts the existing
epoch buffers; a one-token continuation uses the same entry. The owned prepare
program reports the complete convolution/recurrent GDN slot and paged KV geometry
so native request reset clears fresh state and subsequent chunks retain it. The
Qwen control policy keeps Qwen framing/EOS behavior and requires ordinary inference;
it does not use Llama's terminal mapping or an isolated numerical buffer shim.
For the ordinary chat profile, pass `enable_thinking:false` and `temperature:0`.

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
