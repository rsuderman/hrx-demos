# loomcxx pipeline authoring

This tree provides editable C++ authoring for numerical kernels, motifs and
families. Its outputs are ordinary verified Loom modules, consumed by existing
compiler and [Loomify authoring APIs](../tools/loomify-gguf/authoring/command/command_program_builder.h).
The production `.loom` catalog stays in [loom-pipeline](../loom-pipeline/llm/README.md).
Import, linking and compilation use the current HRX pin.

## Add a computation family

1. Add numerical formulas, templates, configuration bindings and kernel launch
   definitions to a family source under `llm/`, such as `normalization.cxx`.
   Related motifs/model adapters/entrypoints belong together. Keep exact math,
   memory, dispatch and launch contracts in this source.
2. Add one build declaration to `llm/BUILD.bazel`. Existing raw-buffer
   public families declare their bridge tuples here as
   `(public_family, provider, imported_family, mode, public_argument_types)`:

   ```starlark
   loomcxx_module(
       name = "normalization",
       src = "normalization.cxx",
       bridges = [
           ("qwen38.rmsnorm_f32.row", "qwen38_rmsnorm_f32_row",
            "cxx.qwen38.rmsnorm_f32.row", "raw",
            ["index", "index", "offset", "f32", "buffer", "buffer", "buffer"]),
           # The actual normalization declaration also maps the shared family.
       ],
   )
   ```

   `raw` forwards arguments unchanged; `zero-origins` appends a zero byte
   offset after every raw buffer, matching the importer's pointer ABI. New
   families without an existing public ABI need no bridges. The assembler's
   mapping file is generated internally, never maintained beside each family.
3. Add focused numerical/contract cases beside the source. `loom_test_matrix`
   shares static fixture expansion, definition-set selection, GPU launch policy
   and test suites; `loom_gpu_test` also serves distinct priority-provider tests.
   Keep numerical oracle, case extents, model configuration and provider fixtures
   explicit in the family source/tests/BUILD declaration.

No new emitter executable, importer/header scaffold, consumer harness, directory
or README is needed for an ordinary family. The shared integration suite grows
only when a new authoring interface needs a witness, not for each algorithm.

## Import, ABI assembly and products

`build/loomcxx.bzl` delivers the exact upstream facade headers as declared inputs,
invokes `loom-import-cxx`, and assembles products through one shared
`support:assemble` executable. Explicit header delivery avoids the pin's builtin
header catalog naming issue without compiler patches or machine paths.

The bounded ABI assembler reads explicit mapping data, validates argument types,
imported device convention, public signatures and provider identities, publishes
coarse public declarations with default-priority device providers, and retains
kernel export spellings. It contains no family names or numerical bodies.
Mappings support void providers with `index`, `offset`, `i1`, `f32` and `buffer`
arguments. Each row owns distinct public/provider/imported symbols. Unsupported
modes/types, duplicates and signature mismatches fail; mixed per-buffer pointer
adaptation, results, other conventions and arbitrary transformations are outside
this interface. A failed mapping is not permission to change the public contract.

Source guards `LOOMCXX_PROVIDERS_ONLY` and `LOOMCXX_CALLERS_ONLY` select products:

- `normalization.loombc` contains kernels and providers.
- `normalization_providers.loombc` contains reusable providers without entries.
- `normalization_callers.loombc` contains kernel callers without implementations.

The macro defaults to all three; `products` may select a subset of `full`,
`providers`, `callers`. Link callers with providers or a deliberately selected
alternative public provider set. Choose one original/alternative definition set
rather than linking duplicate implementations. Serving recipes still select the
existing catalog; no runtime authoring switch is introduced.

The normalization build declaration demonstrates both adaptations through the same assembler.
The shared RMSNorm numerical helper takes C++ pointers, lowered to three
`(buffer, offset)` pairs; its public raw buffers need zero origins. The Qwen
adapter takes typed raw buffers and forwards them unchanged. Decode calls public
Qwen; its adapter and both prefill kernels call public shared RMSNorm. Selected
public providers therefore affect every intended call site. Private `cxx.*`
families implement the computation and are not caller dispatch shortcuts.

## Loomify consumers

Create an owned `ModuleBuilder` from verified definitions, resolve a `KernelRef`,
preflight its interface, invoke it from `CommandProgramBuilder`, then finish with
`LinkAndVerify`. Imported Qwen F32 prefill has `(index)` workload and
`(index, f32, buffer, buffer, buffer)` device arguments, including explicit epsilon.
The current fixed-epsilon Llama model recipe instead consumes the existing
[`DefineLlamaRmsNorm`](../tools/loomify-gguf/authoring/model-full/README.md)
adapter over providers-only output:

```cpp
KernelRef norm;
DefineLlamaRmsNorm(*module, "my.llama_norm", &norm);
```

Its supported width3072/epsilon1e-5 configuration yields the model-compatible
`(index)` workload and `(index, buffer, buffer, buffer)` device interface accepted
by `DefineLlamaStep`. The shared `testing:loomify_consumer_test` checks both paths,
source/builder lifetime independence, owned output roundtrip, exact interface and
owner rejection, and actual numerical closure. This is bounded authoring
integration evidence, not full GGUF/model generation or serving qualification.

## Validation

```sh
bazel test //loomcxx-pipeline/llm:normalization_original_execution_test \
  //loomcxx-pipeline/llm:normalization_alternative_execution_test \
  //loomcxx-pipeline/llm:shared_selection_test \
  //loomcxx-pipeline/llm:qwen_selection_test \
  //loomcxx-pipeline/testing:all //loomcxx-pipeline/support:mapping_test \
  --@hrx_system//loom/config/import:enable=cxx \
  --test_env=LD_LIBRARY_PATH --test_env=IREE_HAL_AMDGPU_LIBHSA_PATH
```

Normalization cases use an independent serial sqrt oracle with widths1/257/1024/
3072/5120, rows1/2/3, positive epsilon variants, independently configured Qwen
divisor128.5 at width257, nonuniform inputs/weights, F32/F16 outputs, subspans,
bitwise guards and input/weight preservation. Tolerances are2e-5 for F32 and2e-3
for F16. Four existing callers include the unchanged rooted Llama output-norm
command. Shared-family priority override produces23 through all four callers;
Qwen priority override produces41 through decode only, making bypass observable.
Support tests cover malformed mappings and imported-signature admission failures.

Manual pin-investigation probes and historical reports live in the external
workspace, outside product source. Last-row prefill, exceptional inputs, other
shapes/targets, full-model serving and performance remain unqualified. Broad
family migration is separate work.
