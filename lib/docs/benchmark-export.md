# Standalone numerical benchmarks from GGUF

`loomify-benchmark` reuses the numerical model selected by Loomify and exports
standalone cases for supported input recipes. It does not run a GPU benchmark
during generation, and needs neither an external tokenizer nor Loom Serve.

Inspect the selected command sequence without copying tensor data:

```sh
loomify-benchmark --gguf=MODEL.gguf --phase=decode \
  --plan-only --output=NEW_PLAN.json
```

Export a bundle for the same workload:

```sh
loomify-benchmark --gguf=MODEL.gguf --phase=decode \
  --context-capacity=512 --history-tokens=128 \
  --allow-partial --output=NEW_BUNDLE
```

Without `--allow-partial`, every selected dispatch must have a runnable recipe.
Partial export keeps every unsupported invocation and its reason in the manifest;
it does not claim full-model benchmark coverage. An export with no runnable cases
fails. Plan inspection remains available independently of fixture coverage.

Prefill defaults to capacity128, with active tokens equal to capacity. Use
`--tokens=128 --active-tokens=97` to select a partial workload. Decode capacity and
active count are both one; contradictory values reject. Context capacity defaults
to512 and history to0. `--seed` controls deterministic synthetic inputs, while
weights come from the actual GGUF. `--platform` selects the maintained generation
platform, currently `amdgpu:gfx11-generic`; it does not select a live device.

`--weights=representative` is the default. Shared operation recipes declare which
immutable weight spans may be represented by one real layer's data. All origins
and original weight keys/ranges remain recorded, and substitutions are explicit.
`--weights=all` preserves distinct real-weight cases. Other scalar, control,
state and alias differences remain separate. Generation has no arbitrary per-case
byte cap by default. `--max-input-bytes=BYTES` supplies an explicit prepared-root
resource budget; the manifest records it separately from kernel support.
Budget-excluded cases retain descriptive resource-limit reasons. Actual source read,
truncation and consistency failures still abort, including partial export.

The normal resource root is discovered through the tool's Bazel runfiles. Use
`--library-root=PATH` for an explicitly supplied maintained numerical catalog.
There is no model-cache lookup or automatic download in the generator.

The output is one module with all emitted benchmark records:

```text
NEW_BUNDLE/
  manifest.json
  benchmark.loombc
  benchmark.loom
  fixtures/<content-hash>.npy
```

Use `--emit-source=false` to omit the readable source form. The module and relative
fixtures can be moved together and consumed without the original GGUF, generator
checkout or resource catalog. Output paths must be new, including plan/report
JSON files. Existing files, empty directories and dangling symlinks are never
replaced. Publication requires Linux atomic no-replace support; unsupported hosts
or filesystems fail instead of overwriting a destination.

Run an emitted case using its `benchmark_symbol` from the manifest:

```sh
loom-benchmark NEW_BUNDLE/benchmark.loombc --dry-run --output=plan.json
loom-benchmark NEW_BUNDLE/benchmark.loombc --device=amdgpu \
  --benchmark=@SYMBOL_FROM_MANIFEST --measure=dispatch_complete \
  --output=results.json
```

Generated `check.case` records invoke the retained kernels directly in a
contiguous sequence. This preserves their exact selected body predicates instead
of routing through a newly specialized command wrapper. Actual execution compiles
the kernels before running them; dry-run alone does not establish compilation or
execution success. Use compile-only only for a product form the runner supports;
the scenario form used in the initial checkpoint rejects that mode.

Valid fixture setup and exact native replay are separate from independent
arithmetic checking. The manifest records authored `safety_checks` (such as tail
canaries and immutable weights/seeds) and separately records `arithmetic_check`.
A native correctness result can therefore mean safety assertions passed while
arithmetic remains `unchecked`; it never implies a mathematical comparison passed.
Root metadata distinguishes logical bytes from fixture-only guard suffix bytes.
The input-footprint preflight counts logical payloads; artifact sizes include
guards and file headers.
Every actual weight source is listed, including operations that read several
parameter tensors. Mutable cases explicitly include reset
work before each target invocation and are labeled `reset_plus_kernel`. Their
scores include those copies; do not subtract an unrelated reset measurement or
present them as pure kernel times.

## Importing measurements

Reports read native snapshot JSON with an explicit result index:

```sh
loomify-benchmark report --manifest=NEW_BUNDLE/manifest.json \
  --index=result-index.json --output=NEW_REPORT.json
```

The index binds the manifest ID and exact result-file SHA-256 to explicit native
candidate coordinates. Result paths are relative to the index file. Copy case
identity/input policy from the export and candidate identifiers from the actual
native result; do not infer them from array positions or kernel names.

```json
{
  "format": "loomify.benchmark.result-index.v1",
  "manifest_id": "MANIFEST_ID",
  "results": [{
    "file": "results.json",
    "sha256": "SHA256_OF_RESULTS_FILE",
    "producer_identity": "ACTUAL_RUNNER_BUILD_IDENTITY",
    "exclusive_device": false,
    "bindings": [{
      "case_id": "CASE_ID",
      "input_policy_digest": "INPUT_POLICY_DIGEST",
      "candidate_id": "ACTUAL_NATIVE_CANDIDATE_ID"
    }]
  }]
}
```

Report counts distinguish selected, runnable, measured and qualified invocation
coverage. Missing or noisy measurements remain visible. A leaf score times its
model multiplicity is a standalone work proxy, not predicted model latency or
tokens/second. Sequence/reset scores are not multiplied by their dispatch-origin
count. Numerical-phase estimates require separately measured compatible sequence
partitions; absent estimates remain absent.

The manifest hashes emitted module/fixture bytes and records copied tensor ranges.
It does not imply a full-checkpoint content hash. `generator_identity` identifies
the generation build (the CLI uses its executable SHA-256); report
`producer_identity` independently identifies the runner used for measurement.

## Development

The generator uses native selected-command inspection and source-metadata-free
bytecode serialization. This development checkpoint needs the companion HRX SDK
providing those APIs; repository pin qualification remains a separate delivery
gate. Generated standalone modules use the existing benchmark runner. Focused
tests live under `lib/loomify/benchmark/testing`; real assets are supplied through
explicit fixture environment variables and are never downloaded by tests.
