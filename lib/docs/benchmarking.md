# Benchmark and optimize authored workloads

For a kernel, numerical sequence, or model performance change, choose an explicit
semantic boundary and oracle, then use upstream Loom tools to check, measure, and
inspect that workload. Ordinary GGUF/module/package generation remains with
[the model builder](gguf-export.md).

See [GPU implementation contracts](hardware-support.md) for the separation of
generation requirements, physical-device qualification and runtime smoke checks.

## 1. Match tools and choose the boundary

Use the HRX revision pinned in `MODULE.bazel` for fixture generation, execution,
and report inspection. Keep HRX immutable: adapt the workload/Loomify or select an
existing published revision through normal pin review, and record upstream
capability gaps. Record the Solutions revision/diff, HRX revision, tool
identities, target/device, input recipe, and measurement policy in the external
workspace. Keep build state, results, and model/NPY inputs outside the checkout;
use the task's isolated venv for Python development tools.

`loom-test`, `loom-benchmark`, and `loom-compile-report` are upstream development
tools; installing the Solutions wheel does not provide them. Obtain a matching
[Loom development installation](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
or build their targets through this checkout's pinned `@hrx_system` dependency.
Check each tool's `--help` before using the flags below. An unavailable capability
is a tooling blocker to record before choosing another route.

Name what the score includes: one kernel, preparation plus projection, a layer,
or an application request. Include necessary packing, copies, resets, and
synchronization equally in competing implementations. For model claims, prove
the implementation and launch shape actually selected by that model. Preserve
indirect dispatch and control flow by executing the authored subject. Compile
reports provide diagnostic evidence, not an execution ABI to reconstruct.

**Ready:** the workload, target, state/reset behavior, and performance boundary
are explicit, and matching tools are available.

## 2. Reuse an operation-owned scenario and oracle

[The gated FFN fixture](../programs/ffn/testing/gated_fixtures.py) provides
`paired_q4_r1_c1_k256_n256_{baseline,fused}` scenarios and benchmark declarations.
They compare two implementations of Q4 gate/up projection plus SwiGLU, including
activation packing. Expected values come from independent logical anchors in
[fixture_helpers.py](../kernels/qwen/testing/common/fixture_helpers.py).
The scenarios check output/guards and unchanged input/weight storage. A separate
GPU parity scenario supplements the independent oracle. This synthetic fixture
needs no GGUF or saved activations and proves only its authored shape/numerics.

From the repository root, set `BENCHMARK_RUN_DIR` to a new absolute external
directory and put matching Loom tools on `PATH`. Build the existing merged module:

```bash
set -euo pipefail
: "${BENCHMARK_RUN_DIR:?Set a new absolute directory outside the checkout}"
mkdir "$BENCHMARK_RUN_DIR"
bazel --output_base="$BENCHMARK_RUN_DIR/bazel" build --config=opt \
  //lib/programs/ffn/testing:gated_text
fixture="$(bazel --output_base="$BENCHMARK_RUN_DIR/bazel" info --config=opt bazel-bin)/lib/programs/ffn/testing/gated_text.loom"

loom-benchmark "$fixture" \
  --benchmark=@bench_paired_q4_r1_c1_k256_n256_fused \
  --dry-run --output="$BENCHMARK_RUN_DIR/plan.json"
```

The [BUILD target](../programs/ffn/testing/BUILD.bazel) merges the fixture with
production providers. A dry run checks selection and planning; it neither
compiles the candidate nor proves correctness. Extend operation-owned checks
with deterministic nonuniform inputs, real packed-format semantics, aliases,
guards, and tolerances. Prefer a small independent scalar or VM reference;
an optimized reference backend can separately own performance.

**Ready:** the plan selects the intended benchmark and the oracle/expectations
cover the changed operation, including its state and memory boundaries.

## 3. Require correctness before timing

Run both concrete scenarios on compatible gfx11 hardware:

```bash
for route in baseline fused; do
  loom-test "$fixture" \
    --case="@paired_q4_r1_c1_k256_n256_$route" \
    --device=amdgpu --target=amdgpu:gfx11-generic \
    --sanitizer=access >"$BENCHMARK_RUN_DIR/$route.correctness.json"
done
```

Inspect reports and exit status: every selected trial must execute and pass,
with no unexpected skip or planning issue. Access sanitization checks the
physical memory envelope; the authored oracle checks the mathematics.
`check.compare` uses independent target/oracle storage from the same recipe;
`check.invoke` is target-only. The normal HAL/VM differential checks a resultless
subject through mutable post-state. Choose a reference that compiles under its
profile; arbitrary GPU code is not automatically a VM/x86 oracle.

**Ready:** all intended checks pass on the exact candidate to be measured.

## 4. Measure matching work and data reuse

Scenario benchmarks default to unchecked target-only execution. Enable their
authored correctness checks explicitly for a checked measurement. This comparison
selects one trial and a deliberate cache-hot reuse policy:

```bash
loom-benchmark "$fixture" \
  --compare=@bench_paired_q4_r1_c1_k256_n256_baseline,@bench_paired_q4_r1_c1_k256_n256_fused \
  --sample=0 --device=amdgpu --target=amdgpu:gfx11-generic \
  --scenario-correctness=true --measure=dispatch_complete \
  --batch-size=64 --input-ring-count=1 \
  --interleave=ABABA --repetitions=2 \
  --output-format=jsonl --output="$BENCHMARK_RUN_DIR/comparison.jsonl"
```

`dispatch_complete` measures host submission through queue completion, normalized
per logical operation. Scenario setup, oracle execution, comparisons, and readback
are outside timing. Use optimized tools without sanitizers for performance and
coordinate exclusive device use. Keep cache-hot, streaming, and model-resident
claims separate: ring count 1 does not model a full model's weight reuse. For
streaming experiments use matched `--input-ring-min-bytes` policies and inspect
effective ring/byte counts. Retain dispatch count, batching, reset costs,
warnings, and variation with scores. Noisy evidence stays inconclusive;
a microbenchmark gain alone does not establish model throughput.

**Ready:** both candidates use equivalent inputs/reuse policies, correctness is
recorded, and timing evidence supports the stated boundary.

## 5. Explain results with separate profile and compile evidence

Collect a separate diagnostic run for one candidate:

```bash
loom-benchmark "$fixture" \
  --benchmark=@bench_paired_q4_r1_c1_k256_n256_fused \
  --sample=0 --device=amdgpu --target=amdgpu:gfx11-generic \
  --scenario-correctness=true --measure=dispatch_complete \
  --batch-size=64 --input-ring-count=1 --profile-final-batch=true \
  --compile-report=details --artifact-bundle-policy=debug \
  --artifact-bundle-dir="$BENCHMARK_RUN_DIR/profile.bundle" \
  --output="$BENCHMARK_RUN_DIR/profile.json"

for report in "$BENCHMARK_RUN_DIR/profile.bundle"/compile_reports/*; do
  loom-compile-report show "$report"
done
```

HAL profiling performs a separate instrumented replay after scoring. Dispatch
timestamps, counters, and traces depend on target support and retained metadata;
check completeness and perturbation warnings. Its device time cannot be
subtracted from the ordinary run's host score. Compile reports explain emitted
code, register/memory use, and compiler decisions. Command-program reports are
available in published HRX `a18539632138e7094efda6732a10763f3dfcd90a`; an older SDK
may provide kernel reports only. Indirect launches retain count-buffer references
and static branch evidence does not identify an executed path. Read reports with
their producing tool version; use `diff` under its explicit identity contract.

Keep compact source/tool/workload identities, exact commands, numerical results,
score policies, warnings, and relevant report observations in the external
workspace. Preserve failed trial configuration/ordinal/entropy for replay.
Then stop task-owned build services and remove disposable builds/venvs/exports
and redundant output. Preserve immutable model inputs and other agents' services.

**Complete:** the numerical/performance claim has reproducible evidence, or its
concrete blocker/inconclusive measurement is recorded without a gain claim.

## NPY and capture limits

Use `check.file.read.npy` for exact external fixtures and explicit expectations
for golden comparisons. `check.case` supports `check.file.write.npy` with
`always`/`on_failure` and an explicit `loom-test --file-output-root`; writes are
evidence, not an oracle. Current `check.scenario` planning rejects those writes;
large-buffer streaming capture is unavailable through that interface. Prefer
compact generated inputs and differential oracles. Automated PGO/capture-to-
scenario generation is future work, not a prerequisite for this workflow.

For larger ports, follow the upstream
[GGML/llama.cpp oracle workflow](https://rocm.github.io/hrx-system/loom/workflows/oracles/ggml-llama-cpp/)
for physical GGUF contracts, selected-backend evidence, semantic boundaries and
matched reuse. Tool details remain upstream:
[correctness](https://rocm.github.io/hrx-system/loom/workflows/test-correctness/),
[benchmarking](https://rocm.github.io/hrx-system/loom/workflows/benchmark/),
[result interpretation](https://rocm.github.io/hrx-system/loom/workflows/benchmark-results/),
and [compile reports](https://rocm.github.io/hrx-system/loom/workflows/compile-reports/).

## Matrix integration coverage

Use `//lib/programs/projection/testing/matrix:integration_iq_full` for the broad
IQ matrix correctness batch. It selects SOURCE and bytecode checks for
`iq4_nl`, `iq3_s`, `q3k`, `canonical`, `shared_preparation` and `policy_qkv`.
The first three full fixtures contain their respective tiny cases, so do not
also select those six tiny modules in this full batch. All tiny-only developer
targets remain available; `classic_tiny` has a distinct direct-kernel provider
and is not covered by this omission rule.

`//lib/programs/projection/testing/matrix:integration_iq_full_modules` builds
the corresponding twelve modules; it is not an execution inventory. Before
hardware execution, expand the test suite with the same source and configured
Bazel wrapper outside the device lease:

```bash
bazel query 'tests(//lib/programs/projection/testing/matrix:integration_iq_full)'
```

Require exact equality with the reviewed twelve concrete
`<family>_<text|bc>_test_execute_amd_gpu_hardware_test` labels, and verify their
hardware resource metadata. Stop on missing, extra or nonhardware members;
never silently filter future suite members. Build the selected products before
taking the shared GPU lease and preserve all scenario, trial and expectation
checks under that lease.

The matrix families share one ordered SOURCE/bytecode provider pair. Legacy
`<family>_provider_<text|bc>` labels still produce their original basenames as
byte-for-byte copies for runfile consumers. Provider sharing changes build
work, not fixture contents, numerical policies or assertions. Measure build
and execution time separately using the same source, pin, configuration and
inputs; reduced action or module counts alone do not prove a speedup.
