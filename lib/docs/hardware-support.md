# GPU implementation contracts and bring-up

The maintained generation recipe is **`amdgpu:gfx11-generic`**. It selects
concrete targets and numerical implementation products through the existing
[platform mapping](../loomify/platforms.cc) and native linker. This is a
generation choice; Loom Serve separately selects a physical device. Another
device may reuse an existing implementation if it satisfies its contracts.
Source reuse or successful compilation alone does not qualify that device.

Model/GGUF configuration, pipeline sequencing, matrix/QKV/FFN composition,
preparation, allocation arithmetic, codecs and ordinary activation math stay
shared. Selected implementations own actual instruction/fragment/lane layouts
and staging schedules. A constant thread or tile count is not, by itself, a
reason to copy its surrounding algorithm into a GPU-specific directory.

## Requirements and their owners

| Contract | Owner and reuse boundary |
| --- | --- |
| Matrix row alignment | The selected [scaled-F16 product](../platforms/amdgpu/gfx11/projection/BUILD.bazel) defines `llm.platform.projection.scaled_f16.row_alignment`, currently 64. Shared [output-row rounding](../programs/projection/matrix.loom) consumes it for allocation and preparation. Actual leaf row/column legality still applies. |
| Attention launch shape | The selected [H256 schedule configuration](../platforms/amdgpu/gfx11/attention_h256_config.loom) defines `llm.platform.attention.h256.query_tile_rows` and `workgroup_size`, currently 16 and 256. Shared dense/span wrappers consume these facts; the current body retains its wave64 fragment/lane contract. |
| Matrix/attention device schedule | [Projection](../platforms/amdgpu/gfx11/projection) and [H256 attention](../platforms/amdgpu/gfx11/attention_h256_schedule.loom) own their actual MMA fragments, lane mapping, LDS staging and numerical traversal. A different instruction/layout strategy can require a different body; compatible targets need not duplicate one. |
| Fixed-width arithmetic | The [compatibility dot reduction](../kernels/projection/qk_mmvq_compat.loom) specifies an ordered 32-lane summation tree. Preserve that numerical algorithm and target requirement; substituting the current wave width changes the arithmetic. |
| Reusable workgroup code | Canonical dot helpers, normalization, GDN, elementwise kernels and Q8 packing include reusable algorithms. Some use target subgroup width; others use workgroup reductions/barriers. Verify their actual launch, lowering and numerical requirements on a new target before copying or retargeting them. |
| Encoded data layout | Q4/Q6 block sizes, byte strides, packing and rounding are format/numerical facts. Keep [codecs](../kernels/projection/f16_staging.loom) shared rather than treating those constants as backend tuning parameters. |

Backend facts are ordinary linked Loom configuration, supplied by the selected
product before specialization/compilation. They are not GGUF values, per-model
defaults, runtime device arguments or arbitrary command-line tuning knobs.
Missing or incompatible definitions must fail native materialization rather
than silently choosing another layout. The existing
[linked-configuration tests](../loomify/testing/linked_config_test.cc) exercise
real shared consumers, allocation/launch consistency, missing/invalid facts,
SOURCE/bytecode reload and final request compilation.

Performance choices are distinct from legality. For example, the current
short-row matrix cutoff of 16 is a shared routing heuristic. Changing a cutoff,
unroll factor or staging policy requires numerical and performance evidence;
the GPU's name does not justify a different value. Expose a linked parameter
only when a real shared consumer needs it, and preserve coherent requirements
between the producer, preparation, allocation and selected kernel.

## Selecting generation and execution independently

The existing local smoke script forwards `--platform` to the converter and
`--device` to Serve independently. Its JSON records both requested values and
the observed runtime devices. Defaults remain `amdgpu:gfx11-generic` and
`amdgpu`; it never guesses a generation recipe from a physical device string.
Supply immutable local assets and already-built tools:

```bash
"$TASK_PYTHON" build_tools/loomify-smoke.py \
  --gguf="$GGUF" --tokenizer="$TOKENIZER" \
  --converter="$CONVERTER" --serve="$SERVE" \
  --platform="$PLATFORM" --device="$DEVICE" \
  --report="$REPORT"
```

Use a task-owned interpreter/state namespace and distinct ports as described in
the [build guide](../../build_tools/README.md). The script checks SOURCE/bytecode
generation parity, progress/accounting and owned shutdown using temporary state.
It is a functional lifecycle check, not an independent numerical oracle or a
sustained throughput benchmark. Model weights are not downloaded by this script.

## Bringing up a real device

1. Establish the actual compiler/runtime target profile, wave/fragment support
   and resource limits. Match them to the selected concrete
   [target requirements](../platforms/amdgpu/gfx11/targets.loom). Register a new
   recipe only with an actual implementation; changing a selector string does
   not provide missing instructions or schedules.
2. Inspect the generated model and selected implementation using the authored
   workloads and upstream compile evidence described in the
   [benchmarking workflow](benchmarking.md). Record static launch rows, physical
   capacity and runtime active rows separately. Compile reports are diagnostic
   evidence, not an execution ABI or proof of physical execution.
3. Reuse existing SOURCE/bytecode component tests with independent numerical
   references, full/tail/inactive cases and memory guards. Then qualify complete
   blocks, dependent decode/cache state, and the actual generated package on the
   device. Keep missing assets and unsupported cases explicit.
4. Use the recipe-aware smoke command for bounded serving/lifecycle checks.
   Measure fixed-work kernel/block and model prefill/decode/request throughput
   separately, retaining before/after values and variability. A component gain
   or compilation pass does not establish a model throughput gain.
5. Update the affected [coverage entries](coverage.md), provider/association tests
   and operation-owned workload scenarios alongside source changes. Keep exact
   input, ABI/grid and state expectations strict; a genuinely different
   implementation needs matching authored cases rather than relaxed checks.
   Retain compact evidence, release owned services/GPU leases and remove
   disposable outputs.

This guide is a family-level contract map, not a second kernel registry. Source,
BUILD ownership and executable tests remain authoritative. No additional GPU
family or backend is qualified by this documentation.
