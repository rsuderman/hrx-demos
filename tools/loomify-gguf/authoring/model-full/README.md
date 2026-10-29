# GGUF-driven complete model-step construction

`llama_step` constructs all numerical stages of a supported Llama 3.2 3B
Q4_K_M model step in ordinary host C++, using `ModuleBuilder`, direct kernel
references, `CommandValues` and native template applications. Descriptor tensor
roles choose parameter names, byte extents and Q4_K/Q6_K providers. The host
unrolls every layer; the emitted commands do not call existing model command
bodies. Numerical kernels, motifs and templates remain existing Loom definitions.

This is a bounded full-computation recipe. It supports single-token prefill and
single-token decode. Batched prefill and a serving adapter are additional consumer
capabilities. Source construction and greedy-token agreement do not establish
independent tensor/logit fidelity of the numerical providers.

## Construction

```cpp
ProgramRef prefill, decode;
KernelRef rms_norm;
module.Kernel("llama32_rmsnorm_f32", &rms_norm);
DefineLlamaStep(descriptor, module, rms_norm, LlamaStepKind::kPrefill,
                "model.prefill", &prefill);
DefineLlamaStep(descriptor, module, rms_norm, LlamaStepKind::kDecode,
                "model.decode", &decode);
module.LinkAndVerify({prefill, decode}, &output, &result);
```

Every returned status and final result must be checked. `module` must already
contain the numerical kernel/helper/template definitions and matching config and
target definitions. The recipe uses existing Llama numerical adapter symbols;
unsupported architecture, geometry, required role, storage or byte extent is
rejected before defining a body. Numerical providers currently require width3072,
FFN8192, vocabulary128256, 28 attention layers, 24 query/8 KV heads, width128,
F32 norms/RoPE factors, tied Q6_K embedding/output, Q4_K query/key/attention-output/
gate/up and Q4_K or Q6_K value/down. The generic authoring API has no such limits.

The recipe preserves separate prefill and decode FFN templates. Template families
are emitted with native `template.apply`, not reinterpreted as kernels. Dependent
weight-view dimensions reuse the same SSA values supplied to those contracts.

The example executable accepts a GGUF and an explicitly supplied numerical
definition module:

```sh
bazel build //tools/loomify-gguf/authoring/model-full:emit_llama_step
bazel-bin/tools/loomify-gguf/authoring/model-full/emit_llama_step \
  MODEL.gguf NUMERICAL_DEFINITIONS.loombc OUTPUT.loombc
```

Use normal Loom linking to assemble the numerical definitions from the existing
model/shared source list and explicit runtime-capacity config. The emitter's
configured IR context admits their target operations. Its strict output roots are
`generated.llama_prefill` and `generated.llama_decode`; unreferenced original model
commands are excluded even when the input numerical module came from mixed source
files. Weights are external and are never embedded in the output.

## Authoring and selecting a numerical kernel

The RMSNorm interface is supplied by the caller as a `KernelRef`, reused across
all 57 normalization launches per step. The recipe checks its owner and declared
workload/argument types before creating a command body; incompatible or foreign
references are rejected without leaving an unfinished definition behind.

For a concrete authored replacement:

```cpp
KernelRef rms_norm;
DefineLlamaRmsNorm(module, "my.rmsnorm", &rms_norm);
DefineLlamaStep(descriptor, module, rms_norm, LlamaStepKind::kPrefill,
                "model.prefill", &prefill);
```

`DefineLlamaRmsNorm` authors a distinct kernel using native Loom builders. Its
256-invocation workgroup handles one row, and its body applies the existing
`llm.rmsnorm.row` numerical motif with width3072 and epsilon1e-5. It reuses that
motif's arithmetic/reduction implementation and preserves the disjoint-buffer
contract. Workload and launch row counts agree and use the existing [1,2048]
range and row-index bounds. This example is native Loom kernel authoring; the separate
[computation example](../computation/README.md) demonstrates loomcxx import.

The emitter's `--authored-rmsnorm` option constructs that replacement and passes
its reference to both full-model roots. Omitting the option selects the existing
RMSNorm kernel. Neither path changes the model command recipe or its oracle:

```sh
bazel-bin/tools/loomify-gguf/authoring/model-full/emit_llama_step \
  MODEL.gguf NUMERICAL_DEFINITIONS.loombc OUTPUT.loombc --authored-rmsnorm
```

Inspect the emitted module for `kernel.launch @authored.llama_rmsnorm` in all114
normalization sites across both roots, and its `template.apply<@llm.rmsnorm.row>`
body. No launch should select `@llama32_rmsnorm_f32` in this variant. Recompute
parameter reflection for the selected execution root before numerical validation.

## Invocation ABI

Both roots have no scalar specializations and seven buffer bindings:

| Binding | Responsibility |
| --- | --- |
| 0 | Immutable parameters packed according to this generated command's reflected named requirements |
| 1 | Residual workspace, at least one3072-element F32 row |
| 2 | Three I32 control words, with the stage-dependent layouts below |
| 3 | KV map resource; dense mode supplies a valid dummy allocation when pooling is disabled |
| 4 | Persistent F16 KV cache, `2 * 28 * cache_capacity * 1024` elements in the existing adapter layout |
| 5 | Mutable I32 token buffer, at least129 elements: current input/output at0, history ring at1..128 |
| 6 | Eight I32 progress words in the existing finish-adapter layout |

Initial prefill control is `{1, 0, canonical_eos}` with cleared cache/progress and
the first input token in token buffer0. Prefill produces
`{next_absolute_position=1, generated_count=1, canonical_eos}`. Decode consumes
that layout, increments position/generated count and publishes the next token and
history entry. The caller must preserve the same mutable state allocations,
provide valid token IDs and capacity bounds, and wait for completion before
consuming output or issuing a dependent invocation. Cache, map, control and token
state remain external; temporary activations belong to the command.

Use the existing named-parameter runtime consumer or gather from the GGUF index
into each reflected fixed-root placement. **A packed buffer from another command
is not interchangeable merely because it has the same total size.** This recipe
requests RoPE factors earlier than the original commands, so their fixed-root
offsets differ. GGUF file offsets, tensor-relative ranges and packed-root offsets
are separate quantities. A test must establish the binding layout before
interpreting a numerical result.

## Validation boundaries

`llama_step_test` checks supported profile admission, all-layer construction,
explicit prefill/decode template paths, strict source linking and text/bytecode
reload, both existing and authored RMSNorm selection, and recoverable rejection
of incompatible/foreign references. Its no-op numerical contracts isolate host construction; open template
applicability is subsequently resolved by the real numerical providers.

Real checkpoint execution requires external assets and target support. Validate
generated source closure, named parameter bytes, normal target compilation,
single-token prefill followed by multiple decode/cache invocations, and an
independent numerical policy. Keep greedy trajectory/state-publication checks
separate from independent cache/residual/logit comparisons. Existing-source tensor
qualification gaps also apply to a generated composition using those providers;
they are not waived by correct module construction.
