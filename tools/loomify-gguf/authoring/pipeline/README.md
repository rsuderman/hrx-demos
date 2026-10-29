# Persistent-pipeline construction

`PipelineBuilder` builds Loom's actual persistent-dataflow `pipeline.def` using
existing native operations. Use it inside `ModuleBuilder::BuildNative`; the module
owner handles failure, linking and output ownership. Pipeline definitions can be
module roots directly, and kernel-scoped definitions/declarations can be passed to
`CommandProgramBuilder::Invoke` through a module-owned `KernelPipelineRef`.

The synchronous definition callback receives specialization/run operands and can:

- Define worker strands with explicit static origins, counts and strides.
- Select an invocation memory pool, allocate storage, form views and bind a FIFO.
- Reserve/publish producer records and acquire/release consumer records.
- Compose a child pipeline into the same invocation.
- Use existing native numerical operations in the current construction region.

Run types may be supplied directly or through `dependent_run`, which receives the
fresh specialization operands for building dependent shapes. The source type
system and substitution verifier remain Loom's. Worker domains are currently
static; more advanced native operations remain available through `Native()`.

All raw native values/types and `PipelineRef` objects are borrowed within one
module construction callback. Do not retain them across callback/module lifetime,
cloning or linking. Resolve a module-owned `KernelPipelineRef` afterward for
command invocation or pipeline-root selection. Return construction errors from
callbacks; the module owner prevents output after a native mutation failure.
Pipeline and strand terminators are added automatically on successful callbacks.

The tests contain complete examples, including initialized producer storage,
consumer retirement, child composition and direct command invocation. They verify
both serialization formats, dependent view/channel substitution, explicit external
pipeline providers and rejection of a generic pipeline as a kernel launch target.
Run them with:

```sh
bazel test //tools/loomify-gguf/authoring/pipeline:pipeline_builder_test
```

The roundtrip test retains `resident.loom` and `resident.loombc` as standard Bazel
undeclared test outputs. They use `target.decl @array` so construction tests remain
independent of a compiled target backend. To materialize on XDNA, supply a concrete
compatible target such as `aie2p.target<array> @array`, enable the existing `xdna`
backend, and compile with the appropriate device profile through normal Loom
tools. Source validation is separate from target compilation and execution.

The test pipeline publishes four initialized integers and consumes the record;
it has no external output. Target compilation can qualify its materialization,
but this fixture does not provide a numerical device-execution oracle. This
builder is one authoring capability; it does not generate a full model from GGUF.
