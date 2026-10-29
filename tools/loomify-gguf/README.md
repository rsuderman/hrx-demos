# loomify-gguf

`loomify-gguf` is the C++ developer foundation for generating Loom model
programs. The current scaffold prints an empty loomcxx function to stdout:

```sh
bazel run //:loomify-gguf > /tmp/model.cxx
bazel run //:loomify-gguf -- --help
```

The `:source_builder` library constructs source independently of CLI/output
handling. The CLI uses pinned HRX build rules and flag handling; it has no model
loading, reporting, registry or compiler dependency. Architecture and tensor
inspection belongs to [loom-model-inspect](../loom-model-inspect/README.md),
using the shared [model loading libraries](../model/README.md).

The source has a stable Loom symbol, `model.pipeline`, and imports as an ordinary
empty function. It is groundwork for model construction; it does not build an
executable model. To import it with the optional pinned HRX frontend:

```sh
bazel run --@hrx_system//loom/config/import:enable=cxx \
  @hrx_system//loom/src/loom/tools/loom-import-cxx -- \
  --root=model_pipeline --to=bc --output=/tmp/model.loombc /tmp/model.cxx
```

Generated source and bytecode should be written outside the checkout during
development. This is a developer target and is not included in the serving wheel.
