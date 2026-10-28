# loomify-gguf

`loomify-gguf` is the C++ developer foundation for generating Loom model
programs from GGUF checkpoints. It currently prints an empty loomcxx function
to stdout:

```sh
bazel run //:loomify-gguf > /tmp/model.cxx
bazel run //:loomify-gguf -- --help
```

The `:source_builder` library owns source construction independently of CLI
and file handling. Both targets use the pinned HRX shared C++ build macros; the
CLI uses HRX flag handling. This source-only tool does not need the C++ importer
or a GPU to run.

The output has a stable Loom symbol, `model.pipeline`, and imports as an ordinary
empty function. It is groundwork for model construction. GGUF input, Qwen 0.8B
validation, numerical providers, and inference are future stages.

To import the output into bytecode using the optional pinned HRX frontend:

```sh
bazel run --@hrx_system//loom/config/import:enable=cxx \
  @hrx_system//loom/src/loom/tools/loom-import-cxx -- \
  --root=model_pipeline --to=bc --output=/tmp/model.loombc /tmp/model.cxx
```

Generated source and bytecode should be written outside the checkout during
development. The tool is a developer target and is not added to the serving
wheel.
