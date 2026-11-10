# loomify-gguf

The thin CLI for the reusable [native converter library](../../lib/docs/gguf-export.md).
That guide covers configuration inspection, numerical module export, Serve packages,
C++ API, supported capabilities and validation.

```sh
bazel build //tools/loomify-gguf:loomify-gguf
bazel-bin/tools/loomify-gguf/loomify-gguf \
  --gguf=/path/to/model.gguf --configuration-output=/path/to/new-inspection
```

Generated output and model assets remain external. The developer converter is not
shipped in the serving wheel. Architecture/tensor reporting remains a separate
[model inspector](../loom-model-inspect/README.md).
