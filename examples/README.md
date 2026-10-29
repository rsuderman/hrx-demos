# Examples

Runnable demonstrations of library APIs belong here. Reusable implementation
code lives under `lib/`; user-facing command-line tools live under `tools/`.

The [connected FFN example](operations/connected_ffn.cc) declares operation
contracts and builds two connected invocations:

```sh
bazel run //examples/operations:connected_ffn
```

Run its library tests with `bazel test //lib/operations/testing:all`.
