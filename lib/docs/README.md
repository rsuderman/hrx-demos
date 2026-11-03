# Library documentation

Reusable code belongs here, with native tests in each library's `testing/`
subdirectory. Executable tools live under `tools/`, runnable demonstrations
under `examples/`, and Python tests under `tests/`.

- [Operations](operations.md): typed operation definitions and graph authoring.
- [Examples](examples.md): runnable graphs and native test commands.
- [Llama FFN structure](llama-ffn-structure.md): descriptor-backed normalized FFN
  region construction, ownership and bounded structural tests.

- [Llama structure](llama-structure.md): immutable ownership, parameter identity,
  runtime inputs and known region cuts.
- [Dispatch registry](dispatch-registry.md): known typed dispatches and owned native providers.
- [Dispatch shapes](dispatch-shapes.md): model-derived logical result inference.
- [Dispatch invocation](dispatch-invocation.md): output-producing native norm and whole FFN authoring.
- [Selected FFN compilation](llama-ffn-region.md): retained GGUF owner, static
  one-row region emission and source/bytecode proof boundaries.
