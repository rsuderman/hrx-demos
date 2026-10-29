# C++ computation in a host-authored command

`testdata/affine.cc` authors a kernel and reusable numerical helper using supported
C++ and the existing loomcxx headers. `emit_example.cc` is ordinary host C++: it
imports that source with `loomc_module_import_cxx`, obtains a module-owned kernel
reference, and passes it directly to `CommandProgramBuilder::Invoke`. It strictly
links the selected command root and writes the complete module as `.loombc`.

The helper remains a function within the numerical computation. The importer
does not construct the command program, and no child command wrapper is needed
to invoke the kernel. The example releases frontend source and imported-module
handles before finalization to exercise the owning handoff.

The example supplies its required exact-pin facade headers through the existing
importer's source-provider callback. Header identities are logical `loomcxx/…`
names, with no dependency on installed headers or machine-specific include paths.

Build or run the emitter with:

```sh
bazel run --@hrx_system//loom/config/import:enable=cxx //tools/loomify-gguf/authoring/computation:emit_example -- path/to/affine.cc output.loombc
```

`computation_import_test` checks that the emitted closed module retains the helper
call and direct kernel launch and reloads as bytecode and text.
`authored_execution_test` executes the host-authored command on AMDGPU and compares
against a separately written scalar oracle. It checks seven signed integer inputs,
two guard elements on each end of the output, and unchanged input storage.

This example establishes the computation-authoring connection. It does not generate
a model from GGUF or qualify persistent pipelines on GPU.
