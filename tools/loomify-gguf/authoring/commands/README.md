# Command values and model bodies

`CommandValues` constructs the planning values and storage used around direct
kernel invocations in a `CommandProgramBuilder`. It uses native Loom operations
and the builder's module/value ownership checks.

```cpp
CommandValues values(*program);
Value count, stride, bytes, scratch, weight;
values.Config("model.width", &count);
values.Constant(4, LOOM_SCALAR_TYPE_OFFSET, &stride);
values.Scale(count, stride, &bytes);
values.Allocate(bytes, 256, &scratch);
values.Parameter(program->Binding(0), "blk.{}.weight",
                 {program->Specialization(0)}, LOOM_SCALAR_TYPE_F32,
                 {count}, &weight);
program->Invoke(kernel, {count}, {count, weight, scratch});
```

The excerpt omits status handling for readability. The emitter example checks
every status and returns errors. Kernel interfaces determine the actual operand
groups; the example does not prescribe a universal numerical signature.

Named parameters are immutable content requirements associated with an explicit
buffer root. Keys and index substitutions name the content; dimensions describe
the logical rank-1/2 view. The runtime parameter provider supplies the named bytes
and command preparation packs their buffer locations. GGUF absolute offsets are
not command-buffer offsets.

The supported planning operations include config reads; integer/address constants;
address casts; index arithmetic and byte scaling; integer mask selection; dependent
parameter views; and aligned global scratch. Structured `If`, `Serial`, and
`Concurrent` callbacks belong to the command builder. Create a `CommandValues`
inside a callback to emit values in that scope. Outer values can be captured;
values created in a closed nested scope cannot escape it.

Malformed key patterns, unsupported ranks/constants, invalid alignments, missing
configuration and operand type mismatches are rejected through read-only preflight
before native construction. These errors leave this program and sibling builders
usable. Floating-point-to-address casts are rejected. Failures after native
mutation follow the shared builder's terminal failure policy. Finalization uses
normal Loom verification and linking.

`emit_body_example.cc` builds a small staged numerical body: a config read controls
scratch byte sizing, a layer-substituted named parameter supplies a bias, and a
planning branch selects a second dispatch's scale. This demonstrates operations
required by the existing projection/layer programs without porting a model.

Validation targets:

- `//tools/loomify-gguf/authoring/commands:command_values_test`: logical view
  identity, config/arithmetic/selection, scope rejection, preflight errors and
  text/bytecode reload.
- `//tools/loomify-gguf/authoring/commands:command_body_execution_test`: executes
  the C++-generated command on AMDGPU for both branch configurations against an
  independent scalar formula. Input/parameter/output owners include guard regions
  and nonzero subview origins. This fixture supplies an already packed parameter
  root; it does not exercise a GGUF file provider or full-model inference.
