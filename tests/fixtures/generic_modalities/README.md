# Generic modality onboarding structural fixtures

Initial catalog: one CHAT executable and one recipe. Addition DATA: another recipe
on that executable, a second CHAT executable, and a nonpreparing ASR executable.
Names and catalog paths intentionally differ from product model names.

Source declarations reuse frozen, reviewed maintained native source closures.
Installed fixtures reuse genuine pinned-compiler output from `native-fixture.zip`,
verified by the literal hashes here. The fixture manifests represent normalized
artifact DATA, not a claim that the uniform producer has emitted them. Source and
installed manifests, exact source identities, role/member expectations and recipe
coordinates in the two release JSON files are authored before any producer runs.
The harness never generates its release oracle from staged/produced inventories.

Tiny prepared assets exercise bookkeeping/config/export only. They do not prove
model loading, native ABI behavior, throughput or inference. Real compiler/producer
and genuine artifact qualification remain separate until their actual APIs are
composed. No mocked reader/selector/checker or source family dispatch is used.

The two *-build.bzl files are literal fixture build DATA for the actual uniform
producer. Its separate harness case reuses these accepted compiled inputs and
executes the real Loom decoder built from the current `MODULE.bazel` HRX pin.
It calls real prepare/stage and the strict independent installed checker, without
filtering producer receipts or deriving release expectations from its output.
The archive contains five bytecode members, their 82 matching source files, and
the retained licenses. The source closures are Qwen 0.8B's 66-file pipeline and
Moonshine's 16 files across four compiler groups. Generic catalog IDs, recipe
assets, serving settings and compiled roles remain fixture-owned DATA; they do
not select model-specific Python behavior.

`native-fixture.json` records the archive/member SHA256 identities, authored
source commit, actual HRX compiler pin, compiler/decoder identities and reviewed
release-data identity. These artifacts are newly compiled with that pin; they
are not historical wheel outputs or bytecode translated by an older decoder.
The ordered groups and policy rewrites use the existing `bytecode_catalogs`
convention: `loom-link INPUTS --mode=merge --to=bc --output=OUTPUT`, followed by
the same pin's `loom-format` and unchanged source/compiled proof checks.

Refreezing for a new compiler or source baseline requires reviewing source
membership, roots, counts, roles, policy and generic serving contracts before
compilation. Then update the archive, source declarations, installed declarations,
receipts and both release files together. Release expectations must not be
derived from producer observations. Current working model edits alone must not
silently change these frozen inputs. The current archive was serialized twice
with identical bytes and contains no task-specific source paths.

Run the required eight-case integration lane from the repository root with an
isolated source-test venv and explicit `LOOM_SERVE_DIR`:

```bash
"$python" -m build_tools.ci_tools.source_tests onboarding \
  --decoder "$decoder" --fixture-directory "$new_task_directory"
```

The shared action verifies and extracts the frozen bundle into a new caller-owned
directory and replaces all inherited `GENERIC_MODALITY_FIXTURE_*` variables.
No historical wheel, source checkout, network download or compiler double is
required. The test script fails if any of its eight tests fail or skip, or if the
expected test count changes. The caller owns cleanup of the extracted directory.

`.github/scripts/build_python_wheel.sh` always runs this lane after building the
wheel, in its own venv with the built wheel installed and the actual pinned
decoder. Failure stops the build before wheel repair, upload and cache admission.
Both GitHub CI and default full local CI invoke that script. The ordinary
`*_test.py` source discovery runs before compilation and does not include this
native integration lane. The development-only `build-wheel.sh --fast` command
builds an unrepaired host wheel and does not run these CI checks.
