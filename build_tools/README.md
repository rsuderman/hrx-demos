# Build and validation commands

These tools build the `hrx-solutions` wheel from the HRX revision pinned in
`MODULE.bazel`, check its payload and client assets, and prepare releases.
Application code and source tests live in `binding/python/` and `tests/`;
generic Loom runtime tests remain in HRX. Model weights are external inputs.

The commands below describe the current layout. The cleanup specification is
[`docs/specs/build-tools-cleanup.md`](../docs/specs/build-tools-cleanup.md).

## Formatting

Before handoff or commit, install the pinned developer tools in a new external
directory, add their executables to `PATH`, then run:

```bash
bash build_tools/format.sh --fix
bash build_tools/format.sh --check
```

With no option the command checks only. Both modes require the versions in
`build_tools/format-tools.env` for Black, clang-format, and Buildifier; these
three tools must already be installed. Formatting is also
checked by the lightweight GitHub workflow and local CI before source capture.
Loom formatting uses `loom-format` from the HRX revision pinned in
`MODULE.bazel`. A verified standalone copy can be reused across checkouts and
raw staged snapshots. Every selected file is still checked; the cache stores a
formatter executable, not formatting results. A C/C++ toolchain is required for
population, but no GPU or model assets are needed.

The ordinary uncached path retains `BAZEL` (including operational wrappers) and
`HRX_FORMAT_BAZEL_ROOT` (absolute external Bazel state). Set
`HRX_FORMATTER_CACHE=off` to explicitly retain custom wrapper build semantics.
Default canonical cache hits bypass the wrapper. Cache entries live under
`HRX_FORMATTER_CACHE_ROOT`, an absolute external directory defaulting to the
project workspace's `tools/loom-format`.

To populate canonical entries, provision the published Linux x86-64 Bazel 9.3.2
binary separately and set `HRX_FORMATTER_BAZEL` to its absolute path. Its SHA256
must be `2e14c55c298752d71e8a0bbd762665ffce5966c9ad00ec5d1a8b82a58e7bfa06`;
Bazelisk and arbitrary wrappers are not population launchers. Acquire it from
`https://releases.bazel.build/9.3.2/release/bazel-9.3.2-linux-x86_64`, verify the
published checksum, and make it executable. Format calls never download it.
Population uses `/usr/bin/clang` and `/usr/bin/clang++`, ignores all Bazel rc files,
sets explicit strict/pinned dependency and task/VM/static-tool options, and uses
a temporary owned Bazel output. It shuts down that output and removes it after
copying and checking the standalone executable. A configured invalid launcher
or failed canonical build is an error. Without canonical setup, unsupported
hosts/pins or local module overrides use the existing uncached path, with no
cache publication.

Entries identify the snapshot's MODULE/lock/version bytes, published HRX pin,
host ABI, compiler/Bazel bytes and maintained recipe. The executable hash is
checked on every use. Partial, corrupt or symlinked entries never execute;
concurrent writers publish atomically. The resolver is the cache owner and
keeps the current entry plus the newest entry for another pin, skipping entries
held by active format calls. Deleting the cache when no formatter is running is
safe. Task cleanup removes population scratch, not these documented reusable
tool entries. No private SDK, shared compiler library or runfiles are retained.
The pre-commit hook selects the staged pin even when the working tree has a
different unstaged pin. Install Bazelisk as `bazel` for the uncached fallback.
The tools are not dependencies of the installed wheel. Setup on Linux x86-64:

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source build_tools/format-tools.env
format_tools="${CODEX_PROJECT_WORKSPACES:-$HOME/codex/project-workspaces}/hrx-solutions/tools/formatting"
python3 -m venv "$format_tools"
"$format_tools/bin/python" -m pip install "black==$BLACK_VERSION" "clang-format==$CLANG_FORMAT_VERSION"
curl --fail --location --silent --show-error \
  "https://github.com/bazel-contrib/buildtools/releases/download/v$BUILDIFIER_VERSION/buildifier-linux-amd64" \
  --output "$format_tools/bin/buildifier"
echo "$BUILDIFIER_LINUX_AMD64_SHA256  $format_tools/bin/buildifier" | sha256sum --check --status
chmod +x "$format_tools/bin/buildifier"
curl --fail --location --silent --show-error \
  "https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-amd64" \
  --output "$format_tools/bin/bazel"
chmod +x "$format_tools/bin/bazel"
export PATH="$format_tools/bin:$PATH"
```

Python 3.10 or newer runs the tools; Black preserves the project's Python 3.9
target. Clang-format uses the LLVM-based root configuration with include sorting
disabled. Buildifier formatting does not enable lint fixes. Its `listsort` and
`label` rewrites are disabled with separate repeated flags so custom Loom merge
rules retain source order and literal label spellings. The command formats
tracked Bazel/Starlark, C/C++ (including `.cxx`), Python, `.loom`, and
`.loom-test` files. Loom uses `--in-place` for fixes and `--check` for
verification. It skips symlinks, deleted/untracked files, build/vendor trees,
and unsupported types such as Jinja, JSON, and TypeScript. Tracked candidates
resolving outside the checkout are rejected before modification.

Install the tracked commit hook once per clone (shared by its worktrees):

```bash
git config --local core.hooksPath .githooks
```

The hook requires Bash, Git, `flock`, and `sha256sum` on Linux.
It checks a disposable snapshot of the **staged contents**, including
partially staged files, through the same `format.sh --check` command. It never
rewrites files or stages fixes. On failure, run `format.sh --fix`, review the
changes, and stage the fixes you intend to commit. The runner automatically
uses the documented external `tools/formatting/bin` installation, including from editor Git hooks.
`HRX_FORMAT_TOOLS` can select another installation directory; otherwise the
pinned tools must be on `PATH`. Hook snapshots are serialized per checkout and removed on exit. Their locks live under the external project
workspace's `tools/formatting-hooks`; `HRX_FORMAT_HOOK_ROOT` can select another
external directory. Git does not enable tracked hooks automatically on clones;
CI remains the shared enforcement point. If you already use a custom hooks
path, arrange for its pre-commit hook to call `bash build_tools/pre-commit.sh`
instead of replacing it.

Two Python preparers are intentionally excluded because their exact bytes are
trusted by operation descriptors and release catalogs:

- `loom-pipeline/recognition/moonshine/preparer/prepare_assets.py`
- `loom-pipeline/speech/pocket_tts/preparer/prepare_assets.py`

Keep the initial bulk formatting separate from functional or tooling commits.
After that formatting commit exists, record its final full hash in
`.git-blame-ignore-revs` in a following bookkeeping commit. To use the tracked
list locally, run `git config blame.ignoreRevsFile .git-blame-ignore-revs`, or use
`git blame --ignore-revs-file .git-blame-ignore-revs PATH`. Clones do not inherit
the local Git setting. Preserve the formatting commit at integration, or update
the tracked hash if history changes.

## Early Loomify CPU validation

Before expensive GPU work, run the reviewed admission/materialization batch from
this checkout root in a caller-owned task venv. Supply the three existing immutable
fixtures (`LOOMIFY_TEST_MINICPM_GGUF`, `LOOMIFY_TEST_GGUF`, and
`LOOMIFY_TEST_HYPERCLOVAX_GGUF`); the helper fails before Bazel if any is missing.
Set an absolute task-owned `LOOM_SERVE_DIR` outside the checkout, disable user site,
and remove inherited `PYTHONPATH`/`PYTHONHOME`. For example:

```bash
TASK_VENV/bin/python build_tools/loomify-validation.py native \
  --bazel /absolute/task-owned/bazel-wrapper \
  --result-dir /absolute/new/evidence/native --jobs 4
```

The wrapper owns the SDK, output base, build-memory limits and explicit action,
host and test environment. The helper creates none of them. It runs only four
reviewed CPU tests (projection materialization, FFN capability, embedding
materialization and configured Llama), propagates the exact fixture variables,
and overrides `local_test_jobs` only for that batch. `--jobs` accepts 1–4 and
defaults to 4; keep hardware execution serial. Missing, stale, empty, failed or
skipped XML returns nonzero. A caller must stop on failure before GPU work.

The new external result directory contains the command log, current Bazel JSON
events and a `result.json` with source/config identities, per-case results,
elapsed time and child resource usage. Optional `--input-record FILE` and repeatable
`--rc FILE` record existing caller input/config identities; `--rc` records a file
already selected by the wrapper and does not configure Bazel. Fixture payloads
are not rehashed. On cancellation, the helper interrupts and reaps the client,
then invokes the same exclusive task wrapper's public `shutdown` with a bounded
wait to reconcile detached Bazel work. A failed/timed-out shutdown records
`cleanup.status=unresolved`; its owner must reconcile that output before reuse
or removal. No unrelated Bazel output or process names are targeted. Linux child `peak_rss_kib` is the largest child's high-water
mark, not a simultaneous aggregate-memory measurement; measure the complete
executor separately when comparing job settings. This early batch supplements
existing final verification; it does not replace local CI or model execution.

### Prebuilt hardware batches

Use the same caller-owned environment to run a reviewed file of concrete hardware
test labels, one per line. This mode does not require the native batch's three
GGUF fixture variables; prepare and identify any inputs needed by the selected
hardware tests separately. Run the required native batch first and stop on failure.

```bash
TASK_VENV/bin/python build_tools/loomify-validation.py hardware \
  --bazel /absolute/task-owned/bazel-wrapper \
  --targets-file /absolute/hardware-targets.txt \
  --gpu-lock /absolute/workspace/gates/qwen08-gpu.lock \
  --result-dir /absolute/new/evidence/hardware
```

The helper accepts only explicit supported `*_execute_amd_gpu_hardware_test`
leaves. It checks their source-owned resource tags and configured rule membership,
builds them, then records launcher/runfile identities before acquiring the shared
GPU lock. Duplicate labels, wildcards, suite aliases and unsupported metadata fail
preflight. If selecting the matrix `integration_iq_full` suite, the caller must
first query its complete test expansion with the same source/wrapper and require
equality with the reviewed twelve-leaf manifest; do not filter away unexpected
host tests or use the module filegroup as an execution inventory.

Hardware execution remains serial through Bazel's public test launcher, preserving
its runfiles and resource tags. The lock is blocking, so prepared immutable work
waits rather than rebuilding after a busy exit. Source, wrapper, explicit config,
input-record and manifest identities are checked again after waiting. The test
command is uncached and records a full JSON profile; any product-building action
inside the lease fails the optimization check. Rebuild outside the lock and repeat
the affected proof. The helper also requires fresh XML and successful, nonempty
Loom trial/expectation reports without skips, planning errors or expected failures.

Results keep build, lock-wait and test intervals separate, with complete reports
and selected launcher/runfile hashes. No other process may mutate this caller's
Bazel output while a batch runs. Cancellation interrupts and reaps the client,
then uses the same public wrapper shutdown. If shutdown cannot confirm retirement
while the hardware lease is held, the foreground helper records unresolved
cleanup and retains the lock while retrying shutdown; the owner must resolve that
blocker, rather than treating the lease as released. Cancellation while waiting
never takes or releases another owner's lock. No GPU concurrency or serving
lifecycle behavior changes are enabled by this command.

### Reusing accepted qualification

An integration may retain an accepted execution only when its complete executed
boundary is associated with final source: relevant source and whole program/launch
products, pin/compiler/runtime/ABI, platform, flags, fixtures/oracles and selected
scenarios must match. Record the prior result, unchanged boundary, retained
coverage and fresh checks in the existing external integration record before
running gates. Target names, pass counts or an unchanged leaf kernel alone do not
establish equivalence. Missing identities, unexpected skips or changed inputs
require affected revalidation rather than assumed reuse.

Final composition checks, required reviews/formatting, applicable SOURCE/bytecode
and negative-contract checks, bounded final-product smoke and owned lifecycle
shutdown remain required. Reuse does not waive full pre-landing local CI or grant
reduced-coverage authorization. Retain compact successful evidence; do not keep
SDKs or exports indefinitely. A later rebuild can associate old execution proof
only when the relevant complete product bytes and runtime/input identities match.
The designated integrator owns the final reuse decision and its external protocol.

## Choose a gate

| Need | Command | Coverage and owning implementation |
| --- | --- | --- |
| Full local CI | `bash build_tools/local-ci.sh [--model08 DIRECTORY] [--run-dir DIRECTORY]` | Source capture, source client assets, Python tests, wheel build, installed CPU and client checks, GPU backend, Open WebUI; `ci_tools/local_ci.py` |
| Explicit reduced coverage | `bash build_tools/local-ci.sh --cpu-only [--run-dir DIRECTORY]` | Same stages through installed UI checks; skips GPU and Open WebUI, still requires Docker and network |
| Build a repaired wheel | `bash build_tools/build-wheel.sh` | Pinned manylinux container; `.github/scripts/build_python_wheel.sh` and `wheel_tools/` |
| Check an existing wheel | `bash build_tools/check-wheel.sh WHEEL NEW_VENV_DIRECTORY [--native-lifecycle]` | Installed payload/import/console-script checks and CPU lifecycle fixtures; `validation/check_installed_{wheel,cpu}.py`; optional empty AMDGPU server lifecycle |
| Real-model backend | `bash build_tools/model-smoke.sh WHEEL NEW_RUN_DIRECTORY MODEL08_DIRECTORY` | Isolated Qwen 0.8B backend/configuration/lifecycle qualification; `validation/hem_model_smoke.py`, profile `ci08` |
| Real-model Open WebUI | `bash build_tools/model-smoke.sh --openwebui WHEEL NEW_RUN_DIRECTORY MODEL08_DIRECTORY` | Public WebUI wiring against isolated native models; `integrations/openwebui/openwebui_configuration_smoke.py` |

Default local CI runs the full GPU and client gate. CPU-only is explicit reduced
coverage and does not satisfy the repository's pre-landing gate without user
approval. `--model08` is incompatible with `--cpu-only`.

For an affected installed CPU contract, invoke the existing
`build_tools.validation.check_installed_cpu` runner from outside the checkout
using the isolated wheel venv and a new absolute `LOOM_SERVE_DIR`. Pass
`--source-root CHECKOUT --evidence EXTERNAL_JSON` and repeat `--test EXACT_ID`
for each full discovered ID, for example
`--test tests.curated_assets_test.CuratedAssetsTest.test_managed_lifecycle_reserves_named_files_and_retained_catalog`.
The runner admits and loads the complete fixture inventory before selection;
do not trim `HELPERS` or `CONTRACTS` in adapters. Unknown or duplicate IDs fail
before test execution. Omitting `--test` retains the full installed suite and
all source parity and payload checks.

### Invocation, prerequisites, and outputs

All four top-level shell commands locate this checkout from their own path and
can be invoked by absolute path from another directory. Relative wheel, model,
and output arguments resolve from the caller's directory. Examples here assume
the repository root. Use external absolute directories for validation state.

- **Local CI:** Linux x86-64; the pinned formatting tools above; Bash, Git, Python 3 with venv/pip, Docker daemon,
  curl, tar, timeout, realpath, and sha256sum. Full mode also needs accessible
  `/dev/kfd` and a render node, the supported AMD backend, and model inputs.
  It captures source inputs into a private snapshot and runs checks there.
  `--run-dir` must be new and outside the checkout. By default, evidence is in
  `$CODEX_PROJECT_WORKSPACES/hrx-solutions/gates/local-ci/`, or
  `~/codex/project-workspaces/hrx-solutions/gates/local-ci/` when unset, under
  a unique run directory. Outputs include `summary.json`, stage logs, source
  manifests, the snapshot's wheel, and installed/real-model evidence; the
  private runtime directory is separately reported and retained.
- **Build wheel:** Linux x86-64 with Docker, Python 3, and network access.
  The container acquires pinned Bazelisk, HRX/build inputs, TheRock toolchain,
  Python packages, and build dependencies. Outputs are under `build/wheel/`,
  with the repaired wheel in `build/wheel/wheelhouse/`. The checkout is mounted
  writable. Use a task worktree or local CI's external source snapshot.
  `HRX_BAZEL_CACHE_DIR` enables an optional external cache unless
  `HRX_BAZEL_CACHE_ENABLED=0`; see the root README for the cache policy.
- **Check wheel:** an existing `.whl`, Python 3 with venv/pip, and a destination
  that does not exist. Installs the exact wheel with `--no-index --no-deps`,
  executes outside the checkout, verifies installed import provenance, and
  writes `wheel-check/installed-cpu-evidence.json` inside the new venv. It
  isolates Loom state below the venv, or the private runtime root supplied by
  local CI. CPU fixtures use an isolated state directory. Native lifecycle
  requires an available AMD GPU; it does not run real-model inference.
- **Model smoke:** an existing wheel, Python 3 with venv/pip, network access for
  installation dependencies, and a new or empty external run directory.
  The backend qualification targets gfx1151. Outputs include `install.log`
  and `lifecycle/` evidence; all owned state stays within the instance's
  namespace. Always pass the immutable local fixture for development:
  `Qwen3.5-0.8B-UD-Q5_K_XL.gguf` and adjacent `tokenizer.json` in the supplied
  directory. The default fixture location is
  `$HOME/Downloads/gguf/models/multimodal/qwen/qwen3.5-0.8b`.
  Omitting `MODEL08_DIRECTORY` on the backend route requests acquisition of
  the pinned recipe; do not use that route for cleanup validation. Open WebUI
  requires an explicit model directory and Docker; it inspects or pulls the
  pinned image and records `webui-image.log` plus lifecycle evidence.

Local CI acquires the shared external `gates/qwen08-gpu.lock` around its GPU
and WebUI stages. When invoking real-model checks directly, acquire that same
workspace lock once around the check and its cleanup. Separate state and ports
do not replace GPU coordination. Never nest an extra lock around local CI.
`validation/model_smoke.py` owns the shell launcher's setup/child supervision.
The model supervisor and helpers preserve signal status and report unresolved
owned cleanup; retain that evidence and do not stop unrelated services.

## Source tests and client assets

Run Python source validation from the repository root in a standalone external
venv. Exclude inherited `PYTHONPATH`/`PYTHONHOME`, disable user site packages,
verify interpreter/dependency provenance, and set an absolute instance-specific
`LOOM_SERVE_DIR` before any Loom command. `ci_tools/source_tests.py` owns the shared source-gate setup and discovery
contract used by hosted CI and local CI. Its install action supplies
`setuptools>=77`, `wheel>=0.45`, `packaging>=24.2`, and `textual>=0.47,<2`.
Use the venv's explicit interpreter:

```sh
"$VENV/bin/python" -m build_tools.ci_tools.source_tests install
"$VENV/bin/python" -m build_tools.ci_tools.source_tests verify
"$VENV/bin/python" -m build_tools.ci_tools.source_tests build_tools
"$VENV/bin/python" -m build_tools.ci_tools.source_tests tests
```

These are source tests, including fixtures and mocked ownership checks, rather
than installed-wheel acceptance. Some application tests invoke real client
runtimes; stage them and export the same environment as local CI's `ui_source`
stage to avoid skipping their client coverage.

Client staging uses new external destinations. Pi and OpenCode staging require
curl/tar, checksum tools, timeout, and network to fetch their pinned Node
archives and locked npm packages. OMP staging requires an existing npm and
network, then uses its separately locked Bun runtime and Bun client lock.
Dependency installation scripts are disabled; preserve lockfiles and digests.

| Client | Acquire runtime | Smallest asset check (no model/GPU) |
| --- | --- | --- |
| Pi | `bash build_tools/integrations/pi/stage-pi-runtime.sh NEW_RUNTIME` | `bash build_tools/integrations/pi/check-pi-assets.sh RUNTIME ASSET_DIRECTORY` |
| OpenCode | `bash build_tools/integrations/opencode/stage-opencode-runtime.sh NEW_RUNTIME` | `bash build_tools/integrations/opencode/check-opencode-assets.sh RUNTIME ASSET_DIRECTORY` |
| OMP | `bash build_tools/integrations/omp/stage-omp-runtime.sh NEW_RUNTIME` | `bash build_tools/integrations/omp/check-omp-assets.sh RUNTIME ASSET_DIRECTORY` |

Staging writes only the selected runtime directory (plus normal package-manager
caches). Pi uses Node 22.22.1/Pi 1.3.0, OpenCode uses Node 24.10.0/OpenCode
2.0.31, and OMP uses Bun 1.3.14/OMP 18.11.2. Staged directories must not already
exist. Asset checks accept any working directory via absolute script paths;
outputs are console results plus temporary/runtime scratch fixtures.

For source assets pass `binding/python/pi_extension`,
`binding/python/opencode_plugin`, or `binding/python/omp_extension`.
For installed assets run:

```sh
bash build_tools/integrations/check-ui-integrations.sh INSTALLED_VENV PI_RUNTIME OPENCODE_RUNTIME OMP_RUNTIME
```

This verifies the package is imported from the venv and selects its copied
assets. It writes console results and uses the staged runtimes' scratch space.
The invoking environment must exclude Python path overrides/user site packages.
The same client checks run against source and installed assets as separate CI
stages. They use synthetic protocol fixtures; backend real-model acceptance
and WebUI qualification remain separate.

Pi's gate runs `pi_extension_test.mjs`, `pi_discovery_test.mjs`, and
`pi_budgets_test.mjs`. OpenCode runs `tests/opencode_adapter_test.mjs`; OMP
runs `tests/omp_adapter_test.mjs` and `tests/omp_loader_test.mjs`. The relocated
`integrations/pi/pi_provider_test.mjs` is retained as an accepted manual provider
safety fixture, intentionally outside the three-file Pi gate. Run it beside an
isolated locked Pi runtime as described in its header; it requires explicit
source or installed asset and scratch directories and uses synthetic loopback
protocol fixtures. It does not qualify real-model serving.

OpenCode/OMP configuration modules retain their unit-tested helpers and
explicitly unwired library `qualify()` functions. These are not standalone
qualification CLIs and are not invoked by the backend or installed asset gates.
No qualification coverage is added or removed by this organization.

## Helper ownership

Run Python module commands below from the repository root with the task venv
interpreter. They require the source helper package to be importable. Absolute
script paths used by installed checks have dedicated bootstraps; do not replace
those with module commands from an outside-checkout working directory without
preserving installed application import provenance.

| Responsibility | Owning module / invocation | Inputs, prerequisites, and outputs |
| --- | --- | --- |
| Local native build setup | `python -m build_tools.wheel_tools.setup_python --rocm=/path/to/rocm [--bazel EXECUTABLE]` | Existing ROCm/TheRock Clang and pinned Bazel/Bazelisk; writes or validates `.bazelrc.configured`; `--force` replaces incompatible settings |
| Archive acceptance | `python -m build_tools.wheel_tools.validate_wheel WHEEL [--sdist SDIST]` | Existing archives; verifies payload/entrypoints/native constraints and optional reviewed sdist assets; console success/error, no GPU |
| Catalog validation | `python -m build_tools.model_tools.loom_catalog validate --inventory INVENTORY --stamp STAMP` | Explicit native compiler inventory; writes validation stamp; smallest source check is `model_tools/loom_catalog_test.py` |
| Catalog preparation | `python -m build_tools.model_tools.loom_catalog prepare --source SOURCE --output OUTPUT` | Existing bytecode source; writes prepared bytecode source; no weight acquisition |
| Catalog Bazel action | `//build_tools:loom_catalog` | Used by `loom-pipeline/bytecode.bzl`; retains runfiles/direct-import handling |
| Cache lifecycle | `ci_tools/wheel_cache.py`, `wheel_cache_summary.py` | Build/container/CI-owned cache directory, fingerprints and diagnostics; writes cache provenance/configuration or summaries; owning unit tests are the smallest check |
| GitHub cache maintenance | `ci_tools/github_wheel_cache.py` | Workflow-owned authenticated GitHub cache metadata and attempted key; may delete eligible remote caches; policy and client unit tests are the smallest check |
| Shared source tests | `python -m build_tools.ci_tools.source_tests {install,verify,build_tools,tests}` | Caller owns venv/environment/runtime fixtures; installs source requirements, verifies provenance, or runs pre-build unit discovery; console results |
| Native onboarding integration | `python -m build_tools.ci_tools.source_tests onboarding --decoder DECODER --fixture-directory NEW_DIRECTORY` | Required after wheel compilation in the manylinux build used by full local/GitHub CI; verifies checked-in frozen native inputs, replaces ambient fixture overrides, runs all eight real producer/checker cases without skips; caller owns extracted directory cleanup; see `tests/fixtures/generic_modalities/README.md` |
| Shared version rules | `versioning.py` | Reads `version.json`; release/tag validation; no independent CLI |

Only two top-level forwarding files remain supported compatibility commands:
`python build_tools/setup_python.py` and `python build_tools/make_release.py`.
They were documented user entrypoints before this cleanup. New examples use
`python -m build_tools.wheel_tools.setup_python` and
`python -m build_tools.release.make_release` from the repository root.

The other fifteen top-level forwarding files were internal build/CI/import
routes. Their consumers now use the owning packages or module commands, and
the old routes are retired after replacement validation. There is no blanket
compatibility promise for arbitrary build-helper file paths. Use these module
prefixes when updating an older local script:

| Retired top-level stems | Canonical owning package |
| --- | --- |
| `local_ci`, `local_ci_source`, `local_ci_runtime`, `wheel_cache`, `wheel_cache_summary`, `github_wheel_cache` | `build_tools.ci_tools` |
| `bazel_compiler_inputs`, `python_packaging`, `validate_wheel` | `build_tools.wheel_tools` |
| `check_installed_cpu`, `check_installed_wheel` | `build_tools.validation` |
| `loom_catalog` | `build_tools.model_tools` |
| `publish_release` | `build_tools.release` |

The top level contains nine files: this guide, `BUILD.bazel`, `versioning.py`,
the four stable shell commands, and the two justified compatibility commands.
Python source tests remain beside their owning implementation. Lockfiles and
client runtime acquisition remain separate for each integration.

`MANIFEST.in` includes build-tool Python, shell, and Bazel files. That rule does
not include integration JavaScript, JSON manifests, or lockfiles. The release
contract publishes a wheel, not an sdist. Integration execution from an sdist
is not established by that manifest; preserve existing scope and audit it
separately when moving inputs.

## Releases

Follow [`RELEASING.md`](../RELEASING.md) for authorization and the complete
workflow. From a clean `main` checkout, the standard-library-only helper
`python -m build_tools.release.make_release --version VERSION --bump-dev --dry-run`
checks version/tag/branch/Git identity without changing files. Without
`--dry-run` it creates version commits and a local tag; it never pushes.
The smallest check is `release/release_test.py` in full build-tool discovery.

`release/publish_release.py` owns workflow operations `validate-tag`,
`validate-wheel`, `github`, and `pypi-status`, all taking `--tag TAG` (and
`--wheel-dir`/`--repository` where applicable). It needs `packaging`; GitHub
publication needs authenticated `gh`, and remote publication/status requires
network. Validation prints results; `github` manages release assets/checksums,
and PyPI status inspects the exact release wheel's availability. The release
workflow owns actual PyPI upload. Release preparation/publication are explicit
operations, not validation shortcuts.

## Host wheel builds

The optional host route builds a `py3-none-linux_x86_64` wheel without Docker or
repair. It requires Linux x86-64, an immutable ROCm/TheRock SDK, the actual pinned
Bazel executable from `.bazelversion` (not Bazelisk), and a standalone task-owned
build venv. Every invocation captures current source into a new external run
directory and starts a fresh private Bazel root, repository caches and Loom state.
Without an explicit `--cache-dir`, disk and remote action caches are disabled.
This is host acceptance; it makes no manylinux portability or full-CI claim.

Prepare the venv once outside the checkout, using paths without whitespace:

```sh
BUILD_VENV=/absolute/task/build-venv
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 python3 -m venv "$BUILD_VENV"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  "$BUILD_VENV/bin/python" -m pip install --no-cache-dir \
  'setuptools>=77' 'wheel>=0.45' 'packaging>=24.2' 'build>=1.2'
```

Provision the actual pinned Bazel binary separately in a task-owned tools
namespace; any acquisition cost belongs outside build timings. The runner
validates the existing venv and imports without installing or upgrading packages.
It requires system/user site packages disabled and no nonempty setuptools system
`distutils.cfg`. It rejects inherited platform/toolchain/packaging overrides,
local `setup.cfg`, `.bazelrc.local`, module overrides and uncommitted changes to
Bazel routing files. An existing `.bazelrc.configured` must exactly match setup's
configuration for the supplied SDK. It excludes Python import/native runtime
path overrides while retaining HOME, XDG and credentials. User distutils config
is disabled through the build frontend's `--global-option=--no-user-cfg` setting.

```sh
RUN=/absolute/new/host-wheel-run
bash build_tools/build-wheel.sh --fast \
  --rocm /absolute/immutable/sdk --venv "$BUILD_VENV" \
  --run-dir "$RUN" --bazel /absolute/tools/pinned-bazel
```

All selected paths must be absolute, external and without whitespace; `RUN` must
not exist or overlap the SDK, venv or Bazel. Outputs include `source-manifest.json`,
`source/`, `wheel/`, `native-metrics/`, phase logs and `summary.json` with source identity, selected
interpreter/dependency/compiler/Bazel identities, wheel hash, timing, exits and
owned Bazel shutdown evidence. The existing backend retains its native Bazel
profile and build-event stream under `native-metrics/`; these support analysis
and execution attribution within the frontend build interval. Packaging and
native work can overlap, so the frontend interval alone is not native time.
Failures retain the run directory and the first
error; unresolved cleanup fails the build. Keep retained evidence and resolve
owned work before manually removing any output root.

Optional disk action-cache reuse is a developer opt-in for the **same stable
host, SDK and toolchain**. It retains fresh source/output/server/repository roots,
package staging and wheel validation. It does not make the host hermetic or prove
complete SDK/system equivalence:

```sh
bash build_tools/build-wheel.sh --fast \
  --rocm /absolute/immutable/sdk --venv "$BUILD_VENV" \
  --run-dir /absolute/new/cached-run --bazel /absolute/tools/pinned-bazel \
  --cache-dir /absolute/task/private-cache
```

The cache directory must be dedicated, external, same-user, mode0700 and without
symlink components or overlaps with the source/run/SDK/venv/Bazel. A new directory
is created privately. Its exclusive nonblocking lease is held until owned work
has retired; a concurrent invocation using the same cache fails. Selected host,
Bazel/compiler binary and SDK path/receipt identities must match. These bounded
records do not identify all headers, libraries, compiler resources or SDK bytes.
Bazel's declared action inputs/configuration handle ordinary source changes;
source commit and packaged Python bytes are not global cache keys.

**Use a fresh cache or clear the retired cache after changing external SDK
contents, compiler resources/configuration, system headers/libraries/tools or
interpreters, or after any uncertain unmanaged host change or cache modification.**
Keep those inputs stable during each build. Reusing the same paths or recorded
binary hashes does not remove this manual invalidation obligation. Remote caches
and execution stay disabled; inherited cache selections and container compiler
stamps are excluded. Completed disk-cache actions can be reused after an ordinary
failed build once its owned cleanup completes.

`--cache-mode read-write` is the default when `--cache-dir` is supplied.
`--cache-dir PATH --cache-mode disabled` performs the same lease/identity work
with disk action reads/writes disabled, for a matched control. `--cache-mode`
requires `--cache-dir`. Both explicit modes use equivalent empty compiler-input
stamps; per-run summaries record the effective mode, selections and native
configuration. Include all lease/identity/capture/build/validation/shutdown costs
when comparing the same candidate's disabled, population, warm and Python-marker
runs. The earlier cold measurement is historical evidence, not a matched control.

If an invocation is interrupted or cleanup is unresolved, the retained
`active.json` blocks reuse. Inspect its run directory and exact owner PID/start
identity. Confirm all work belonging to that invocation has retired, including
its private Bazel server through that run's launcher; PID absence alone does not
prove its descendants retired, and a reused PID is not authority to stop anything.
Only after that proof may the task owner remove `active.json` or clear the cache.
The runner never automatically adopts an unresolved lease, deletes a live output
root or stops unrelated work. Cache and run evidence remain external.

Run the existing installed wheel check separately, outside the source checkout,
with the verified build venv first on PATH so its `python3` creates the fresh
installation venv. Use a distinct, nonexistent destination and the exact wheel
reported by the build:

```sh
cd /absolute/task
PATH="$BUILD_VENV/bin:/usr/bin:/bin" \
  bash "$RUN/source/build_tools/check-wheel.sh" \
  "$RUN/wheel/EXACT_WHEEL_NAME.whl" /absolute/new/installed-venv
```

Record installed-check status/interpreter evidence separately from build success.
The build command checks archive contents, filename/internal host tags and source
integrity; the installed command checks existing public/payload/native-library,
catalog/helper and CPU contracts. Neither replaces the mandatory default full
container/GPU/Pi local CI gate before landing.

Wheel packaging parses the authored JSON recipes with the runtime parser, resolves
their concrete pipeline descriptors, and selects catalog assets from the descriptor
bundle union. Finite Bazel producers still own compiler inputs and trusted helper
bytes. Recipe JSON is copied byte for byte; executable source is compiled. Qwen
`pipeline.json` and catalog build hashes retain program identity without a Python
tensor-shape profile. Moonshine catalog metadata retains native program and asset
locators; acquisition and settings live in the common authored recipe.

`validation/recipe_inventory.py` independently checks the three released recipe
IDs, logical paths and immutable HF coordinates. Wheel/source archive checks reject
missing, duplicate, swapped, changed and unregistered declarations. Installed checks
compare source bytes and exercise the actual parser, binding, effective settings
and modality definition constructor; typed synthetic references in this CPU setup
check do not qualify model acquisition or native inference.
