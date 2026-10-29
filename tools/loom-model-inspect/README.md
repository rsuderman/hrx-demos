# loom-model-inspect

Load a local GGUF and print its architecture, layer tensor roles and complete
storage catalog without loading tensor payloads:

```sh
bazel run //:loom-model-inspect -- --gguf=/path/to/model.gguf
bazel run //:loom-model-inspect -- --gguf /path/to/model.gguf
bazel run //:loom-model-inspect -- --gguf=/path/to/model.gguf --layer=0
bazel run //:loom-model-inspect -- --gguf=/path/to/model.gguf --tensor=blk.0.attn_q.weight
bazel run //:loom-model-inspect -- --help
```

The summary shows the source path, model identity, attention/recurrent layer
counts, geometry, normalization and relevant position/state metadata. Widths
and attention heads are marked uniform only when the layers agree; varying
values refer to the per-layer details. Floating-point summary values use six
significant digits for readability without changing the loaded descriptor.

Storage totals count each catalog entry once, including unbound tensors and
counting tied output weights only once. Per-type counts and encoded byte totals
describe the GGUF storage, not runtime memory requirements. File size and tensor
data offset are reported separately. The Storage section holds file-wide
counts and encoded bytes. The View section counts the unique entries actually
shown, their encoded bytes, and all model-role references to those entries
(including references from hidden layers). Logical role counts are separate
from stored tensor counts.

Global and per-layer roles are grouped by architecture meaning, with attention
or recurrent roles before feed-forward roles. Each tensor's first displayed
role includes its actual name, numeric/readable ggml storage type, normalized
extents, encoded bytes and absolute file range. Further roles sharing that
tensor reference its first displayed role; a tied output explicitly references
the embedding. Unrecognized roles remain visible in an Other group. Default
output includes unassigned catalog entries after the layers without repeating
a second full catalog.

Names are escaped losslessly. Shapes follow GGML order and are padded to four
dimensions; original rank is unavailable. Layer indices identify model layers;
attention and recurrent ordinals are separate logical identities, not physical
cache offsets. GDN heads come from SSM geometry rather than attention metadata.

`--layer=N` (or `--layer N`) prints the full model summary, globals and the
selected model layer. The index must be a nonnegative decimal uint32 matching
a model layer, and repeated selectors are rejected. Tensor details are complete
at the first occurrence in the selected view even if hidden layers also use
that tensor. Selection preserves model-wide counts and storage totals.

Every full tensor detail lists all model-wide global/layer role references,
in semantic order, or marks the entry unassigned. `--tensor=EXACT_NAME` (or
`--tensor EXACT_NAME`) prints the model summary and one complete tensor record
with all references. Matching uses raw name bytes, without globbing, case
folding or interpreting the escaped display spelling. Quote names with spaces
or shell metacharacters. An explicitly empty selector matches an admitted empty
name; it does not select all tensors. Use the equals form for names beginning
with `--`. Layer and tensor selectors are mutually exclusive, and repeated
selectors or unknown names fail before publishing a report.

The shared [model libraries](../model/README.md) own catalog reading and
architecture validation. The inspector owns file I/O and human presentation.
Supported descriptors are dense `llama` and `qwen35`; unsupported variants
produce diagnostics. Storage facts and structural admission do not qualify
numerical execution, choose runtime capacities or construct a pipeline.

A GGUF path is required. Invalid input produces a nonzero exit and no successful
report. The inspector disables stdio buffering before reading and writes the
complete report only after loading, validation and checked storage aggregation
succeed. No GPU, tokenizer, compiler/importer or server is needed. This is a
developer target and is not
included in the serving wheel.

Run focused tests with:

```sh
bazel test //tools/model/testing:all //tools/loom-model-inspect/testing:all
```
