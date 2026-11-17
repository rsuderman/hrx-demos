#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
set -euo pipefail

# Check a disposable copy of the index, including partially staged files.
root=$(git rev-parse --show-toplevel)
cd -- "$root"
state=${HRX_FORMAT_HOOK_ROOT:-${CODEX_PROJECT_WORKSPACES:-$HOME/codex/project-workspaces}/hrx-solutions/tools/formatting-hooks}
mkdir -p -- "$state"
state=$(realpath -- "$state")
case "$state" in
  "$root"|"$root"/*)
    echo 'HRX_FORMAT_HOOK_ROOT must be outside the checkout.' >&2
    exit 2 ;;
esac
# A stable snapshot path permits Bazel reuse across commits. Serialize each
# checkout's snapshots; different worktrees have independent locks and indexes.
identity=$(printf '%s' "$root" | sha256sum)
state="$state/${identity%% *}"
mkdir -p -- "$state"
exec 9>"$state/lock"
flock 9
snapshot="$state/staged"
trap 'rm -rf -- "$snapshot"; rm -f -- "$state/index-info"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -- "$snapshot"
git ls-files --stage -z > "$state/index-info"
while IFS= read -r -d '' entry; do
  metadata=${entry%%$'\t'*}
  name=${entry#*$'\t'}
  read -r mode object stage <<< "$metadata"
  if [[ $stage != 0 ]]; then
    echo 'Resolve index conflicts before checking formatting.' >&2
    exit 1
  fi
  mkdir -p -- "$(dirname -- "$snapshot/$name")"
  case "$mode" in
    100644|100755)
      # Read raw blobs: checkout-index would apply smudge filters and EOL rules.
      git cat-file blob "$object" > "$snapshot/$name"
      chmod "${mode#100}" -- "$snapshot/$name" ;;
    120000) git checkout-index --prefix="$snapshot/" -- "$name" ;;
    *) echo "Unsupported staged file mode: $mode ($name)" >&2; exit 1 ;;
  esac
done < "$state/index-info"
# Git exports index/worktree variables to hooks. Do not let the snapshot's Git
# commands accidentally operate on the real repository or an alternate index.
mapfile -t git_environment < <(git rev-parse --local-env-vars)
unset "${git_environment[@]}"
# Reconstruct only the file inventory. Re-adding files could run filters or
# newline conversion and change the exact staged bytes being checked.
git -C "$snapshot" init --quiet --template=
git -C "$snapshot" update-index -z --index-info < "$state/index-info"
cd -- "$snapshot"
if ! bash "$snapshot/build_tools/format.sh" --check; then
  echo 'Commit blocked by formatting. Run bash build_tools/format.sh --fix, review, and stage the fixes.' >&2
  exit 1
fi
