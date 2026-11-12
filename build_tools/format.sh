#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
  echo 'Usage: bash build_tools/format.sh [--check|--fix]'
  echo 'Check is the default. Install the pinned tools as described in build_tools/README.md.'
}

mode=${1:---check}
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "$mode" in
  --check|--fix) ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$root"
if [[ $(git rev-parse --show-toplevel) != "$root" ]]; then
  echo 'Formatting requires the source Git checkout.' >&2
  exit 2
fi
# shellcheck source=build_tools/format-tools.env
source "$root/build_tools/format-tools.env"

missing() {
  echo "$1 Install the pinned tools using build_tools/README.md." >&2
  exit 2
}
for tool in buildifier clang-format black; do
  command -v "$tool" >/dev/null || missing "Missing formatter: $tool."
done

observed=$(buildifier --version)
first=${observed%%$'\n'*}
first=${first%"${first##*[![:space:]]}"}
[[ $first == "buildifier version: $BUILDIFIER_VERSION" ]] || missing "Expected buildifier $BUILDIFIER_VERSION; found $first."
observed=$(clang-format --version)
first=${observed%%$'\n'*}
[[ $first == "clang-format version $CLANG_FORMAT_VERSION" || $first == "clang-format version $CLANG_FORMAT_VERSION "* ]] || missing "Expected clang-format $CLANG_FORMAT_VERSION; found $first."
observed=$(black --version)
first=${observed%%$'\n'*}
[[ $first == "black, $BLACK_VERSION" || $first == "black, $BLACK_VERSION "* ]] || missing "Expected Black $BLACK_VERSION; found $first."

# Capture the NUL-delimited list explicitly: process substitution would hide a
# failing Git selector and could report an empty checkout as successfully checked.
list_dir=$(realpath -- "${TMPDIR:-/tmp}")
case "$list_dir" in
  "$root"|"$root"/*)
    echo 'Formatting requires a TMPDIR outside the source checkout.' >&2
    exit 2
    ;;
esac
selection=$(mktemp "$list_dir/hrx-format-files.XXXXXX")
trap 'rm -f -- "$selection"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
if ! git ls-files -z > "$selection"; then
  echo 'Cannot list tracked formatter inputs; check the Git index.' >&2
  exit 2
fi

bazel_files=()
cpp_files=()
python_files=()
while IFS= read -r -d '' name; do
  case "$name" in
    .deps/*|build/*|dist/*|bazel-*/*|third_party/*|vendor/*) continue ;;
    # These preparers are trusted, byte-pinned payloads in release catalogs.
    loom-pipeline/recognition/moonshine/preparer/prepare_assets.py|loom-pipeline/speech/pocket_tts/preparer/prepare_assets.py) continue ;;
  esac
  case "$name" in
    BUILD|*/BUILD|BUILD.bazel|*/BUILD.bazel|WORKSPACE|*/WORKSPACE|WORKSPACE.bazel|*/WORKSPACE.bazel|MODULE.bazel|*/MODULE.bazel|*.bzl) kind=bazel ;;
    *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp) kind=cpp ;;
    *.py) kind=python ;;
    *) continue ;;
  esac
  # Symlinks, deleted entries, generated output and untracked files are not inputs.
  [[ ! -L $name && -f $name ]] || continue
  physical=$(realpath -- "$name")
  case "$physical" in
    "$root"/*) ;;
    *) missing "Tracked formatter input resolves outside this checkout: $name." ;;
  esac
  case "$kind" in
    bazel) bazel_files+=("$name") ;;
    cpp) cpp_files+=("$name") ;;
    python) python_files+=("$name") ;;
  esac
done < "$selection"

echo "Formatting $mode: ${#bazel_files[@]} Bazel/Starlark, ${#cpp_files[@]} C/C++, ${#python_files[@]} Python files."
status=0
if (( ${#bazel_files[@]} )); then
  # Loom merge macros consume ordered sources and literal label spellings.
  # Buildifier accepts these as repeated flags, not a comma-separated value.
  if [[ $mode == --fix ]]; then
    buildifier -mode=fix -lint=off -buildifier_disable=listsort -buildifier_disable=label -- "${bazel_files[@]}" || status=1
  else
    buildifier -mode=check -lint=off -buildifier_disable=listsort -buildifier_disable=label -- "${bazel_files[@]}" || status=1
  fi
fi
if (( ${#cpp_files[@]} )); then
  if [[ $mode == --fix ]]; then
    clang-format --style=file -i -- "${cpp_files[@]}" || status=1
  else
    clang-format --style=file --dry-run --Werror --ferror-limit=0 -- "${cpp_files[@]}" || status=1
  fi
fi
if (( ${#python_files[@]} )); then
  if [[ $mode == --fix ]]; then
    black --no-cache --config "$root/pyproject.toml" -- "${python_files[@]}" || status=1
  else
    black --no-cache --check --diff --config "$root/pyproject.toml" -- "${python_files[@]}" || status=1
  fi
fi
exit "$status"
