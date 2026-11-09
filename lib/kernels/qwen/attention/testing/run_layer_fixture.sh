#!/usr/bin/env bash
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
# The native reader is test-only glue in an allowed upstream package. It emits
# current command-product placements, never assumed or retained bank offsets.
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo 'usage: run_layer_fixture.sh PYTHON REFLECTOR CHECKER MODULE QK_KIND OUTPUT' >&2
  exit 2
fi
fixture_python=$1
fixture_reflector=$2
fixture_checker=$3
fixture_module=$4
fixture_kind=$5
fixture_output=$6
mkdir -p "$fixture_output"
"$fixture_reflector" "$fixture_module" attention_layer > "$fixture_output/reflection.tsv"
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 "$fixture_python" \
  "$(dirname "$0")/layer_fixture.py" "$fixture_output/reflection.tsv" \
  "$fixture_output" --query-kind "$fixture_kind"
"$fixture_checker" "$fixture_output/check.loom" --library="$fixture_module" \
  --device=amdgpu > "$fixture_output/result.json" 2> "$fixture_output/stderr.log"
"$fixture_checker" "$fixture_output/chain.loom" --library="$fixture_module" \
  --device=amdgpu > "$fixture_output/chain-result.json" 2> "$fixture_output/chain-stderr.log"
