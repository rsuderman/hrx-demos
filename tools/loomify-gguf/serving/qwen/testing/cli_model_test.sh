#!/usr/bin/env bash
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
set -eu
: "${LOOMIFY_TEST_QWEN_GGUF:?manual gate requires immutable Qwen GGUF}"
: "${LOOMIFY_TEST_QWEN_TOKENIZER:?manual gate requires matching tokenizer.json}"
converter="$TEST_SRCDIR/$TEST_WORKSPACE/tools/loomify-gguf/loomify-gguf"
cd "$TEST_TMPDIR"
must_fail() {
  if "$@"; then
    exit 1
  else
    status=$?
  fi
  test "$status" -eq 1
}
# Run outside the checkout: the converter must find its owned providers through
# runfiles, rather than a development-only old model catalog/source directory.
for format in bc loom; do
  destination="$TEST_TMPDIR/cli-qwen-$format"
  "$converter" --stage=model --gguf="$LOOMIFY_TEST_QWEN_GGUF" \
    --tokenizer="$LOOMIFY_TEST_QWEN_TOKENIZER" --output-dir="$destination" \
    --format="$format" --context-capacity=512 --prefill-capacity=128
  test -f "$destination/model.json"
  test -f "$destination/prepare.loom"
  test -f "$destination/control.loom"
  test -f "$destination/weights.loom"
  if [ "$format" = bc ]; then test -f "$destination/model.loombc"; else test -f "$destination/model.loom"; fi
  test ! -e "$destination/model.json.tmp"
  must_fail "$converter" --stage=model --gguf="$LOOMIFY_TEST_QWEN_GGUF" \
      --tokenizer="$LOOMIFY_TEST_QWEN_TOKENIZER" --output-dir="$destination"
  mv "$destination" "$destination-relocated"
done
must_fail "$converter" --stage=model --gguf="$LOOMIFY_TEST_QWEN_GGUF" \
    --tokenizer="$LOOMIFY_TEST_QWEN_TOKENIZER" --output-dir=invalid-capacity \
    --prefill-capacity=513
test ! -e invalid-capacity

must_fail "$converter" --stage=model --gguf="$LOOMIFY_TEST_QWEN_GGUF" \
  --tokenizer="$TEST_TMPDIR/absent-tokenizer.json" --output-dir=missing-tokenizer
test ! -e missing-tokenizer
must_fail "$converter" --stage=model --gguf="$TEST_TMPDIR/absent-model.gguf" \
  --tokenizer="$LOOMIFY_TEST_QWEN_TOKENIZER" --output-dir=missing-gguf
test ! -e missing-gguf
printf 'owned output sentinel' > "$TEST_TMPDIR/output-parent-file"
must_fail "$converter" --stage=model --gguf="$LOOMIFY_TEST_QWEN_GGUF" \
  --tokenizer="$LOOMIFY_TEST_QWEN_TOKENIZER" --output-dir="$TEST_TMPDIR/output-parent-file/child"
test "$(cat "$TEST_TMPDIR/output-parent-file")" = 'owned output sentinel'
test ! -e "$TEST_TMPDIR/output-parent-file/child/model.json"
