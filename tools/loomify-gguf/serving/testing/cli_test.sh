#!/usr/bin/env bash
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
set -eu
converter="$TEST_SRCDIR/$TEST_WORKSPACE/tools/loomify-gguf/loomify-gguf"
cd "$TEST_TMPDIR"
printf '{}' > tokenizer.json
printf 'not a model' > malformed.gguf
expect_failure() {
  if "$converter" "$@" > stdout 2> stderr; then
    cat stderr
    exit 1
  fi
}
expect_failure --stage=model --gguf=malformed.gguf --tokenizer=tokenizer.json --output-dir=package
 test ! -e package
expect_failure --stage=model --gguf=malformed.gguf --tokenizer=tokenizer.json --output-dir=package --prefill-capacity=513
 test ! -e package
expect_failure --stage=model --gguf=malformed.gguf --output=embedding.loombc
 test ! -e embedding.loombc
expect_failure --stage=embedding --gguf=malformed.gguf --output=embedding.loombc
 test ! -e embedding.loombc
expect_failure --stage=unknown --gguf=malformed.gguf --output=embedding.loombc
 test ! -e embedding.loombc
