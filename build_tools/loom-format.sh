#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
set -euo pipefail

# Build the formatter from MODULE.bazel's HRX pin, never an unrelated PATH tool.
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
state=${HRX_FORMAT_BAZEL_ROOT:-${CODEX_PROJECT_WORKSPACES:-$HOME/codex/project-workspaces}/hrx-solutions/tools/formatting-bazel}
if [[ $state != /* ]]; then
  echo 'HRX_FORMAT_BAZEL_ROOT must be an absolute external path.' >&2
  exit 2
fi
mkdir -p -- "$state"
state=$(realpath -- "$state")
case "$state" in
  "$root"|"$root"/*)
    echo 'HRX_FORMAT_BAZEL_ROOT must be outside the checkout.' >&2
    exit 2 ;;
esac
if ! command -v "${BAZEL:-bazel}" >/dev/null; then
  echo 'Install Bazelisk or the Bazel version in .bazelversion; see build_tools/README.md.' >&2
  exit 2
fi
cd -- "$root"
exec "${BAZEL:-bazel}" --output_user_root="$state" run \
  --noshow_progress --ui_event_filters=-info --symlink_prefix=/ \
  --@hrx_system//runtime/config/hal:drivers=task \
  --@hrx_system//loom/config/execute:enable=iree_vm \
  @hrx_system//loom/src/loom/tools/loom-format -- "$@"
