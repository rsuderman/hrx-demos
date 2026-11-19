#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: MIT
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec python3 "$root/build_tools/loom_format_cache.py" "$@"
