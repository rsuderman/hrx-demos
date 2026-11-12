# Copyright 2026 AMD ROCm Software
# SPDX-License-Identifier: MIT
"""Check independent row completeness and the explicitly selected local anchors."""

import argparse
import json
from pathlib import Path
import os
import site
import sys

sys.dont_write_bytecode = True
assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
from render_reference import load_adapter

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True, type=Path)
p.add_argument("--reference", required=True, type=Path)
p.add_argument("--source-override", action="store_true")
a = p.parse_args()
adapter = load_adapter(a.adapter)
rows = json.loads(a.reference.read_text())
assert rows
for row in rows:
    assert row["revision"] == adapter.REVISION
    if "error" not in row:
        assert isinstance(row["token_ids"], list), "complete token vector required"
        assert all(isinstance(token, int) and token >= 0 for token in row["token_ids"])
        assert row["utf8_hex"] == row["text"].encode().hex()
        assert row["bytes"] == len(row["text"].encode())
adapter.check_reference(rows, a.source_override)
print(f"PASS: {adapter.REVISION} {len(rows)} independent oracle rows")
